"""Checkpoints that can actually be resumed from.

A Colab session dies well before 24h of training finishes, so every run has to
survive being killed at an arbitrary moment and pick up where it left off. That
means more than the weights: the optimizer moments, the scheduler position, the
RNG streams and the epoch counter all have to come back, or the resumed run is a
different run that happens to start from the same parameters.

**The sink is Hugging Face Hub by default, not Drive.** `colab drivemount` needs
a TTY and cannot be driven by an agent (see AGENTS.md); the Hub works headless
with a token. Drive stays available for a human-run session.

Two failure modes this guards against, both of which have eaten real runs:

* **A truncated checkpoint.** The VM can die mid-write, leaving a file that
  loads as garbage or not at all. We write to a temp path and rename — rename is
  atomic within a filesystem — and we keep the previous checkpoint until the new
  one is safely in place.
* **A checkpoint from a different run.** The config hash is stored inside and
  checked on load, so a renamed directory or a changed hyper-parameter fails
  loudly instead of resuming into nonsense.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from drugrx.config import RunConfig


@dataclass
class TrainState:
    """Everything that is not the model but still defines where we are."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None
    best_epoch: int = -1
    #: Epochs since `best_metric` improved — what the patience kill rule reads.
    epochs_since_best: int = 0
    #: Wall-clock seconds already spent, summed across sessions. A resumed run
    #: must not get a fresh 24h budget.
    elapsed_seconds: float = 0.0


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _set_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])
        except (RuntimeError, ValueError):
            # Different GPU count on the new VM. The data order will diverge
            # slightly; say so rather than crashing a 20-hour resume.
            print("[ckpt] warning: could not restore CUDA RNG state (GPU count changed)")


class Checkpointer:
    """Saves to a local directory, and optionally mirrors to the Hub.

    `local_dir` is on the VM, so it dies with the VM — it exists so that a
    *kernel* restart is cheap. The Hub copy is what survives the VM.
    """

    def __init__(
        self,
        config: RunConfig,
        local_dir: str | Path = "/content/ckpt",
        hub_repo: str = "",
        hub_token: str = "",
        keep_last: int = 2,
    ) -> None:
        self.config = config
        self.dir = Path(local_dir) / config.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.hub_repo = hub_repo or config.hub_repo
        self.hub_token = hub_token or os.environ.get("HF_TOKEN", "")
        self._api = None

        if self.hub_repo and self.hub_token:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=self.hub_token)
                self._api.create_repo(
                    self.hub_repo, repo_type="model", private=True, exist_ok=True
                )
            except Exception as exc:  # noqa: BLE001 — never fail a run over the mirror
                print(f"[ckpt] hub disabled: {exc}")
                self._api = None
        elif self.hub_repo:
            print("[ckpt] hub repo set but no HF_TOKEN; checkpoints stay on the VM only")

        (self.dir / "config.json").write_text(json.dumps(config.to_json(), indent=2, default=str))

    # ── writing ───────────────────────────────────────────────────────

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        state: TrainState,
        scheduler: Any = None,
        scaler: Any = None,
        ema: Any = None,
        is_best: bool = False,
    ) -> Path:
        payload = {
            "config_hash": self.config.run_id,
            "config": self.config.to_json(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "ema": ema.state_dict() if ema is not None else None,
            "state": state.__dict__,
            "rng": _rng_state(),
            "saved_at": time.time(),
        }

        path = self.dir / f"epoch{state.epoch:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomic within the filesystem

        # "latest" is a copy, not a symlink: the Hub does not keep symlinks and
        # a dangling one on resume is a confusing way to lose a run.
        latest = self.dir / "latest.pt"
        shutil.copy2(path, latest.with_suffix(".pt.tmp"))
        os.replace(latest.with_suffix(".pt.tmp"), latest)

        if is_best:
            best = self.dir / "best.pt"
            shutil.copy2(path, best.with_suffix(".pt.tmp"))
            os.replace(best.with_suffix(".pt.tmp"), best)

        self._prune()

        # The best checkpoint always goes up — it is the artifact that matters.
        # `latest` goes up on a cadence, because mirroring 71 MB every epoch
        # takes longer than an epoch does and buys only a few minutes of
        # recovery granularity.
        every = max(1, self.config.hub_every)
        if is_best:
            self._mirror(self.dir / "best.pt", "best.pt")
        if state.epoch % every == 0 or is_best:
            self._mirror(latest, "latest.pt")
        return path

    def _prune(self) -> None:
        epochs = sorted(self.dir.glob("epoch*.pt"))
        for stale in epochs[: max(0, len(epochs) - self.keep_last)]:
            stale.unlink(missing_ok=True)

    def _mirror(self, path: Path, name: str) -> None:
        if self._api is None:
            return
        try:
            self._api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"{self.config.run_id}/{name}",
                repo_id=self.hub_repo,
                repo_type="model",
            )
        except Exception as exc:  # noqa: BLE001
            # An upload failure must never kill a run that is otherwise fine.
            print(f"[ckpt] hub upload failed ({name}): {exc}")

    # ── reading ───────────────────────────────────────────────────────

    def _fetch_from_hub(self, name: str = "latest.pt") -> Optional[Path]:
        if self._api is None:
            return None
        try:
            from huggingface_hub import hf_hub_download

            got = hf_hub_download(
                repo_id=self.hub_repo,
                filename=f"{self.config.run_id}/{name}",
                repo_type="model",
                token=self.hub_token,
            )
            return Path(got)
        except Exception as exc:  # noqa: BLE001
            print(f"[ckpt] nothing to resume from on the hub: {exc}")
            return None

    def resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any = None,
        scaler: Any = None,
        ema: Any = None,
        map_location: str = "cuda",
    ) -> TrainState:
        """Load the newest checkpoint, from the VM or failing that the Hub.

        Returns a fresh `TrainState` when there is nothing to resume from, so a
        caller can always treat this as "give me the state to start at".
        """
        path = self.dir / "latest.pt"
        if not path.exists():
            fetched = self._fetch_from_hub("latest.pt")
            if fetched is None:
                print("[ckpt] starting from scratch")
                return TrainState()
            path = fetched

        payload = torch.load(path, map_location=map_location, weights_only=False)

        stored = payload.get("config_hash")
        if stored != self.config.run_id:
            raise RuntimeError(
                f"checkpoint {path} belongs to run {stored!r}, not {self.config.run_id!r}. "
                "A defining config field changed — start a new run, or restore the old config."
            )

        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer"):
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler"):
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler"):
            scaler.load_state_dict(payload["scaler"])
        if ema is not None and payload.get("ema"):
            ema.load_state_dict(payload["ema"])
        if payload.get("rng"):
            _set_rng_state(payload["rng"])

        state = TrainState(**payload["state"])
        print(
            f"[ckpt] resumed {self.config.run_id} at epoch {state.epoch}, "
            f"step {state.global_step}, {state.elapsed_seconds / 3600:.2f}h spent, "
            f"best {self.config.monitor}={state.best_metric}"
        )
        return state
