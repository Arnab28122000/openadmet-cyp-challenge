"""Checkpoint and resume for the vmapped CYP ensembles, mirrored to the Hub.

A Colab Pro+ A100 has died mid-experiment twice in this session — the
instability `colabtools` #5950 describes. Results were already being pushed as
they landed, which saved completed runs, but a run killed *in progress* still
started from scratch. With long schedules (3,000 epochs, optima past 200) that
is the difference between losing minutes and losing an hour.

What a resumable checkpoint has to carry, and why each matters:

* **stacked params and buffers** — the ensemble itself;
* **optimizer state** — Adam's moments. Restoring weights without them restarts
  the optimiser cold and the loss visibly jumps;
* **scheduler state** — a cosine schedule resumed at the wrong position trains at
  the wrong learning rate for the rest of the run;
* **epoch, best score, best epoch, patience counter** — or early stopping
  silently restarts its clock and the run trains far past where it should;
* **the best state so far**, separately from the live state, because the live one
  is usually worse by the time a crash happens;
* **RNG state** — so a resumed run is the same run rather than a similar one.

The config hash guards against resuming into a checkpoint from a different
architecture, which loads shapes that happen to match and trains on nonsense.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

HUB_REPO_DEFAULT = "arnab2812/drugrx-cyp-runs"


def config_hash(cfg: dict) -> str:
    """Identity of the *architecture and data*, not the schedule length."""
    keep = {k: cfg[k] for k in sorted(cfg)
            if k in {"d_model", "n_layers", "dropout", "members", "seed",
                     "in_dim", "n_heads", "aux_weight", "lr"}}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:10]


def _rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _set_rng(s: dict) -> None:
    try:
        random.setstate(s["python"])
        np.random.set_state(s["numpy"])
        torch.set_rng_state(s["torch"].cpu() if hasattr(s["torch"], "cpu") else s["torch"])
        if torch.cuda.is_available() and s.get("cuda"):
            torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])
    except Exception as exc:  # noqa: BLE001
        print(f"[resume] RNG not restored ({exc}); data order will differ", flush=True)


class EnsembleCheckpoint:
    def __init__(self, tag: str, seed: int, cfg: dict,
                 local_dir: str = "/content/ckpt", hub_repo: str = HUB_REPO_DEFAULT,
                 push_every: int = 50) -> None:
        self.tag, self.seed, self.cfg = tag, seed, cfg
        self.hash = config_hash(cfg)
        self.name = f"{tag}_s{seed}_{self.hash}"
        self.dir = Path(local_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{self.name}.pt"
        self.hub_repo = hub_repo if os.environ.get("HF_TOKEN") else ""
        self.push_every = push_every
        self._api = None
        if self.hub_repo:
            try:
                from huggingface_hub import HfApi

                self._api = HfApi(token=os.environ["HF_TOKEN"])
                self._api.create_repo(self.hub_repo, repo_type="dataset",
                                      private=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[resume] hub disabled: {exc}", flush=True)

    # ── write ────────────────────────────────────────────────────────
    def save(self, *, epoch: int, params, buffers, opt, sched, best: float,
             best_epoch: int, since: int, curve: list, best_state=None,
             force_push: bool = False) -> None:
        payload = {
            "hash": self.hash, "cfg": self.cfg, "tag": self.tag, "seed": self.seed,
            "epoch": epoch, "best": best, "best_epoch": best_epoch, "since": since,
            "curve": curve,
            "params": {k: v.detach().cpu() for k, v in params.items()},
            "buffers": {k: v.detach().cpu() for k, v in buffers.items()},
            "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
            "best_state": None if best_state is None else
                ({k: v.detach().cpu() for k, v in best_state[0].items()},
                 {k: v.detach().cpu() for k, v in best_state[1].items()}),
            "rng": _rng_state(),
        }
        tmp = self.path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, self.path)          # atomic: a half-written checkpoint
                                            # is worse than none
        if self._api and (force_push or epoch % self.push_every == 0):
            try:
                self._api.upload_file(path_or_fileobj=str(self.path),
                                      path_in_repo=f"ckpt/{self.name}.pt",
                                      repo_id=self.hub_repo, repo_type="dataset")
            except Exception as exc:  # noqa: BLE001
                print(f"[resume] push failed at epoch {epoch}: {exc}", flush=True)

    # ── read ─────────────────────────────────────────────────────────
    def load(self, device: str = "cuda") -> dict | None:
        """Local checkpoint first, then the Hub. None means start fresh."""
        src = self.path if self.path.exists() else None
        if src is None and self._api:
            try:
                from huggingface_hub import hf_hub_download

                src = Path(hf_hub_download(self.hub_repo, f"ckpt/{self.name}.pt",
                                           repo_type="dataset",
                                           token=os.environ["HF_TOKEN"]))
                print(f"[resume] pulled checkpoint from the hub", flush=True)
            except Exception:  # noqa: BLE001
                return None
        if src is None:
            return None
        blob = torch.load(src, map_location=device, weights_only=False)
        if blob.get("hash") != self.hash:
            print(f"[resume] checkpoint is for config {blob.get('hash')}, not {self.hash} — "
                  "ignoring it rather than loading mismatched weights", flush=True)
            return None
        if blob.get("rng"):
            _set_rng(blob["rng"])
        print(f"[resume] resuming {self.name} at epoch {blob['epoch']} "
              f"(best {blob['best']:.4f} @ep{blob['best_epoch']}, "
              f"{blob['since']} epochs since improvement)", flush=True)
        return blob
