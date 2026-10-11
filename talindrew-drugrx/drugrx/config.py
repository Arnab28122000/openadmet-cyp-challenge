"""Run configuration, and the id derived from it.

A run's id is a hash of its config, so two runs with the same settings resume
each other and a changed setting starts a new run instead of silently poisoning
an old checkpoint. That has to include everything that changes the result —
model, data, optimisation, seed — and nothing that does not, like how often we
log, or a resume would be impossible after changing the log interval.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any


#: Fields that do not change what the model learns, and so are excluded from the
#: run id. Keep this list short and justified: anything wrongly listed here lets
#: two genuinely different runs share a checkpoint.
NON_DEFINING = frozenset({
    "log_every", "eval_every", "checkpoint_every", "notes", "hub_repo",
    "heartbeat_every", "max_hours", "hub_every",
})


@dataclass
class RunConfig:
    """Everything a run needs. Serialised next to every checkpoint."""

    # ── identity ──────────────────────────────────────────────────────
    #: Short human name; part of the id, so renaming forks the run.
    name: str = "unnamed"
    seed: int = 0

    # ── data ──────────────────────────────────────────────────────────
    #: Dataset key the loader resolves (see drugrx/data/).
    dataset: str = ""
    #: "scaffold" for headline numbers. "random" is allowed for debugging only
    #: and is reported separately, never as the result (see AGENTS.md).
    split: str = "scaffold"
    split_seed: int = 0
    tasks: tuple[str, ...] = ()
    max_atoms: int = 128

    # ── model ─────────────────────────────────────────────────────────
    arch: str = ""
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    #: Extra arch knobs (recursion depth, expert count…) without a schema change.
    arch_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── optimisation ──────────────────────────────────────────────────
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 128
    grad_accum: int = 1
    epochs: int = 100
    warmup_steps: int = 500
    #: **WSD (warmup-stable-decay) by default, not cosine.** A Colab session can
    #: die at hour 9; you can cool down from any WSD checkpoint, but a cosine
    #: schedule interrupted partway gives a model that was never annealed.
    schedule: str = "wsd"
    #: Fraction of the run spent in WSD's decay phase.
    decay_frac: float = 0.2
    grad_clip: float = 1.0
    #: "bf16" on Ada (sm_89) is safe and avoids fp16's loss-scaling pitfalls.
    precision: str = "bf16"
    #: EMA was worth 7.5 points in TRM's own ablation and costs one extra copy
    #: of the weights. On by default; 0.0 disables it.
    ema_decay: float = 0.999
    compile: bool = False

    # ── stopping ──────────────────────────────────────────────────────
    #: Epochs without validation improvement before the run is killed.
    patience: int = 10
    #: The metric the kill rules and "best checkpoint" watch.
    monitor: str = "val/loss"
    monitor_mode: str = "min"
    #: A run that cannot beat this by the halfway point is killed. Set it from
    #: the Phase 0 baseline, so "no better than fingerprints" ends the run.
    baseline: float | None = None

    # ── housekeeping (not part of the id) ─────────────────────────────
    log_every: int = 50
    eval_every: int = 1
    checkpoint_every: int = 1
    heartbeat_every: int = 30
    max_hours: float = 23.0
    hub_repo: str = ""
    #: Mirror `latest.pt` to the Hub every N epochs. The *best* checkpoint is
    #: always mirrored when it improves, so a crash never costs more than the
    #: last N epochs of progress — while a 71 MB upload at ~2.5 MB/s every epoch
    #: would otherwise cost ~28 s per epoch, which is longer than an epoch takes.
    hub_every: int = 5
    notes: str = ""

    def defining(self) -> dict[str, Any]:
        """The parts of the config that determine the result."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in NON_DEFINING
        }

    @property
    def run_id(self) -> str:
        blob = json.dumps(self.defining(), sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode()).hexdigest()[:10]
        return f"{self.name}-{digest}"

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "run_id": self.run_id}

    @classmethod
    def from_json(cls, blob: dict[str, Any]) -> "RunConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in blob.items() if k in known}
        # tuples survive a JSON round trip as lists; the id hash would differ.
        for key in ("tasks",):
            if key in kwargs and isinstance(kwargs[key], list):
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)
