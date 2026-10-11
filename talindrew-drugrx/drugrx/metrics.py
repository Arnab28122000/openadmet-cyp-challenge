"""Metrics, heartbeat, and the kill rules.

The local monitor cannot see inside the VM, so the run has to say what it is
doing in files the monitor can pull: an append-only JSONL of metrics, and a
heartbeat that is rewritten often. Append-only matters — a resumed run must add
to the record rather than replace it, or the history of a 24h run that spanned
three sessions is just the last session.

The heartbeat is what distinguishes **slow** from **dead**, which is the single
most expensive confusion in a long run: killing a slow run wastes the work, and
waiting on a dead one wastes the units.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from drugrx.config import RunConfig


class MetricLog:
    """Append-only JSONL. One object per line, flushed immediately.

    Flushing every line is deliberate: a VM that dies takes the buffer with it,
    and the lines we lose are exactly the ones that would explain why.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def log(self, **fields: Any) -> None:
        fields.setdefault("t", time.time())
        self._fh.write(json.dumps(fields, default=float) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


class Heartbeat:
    """A file whose mtime and contents say the run is alive and where it is."""

    def __init__(self, path: str | Path, every: float = 30.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.every = every
        self._last = 0.0

    def beat(self, force: bool = False, **fields: Any) -> None:
        now = time.time()
        if not force and now - self._last < self.every:
            return
        self._last = now
        payload = {"t": now, "pid": os.getpid(), **fields}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, default=float))
        os.replace(tmp, self.path)


@dataclass
class KillVerdict:
    stop: bool
    reason: str = ""


class KillRules:
    """The written-down conditions under which a run is stopped.

    They live in code rather than in a human's judgement because the whole point
    is that they apply at 3am without anyone deciding to be disciplined. Each
    one is from docs/drugrx-training/AGENTS.md.
    """

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.lower_is_better = config.monitor_mode == "min"

    def _worse(self, a: float, b: float) -> bool:
        """Is `a` worse than `b`?"""
        return a > b if self.lower_is_better else a < b

    def check(
        self,
        *,
        epoch: int,
        epochs_since_best: int,
        current: Optional[float],
        elapsed_hours: float,
        diverged: bool = False,
    ) -> KillVerdict:
        if diverged:
            return KillVerdict(True, "loss is NaN or diverged")

        if epochs_since_best >= self.config.patience:
            return KillVerdict(
                True,
                f"no improvement in {self.config.monitor} for {epochs_since_best} epochs "
                f"(patience {self.config.patience})",
            )

        if elapsed_hours >= self.config.max_hours:
            return KillVerdict(True, f"time budget spent ({elapsed_hours:.1f}h)")

        # "Worse than the baseline at the halfway point." A deep model that
        # cannot beat ECFP4+LightGBM is the expected outcome often enough that
        # it has to end the run automatically rather than be argued about.
        if (
            self.config.baseline is not None
            and current is not None
            and epoch >= self.config.epochs // 2
            and self._worse(current, self.config.baseline)
        ):
            return KillVerdict(
                True,
                f"at epoch {epoch} (halfway) {self.config.monitor}={current:.4f} is still worse "
                f"than the baseline {self.config.baseline:.4f}",
            )

        return KillVerdict(False)


def is_improvement(current: float, best: float | None, mode: str = "min") -> bool:
    if best is None:
        return True
    return current < best if mode == "min" else current > best
