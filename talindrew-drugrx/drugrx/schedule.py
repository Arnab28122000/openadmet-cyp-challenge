"""Learning-rate schedules, chosen for a machine that dies without warning.

**WSD (warmup-stable-decay) is the default, not cosine.** A Colab Pro+ session
is advertised as "up to 24 hours" and is reported dying after 40–60 minutes; a
cosine schedule that is interrupted at hour 9 leaves a model that was never
annealed, and annealing is most of the final quality. WSD holds a constant rate
through the stable phase, so **any** checkpoint can be cooled down into a
finished model with a short decay run.

That property is what makes a 24h budget spread over an unknown number of
sessions tractable at all.
"""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def wsd(
    optimizer,
    total_steps: int,
    warmup_steps: int = 500,
    decay_frac: float = 0.2,
    min_ratio: float = 0.0,
) -> LambdaLR:
    """Warmup → constant → decay to `min_ratio` of the peak.

    `total_steps` is a plan, not a promise: if the run is cut short, the stable
    phase simply never ends, and the checkpoint is still usable after a separate
    decay run. That is the point of the schedule.
    """
    decay_steps = max(1, int(total_steps * decay_frac))
    stable_end = max(warmup_steps, total_steps - decay_steps)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        if step < stable_end:
            return 1.0
        progress = (step - stable_end) / max(1, total_steps - stable_end)
        # 1-sqrt decay: the shape the WSD work found beat linear and cosine for
        # the same decay budget.
        return max(min_ratio, 1.0 - math.sqrt(min(1.0, progress)))

    return LambdaLR(optimizer, fn)


def cosine(optimizer, total_steps: int, warmup_steps: int = 500,
           min_ratio: float = 0.01) -> LambdaLR:
    """Kept for ablations only. See the module docstring for why not by default."""

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, fn)


def build(name: str, optimizer, total_steps: int, warmup_steps: int = 500,
          decay_frac: float = 0.2) -> LambdaLR:
    if name == "wsd":
        return wsd(optimizer, total_steps, warmup_steps, decay_frac)
    if name == "cosine":
        return cosine(optimizer, total_steps, warmup_steps)
    if name == "constant":
        return LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, warmup_steps)))
    raise ValueError(f"unknown schedule {name!r}")


class EMA:
    """Exponential moving average of the weights.

    Worth 7.5 points in TRM's own ablation and costs one extra copy of the
    parameters — the best ratio of benefit to complexity on the shortlist. The
    shadow weights are what gets evaluated and shipped; the live weights are
    only what training updates.
    """

    def __init__(self, model, decay: float = 0.999) -> None:
        import torch

        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: dict = {}
        self._torch = torch

    def update(self, model) -> None:
        with self._torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def apply_to(self, model) -> None:
        """Swap the EMA weights in, keeping the live ones to restore."""
        self._backup = {
            k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow
        }
        model.load_state_dict(
            {k: (self.shadow[k].to(v.dtype) if k in self.shadow else v)
             for k, v in model.state_dict().items()},
            strict=False,
        )

    def restore(self, model) -> None:
        if self._backup:
            model.load_state_dict(self._backup, strict=False)
            self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, blob: dict) -> None:
        self.decay = blob.get("decay", self.decay)
        self.shadow = blob["shadow"]
