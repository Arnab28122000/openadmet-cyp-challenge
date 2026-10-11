"""GPU-resident training: the whole dataset in VRAM, an ensemble trained at once.

The previous loop used batch 128 and about 2 GB of a 40 GB card, with a
DataLoader shuttling tensors from host memory every step. For a dataset this
small that is almost all overhead: 46k × 3,091 float32 is **~570 MB**, so the
entire feature matrix, labels and mask fit in VRAM with room to spare and the
DataLoader can be deleted outright. Batches become index slices into a resident
tensor, which costs a kernel launch instead of a host-to-device copy.

**The useful way to spend the rest of the card is an ensemble, not a bigger
batch.** Past a few thousand rows a larger batch mostly buys noise reduction
nobody asked for, whereas `K` independently-initialised models trained
simultaneously under `vmap` give what we actually need: calibrated uncertainty.
The playground ranks candidates, and a confident wrong score is worse than an
honest wide interval — ensembling is the cheapest honest uncertainty there is,
and here it is nearly free because the GPU was idle anyway.

`torch.func.stack_module_state` + `vmap` runs all K members as one batched
kernel, so K=8 costs far less than 8 sequential runs. Memory scales with K, so
`fit_ensemble` reports what it reserved and the caller can size K to the card.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad_and_value, stack_module_state, vmap


@dataclass
class GPUData:
    """Everything the loop needs, resident on the device."""

    x: torch.Tensor          # (n, d)
    y: torch.Tensor          # (n, t)
    mask: torch.Tensor       # (n, t) bool
    n: int

    @classmethod
    def from_numpy(cls, x: np.ndarray, y: np.ndarray, mask: np.ndarray,
                   device: str = "cuda") -> "GPUData":
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)).to(device)
        mt = torch.from_numpy(np.ascontiguousarray(mask)).to(device)
        return cls(xt, yt, mt, xt.shape[0])

    def bytes(self) -> int:
        return sum(t.element_size() * t.nelement() for t in (self.x, self.y, self.mask))


def suggest_batch_size(data: GPUData, d_model: int, n_tasks: int,
                       ensemble: int = 1, headroom: float = 0.9) -> int:
    """Largest power-of-two batch that should fit, leaving `headroom` spare.

    Headroom is 0.9 rather than something timid **because the OOM backoff below
    makes an over-estimate cheap**: a too-large batch costs one halving and a
    cache flush, where a too-small one leaves the card idle for the whole run.
    Without the retry this number would have to be conservative; with it, aiming
    high is the right default.
    """
    free, total = torch.cuda.mem_get_info()
    budget = free * headroom - data.bytes()
    if budget <= 0:
        return 256
    # Per-row activation cost. The first version of this used a factor of 3 for
    # "forward + grad + optimiser slack" and chose a full-dataset batch across
    # 16 members, which allocated 38.7 GB of a 39.5 GB card and died. The
    # autograd graph holds every intermediate for the backward pass, so the real
    # multiplier is far larger than 3 — and the estimate is only a starting
    # point now: `fit_ensemble` halves the batch and retries on OOM, which is
    # what actually makes this safe. Prediction gets you close; retry gets you
    # correct.
    per_row = 4 * d_model * 6 * ensemble * 12 + 4 * n_tasks * ensemble * 8
    raw = max(256, int(budget / max(1, per_row)))
    return min(1 << (raw.bit_length() - 1), data.n)


def fit_ensemble(
    make_model: Callable[[int], nn.Module],
    data: GPUData,
    valid: GPUData,
    n_members: int = 8,
    per_member_labels: bool = False,
    epochs: int = 60,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    batch_size: int | None = None,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 5,
    patience: int = 15,
    score_fn: Callable[[dict, dict], float] | None = None,
    on_epoch: Callable[[int, float, torch.Tensor], None] | None = None,
) -> tuple[dict, dict, list[float]]:
    """Train `n_members` models simultaneously. Returns (params, buffers, losses).

    Each member gets its own initialisation (different seed) but sees the same
    batches, which is the cheap form of ensembling: the diversity comes from
    initialisation and from the loss landscape, not from resampling the data.

    **`per_member_labels` is how the spare VRAM earns its keep.** With it, `data.y`
    and `data.mask` carry a leading member axis and each member trains against
    its *own* label matrix — which turns one run into K independent draws. That is
    what builds a permutation **null distribution** instead of a single realisation
    of it: comparing a real effect against one permuted draw is the same mistake
    as reading a 15/21 tally, since at effects of 0.01–0.03 the permutation arm's
    own sampling variance is the same size as the thing being measured.

    The memory goes to **concurrency**, not to a larger batch: batch size buys
    gradient-noise reduction this dataset has no use for, while independent draws
    are precisely the binding resource for resolving small effects.
    """
    torch.manual_seed(seed)
    members = []
    for k in range(n_members):
        torch.manual_seed(seed + k * 7919)
        members.append(make_model(k).to(device))

    params, buffers = stack_module_state(members)
    base = members[0].to("meta")

    # With per-member labels the batch slice has a leading member axis, so the
    # label tensors map over dim 0 alongside the parameters instead of being
    # broadcast.
    label_dims = 0 if per_member_labels else None

    def loss_fn(p, b, xb, yb, mb):
        out = functional_call(base, (p, b), (), {"x": xb})
        preds = out if isinstance(out, torch.Tensor) else out["logits"]
        diff = (preds - yb) ** 2
        # Masked mean: a cell nobody measured must contribute nothing, not zero.
        denom = mb.sum().clamp(min=1.0)
        return (diff * mb).sum() / denom

    # `randomness="different"` is required because dropout is a random op, and
    # it is also what we want: each member must draw its own dropout mask, or
    # the "ensemble" is K models sharing one source of noise and its spread
    # understates the uncertainty it exists to measure.
    grad_fn = vmap(grad_and_value(loss_fn),
                   in_dims=(0, 0, None, label_dims, label_dims),
                   randomness="different")

    opt = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=weight_decay,
                            fused=torch.cuda.is_available())

    if batch_size is None:
        batch_size = suggest_batch_size(data, 512, data.y.shape[1], n_members)
    batch_size = min(batch_size, data.n)

    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[fast] {n_members} members, batch {batch_size}, "
          f"data {data.bytes()/1e6:.0f} MB resident, {reserved:.1f} GB reserved", flush=True)

    history: list[float] = []
    scores: list[float] = []
    best: dict = {"value": None, "epoch": -1, "params": None, "buffers": None}
    since_best = 0

    for epoch in range(epochs):
        t0 = time.time()
        perm = torch.randperm(data.n, device=device)
        total = 0.0
        steps = 0
        start = 0
        while start < data.n:
            idx = perm[start:start + batch_size]
            yb = data.y[:, idx] if per_member_labels else data.y[idx]
            mb = data.mask[:, idx] if per_member_labels else data.mask[idx]
            try:
                grads, losses = grad_fn(params, buffers, data.x[idx], yb, mb)
            except torch.OutOfMemoryError:
                # Back off and carry on rather than losing the run. Halving is
                # the right step: it converges in a few tries and never thrashes.
                if batch_size <= 128:
                    raise
                batch_size //= 2
                torch.cuda.empty_cache()
                print(f"[fast] OOM — batch halved to {batch_size}", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            for name, g in grads.items():
                params[name].grad = g
            opt.step()
            total += float(losses.mean())
            steps += 1
            start += batch_size

        mean_loss = total / max(1, steps)
        history.append(mean_loss)

        # **Early stopping, on a validation score, with a best-epoch snapshot.**
        # The first version of this ran a fixed 80 epochs and finished at a
        # training loss of 0.0355 — far past the epoch-8 peak every earlier run
        # found — and lost to the single model it was meant to improve on. The
        # speed was real; removing the discipline that made the single model
        # good was the mistake.
        if score_fn is not None:
            value = score_fn(params, buffers)
            scores.append(value)
            if best["value"] is None or value < best["value"]:
                best.update(
                    value=value, epoch=epoch,
                    params={k: v.detach().clone() for k, v in params.items()},
                    buffers={k: v.detach().clone() for k, v in buffers.items()},
                )
                since_best = 0
            else:
                since_best += 1
            if since_best >= patience:
                print(f"[fast] stopping at epoch {epoch}: no improvement for "
                      f"{patience} epochs (best {best['value']:.4f} @ep{best['epoch']})",
                      flush=True)
                break
        if epoch % log_every == 0 or epoch == epochs - 1:
            peak = torch.cuda.max_memory_allocated() / 1e9
            extra = f" val {scores[-1]:.4f}" if scores else ""
            print(f"[fast] epoch {epoch:3d} loss {mean_loss:.4f}{extra} "
                  f"{time.time()-t0:.2f}s peak {peak:.1f} GB", flush=True)
        if on_epoch is not None:
            on_epoch(epoch, mean_loss, params)

    # Hand back the best snapshot, not whatever the last epoch happened to be.
    if best["params"] is not None:
        print(f"[fast] returning best epoch {best['epoch']} (val {best['value']:.4f})", flush=True)
        return best["params"], best["buffers"], history
    return params, buffers, history


def predict_ensemble(base: nn.Module, params: dict, buffers: dict,
                     x: torch.Tensor, batch_size: int = 8192,
                     mc_dropout: bool = False) -> torch.Tensor:
    """Per-member predictions, shape (members, rows, tasks).

    Returned un-averaged on purpose: the **spread across members is the
    uncertainty**, and averaging here would throw away the thing worth having.

    **`base.eval()` is called here, and that is a fix rather than a tidy-up.**
    A freshly constructed module has `training=True`, and no caller in this
    repo ever set otherwise — so every prediction this programme made, from
    early-stopping validation scores to the shipped artifact's card numbers,
    ran with dropout active at 0.1. It is consistent across arms, so the
    comparative results stand, but it is wrong in two ways that matter here:
    it adds noise that penalises the neural arm against gradient boosting,
    and it puts dropout noise inside the member spread that Phase 4 wants to
    serve as calibrated uncertainty. `mc_dropout=True` asks for the old
    behaviour deliberately, which is the only context where it is correct.
    """
    base.train(mc_dropout)
    def one(p, b, xb):
        out = functional_call(base, (p, b), (), {"x": xb})
        return out if isinstance(out, torch.Tensor) else out["logits"]

    # At prediction time the modules are in eval mode, so there is no
    # randomness left to vary; "different" is kept for safety if a caller
    # predicts with dropout on (MC-dropout), where per-member draws are correct.
    batched = vmap(one, in_dims=(0, 0, None), randomness="different")
    chunks = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            chunks.append(batched(params, buffers, x[start:start + batch_size]))
    return torch.cat(chunks, dim=1)
