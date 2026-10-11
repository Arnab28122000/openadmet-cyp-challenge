"""Train the CYP challenge model and write a submission.

Design follows what this programme measured rather than what is fashionable:

* **Features are count-ECFP4 + RDKit2D + Avalon + ErG** (3,091 dims). H5 found
  Avalon + ErG improved 14/21 TDC endpoints, and **three CYP endpoints were in the
  top five gains** — against my own prediction that the CYP substructure panel
  would not benefit. This challenge is a CYP panel, so that is the single most
  relevant result we hold.
* **The loss is interval-aware**, because the metric is. A prediction anywhere
  inside the credible interval scores zero, so chasing the point estimate spends
  capacity on error the leaderboard does not count. CYP3A4 alone has 40% of its
  labels below pIC50 4, where the intervals are widest.
* **20 auxiliary heads carry 47,360 labels** — TDI-condition pIC50, Emax under
  *both* assay conditions, the four `is_TDI` flags, and single-concentration log2
  fold-change — against 6,525 primary ones. The primary task is thin; the
  auxiliary supervision is where the chemistry comes from. H7a showed auxiliary
  heads give a real if small gain, and that was on an encoder already well fed.
  Here it should matter more.
  (It was 14 heads / 38,137 labels until an `endswith` filter in the loader was
  found to be silently dropping 6 heads — including the Emax values measured under
  the *primary* assay condition, the most on-target auxiliary signal available.
  See `drugrx/cyp/data.py`.)
* **An ensemble, trained concurrently under `vmap`.** Averaging reduces variance
  on a 750-compound test set where single-model noise is large, and the spread is
  the uncertainty we would need before claiming anything.

Validation is a **scaffold split with ST-RAE**, never random: this programme has
measured what random splits do to an estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import CYPS, PRIMARY, load  # noqa: E402
from drugrx.cyp.metric import macro_st_rae  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402


class CypNet(nn.Module):
    """Shared trunk, one output per head. Primary heads come first."""

    def __init__(self, in_dim: int, n_heads: int, d_model: int = 1024,
                 n_layers: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)]
            d = d_model
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def build_loss(n_primary: int, aux_weight: float, huber: float = 1.0):
    """Interval hinge on the primary heads, masked MSE on the auxiliary ones."""

    def loss_fn(preds, y, mask, lo, hi, lo_mask):
        p_pred, p_lo, p_hi = preds[:, :n_primary], lo, hi
        below = torch.clamp(p_lo - p_pred, min=0.0)
        above = torch.clamp(p_pred - p_hi, min=0.0)
        dist = below + above
        quad = torch.minimum(dist, torch.full_like(dist, huber))
        primary = 0.5 * quad.pow(2) + huber * (dist - quad)
        primary = (primary * lo_mask).sum() / lo_mask.sum().clamp(min=1.0)

        a_pred, a_y, a_m = preds[:, n_primary:], y[:, n_primary:], mask[:, n_primary:]
        aux = (((a_pred - a_y) ** 2) * a_m).sum() / a_m.sum().clamp(min=1.0)
        return primary + aux_weight * aux

    return loss_fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp")
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(workers=args.workers)
    frame = data.frame
    heads = [*PRIMARY, *data.auxiliary]

    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    x_test = featurise(data.test["smiles"].tolist(), workers=args.workers,
                       counts=True, desc_mode="rdkit2d", extra_fps=True)

    y = frame[heads].to_numpy(dtype=np.float32)
    mask = ~np.isnan(y)
    # Standardise every head so no target dominates on units; the primary heads
    # are kept on their native pIC50 scale because the interval bounds are too.
    norm = {}
    for i, h in enumerate(heads):
        if h in PRIMARY:
            norm[h] = (0.0, 1.0)
            continue
        col = y[:, i][mask[:, i]]
        mu, sd = (float(col.mean()), float(col.std()) or 1.0) if col.size else (0.0, 1.0)
        y[:, i] = (y[:, i] - mu) / sd
        norm[h] = (mu, sd)
    y = np.nan_to_num(y, nan=0.0)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(dtype=np.float32)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(dtype=np.float32)
    lo_mask = np.isfinite(lo) & np.isfinite(hi)
    lo, hi = np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=0.0)

    # scaffold_split always reserves a third fold, so the two fractions must sum
    # to less than one; the remainder is merged into validation here because this
    # task has its own blinded test set and needs no local hold-out beyond it.
    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=args.seed)
    va = np.concatenate([va, rest]) if len(rest) else va
    print(f"[cyp] scaffold split: {len(tr)} train, {len(va)} valid", flush=True)

    _, stats = standardise_descriptors(x[tr])
    x, _s = standardise_descriptors(x, stats=stats)
    x_test, _s = standardise_descriptors(x_test, stats=stats)

    dev = args.device
    X = torch.from_numpy(np.ascontiguousarray(x, np.float32)).to(dev)
    Y = torch.from_numpy(np.ascontiguousarray(y, np.float32)).to(dev)
    M = torch.from_numpy(np.ascontiguousarray(mask)).to(dev)
    LO = torch.from_numpy(np.ascontiguousarray(lo, np.float32)).to(dev)
    HI = torch.from_numpy(np.ascontiguousarray(hi, np.float32)).to(dev)
    LM = torch.from_numpy(np.ascontiguousarray(lo_mask)).to(dev)
    XT = torch.from_numpy(np.ascontiguousarray(x_test, np.float32)).to(dev)
    tr_t = torch.from_numpy(tr).to(dev)

    members = []
    for k in range(args.members):
        torch.manual_seed(args.seed + k * 7919)
        members.append(CypNet(x.shape[1], len(heads), args.d_model, 3, args.dropout).to(dev))
    params, buffers = stack_module_state(members)
    base = members[0].to("meta")
    loss_core = build_loss(len(PRIMARY), args.aux_weight)

    def loss_fn(p, b, xb, yb, mb, lob, hib, lmb):
        return loss_core(functional_call(base, (p, b), (), {"x": xb}), yb, mb, lob, hib, lmb)

    grad_fn = vmap(grad_and_value(loss_fn), in_dims=(0, 0, None, None, None, None, None, None),
                   randomness="different")
    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.01,
                            fused=torch.cuda.is_available())

    def predict(xs: torch.Tensor, mc_dropout: bool = False) -> torch.Tensor:
        """Predict with the ensemble. **`eval()` matters here and was missing.**

        Without it the meta module stays in train mode and `functional_call`
        applies dropout at inference — 0.15 here, so 15% of units are zeroed in
        every prediction. That handicaps the neural arm in every comparison
        against gradient boosting, which has no such behaviour, and it inflates
        the member spread that would otherwise serve as calibrated uncertainty.
        A peer session measured the same bug in `fast_train.predict_ensemble`:
        eval mode better on 18/21 endpoints, mean gain +0.0024, member-spread
        inflated 1.27x.

        `mc_dropout=True` asks for the old behaviour deliberately, which is a
        legitimate uncertainty estimate — but it is not how a model should be
        scored.
        """
        base.train(mc_dropout)

        def one(p, b, xb):
            return functional_call(base, (p, b), (), {"x": xb})
        batched = vmap(one, in_dims=(0, 0, None), randomness="different")
        chunks = []
        with torch.no_grad():
            for s in range(0, xs.shape[0], 8192):
                chunks.append(batched(params, buffers, xs[s:s + 8192]))
        return torch.cat(chunks, dim=1)

    def valid_strae() -> tuple[float, dict]:
        mean = predict(X[va]).mean(0).float().cpu().numpy()
        preds = {c: mean[:, i] for i, c in enumerate(PRIMARY)}
        truth = {c: (frame[c].to_numpy()[va], lo[va, i], hi[va, i])
                 for i, c in enumerate(PRIMARY)}
        return macro_st_rae(preds, truth)

    best = {"score": float("inf"), "epoch": -1, "params": None, "buffers": None}
    since = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        perm = tr_t[torch.randperm(len(tr_t), device=dev)]
        total, steps, start = 0.0, 0, 0
        bs = args.batch_size
        while start < len(perm):
            idx = perm[start:start + bs]
            try:
                grads, losses = grad_fn(params, buffers, X[idx], Y[idx], M[idx],
                                        LO[idx], HI[idx], LM[idx])
            except torch.OutOfMemoryError:
                if bs <= 128:
                    raise
                bs //= 2
                torch.cuda.empty_cache()
                print(f"[cyp] OOM, batch -> {bs}", flush=True)
                continue
            opt.zero_grad(set_to_none=True)
            for n_, g in grads.items():
                params[n_].grad = g
            opt.step()
            total += float(losses.mean()); steps += 1; start += bs

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            score, per = valid_strae()
            if score < best["score"]:
                best.update(score=score, epoch=epoch,
                            params={k: v.detach().clone() for k, v in params.items()},
                            buffers={k: v.detach().clone() for k, v in buffers.items()})
                since = 0
            else:
                since += 5
            print(f"[cyp] epoch {epoch:4d} loss {total/max(1,steps):.4f} "
                  f"MA-ST-RAE {score:.4f} best {best['score']:.4f}@{best['epoch']} "
                  f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
            if since >= args.patience:
                print(f"[cyp] stopping at {epoch}", flush=True)
                break

    if best["params"] is not None:
        params, buffers = best["params"], best["buffers"]
    score, per = valid_strae()
    print(f"\n[cyp] best MA-ST-RAE {best['score']:.4f} (epoch {best['epoch']}), "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    for k, v in per.items():
        print(f"    {k}: {v:.4f}", flush=True)

    val_mean = predict(X[va]).mean(0).float().cpu().numpy()
    test_pred = predict(XT)
    mean = test_pred.mean(0).float().cpu().numpy()
    # Saved so the blending/offset step can reuse them without retraining.
    np.savez(out / "val_pred.npz", val=val_mean[:, :len(PRIMARY)],
             test=mean[:, :len(PRIMARY)], va_index=va)
    spread = test_pred.std(0).float().cpu().numpy()
    sub = pd.DataFrame({"Molecule_Name": data.test["Molecule_Name"],
                        "SMILES": data.test["SMILES"]})
    for i, c in enumerate(PRIMARY):
        sub[c] = mean[:, i]
    sub.to_csv(out / "submission.csv", index=False)

    (out / "report.json").write_text(json.dumps({
        "valid_macro_st_rae": best["score"], "per_endpoint": per,
        "best_epoch": best["epoch"], "members": args.members,
        "n_train": int(len(tr)), "n_valid": int(len(va)),
        "heads": heads, "aux_weight": args.aux_weight,
        "mean_ensemble_spread": float(np.nanmean(spread[:, :len(PRIMARY)])),
        "pred_std_per_endpoint": {c: float(mean[:, i].std()) for i, c in enumerate(PRIMARY)},
    }, indent=2, default=float))
    print(f"[cyp] wrote {out/'submission.csv'} and report.json", flush=True)
    # The organisers reject near-constant columns; check before submitting.
    for i, c in enumerate(PRIMARY):
        print(f"    {c}: pred sd {mean[:, i].std():.3f} range "
              f"{mean[:, i].min():.2f}–{mean[:, i].max():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
