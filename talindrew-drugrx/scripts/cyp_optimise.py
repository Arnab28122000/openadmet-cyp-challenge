"""Exploit the metric, not just the model: blending and per-endpoint offsets.

Every model arm so far lands at 0.434–0.443 with a bootstrap sd of ~0.03 — they
are indistinguishable, which is this programme's recurring result. So the
remaining gains are not in the model class. They are in the **metric**, which is
unusual enough to be worth attacking directly:

* Error is measured to the **nearest credible-interval bound**, and a prediction
  anywhere inside scores **zero**. The intervals are not symmetric about the point
  estimate and are much wider for low-activity compounds (CYP3A4: 40% of labels
  below pIC50 4).
* Therefore the ST-RAE-optimal prediction is **not** the conditional mean. Shifting
  predictions toward the wider side of the interval converts near-misses into
  exact zeros, and that is free score the MAE-optimal model leaves behind.

Two cheap, data-backed moves, both fitted on validation and applied blind to test:

1. **Per-endpoint additive offset** — one scalar per CYP, chosen to minimise
   validation ST-RAE. Four parameters total, so the overfitting risk is small and
   is measured by bootstrap rather than assumed.
2. **Blend weight between the net and the gradient-boosting model** — they are
   statistically tied, which is precisely when averaging helps, because their
   errors are not perfectly correlated.

Everything is chosen on validation only. The blinded test set is never consulted.
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

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.metric import macro_st_rae, st_rae  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402


def fit_offset(pred, y, lo, hi, grid=np.arange(-1.5, 1.51, 0.02)):
    """Additive shift minimising ST-RAE. Returns (offset, score_before, score_after)."""
    base = st_rae(pred, y, lo, hi)
    scores = [st_rae(pred + d, y, lo, hi) for d in grid]
    i = int(np.nanargmin(scores))
    return float(grid[i]), float(base), float(scores[i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xgb-rounds", type=int, default=1200)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    x_test = featurise(data.test["smiles"].tolist(), workers=args.workers,
                       counts=True, desc_mode="rdkit2d", extra_fps=True)

    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=args.seed)
    va = np.concatenate([va, rest]) if len(rest) else va
    _, stats = standardise_descriptors(x[tr])
    x, _ = standardise_descriptors(x, stats=stats)
    x_test, _ = standardise_descriptors(x_test, stats=stats)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(dtype=float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(dtype=float)

    # ── gradient boosting, per endpoint, with test predictions ───────
    import xgboost as xgb

    val_xgb, test_xgb = {}, {}
    t0 = time.time()
    for c in PRIMARY:
        y = frame[c].to_numpy(dtype=float)
        itr = tr[np.isfinite(y[tr])]
        ival = va[np.isfinite(y[va])]
        m = xgb.XGBRegressor(n_estimators=args.xgb_rounds, learning_rate=0.03, max_depth=7,
                             subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                             device="cuda", tree_method="hist", verbosity=0,
                             eval_metric="mae", early_stopping_rounds=80)
        m.fit(x[itr], y[itr], eval_set=[(x[ival], y[ival])], verbose=False)
        val_xgb[c] = m.predict(x[va])
        test_xgb[c] = m.predict(x_test)
    print(f"[opt] xgb trained in {time.time()-t0:.0f}s", flush=True)

    # ── the trained ensemble's predictions, if present ───────────────
    val_nn, test_nn = {}, {}
    nn_path = Path("/content/out/cyp/val_pred.npz")
    if nn_path.exists():
        z = np.load(nn_path, allow_pickle=True)
        for i, c in enumerate(PRIMARY):
            val_nn[c] = z["val"][:, i]
            test_nn[c] = z["test"][:, i]
        print("[opt] loaded neural-net predictions", flush=True)

    truth = {c: (frame[c].to_numpy()[va], lo[va, i], hi[va, i]) for i, c in enumerate(PRIMARY)}
    report = {}

    base_macro, base_per = macro_st_rae(val_xgb, truth)
    print(f"[opt] xgb alone: {base_macro:.4f}", flush=True)

    # ── blend, if both model families are available ──────────────────
    best_w, blended_val, blended_test = 1.0, val_xgb, test_xgb
    if val_nn:
        best, best_w = None, 1.0
        for w in np.arange(0.0, 1.01, 0.05):
            mix = {c: w * val_xgb[c] + (1 - w) * val_nn[c] for c in PRIMARY}
            s, _ = macro_st_rae(mix, truth)
            if best is None or s < best:
                best, best_w = s, float(w)
        blended_val = {c: best_w * val_xgb[c] + (1 - best_w) * val_nn[c] for c in PRIMARY}
        blended_test = {c: best_w * test_xgb[c] + (1 - best_w) * test_nn[c] for c in PRIMARY}
        print(f"[opt] best blend w(xgb)={best_w:.2f} → {best:.4f}", flush=True)
        report["blend_weight_xgb"] = best_w
        report["blend_macro"] = best

    # ── per-endpoint offsets fitted on validation ────────────────────
    offsets, after = {}, {}
    for i, c in enumerate(PRIMARY):
        y, l, h = truth[c]
        d, before, aft = fit_offset(np.asarray(blended_val[c]), y, l, h)
        offsets[c], after[c] = d, aft
        print(f"    {c}: offset {d:+.2f}  {before:.4f} → {aft:.4f}", flush=True)

    final_val = {c: blended_val[c] + offsets[c] for c in PRIMARY}
    final_macro, final_per = macro_st_rae(final_val, truth)
    print(f"\n[opt] FINAL validation MA-ST-RAE {final_macro:.4f} "
          f"(from {base_macro:.4f} xgb-alone)", flush=True)

    # Bootstrap the improvement so a small gain is not read as a result.
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(400):
        per_draw = []
        for i, c in enumerate(PRIMARY):
            y, l, h = truth[c]
            k = np.isfinite(y)
            idx = rng.integers(0, int(k.sum()), int(k.sum()))
            yy, ll, hh = y[k][idx], l[k][idx], h[k][idx]
            a = st_rae(np.asarray(val_xgb[c])[k][idx], yy, ll, hh)
            b = st_rae(np.asarray(final_val[c])[k][idx], yy, ll, hh)
            per_draw.append(a - b)
        gains.append(np.mean(per_draw))
    gains = np.array(gains)
    print(f"[opt] improvement over xgb-alone: {gains.mean():+.4f} "
          f"[{np.percentile(gains,2.5):+.4f}, {np.percentile(gains,97.5):+.4f}] "
          f"→ {'REAL' if np.percentile(gains,2.5) > 0 else 'inside noise'}", flush=True)

    sub = pd.DataFrame({"Molecule_Name": data.test["Molecule_Name"],
                        "SMILES": data.test["SMILES"]})
    for c in PRIMARY:
        sub[c] = np.asarray(blended_test[c]) + offsets[c]
    sub.to_csv(out / "submission.csv", index=False)

    report.update({
        "xgb_alone": base_macro, "final": final_macro, "per_endpoint": final_per,
        "offsets": offsets,
        "improvement_mean": float(gains.mean()),
        "improvement_ci": [float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))],
        "pred_std": {c: float(sub[c].std()) for c in PRIMARY},
    })
    (out / "optimise_report.json").write_text(json.dumps(report, indent=2, default=float))
    print("\n[opt] submission column spread (organisers reject near-constant):", flush=True)
    for c in PRIMARY:
        print(f"    {c}: sd {sub[c].std():.3f} range {sub[c].min():.2f}–{sub[c].max():.2f}",
              flush=True)
    print(f"[opt] wrote {out/'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
