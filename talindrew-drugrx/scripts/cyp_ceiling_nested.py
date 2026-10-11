"""Two corrections before anything else: the noise ceiling, and nested selection.

**1. Is CYP2D6 a modelling failure or an assay-noise ceiling?** Those take
opposite actions. 0.6013 against a 0.6116 median floor means the model barely
beats predicting the median — but if replicate variance on 2D6 is wide, 0.60 is
near-optimal and effort should go to CYP1A2 (0.4760) instead, which has real room
and no mechanistic excuse. The challenge ships a per-compound `_std` for every
label, so the signal-to-noise ratio is measurable directly rather than argued.

**2. The blend weight was selected on the fold the gain was measured on.** w=0.45
was chosen by scanning a continuous parameter on validation and reporting the
best point; the +0.0283 "improvement" with a CI excluding zero is conditional on
that choice, and a bootstrap does not correct for selection. At a gain of 0.028
against a bootstrap sd of 0.03 this is exactly the ratio where selection optimism
matters. Nested selection fixes it: choose w on inner folds, measure on an outer
fold never used for the choice, and report the gain at the **inner-chosen** w
rather than the outer-best. The per-endpoint offsets get the same treatment —
0.003 is well inside what pure selection noise produces on a fold this size.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import macro_st_rae, st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def noise_ceiling(frame) -> dict:
    """Signal-to-noise per endpoint from the shipped per-compound std."""
    out = {}
    for c in PRIMARY:
        y = frame[c].to_numpy(dtype=float)
        sd = frame[f"{c}_std"].to_numpy(dtype=float) if f"{c}_std" in frame else np.full_like(y, np.nan)
        k = np.isfinite(y) & np.isfinite(sd)
        if k.sum() < 50:
            continue
        total_var = float(np.var(y[k]))
        noise_var = float(np.mean(sd[k] ** 2))
        # Fraction of observed variance that is measurement noise. A perfect
        # model cannot explain it, so (1 - frac) bounds achievable R².
        frac = noise_var / total_var if total_var > 0 else np.nan
        out[c] = {"n": int(k.sum()), "label_sd": float(np.std(y[k])),
                  "median_measurement_sd": float(np.median(sd[k])),
                  "noise_var_fraction": frac, "max_achievable_r2": float(max(0.0, 1 - frac))}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/ceiling_nested.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame

    ceiling = noise_ceiling(frame)
    print("\n=== assay noise ceiling (from the shipped per-compound std) ===", flush=True)
    for c, v in ceiling.items():
        print(f"  {c}: n={v['n']:5d} label sd {v['label_sd']:.3f} | measurement sd "
              f"{v['median_measurement_sd']:.3f} | noise is {v['noise_var_fraction']:.1%} of "
              f"variance | max achievable R2 {v['max_achievable_r2']:.3f}", flush=True)

    # ── nested selection of the blend weight and offsets ─────────────
    import xgboost as xgb

    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(dtype=float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(dtype=float)

    rng = np.random.default_rng(0)
    outer_naive, outer_nested = [], []
    for fold in range(args.folds):
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.80,
                                      frac_valid=0.10, seed=100 + fold)
        outer = np.concatenate([va, rest]) if len(rest) else va
        # inner split of the training part only
        inner_tr, inner_va, _r = scaffold_split([frame["smiles"].iloc[i] for i in tr],
                                                frac_train=0.85, frac_valid=0.10, seed=200 + fold)
        inner_tr, inner_va = tr[inner_tr], tr[inner_va]

        _, stats = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stats)

        def fit(idx_tr, idx_eval, seed):
            p = {}
            for c in PRIMARY:
                y = frame[c].to_numpy(dtype=float)
                a = idx_tr[np.isfinite(y[idx_tr])]
                b = idx_eval[np.isfinite(y[idx_eval])]
                if len(a) < 50 or len(b) < 10:
                    p[c] = np.full(len(idx_eval), np.nanmedian(y[idx_tr])); continue
                m1 = xgb.XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=7,
                                      subsample=0.8, colsample_bytree=0.5, device="cuda",
                                      tree_method="hist", verbosity=0, eval_metric="mae",
                                      early_stopping_rounds=60, random_state=seed)
                m1.fit(xs[a], y[a], eval_set=[(xs[b], y[b])], verbose=False)
                m2 = xgb.XGBRegressor(n_estimators=700, learning_rate=0.05, max_depth=4,
                                      subsample=0.6, colsample_bytree=0.8, device="cuda",
                                      tree_method="hist", verbosity=0, eval_metric="mae",
                                      early_stopping_rounds=60, random_state=seed + 1)
                m2.fit(xs[a], y[a], eval_set=[(xs[b], y[b])], verbose=False)
                p[c] = (m1.predict(xs[idx_eval]), m2.predict(xs[idx_eval]))
            return p

        inner = fit(inner_tr, inner_va, fold)
        truth_in = {c: (frame[c].to_numpy()[inner_va], lo[inner_va, i], hi[inner_va, i])
                    for i, c in enumerate(PRIMARY)}
        best_w, best_s = 1.0, None
        for w in np.arange(0.0, 1.01, 0.05):
            mix = {c: w * inner[c][0] + (1 - w) * inner[c][1] for c in PRIMARY}
            s, _ = macro_st_rae(mix, truth_in)
            if best_s is None or s < best_s:
                best_s, best_w = s, float(w)

        out_p = fit(tr, outer, fold)
        truth_out = {c: (frame[c].to_numpy()[outer], lo[outer, i], hi[outer, i])
                     for i, c in enumerate(PRIMARY)}
        nested = {c: best_w * out_p[c][0] + (1 - best_w) * out_p[c][1] for c in PRIMARY}
        s_nested, _ = macro_st_rae(nested, truth_out)
        # Naive: pick the weight ON the outer fold, which is what we did before.
        naive_best = min(
            (macro_st_rae({c: w * out_p[c][0] + (1 - w) * out_p[c][1] for c in PRIMARY},
                          truth_out)[0] for w in np.arange(0.0, 1.01, 0.05)))
        single, _ = macro_st_rae({c: out_p[c][0] for c in PRIMARY}, truth_out)
        outer_nested.append(s_nested - single)
        outer_naive.append(naive_best - single)
        print(f"  fold {fold}: inner-chosen w={best_w:.2f} | single {single:.4f} "
              f"| nested-blend {s_nested:.4f} ({s_nested-single:+.4f}) "
              f"| naive-best {naive_best:.4f} ({naive_best-single:+.4f})", flush=True)

    nn, na = np.array(outer_nested), np.array(outer_naive)
    print(f"\nblend gain, HONEST (nested): {nn.mean():+.4f} ± {nn.std():.4f} over {len(nn)} folds")
    print(f"blend gain, OPTIMISTIC (weight chosen on the measured fold): {na.mean():+.4f}")
    print(f"selection optimism: {na.mean()-nn.mean():+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"noise_ceiling": ceiling, "nested_gain_mean": float(nn.mean()),
         "nested_gain_sd": float(nn.std()), "naive_gain_mean": float(na.mean()),
         "selection_optimism": float(na.mean() - nn.mean()),
         "per_fold_nested": nn.tolist(), "per_fold_naive": na.tolist()},
        indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
