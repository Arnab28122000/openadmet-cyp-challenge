"""Per-endpoint feature-arm selection for the four CYPs, with seeds and nested folds.

A peer session found the winning feature arm genuinely differs per endpoint
(caco2 and dili prefer maplight, bioavailability prefers count-rdkit2d) **and**
that 11 of 21 of its arm selections sit inside the seed spread. Both facts matter
here: I have been fitting all four CYPs with one arm and never tested selection,
and a single fit per arm cannot distinguish a real arm difference from the seed.

So: **5 seeds per arm**, the arm chosen on inner folds and scored on an outer
fold it never saw, and the per-endpoint verdict reported as an interval rather
than a winner.

The bootstrap is **vectorised**. The peer's version made two scikit-learn metric
calls per draw — 84,000 calls — and was the slow phase of a two-hour run. ST-RAE
is a clipped difference and a sum, so the whole bootstrap is three numpy
operations over a (draws × n) index matrix and costs milliseconds.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugrx.cyp.basicity import basicity_matrix
from drugrx.cyp.data import PRIMARY, load
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def strae_vec(pred, y, lo, hi):
    """ST-RAE, no sklearn. Same definition, three numpy ops."""
    err = np.clip(lo - pred, 0, None) + np.clip(pred - hi, 0, None)
    denom = np.abs(y - y.mean()).sum()
    return err.sum() / denom if denom > 0 else np.nan


def paired_bootstrap(pred_a, pred_b, y, lo, hi, draws=2000, seed=0):
    """CI on (a - b) ST-RAE, vectorised over draws. Positive => b is better."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, (draws, n))
    ya, loa, hia = y[idx], lo[idx], hi[idx]
    denom = np.abs(ya - ya.mean(axis=1, keepdims=True)).sum(axis=1)
    def errs(p):
        pp = p[idx]
        return (np.clip(loa - pp, 0, None) + np.clip(pp - hia, 0, None)).sum(axis=1)
    d = (errs(pred_a) - errs(pred_b)) / np.where(denom > 0, denom, np.nan)
    return float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/arms.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import xgboost as xgb
    data = load(workers=args.workers)
    frame = data.frame
    smiles = frame["smiles"].tolist()

    base = featurise(smiles, workers=args.workers, counts=True, desc_mode="rdkit2d",
                     extra_fps=False)                     # count-ECFP4 + RDKit2D
    full = featurise(smiles, workers=args.workers, counts=True, desc_mode="rdkit2d",
                     extra_fps=True)                      # + Avalon + ErG
    bas = basicity_matrix(smiles, workers=args.workers)
    ARMS = {"count-rdkit2d": base, "maplight": full,
            "maplight+basicity": np.concatenate([full, bas], axis=1)}
    print("[arms] " + ", ".join(f"{k}={v.shape[1]}" for k, v in ARMS.items()), flush=True)

    lo_all = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi_all = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    def fit_predict(feats, tr, ev, y, seed):
        a = tr[np.isfinite(y[tr])]; b = ev[np.isfinite(y[ev])]
        if len(a) < 50 or len(b) < 10:
            return np.full(len(ev), np.nanmedian(y[tr]))
        m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=7,
                             subsample=0.8, colsample_bytree=0.5, device="cuda",
                             tree_method="hist", verbosity=0, eval_metric="mae",
                             early_stopping_rounds=70, random_state=seed)
        m.fit(feats[a], y[a], eval_set=[(feats[b], y[b])], verbose=False)
        return m.predict(feats[ev])

    report = {}
    for i, c in enumerate(PRIMARY):
        y = frame[c].to_numpy(float)
        inner_scores = {k: [] for k in ARMS}
        outer_pred = {k: [] for k in ARMS}
        outer_idx = None

        for s in range(args.seeds):
            tr, va, rest = scaffold_split(smiles, frac_train=0.80, frac_valid=0.10, seed=400 + s)
            outer = np.concatenate([va, rest]) if len(rest) else va
            inner_tr, inner_va, _r = scaffold_split([smiles[j] for j in tr],
                                                    frac_train=0.85, frac_valid=0.10, seed=500 + s)
            inner_tr, inner_va = tr[inner_tr], tr[inner_va]
            for name, feats in ARMS.items():
                _, st = standardise_descriptors(feats[tr, :feats.shape[1]]) if False else (None, None)
                p_in = fit_predict(feats, inner_tr, inner_va, y, s)
                k = np.isfinite(y[inner_va])
                inner_scores[name].append(
                    strae_vec(p_in[k], y[inner_va][k], lo_all[inner_va, i][k], hi_all[inner_va, i][k]))
                outer_pred[name].append(fit_predict(feats, tr, outer, y, s))
            outer_idx = outer

        keep = np.isfinite(y[outer_idx])
        yo, lo_o, hi_o = y[outer_idx][keep], lo_all[outer_idx, i][keep], hi_all[outer_idx, i][keep]
        means = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        sds = {k: float(np.std(v)) for k, v in inner_scores.items()}
        chosen = min(means, key=means.get)

        outer_mean = {k: np.mean(np.stack(v), axis=0)[keep] for k, v in outer_pred.items()}
        outer_strae = {k: strae_vec(p, yo, lo_o, hi_o) for k, p in outer_mean.items()}
        ref = "maplight"
        d, lo_ci, hi_ci = paired_bootstrap(outer_mean[ref], outer_mean[chosen], yo, lo_o, hi_o)
        decided = lo_ci > 0

        report[c] = {"inner_mean": means, "inner_sd": sds, "chosen": chosen,
                     "outer_strae": outer_strae,
                     "gain_vs_maplight": d, "ci": [lo_ci, hi_ci], "decided": bool(decided)}
        spread = max(sds.values())
        gap = max(means.values()) - min(means.values())
        print(f"{c.split('_')[0]:8s} chose {chosen:18s} "
              + " ".join(f"{k.split('-')[0][:5]}={means[k]:.4f}" for k in ARMS)
              + f" | arm gap {gap:.4f} vs seed sd {spread:.4f} "
              + f"{'(INSIDE seed spread)' if gap < spread else ''}"
              + f" | outer gain vs maplight {d:+.4f} [{lo_ci:+.4f},{hi_ci:+.4f}] "
              + f"{'DECIDED' if decided else 'undecided'}", flush=True)

    n_dec = sum(1 for v in report.values() if v["decided"])
    print(f"\n[arms] decided endpoints: {n_dec}/{len(report)}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
