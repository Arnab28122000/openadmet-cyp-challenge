"""H11: do protonation features rescue CYP2D6? Five folds, per endpoint, with sd."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugrx.cyp.basicity import NAMES, basicity_matrix
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/basicity.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    import xgboost as xgb
    data = load(workers=args.workers)
    frame = data.frame
    smiles = frame["smiles"].tolist()
    x = featurise(smiles, workers=args.workers, counts=True, desc_mode="rdkit2d", extra_fps=True)
    bas = basicity_matrix(smiles, workers=args.workers)
    print(f"[bas] base {x.shape}, basicity {bas.shape}", flush=True)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    res = {c: {"base": [], "with": []} for c in PRIMARY}

    for fold in range(args.folds):
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=300 + fold)
        va = np.concatenate([va, rest]) if len(rest) else va
        _, st = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=st)
        # Basicity columns are counts and charges; standardise them separately so
        # they are on a comparable scale without touching the fingerprint blocks.
        mu, sd = bas[tr].mean(0), bas[tr].std(0); sd[sd < 1e-8] = 1.0
        bs = (bas - mu) / sd
        xb = np.concatenate([xs, bs], axis=1)

        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            a = tr[np.isfinite(y[tr])]; b = va[np.isfinite(y[va])]
            if len(a) < 50 or len(b) < 20:
                continue
            for tag, feats in (("base", xs), ("with", xb)):
                m = xgb.XGBRegressor(n_estimators=900, learning_rate=0.04, max_depth=7,
                                     subsample=0.8, colsample_bytree=0.5, device="cuda",
                                     tree_method="hist", verbosity=0, eval_metric="mae",
                                     early_stopping_rounds=70, random_state=fold)
                m.fit(feats[a], y[a], eval_set=[(feats[b], y[b])], verbose=False)
                p = m.predict(feats[b])
                res[c][tag].append(st_rae(p, y[b], lo[b, i], hi[b, i]))

    print(f"\n{'endpoint':10s} {'base':>16s} {'with basicity':>18s} {'delta':>10s}")
    out = {}
    for c in PRIMARY:
        a, b = np.array(res[c]["base"]), np.array(res[c]["with"])
        if not len(a):
            continue
        d = a.mean() - b.mean()
        out[c] = {"base_mean": float(a.mean()), "base_sd": float(a.std()),
                  "with_mean": float(b.mean()), "with_sd": float(b.std()),
                  "delta": float(d), "folds": len(a),
                  "wins": int((b < a).sum())}
        print(f"{c.split('_')[0]:10s} {a.mean():.4f}±{a.std():.4f}  {b.mean():.4f}±{b.std():.4f}  "
              f"{d:+.4f}  ({int((b<a).sum())}/{len(a)} folds better)", flush=True)
    macro_d = np.mean([v["delta"] for v in out.values()])
    print(f"\nmacro delta {macro_d:+.4f} (positive = basicity helps)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"per_endpoint": out, "macro_delta": float(macro_d)},
                                         indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
