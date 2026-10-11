"""Re-measure every baseline under the OFFICIAL metric, and size the damage.

The denominator we used for this programme's whole first phase was the standard
RAE one (`sum|y - mean(y)|`). The organisers instead soft-threshold the naive mean
predictor too, which makes 1.0 mean "as good as predicting the mean". Our numbers
read ~0.44 where the leaderboard scored the same model at 0.8752.

Two questions this answers, both of which change what to do next:

1. **Was it only mis-measurement, or is there also a generalisation gap?** If the
   validation score under the official metric is ~0.87, the model behaves the same
   on validation and test and there is no gap to hunt -- we were simply reading the
   wrong number. If validation still reads ~0.5 officially, the test set is genuinely
   harder and the applicability-domain question is live.
2. **How far above the trivial baseline are we really?** Under the official metric
   the mean predictor is exactly 1.0 by construction, so this prints the honest
   margin for every arm, including the constant floors.

Both metrics are printed side by side so the correction factor is visible per
endpoint rather than assumed uniform -- the macro average re-weights endpoints
differently under the two denominators, which is the part of the correction that
can actually flip a decision.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load                        # noqa: E402
from drugrx.cyp.metric import st_rae, st_rae_mad                 # noqa: E402
from drugrx.data.splits import scaffold_split                    # noqa: E402
from drugrx.features import featurise, standardise_descriptors   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/cyp_remeasure/remeasure.json")
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo_all = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi_all = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    import xgboost as xgb

    rows = []
    for rep in range(args.repeats):
        tr, te, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=rep)
        te = np.concatenate([te, rest]) if len(rest) else te
        _, stt = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stt)
        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            itr = tr[np.isfinite(y[tr])]; ite = te[np.isfinite(y[te])]
            if len(itr) < 100 or len(ite) < 40:
                continue
            lor, hir, yr = lo_all[ite, i], hi_all[ite, i], y[ite]
            m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                 device="cuda", tree_method="hist", verbosity=0,
                                 eval_metric="mae", early_stopping_rounds=80)
            m.fit(xs[itr], y[itr], eval_set=[(xs[ite], y[ite])], verbose=False)
            pred = m.predict(xs[ite])
            arms = {
                "xgb": pred,
                # The two constant floors, which the official metric makes
                # interpretable: train-mean must land near 1.0.
                "train_mean": np.full(len(ite), np.nanmean(y[itr])),
                "train_median": np.full(len(ite), np.nanmedian(y[itr])),
                "interval_mid": np.full(len(ite), np.nanmedian(
                    (lo_all[itr, i] + hi_all[itr, i]) / 2.0)),
            }
            for name, p in arms.items():
                rows.append({"rep": rep, "endpoint": c, "arm": name,
                             "official": st_rae(p, yr, lor, hir),
                             "mad": st_rae_mad(p, yr, lor, hir),
                             "mae": float(np.mean(np.abs(p - yr))),
                             "pred_sd": float(np.std(p)), "true_sd": float(np.std(yr)),
                             "n": int(len(ite))})

    import pandas as pd
    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    print("\n[remeasure] macro, both denominators (3 scaffold repeats)", flush=True)
    print(f"    {'arm':14s} {'OFFICIAL':>9s} {'old (mad)':>10s} {'ratio':>7s} "
          f"{'MAE':>7s} {'pred sd':>8s} {'true sd':>8s}", flush=True)
    for arm, g in df.groupby("arm"):
        mo = g.groupby("rep")["official"].mean().mean()
        mm = g.groupby("rep")["mad"].mean().mean()
        print(f"    {arm:14s} {mo:9.4f} {mm:10.4f} {mm/mo:7.2f} "
              f"{g['mae'].mean():7.4f} {g['pred_sd'].mean():8.3f} {g['true_sd'].mean():8.3f}")

    print("\n[remeasure] per endpoint (xgb), to show the correction is NOT uniform", flush=True)
    for c, g in df[df.arm == "xgb"].groupby("endpoint"):
        print(f"    {c[:24]:24s} official {g['official'].mean():.4f}  "
              f"old {g['mad'].mean():.4f}  ratio {g['mad'].mean()/g['official'].mean():.2f}")
    print("\n[remeasure] the leaderboard says our submission scored 0.8752 "
          "(rank 163); 1.0 == predicting the mean.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
