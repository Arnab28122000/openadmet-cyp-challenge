r"""Where does the error live? Bin by label extremity and look.

The structural version of distribution shift is dead: the challenge training set
covers 100% of the blind test compounds at Tanimoto 0.3, median similarity 0.581.
The **label** version is not, and the evidence for it got stronger:

* implied test label sd ~1.29 against training's 0.955;
* R-squared collapses 0.318 (validation) -> 0.045 (test);
* ST-RAE degrades only 0.737 -> 0.875 over the same move.

A metric that degrades mildly while another collapses is diagnostic, not
incidental. Same structures carrying more extreme activities is the definition of
an activity-cliff problem, it is exactly where fingerprint models fail, and it is a
**representation** claim rather than a calibration one.

This is the cheap test and it had not been run. Validation molecules are binned by
label extremity and each bin is characterised by:

* `mean_hinge` -- mean soft-threshold error per molecule, i.e. the actual
  contribution to the ST-RAE **numerator**. Per-bin ST-RAE is deliberately *not*
  reported: its denominator is subset-dependent, so four bins would each be
  normalised differently and the comparison would be meaningless.
* `share_num` -- that bin's share of the total numerator. This is the number that
  says whether the tails dominate the ranked metric.
* `r2`, `mae`, `bias` -- for the diagnostics the leaderboard shows.
* `width` -- mean interval width, because the whole asymmetry hinges on whether
  extreme compounds carry narrow bands (errors charged in full) or wide ones
  (errors forgiven).

If error and numerator mass concentrate in the tails, the acquisition criterion for
new data is **tail coverage**, not structural coverage -- and Veith's 87.6%
structural coverage at 0.0% incremental was screened on the wrong axis. So Veith's
own label distribution is inspected here too, on the same axis.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load                        # noqa: E402
from drugrx.cyp.metric import soft_threshold_error, st_rae       # noqa: E402
from drugrx.data.splits import scaffold_split                    # noqa: E402
from drugrx.features import featurise, standardise_descriptors   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--out", default="/content/out/cyp_tail/tail.json")
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
            if len(itr) < 100 or len(ite) < 60:
                continue
            m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                 device="cuda", tree_method="hist", verbosity=0,
                                 eval_metric="mae", early_stopping_rounds=80)
            m.fit(xs[itr], y[itr], eval_set=[(xs[ite], y[ite])], verbose=False)
            p = m.predict(xs[ite])
            yy, l, h = y[ite], lo_all[ite, i], hi_all[ite, i]
            hinge = soft_threshold_error(p, l, h)
            total = hinge.sum()

            # Bin by the label itself: low activity -> high activity.
            edges = np.quantile(yy, np.linspace(0, 1, args.bins + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            b = np.clip(np.digitize(yy, edges[1:-1]), 0, args.bins - 1)
            for k in range(args.bins):
                msk = b == k
                if msk.sum() < 5:
                    continue
                ss = float(np.sum((yy[msk] - p[msk]) ** 2))
                st = float(np.sum((yy[msk] - yy[msk].mean()) ** 2))
                rows.append({
                    "rep": rep, "endpoint": c, "bin": k, "n": int(msk.sum()),
                    "y_lo": float(yy[msk].min()), "y_hi": float(yy[msk].max()),
                    "y_mean": float(yy[msk].mean()),
                    "mean_hinge": float(hinge[msk].mean()),
                    "share_num": float(hinge[msk].sum() / total) if total > 0 else np.nan,
                    "mae": float(np.mean(np.abs(p[msk] - yy[msk]))),
                    "bias": float(np.mean(p[msk] - yy[msk])),
                    "r2": 1.0 - ss / st if st > 0 else np.nan,
                    "width": float(np.mean(h[msk] - l[msk])),
                    "frac_outside": float((hinge[msk] > 0).mean()),
                })

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    print(f"\n[tail] pooled over endpoints and {args.repeats} repeats, "
          f"bin 0 = least active ... bin {args.bins-1} = most active", flush=True)
    print(f"    {'bin':>3s} {'y range':>14s} {'n':>5s} {'mean_hinge':>11s} "
          f"{'share_num':>10s} {'MAE':>7s} {'bias':>7s} {'R2':>8s} "
          f"{'width':>7s} {'outside':>8s}", flush=True)
    for k, g in df.groupby("bin"):
        print(f"    {k:3d} {g['y_mean'].min():6.2f}-{g['y_mean'].max():6.2f} "
              f"{int(g['n'].mean()):5d} {g['mean_hinge'].mean():11.4f} "
              f"{g['share_num'].mean():10.1%} {g['mae'].mean():7.3f} "
              f"{g['bias'].mean():+7.3f} {g['r2'].mean():8.3f} "
              f"{g['width'].mean():7.3f} {g['frac_outside'].mean():8.1%}")

    top = df[df.bin == df.bin.max()]
    bot = df[df.bin == 0]
    mid = df[df.bin == df.bin.max() // 2]
    print(f"\n[tail] most-active bin carries {top['share_num'].mean():.1%} of the "
          f"ST-RAE numerator at {top['n'].mean()/df.groupby(['rep','endpoint'])['n'].sum().mean():.1%} "
          f"of the molecules", flush=True)
    print(f"[tail] mean hinge: least active {bot['mean_hinge'].mean():.4f}  "
          f"middle {mid['mean_hinge'].mean():.4f}  most active {top['mean_hinge'].mean():.4f}",
          flush=True)
    print(f"[tail] interval width: least active {bot['width'].mean():.3f}  "
          f"most active {top['width'].mean():.3f}", flush=True)
    print(f"[tail] bias: least active {bot['bias'].mean():+.3f}  "
          f"most active {top['bias'].mean():+.3f}  "
          f"(negative = we under-predict that bin)", flush=True)
    conc = top["share_num"].mean() > 2.0 / args.bins
    print(f"\n[tail] VERDICT: error {'DOES' if conc else 'does NOT'} concentrate in the "
          f"active tail -> acquisition criterion is "
          f"{'TAIL coverage' if conc else 'structural coverage'}", flush=True)

    # Re-screen the external source on the axis that matters.
    if Path(args.external).exists():
        ext = pd.read_parquet(args.external)
        vc = [c for c in ext.columns if c.startswith("veith_")]
        print(f"\n[tail] Veith label distribution, on the same axis "
              f"({len(vc)} columns)", flush=True)
        for c in vc[:8]:
            v = ext[c].to_numpy(float); v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            print(f"    {c[:34]:34s} n={len(v):6d} min {v.min():6.2f} "
                  f"max {v.max():6.2f} mean {v.mean():6.2f} sd {v.std():5.2f}")
        print("    challenge, for comparison:", flush=True)
        for c in PRIMARY:
            v = frame[c].to_numpy(float); v = v[np.isfinite(v)]
            print(f"    {c[:34]:34s} n={len(v):6d} min {v.min():6.2f} "
                  f"max {v.max():6.2f} mean {v.mean():6.2f} sd {v.std():5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
