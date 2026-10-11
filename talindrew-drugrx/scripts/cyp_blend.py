"""Blend the two tied-but-different models: xgb on all features, TabICL on 500.

The in-context result was a macro tie with a per-endpoint split:

| endpoint | xgb | icl_top | icl_top - xgb |
|---|---|---|---|
| CYP1A2 | 0.5397 | 0.5335 | **-0.0062** |
| CYP2C9 | 0.3766 | 0.3665 | **-0.0101** |
| CYP2D6 | 0.6077 | 0.6253 | +0.0175 |
| CYP3A4 | 0.3020 | 0.2904 | **-0.0116** |

Macro: 0.4565 vs 0.4597, inside noise (-0.0032 +/- 0.0051). TabICL wins three of
four endpoints and loses one badly enough to erase it -- and its per-repeat sd is
**half** xgb's (0.0043 vs 0.0095). Two models that are statistically tied overall
while disagreeing per endpoint is precisely the condition under which averaging
helps, because the errors are not perfectly correlated; blending is also the only
intervention in this whole programme that has ever cleared the noise floor
(+0.0283, CI [+0.0160, +0.0427]).

**Picking the better model per endpoint is not the experiment.** With four
endpoints and five repeats that is selection on ~20 numbers and would manufacture
a result. So the blend weight is fitted on a **separate split** the score is not
read from, and a single global weight is reported alongside the per-endpoint one
so the cost of the extra freedom is visible rather than assumed away:

* `xgb`, `icl` -- the two arms alone, on the report split.
* `blend50` -- a fixed 50/50 average. **No fitted parameters at all**, so it
  cannot overfit; if the gain is real this should already show it.
* `blend_global` -- one weight for all four endpoints, fitted on the fit split.
* `blend_per_ep` -- one weight per endpoint, fitted on the fit split. Four
  parameters, and the arm most likely to flatter itself.

Paired across repeats against `xgb`.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load          # noqa: E402
from drugrx.cyp.metric import st_rae               # noqa: E402
from drugrx.data.splits import scaffold_split      # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402


def spearman_rank_select(xtr, ytr, k):
    from scipy.stats import rankdata
    yr = rankdata(ytr) - np.mean(rankdata(ytr))
    xr = np.apply_along_axis(rankdata, 0, xtr)
    xr = xr - xr.mean(0)
    num = xr.T @ yr
    den = np.sqrt((xr ** 2).sum(0) * (yr ** 2).sum()) + 1e-12
    return np.argsort(-np.abs(num / den))[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-features", type=int, default=500)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/cyp_blend/blend.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo_all = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi_all = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    import xgboost as xgb
    from tabicl import TabICLRegressor

    WGRID = np.arange(0.0, 1.001, 0.05)
    rows = []
    for rep in range(args.repeats):
        # train / fit the blend weight / report. Three splits, never reused.
        tr, fit, rep_idx = scaffold_split(frame["smiles"].tolist(), frac_train=0.70,
                                          frac_valid=0.15, seed=rep)
        if len(rep_idx) == 0:
            print(f"[blend] rep {rep}: no third split -- skipped", flush=True); continue
        _, stt = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stt)
        print(f"[blend] rep {rep}: {len(tr)} train / {len(fit)} fit / {len(rep_idx)} report",
              flush=True)

        pf, pr = {}, {}           # predictions on the fit and report splits
        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            itr = tr[np.isfinite(y[tr])]
            ifit = fit[np.isfinite(y[fit])]
            irep = rep_idx[np.isfinite(y[rep_idx])]
            if len(itr) < 100 or len(ifit) < 40 or len(irep) < 40:
                continue
            m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                 device="cuda", tree_method="hist", verbosity=0,
                                 eval_metric="mae", early_stopping_rounds=80)
            m.fit(xs[itr], y[itr], eval_set=[(xs[ifit], y[ifit])], verbose=False)
            sel = spearman_rank_select(xs[itr], y[itr], args.n_features)
            t0 = time.time()
            r2 = TabICLRegressor(device="cuda", random_state=0)
            r2.fit(np.ascontiguousarray(xs[itr][:, sel], np.float32), y[itr])
            pf[c] = {"xgb": m.predict(xs[ifit]),
                     "icl": np.asarray(r2.predict(
                         np.ascontiguousarray(xs[ifit][:, sel], np.float32)), float).ravel()}
            pr[c] = {"xgb": m.predict(xs[irep]),
                     "icl": np.asarray(r2.predict(
                         np.ascontiguousarray(xs[irep][:, sel], np.float32)), float).ravel()}
            pf[c]["truth"] = (y[ifit], lo_all[ifit, i], hi_all[ifit, i])
            pr[c]["truth"] = (y[irep], lo_all[irep, i], hi_all[irep, i])
            print(f"    {c[:22]:22s} fitted ({time.time()-t0:.0f}s icl)", flush=True)

        if not pr:
            continue

        def score_w(split, weights):
            """Macro ST-RAE with w = weight on xgb, per endpoint."""
            vals = []
            for c, d in split.items():
                w = weights[c] if isinstance(weights, dict) else weights
                p = w * d["xgb"] + (1 - w) * d["icl"]
                vals.append(st_rae(p, *d["truth"]))
            return float(np.mean(vals))

        # one global weight, chosen on the fit split only
        w_global = float(WGRID[int(np.argmin([score_w(pf, w) for w in WGRID]))])
        # one weight per endpoint, also on the fit split only
        w_per = {}
        for c, d in pf.items():
            tot = [st_rae(w * d["xgb"] + (1 - w) * d["icl"], *d["truth"]) for w in WGRID]
            w_per[c] = float(WGRID[int(np.argmin(tot))])

        # Store PER ENDPOINT, not just the macro. Storing only macros is what
        # made cyp_blend the one experiment that could not be re-weighted
        # arithmetically when the metric denominator was corrected -- it had to be
        # re-run instead. Per-endpoint rows make any future re-weighting free.
        for c, dd in pr.items():
            w_g = w_global
            rows.append({
                "rep": rep, "endpoint": c,
                "xgb": st_rae(dd["xgb"], *dd["truth"]),
                "icl": st_rae(dd["icl"], *dd["truth"]),
                "blend50": st_rae(0.5 * dd["xgb"] + 0.5 * dd["icl"], *dd["truth"]),
                "blend_global": st_rae(w_g * dd["xgb"] + (1 - w_g) * dd["icl"], *dd["truth"]),
                "blend_per_ep": st_rae(w_per[c] * dd["xgb"] + (1 - w_per[c]) * dd["icl"],
                                       *dd["truth"]),
                "w_global": w_g, "w_endpoint": w_per[c]})
        row = {"rep": rep, "xgb": score_w(pr, 1.0), "icl": score_w(pr, 0.0),
               "blend50": score_w(pr, 0.5), "blend_global": score_w(pr, w_global),
               "blend_per_ep": score_w(pr, w_per)}
        print(f"    -> xgb {row['xgb']:.4f} icl {row['icl']:.4f} "
              f"50/50 {row['blend50']:.4f} global {row['blend_global']:.4f} "
              f"(w={w_global:.2f}) per-ep {row['blend_per_ep']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    arms = ["xgb", "icl", "blend50", "blend_global", "blend_per_ep"]
    macro = df.groupby("rep")[arms].mean()
    print("\n[blend] OFFICIAL macro ST-RAE (1.0 == predicting the mean)", flush=True)
    for k in arms:
        print(f"    {k:13s} {macro[k].mean():.4f} (per-repeat sd {macro[k].std(ddof=1):.4f})")
    print("\n[blend] per-endpoint signs vs xgb (mixed => a macro verdict is fragile)",
          flush=True)
    for k in arms[1:]:
        sg = ["+" if (g["xgb"] - g[k]).mean() > 0 else "-" for _, g in df.groupby("endpoint")]
        print(f"    {k:13s} {' '.join(sg)}  "
              f"{'MIXED' if len(set(sg)) > 1 else 'uniform'}")
    print("\n[blend] paired vs xgb (positive = better than xgb)", flush=True)
    for k in arms[1:]:
        d = macro["xgb"] - macro[k]
        se = d.std(ddof=1) / np.sqrt(len(d)); t = d.mean() / se if se > 0 else float("nan")
        wins = int((d > 0).sum())
        tag = ("BETTER" if abs(t) > 2.5 and wins == len(d) else
               "WORSE (consistent)" if abs(t) > 2.5 and wins == 0 else "inside noise")
        print(f"    {k:13s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, "
              f"wins {wins}/{len(d)}, t={t:+.2f}) -> {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
