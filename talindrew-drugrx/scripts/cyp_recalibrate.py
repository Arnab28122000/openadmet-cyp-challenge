"""Affine recalibration: fix the under-dispersion the leaderboard exposed.

Measured on validation under the **official** metric: our predictions carry a
standard deviation of 0.481 against the labels' 0.944. They are compressed to
roughly half the spread of the thing they predict, which is what produced a
leaderboard R-squared of 0.045 alongside a respectable Spearman of 0.62 -- a model
that ranks molecules well and places them wrongly.

Shrinkage of this kind is the *expected* behaviour of the interval-hinge loss: a
prediction anywhere inside the credible interval costs nothing, so there is no
gradient pulling predictions out toward the extremes of the label distribution,
and the optimiser is free to park everything near the middle where the bands are
widest.

The earlier decode-map experiment (`cyp_decode.py`) tested **additive** shifts,
per bin, and found nothing beyond a single scalar per endpoint. It never tested a
**scale** term, which is the axis the measurement says is actually broken. So:

    p' = m + alpha * (p - m) + beta        (per endpoint; m = mean of p on the fit split)

Four arms, cheapest-first, every parameter fitted on a split the score is not read
from:

* `raw` -- the incumbent.
* `shift` -- beta only. This is what we submitted (one scalar per endpoint).
* `scale` -- alpha only, one parameter, directly targeting the under-dispersion.
* `affine` -- both. Two parameters per endpoint.
* `match_sd` -- alpha set analytically so the prediction sd equals the *training
  label* sd, with no fitting at all. A zero-free-parameter control: if `scale`
  beats this, the gain came from fitting; if it does not, the whole effect is just
  "match the spread" and needs no optimiser.

Scored with the official ST-RAE, and MAE and R-squared are reported alongside,
because the leaderboard shows all three and the under-dispersion story predicts
they move together. Paired across repeats.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load                        # noqa: E402
from drugrx.cyp.metric import st_rae                             # noqa: E402
from drugrx.data.splits import scaffold_split                    # noqa: E402
from drugrx.features import featurise, standardise_descriptors   # noqa: E402

ALPHAS = np.arange(0.5, 3.01, 0.05)
BETAS = np.arange(-1.0, 1.001, 0.02)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/cyp_recal/recal.json")
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
        tr, fit, rep_idx = scaffold_split(frame["smiles"].tolist(), frac_train=0.70,
                                          frac_valid=0.15, seed=rep)
        if len(rep_idx) == 0:
            continue
        _, stt = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stt)
        print(f"[recal] rep {rep}: {len(tr)} train / {len(fit)} fit / {len(rep_idx)} report",
              flush=True)

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
            pf, pr = m.predict(xs[ifit]), m.predict(xs[irep])
            lof, hif, yf = lo_all[ifit, i], hi_all[ifit, i], y[ifit]
            lor, hir, yr = lo_all[irep, i], hi_all[irep, i], y[irep]

            centre = float(pf.mean())          # fitted on the fit split only
            def apply(p, a, b):
                return centre + a * (p - centre) + b

            # one-dimensional searches, then a small joint grid for `affine`
            b_only = BETAS[int(np.argmin([st_rae(apply(pf, 1.0, b), yf, lof, hif)
                                          for b in BETAS]))]
            a_only = ALPHAS[int(np.argmin([st_rae(apply(pf, a, 0.0), yf, lof, hif)
                                           for a in ALPHAS]))]
            best, ab = None, (1.0, 0.0)
            for a in ALPHAS:
                sc = [st_rae(apply(pf, a, b), yf, lof, hif) for b in BETAS]
                j = int(np.argmin(sc))
                if best is None or sc[j] < best:
                    best, ab = sc[j], (float(a), float(BETAS[j]))
            # zero-free-parameter control: match the TRAINING label sd
            sd_p = float(pf.std()) or 1e-9
            a_match = float(np.nanstd(y[itr]) / sd_p)

            def row(name, p):
                ss = float(np.sum((yr - p) ** 2))
                st = float(np.sum((yr - yr.mean()) ** 2))
                return {"rep": rep, "endpoint": c, "arm": name,
                        "official": st_rae(p, yr, lor, hir),
                        "mae": float(np.mean(np.abs(p - yr))),
                        "r2": 1.0 - ss / st if st > 0 else float("nan"),
                        "pred_sd": float(np.std(p)), "true_sd": float(np.std(yr))}

            rows += [row("raw", pr),
                     row("shift", apply(pr, 1.0, float(b_only))),
                     row("scale", apply(pr, float(a_only), 0.0)),
                     row("affine", apply(pr, ab[0], ab[1])),
                     row("match_sd", apply(pr, a_match, 0.0))]
            print(f"    {c[:22]:22s} alpha(fit) {a_only:.2f} beta(fit) {b_only:+.2f} "
                  f"affine ({ab[0]:.2f},{ab[1]:+.2f}) match_sd alpha {a_match:.2f}",
                  flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    arms = ["raw", "shift", "scale", "affine", "match_sd"]
    macro = {a: df[df.arm == a].groupby("rep")["official"].mean() for a in arms}
    print("\n[recal] OFFICIAL macro ST-RAE (1.0 == predicting the mean)", flush=True)
    for a in arms:
        g = df[df.arm == a]
        print(f"    {a:9s} {macro[a].mean():.4f} (sd {macro[a].std(ddof=1):.4f})  "
              f"MAE {g['mae'].mean():.4f}  R2 {g['r2'].mean():+.4f}  "
              f"pred sd {g['pred_sd'].mean():.3f} vs true {g['true_sd'].mean():.3f}")
    print("\n[recal] paired vs raw (positive = better)", flush=True)
    for a in arms[1:]:
        d = macro["raw"] - macro[a]
        se = d.std(ddof=1) / np.sqrt(len(d)); t = d.mean() / se if se > 0 else float("nan")
        w = int((d > 0).sum())
        tag = ("BETTER" if abs(t) > 2.5 and w == len(d) else
               "WORSE (consistent)" if abs(t) > 2.5 and w == 0 else "inside noise")
        print(f"    {a:9s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, wins {w}/{len(d)}, "
              f"t={t:+.2f}) -> {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
