r"""Separate the model from the decision: a monotone 1-D decode map fitted to the metric.

Standing result of this programme: the model class is saturated. Features moved
the score, extra capacity did not (depth p=0.723, width +0.0000), 800 extra
epochs is noise (band sd 0.0095), and the one verified gain was a per-endpoint
*additive offset* fitted to the metric post hoc. That offset is the one-parameter
member of a family, and the family is what this script tests.

**The theory says what the optimum is.** ST-RAE charges `dist(a, [lo, hi])`, which
is convex and piecewise-linear in the prediction `a`, so the risk derivative is
`E[w.1{hi < a}] - E[w.1{lo > a}]` and the optimum satisfies

    P(lo > a*) == P(hi < a*)        equivalently   F_lo(a*) + F_hi(a*) == 1

i.e. **a\* is the weighted median of the pooled set of all `lo` and `hi` draws** --
not the conditional mean, and (contrary to my first guess) not a mode either: the
loss is unbounded, so the Bayes act is median-type however multimodal the
predictive is.

**But the useful consequence is not a generative model.** `a*` depends on `x` only
through the predictive distribution, and the interval geometry here is dominated
by *width*, which is strongly dependent on the label itself (r = -0.73 to -0.89
between width and y). So the correction is almost entirely a function of the point
prediction, and can be learned directly as a one-dimensional map

    y_hat  ->  y_hat + delta(y_hat)

fitted by ERM against the metric on held-out molecules. No density estimate, no
diffusion, no flow. Measured offline on the real labels this is worth roughly
-0.027 macro ST-RAE against -0.005 for the global per-endpoint offset, because a
scalar cannot represent a correction that varies with predicted activity.

**Three disciplines, each of which has burned this programme before.**

1. **The map is fitted on molecules it is not scored on.** Three-way scaffold
   split: train the model, fit `delta` on a second split, report on a third. An
   8-parameter table on ~650 molecules per endpoint with bootstrap sd ~0.03 is
   exactly the regime where fitting and reporting on one split invents a result --
   see the 2026-10-05 winner's-curse entry.
2. **Monotonicity is enforced, not hoped for.** An unconstrained per-bin table can
   re-order molecules (a less active compound predicted higher than a more active
   one) while ST-RAE, which scores each molecule independently, notices nothing.
   Isotonic projection (PAVA) on the bin outputs keeps the map order-preserving,
   and both variants are reported so the cost of the constraint is visible.
3. **It is fitted per model arm.** The net is already hinge-trained, so it may
   already emit something near `a*`; the gain measured on a mean-trained
   gradient-boosting arm need not transfer. xgb and the net are therefore decoded
   separately rather than sharing one table.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load          # noqa: E402
from drugrx.cyp.metric import st_rae               # noqa: E402
from drugrx.data.splits import scaffold_split      # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402


def hinge(pred, lo, hi):
    return np.clip(lo - pred, 0, None) + np.clip(pred - hi, 0, None)


def fit_global_offset(pred, lo, hi, grid):
    """The incumbent: one scalar per endpoint."""
    tot = [hinge(pred + d, lo, hi).sum() for d in grid]
    return float(grid[int(np.argmin(tot))])


def fit_decode_map(pred, lo, hi, n_bins, grid, monotone=True):
    """Per-bin additive shift, optionally projected to an order-preserving map.

    Bins are quantiles of the prediction. The metric is a sum over molecules, so
    bins are independent and each bin's shift is an exact 1-D grid minimisation --
    no joint optimisation and no local minima.
    """
    edges = np.quantile(pred, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, n_bins - 1)
    centres = np.array([pred[idx == b].mean() if (idx == b).any() else np.nan
                        for b in range(n_bins)])
    shifts = np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:                      # too few to fit; leave uncorrected
            continue
        tot = [hinge(pred[m] + d, lo[m], hi[m]).sum() for d in grid]
        shifts[b] = grid[int(np.argmin(tot))]
    if monotone:
        # Project so that bin centre + shift is non-decreasing in the bin centre.
        # Order-preserving is a property of the *output*, not of the shift.
        ok = np.isfinite(centres)
        out = centres[ok] + shifts[ok]
        out = _pava(out)
        shifts[ok] = out - centres[ok]
    return edges, centres, shifts


def _pava(y):
    """Pool-adjacent-violators: nearest non-decreasing sequence (unit weights)."""
    y = np.asarray(y, float)
    n = len(y)
    out: list[list[float]] = []
    for v in y:
        out.append([float(v), 1.0])
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2 = out.pop()
            v1, w1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    res = []
    for v, w in out:
        res.extend([v] * int(round(w)))
    return np.asarray(res[:n])


def apply_map(pred, edges, centres, shifts):
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, len(shifts) - 1)
    return pred + shifts[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5, help="independent scaffold splits")
    ap.add_argument("--bins", nargs="*", type=int, default=[2, 4, 8, 16])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--xgb-rounds", type=int, default=1200)
    ap.add_argument("--out", default="/content/out/cyp_decode/decode.json")
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo_all = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi_all = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    grid = np.arange(-1.5, 1.501, 0.01)

    import xgboost as xgb

    rows = []
    for rep in range(args.repeats):
        # train / fit-the-decoder / report. Three splits, never reused.
        tr, fit, rep_idx = scaffold_split(frame["smiles"].tolist(), frac_train=0.70,
                                          frac_valid=0.15, seed=rep)
        if len(rep_idx) == 0:
            print(f"[decode] rep {rep}: no third split -- skipped", flush=True); continue
        _, stt = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stt)
        print(f"[decode] rep {rep}: {len(tr)} train / {len(fit)} fit / {len(rep_idx)} report",
              flush=True)

        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            itr = tr[np.isfinite(y[tr])]
            ifit = fit[np.isfinite(y[fit])]
            irep = rep_idx[np.isfinite(y[rep_idx])]
            if len(itr) < 100 or len(ifit) < 40 or len(irep) < 40:
                continue
            m = xgb.XGBRegressor(n_estimators=args.xgb_rounds, learning_rate=0.03,
                                 max_depth=7, subsample=0.8, colsample_bytree=0.5,
                                 reg_lambda=1.0, device="cuda", tree_method="hist",
                                 verbosity=0, eval_metric="mae", early_stopping_rounds=80)
            m.fit(xs[itr], y[itr], eval_set=[(xs[ifit], y[ifit])], verbose=False)
            pf, pr = m.predict(xs[ifit]), m.predict(xs[irep])
            lof, hif = lo_all[ifit, i], hi_all[ifit, i]
            lor, hir = lo_all[irep, i], hi_all[irep, i]
            yr = y[irep]

            raw = st_rae(pr, yr, lor, hir)
            d0 = fit_global_offset(pf, lof, hif, grid)
            off = st_rae(pr + d0, yr, lor, hir)
            r = {"rep": rep, "endpoint": c, "n_report": int(len(irep)),
                 "raw": raw, "offset": off, "offset_value": d0}
            for nb in args.bins:
                for mono in (True, False):
                    e, ce, sh = fit_decode_map(pf, lof, hif, nb, grid, monotone=mono)
                    sc = st_rae(apply_map(pr, e, ce, sh), yr, lor, hir)
                    r[f"map{nb}{'_mono' if mono else '_free'}"] = sc
            rows.append(r)
            print(f"    {c:30s} raw {raw:.4f} offset {off:.4f} "
                  + " ".join(f"m{nb}:{r[f'map{nb}_mono']:.4f}" for nb in args.bins), flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    print("\n[decode] macro per repeat, then paired against the incumbent offset", flush=True)
    macro = df.groupby("rep").mean(numeric_only=True)
    cols = ["raw", "offset"] + [f"map{nb}_{k}" for nb in args.bins for k in ("mono", "free")]
    for c_ in cols:
        if c_ in macro:
            print(f"    {c_:14s} {macro[c_].mean():.4f} (per-repeat sd {macro[c_].std(ddof=1):.4f})")

    print("\n[decode] paired vs offset (positive = better than the incumbent)", flush=True)
    for c_ in cols:
        if c_ in ("raw", "offset") or c_ not in macro:
            continue
        d = macro["offset"] - macro[c_]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        print(f"    {c_:14s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, "
              f"wins {int((d>0).sum())}/{len(d)}, t={t:+.2f}) -> "
              f"{'REAL' if abs(t) > 2.5 and int((d>0).sum()) in (0, len(d)) else 'inside noise'}")
    print("\n[decode] and vs raw, to size the whole decode step:", flush=True)
    for c_ in ("offset", f"map{args.bins[-1]}_mono"):
        if c_ in macro:
            d = macro["raw"] - macro[c_]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"    raw -> {c_:14s} {d.mean():+.4f} +/- {se:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
