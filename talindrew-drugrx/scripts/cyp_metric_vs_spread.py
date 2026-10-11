r"""How much does ST-RAE move with the evaluation set's label spread, model held fixed?

ST-RAE divides by the error of a constant predictor at mean(y) **of the evaluation set**, so
the ratio is a property of that set as well as of the model. The question raised in review:
if realistic changes in label concentration move the ratio by anything near 0.2, a large part
of the +0.2114 validation-to-leaderboard gap is metric geometry rather than model deficiency.

An earlier banding experiment answered this badly. It split validation by similarity to
training and found the denominator flat at 1.016x -- but that is a report that **the bands did
not differ in label spread**, i.e. the experiment never varied the quantity the mechanism runs
on. This varies it directly.

Design: draw subsets of fixed size from validation with sampling weights
`exp(beta * |y - mean(y)|)`, sweeping beta to produce subsets whose realised label sd ranges
from far below validation's to far above it. One fixed model throughout (the longdeep
5-seed mean), so every change in the ratio is the metric's.

Two reference points matter. Validation sd is 0.944. The **test** sd, backed out of the
leaderboard's own MAE and R² columns -- which the organisers compute and our denominator
error never touched -- is **~1.29**: from R^2 = 1 - MSE/sd^2 and, for Gaussian errors,
RMSE = MAE*sqrt(pi/2), giving sd = MAE*sqrt(pi/2)/sqrt(1-R^2) = 1.0063*1.2533/sqrt(0.9548)
= 1.29. It reproduces on the newer row (MAE 1.0, R^2 0.0552 -> 1.289).
"""
from __future__ import annotations

import glob
import statistics as st

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error


def main() -> int:
    d = load()
    fr = d.frame
    runs = sorted(glob.glob("data/cyp_cache/cyp_h12/h12-longdeep_s*.npz"))
    idx = np.load(runs[0])["va_index"]
    pred = np.mean([np.load(r)["val"] for r in runs], axis=0)

    cols = []
    for i, c in enumerate(PRIMARY):
        y = fr[c].to_numpy(float)[idx]
        lo = fr[f"{c}_conf_low"].to_numpy(float)[idx]
        hi = fr[f"{c}_conf_high"].to_numpy(float)[idx]
        k = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(pred[:, i])
        cols.append((y[k], lo[k], hi[k], pred[k, i]))

    rng = np.random.default_rng(0)
    print(f"{'beta':>6s} {'sd(y)':>7s} {'NUMER/n':>9s} {'DENOM/n':>9s} {'ST-RAE':>8s} "
          f"{'+/-':>7s}")
    rows = []
    for beta in (-2.5, -1.5, -0.8, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        reps_sd, reps_r, reps_n, reps_d = [], [], [], []
        for _ in range(40):
            num = den = tot = 0.0
            sds = []
            for y, lo, hi, p in cols:
                n_take = max(int(0.5 * len(y)), 30)
                w = np.exp(beta * np.abs(y - y.mean()))
                w = w / w.sum()
                sel = rng.choice(len(y), size=n_take, replace=False,
                                 p=w if np.isfinite(w).all() else None)
                ys, ls, hs, ps = y[sel], lo[sel], hi[sel], p[sel]
                num += soft_threshold_error(ps, ls, hs).sum()
                den += soft_threshold_error(np.full(len(ys), ys.mean()), ls, hs).sum()
                tot += len(ys)
                sds.append(ys.std())
            reps_sd.append(float(np.mean(sds)))
            reps_r.append(num / den)
            reps_n.append(num / tot)
            reps_d.append(den / tot)
        m_sd, m_r = st.mean(reps_sd), st.mean(reps_r)
        print(f"{beta:6.1f} {m_sd:7.3f} {st.mean(reps_n):9.4f} {st.mean(reps_d):9.4f} "
              f"{m_r:8.4f} {st.stdev(reps_r):7.4f}")
        rows.append((m_sd, m_r))

    # Interpolate the ratio at validation's spread and at the test estimate.
    rows.sort()
    xs = np.array([r[0] for r in rows]); ys_ = np.array([r[1] for r in rows])
    for label, target in (("validation (0.944)", 0.944), ("TEST estimate (1.29)", 1.29)):
        v = float(np.interp(target, xs, ys_))
        print(f"\n  ST-RAE at sd {target:.3f}  [{label}] : {v:.4f}")
    v_val = float(np.interp(0.944, xs, ys_))
    v_test = float(np.interp(1.29, xs, ys_))
    print(f"\n  metric-geometry shift from validation's spread to the test estimate: "
          f"{v_test - v_val:+.4f}")
    print(f"  observed validation->leaderboard gap: +0.2114")
    print(f"  share explained by label spread alone: "
          f"{100 * (v_test - v_val) / 0.2114:+.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
