r"""Is the bin-5 numerator large because the errors are large, or because the bin is?

The auxiliary ablation's tail read prints a **sum** of soft-threshold error per
bin, because that is what the ST-RAE numerator is. A sum confounds per-molecule
error with bin population, so "bin 5 is 133 and the others are 35-80" cannot be
read as "the tail is three times harder" without the counts beside it. A 5-unit
reduction over 400 molecules and a 5-unit reduction over 40 are different claims.

Bins are quantiles of the labels, so populations *should* be equal by
construction -- but pIC50 panels are full of ties at the assay's limits, and
`searchsorted` on tied quantile edges collapses bins. If a large share of labels
sit at a detection threshold, the lowest bin swallows them and its sum inflates
for reasons that have nothing to do with prediction quality. That is the specific
worry about bin 1 being the second-worst bin when it should be the easy one:
either it is heavily populated, or those labels are censored measurements whose
"value" is a threshold.

So this prints, per head: label ties, the realised bin populations, the per-bin
**mean** error beside the sum, and a censoring check on the interval bounds --
degenerate or one-sided `lo`/`hi`, and how much of each bin's error comes from
rows whose interval is suspiciously narrow or pinned to the label.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/cyp_cache/cyp_aux/aux-none_s0.npz")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    data = load()
    f = data.frame
    z = np.load(args.npz)
    idx, pred = z["va_index"], z["val"]
    nb = args.bins

    grand = np.zeros(nb)
    grand_n = np.zeros(nb)
    for i, c in enumerate(PRIMARY):
        y = f[c].to_numpy(float)[idx]
        lo = f[f"{c}_conf_low"].to_numpy(float)[idx]
        hi = f[f"{c}_conf_high"].to_numpy(float)[idx]
        keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(pred[:, i])
        yy, ll, hh, pp = y[keep], lo[keep], hi[keep], pred[keep, i]
        if len(yy) < nb:
            continue
        vals, cnts = np.unique(yy, return_counts=True)
        top = sorted(zip(cnts, vals), reverse=True)[:3]
        width = hh - ll
        print(f"\n=== {c} ===")
        print(f"  n={len(yy)}  distinct labels={len(vals)}  "
              f"range {yy.min():.2f}..{yy.max():.2f}")
        print("  most common labels: "
              + ", ".join(f"{v:.3f} x{n}" for n, v in top)
              + f"   (top label is {100 * top[0][0] / len(yy):.1f}% of rows)")
        print(f"  interval width: median {np.median(width):.3f}  "
              f"min {width.min():.3f}  zero-width rows {int((width <= 0).sum())}  "
              f"label outside [lo,hi] {int(((yy < ll) | (yy > hh)).sum())}")

        edges = np.quantile(yy, np.linspace(0, 1, nb + 1))
        which = np.clip(np.searchsorted(edges, yy, side="right") - 1, 0, nb - 1)
        err = soft_threshold_error(pp, ll, hh)
        print(f"  quantile edges: " + " ".join(f"{e:.3f}" for e in edges))
        print(f"  {'bin':>4s} {'n':>5s} {'sum':>9s} {'mean':>8s} "
              f"{'medwidth':>9s} {'inside%':>8s}")
        for b in range(nb):
            m = which == b
            n = int(m.sum())
            if n == 0:
                print(f"  {b+1:>4d} {0:>5d}         -        -         -        -")
                continue
            inside = 100.0 * float((err[m] == 0).mean())
            print(f"  {b+1:>4d} {n:>5d} {err[m].sum():9.2f} {err[m].mean():8.4f} "
                  f"{np.median(width[m]):9.3f} {inside:7.1f}%")
            grand[b] += err[m].sum()
            grand_n[b] += n

    print("\n=== pooled over the four heads (one seed, control arm) ===")
    print(f"  {'bin':>4s} {'n':>6s} {'sum':>9s} {'mean':>8s} {'share of numerator':>20s}")
    tot = grand.sum()
    for b in range(nb):
        share = 100.0 * grand[b] / tot if tot else float("nan")
        mean = grand[b] / grand_n[b] if grand_n[b] else float("nan")
        print(f"  {b+1:>4d} {int(grand_n[b]):>6d} {grand[b]:9.2f} {mean:8.4f} "
              f"{share:19.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
