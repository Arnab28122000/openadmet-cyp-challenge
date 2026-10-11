r"""Re-measure the decode-map hypothesis under the corrected metric. No GPU.

H7 asked whether a richer decoder beats the one-parameter additive offset. It was
answered under the **superseded denominator** (`raw 0.4660`, `offset 0.4540`, maps
inside noise, 16 bins worse) and is one of the two CYP conclusions still resting on it.
Depth and width both reversed on re-measurement, so this cannot be assumed to survive.

It needs no training, because it is a question about **post-processing predictions**, and
51 runs of stored validation predictions already exist. What it does need is honest
splitting: a decode map fitted and evaluated on the same rows will always look good. So
the 922 validation rows are halved, the map is fitted on one half and scored on the
other, both directions are run, and the two are averaged -- a 2-fold cross-fit.

The comparison is **paired within run**: every arm sees the same rows, so the
run-to-run spread that dominates everything in this programme cancels.

Arms, in increasing flexibility: nothing, a per-endpoint additive offset (the
incumbent), and a monotone per-bin decode map at 2/4/8/16 bins. Monotonicity is imposed
by a pool-adjacent-violators pass, which the earlier study found cut the 16-bin damage
from -0.0197 to -0.0061.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae

OFFSET_GRID = np.arange(-0.40, 0.4001, 0.02)


def _best_offset(p, y, lo, hi):
    """The incumbent: one scalar per endpoint, chosen on the fit half.

    Minimises the metric's **numerator**, not the ratio. The denominator depends only
    on the rows and the labels, never on the prediction, so the argmin is identical --
    and the ratio is genuinely undefined when the mean predictor is itself perfect
    (the 0/0 that `tests/test_official_metric.py` documents). At 16 bins each bin holds
    ~7-10 rows per endpoint, where that happens often enough to crash a `nanargmin`.
    """
    err = [float((np.clip(lo - (p + d), 0, None) + np.clip((p + d) - hi, 0, None)).sum())
           for d in OFFSET_GRID]
    return float(OFFSET_GRID[int(np.argmin(err))])


def _pava(v, w):
    """Pool adjacent violators: nearest non-decreasing sequence, weighted."""
    v, w = list(map(float, v)), list(map(float, w))
    i = 0
    while i < len(v) - 1:
        if v[i] > v[i + 1]:
            tot = w[i] + w[i + 1]
            merged = (v[i] * w[i] + v[i + 1] * w[i + 1]) / tot if tot else v[i]
            v[i:i + 2] = [merged]
            w[i:i + 2] = [tot]
            i = max(i - 1, 0)
        else:
            i += 1
    out, k = [], 0
    for val, weight in zip(v, w):
        n = int(round(weight))
        out.extend([val] * max(n, 1))
        k += 1
    return out


def _bin_map(p_fit, y_fit, lo_fit, hi_fit, nb, monotone=True):
    """A per-bin additive offset, bins taken on the PREDICTION (available at test)."""
    edges = np.quantile(p_fit, np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    deltas, weights = [], []
    for b in range(nb):
        m = (p_fit >= edges[b]) & (p_fit < edges[b + 1])
        if m.sum() < 5:
            deltas.append(0.0); weights.append(1.0); continue
        deltas.append(_best_offset(p_fit[m], y_fit[m], lo_fit[m], hi_fit[m]))
        weights.append(float(m.sum()))
    if monotone and nb > 1:
        order = np.argsort(np.array(deltas))
        if not np.all(order == np.arange(nb)):
            deltas = _pava(deltas, [1.0] * nb)[:nb]
    return edges, np.array(deltas, float)


def _apply(edges, deltas, p):
    out = p.copy()
    for b in range(len(deltas)):
        m = (p >= edges[b]) & (p < edges[b + 1])
        out[m] = p[m] + deltas[b]
    return out


def main() -> int:
    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    runs = []
    for sub in ("cyp_loss", "cyp_aux", "cyp_long", "cyp_h12"):
        d = Path("data/cyp_cache") / sub
        if d.exists():
            runs += [f.with_suffix(".npz") for f in sorted(d.glob("*.json"))
                     if f.with_suffix(".npz").exists()]
    print(f"{len(runs)} stored runs; metric {METRIC_ID}; paired within run, 2-fold cross-fit\n")

    arms = {}
    rng = np.random.default_rng(0)
    for npz in runs:
        z = np.load(npz)
        idx, pred = z["va_index"], z["val"]
        half = rng.permutation(len(idx)) < len(idx) // 2
        per_arm = {}
        for name in ["raw", "offset"] + [f"map{n}" for n in (2, 4, 8, 16)]:
            sc = []
            for fold in (half, ~half):
                fit, ev = fold, ~fold
                outs = []
                for i, c in enumerate(PRIMARY):
                    p, yy = pred[:, i], y[c][idx]
                    ll, hh = lo[c][idx], hi[c][idx]
                    k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) & np.isfinite(p)
                    kf, ke = k & fit, k & ev
                    if kf.sum() < 20 or ke.sum() < 20:
                        outs.append(np.nan); continue
                    if name == "raw":
                        pe = p[ke]
                    elif name == "offset":
                        pe = p[ke] + _best_offset(p[kf], yy[kf], ll[kf], hh[kf])
                    else:
                        nb = int(name[3:])
                        e, dl = _bin_map(p[kf], yy[kf], ll[kf], hh[kf], nb)
                        pe = _apply(e, dl, p[ke])
                    outs.append(st_rae(pe, yy[ke], ll[ke], hh[ke]))
                sc.append(float(np.nanmean(outs)))
            per_arm[name] = float(np.mean(sc))
        for k_, v in per_arm.items():
            arms.setdefault(k_, []).append(v)

    base = arms["offset"]
    print(f"{'arm':10s} {'macro':>8s} {'vs raw':>9s} {'vs offset':>11s} {'se':>8s} "
          f"{'t':>7s} {'wins':>8s} verdict")
    for name, vals in arms.items():
        d_raw = st.mean(vals) - st.mean(arms["raw"])
        if name == "offset":
            d = [v - r for v, r in zip(vals, arms["raw"])]
            m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
            print(f"{name:10s} {st.mean(vals):8.4f} {d_raw:+9.4f} {'(base)':>11s} "
                  f"{se:8.4f} {m/se:+7.2f} {sum(1 for x in d if x<0):4d}/{len(d)}  "
                  f"{'REAL' if abs(m/se) > 2.5 else 'inside noise'}")
            continue
        if name == "raw":
            print(f"{name:10s} {st.mean(vals):8.4f} {'—':>9s} {'—':>11s}")
            continue
        d = [v - b for v, b in zip(vals, base)]
        m, se = st.mean(d), st.stdev(d) / len(d) ** 0.5
        t = m / se if se else float("nan")
        print(f"{name:10s} {st.mean(vals):8.4f} {d_raw:+9.4f} {m:+11.4f} {se:8.4f} "
              f"{t:+7.2f} {sum(1 for x in d if x<0):4d}/{len(d)}  "
              f"{'BEATS OFFSET' if t < -2.5 else ('WORSE' if t > 2.5 else 'inside noise')}")
    print("\nNegative is better. 'vs offset' is the paired difference against the incumbent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
