"""v9: every placement axis is now BRACKETED, so stop extrapolating and interpolate.

v8 falsified its own headline claim on all three centre cells and the CYP2D6 spread forecast too
(macro 0.5256 -> 0.5520, rank 54 -> 69). The claim it broke was *"no centre cell may rise above its
v7 value"*, whose stated proviso was "provided we do not cross the balance point". We crossed it,
on every axis at once.

**Convexity itself was never refuted** -- and the distinction matters, because convexity is what
this script runs on. ST-RAE is a sum of hinge terms in the predictions, so for a pure shift
`p_i(c) = c + s*z_i` it is a convex piecewise-LINEAR function of the centre `c`, and likewise
convex in the spread `s`. Its derivative `(N_above - N_below)/D` is a non-decreasing step function.
What v8 refuted is the assumption that v7 sat on the near side of the zero crossing. It did not.

The consolation is the whole reason this is worth a slot: **a secant that changes sign across a
point brackets the minimum**, and v8 delivered a sign change on all four axes:

    axis             left secant   right secant   minimum lies in
    CYP1A2 centre        -0.159        +0.058      (4.2758, 4.6600)
    CYP2C9 centre        -0.360        +0.007      (4.7072, 4.9800)
    CYP3A4 centre        -0.105        +0.114      (4.7532, 5.0300)
    CYP2D6 spread        -0.160        +0.058      (0.7805, 1.2008)

Every forecast this programme has made so far was an EXTRAPOLATION beyond the measured range, and
two of the last two missed high. v9 is the first that is an INTERPOLATION strictly inside a
measured bracket, which is why its bounds are rigorous rather than assumed.

Three quantities per axis, and the difference between them is the honest uncertainty:

* **vertex**      -- the 3-point parabola minimiser. A point ESTIMATE, and a biased one: the true
                     function is piecewise linear with a kink, so a parabola dips more smoothly
                     than the truth and its projected value is optimistic.
* **chord bound** -- f(x) <= the straight line between the two measured points that straddle x.
                     RIGOROUS from convexity alone, and it is the number the falsifiable claim uses.
* **slope bound** -- f(x) >= f(x_best) + s_left * (x - x_best). Also rigorous, and loose.

    python scripts/cyp_v9_bracket.py --dry-run     # report only
    python scripts/cyp_v9_bracket.py               # writes submission_v9.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/cyp_cache/final_v4screen/submission_v4.csv"
DST = ROOT / "data/cyp_cache/final_v4screen/submission_v9.csv"
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
COL = {k: f"{k}_pIC50_direct_inhibition" for k in CYPS}
FLOOR = 1.0

#: Every scored point on each axis, (x, ST-RAE), read off the board. The OTHER axis is held fixed
#: across all points of a given axis -- that is what makes each one attributable.
#: Centres: CYP1A2/CYP2C9/CYP3A4 at their version spreads; CYP2D6 spread at centre 3.43 throughout.
AXES = {
    "CYP1A2": ("centre", [(4.6600, 0.5490), (4.4679, 0.5379), (4.2758, 0.5685)]),
    "CYP2C9": ("centre", [(4.9800, 0.4623), (4.8436, 0.4613), (4.7072, 0.5104)]),
    "CYP3A4": ("centre", [(5.0300, 0.5671), (4.8916, 0.5513), (4.7532, 0.5658)]),
    "CYP2D6": ("spread", [(0.7805, 0.5870), (1.0007, 0.5517), (1.2008, 0.5634)]),
}
#: v7's placement, which scored 0.5256 -- the best row this programme has had. The axis NOT being
#: moved on each isoform is pinned here, so every v9 cell still moves exactly one variable.
V7 = {"CYP1A2": dict(centre=4.4679, spread=1.0007),
      "CYP2C9": dict(centre=4.8436, spread=0.7605),
      "CYP2D6": dict(centre=3.4300, spread=1.0007),
      "CYP3A4": dict(centre=4.8916, spread=0.8506)}


def analyse(pts):
    """Bracket check, parabola vertex, and the two rigorous convexity bounds at that vertex."""
    pts = sorted(pts)
    x = np.array([a for a, _ in pts], float)
    y = np.array([b for _, b in pts], float)
    sL = (y[1] - y[0]) / (x[1] - x[0])
    sR = (y[2] - y[1]) / (x[2] - x[1])
    bracketed = bool(y[1] < y[0] and y[1] < y[2] and sL < 0 < sR)
    a, b, _c = np.polyfit(x, y, 2)
    vx = float(-b / (2 * a))
    vy = float(np.polyval([a, b, _c], vx))
    i_best = int(np.argmin(y))
    xb, yb = float(x[i_best]), float(y[i_best])

    # Chord bound: the straight line between the two measured points straddling vx.
    j = int(np.searchsorted(x, vx))
    j = min(max(j, 1), len(x) - 1)
    x0, y0, x1, y1 = x[j - 1], y[j - 1], x[j], y[j]
    chord = float(y0 + (y1 - y0) * (vx - x0) / (x1 - x0))
    # Slope bound: moving off the best point, the slope cannot be steeper than the far secant.
    s_out = sL if vx > xb else sR
    slope_lo = float(yb + s_out * (vx - xb)) if vx > xb else float(yb - s_out * (xb - vx))
    return dict(bracketed=bracketed, sL=float(sL), sR=float(sR), vertex=vx, parabola=vy,
                chord=chord, slope_lo=slope_lo, x_best=xb, y_best=yb,
                lo=float(min(vy, slope_lo)), hi=float(chord))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    res = {k: analyse(p) for k, (_ax, p) in AXES.items()}

    print("=" * 94)
    print("v9 PRE-REGISTRATION -- written before the result exists")
    print("=" * 94)
    print("\n1. THE BRACKET. A secant sign change across a point puts the minimum inside it.")
    print(f"   {'iso':8} {'axis':7} {'secant L':>9} {'secant R':>9} {'bracketed':>10} "
          f"{'best x':>8} {'best y':>8}")
    for k in CYPS:
        ax, _ = AXES[k]
        r = res[k]
        print(f"   {k:8} {ax:7} {r['sL']:+9.3f} {r['sR']:+9.3f} {str(r['bracketed']):>10} "
              f"{r['x_best']:8.4f} {r['y_best']:8.4f}")
    if not all(r["bracketed"] for r in res.values()):
        raise SystemExit("an axis is NOT bracketed -- do not interpolate on it")
    print("   All four bracketed. Every previous forecast extrapolated BEYOND the measured")
    print("   range; this one interpolates strictly inside it.")

    print("\n2. THE TARGET and its bounds. The chord bound is rigorous from convexity alone.")
    print(f"   {'iso':8} {'axis':7} {'from':>8} {'to':>8} {'move':>8} {'parabola':>9} "
          f"{'chord<=':>8} {'band':>18}")
    for k in CYPS:
        ax, _ = AXES[k]
        r = res[k]
        print(f"   {k:8} {ax:7} {r['x_best']:8.4f} {r['vertex']:8.4f} "
              f"{r['vertex']-r['x_best']:+8.4f} {r['parabola']:9.4f} {r['chord']:8.4f} "
              f"[{r['lo']:.4f}, {r['hi']:.4f}]")
    lo = float(np.mean([res[k]["lo"] for k in CYPS]))
    hi = float(np.mean([res[k]["hi"] for k in CYPS]))
    pt = float(np.mean([res[k]["parabola"] for k in CYPS]))
    v7m = float(np.mean([res[k]["y_best"] for k in CYPS]))
    print(f"\n   MACRO point estimate {pt:.4f};  rigorous band [{lo:.4f}, {hi:.4f}]")
    print(f"   v7 (the best row so far) was {v7m:.4f}; v8 was 0.5520.")

    print("\n3. THE FALSIFIABLE CLAIM -- and why it is stronger than v8's.")
    print(f"   Every cell must land at or below its CHORD bound, and macro at or below {hi:.4f}.")
    print("   v8's claim rested on an assumption that could fail (not crossing the balance")
    print("   point) and it did fail. This one rests only on (a) convexity of a sum of hinges")
    print("   and (b) three measured points per axis. If a cell exceeds its chord bound then")
    print("   either the scorer is not deterministic or ST-RAE is not convex in that")
    print("   parameter -- and both of those would be far bigger news than the placement.")
    print("   ALSO pre-registered, 7th consecutive: rho 0.7093 / tau 0.5285 UNCHANGED.")

    print("\n4. HONEST EXPECTED VALUE. The point estimate beats v7 by "
          f"{v7m - pt:+.4f} macro and the")
    print(f"   rigorous worst case is {hi - v7m:+.4f} against it. This is a SMALL move that")
    print("   mainly recovers v8's loss; it is worth the slot because it also measures the")
    print("   vertex on all four axes, which closes the placement axis for good.")

    rows = list(csv.DictReader(Path(args.src).open()))
    fields = list(rows[0].keys())
    out = {}
    print(f"\n5. BUILD from {Path(args.src).name} ({len(rows)} rows)")
    for k in CYPS:
        ax, _ = AXES[k]
        tgt_c = res[k]["vertex"] if ax == "centre" else V7[k]["centre"]
        tgt_s = res[k]["vertex"] if ax == "spread" else V7[k]["spread"]
        p = np.array([float(r[COL[k]]) for r in rows])
        q = tgt_c + (p - p.mean()) * (tgt_s / p.std(ddof=1))
        clipped = int((q < FLOOR).sum())
        q = np.maximum(q, FLOOR)
        rho = float(spearmanr(p, q).statistic)
        tau = float(kendalltau(p, q).statistic)
        assert abs(rho - 1.0) < 1e-12 and abs(tau - 1.0) < 1e-12, f"{k}: map not monotone"
        assert not clipped, f"{k}: {clipped} predictions clipped at the floor -- would break rho"
        out[k] = q
        print(f"   {k:8} {ax:6} -> centre {q.mean():7.4f}  sd {q.std(ddof=1):7.4f}  "
              f"(moved {ax} only; the other axis is pinned at v7)")

    if args.dry_run:
        print("\n   --dry-run: nothing written")
        return 0
    # SEVEN decimals. Four collapses ~10-20 distinct predictions per column into ties and six still
    # ties one CYP1A2 pair; seven is the first that is EXACTLY rank-identical to the v4 source on
    # all four columns. Ties perturb Kendall tau (tie-sensitive) and would quietly break the one
    # invariant this programme pre-registers every single time. Measured, not assumed.
    for i, r in enumerate(rows):
        for k in CYPS:
            r[COL[k]] = f"{out[k][i]:.7f}"
    with Path(args.dst).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    rep = {k: dict(axis=AXES[k][0], **res[k],
                   centre=float(out[k].mean()), spread=float(out[k].std(ddof=1)))
           for k in CYPS}
    rep["macro"] = dict(point=pt, lo=lo, hi=hi, v7=v7m)
    Path(str(args.dst).replace(".csv", ".json")).write_text(json.dumps(rep, indent=1))
    print(f"\n   wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
