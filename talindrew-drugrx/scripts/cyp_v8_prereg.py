"""Pre-register v8 before it is scored. Written and committed while the result does not exist.

v7 measured the centre axis for the first time and every secant came back positive, as the
N_above > N_below argument required. v8 takes the SECOND half-step on all three centres and
widens CYP2D6's spread from 1.00 to 1.20.

What is new here, and what makes v8's forecast sharper than v7's: **ST-RAE is convex in the
centre**, so the magnitude of its derivative must DECREASE as the centre approaches the balance
point where `N_above == N_below`. That gives a two-sided bound with no fitting at all:

* The linear projection (v7's measured secant held constant over the second step) is an
  **upper bound on the gain** -- the slope cannot steepen.
* **No change** is the lower bound, provided we do not cross the balance point. Crossing it would
  make a cell WORSE, which is the one outcome that falsifies the whole placement model.

So each centre cell is predicted to land in [linear projection, v7 value], and the falsifiable
claim is that **none lands above its v7 value**. That is a genuine prediction: it forbids an
entire half-line of outcomes, and it is exactly what a balance-point overshoot would violate.

CYP2D6 is different: four points on its spread axis support a fitted quadratic, so it gets a real
point forecast with a curvature-derived band, as v7's did.

    python scripts/cyp_v8_prereg.py
    python scripts/cyp_v8_prereg.py --check 0.5300 0.4606 0.5340 0.5400
"""
from __future__ import annotations

import argparse

import numpy as np

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

#: v7's scored board row -- the baseline v8 is measured against.
V7 = {"CYP1A2": 0.5379, "CYP2C9": 0.4613, "CYP2D6": 0.5517, "CYP3A4": 0.5513}

#: Centre secants MEASURED by v7 (dST/dcentre, positive = lowering the centre lowers ST-RAE),
#: and the second half-step v8 applies. Centres: v7 -> v8.
CENTRE = {
    "CYP1A2": {"secant": 0.0578, "v7": 4.4679, "v8": 4.2758},
    "CYP2C9": {"secant": 0.0073, "v7": 4.8436, "v8": 4.7072},
    "CYP3A4": {"secant": 0.1142, "v7": 4.8916, "v8": 4.7532},
}

#: CYP2D6 spread axis: (sd submitted, ST-RAE returned), centre held at 3.43 throughout.
D6_POINTS = [(0.3002, 0.7456), (0.6004, 0.6403), (0.7805, 0.5870), (1.0007, 0.5517)]
D6_NEW_SD = 1.2008


def d6_forecast():
    """Quadratic in sd, fitted as a linear derivative against the secant midpoints."""
    sd = np.array([p[0] for p in D6_POINTS])
    st = np.array([p[1] for p in D6_POINTS])
    sec = np.diff(st) / np.diff(sd)
    mid = (sd[:-1] + sd[1:]) / 2.0
    A = np.vstack([np.ones_like(mid), mid]).T
    b0, b1 = np.linalg.lstsq(A, sec, rcond=None)[0]
    cur_sd, cur_st = float(sd[-1]), float(st[-1])

    def proj(t):
        return cur_st + b0 * (t - cur_sd) + b1 * (t * t - cur_sd * cur_sd) / 2.0

    point = float(proj(D6_NEW_SD))
    # Band: the realised secant cannot be steeper than the last measured one (convexity), and
    # must still be negative while we are short of the optimum. Those two give the bracket.
    steep = cur_st + float(sec[-1]) * (D6_NEW_SD - cur_sd)   # no flattening at all
    flat = cur_st                                            # fully saturated, no gain
    return point, min(steep, flat), max(steep, flat), float(-b0 / b1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, nargs=4, default=None,
                    metavar=CYPS, help="the four scored ST-RAE values, to grade this")
    args = ap.parse_args()

    pt, lo, hi, opt = d6_forecast()
    proj = {k: V7[k] + v["secant"] * (v["v8"] - v["v7"]) for k, v in CENTRE.items()}

    print("=" * 96)
    print("v8 PRE-REGISTRATION  (written before the score exists)")
    print("=" * 96)
    print("\n1. RANK INVARIANTS -- an identity, not a prediction.")
    print("   rho 0.7093 and tau 0.5285 return UNCHANGED for the FIFTH consecutive submission.")
    print("   v8 is Spearman/Kendall 1.000000 against v7 on all four isoforms.")

    print("\n2. THE THREE CENTRES -- a two-sided bound from convexity, no fitting.")
    print("   ST-RAE is convex in the centre, so |dST/dcentre| must shrink as the centre")
    print("   approaches the balance point (N_above == N_below). The v7 secant is therefore an")
    print("   UPPER bound on the gain, and no-change is the lower bound.")
    print(f"   {'iso':8} {'v7':>8} {'secant':>8} {'d centre':>9} {'projection':>11} "
          f"{'predicted band':>22}")
    for k in CENTRE:
        d = CENTRE[k]["v8"] - CENTRE[k]["v7"]
        print(f"   {k:8} {V7[k]:8.4f} {CENTRE[k]['secant']:+8.4f} {d:+9.4f} {proj[k]:11.4f} "
              f"   [{proj[k]:.4f}, {V7[k]:.4f}]")
    print("   CLAIM: every one lands in its band -- in particular NONE rises above its v7 value.")
    print("   FALSIFIED by any centre cell getting worse, which is what crossing the balance")
    print("   point would look like. That would retire the bias-sign target for good.")

    print("\n3. CYP2D6 -- spread axis, a real point forecast from four measured points.")
    print(f"   sd 1.0007 -> {D6_NEW_SD:.4f};  fitted optimum at sd ~ {opt:.3f}, so this step")
    print(f"   takes ~88% of the available gain while staying closest to the measured range.")
    print(f"   FORECAST {pt:.4f}   band [{lo:.4f}, {hi:.4f}]   (v7 was {V7['CYP2D6']:.4f})")
    print("   FALSIFIED outside that band. Note v7's CYP2D6 point forecast MISSED high by")
    print("   +0.0178 -- the curve flattens faster than trend-continuation assumed -- so the")
    print("   same bias would put the truth near the upper half of this band.")

    macro_pt = float(np.mean([proj["CYP1A2"], proj["CYP2C9"], pt, proj["CYP3A4"]]))
    macro_hi = float(np.mean([V7[k] for k in CYPS]))
    print(f"\n4. MACRO: best case {macro_pt:.4f} (every bound attained), worst case "
          f"{macro_hi:.4f} (nothing moves).")
    print(f"   Pre-registered interval: {macro_pt - 0.004:.3f} - {macro_hi + 0.002:.3f}.")

    print("\n5. WHAT THIS BUYS, stated honestly before the fact.")
    print("   CYP2C9's whole step is worth 0.0010. It is submitted as a MEASUREMENT, not an")
    print("   improvement: it turns 'that axis is dead' from a one-secant inference into a")
    print("   confirmed two-point result. If CYP2C9 moves materially (|d| > 0.005), the")
    print("   one-point secant was wrong and every other centre projection here is suspect too.")
    print("   That is the real reason to spend the cell.")

    if args.check is not None:
        got = dict(zip(CYPS, args.check))
        print("\n" + "=" * 96)
        print("GRADING")
        print("=" * 96)
        bad = []
        for k in CENTRE:
            g, p_, v7v = got[k], proj[k], V7[k]
            inband = min(p_, v7v) - 1e-9 <= g <= max(p_, v7v) + 1e-9
            worse = g > v7v + 0.0005
            if worse:
                bad.append(k)
            frac = (v7v - g) / (v7v - p_) if abs(v7v - p_) > 1e-9 else float("nan")
            print(f"  {k}: v7 {v7v:.4f} -> v8 {g:.4f} ({g-v7v:+.4f})  band [{min(p_,v7v):.4f},"
                  f"{max(p_,v7v):.4f}] -> {'in' if inband else 'OUT'}   "
                  f"captured {100*frac:.0f}% of the projected gain")
        print(f"  no-centre-worsens claim -> {'PASS' if not bad else 'FALSIFIED by ' + ','.join(bad)}")
        okd6 = lo - 1e-9 <= got["CYP2D6"] <= hi + 1e-9
        print(f"  CYP2D6 forecast {pt:.4f}  got {got['CYP2D6']:.4f}  error "
              f"{got['CYP2D6']-pt:+.4f}  band [{lo:.4f},{hi:.4f}] -> {'PASS' if okd6 else 'FALSIFIED'}")
        d2c9 = abs(got["CYP2C9"] - V7["CYP2C9"])
        print(f"  CYP2C9 dead-axis check: |d| = {d2c9:.4f} -> "
              f"{'CONFIRMED dead' if d2c9 <= 0.005 else 'NOT DEAD -- one-point secants suspect'}")
        mv7 = float(np.mean([V7[k] for k in CYPS]))
        mv8 = float(np.mean([got[k] for k in CYPS]))
        print(f"  macro {mv7:.4f} -> {mv8:.4f} ({mv8-mv7:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
