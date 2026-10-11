"""Where is CYP2D6's ST-RAE optimum in spread, and how far should v8 step toward it?

Four scored board points now sit on this axis with the centre held at 3.43 throughout, so every
difference is attributable to spread alone. The secants are flattening fast (-0.351, -0.296,
-0.161), which means the optimum is close and OVERSHOOTING IT COSTS -- past the optimum the
derivative changes sign and further widening makes the score worse.

Model: fit the derivative (secant vs its own midpoint) linearly, i.e. treat ST-RAE as locally
quadratic in sd. The zero crossing is the optimum. Then compare candidate steps on what they
actually buy against how far they extrapolate beyond the measured range.

The point is NOT to land exactly on the estimated optimum. Near an optimum the gain is
second-order -- the last stretch buys almost nothing -- while the extrapolation risk is
first-order. So the step that captures most of the gain closest to the data is the right one.

    python scripts/cyp_d6_spread_opt.py
"""
from __future__ import annotations

import numpy as np

#: (sd actually submitted, ST-RAE the board returned). Centre held at 3.43 for all four.
POINTS = [(0.3002, 0.7456), (0.6004, 0.6403), (0.7805, 0.5870), (1.0007, 0.5517)]
CANDIDATES = (1.10, 1.15, 1.20, 1.25, 1.31, 1.40)


def main() -> int:
    sd = np.array([p[0] for p in POINTS])
    st = np.array([p[1] for p in POINTS])
    sec = np.diff(st) / np.diff(sd)
    mid = (sd[:-1] + sd[1:]) / 2.0

    print("measured spread axis (centre held at 3.43 throughout):")
    for a, b in POINTS:
        print(f"   sd {a:.4f}  ->  ST-RAE {b:.4f}")
    print("\nsecants, against their own midpoints:")
    for m, s in zip(mid, sec):
        print(f"   midpoint sd {m:.4f}  ->  dST/dsd {s:+.4f}")

    # Linear fit of derivative vs sd == quadratic ST-RAE. 3 points, 1 df: weak, and labelled so.
    A = np.vstack([np.ones_like(mid), mid]).T
    coef, *_ = np.linalg.lstsq(A, sec, rcond=None)
    b0, b1 = float(coef[0]), float(coef[1])
    opt = -b0 / b1
    resid = sec - (b0 + b1 * mid)
    print(f"\n   dST/dsd ~ {b0:+.4f} {b1:+.4f}*sd      residuals {np.round(resid, 4)}")
    print(f"   => ST-RAE optimum at sd ~ {opt:.3f}   (3 points, 1 df -- a weak fit, treat as a "
          f"direction, not a target)")

    cur_sd, cur_st = sd[-1], st[-1]
    print(f"\ncandidate steps from the current sd {cur_sd:.4f} (ST-RAE {cur_st:.4f}):")
    print(f"   {'target':>8} {'gain':>9} {'proj ST-RAE':>12} {'% of max gain':>14} "
          f"{'beyond data':>12}")
    # Integral of the fitted derivative from cur_sd to t, i.e. the quadratic's own prediction.
    def proj(t):
        return cur_st + b0 * (t - cur_sd) + b1 * (t * t - cur_sd * cur_sd) / 2.0

    best = proj(opt)
    max_gain = cur_st - best
    for t in CANDIDATES:
        g = cur_st - proj(t)
        print(f"   {t:8.2f} {-g:+9.4f} {proj(t):12.4f} {100*g/max_gain:13.0f}% "
              f"{t - cur_sd:+12.3f}")
    print(f"\n   the optimum itself ({opt:.3f}) is worth {max_gain:.4f} in total.")
    print("   Note how flat it is near the top: that is why landing exactly on the estimated")
    print("   optimum is not worth the extrapolation. Pick the step that takes most of the gain")
    print("   closest to the measured range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
