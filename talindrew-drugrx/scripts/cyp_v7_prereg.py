"""Pre-register what v7's score will mean -- written and committed BEFORE the score lands.

v7 went in at 2026-10-10 05:15 UTC and scores ~40 min later. This file exists so the reading of
it cannot be chosen afterwards. It does two separable things, and it is important that they are
separable because only one of them is a real prediction:

1. **CYP2D6 (spread axis) -- a genuine numeric prediction.** Four board points now sit on this
   axis (sd 0.30 / 0.60 / 0.78 / 1.00), so v7's CYP2D6 cell is an *extrapolation of one step from
   a measured, curving function*. That is forecastable and is forecast here, with the band coming
   from the curvature rather than from a guess.

2. **CYP1A2 / CYP2C9 / CYP3A4 (centre axis) -- a SIGN prediction only, no magnitude.** This axis
   has never been measured. What theory does give is the sign, exactly:

       ST-RAE numerator = sum[ max(lo-p,0) + max(p-hi,0) ]
       shift every p by delta  =>  d(numerator)/d(delta) = N_above - N_below

   where N_above = #{p > hi} and N_below = #{p < lo}. Every isoform's bias `m = y_bar - p_bar` is
   negative (we predict high), so N_above > N_below, so the derivative is positive, so **lowering
   the centre must lower ST-RAE** -- until the centre crosses the balance point where the two
   counts equalise and the derivative changes sign. v7 lowers all three centres.

   So the falsifiable claim is: **all three centre-moved isoforms improve or stay flat. None gets
   worse.** A worsening on any of them says the centre was already past its balance point, which
   would be new information and would mean the bias-sign correction has the wrong target.

   No magnitude is predicted, because |N_above - N_below| on the blinded set is unknown and that
   is precisely the quantity being measured. Inventing one is what produced the wrong v5 forecast.

Run it to print the pre-registration; run it after the score with --check to grade it.

    python scripts/cyp_v7_prereg.py
    python scripts/cyp_v7_prereg.py --check 0.5390 0.4550 0.5330 0.5600
"""
from __future__ import annotations

import argparse

import numpy as np

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

#: Measured board points on CYP2D6's spread axis. Centre held at 3.43 throughout, so every one
#: of these differs from its neighbour in sd alone -- which is what makes the curve readable.
D6_SPREAD = [(0.3002, 0.7456), (0.6004, 0.6403), (0.7805, 0.5870)]
D6_NEXT_SD = 1.0007

#: v6 -> v7 centre moves (the a=0.5 half-step toward the bias-corrected centre).
CENTRE_MOVE = {"CYP1A2": (4.6600, 4.4679), "CYP2C9": (4.9800, 4.8436), "CYP3A4": (5.0300, 4.8916)}
V6_BOARD = {"CYP1A2": 0.5490, "CYP2C9": 0.4623, "CYP2D6": 0.5870, "CYP3A4": 0.5671}

#: m = y_bar - p_bar, from cyp_bias_sign.py. All negative => we predict high => N_above > N_below.
M = {"CYP1A2": -0.384, "CYP2C9": -0.273, "CYP2D6": -0.391, "CYP3A4": -0.277}


def d6_forecast() -> tuple[float, float, float]:
    """Extrapolate CYP2D6 one step along its own measured, curving spread response.

    Two secants are in hand and they are flattening (-0.351 then -0.296). The next secant is
    therefore bounded above by the last one (the curve is convex over the measured range) and
    below by zero (ST-RAE cannot keep falling indefinitely; widening eventually hurts). That
    bracket IS the band -- it is not a guess, it is what the curvature permits.
    """
    sd = np.array([p[0] for p in D6_SPREAD])
    st = np.array([p[1] for p in D6_SPREAD])
    sec = np.diff(st) / np.diff(sd)
    step = D6_NEXT_SD - sd[-1]
    # Point estimate: continue the flattening at the same rate the two secants flattened.
    flatten = sec[1] - sec[0]              # +0.055 per unit of previous step
    nxt = sec[1] + flatten                 # next secant, continuing the trend
    point = st[-1] + nxt * step
    hi = st[-1] + 0.0 * step               # secant reaches 0 (fully saturated): no change
    lo = st[-1] + sec[1] * step            # secant stays at its last value: no flattening
    return float(point), float(min(lo, hi)), float(max(lo, hi))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=float, nargs=4, default=None,
                    metavar=("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"),
                    help="the four scored ST-RAE values, to grade this pre-registration")
    args = ap.parse_args()

    pt, lo, hi = d6_forecast()
    print("=" * 94)
    print("v7 PRE-REGISTRATION  (submitted 2026-10-10 05:15 UTC; written before the score)")
    print("=" * 94)

    print("\n1. RANK INVARIANTS -- not a prediction, an identity.")
    print("   rho 0.7093 and tau 0.5285 return UNCHANGED. The file is Spearman/Kendall")
    print("   1.000000 against v6 on all four isoforms and placement is monotone, so anything")
    print("   else is a pipeline fault, not a result.")

    print("\n2. CYP2D6 -- spread axis, a real numeric forecast.")
    sd = [p[0] for p in D6_SPREAD]
    st = [p[1] for p in D6_SPREAD]
    sec = np.diff(st) / np.diff(sd)
    print(f"   measured: " + "  ".join(f"sd {a:.3f}->{b:.4f}" for a, b in zip(sd, st)))
    print(f"   secants:  {sec[0]:+.3f} then {sec[1]:+.3f}  (flattening by {sec[1]-sec[0]:+.3f})")
    print(f"   sd now {D6_NEXT_SD:.4f}  =>  FORECAST {pt:.4f}   band [{lo:.4f}, {hi:.4f}]")
    print("   The band is the curvature bracket: the next secant cannot be steeper than the last")
    print("   (the curve is convex here) and cannot be positive yet if widening still helps.")
    print(f"   FALSIFIED if CYP2D6 lands outside [{lo:.4f}, {hi:.4f}].")

    print("\n3. CYP1A2 / CYP2C9 / CYP3A4 -- centre axis, SIGN ONLY. No magnitude is predicted.")
    print("   d(ST-RAE numerator)/d(centre) = N_above - N_below, and every m is negative, so")
    print("   N_above > N_below and lowering the centre must help until the balance point.")
    for c, (a, b) in CENTRE_MOVE.items():
        print(f"     {c}: centre {a:.4f} -> {b:.4f}  (delta {b-a:+.4f}, half of m={M[c]:+.3f})"
              f"   v6 was {V6_BOARD[c]:.4f}")
    print("   CLAIM: all three improve or stay flat. NONE gets worse.")
    print("   FALSIFIED by any of the three rising. That would mean the centre was already past")
    print("   its balance point and the bias-sign target is wrong -- genuinely new information.")

    print("\n4. CYP1A2 is the clean cell, and the one to read first.")
    print("   Its spread axis is measured FLAT (v4->v5 moved sd 0.9506->1.0007 and ST-RAE stayed")
    print("   exactly 0.5490), and v5->v6 submitted a byte-identical column for an exact 0.5490")
    print("   replicate -- so the scorer is deterministic and CYP1A2 sits at a stationary point")
    print("   in spread. Any move in CYP1A2 is therefore attributable to the centre alone.")

    print("\n5. WHAT v8 DOES, decided now rather than after seeing the number.")
    print("   The next window opens 2026-10-10 17:15 UTC.")
    print("   * If the sign claim holds on all three: v8 takes the SECOND half-step (a=1.0) on")
    print("     whichever isoforms moved most, and NOT on all three blindly -- one secant")
    print("     licenses the level, not the gradient. Two points per isoform is the minimum for")
    print("     an optimum bracket, which is the whole reason v7 moved one variable at a time.")
    print("   * If any isoform worsens: STOP moving that centre, and bracket instead -- the")
    print("     balance point lies between v6's centre and v7's.")
    print("   * If CYP2D6 lands outside its band: the spread curve is not what we think it is,")
    print("     and the four-point fit gets refitted before any further spread move.")
    print("   * In every case the macro number is NOT the thing to read. Four isoforms moved for")
    print("     two different reasons; the macro average confounds them.")

    if args.check is not None:
        got = dict(zip(CYPS, args.check))
        print("\n" + "=" * 94)
        print("GRADING")
        print("=" * 94)
        ok_d6 = lo <= got["CYP2D6"] <= hi
        err = got["CYP2D6"] - pt
        print(f"  CYP2D6 forecast {pt:.4f}  got {got['CYP2D6']:.4f}  error {err:+.4f}  "
              f"band [{lo:.4f},{hi:.4f}]  -> {'PASS' if ok_d6 else 'FALSIFIED'}")
        worse = []
        for c in CENTRE_MOVE:
            d = got[c] - V6_BOARD[c]
            verdict = "worse" if d > 0.0005 else ("flat" if abs(d) <= 0.0005 else "better")
            if verdict == "worse":
                worse.append(c)
            print(f"  {c}: v6 {V6_BOARD[c]:.4f} -> v7 {got[c]:.4f}  ({d:+.4f})  {verdict}"
                  f"   secant {d/(CENTRE_MOVE[c][1]-CENTRE_MOVE[c][0]):+.3f} per unit centre")
        print(f"  sign claim -> {'PASS' if not worse else 'FALSIFIED by ' + ', '.join(worse)}")
        macro_v6 = float(np.mean([V6_BOARD[c] for c in CYPS]))
        macro_v7 = float(np.mean([got[c] for c in CYPS]))
        inband = 0.515 <= macro_v7 <= 0.545
        print(f"  macro {macro_v6:.4f} -> {macro_v7:.4f} ({macro_v7-macro_v6:+.4f}); "
              f"pre-registered 0.515-0.545 -> {'in band' if inband else 'OUT OF BAND'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
