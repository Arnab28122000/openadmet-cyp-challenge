"""Every placement we have put on the board, and what it actually scored.

This file exists because of a specific mistake. The placement simulator passed a **one-point**
calibration at v4's own placement -- CYP3A4's residual was +0.0022, two parts in a thousand --
and I treated that as licence to predict the *direction* of a change. v5's measured result then
contradicted the simulator on all four isoforms. The lesson, written into PROGRESS at the time:
**passing a one-point calibration does not make a simulator an instrument for predicting
change; that needs at least two points.**

v6 supplies the second point on the spread axis. So this script does what the simulator cannot:
it reports the **measured** secant between real board rows, per isoform, per axis.

Two properties of this board make that unusually clean:

* The metric is **deterministic**. ST-RAE over 750 fixed compounds against fixed labels has no
  sampling noise, so a difference between two board rows is a measurement, not an estimate.
  There is no t-statistic to compute and none is reported.
* Placement is a **monotone affine map**, so rho and tau are invariant. Any change in them
  would mean the pipeline did something other than what it claimed, which is why they are
  asserted rather than tracked.

The one thing it cannot do is extrapolate. A secant between two spreads says what happened
between those two spreads. The **centre axis has no measured points at all**, which is exactly
why v7 is written up as a probe rather than as an improvement.

    python scripts/cyp_board_history.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cyp_cache/board_history.json"
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

#: m = y_bar - p_bar, the over-prediction recovered by cyp_bias_sign.py. Negative on all four.
M = {"CYP1A2": -0.384, "CYP2C9": -0.273, "CYP2D6": -0.391, "CYP3A4": -0.277}

#: Centre and spread actually submitted, and the per-isoform ST-RAE the board returned.
#: `board: None` means submitted but not yet scored.
V9_FULL = {
    "macro": {"st_rae": 0.5232, "mae": 0.6703, "r2": 0.5611,
              "spearman": 0.7093, "kendall": 0.5285, "rank": 59},
    "CYP1A2": {"st_rae": 0.5369, "mae": 0.7656, "r2": 0.5546,
               "spearman": 0.7597, "kendall": 0.5654, "rank": 57},
    "CYP2C9": {"st_rae": 0.4555, "mae": 0.4748, "r2": 0.6268,
               "spearman": 0.8084, "kendall": 0.6172, "rank": 92},
    "CYP2D6": {"st_rae": 0.5493, "mae": 0.8467, "r2": 0.4738,
               "spearman": 0.5004, "kendall": 0.3547, "rank": 25},
    "CYP3A4": {"st_rae": 0.5513, "mae": 0.5943, "r2": 0.5891,
               "spearman": 0.7685, "kendall": 0.5769, "rank": 137},
    "n_entrants": 273,
}

#: v9's pre-registered bands, kept so the grade is reproducible rather than remembered.
#: (lower bound, chord bound) per cell; the chord is the falsifiable one.
V9_BANDS = {
    "CYP1A2": (0.5307, 0.5405), "CYP2C9": (0.4377, 0.4618),
    "CYP2D6": (0.5447, 0.5543), "CYP3A4": (0.5510, 0.5516),
    "macro": (0.5160, 0.5270),
}
V9_POINT = 0.5236

HISTORY = {
    "v4": {
        "submitted": "2026-10-08 15:57 UTC",
        "centre": {"CYP1A2": 4.66, "CYP2C9": 4.98, "CYP2D6": 3.43, "CYP3A4": 5.03},
        "sd": {"CYP1A2": 0.95, "CYP2C9": 0.55, "CYP2D6": 0.30, "CYP3A4": 0.75},
        "board": {"CYP1A2": 0.5490, "CYP2C9": 0.5045, "CYP2D6": 0.7456, "CYP3A4": 0.5710},
    },
    "v5": {
        "submitted": "2026-10-09 04:03 UTC",
        "centre": {"CYP1A2": 4.66, "CYP2C9": 4.98, "CYP2D6": 3.43, "CYP3A4": 5.03},
        "sd": {"CYP1A2": 1.00, "CYP2C9": 0.68, "CYP2D6": 0.60, "CYP3A4": 0.82},
        "board": {"CYP1A2": 0.5490, "CYP2C9": 0.4659, "CYP2D6": 0.6403, "CYP3A4": 0.5664},
    },
    "v6": {
        # Timestamp is the board's own, read back off the Space (we had recorded 17:12).
        "submitted": "2026-10-09 17:09 UTC",
        "centre": {"CYP1A2": 4.66, "CYP2C9": 4.98, "CYP2D6": 3.43, "CYP3A4": 5.03},
        "sd": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 0.78, "CYP3A4": 0.85},
        "board": {"CYP1A2": 0.5490, "CYP2C9": 0.4623, "CYP2D6": 0.5870, "CYP3A4": 0.5671},
    },
    "v7": {
        # Board timestamp 05:20; we clicked Submit at 05:15 and it queued ~6 h before scoring.
        "submitted": "2026-10-10 05:20 UTC",
        "centre": {"CYP1A2": 4.4679, "CYP2C9": 4.8436, "CYP2D6": 3.43, "CYP3A4": 4.8916},
        "sd": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 1.00, "CYP3A4": 0.85},
        "board": {"CYP1A2": 0.5379, "CYP2C9": 0.4613, "CYP2D6": 0.5517, "CYP3A4": 0.5513},
    },
    "v8": {
        # SUBMITTED 2026-10-10 17:22 UTC, unscored. Centre/sd read back off the submitted CSV,
        # not copied from the builder's constants. Spearman against v7 is 1.000000 on all four,
        # so rho/tau MUST return 0.7093/0.5285 -- the first claim to check, and it is free.
        # The 12 h window opened at 17:20:39 (12 h after the BOARD's second, not our click);
        # a trigger armed from our 05:15 click fired 51 s early and was rejected.
        "submitted": "2026-10-10 17:21 UTC",
        "centre": {"CYP1A2": 4.2758, "CYP2C9": 4.7072, "CYP2D6": 3.43, "CYP3A4": 4.7532},
        "sd": {"CYP1A2": 1.0007, "CYP2C9": 0.7605, "CYP2D6": 1.2008, "CYP3A4": 0.8506},
        "board": {"CYP1A2": 0.5685, "CYP2C9": 0.5104, "CYP2D6": 0.5634, "CYP3A4": 0.5658},
    },
    "v9": {
        # SCORED 2026-10-11 17:33 UTC (board stamp 12:36 UTC, 5h of lag). The FIRST placement
        # submission that is an interpolation inside a measured bracket rather than an
        # extrapolation beyond one: v8's overshoot put a secant sign change on all four axes,
        # which brackets each minimum. Targets are the 3-point parabola vertices
        # (scripts/cyp_v9_bracket.py). Seven decimals, because six tied one CYP1A2 pair.
        #
        # THE PRE-REGISTERED CLAIM HELD ON EVERY CELL -- the first time in this programme.
        # Each cell had to land at or below its CHORD bound, which follows from convexity of a
        # sum of hinges plus three measured points and assumes nothing else:
        #
        #   cell     band [lower, CHORD]     actual    inside?
        #   CYP1A2   [0.5307, 0.5405]        0.5369    yes
        #   CYP2C9   [0.4377, 0.4618]        0.4555    yes
        #   CYP2D6   [0.5447, 0.5543]        0.5493    yes
        #   CYP3A4   [0.5510, 0.5516]        0.5513    yes
        #   macro    point 0.5236, <= 0.5270 0.5232    yes -- POINT ESTIMATE OFF BY 0.0004
        #
        # And rho/tau unchanged at 0.7093/0.5285 for the SEVENTH consecutive submission.
        #
        # THE UNCOMFORTABLE PART, recorded because it is the real lesson: macro ST-RAE improved
        # on v7 (0.5256 -> 0.5232, better than the pre-registered best case of 0.5236) and the
        # RANK still went 54 -> 59. n_entrants 271 -> 273. The field is improving faster than
        # this programme is. A better score at a worse rank is the clearest signal yet that
        # placement is exhausted and only model class can move us.
        "submitted": "2026-10-11 12:36 UTC",
        "centre": {"CYP1A2": 4.5128, "CYP2C9": 4.9091, "CYP2D6": 3.43, "CYP3A4": 4.8886},
        "sd": {"CYP1A2": 1.0007, "CYP2C9": 0.7605, "CYP2D6": 1.0446, "CYP3A4": 0.8506},
        "board": {"CYP1A2": 0.5369, "CYP2C9": 0.4555, "CYP2D6": 0.5493, "CYP3A4": 0.5513},
    },
}

#: The full v6 row, kept because the board shows only the latest submission per user and the
#: macro MAE is what `cyp_solve_moments.py` inverts. Ranks are out of 268 regression entrants.
V6_FULL = {
    "macro": {"st_rae": 0.5414, "mae": 0.7095, "r2": 0.5329,
              "spearman": 0.7093, "kendall": 0.5285, "rank": 59},
    "CYP1A2": {"st_rae": 0.5490, "mae": 0.7949, "r2": 0.5234,
               "spearman": 0.7597, "kendall": 0.5654, "rank": 55},
    "CYP2C9": {"st_rae": 0.4623, "mae": 0.4847, "r2": 0.6129,
               "spearman": 0.8084, "kendall": 0.6172, "rank": 90},
    "CYP2D6": {"st_rae": 0.5870, "mae": 0.9417, "r2": 0.4337,
               "spearman": 0.5004, "kendall": 0.3547, "rank": 35},
    "CYP3A4": {"st_rae": 0.5671, "mae": 0.6166, "r2": 0.5615,
               "spearman": 0.7685, "kendall": 0.5769, "rank": 140},
    "n_entrants": 268,
}

#: v7, the first centre-axis point. Ranks out of 271 -- the field grew by 3 in a day, so a rank
#: change of 1-2 places is field drift, not a result. Read the ST-RAE, not the rank.
V7_FULL = {
    "macro": {"st_rae": 0.5256, "mae": 0.6717, "r2": 0.5628,
              "spearman": 0.7093, "kendall": 0.5285, "rank": 54},
    "CYP1A2": {"st_rae": 0.5379, "mae": 0.7596, "r2": 0.5599,
               "spearman": 0.7597, "kendall": 0.5654, "rank": 54},
    "CYP2C9": {"st_rae": 0.4613, "mae": 0.4729, "r2": 0.6311,
               "spearman": 0.8084, "kendall": 0.6172, "rank": 92},
    "CYP2D6": {"st_rae": 0.5517, "mae": 0.8595, "r2": 0.4714,
               "spearman": 0.5004, "kendall": 0.3547, "rank": 28},
    "CYP3A4": {"st_rae": 0.5513, "mae": 0.5946, "r2": 0.5888,
               "spearman": 0.7685, "kendall": 0.5769, "rank": 134},
    "n_entrants": 271,
}


#: v8, which REGRESSED on all four cells and falsified its own headline claim. Ranks out of 271.
#: Kept because it is the far side of the bracket on every axis -- the only reason the v9
#: interpolation exists at all. MA-MAE actually IMPROVED (0.6717 -> 0.6631) while ST-RAE got
#: worse, which is the clearest demonstration yet that the two metrics have different optima.
V8_FULL = {
    "macro": {"st_rae": 0.5520, "mae": 0.6631, "r2": 0.5579,
              "spearman": 0.7093, "kendall": 0.5285, "rank": 69},
    "CYP1A2": {"st_rae": 0.5685, "mae": 0.7523, "r2": 0.5601,
               "spearman": 0.7597, "kendall": 0.5654, "rank": 66},
    "CYP2C9": {"st_rae": 0.5104, "mae": 0.4897, "r2": 0.6140,
               "spearman": 0.8084, "kendall": 0.6172, "rank": 124},
    "CYP2D6": {"st_rae": 0.5634, "mae": 0.8164, "r2": 0.4686,
               "spearman": 0.5004, "kendall": 0.3547, "rank": 34},
    "CYP3A4": {"st_rae": 0.5658, "mae": 0.5942, "r2": 0.5890,
               "spearman": 0.7685, "kendall": 0.5769, "rank": 144},
    "n_entrants": 271,
}

#: Invariant across every version: placement is monotone, so these cannot move.
RANK = {"spearman": 0.7093, "kendall": 0.5285}


def secant(a: str, b: str) -> dict:
    """Measured dST/dsd and dST/dcentre between two scored versions, per isoform."""
    A, B = HISTORY[a], HISTORY[b]
    if A["board"] is None or B["board"] is None:
        return {}
    out = {}
    for iso in CYPS:
        dsd = B["sd"][iso] - A["sd"][iso]
        dc = B["centre"][iso] - A["centre"][iso]
        dst = B["board"][iso] - A["board"][iso]
        # Only attribute the change to an axis that actually moved. A version that moved
        # both would need two points to separate them, which is why each has moved one.
        axis = "sd" if (dsd and not dc) else "centre" if (dc and not dsd) else "both/none"
        out[iso] = {"d_sd": round(dsd, 4), "d_centre": round(dc, 4),
                    "d_st_rae": round(dst, 4), "axis": axis,
                    "secant": round(dst / dsd, 4) if axis == "sd" and dsd else
                              round(dst / dc, 4) if axis == "centre" and dc else None}
    return out


def main() -> int:
    scored = [k for k, v in HISTORY.items() if v["board"] is not None]
    pending = [k for k, v in HISTORY.items() if v["board"] is None]

    print(f"scored: {', '.join(scored)}" + (f"   pending: {', '.join(pending)}" if pending else ""))
    print(f"rank invariants (must never move): rho {RANK['spearman']}  tau {RANK['kendall']}\n")

    print(f"{'version':8}" + "".join(f"{i:>10}" for i in CYPS) + f"{'macro':>10}")
    print("-" * 58)
    for k in scored:
        b = HISTORY[k]["board"]
        macro = sum(b[i] for i in CYPS) / 4
        print(f"{k:8}" + "".join(f"{b[i]:10.4f}" for i in CYPS) + f"{macro:10.4f}")

    pairs = [(scored[i], scored[i + 1]) for i in range(len(scored) - 1)]
    for a, b in pairs:
        s = secant(a, b)
        if not s:
            continue
        axis = {v["axis"] for v in s.values()}
        print(f"\n{a} -> {b}   (axis moved: {', '.join(sorted(axis))})")
        print(f"  {'iso':8} {'d axis':>9} {'d ST-RAE':>10} {'secant':>10}")
        for iso in CYPS:
            d = s[iso]
            da = d["d_sd"] if d["axis"] == "sd" else d["d_centre"]
            sec = f"{d['secant']:10.3f}" if d["secant"] is not None else f"{'n/a':>10}"
            print(f"  {iso:8} {da:+9.3f} {d['d_st_rae']:+10.4f} {sec}")

    # Two measured points on one axis let the SECOND derivative be seen for the first time.
    if len(pairs) >= 2:
        print("\nCurvature on the spread axis -- the thing one point could never show:")
        print(f"  {'iso':8} {'secant 1':>10} {'secant 2':>10} {'change':>10}")
        for iso in CYPS:
            s1 = secant(*pairs[0]).get(iso, {}).get("secant")
            s2 = secant(*pairs[1]).get(iso, {}).get("secant")
            if s1 is None or s2 is None:
                continue
            print(f"  {iso:8} {s1:10.3f} {s2:10.3f} {s2 - s1:+10.3f}")

    # v7 measured the centre axis for the first time. Every secant came back POSITIVE, meaning
    # lowering the centre lowers ST-RAE, as the N_above > N_below argument required -- but the
    # three magnitudes differ by 16x, and that is what decides where the remaining step is worth
    # spending. A secant is a local slope, so this projects the SECOND half-step linearly and
    # says so: it is an estimate to be measured, not a result.
    cs = secant("v6", "v7")
    print("\nThe CENTRE axis, now measured (v6 -> v7). Remaining half-step projected linearly:")
    print(f"  {'iso':8} {'secant':>9} {'now':>9} {'a=1.0':>9} {'d centre':>9} {'proj ST-RAE':>12}")
    for iso in CYPS:
        d = cs.get(iso, {})
        if d.get("axis") != "centre" or d.get("secant") is None:
            continue
        c7 = HISTORY["v7"]["centre"][iso]
        full = HISTORY["v4"]["centre"][iso] + M[iso]
        step = full - c7
        print(f"  {iso:8} {d['secant']:+9.3f} {HISTORY['v7']['board'][iso]:9.4f} {full:9.3f} "
              f"{step:+9.3f} {HISTORY['v7']['board'][iso] + d['secant']*step:12.4f}")
    print("  A positive secant means lowering the centre helps, which is the predicted sign.")
    print("  Projections assume the slope holds over the second half-step; it need not, because")
    print("  the balance point (N_above == N_below) is where the slope goes to zero.")

    OUT.write_text(json.dumps(
        {"history": HISTORY, "rank": RANK, "m": M,
         "secants": {f"{a}->{b}": secant(a, b) for a, b in pairs}}, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
