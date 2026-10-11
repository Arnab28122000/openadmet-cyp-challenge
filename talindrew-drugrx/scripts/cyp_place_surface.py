"""Evaluate the placement simulator at named placements, and calibrate it against the board.

`cyp_place_sim.py` finds its own optimum and checks itself at one point. This asks the
complementary question: **what does the simulator predict for placements we have actually
submitted, or are about to?** That turns it from an oracle into an instrument with a known
error, which is the only form in which it should be trusted.

The calibration is a single subtraction, and its assumption is stated rather than hidden:

    corrected(x) = sim(x) - [ sim(v4) - board(v4) ]

i.e. the simulator's error is taken to be a **constant offset** over the region of the
(centre, sd) surface being read. That is believable for small moves and not for large ones,
so the residual is printed per isoform and any isoform whose residual exceeds OFFSET_TRUST is
marked unusable rather than quietly corrected. On the 2026-10-09 run the residuals were
-0.0882 / -0.0350 / +0.0995 / +0.0034: **CYP2C9 and CYP3A4 are instruments, CYP1A2 and CYP2D6
are not.**

The placement v5 shipped is evaluated here too, which makes this a falsifiable prediction
rather than a retrospective fit: v5's board row lands about two hours after upload, and the
corrected numbers below say what it should score.

    python scripts/cyp_place_surface.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_place_sim import (ACTIVE_CUT, INACTIVE_R, SUBMITTED, bound_lookup,
                           soft_threshold_error, solve_mixture)
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.screen import screen_labels

ROOT = Path(__file__).resolve().parents[1]
OFFSET_TRUST = 0.05        # beyond this the simulator is not an instrument on that isoform

#: Solved from our own board row (scripts/cyp_solve_moments.py + cyp_bias_sign.py).
MOMENTS = {"CYP1A2": (4.2758, 1.4394), "CYP2C9": (4.7072, 0.9655),
           "CYP2D6": (3.0390, 1.6600), "CYP3A4": (4.7532, 1.1603)}
R = {"CYP1A2": 0.7747, "CYP2C9": 0.8215, "CYP2D6": 0.5180, "CYP3A4": 0.7832}

#: The placements worth reading off the surface.
PLACEMENTS = {
    "v4 (on the board)": {"CYP1A2": (4.66, 0.95), "CYP2C9": (4.98, 0.55),
                          "CYP2D6": (3.43, 0.30), "CYP3A4": (5.03, 0.75)},
    "v5 (submitted 04:05)": {"CYP1A2": (4.66, 1.00), "CYP2C9": (4.98, 0.68),
                             "CYP2D6": (3.43, 0.60), "CYP3A4": (5.03, 0.82)},
    "v6 cand a=0.5 @v5 sd": {"CYP1A2": (4.468, 1.00), "CYP2C9": (4.844, 0.68),
                             "CYP2D6": (3.234, 0.60), "CYP3A4": (4.892, 0.82)},
    "v6 cand a=1.0 @v5 sd": {"CYP1A2": (4.276, 1.00), "CYP2C9": (4.707, 0.68),
                             "CYP2D6": (3.039, 0.60), "CYP3A4": (4.753, 0.82)},
    # v5 widened all four; the distribution-free bound only ever demanded CYP2D6. These
    # candidates undo the three over-widenings and keep the one that was justified.
    "v4 sd, 2D6 -> 0.87": {"CYP1A2": (4.66, 0.95), "CYP2C9": (4.98, 0.55),
                           "CYP2D6": (3.43, 0.87), "CYP3A4": (5.03, 0.75)},
    "a=0.5, v4 sd, 2D6 .87": {"CYP1A2": (4.468, 0.95), "CYP2C9": (4.844, 0.55),
                              "CYP2D6": (3.234, 0.87), "CYP3A4": (4.892, 0.75)},
    "a=1.0, v4 sd, 2D6 .87": {"CYP1A2": (4.276, 0.95), "CYP2C9": (4.707, 0.55),
                              "CYP2D6": (3.039, 0.87), "CYP3A4": (4.753, 0.75)},
    # The simulator's own unconstrained optimum, for reference. Only its CYP2C9 and CYP3A4
    # cells are trustworthy -- the other two isoforms fail the residual check.
    "sim optimum": {"CYP1A2": (4.526, 0.936), "CYP2C9": (4.857, 0.531),
                    "CYP2D6": (3.139, 0.871), "CYP3A4": (4.903, 0.696)},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--draws", type=int, default=12)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/place_surface.json"))
    args = ap.parse_args()

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    screen_labels(frame)

    pool = []
    for c in PRIMARY:
        m = frame[c].notna() & (2.0 ** frame[f"log2fc_{c[:6]}"] >= INACTIVE_R)
        pool.append(frame.loc[m, c].to_numpy(float))
    floor_pool = np.concatenate(pool)
    rng = np.random.default_rng(0)

    report, rows = {}, {}
    for c in PRIMARY:
        iso = c[:6]
        y_tr = frame[c].to_numpy(float)
        lo_tr = frame[f"{c}_conf_low"].to_numpy(float)
        hi_tr = frame[f"{c}_conf_high"].to_numpy(float)
        ok = np.isfinite(y_tr) & np.isfinite(lo_tr) & np.isfinite(hi_tr)
        bounds = bound_lookup(y_tr[ok], lo_tr[ok], hi_tr[ok])
        mean_t, sd_t = MOMENTS[iso]
        sd_active = float(np.std(y_tr[ok & (y_tr >= ACTIVE_CUT)]))
        w, mu = solve_mixture(mean_t, sd_t, floor_pool, sd_active)
        n0 = int(args.n * w)
        y = np.concatenate([rng.choice(floor_pool, n0, replace=True),
                            rng.normal(mu, sd_active, args.n - n0)])
        lo, hi = bounds(y)
        yz = (y - y.mean()) / y.std()
        den = soft_threshold_error(np.full(len(y), y.mean()), lo, hi).sum()
        r = R[iso]

        def at(centre: float, sd: float, _yz=yz, _lo=lo, _hi=hi, _den=den, _r=r) -> float:
            """Mean simulated ST-RAE over independent predictor draws at this placement."""
            tot = 0.0
            g = np.random.default_rng(12345)
            for _ in range(args.draws):
                z = _r * _yz + np.sqrt(max(1 - _r * _r, 0.0)) * g.normal(size=len(_yz))
                z = (z - z.mean()) / z.std()
                tot += soft_threshold_error(centre + sd * z, _lo, _hi).sum() / _den
            return tot / args.draws

        sub = SUBMITTED[iso]
        resid = at(*PLACEMENTS["v4 (on the board)"][iso]) - sub["board"]
        rows[iso] = {"resid": resid, "trust": abs(resid) <= OFFSET_TRUST,
                     "board_v4": sub["board"],
                     "sim": {k: at(*v[iso]) for k, v in PLACEMENTS.items()}}
        report[iso] = rows[iso]

    print("Calibration: sim at v4's placement minus v4's actual board score.\n")
    print(f"{'iso':8} {'board v4':>9} {'sim v4':>8} {'residual':>9}   usable?")
    for iso, d in rows.items():
        print(f"{iso:8} {d['board_v4']:9.4f} {d['sim']['v4 (on the board)']:8.4f} "
              f"{d['resid']:+9.4f}   {'yes' if d['trust'] else 'NO -- do not correct, do not act'}")

    print(f"\nCorrected predictions (sim - residual). Isoforms marked NO above are shown for")
    print("completeness and must not be used to choose a placement.\n")
    hdr = f"{'placement':24}" + "".join(f"{i:>10}" for i in rows) + f"{'macro':>10}{'  (trusted only)':>18}"
    print(hdr)
    print("-" * len(hdr))
    for name in PLACEMENTS:
        vals, trusted = [], []
        line = f"{name:24}"
        for iso, d in rows.items():
            v = d["sim"][name] - d["resid"]
            vals.append(v)
            if d["trust"]:
                trusted.append(v)
            line += f"{v:10.4f}"
        line += f"{np.mean(vals):10.4f}{np.mean(trusted):18.4f}"
        print(line)

    print(f"\nBoard macro for v4 was 0.5925 across all four isoforms "
          f"({np.mean([d['board_v4'] for d in rows.values()]):.4f} from the per-isoform rows).")
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
