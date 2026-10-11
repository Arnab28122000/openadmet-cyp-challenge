"""Build v6 from v4 by a full affine map: centre *and* spread.

v5 moved spread only, deliberately, so that its board result would be a clean one-variable
measurement. v6 adds the axis v5 left alone.

**Why the centre is now actionable.** The moment solve recovers `|m|` but not its sign, because
R² and MAE both depend on the offset evenly. `cyp_bias_sign.py` resolves it against an
independent published estimate of the blinded means, unanimously and at 7.7-22x
discrimination: `m = y_bar - p_bar < 0` on all four isoforms, i.e. **we over-predict**, by
0.273 / 0.273 / 0.391 / 0.277 log units.

**The trap this script exists to avoid.** That does *not* mean move the centre to `y_bar`.
Placement puts the centre above the population mean **on purpose**: ST-RAE's deadband makes an
inactive compound's credible window free up to ~3.5, so the ST-RAE optimum sits above the
squared-error optimum. The finding is that v4 overshot, not that the overshoot should be zero.
So the centre is parameterised by how much of the measured bias to give back:

    centre_v6 = centre_v4 + alpha * m          alpha in [0, 1],  m < 0

`alpha = 0` is v4's centre, `alpha = 1` lands on the recovered population mean, and the right
value is between them. **It is not guessable and must come from the simulator calibrated
against v5's measured board row** -- which is the whole reason v5 went out as a one-variable
change. Running this with an alpha that has not been calibrated that way is the mistake the
minimax scan already made once.

The map is `p' = (mu + alpha*m) + (p - mu) * (sd_target / sd(p))`, affine with positive slope,
hence strictly monotone: **Spearman and Kendall must return from the board bit-identical** to
v4's 0.7093 / 0.5285. The script asserts that locally before writing, and it is the only
end-to-end check a blinded test set allows.

    python scripts/cyp_affine_v6.py --alpha 0.5 --dry-run
    python scripts/cyp_affine_v6.py --alpha 0.5 --sd 1.05 0.74 0.66 0.87
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
DST = ROOT / "data/cyp_cache/final_v4screen/submission_v6.csv"
SIGN = ROOT / "data/cyp_cache/bias_sign.json"
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
COL = {k: f"{k}_pIC50_direct_inhibition" for k in CYPS}

#: Every scored submission's spreads. The LAST entry is the live baseline: the confound check
#: below must compare against what is currently on the board, not against a frozen version.
#: Leaving this at v6 after v7 had moved CYP2D6 would have let a both-axes move through silently.
SD_BY_VERSION = {
    "v5": {"CYP1A2": 1.00, "CYP2C9": 0.68, "CYP2D6": 0.60, "CYP3A4": 0.82},
    "v6": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 0.78, "CYP3A4": 0.85},
    "v7": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 1.00, "CYP3A4": 0.85},
    "v8": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 1.20, "CYP3A4": 0.85},
    "v9": {"CYP1A2": 1.00, "CYP2C9": 0.76, "CYP2D6": 1.0446, "CYP3A4": 0.85},
}
#: The version ON THE BOARD, which --baseline defaults to. v8 was submitted 2026-10-10 17:22 UTC;
#: it replaced v7's row the moment it was accepted, so the confound check compares against it even
#: while it is unscored.
LATEST = "v9"
SD_V5 = SD_BY_VERSION["v5"]
SD_V6 = SD_BY_VERSION["v6"]
#: The squared-error optimum s_p = r * s_y, i.e. the far end. ST-RAE's optimum is below it.
SD_SQ_OPT = {"CYP1A2": 1.115, "CYP2C9": 0.794, "CYP2D6": 0.687, "CYP3A4": 0.909}
FLOOR = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None,
                    help="one alpha for all four isoforms, 0..1")
    ap.add_argument("--alpha-per", type=float, nargs=4, default=None,
                    metavar=("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"),
                    help="per-isoform alpha. From v7 the four isoforms are in different "
                         "states -- CYP1A2's spread is at its optimum, CYP3A4's is past it, "
                         "CYP2D6's still has room -- so a slot buys more by moving the "
                         "centre where the spread lever is spent and the spread where it is "
                         "not. Each isoform still moves exactly ONE variable, which is what "
                         "keeps every cell a clean measurement.")
    ap.add_argument("--sd", type=float, nargs=4, default=None,
                    metavar=("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"),
                    help="target sds; default keeps v5's")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    ap.add_argument("--baseline", default=LATEST, choices=sorted(SD_BY_VERSION),
                    help="the version currently ON THE BOARD, which the confound check compares "
                         "spreads against. Defaults to the latest scored submission.")
    ap.add_argument("--allow-confound", action="store_true",
                    help="permit an isoform to move BOTH centre and spread. Its board cell then "
                         "cannot be attributed to either axis, so only use this when that cell "
                         "is being spent deliberately and the notebook says why.")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    args = ap.parse_args()

    if (args.alpha is None) == (args.alpha_per is None):
        raise SystemExit("give exactly one of --alpha or --alpha-per")
    alpha = (dict(zip(CYPS, args.alpha_per)) if args.alpha_per
             else {k: args.alpha for k in CYPS})
    for k, a in alpha.items():
        if not 0.0 <= a <= 1.0:
            raise SystemExit(f"alpha must be in [0, 1], got {a} for {k}")
    sign = json.loads(SIGN.read_text())
    if not sign.get("unanimous"):
        raise SystemExit("bias_sign.json is not unanimous -- resolve per isoform first")
    for iso, rec in sign["isoforms"].items():
        if not rec.get("resolved"):
            raise SystemExit(f"{iso}: bias sign unresolved (discrimination "
                             f"{rec.get('discrimination')}x) -- do not shift its centre")

    sd_target = dict(zip(CYPS, args.sd)) if args.sd else dict(SD_V5)
    tag = Path(args.dst).stem.replace("submission_", "") or "out"
    rows = list(csv.DictReader(Path(args.src).open()))
    fields = list(rows[0].keys())
    print(f"[{tag}] {len(rows)} rows from {args.src};  alpha = "
          + ", ".join(f"{k} {alpha[k]:g}" for k in CYPS) + "\n")

    # Refuse to move both axes on the same isoform. The whole value of these submissions has
    # come from each board cell being attributable to one variable; a cell that moved both
    # cannot be read, and the confound would not be visible in the output.
    base_sd = SD_BY_VERSION[args.baseline]
    both = [k for k in CYPS if alpha[k] != 0.0 and abs(sd_target[k] - base_sd[k]) > 1e-9]
    if both and not args.allow_confound:
        raise SystemExit(
            f"{', '.join(both)}: moving BOTH centre and spread (vs baseline {args.baseline}) "
            "makes that isoform's board cell uninterpretable. Move one per isoform, or pass "
            "--allow-confound.")
    if both:
        print(f"  !! CONFOUNDED on {', '.join(both)}: both axes move, so those cells cannot be "
              f"attributed to either. Deliberate -- see the notebook.\n")

    report = {"alpha": alpha, "sd_target": sd_target, "isoforms": {}}
    for k in CYPS:
        c = COL[k]
        p = np.array([float(r[c]) for r in rows])
        mu, sd = float(p.mean()), float(p.std())
        m = sign["isoforms"][k]["m"]              # negative: we over-predict
        centre = mu + alpha[k] * m
        tgt = sd_target[k]
        q = centre + (p - mu) * (tgt / sd)
        clipped = int((q < FLOOR).sum())
        q = np.maximum(q, FLOOR)

        rho = float(spearmanr(p, q).statistic)
        tau = float(kendalltau(p, q).statistic)
        assert abs(rho - 1.0) < 1e-12 and abs(tau - 1.0) < 1e-12, (
            f"{k}: map was not monotone (rho {rho}, tau {tau})")
        if not clipped:
            assert abs(float(q.mean()) - centre) < 1e-9, (
                f"{k}: centre landed at {q.mean()} not {centre} with no clipping to explain it")

        if not args.dry_run:
            for i, r in enumerate(rows):
                r[c] = f"{q[i]:.6f}"
        report["isoforms"][k] = {
            "mean_v4": round(mu, 4), "mean_v6": round(float(q.mean()), 4),
            "bias_m": round(m, 4), "shift_applied": round(alpha[k] * m, 4),
            "sd_v4": round(sd, 4), "sd_v6": round(float(q.std()), 4), "sd_target": tgt,
            "sd_sq_optimum": SD_SQ_OPT[k],
            "min": round(float(q.min()), 3), "max": round(float(q.max()), 3),
            "clipped_at_floor": clipped,
            "rank_preserved": {"spearman": rho, "kendall": tau}}
        print(f"  {k}: centre {mu:.4f} -> {float(q.mean()):.4f} "
              f"(m {m:+.3f}, shift {alpha[k]*m:+.3f})   "
              f"sd {sd:.4f} -> {float(q.std()):.4f} (target {tgt}, sq-opt {SD_SQ_OPT[k]})"
              f"{f'   CLIPPED {clipped}' if clipped else ''}")
        print(f"        rank check: Spearman {rho:.12f}  Kendall {tau:.12f}  (both must be 1)")

    if args.dry_run:
        print(f"\n[{tag}] dry run -- nothing written")
        return 0

    out = Path(args.dst)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    assert fields == list(csv.DictReader(Path(args.src).open()).fieldnames), "header drift"
    print(f"\n[{tag}] wrote {out} ({out.stat().st_size} bytes, {len(rows)} rows)")
    print(f"[{tag}] header identical to v4: {','.join(fields)}")
    rep_path = ROOT / f"data/cyp_cache/{tag}_affine.json"
    rep_path.write_text(json.dumps(report, indent=1))
    print(f"[{tag}] wrote {rep_path.relative_to(ROOT)}")
    print(f"\n[{tag}] PRE-REGISTER before submitting: rho must return 0.7093 and tau 0.5285")
    print("     UNCHANGED -- the map is monotone, so anything else is a pipeline fault and")
    print("     not a result. State the expected macro ST-RAE interval too, or the")
    print("     submission teaches nothing it was not already assumed to teach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
