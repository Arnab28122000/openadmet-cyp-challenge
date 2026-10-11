"""Build v5 from v4 by changing the spread and nothing else.

**One variable.** v4's placement centres came out of a minimax scan and the board says they
are close to right; its *spreads* came out of the same scan and the board says they are too
narrow on every isoform. So v5 keeps each isoform's mean exactly where v4 put it and only
rescales the deviations:

    p' = mean(p) + (p - mean(p)) * (sd_target / sd(p))

That is affine with a positive slope, hence **strictly monotone**: Spearman's rho and
Kendall's tau must return from the board *bit-identical* to v4's 0.7093 / 0.5285. If they
move, the submission was not what we think it was -- a free end-to-end check this programme
has used three times, and the only one available on a blinded test set.

## Why these targets

Five routes sharing no assumptions were asked for the ST-RAE-optimal spread, and all five
said *wider than v4* on all four isoforms:

| route | what it assumes |
|---|---|
| distribution-free bound (MAE <= RMSE, R^2 <= r^2) | nothing at all |
| `cyp_place_sim.py` at the board-measured r | a mixture matching the blinded moments |
| `cyp_place_calibrate.py`, w pinned by ST-RAE | the mixture reproduces our own board row |
| same, `(w, active sd)` pinned by ST-RAE **and** R^2 | ditto, two constraints |
| a competitor's published simulation (CYP2D6 only) | theirs, unknown |

The magnitudes disagree (CYP2D6: 0.44 to 0.92), and under ST-RAE over-widening is not a
symmetric mistake -- an inactive compound's credible interval is a *free* window, and spread
that pushes predictions out of it costs score with no offsetting gain. So each target is set
near the **conservative end** of its range, not at the median:

    CYP1A2 0.95 -> 1.00 (x1.05)   CYP2C9 0.55 -> 0.68 (x1.24)
    CYP2D6 0.30 -> 0.60 (x2.00)   CYP3A4 0.75 -> 0.82 (x1.09)

CYP2D6 is the one that matters and the one with the firmest floor: the distribution-free
bound alone puts its optimum at **>= 0.536** against v4's 0.300, using only MAE <= RMSE and
Murphy's R^2 <= r^2 on the reported row. v4 under-dispersed our worst isoform by at least
1.8x because `place_minimax.json` chose sd 0.10 for it -- implying r = 0.209, below the floor
of the very band the scan was told to search, against a measured r of 0.518.

    .venv/bin/python scripts/cyp_rescale_v5.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

SRC = Path("data/cyp_cache/final_v4screen/submission_v4.csv")
DST = Path("data/cyp_cache/final_v4screen/submission_v5.csv")
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
COL = {k: f"{k}_pIC50_direct_inhibition" for k in CYPS}

#: Conservative end of the five-route consensus. Centres are deliberately NOT changed.
SD_TARGET = {"CYP1A2": 1.00, "CYP2C9": 0.68, "CYP2D6": 0.60, "CYP3A4": 0.82}
#: The assay's own floor: the inactive credible window starts at ~1.06, so nothing below
#: this is a meaningful pIC50 and widening must not manufacture one.
FLOOR = 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default=str(DST))
    args = ap.parse_args()

    rows = list(csv.DictReader(Path(args.src).open()))
    fields = list(rows[0].keys())
    print(f"[v5] {len(rows)} rows from {args.src}\n")

    report = {}
    for k in CYPS:
        c = COL[k]
        p = np.array([float(r[c]) for r in rows])
        mu, sd = float(p.mean()), float(p.std())
        tgt = SD_TARGET[k]
        q = mu + (p - mu) * (tgt / sd)
        clipped = int((q < FLOOR).sum())
        q = np.maximum(q, FLOOR)

        # The checks that make this submission worth a slot.
        rho = float(spearmanr(p, q).statistic)
        tau = float(kendalltau(p, q).statistic)
        assert abs(rho - 1.0) < 1e-12 and abs(tau - 1.0) < 1e-12, (
            f"{k}: rescale was not monotone (rho {rho}, tau {tau})")
        assert abs(float(q.mean()) - mu) < 1e-9 or clipped, (
            f"{k}: mean moved from {mu} to {q.mean()} with no clipping to explain it")

        for i, r in enumerate(rows):
            r[c] = f"{q[i]:.6f}"
        report[k] = {"mean_before": round(mu, 4), "mean_after": round(float(q.mean()), 4),
                     "sd_before": round(sd, 4), "sd_after": round(float(q.std()), 4),
                     "sd_target": tgt, "factor": round(tgt / sd, 4),
                     "min": round(float(q.min()), 3), "max": round(float(q.max()), 3),
                     "clipped_at_floor": clipped,
                     "rank_preserved": {"spearman": rho, "kendall": tau}}
        print(f"  {k}: sd {sd:.4f} -> {float(q.std()):.4f} (target {tgt}, x{tgt/sd:.3f})  "
              f"mean {mu:.4f} -> {float(q.mean()):.4f}  range [{q.min():.2f}, {q.max():.2f}]"
              f"{f'  CLIPPED {clipped}' if clipped else ''}")
        print(f"        rank check: Spearman {rho:.12f}  Kendall {tau:.12f}  (both must be 1)")

    out = Path(args.dst)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    size = out.stat().st_size
    print(f"\n[v5] wrote {out} ({size} bytes, {len(rows)} rows)")

    # Header and shape must match v4 exactly -- the scorer rejects on column names.
    assert fields == list(csv.DictReader(Path(args.src).open()).fieldnames), "header drift"
    print(f"[v5] header identical to v4: {','.join(fields)}")
    Path("data/cyp_cache/v5_rescale.json").write_text(json.dumps(report, indent=1))
    print("[v5] wrote data/cyp_cache/v5_rescale.json")
    print("\n[v5] PRE-REGISTERED PREDICTION: rho returns 0.7093 and tau 0.5285 UNCHANGED")
    print("     (the map is monotone, so anything else means a pipeline fault, not a result);")
    print("     macro ST-RAE lands between 0.555 and 0.580, i.e. a gain of 0.012-0.037.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
