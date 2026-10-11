"""How much of our ST-RAE deficit is ranking, and how much is placement?

The board says these are different things and we had been treating the second as nearly
settled. At rank 30, `tguttenb1` scores macro ST-RAE **0.4737** with macro Spearman **0.7223**
-- +0.013 of ranking over ours -- while scoring macro R2 **0.5907** against our **0.4472**.
Nearly the whole 0.119 of ST-RAE between us sits in R2 at the same ranking, and R2 at fixed
ranking is placement.

Murphy (1988) decomposes it exactly. With `k = s_p/s_y` the spread ratio and `b = (p_bar -
y_bar)/s_y` the standardised bias,

    R2 = r^2 - (r - k)^2 - b^2

so `r^2` is a hard ceiling reached only at `k = r, b = 0`. Everything below the ceiling is
recoverable **by an affine map alone**, which is monotone and therefore cannot disturb rho or
tau. This script measures, per isoform:

* the ceiling `r^2` implied by our own reported rho,
* where our R2 actually sits,
* the two components of the shortfall, spread and bias, separately,
* and the same for the best entries, to check the ceiling is real rather than an artefact of
  the Gaussian-copula step `r = 2 sin(pi rho / 6)`.

Do it per isoform, never on the macro row: the macro columns average rho and R2 separately,
and by Jensen `mean(r^2) >= (mean r)^2`, so a macro R2 legitimately exceeds the r^2 computed
from a macro rho. Reading that as a violated bound is a mistake this script exists to avoid.

    python scripts/cyp_murphy_gap.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


DUMP = Path("/private/tmp/claude-501/-Users-arnab-Documents-talindrew"
            "/c617177f-c1cb-4037-80b1-ab3bb279b316/scratchpad/board_oct9")
ISO_TABLE = {"CYP1A2": "lb_61.csv", "CYP2C9": "lb_64.csv",
             "CYP2D6": "lb_67.csv", "CYP3A4": "lb_70.csv"}
SUBMITTED = Path("data/cyp_cache/final_v4screen/submission_v4.csv")
V5 = Path("data/cyp_cache/final_v4screen/submission_v5.csv")


def pearson_from_spearman(rho: float) -> float:
    """Gaussian-copula inversion. An approximation -- treated as such below."""
    return 2.0 * math.sin(math.pi * rho / 6.0)


def read_iso(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open()):
        try:
            out.append(dict(user=r["Username"], rank=int(r["Rank"]),
                            strae=float(r["ST-RAE"]), mae=float(r["MAE"]),
                            r2=float(r["R²"]), rho=float(r["Spearman's ρ"]),
                            tau=float(r["Kendall's τ"])))
        except (ValueError, KeyError):
            continue
    return [x for x in out if all(v == v for v in
                                  (x["strae"], x["r2"], x["rho"]))]


def main() -> int:
    print("Per isoform. 'ceiling' is r^2 with r = 2 sin(pi rho / 6) from our OWN rho;")
    print("'gap' is ceiling - our R2, i.e. what an affine map could recover at fixed ranking.\n")
    print(f"{'iso':8} {'our rho':>8} {'r':>7} {'ceiling':>8} {'our R2':>8} {'gap':>8} "
          f"{'best R2':>8} {'that rho':>9} {'their ceil':>11}")
    print("-" * 82)

    total_gap, rows_for_sd = 0.0, {}
    for iso, f in ISO_TABLE.items():
        rows = read_iso(DUMP / f)
        ours = next(x for x in rows if x["user"] == "talindrew")
        r = pearson_from_spearman(ours["rho"])
        ceil = r * r
        gap = ceil - ours["r2"]
        total_gap += gap
        best = max(rows, key=lambda x: x["r2"])
        br = pearson_from_spearman(best["rho"])
        rows_for_sd[iso] = (ours, rows)
        print(f"{iso:8} {ours['rho']:8.4f} {r:7.4f} {ceil:8.4f} {ours['r2']:8.4f} "
              f"{gap:+8.4f} {best['r2']:8.4f} {best['rho']:9.4f} {br*br:11.4f}")

    print(f"\nmacro recoverable R2 at fixed ranking: {total_gap/4:+.4f}")

    # Split the gap into its two Murphy components, using the spread of our own file
    # against the test sd recovered earlier from the board row.
    sy = {"CYP1A2": 1.439, "CYP2C9": 0.966, "CYP2D6": 1.3253, "CYP3A4": 1.160}
    print("\nSplitting the shortfall: R2 = r^2 - (r - k)^2 - b^2,  k = s_p/s_y,  b = bias/s_y")
    print(f"{'iso':8} {'s_p (v4)':>9} {'s_y':>7} {'k':>7} {'r':>7} {'spread loss':>12} "
          f"{'bias loss':>10} {'s_p for k=r':>12} {'v5 s_p':>8}")
    print("-" * 88)
    import pandas as pd
    if not SUBMITTED.exists():
        print(f"  (no {SUBMITTED}; spread split skipped)")
        return 0
    v4 = pd.read_csv(SUBMITTED)
    v5 = pd.read_csv(V5) if V5.exists() else None
    for iso in ISO_TABLE:
        col = f"{iso}_pIC50_direct_inhibition"
        ours, _ = rows_for_sd[iso]
        sp = float(v4[col].std(ddof=0))
        sp5 = float(v5[col].std(ddof=0)) if v5 is not None else float("nan")
        r = pearson_from_spearman(ours["rho"])
        k = sp / sy[iso]
        spread_loss = (r - k) ** 2
        bias_loss = max(0.0, r * r - spread_loss - ours["r2"])
        print(f"{iso:8} {sp:9.4f} {sy[iso]:7.4f} {k:7.4f} {r:7.4f} {spread_loss:12.4f} "
              f"{bias_loss:10.4f} {r*sy[iso]:12.4f} {sp5:8.4f}")

    print("\nRead the last two columns together: 's_p for k=r' is the spread that zeroes the")
    print("spread term, and 'v5 s_p' is what the pending v5 submission actually ships.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
