"""Where does our deficit against the field actually live? Per isoform, in sigma* units.

A macro rho of 0.7093 against leaders near 0.85 reads as one uniform gap. It is not necessarily
one: the macro is an average over four isoforms whose individual rhos range from 0.50 to 0.81,
and a deficit concentrated in one cell is a completely different research problem from a deficit
spread evenly over four.

This matters because the literature says the whole model-class ladder -- tuned GBM to 2026 state
of the art -- is worth 4-8% of absolute error on tasks of this shape. If our deficit is uniform,
nothing in that ladder closes it. If it is concentrated, the concentrated cell may have a
specific, diagnosable cause that is not a model-class problem at all.

    python scripts/cyp_where_is_the_gap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyp_board_fetch import boards, fetch  # noqa: E402

CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
US = "talindrew"
#: Blinded-truth sd per isoform, recovered by the moment inversion (cyp_solve_moments.py).
S_Y = {"CYP1A2": 1.439, "CYP2C9": 0.966, "CYP2D6": 1.660, "CYP3A4": 1.160}


def r_from_rho(rho: float) -> float:
    """Pearson r from Spearman rho under bivariate normality (no ties)."""
    return float(2.0 * np.sin(np.pi * rho / 6.0))


def sigma_star(rho: float, s_y: float) -> float:
    r = r_from_rho(rho)
    return float(s_y * np.sqrt(max(0.0, 1.0 - r * r)))


def main() -> int:
    bs = boards(fetch())
    print(f"{'isoform':9} {'ours rho':>9} {'best rho':>9} {'p90 rho':>8} | "
          f"{'our sig*':>9} {'best sig*':>10} {'CUT NEEDED':>11} {'our rank':>9}")
    print("-" * 92)
    rows = {}
    for iso in CYPS:
        key = next((k for k in bs if "Live" in k and "Regression" in k and iso in k), None)
        if key is None:
            continue
        b = bs[key]
        hdr, data = b["headers"], b["data"]
        i_rho = next(i for i, h in enumerate(hdr) if "Spearman" in h)
        i_user = hdr.index("Username")
        i_rank = hdr.index("Rank")
        rhos, ours, our_rank = [], None, None
        for r in data:
            try:
                v = float(r[i_rho])
            except (TypeError, ValueError):
                continue
            rhos.append(v)
            if US in str(r[i_user]):
                ours, our_rank = v, r[i_rank]
        if ours is None or not rhos:
            continue
        rhos = np.array(rhos)
        best, p90 = float(rhos.max()), float(np.quantile(rhos, 0.90))
        s = S_Y[iso]
        so, sb = sigma_star(ours, s), sigma_star(best, s)
        cut = 1.0 - sb / so
        rows[iso] = dict(ours=ours, best=best, p90=p90, sig_ours=so, sig_best=sb, cut=cut)
        print(f"{iso:9} {ours:9.4f} {best:9.4f} {p90:8.4f} | {so:9.3f} {sb:10.3f} "
              f"{100*cut:10.1f}% {str(our_rank):>9}")

    print("-" * 92)
    if rows:
        macro_cut = float(np.mean([v["cut"] for v in rows.values()]))
        tot_ours = float(np.mean([v["sig_ours"] for v in rows.values()]))
        tot_best = float(np.mean([v["sig_best"] for v in rows.values()]))
        print(f"{'MEAN':9} {'':9} {'':9} {'':8} | {tot_ours:9.3f} {tot_best:10.3f} "
              f"{100*macro_cut:10.1f}%")
        worst = max(rows, key=lambda k: rows[k]["cut"])
        best_cell = min(rows, key=lambda k: rows[k]["cut"])
        print(f"\n  widest gap:    {worst}  ({100*rows[worst]['cut']:.1f}% cut needed)")
        print(f"  narrowest gap: {best_cell}  ({100*rows[best_cell]['cut']:.1f}% cut needed)")
        spread = rows[worst]["cut"] - rows[best_cell]["cut"]
        print(f"  spread across isoforms: {100*spread:.1f} percentage points")
        print("\n  A uniform deficit and a concentrated one call for different work. If the")
        print("  spread is large, the macro number has been hiding the actual problem, and the")
        print("  worst cell deserves a diagnosis of its own rather than a better model class.")
        # Against the realistic model-class ceiling rather than against the single best entrant.
        print("\n  Against the measured model-class ceiling (4-8% of absolute error), how much")
        print("  of each gap could the BEST AVAILABLE MODEL CLASS close?")
        for iso, v in rows.items():
            closes_lo, closes_hi = 0.04 / v["cut"], 0.08 / v["cut"]
            print(f"    {iso:9} needs {100*v['cut']:5.1f}%  ->  model class covers "
                  f"{100*min(closes_lo,1):4.0f}-{100*min(closes_hi,1):3.0f}% of it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
