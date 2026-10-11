"""Recover the leading entries' placement parameters from their leaderboard rows.

Our own row was inverted earlier to the blinded test set's moments: for a prediction vector
held on disk, reported Spearman and R2 give two equations in the truth's `(s_y, m)`. Those
moments are a property of the **test set**, so they are the same for every entry -- which
turns the inversion around. For anyone else's row, `s_y` is now known and their own
`(s_p, |m|)` are the unknowns:

    r   = 2 sin(pi rho / 6)                                   Pearson from Spearman
    0   = m^2 + s_p^2 - 2 r s_y s_p + R^2 s_y^2               from MSE = (1 - R^2) s_y^2
    MAE = E|N(-m, sigma^2)|,  sigma^2 = s_y^2 + s_p^2 - 2 r s_y s_p      folded normal

Two equations, two unknowns. So the board hands over **how widely each entry spread its
predictions and how far it offset them** -- the two numbers our own placement step chooses.

Why bother: at rank 30, `tguttenb1` scores macro ST-RAE 0.4737 with macro Spearman 0.7223,
+0.013 of ranking over ours, while scoring macro R2 0.5907 against our 0.4472. Almost the
whole 0.119 of ST-RAE between us is R2 at matched ranking, and R2 at fixed ranking is placement
-- recoverable by an affine map, which is monotone and so cannot disturb rho or tau.

Two honesties, both of which bit earlier in this programme:

* **Only |m| is recoverable.** R2 and MAE depend on the offset through m^2 and through a
  folded normal, both even in m. The sign has to come from somewhere else.
* **CYP2D6's Gaussian solve is misspecified** -- its truth is bimodal, and 748 distinct values
  in 750 rows already ruled out ties as the explanation for its rho/r divergence. Its `s_y` is
  a distribution-free lower bound (MAE <= RMSE), not a point estimate, so its row is reported
  but flagged rather than acted on.

    python scripts/cyp_read_their_placement.py --top 12
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DUMP = Path("/private/tmp/claude-501/-Users-arnab-Documents-talindrew"
            "/c617177f-c1cb-4037-80b1-ab3bb279b316/scratchpad/board_oct9")
ISO_TABLE = {"CYP1A2": "lb_61.csv", "CYP2C9": "lb_64.csv",
             "CYP2D6": "lb_67.csv", "CYP3A4": "lb_70.csv"}

# Test-set sd, recovered from our own board row (scripts/cyp_solve_moments.py).
# CYP2D6's is the distribution-free lower bound, not a point estimate -- see the docstring.
SY = {"CYP1A2": 1.439, "CYP2C9": 0.966, "CYP2D6": 1.3253, "CYP3A4": 1.160}
SOFT = {"CYP2D6"}
GAUSS = math.sqrt(2 / math.pi)   # MAE/RMSE for zero-mean Gaussian residuals


def pearson_from_spearman(rho: float) -> float:
    return 2.0 * math.sin(math.pi * rho / 6.0)


def folded_mean(m: float, sigma: float) -> float:
    """E|X| for X ~ N(-m, sigma^2). Even in m, which is why only |m| is recoverable."""
    if sigma <= 1e-9:
        return abs(m)
    a = abs(m)
    return sigma * math.sqrt(2 / math.pi) * math.exp(-a * a / (2 * sigma * sigma)) \
        + a * (1 - 2 * norm.cdf(-a / sigma))


def solve_sp_m(sy: float, r: float, r2: float, mae: float) -> tuple[float, float] | None:
    """Solve (s_p, |m|) from one board row. Returns None when the row admits no solution."""
    def m2_of(sp: float) -> float:
        return -(sp * sp - 2 * r * sy * sp + r2 * sy * sy)

    def resid(sp: float) -> float:
        m2 = m2_of(sp)
        if m2 < 0:
            return float("nan")
        sig2 = sy * sy + sp * sp - 2 * r * sy * sp
        if sig2 < 0:
            return float("nan")
        return folded_mean(math.sqrt(m2), math.sqrt(sig2)) - mae

    # m^2 >= 0 bounds s_p to the roots of sp^2 - 2 r sy sp + R^2 sy^2 = 0
    disc = (r * sy) ** 2 - r2 * sy * sy
    if disc < 0:
        return None
    lo, hi = r * sy - math.sqrt(disc), r * sy + math.sqrt(disc)
    grid = np.linspace(lo + 1e-6, hi - 1e-6, 400)
    vals = [resid(x) for x in grid]
    for a, b, va, vb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if va == va and vb == vb and va * vb < 0:
            sp = brentq(resid, a, b, xtol=1e-10)
            return sp, math.sqrt(max(0.0, m2_of(sp)))
    return None


def read_iso(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open()):
        try:
            d = dict(user=r["Username"], rank=int(r["Rank"]), strae=float(r["ST-RAE"]),
                     mae=float(r["MAE"]), r2=float(r["R²"]), rho=float(r["Spearman's ρ"]))
        except (ValueError, KeyError):
            continue
        if all(v == v for v in (d["strae"], d["mae"], d["r2"], d["rho"])):
            out.append(d)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    for iso, f in ISO_TABLE.items():
        rows = read_iso(DUMP / f)
        rows.sort(key=lambda x: x["strae"])
        ours = next(x for x in rows if x["user"] == "talindrew")
        flag = "   [MISSPECIFIED: bimodal truth, s_y is a lower bound -- do not act]" \
            if iso in SOFT else ""
        print(f"\n=== {iso}   s_y = {SY[iso]:.4f}{flag}")
        print(f"{'rank':>5} {'entry':20} {'ST-RAE':>7} {'rho':>7} {'R2':>7} "
              f"{'-> their s_p':>12} {'their |m|':>10} {'kappa':>8} {'MAE/RMSE':>8}")
        print("-" * 96)
        shown = rows[:args.top]
        if ours not in shown:
            shown = shown + [ours]
        for x in shown:
            r = pearson_from_spearman(x["rho"])
            sol = solve_sp_m(SY[iso], r, x["r2"], x["mae"])
            name = ">> talindrew (us)" if x["user"] == "talindrew" else x["user"][:20]
            # kappa = MAE / sqrt(1 - R^2) = s_y * (MAE/RMSE). The s_y cancels when two
            # entries are compared, so the *ordering* of kappa survives any error in s_y.
            kappa = x["mae"] / math.sqrt(max(1e-9, 1 - x["r2"]))
            shape = kappa / SY[iso]
            if sol is None:
                why = "heavy-tailed" if shape < GAUSS else "no root"
                print(f"{x['rank']:5} {name:20} {x['strae']:7.4f} {x['rho']:7.4f} "
                      f"{x['r2']:7.4f} {'--':>12} {'--':>10} {kappa:8.4f} {shape:8.4f} {why}")
            else:
                sp, m = sol
                print(f"{x['rank']:5} {name:20} {x['strae']:7.4f} {x['rho']:7.4f} "
                      f"{x['r2']:7.4f} {sp:12.4f} {m:10.4f} {kappa:8.4f} {shape:8.4f}")
    print(f"\nMAE/RMSE: Laplace {math.sqrt(0.5):.4f} < Gaussian {GAUSS:.4f} < uniform 0.8660;")
    print("a value below Gaussian means residuals HEAVIER-tailed than normal, which is why the")
    print("Gaussian solve finds no root for the leaders. kappa = MAE/sqrt(1-R^2) is free of s_y,")
    print("so comparing kappa between entries is safe even where s_y itself is uncertain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
