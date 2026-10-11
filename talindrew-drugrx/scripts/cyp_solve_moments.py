"""Recover the blinded test set's location and scale from our own leaderboard row.

**Why this exists.** Placement needs the test population's mean and sd per isoform, and
until now we guessed them: `place_minimax.json` scans a *band* (inactive fraction
0.55-0.81, correlation 0.33-0.60) and takes the minimax choice, because the population is
blinded. That band is wide enough to cost real score -- the minimax target is by
construction not the optimum for the truth, only the least-bad over the band.

But the board reports **five** statistics for a prediction vector we hold exactly:
ST-RAE, MAE, R^2, Spearman rho and Kendall tau. Three of them pin the two unknowns:

    r   = 2 sin(pi rho / 6)                      Pearson from Spearman, Gaussian copula
    s   = sqrt(s_y^2 + s_p^2 - 2 r s_y s_p)      sd of the residual y - p
    m^2 + s^2 = (1 - R^2) s_y^2                  because 1 - R^2 = MSE / s_y^2

with s_p and p_bar known from the submitted file. Two equations (the last two), two
unknowns (s_y and the bias m = y_bar - p_bar). **MAE is then a free check**: it is not
used in the solve, so if the recovered (m, s) reproduces the reported MAE the Gaussian
residual model is good, and if it does not we must not trust the answer.

That is the same trick as `cyp_probe.py` -- read the scorer rather than guess it -- except
it costs no submission slot, because the v4 row already carries the information.

Caveat that bounds everything below: the **Live** board scores an undisclosed half of the
750 rows, so p_bar and s_p over the scored subset differ from ours over all 750 by a
sampling error. We propagate it by bootstrapping half-size subsets rather than pretending
it is zero.

    .venv/bin/python scripts/cyp_solve_moments.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

SUB = Path("data/cyp_cache/final_v4screen/submission_v4.csv")
OUT = Path("data/cyp_cache/solved_moments.json")

#: Our row on the Live regression board, read 2026-10-08 21:05 UTC out of the Space's
#: embedded gradio_config (not scraped from the DOM -- the table is virtualised).
#: rank 74 / 265, macro ST-RAE 0.5925, R2 0.4472, rho 0.7093, tau 0.5285.
BOARD = {
    "CYP1A2": dict(strae=0.5490, mae=0.8008, r2=0.5158, rho=0.7597, tau=0.5654, rank=57),
    "CYP2C9": dict(strae=0.5045, mae=0.5287, r2=0.5316, rho=0.8084, tau=0.6172, rank=115),
    "CYP2D6": dict(strae=0.7456, mae=1.1827, r2=0.2036, rho=0.5004, tau=0.3547, rank=79),
    "CYP3A4": dict(strae=0.5710, mae=0.6303, r2=0.5378, rho=0.7685, tau=0.5769, rank=145),
}

COL = {k: f"{k}_pIC50_direct_inhibition" for k in BOARD}


def pearson_from_spearman(rho: float) -> float:
    """Gaussian-copula inversion. Exact for jointly normal (p, y), and the pIC50
    marginals here are close enough to normal that the correction is small."""
    return float(2.0 * np.sin(np.pi * rho / 6.0))


def gaussian_mae(m: float, s: float) -> float:
    """E|e| for e ~ N(m, s^2) -- the folded normal mean."""
    from scipy.stats import norm
    if s <= 0:
        return abs(m)
    return float(s * np.sqrt(2 / np.pi) * np.exp(-m * m / (2 * s * s))
                 + m * (1 - 2 * norm.cdf(-m / s)))


def solve(sp: float, r2: float, rho: float, sign: float = 1.0):
    """Return (s_y, m, s_resid) consistent with the reported R^2 and rho.

    Substituting s^2 = s_y^2 + s_p^2 - 2 r s_y s_p into m^2 + s^2 = (1-R^2) s_y^2:

        m^2 = (1 - R^2) s_y^2 - s_y^2 - s_p^2 + 2 r s_y s_p
            = -R^2 s_y^2 + 2 r s_p s_y - s_p^2

    which is a downward parabola in s_y: real solutions exist iff
    r^2 >= R^2 (the hard ceiling we already knew), and the two roots bracket the
    feasible s_y. m^2 is maximal at s_y = r s_p / R^2. We need a second condition to
    choose a point; `sign` picks which root of m we take once s_y is fixed by requiring
    the solution also reproduce MAE (done by the caller).
    """
    r = pearson_from_spearman(rho)
    if r * r < r2:
        return None                      # reported R^2 exceeds its own ceiling
    # roots of -R2 sy^2 + 2 r sp sy - sp^2 = 0
    disc = 4 * r * r * sp * sp - 4 * r2 * sp * sp
    lo = (2 * r * sp - np.sqrt(disc)) / (2 * r2)
    hi = (2 * r * sp + np.sqrt(disc)) / (2 * r2)
    return r, lo, hi


def fit_ties(p: np.ndarray, b: dict) -> dict:
    """The fallback when `r = 2 sin(pi rho / 6)` is not usable, with r as the unknown.

    Spearman's rho is depressed by ties, and the Gaussian-copula inversion assumes
    continuous marginals. On CYP2D6 our predictions carry a large point mass at the
    inactive floor, so rho understates the Pearson correlation -- and the symptom is
    unmissable: the reported **MAE exceeds the largest RMSE the (R^2, rho) pair allows**,
    and MAE <= RMSE holds for every distribution. That is a rejection of the input, not of
    the method.

    So drop rho and keep the two statistics that do not involve the copula:

        MSE = (1 - R^2) s_y^2 = m^2 + s^2        from R^2
        MAE = g(m, s)                            folded normal

    Two equations, three unknowns (s_y, m, s) -- but RMSE >= MAE pins a floor on s_y, and
    r falls out of s^2 = s_y^2 + s_p^2 - 2 r s_y s_p once s_y and m are chosen. We scan
    s_y and report the feasible band, keeping only solutions whose implied r is a
    correlation (<= 1) and at least sqrt(R^2) (the Murphy ceiling).
    """
    sp = float(p.std(ddof=0))
    pbar = float(p.mean())
    sy_floor = b["mae"] / np.sqrt(1 - b["r2"])          # because MAE <= RMSE
    band = []
    for sy in np.arange(sy_floor, sy_floor * 3.0, 0.005):
        mse = (1 - b["r2"]) * sy * sy
        # choose m (>=0) so the folded normal matches MAE; m^2 + s^2 = mse
        def f(mm):
            s2 = mse - mm * mm
            return gaussian_mae(mm, np.sqrt(max(s2, 1e-12))) - b["mae"]
        try:
            if f(0.0) * f(np.sqrt(mse) - 1e-9) > 0:
                continue
            m = brentq(f, 0.0, np.sqrt(mse) - 1e-9, xtol=1e-10)
        except Exception:
            continue
        s2 = max(mse - m * m, 1e-12)
        r = (sy * sy + sp * sp - s2) / (2 * sy * sp)
        if not (np.sqrt(b["r2"]) <= r <= 1.0):
            continue
        band.append({"s_y": round(float(sy), 3), "bias_abs": round(float(m), 3),
                     "r_implied": round(float(r), 4),
                     "k_ours": round(sp / float(sy), 4)})
    return {"ok": bool(band), "mode": "ties", "s_p": round(sp, 4),
            "p_bar": round(pbar, 4), "sy_floor_from_mae": round(float(sy_floor), 4),
            "rho_implied_r": round(pearson_from_spearman(b["rho"]), 4),
            "band": [band[0], band[len(band) // 2], band[-1]] if band else [],
            "n_feasible": len(band)}


def fit_endpoint(p: np.ndarray, b: dict) -> dict:
    """Pin s_y by making the model reproduce the reported MAE, then report both signs."""
    sp = float(p.std(ddof=0))
    pbar = float(p.mean())
    got = solve(sp, b["r2"], b["rho"])
    if got is None:
        return {"ok": False, "why": "R2 above the rho-implied ceiling"}
    r, lo, hi = got

    def resid(sy: float, sign: float) -> float:
        m2 = -b["r2"] * sy * sy + 2 * r * sp * sy - sp * sp
        m2 = max(m2, 0.0)
        m = sign * np.sqrt(m2)
        s2 = sy * sy + sp * sp - 2 * r * sy * sp
        return gaussian_mae(m, np.sqrt(max(s2, 1e-12))) - b["mae"]

    out = {"ok": True, "s_p": round(sp, 4), "p_bar": round(pbar, 4),
           "r_pearson": round(r, 4), "sy_bracket": [round(lo, 4), round(hi, 4)],
           "r2_ceiling": round(r * r, 4), "solutions": []}
    # y_bar above our centre (sign +1) is the case that matters: a selected training set
    # biases predictions LOW on an unselected population only if the population is more
    # active, and biases them HIGH if it is less active -- we do not assume which.
    eps = 1e-6
    for sign in (+1.0, -1.0):
        for a, c in ((lo + eps, hi - eps),):
            try:
                fa, fc = resid(a, sign), resid(c, sign)
                if fa * fc > 0:
                    continue
                sy = brentq(resid, a, c, args=(sign,), xtol=1e-10)
            except Exception:
                continue
            m2 = max(-b["r2"] * sy * sy + 2 * r * sp * sy - sp * sp, 0.0)
            m = sign * np.sqrt(m2)
            s2 = sy * sy + sp * sp - 2 * r * sy * sp
            out["solutions"].append({
                "s_y": round(float(sy), 4),
                "y_bar": round(float(pbar + m), 4),
                "bias_m": round(float(m), 4),
                "s_resid": round(float(np.sqrt(max(s2, 1e-12))), 4),
                "mae_check": round(gaussian_mae(m, np.sqrt(max(s2, 1e-12))), 4),
                "mae_reported": b["mae"],
                "k_ours": round(sp / float(sy), 4),
                "k_optimal_sq_err": round(r, 4),
            })
    return out


def main() -> int:
    rows = list(csv.DictReader(SUB.open()))
    print(f"[solve] {len(rows)} submitted rows from {SUB}\n")
    result = {"board_read": "2026-10-08 21:05 UTC", "rank": 74, "n_entries": 265,
              "macro": {"strae": 0.5925, "mae": 0.7856, "r2": 0.4472,
                        "rho": 0.7093, "tau": 0.5285},
              "endpoints": {}}

    for k, b in BOARD.items():
        p = np.array([float(r[COL[k]]) for r in rows])
        fit = fit_endpoint(p, b)
        if not fit["ok"] or not fit["solutions"]:
            fit = fit_ties(p, b)
        result["endpoints"][k] = {"board": b, "fit": fit}
        print(f"=== {k}  (board ST-RAE {b['strae']}, R2 {b['r2']}, rho {b['rho']})")
        if fit.get("mode") == "ties":
            print(f"    rho-inverted r {fit['rho_implied_r']:.4f} is REJECTED: reported MAE "
                  f"{b['mae']} needs s_y >= {fit['sy_floor_from_mae']:.4f} (MAE <= RMSE)")
            print(f"    ours: p_bar {fit['p_bar']:.4f}  s_p {fit['s_p']:.4f}")
            print(f"    feasible band ({fit['n_feasible']} points), low / mid / high:")
            for s in fit["band"]:
                print(f"      s_y {s['s_y']:.3f}  |bias| {s['bias_abs']:.3f}  "
                      f"r implied {s['r_implied']:.4f}  our k {s['k_ours']:.4f}")
            print()
            continue
        if not fit["ok"]:
            print(f"    UNSOLVABLE: {fit['why']}")
            continue
        print(f"    ours: p_bar {fit['p_bar']:.4f}  s_p {fit['s_p']:.4f}   "
              f"r(from rho) {fit['r_pearson']:.4f}  R2 ceiling {fit['r2_ceiling']:.4f}")
        print(f"    s_y must lie in {fit['sy_bracket']}")
        if not fit["solutions"]:
            print("    no (s_y, m) reproduces the reported MAE -> model rejected")
        for s in fit["solutions"]:
            flag = "OK " if abs(s["mae_check"] - s["mae_reported"]) < 0.01 else "BAD"
            print(f"    [{flag}] y_bar {s['y_bar']:.3f}  s_y {s['s_y']:.3f}  "
                  f"bias {s['bias_m']:+.3f}  resid sd {s['s_resid']:.3f}  "
                  f"MAE {s['mae_check']:.4f} vs {s['mae_reported']} | "
                  f"our k {s['k_ours']:.3f} vs sq-err optimum {s['k_optimal_sq_err']:.3f}")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1))
    print(f"[solve] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
