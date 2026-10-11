"""Fit the learning curve WITH a plateau term, and ask whether the gap is reachable at any n.

`cyp_data_scaling.py` measures sigma* against training size. This fits it, and the functional
form matters more than the fit quality does.

`research/28-does-compute-help.md` already settled which form to use, citing Rosenfeld et al.,
"A Constructive Prediction of the Generalization Error Across Scales" (ICLR 2020,
arXiv:1909.12673), which fits generalisation error across six benchmarks as

    eps(m, n) = a * n^-alpha + b * m^-beta + c

The **irreducible term `c`** is the whole point. This programme has already measured that the
model-size direction is on its plateau (capacity +0.0019 at p = 0.723, width +0.0000), so `b*m^-beta`
is spent; what was never measured is the `n` direction. A **pure** power law `a*n^-alpha` has no
plateau and therefore *cannot* express "more data stops helping" -- it will always promise that
some finite n reaches any target. Fitting the plateau is what makes the question falsifiable:

    if the fitted c exceeds the sigma* the field's best requires, then NO amount of data closes
    the gap with this model class, and the lever is model class, not data.

So this reports both fits, and the decision rests on `c`.

### Honesty about identifiability

`alpha` and `c` trade off badly over a 6.7x range of n: a curve that is flattening can be fitted
either as a small alpha with no floor or a large alpha with a high floor. The script therefore
bootstraps over **seeds** (not points -- points within a seed share a split and a nested
subsample, so they are not independent) and reports the interval on `c`. A `c` whose interval
spans the target is an honest "cannot tell from this range", not a verdict.

    python scripts/cyp_scaling_fit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cyp_cache"
BOOT = 2000

#: sigma* each isoform needs to match the field's best, from cyp_lift_decompose.py. These are in
#: BOARD s_y units; the learning curve is in VALIDATION s_y units, so they are rescaled below by
#: the measured s_y ratio rather than compared naively.
NEEDED_BOARD = {"CYP1A2": 0.718, "CYP2C9": 0.440, "CYP3A4": 0.543}


def short(c: str) -> str:
    return c.split("_")[0]


def fit_plateau(n: np.ndarray, s: np.ndarray) -> tuple[float, float, float, float]:
    """Least squares on sigma = a n^-alpha + c. Returns (a, alpha, c, R2)."""
    from scipy.optimize import curve_fit

    def f(x, a, alpha, c):
        return a * np.power(x, -alpha) + c

    # c is bounded below by 0 and above by the smallest observed sigma: a floor cannot exceed
    # a value the model already achieved.
    try:
        p, _ = curve_fit(f, n, s, p0=(1.0, 0.15, 0.5 * s.min()),
                         bounds=([0.0, 0.0, 0.0], [50.0, 2.0, s.min()]), maxfev=40000)
    except Exception:
        return (np.nan,) * 4
    pred = f(n, *p)
    r2 = 1.0 - np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2)
    return float(p[0]), float(p[1]), float(p[2]), float(r2)


def fit_power(n: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    """Pure power law in logs: log sigma = log a - alpha log n. Returns (alpha, R2)."""
    ln, ls = np.log(n), np.log(s)
    A = np.vstack([np.ones_like(ln), -ln]).T
    coef, *_ = np.linalg.lstsq(A, ls, rcond=None)
    pred = A @ coef
    return float(coef[1]), float(1.0 - np.sum((ls - pred) ** 2) / np.sum((ls - ls.mean()) ** 2))


def main() -> int:
    rows = json.loads((CACHE / "data_scaling.json").read_text())
    lift = {r["iso"]: r for r in json.loads((CACHE / "lift_decompose.json").read_text())}
    isos = [c for c in sorted({r["iso"] for r in rows})]
    seeds = sorted({r["seed"] for r in rows})
    rng = np.random.default_rng(0)
    print(f"[fit] {len(rows)} points, {len(seeds)} seeds, {len(isos)} isoforms\n")

    print("=== both fits, and what the plateau implies ===")
    print(f"{'iso':8} {'pure alpha':>11} {'R2':>5} | {'a':>7} {'alpha':>7} {'c (floor)':>10} "
          f"{'R2':>5} | {'needed sig*':>12} {'reachable?':>26}")
    for c in isos:
        sh = short(c)
        rs = [r for r in rows if r["iso"] == c and np.isfinite(r["sigma"]) and r["n_drc"] > 0]
        if len(rs) < 8:
            continue
        n = np.array([r["n_drc"] for r in rs], float)
        s = np.array([r["sigma"] for r in rs], float)
        a_p, r2_p = fit_power(n, s)
        a, alpha, c0, r2 = fit_plateau(n, s)

        if sh not in NEEDED_BOARD:
            print(f"{sh:8} {a_p:11.4f} {r2_p:5.2f} | {a:7.3f} {alpha:7.4f} {c0:10.3f} {r2:5.2f} "
                  f"| {'n/a':>12} {'excluded (see lift decomp)':>26}")
            continue

        # Rescale the board-units target into validation units using the measured **sigma***
        # ratio, NOT the s_y ratio. sigma* is an absolute error, and cyp_lift_decompose.py
        # measured that it TRANSFERS (board/validation 1.07 / 1.00 / 1.10). Dividing by the s_y
        # ratio (1.38) instead would hold r fixed, which is the wrong invariance: it is sigma*
        # that carries across the two sets, and r is what differs because s_y differs.
        ratio = lift[sh]["sig_board"] / lift[sh]["sig_val"]
        need_val = NEEDED_BOARD[sh] / ratio

        # bootstrap the floor over SEEDS
        floors = []
        for _ in range(BOOT):
            pick = rng.choice(seeds, size=len(seeds), replace=True)
            sel = [r for sd in pick for r in rs if r["seed"] == sd]
            nn = np.array([r["n_drc"] for r in sel], float)
            ss = np.array([r["sigma"] for r in sel], float)
            _, _, cb, _ = fit_plateau(nn, ss)
            if np.isfinite(cb):
                floors.append(cb)
        lo, hi = (np.percentile(floors, [2.5, 97.5]) if floors else (np.nan, np.nan))

        if not np.isfinite(c0):
            verdict = "fit failed"
        elif lo > need_val:
            # the WHOLE interval above the target, not merely the point estimate -- a point
            # estimate above a target whose CI spans it is not a verdict, it is noise
            verdict = "NO -- floor CI above target"
        elif hi > need_val:
            verdict = "cannot tell (CI spans it)"
        else:
            nn_need = (a / max(need_val - c0, 1e-9)) ** (1.0 / alpha) if alpha > 0 else np.inf
            verdict = f"yes, at {nn_need / n.max():.0f}x data"
        print(f"{sh:8} {a_p:11.4f} {r2_p:5.2f} | {a:7.3f} {alpha:7.4f} {c0:10.3f} {r2:5.2f} "
              f"| {need_val:12.3f} {verdict:>26}")
        print(f"{'':8} {'':11} {'':5} | floor 95% CI over seeds [{lo:.3f}, {hi:.3f}]  "
              f"target in validation units {need_val:.3f}  (board {NEEDED_BOARD[sh]:.3f} / "
              f"sigma* ratio {ratio:.2f})")

    # The per-point log-log fit runs on all 30 (seed x fraction) points, where seed-to-seed
    # scatter swamps a shallow n-trend -- hence R2 0.03-0.19 on three isoforms. Averaging over
    # seeds at each fraction removes that scatter and leaves the trend, which is the better
    # estimator of alpha even though it discards the within-fraction spread.
    print("\n=== alpha on the SEED-AVERAGED curve (6 points, scatter removed) ===")
    print(f"{'iso':8} {'alpha':>8} {'R2':>6} {'sig* 15%':>9} {'sig* 100%':>10} {'total drop':>11} "
          f"{'cut needed':>11} {'n factor':>10}")
    fracs = sorted({r["frac"] for r in rows})
    for c in isos:
        sh = short(c)
        pts = []
        for f in fracs:
            sel = [r for r in rows if r["iso"] == c and r["frac"] == f and np.isfinite(r["sigma"])]
            if sel:
                pts.append((float(np.mean([r["n_drc"] for r in sel])),
                            float(np.mean([r["sigma"] for r in sel]))))
        if len(pts) < 4:
            continue
        n = np.array([p[0] for p in pts]); s = np.array([p[1] for p in pts])
        alpha, r2 = fit_power(n, s)
        drop = 1.0 - s[-1] / s[0]
        if sh in NEEDED_BOARD:
            ratio = lift[sh]["sig_board"] / lift[sh]["sig_val"]
            need_val = NEEDED_BOARD[sh] / ratio
            cut = 1.0 - need_val / s[-1]
            fac = (1.0 - cut) ** (-1.0 / alpha) if alpha > 0 and cut < 1 else np.inf
            cut_s, fac_s = f"{cut*100:10.0f}%", (f"{fac:9.0f}x" if fac < 1e6 else "   >1e6x")
        else:
            cut_s, fac_s = f"{'n/a':>11}", f"{'n/a':>10}"
        print(f"{sh:8} {alpha:8.4f} {r2:6.2f} {s[0]:9.3f} {s[-1]:10.3f} {drop*100:10.1f}% "
              f"{cut_s} {fac_s}")
    print("  Typical ML scaling exponents are 0.1-0.5. Every isoform here is 0.02-0.08, so the")
    print("  returns to more data of the same kind are several times shallower than normal.")

    print("\n=== reading this ===")
    print("  The pure power law has no plateau, so it always promises some finite n reaches any")
    print("  target -- it cannot answer the question and is shown only for comparison.")
    print("  The decision is `c`: a floor above the target means more data of the same kind")
    print("  cannot close the gap with this model class, whatever the quantity.")
    print("  A floor CI that spans the target is an honest 'not resolvable over a 6.7x range of n'")
    print("  and calls for wider n, not for a verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
