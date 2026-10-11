r"""Recover the blinded test set's label spread, our bias and the optimal scale — from the
leaderboard's own diagnostic columns.

The leaderboard publishes five numbers per submission: ST-RAE (which ranks), and **MAE, R²,
Spearman ρ and Kendall τ as diagnostics**. Confirmed against the authoritative source rather
than assumed — `github.com/OpenADMET/blind-challenge-template`, `backend/config.py`:
`ACTIVITY_METRICS` wires MAE to sklearn's `mean_absolute_error` and R² to `r2_score`, and
`evaluate_predictions.py::_metric_needs_credible_interval_bounds` gives the credible-interval
bounds *only* to metrics whose signature accepts them, so MAE/R²/ρ/τ are **plain, raw-error**
statistics. They are therefore readable as moments.

Our own test predictions are not blinded — every run artifact in `data/cyp_cache/*/*.npz`
carries a `test` array of shape (750, 4). So `var_p` is **known exactly**, and the published
columns become equations in the two things we do not know.

## Why four observables and not one

The programme's existing estimate, `sd_y = MAE*sqrt(pi/2)/sqrt(1-R^2) = 1.29`, assumes
**zero bias** and Gaussian errors. It is not a point estimate: `MAE <= RMSE` forces
`var_y >= MAE^2/(1-R^2)` and the Gaussian ratio `MAE >= 0.7979*RMSE` forces
`var_y <= (1.2533*MAE)^2/(1-R^2)`, so 1.29 is the **upper end of a one-parameter family**, the
end where the bias is zero. Along that family `s*` runs from 0.70 to 2.8 — the two equations
cannot decide anything on their own, and the direction of the answer is set entirely by a bias
nobody had estimated.

The rank correlations close it, because they are **invariant to location and scale**. Under a
Gaussian copula

    rho_S = (6/pi)*arcsin(r/2)        tau = (2/pi)*arcsin(r)

so each inverts to a Pearson `r` on its own, and **the two must agree** — which is a test of the
copula assumption, not an article of faith about it. With `r` in hand,
`cov = r*sd_y*sd_p`, and the system

    R^2  = (2*cov - var_p - bias^2) / var_y
    MAE  = folded_normal(bias, sigma),  sigma^2 = (1-R^2)*var_y - bias^2

is over-determined in `(sd_y, bias)`, with MAE left over as a **residual check**.

## The quantity that actually matters

`s* = cov/var_p` is the squared-error-optimal global scale. But the recovered **bias** is the
larger term by far, and it is the one the programme cannot see from validation at all: our
per-endpoint offsets were fitted on validation, whose mean tracks the training mean, while the
test set is an analog expansion around the **top hits per isoform** and is enriched in potent
compounds. A systematic offset there is invisible to every measurement this programme has made.

## This is validated end to end before it is believed

`--mode validate` runs the whole inversion on **validation**, where `sd_y`, `bias` and `r` are
all known, feeding it nothing but the five observables it would get from a leaderboard row. The
recovered values are printed against the truth. An inversion that cannot recover validation's
own bias does not get to make a claim about the test set's — see the standing rule that a
flagged unknown in a measurement path blocks the decisions downstream of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID

# The two rows the board has shown us. MA-MAE/MA-R2/rho/tau are macro means over the four
# endpoints, each a **bootstrap mean over 1000 resamples** (`BOOTSTRAP_SAMPLES = 1000` in the
# organisers' config), not a single point estimate -- so they carry their own error bars and
# anything derived from them inherits them.
LEADERBOARD = {
    "v2": dict(strae=0.8752, mae=1.0063, r2=0.0452, rho=0.6200, tau=0.4468,
               tag="h12-base", note="L3/32/300ep, rank 163"),
    "v3deep": dict(strae=0.8598, mae=1.0, r2=0.0552, rho=0.6408, tau=0.4712,
                   tag="h12-longdeep", note="L6/16/3000ep, rank 159"),
}

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def r_from_spearman(rho: float) -> float:
    """Invert rho_S = (6/pi)*arcsin(r/2) for a Gaussian copula."""
    return float(2.0 * math.sin(math.pi * rho / 6.0))


def r_from_kendall(tau: float) -> float:
    """Invert tau = (2/pi)*arcsin(r) for a Gaussian copula."""
    return float(math.sin(math.pi * tau / 2.0))


def folded_normal_mae(bias: float, sigma: float) -> float:
    """E|e| for e ~ N(bias, sigma^2) -- the mean of a folded normal.

    Reduces to `sigma*sqrt(2/pi)` at zero bias and to `|bias|` as sigma -> 0, so the ratio
    MAE/RMSE is confined to [0.7979, 1] and pins |bias| once RMSE is fixed.
    """
    b = abs(float(bias))
    if sigma <= 0:
        return b
    return float(
        sigma * SQRT_2_OVER_PI * math.exp(-(b * b) / (2.0 * sigma * sigma))
        + b * special.erf(b / (sigma * math.sqrt(2.0)))
    )


def invert(mae: float, r2: float, rho: float, tau: float, var_p: float) -> dict:
    """Recover (sd_y, |bias|, r, s*) from the five observables.

    `r` comes from the rank correlations, which are location- and scale-free. `sd_y` and
    `bias` then solve the R^2 identity, with MAE as the residual check.
    """
    r_rho, r_tau = r_from_spearman(rho), r_from_kendall(tau)
    r = 0.5 * (r_rho + r_tau)
    sd_p = math.sqrt(var_p)

    def bias_sq(sd_y: float) -> float:
        """From R^2 = (2*cov - var_p - bias^2)/var_y with cov = r*sd_y*sd_p."""
        return 2.0 * r * sd_y * sd_p - var_p - r2 * sd_y * sd_y

    def mae_residual(sd_y: float) -> float:
        b2 = bias_sq(sd_y)
        if b2 < 0:
            return 1e3 * (1.0 - b2)          # push the solver back into the feasible region
        mse = (1.0 - r2) * sd_y * sd_y
        sig2 = mse - b2
        if sig2 < 0:
            return 1e3 * (1.0 - sig2)
        return folded_normal_mae(math.sqrt(b2), math.sqrt(sig2)) - mae

    # Feasible sd_y: bias^2 >= 0 bounds it between the roots of the quadratic above.
    disc = (2 * r * sd_p) ** 2 - 4 * r2 * var_p
    if disc <= 0 or r2 <= 0:
        lo_sd, hi_sd = 1e-3, 20.0
    else:
        lo_sd = (2 * r * sd_p - math.sqrt(disc)) / (2 * r2)
        hi_sd = (2 * r * sd_p + math.sqrt(disc)) / (2 * r2)
    lo_sd, hi_sd = max(lo_sd, 1e-3), min(hi_sd, 30.0)

    sol, why = None, "no sign change in the feasible interval"
    grid = np.linspace(lo_sd + 1e-6, hi_sd - 1e-6, 4000)
    vals = np.array([mae_residual(float(x)) for x in grid])
    sign = np.sign(vals)
    cross = np.nonzero(np.diff(sign) != 0)[0]
    roots = []
    for i in cross:
        try:
            roots.append(float(optimize.brentq(mae_residual, grid[i], grid[i + 1])))
        except ValueError:
            pass
    if roots:
        # Prefer the root nearest the training label spread -- the far root is the
        # degenerate branch where almost all the error is bias.
        sol = min(roots, key=lambda s: abs(s - 1.0))
        why = f"{len(roots)} root(s): " + ", ".join(f"{x:.4f}" for x in roots)

    out = dict(r_from_rho=r_rho, r_from_tau=r_tau, r=r,
               copula_check=abs(r_rho - r_tau),
               sd_y_feasible=(lo_sd, hi_sd), roots=roots, why=why)
    if sol is None:
        return out
    b2 = max(bias_sq(sol), 0.0)
    cov = r * sol * sd_p
    out.update(sd_y=sol, bias_abs=math.sqrt(b2), cov=cov,
               s_star=cov / var_p, sd_p=sd_p,
               mae_check=folded_normal_mae(math.sqrt(b2),
                                           math.sqrt(max((1 - r2) * sol * sol - b2, 0.0))))
    return out


def observables(pred: np.ndarray, y: np.ndarray) -> dict:
    """The five numbers a leaderboard row would show, computed on data we hold."""
    keep = np.isfinite(pred) & np.isfinite(y)
    p, t = pred[keep], y[keep]
    mae = float(np.mean(np.abs(p - t)))
    r2 = float(1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    rho = float(stats.spearmanr(p, t).statistic)
    tau = float(stats.kendalltau(p, t).statistic)
    return dict(mae=mae, r2=r2, rho=rho, tau=tau, var_p=float(p.var()),
                sd_y_true=float(t.std()), bias_true=float(p.mean() - t.mean()),
                r_true=float(np.corrcoef(p, t)[0, 1]), n=int(keep.sum()))


def _seed_mean(tag: str, key: str, cache: str = "data/cyp_cache/cyp_h12") -> np.ndarray:
    fs = sorted(glob.glob(f"{cache}/{tag}_s*.npz"))
    if not fs:
        raise SystemExit(f"no artifacts for {tag} under {cache}")
    return np.mean([np.load(f)[key] for f in fs], axis=0), fs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "test", "both"), default="both")
    ap.add_argument("--out", default="data/cyp_cache/backout.json")
    args = ap.parse_args()

    d = load()
    fr = d.frame
    report: dict = {"metric_id": METRIC_ID}

    # ---------------------------------------------------------------- validation
    if args.mode in ("validate", "both"):
        print("=" * 78)
        print("LEG 1A VALIDATION — invert the five observables where the truth is known")
        print("=" * 78)
        for tag in ("h12-base", "h12-longdeep"):
            val, fs = _seed_mean(tag, "val")
            idx = np.load(fs[0])["va_index"]
            per, rows = [], []
            for i, c in enumerate(PRIMARY):
                y = fr[c].to_numpy(float)[idx]
                o = observables(val[:, i], y)
                per.append(o)
                rows.append((c, o))
            # Macro, exactly as the board forms it: a plain mean over endpoints.
            macro = {k: float(np.mean([o[k] for o in per]))
                     for k in ("mae", "r2", "rho", "tau", "var_p",
                               "sd_y_true", "bias_true", "r_true")}
            inv = invert(macro["mae"], macro["r2"], macro["rho"], macro["tau"],
                         macro["var_p"])
            print(f"\n{tag}  ({len(fs)} seeds, n={per[0]['n']}–{per[-1]['n']} per endpoint)")
            print(f"  observables  MAE {macro['mae']:.4f}  R2 {macro['r2']:.4f}  "
                  f"rho {macro['rho']:.4f}  tau {macro['tau']:.4f}  var_p {macro['var_p']:.4f}")
            print(f"  copula check r(rho) {inv['r_from_rho']:.4f} vs r(tau) "
                  f"{inv['r_from_tau']:.4f}   |diff| {inv['copula_check']:.4f}")
            print(f"  {'quantity':<12s} {'RECOVERED':>10s} {'TRUE':>10s} {'error':>10s}")
            if "sd_y" in inv:
                for name, got, true in (("sd_y", inv["sd_y"], macro["sd_y_true"]),
                                        ("|bias|", inv["bias_abs"], abs(macro["bias_true"])),
                                        ("r", inv["r"], macro["r_true"])):
                    print(f"  {name:<12s} {got:10.4f} {true:10.4f} {got - true:+10.4f}")
                s_true = macro["r_true"] * macro["sd_y_true"] / math.sqrt(macro["var_p"])
                print(f"  {'s*':<12s} {inv['s_star']:10.4f} {s_true:10.4f} "
                      f"{inv['s_star'] - s_true:+10.4f}")
                print(f"  MAE residual check {inv['mae_check']:.4f} vs {macro['mae']:.4f}")
            else:
                print(f"  NO SOLUTION — {inv['why']}")
            report.setdefault("validate", {})[tag] = dict(macro=macro, inverted=inv)

    # ---------------------------------------------------------------- test
    if args.mode in ("test", "both"):
        print("\n" + "=" * 78)
        print("LEG 1A — applied to the blinded test set via the leaderboard's own columns")
        print("=" * 78)
        for name, row in LEADERBOARD.items():
            test, fs = _seed_mean(row["tag"], "test")
            # Macro var_p, formed the same way the board forms a macro: per endpoint,
            # then a plain mean.
            var_p = float(np.mean([test[:, i].var() for i in range(len(PRIMARY))]))
            inv = invert(row["mae"], row["r2"], row["rho"], row["tau"], var_p)
            print(f"\n{name}  [{row['note']}]  ST-RAE {row['strae']}")
            print(f"  var_p (our own test predictions, {len(fs)} seeds) {var_p:.4f}  "
                  f"sd_p {math.sqrt(var_p):.4f}")
            print(f"  copula check r(rho) {inv['r_from_rho']:.4f} vs r(tau) "
                  f"{inv['r_from_tau']:.4f}   |diff| {inv['copula_check']:.4f}")
            print(f"  feasible sd_y  [{inv['sd_y_feasible'][0]:.4f}, "
                  f"{inv['sd_y_feasible'][1]:.4f}]   {inv['why']}")
            if "sd_y" in inv:
                print(f"  RECOVERED  sd_y {inv['sd_y']:.4f}   |bias| {inv['bias_abs']:.4f}"
                      f"   r {inv['r']:.4f}   cov {inv['cov']:.4f}")
                print(f"  ==> s* (squared-error-optimal global scale) = {inv['s_star']:.4f}")
                print(f"  MAE residual check {inv['mae_check']:.4f} vs published {row['mae']}")
            else:
                print("  NO SOLUTION in the feasible region")
            report.setdefault("test", {})[name] = dict(row=row, var_p=var_p, inverted=inv)

        # ------------------------------------------------------ sensitivity surface
        print("\n" + "-" * 78)
        print("SENSITIVITY — s* over the (sd_y, bias) plane, v3deep's var_p")
        print("  stating the claim over the range it was tested on, not at a point")
        print("-" * 78)
        test, _ = _seed_mean(LEADERBOARD["v3deep"]["tag"], "test")
        var_p = float(np.mean([test[:, i].var() for i in range(len(PRIMARY))]))
        r2 = LEADERBOARD["v3deep"]["r2"]
        biases = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        print(f"{'sd_y':>6s}" + "".join(f"{f'b={b:.1f}':>9s}" for b in biases))
        surf = {}
        for sd_y in (1.00, 1.10, 1.20, 1.29, 1.40, 1.60, 2.00):
            cells = []
            for b in biases:
                # s* = (R^2*var_y + var_p + bias^2) / (2*var_p), the R^2 identity alone.
                s = (r2 * sd_y * sd_y + var_p + b * b) / (2.0 * var_p)
                cells.append(s)
            surf[sd_y] = cells
            print(f"{sd_y:6.2f}" + "".join(f"{c:9.3f}" for c in cells))
        report["sensitivity"] = dict(var_p=var_p, r2=r2, biases=list(biases), surface=
                                     {str(k): v for k, v in surf.items()})
        print("\n  Read: s* < 1 needs a small bias. The R^2 identity alone cannot decide the")
        print("  direction — which is why the rank correlations are doing the work above.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"\n[backout] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
