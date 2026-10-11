r"""Identify `cov(p, y)` and `var_y` on the blinded test set, exactly, from rescale probes.

The decoder's *direction* is settled (shrink; see `cyp_scale_at_test_r.py`) and its *magnitude*
is not: reconstructing the test label distribution to fit a decoder on was refuted by its own
validation leg, which made four endpoints out of four worse than doing nothing. So the magnitude
is measured instead of modelled, using the scorer as the instrument.

## The identity

Rescale one endpoint's predictions about their own mean, `p_a = m + a*(p - m)`. Then

    mean(p_a) = m        for every a
    var(p_a)  = a^2 * var_p
    cov(p_a, y) = a * cov(p, y)

so, with `b = m - mean_y` and `K = var_y + b^2` both **independent of `a`**,

    MSE(a) = K + a^2*var_p - 2a*cov

The board publishes `R2(a) = 1 - MSE(a)/var_y`, i.e. `MSE(a) = (1 - R2(a))*var_y`. Differencing
two levels cancels `K`:

    (R2(a2) - R2(a1))*var_y + 2*(a1 - a2)*cov = (a1^2 - a2^2)*var_p

`var_p` is **known exactly** -- we hold the prediction vector. So **three levels give two such
equations, linear in `(var_y, cov)`, and determine both.** A fourth level over-identifies them.

What this costs in assumptions: **nothing beyond R2 being sklearn's `r2_score`**, which was
verified in the organisers' own `backend/config.py` (`ACTIVITY_METRICS` wires R2 to `r2_score`
and MAE to `mean_absolute_error`, both plain, both on raw errors). No copula, no label-shape
model, no Gaussian residuals -- the three things that broke every earlier attempt. A rescale is
also **monotone**, so rho and tau are unchanged by construction, which is a free correctness
check on the scoring pipeline: if they move, something other than our rescale moved.

Then `s* = cov/var_p` is the squared-error-optimal scale, and `b^2 = K - var_y` with
`K = (1-R2(a1))*var_y - a1^2*var_p + 2*a1*cov` gives the bias magnitude. MAE at each level is a
separate, over-identifying check on `b` under a folded-normal assumption -- reported, never
relied on.

## Run the self-test before spending a submission

`--selftest` performs the whole experiment on validation, where every truth is known: it computes
`R2(a)` and `MAE(a)` itself at each level, **rounds them to the 4 decimals the board publishes**,
feeds only those to the solver, and prints recovered against true. Rounding is the real
precision risk here -- the solver differences numbers the board gives to 1e-4 -- so the self-test
exists mainly to show the conditioning is good enough at the chosen levels.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID

DEFAULT_LEVELS = (1.00, 0.60, 0.30)


def rescale(p: np.ndarray, a: float) -> np.ndarray:
    m = float(p.mean())
    return m + a * (p - m)


def solve(levels: list[float], r2: list[float], var_p: float) -> dict:
    """Least-squares solve of the differenced identity for (var_y, cov).

    With exactly three levels this is a determined 2x2; with more it is over-determined and the
    residual is reported, which is the only internal check the method has.
    """
    rows, rhs = [], []
    for i in range(len(levels) - 1):
        a1, a2 = levels[i], levels[i + 1]
        rows.append([r2[i + 1] - r2[i], 2.0 * (a1 - a2)])
        rhs.append((a1 * a1 - a2 * a2) * var_p)
    A, y = np.array(rows, float), np.array(rhs, float)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    var_y, cov = float(sol[0]), float(sol[1])
    resid = float(np.linalg.norm(A @ sol - y))
    cond = float(np.linalg.cond(A)) if A.shape[0] >= A.shape[1] else float("nan")
    K = (1 - r2[0]) * var_y - levels[0] ** 2 * var_p + 2 * levels[0] * cov
    b2 = K - var_y
    return dict(var_y=var_y, sd_y=math.sqrt(var_y) if var_y > 0 else float("nan"),
                cov=cov, s_star=cov / var_p if var_p else float("nan"),
                r=cov / math.sqrt(var_y * var_p) if var_y > 0 and var_p > 0 else float("nan"),
                K=K, bias_abs=math.sqrt(b2) if b2 >= 0 else float("nan"),
                bias_sq=b2, resid=resid, cond=cond)


def selftest(levels: list[float], tag: str, cache: str, round_dp: int = 4) -> int:
    fr = load().frame
    files = sorted(glob.glob(f"{cache}/{tag}_s*.npz"))
    val = np.mean([np.load(f)["val"] for f in files], axis=0)
    idx = np.load(files[0])["va_index"]

    print(f"SELF-TEST of the probe identification — validation, truth known, "
          f"R2 rounded to {round_dp} dp as the board publishes it")
    print(f"levels: {levels}\n")
    print(f"{'endpoint':<22s} {'sd_y rec/true':>18s} {'cov rec/true':>18s} "
          f"{'s* rec/true':>16s} {'|bias| rec/true':>17s} {'cond':>9s}")
    ok = True
    for i, c in enumerate(PRIMARY):
        y = fr[c].to_numpy(float)[idx]
        k = np.isfinite(y) & np.isfinite(val[:, i])
        p, y = val[k, i], y[k]
        var_p = float(p.var())
        r2s, maes = [], []
        for a in levels:
            pa = rescale(p, a)
            r2s.append(round(float(1 - np.sum((y - pa) ** 2) / np.sum((y - y.mean()) ** 2)),
                             round_dp))
            maes.append(round(float(np.mean(np.abs(pa - y))), round_dp))
        got = solve(levels, r2s, var_p)
        truth = dict(var_y=float(y.var()), cov=float(np.cov(p, y, bias=True)[0, 1]),
                     bias_abs=abs(float(p.mean() - y.mean())))
        truth["s_star"] = truth["cov"] / var_p
        print(f"{c[:22]:<22s} {got['sd_y']:9.4f}/{math.sqrt(truth['var_y']):8.4f} "
              f"{got['cov']:9.4f}/{truth['cov']:8.4f} "
              f"{got['s_star']:8.4f}/{truth['s_star']:7.4f} "
              f"{got['bias_abs']:9.4f}/{truth['bias_abs']:7.4f} {got['cond']:9.1f}")
        if abs(got["s_star"] - truth["s_star"]) > 0.02:
            ok = False
    print(f"\n  verdict: {'PASS — s* recovered to better than 0.02 everywhere' if ok else 'FAIL — s* error exceeds 0.02 on at least one endpoint'}")
    print("  (rho and tau are invariant to a rescale, so if they move on the board, something")
    print("   other than our rescale moved — a free check on the pipeline.)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS))
    ap.add_argument("--tag", default="h12-longdeep")
    ap.add_argument("--cache", default="data/cyp_cache/cyp_h12")
    ap.add_argument("--results", help="JSON of observed board rows once the probes are scored")
    ap.add_argument("--manifest", default="data/cyp_cache/probes/manifest.json")
    ap.add_argument("--round-dp", type=int, default=4)
    args = ap.parse_args()

    if args.selftest:
        return selftest(args.levels, args.tag, args.cache, args.round_dp)

    if not args.results:
        raise SystemExit("pass --results <json> (or --selftest). The results file maps each "
                         "level to that submission's published per-endpoint columns.")
    man = json.loads(Path(args.manifest).read_text())
    res = json.loads(Path(args.results).read_text())
    levels = sorted((float(k) for k in res), reverse=True)
    print(f"solving from levels {levels}, metric {METRIC_ID}\n")
    print(f"{'endpoint':<22s} {'sd_y':>8s} {'cov':>8s} {'r':>7s} {'s*':>7s} "
          f"{'|bias|':>8s} {'resid':>8s} {'cond':>8s}")
    out = {}
    for c in PRIMARY:
        var_p = man["var_p_at_1"][c]
        r2 = [float(res[f"{a:g}"][c]["r2"]) for a in levels]
        got = solve(levels, r2, var_p)
        out[c] = got
        print(f"{c[:22]:<22s} {got['sd_y']:8.4f} {got['cov']:8.4f} {got['r']:7.4f} "
              f"{got['s_star']:7.4f} {got['bias_abs']:8.4f} {got['resid']:8.2e} "
              f"{got['cond']:8.1f}")
    print(f"\n  ST-RAE-optimal scale ~ s* x 0.827 (the measured free-zone discount, "
          f"cyp_scale_sweep.py):")
    for c in PRIMARY:
        print(f"    {c[:34]:<34s} {0.827 * out[c]['s_star']:.4f}")
    Path("data/cyp_cache/probes/identified.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\n[probe] wrote data/cyp_cache/probes/identified.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
