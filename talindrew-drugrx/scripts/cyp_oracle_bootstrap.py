r"""Run-level bootstrap for the oracle-gap fit. 630 pairs are not 630 observations.

`cyp_oracle_rho.py` fits over all C(36,2)=630 pairs and reports a residual difference
of -0.00395 +/- 0.00081 (t=-4.89). **That standard error is optimistic**: each of the 36
runs appears in 35 pairs, so the pairs are heavily dependent and the effective sample
size is nearer 36 than 630. R^2=0.943 and the sign of the residual are far outside any
plausible correction, but the +/- is the number other people would compute against, so
it needs to be right.

This resamples **runs**, not pairs (a cluster bootstrap over the dependency structure):
draw 36 runs with replacement, rebuild the pairs among the drawn runs, refit, recompute
the statistic. Pairs of a run with itself are excluded -- they have rho=1 and gap=0 by
construction and would bias the fit toward the constrained end.
"""
from __future__ import annotations

import itertools, json, sys
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error, st_rae


def main() -> int:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    models = {}
    for sub in ("cyp_loss", "cyp_aux"):
        for f in sorted((Path("data/cyp_cache") / sub).glob("*.json")):
            npz = f.with_suffix(".npz")
            if npz.exists():
                tag = json.loads(f.read_text()).get("tag", f.stem)
                models[f.stem] = (npz, tag.replace("aux-", "").replace("loss-", ""))

    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    names = sorted(models)
    cache, idx = {}, None
    for n in names:
        z = np.load(models[n][0])
        idx = z["va_index"] if idx is None else idx
        p = z["val"]
        cache[n] = {
            "errs": {c: soft_threshold_error(p[:, i], lo[c][idx], hi[c][idx])
                     for i, c in enumerate(PRIMARY)},
            "p": p, "arm": models[n][1],
            "macro": float(np.nanmean([st_rae(p[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                      for i, c in enumerate(PRIMARY)])),
        }

    # Precompute every pair once; the bootstrap only re-selects among them.
    pdata = {}
    for n1, n2 in itertools.combinations(names, 2):
        a, b = cache[n1], cache[n2]
        rhos, out = [], np.empty_like(a["p"])
        for i, c in enumerate(PRIMARY):
            ea, eb = a["errs"][c], b["errs"][c]
            k = np.isfinite(ea) & np.isfinite(eb)
            if ea[k].std() > 0 and eb[k].std() > 0:
                rhos.append(float(np.corrcoef(ea[k], eb[k])[0, 1]))
            out[:, i] = np.where(eb < ea, b["p"][:, i], a["p"][:, i])
        orac = float(np.nanmean([st_rae(out[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                for i, c in enumerate(PRIMARY)]))
        pdata[(n1, n2)] = (float(np.mean(rhos)),
                           orac - min(a["macro"], b["macro"]),
                           a["arm"] == b["arm"])

    def stat(keys):
        rho = np.array([pdata[k][0] for k in keys])
        gap = np.array([pdata[k][1] for k in keys])
        same = np.array([pdata[k][2] for k in keys])
        if same.sum() < 2 or (~same).sum() < 2:
            return None
        A = np.vstack([np.ones_like(rho), np.sqrt(1 - rho)]).T
        coef, *_ = np.linalg.lstsq(A, gap, rcond=None)
        r = gap - A @ coef
        ss = ((gap - gap.mean()) ** 2).sum()
        return coef[1], coef[0], 1 - (r ** 2).sum() / ss, r[same].mean() - r[~same].mean()

    allk = list(pdata)
    slope0, icept0, r20, diff0 = stat(allk)
    print(f"full sample: slope {slope0:+.4f}  intercept {icept0:+.4f}  "
          f"R^2 {r20:.4f}  residual diff {diff0:+.5f}")

    rng = np.random.default_rng(0)
    slopes, icepts, r2s, diffs = [], [], [], []
    for _ in range(B):
        draw = rng.choice(len(names), size=len(names), replace=True)
        keys = []
        for i, j in itertools.combinations(range(len(draw)), 2):
            if draw[i] == draw[j]:
                continue            # a run paired with itself: rho=1, gap=0
            a, b = names[draw[i]], names[draw[j]]
            keys.append((a, b) if (a, b) in pdata else (b, a))
        s = stat(keys)
        if s is None:
            continue
        slopes.append(s[0]); icepts.append(s[1]); r2s.append(s[2]); diffs.append(s[3])

    def rep(label, v, point):
        v = np.array(v)
        lo_, hi_ = np.percentile(v, [2.5, 97.5])
        print(f"  {label:18s} {point:+.5f}  bootstrap sd {v.std(ddof=1):.5f}  "
              f"95% CI [{lo_:+.5f}, {hi_:+.5f}]")

    print(f"\nrun-level bootstrap, B={len(diffs)} (resampling the 36 runs, not the pairs)")
    rep("slope", slopes, slope0)
    rep("intercept", icepts, icept0)
    rep("R^2", r2s, r20)
    rep("residual diff", diffs, diff0)

    d = np.array(diffs)
    print(f"\n  residual diff: naive t was -4.89 on se 0.00081")
    print(f"  clustered t = {diff0 / d.std(ddof=1):+.2f} on se {d.std(ddof=1):.5f}")
    print(f"  share of bootstrap draws with the same sign: "
          f"{100 * (np.sign(d) == np.sign(diff0)).mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
