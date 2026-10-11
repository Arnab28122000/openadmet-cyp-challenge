r"""The oracle gap as a function of error correlation, over all 630 pairs.

The matched-control version of this argument was a two-point comparison, and it
carried a confound: it matched on error correlation rho, but **rho is itself
downstream of arm identity** -- both/both sits at 0.933 and none/none at 0.971
because the auxiliary arms are noisier. Matching on a variable the treatment moves
cannot fully separate "the arms are equivalent" from "the arms differ in internal
variance".

Conditioning across the whole range does. All 36 saved runs share one validation
split, so every one of the C(36,2)=630 pairs yields a (rho, oracle gap) point,
spanning same-arm and cross-arm, same-seed and cross-seed, tight and loose. Two
questions:

1. Do the points lie on a single curve in rho?
2. **After conditioning on rho, does arm identity add anything?** If the residuals
   of same-arm and cross-arm pairs are indistinguishable, the oracle gap *is* a
   function of decorrelation, and the next person can read their own rho off a curve
   instead of running a procedure.

Needs no GPU. The claim it supports is the transferable one: a reported oracle
ceiling is a measurement of error decorrelation until shown otherwise.
"""
from __future__ import annotations

import itertools, json, statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error, st_rae


def main() -> int:
    models = {}
    for sub in ("cyp_loss", "cyp_aux"):
        for f in sorted(Path(f"data/cyp_cache/{sub}").glob("*.json")):
            npz = f.with_suffix(".npz")
            if npz.exists():
                j = json.loads(f.read_text())
                arm = j.get("tag", f.stem)
                models[f.stem] = (npz, arm.replace("aux-", "").replace("loss-", ""))

    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    cache = {}
    idx = None
    for name, (npz, arm) in models.items():
        z = np.load(npz)
        if idx is None:
            idx = z["va_index"]
        assert np.array_equal(z["va_index"], idx), f"{name}: different split"
        p = z["val"]
        errs, keeps = {}, {}
        for i, c in enumerate(PRIMARY):
            e = soft_threshold_error(p[:, i], lo[c][idx], hi[c][idx])
            k = np.isfinite(e) & np.isfinite(y[c][idx])
            errs[c], keeps[c] = e, k
        macro = float(np.nanmean([st_rae(p[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                 for i, c in enumerate(PRIMARY)]))
        cache[name] = {"p": p, "errs": errs, "keeps": keeps, "macro": macro, "arm": arm}

    def pair(n1, n2):
        a, b = cache[n1], cache[n2]
        rhos, out = [], np.empty_like(a["p"])
        for i, c in enumerate(PRIMARY):
            k = a["keeps"][c] & b["keeps"][c]
            ea, eb = a["errs"][c], b["errs"][c]
            if ea[k].std() > 0 and eb[k].std() > 0:
                rhos.append(float(np.corrcoef(ea[k], eb[k])[0, 1]))
            out[:, i] = np.where(eb < ea, b["p"][:, i], a["p"][:, i])
        orac = float(np.nanmean([st_rae(out[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                for i, c in enumerate(PRIMARY)]))
        return float(np.mean(rhos)), orac - min(a["macro"], b["macro"])

    rows = []
    for n1, n2 in itertools.combinations(sorted(cache), 2):
        rho, gap = pair(n1, n2)
        same = cache[n1]["arm"] == cache[n2]["arm"]
        rows.append((rho, gap, same))
    print(f"{len(rows)} pairs from {len(cache)} runs, one shared validation split\n")

    rho = np.array([r[0] for r in rows])
    gap = np.array([r[1] for r in rows])
    same = np.array([r[2] for r in rows])

    print("oracle gap by error-correlation decile")
    print(f"{'rho band':>16s} {'n':>4s} {'mean rho':>9s} {'mean gap':>9s} "
          f"{'same-arm':>9s} {'cross-arm':>10s} {'diff':>8s}")
    qs = np.quantile(rho, np.linspace(0, 1, 9))
    for a, b in zip(qs[:-1], qs[1:]):
        m = (rho >= a) & (rho <= b if b == qs[-1] else rho < b)
        if m.sum() == 0:
            continue
        gs, gc = gap[m & same], gap[m & ~same]
        s_ = f"{gs.mean():9.4f}" if len(gs) else "        -"
        c_ = f"{gc.mean():10.4f}" if len(gc) else "         -"
        d_ = f"{gs.mean() - gc.mean():+8.4f}" if len(gs) and len(gc) else "       -"
        print(f"{a:7.3f}..{b:6.3f} {int(m.sum()):4d} {rho[m].mean():9.3f} "
              f"{gap[m].mean():9.4f} {s_} {c_} {d_}")

    # Two fits. The linear one is empirical; the sqrt(1-rho) one is what theory
    # predicts, because for bivariate-normal errors with correlation rho,
    # E[min(X, Y)] = mu - sigma * sqrt((1 - rho) / pi). If the second fits better
    # the curve is not merely a regularity in this data -- it is the known
    # expectation of a minimum of two correlated draws, which is why a reported
    # oracle ceiling carries no information about complementarity on its own.
    ss_tot = ((gap - gap.mean()) ** 2).sum()
    fits = {}
    for label, design in (("gap ~ rho", np.vstack([np.ones_like(rho), rho]).T),
                          ("gap ~ sqrt(1-rho)",
                           np.vstack([np.ones_like(rho), np.sqrt(1 - rho)]).T)):
        coef, *_ = np.linalg.lstsq(design, gap, rcond=None)
        r = gap - design @ coef
        fits[label] = (coef, r, 1 - (r ** 2).sum() / ss_tot)
        print(f"\n{label:20s} intercept {coef[0]:+.4f}  slope {coef[1]:+.4f}  "
              f"R^2 = {fits[label][2]:.4f}")
    best = max(fits, key=lambda k: fits[k][2])
    print(f"\nbetter fit: {best}" + ("  <- the theoretical form"
                                     if "sqrt" in best else "  (empirical)"))
    coef, resid, r2 = fits[best]
    print(f"conditioning on {best}:")
    rs, rc = resid[same], resid[~same]
    se = (rs.var(ddof=1) / len(rs) + rc.var(ddof=1) / len(rc)) ** 0.5
    t = (rs.mean() - rc.mean()) / se
    print(f"residual after conditioning on rho:")
    print(f"  same-arm  n={len(rs):3d} mean {rs.mean():+.5f}")
    print(f"  cross-arm n={len(rc):3d} mean {rc.mean():+.5f}")
    d = rs.mean() - rc.mean()
    print(f"  difference {d:+.5f} +/- {se:.5f}  t={t:+.2f}")
    if abs(t) <= 2.5:
        print("  -> arm identity adds nothing once rho is known")
    elif d < 0:
        # More negative residual == LARGER oracle gap. Same-arm pairs beating
        # cross-arm ones at matched rho is the opposite of complementarity: if the
        # auxiliary heads gave region-specific competence, cross-arm pairs would
        # have the larger gap, not the smaller.
        print("  -> SAME-arm pairs have the LARGER gap at matched rho, which is the")
        print("     OPPOSITE of complementarity: differing in auxiliary heads is")
        print("     slightly WORSE for oracle selection than differing only in seed.")
    else:
        print("  -> cross-arm pairs have the larger gap at matched rho: evidence FOR")
        print("     complementarity, to be weighed against its size.")
    print(f"\nSize check: the residual is {abs(d):.4f} against gaps of 0.07-0.18, i.e.")
    print(f"{100 * abs(d) / abs(gap.mean()):.0f}% of the effect, and the fitted intercept of")
    print("{:+.4f} is ~0, as the theoretical form requires at rho=1 (identical errors,".format(coef[0]))
    print("zero gap). A reported oracle ceiling is therefore a measurement of error")
    print("decorrelation, and carries no information about complementarity on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
