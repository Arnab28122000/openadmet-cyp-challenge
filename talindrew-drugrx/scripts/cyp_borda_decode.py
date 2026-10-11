"""The Spearman-optimal score is the EXPECTED RANK, not the conditional mean.

`cyp_decode_ceiling.py` bounded every map of the prediction `p` and found +0.028 in-sample,
negative cross-fitted. That result is correct and it is **narrower than how it was summarised**:
it bounds the 1-D family {f: p -> y-hat}. It says our model is already monotonically calibrated
(MCB in the MCB-DSC-UNC decomposition) -- a scalar map can remove miscalibration but cannot add
discrimination. It says nothing about transforms that depend on a per-item quantity.

The rank-optimal score is one of those. Under independent Y_i ~ N(p_i, s_i^2),

    E[rank_i]  =  sum_j P(Y_i > Y_j)  =  sum_j Phi( (p_i - p_j) / sqrt(s_i^2 + s_j^2) )

the Borda score. **If every s_i is equal this is a strictly increasing function of p_i**, so the
ranking -- and therefore Spearman -- is unchanged. The entire effect comes from
HETEROSCEDASTICITY: an uncertain item is shrunk toward the middle of the order, because its
probability of beating anything is closer to 1/2.

That built-in degeneracy is also the correctness test, and this script runs it: with a constant
sigma the Borda ranking must match the raw ranking EXACTLY. If it does not, the implementation is
wrong and every other number here is void.

Three arms, all scored on the same compounds with the same splits:

* **raw**      -- rank by p. The incumbent.
* **borda**    -- rank by the expected rank, using an ensemble's per-item spread as sigma-hat.
* **oracle2d** -- bin on (p, sigma-hat) and give each cell the mean label, fitted on the scored
                  compounds. The CEILING over all functions of (p, sigma-hat), unattainable by
                  construction, and reported next to a cross-fitted version so the gap is visible.

    python scripts/cyp_borda_decode.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

#: Ensemble members whose spread becomes sigma-hat. Each sees a different bootstrap of the
#: training rows AND a different model seed, so the spread reflects both data and fit variance.
N_MEMBERS = 8
BIN_SWEEP = (3, 4, 6)


def borda(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Expected rank under independent Gaussians. O(n^2) and n<=1000 here, so done densely."""
    d = p[:, None] - p[None, :]
    v = np.sqrt(s[:, None] ** 2 + s[None, :] ** 2)
    v = np.maximum(v, 1e-9)
    z = norm.cdf(d / v)
    np.fill_diagonal(z, 0.5)
    return z.sum(axis=1)


def oracle_2d(p, s, y, nb, fit=None):
    """Bin on (p, sigma) quantiles, give each cell its own mean label.

    This is the unconstrained optimum over functions of (p, sigma) at this resolution -- the
    ceiling the Borda transform is one member of. `fit` selects the rows the cell means are
    estimated from; passing all rows gives the (unattainable) oracle.
    """
    fit = np.arange(len(p)) if fit is None else fit
    pe = np.quantile(p[fit], np.linspace(0, 1, nb + 1)); pe[0], pe[-1] = -np.inf, np.inf
    se = np.quantile(s[fit], np.linspace(0, 1, nb + 1)); se[0], se[-1] = -np.inf, np.inf
    pi_f, si_f = np.digitize(p[fit], pe[1:-1]), np.digitize(s[fit], se[1:-1])
    pi_a, si_a = np.digitize(p, pe[1:-1]), np.digitize(s, se[1:-1])
    out = p.astype(float).copy()
    for a in range(nb):
        for b in range(nb):
            m = (pi_f == a) & (si_f == b)
            tgt = (pi_a == a) & (si_a == b)
            if tgt.any() and m.sum() >= 5:
                out[tgt] = float(np.mean(y[fit][m]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/borda_decode.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[borda] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[borda] seed {seed}", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            okv = np.isfinite(y[va])
            if len(idx) < 50 or okv.sum() < 60:
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 + seed)
            preds = []
            for m in range(N_MEMBERS):
                bs = rng.choice(idx, size=len(idx), replace=True)
                preds.append(gbm(seed * 100 + m).fit(
                    X[bs], yy[bs], sample_weight=w[bs]).predict(X[va])[okv])
            P = np.vstack(preds)
            p, s = P.mean(axis=0), P.std(axis=0, ddof=1)
            yv = y[va][okv]

            rho_raw = float(spearmanr(p, yv).statistic)
            rho_bor = float(spearmanr(borda(p, s), yv).statistic)

            # Correctness test: a CONSTANT sigma must leave the ranking untouched.
            rho_const = float(spearmanr(borda(p, np.full_like(s, s.mean())), yv).statistic)
            degenerate_ok = abs(rho_const - rho_raw) < 1e-9

            rec = dict(seed=seed, iso=c, n=int(okv.sum()),
                       rho_raw=rho_raw, rho_borda=rho_bor,
                       d_borda=rho_bor - rho_raw,
                       sigma_cv=float(s.std(ddof=1) / max(s.mean(), 1e-9)),
                       const_sigma_identical=bool(degenerate_ok),
                       secs=round(time.time() - t0, 1))
            half = rng.permutation(len(p))
            h1, h2 = half[: len(p) // 2], half[len(p) // 2:]
            for nb in BIN_SWEEP:
                orc = oracle_2d(p, s, yv, nb)
                cf = np.empty(len(p), float)
                for f_i, a_i in ((h1, h2), (h2, h1)):
                    tmp = oracle_2d(p, s, yv, nb, fit=f_i)
                    cf[a_i] = tmp[a_i]
                rec[f"rho_oracle2d_{nb}"] = float(spearmanr(orc, yv).statistic)
                rec[f"rho_cf2d_{nb}"] = float(spearmanr(cf, yv).statistic)
            rows.append(rec)
            flag = "" if degenerate_ok else "   ** CONST-SIGMA TEST FAILED **"
            print(f"   {short(c)}: n={rec['n']:4d}  raw {rho_raw:.4f}  borda {rho_bor:.4f}  "
                  f"d {rec['d_borda']:+.4f}   sigma CV {rec['sigma_cv']:.2f}{flag}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1

    bad = [r for r in rows if not r["const_sigma_identical"]]
    print("\n" + "=" * 86)
    if bad:
        print(f"!! CONST-SIGMA DEGENERACY TEST FAILED on {len(bad)} cells -- results are VOID.")
        return 2
    print("const-sigma degeneracy test PASSED on every cell: with equal sigma the Borda ranking")
    print("is identical to the raw ranking, so any effect below is heteroscedasticity and")
    print("nothing else.")

    print("\n=== Borda (expected rank) vs raw, paired over seeds ===")
    print(f"{'iso':8} {'raw rho':>9} {'borda rho':>10} {'delta':>9} {'t':>7} {'wins':>7} "
          f"{'sigma CV':>9}")
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if not rs:
            continue
        d = np.array([r["d_borda"] for r in rs])
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))) if len(d) > 1 and d.std() else 0.0
        print(f"{short(c):8} {np.mean([r['rho_raw'] for r in rs]):9.4f} "
              f"{np.mean([r['rho_borda'] for r in rs]):10.4f} {d.mean():+9.4f} {t:7.2f} "
              f"{sum(d > 0):3d}/{len(d):<3d} {np.mean([r['sigma_cv'] for r in rs]):9.2f}")
    dall = np.array([r["d_borda"] for r in rows])
    print(f"{'MACRO':8} {'':9} {'':10} {dall.mean():+9.4f} {'':7} {sum(dall > 0):3d}/{len(dall):<3d}")

    print("\n=== ceiling over ALL functions of (p, sigma) -- the family Borda belongs to ===")
    print(f"  {'bins':>5} {'oracle rho':>11} {'crossfit rho':>13} {'oracle gain':>12} "
          f"{'REAL gain':>11}")
    base = float(np.mean([r["rho_raw"] for r in rows]))
    for nb in BIN_SWEEP:
        o = float(np.mean([r[f"rho_oracle2d_{nb}"] for r in rows]))
        x = float(np.mean([r[f"rho_cf2d_{nb}"] for r in rows]))
        print(f"  {nb:5d} {o:11.4f} {x:13.4f} {o-base:+12.4f} {x-base:+11.4f}")
    print("\n  The oracle is fitted on the compounds it scores and is unattainable. If the")
    print("  cross-fitted column is negative at every resolution, then NO function of")
    print("  (p, sigma) helps -- and Borda, being one such function, cannot either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
