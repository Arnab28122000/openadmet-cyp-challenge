"""How much MORE data would a 22% cut in absolute error need? Price the data lever before buying it.

`cyp_lift_decompose.py` restated the gap to the field in the units the model actually controls:
closing it needs **sigma* 0.727 -> 0.567 pIC50, a 22% cut in absolute error**, where every lever
ever measured on this programme was worth about 1%. The only lever class with the right order of
magnitude is more data -- and the only intervention that ever worked here was a *data* insight
(screen-derived interval labels, +0.0417).

So measure the returns before paying for them. This fits a learning curve on the data already in
hand: train the production recipe on nested subsamples of the training pool, score on a **fixed**
held-out set per seed, and fit

    log sigma* = a - alpha * log n

The exponent `alpha` gives the data multiplier required for any target error cut:

    n_factor = (1 - cut) ** (-1 / alpha)

### Why this is an UPPER BOUND on what external data could deliver, not an estimate

The curve measures returns to **in-distribution** data: more compounds from the same assay, the
same laboratory and the same chemistry. Any real external source (ChEMBL CYP IC50s, PubChem
bioassays) is a different assay with different protocols and a different chemical distribution,
and this programme has already measured that penalty once -- **10.5k external Veith inactives
made the model WORSE, 3/3 seeds**. So the honest reading is one-directional:

    if even ideal in-distribution data at N x cannot deliver 22%, external data cannot either.

That makes a negative result here decisive and a positive result merely permissive.

### The extrapolation is an extrapolation

A power law fitted over a 6.7x range of n and then read at 3-30x is exactly the move that fails.
The script therefore prints the measured range, the fit quality, and the required factor as a
**band** across isoforms rather than a single number, and flags when the required factor sits
beyond an order of magnitude past the data.

    python scripts/cyp_data_scaling.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import allr_floor, screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
FRACTIONS = (0.15, 0.30, 0.45, 0.60, 0.80, 1.00)
#: the cut in absolute error that reaching the field's best requires, per cyp_lift_decompose.py
TARGET_CUT = {"CYP1A2": 0.21, "CYP2C9": 0.20, "CYP3A4": 0.25}


def short(col: str) -> str:
    return col.split("_")[0]


def sigma_star(s_y: float, rho: float) -> float:
    """Residual sd after an optimal affine map -- the placement-free accuracy figure."""
    r = 2.0 * np.sin(np.pi * rho / 6.0)
    return float(s_y * np.sqrt(max(1e-9, 1.0 - r * r)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/data_scaling.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, _floor = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[scale] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        rng = np.random.default_rng(1000 + seed)
        order = rng.permutation(tr)          # nested subsamples: prefix of one shuffle
        print(f"\n[scale] seed {seed}: pool {len(tr)} train / {len(va)} fixed val", flush=True)

        for f in FRACTIONS:
            keep_tr = order[: max(1, int(round(f * len(order))))]
            for c in PRIMARY:
                # Label assembly copied verbatim from cyp_split_match_board.py's drc+screen arm,
                # so the only thing varying across this experiment is how much training data.
                lab = labs[c]
                y = frame[c].to_numpy(float)
                kind = lab["kind"].to_numpy()
                use = np.zeros(len(frame), bool)
                use[keep_tr] = True
                drc = use & np.isfinite(y)
                ps = use & np.isin(kind, ("inactive", "weak", "active"))
                yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                idx = np.flatnonzero(drc | ps)
                if len(idx) < 50:
                    continue
                t0 = time.time()
                m = lgb.LGBMRegressor(
                    n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                    n_jobs=10, verbose=-1)
                m.fit(X[idx], yy[idx], sample_weight=w[idx])
                p = m.predict(X[va])
                ok = np.isfinite(y[va])
                if ok.sum() < 30:
                    continue
                yv, pv = y[va][ok], p[ok]
                rho = float(spearmanr(pv, yv).statistic)
                s_y = float(yv.std(ddof=1))
                # n is the number of real DRC labels actually trained on -- the quantity a data
                # purchase would increase. Pseudo-labels are counted separately.
                n_drc = int(drc.sum())
                rows.append(dict(seed=seed, iso=c, frac=f, n_drc=n_drc, n_total=int(len(idx)),
                                 rho=rho, s_y=s_y, sigma=sigma_star(s_y, rho),
                                 secs=round(time.time() - t0, 1)))
            print(f"   frac {f:.2f}: " + "  ".join(
                f"{short(c)} n={r['n_drc']} rho {r['rho']:.4f} sig* {r['sigma']:.3f}"
                for c in PRIMARY
                for r in rows if r["seed"] == seed and r["frac"] == f and r["iso"] == c),
                flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== learning curve: sigma* against real DRC labels trained on ===")
    print(f"{'iso':8} " + " ".join(f"{int(f*100):>7}%" for f in FRACTIONS))
    for c in PRIMARY:
        line = f"{short(c):8} "
        for f in FRACTIONS:
            v = [r["sigma"] for r in rows if r["iso"] == c and r["frac"] == f]
            line += f"{np.mean(v):8.3f}" if v else f"{'-':>8}"
        print(line)
    print(f"{'n_drc':8} " + " ".join(
        f"{int(np.mean([r['n_drc'] for r in rows if r['frac'] == f and r['iso'] == PRIMARY[0]])):>8}"
        for f in FRACTIONS))

    print("\n=== power-law fit  log sigma* = a - alpha log n,  and the data factor needed ===")
    print(f"{'iso':8} {'alpha':>7} {'R2':>6} {'n range':>14} {'target cut':>11} "
          f"{'n factor':>10} {'extra labels':>13}")
    summary = {}
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c and r["n_drc"] > 0 and np.isfinite(r["sigma"])]
        if len(rs) < 6:
            continue
        ln = np.log([r["n_drc"] for r in rs])
        ls = np.log([r["sigma"] for r in rs])
        A = np.vstack([np.ones_like(ln), -ln]).T
        coef, *_ = np.linalg.lstsq(A, ls, rcond=None)
        pred = A @ coef
        r2 = 1.0 - np.sum((ls - pred) ** 2) / np.sum((ls - ls.mean()) ** 2)
        alpha = float(coef[1])
        cut = TARGET_CUT.get(short(c))
        n_now = max(r["n_drc"] for r in rs)
        if cut is None or alpha <= 0:
            print(f"{short(c):8} {alpha:7.4f} {r2:6.2f} {f'{min(r[chr(110)+chr(95)+chr(100)+chr(114)+chr(99)] for r in rs)}-{n_now}':>14} "
                  f"{'n/a':>11} {'n/a':>10} {'n/a':>13}"
                  + ("   <- alpha <= 0: more data does not help here" if alpha <= 0 else ""))
            continue
        factor = float((1.0 - cut) ** (-1.0 / alpha))
        summary[short(c)] = dict(alpha=alpha, r2=float(r2), factor=factor, n_now=n_now)
        print(f"{short(c):8} {alpha:7.4f} {r2:6.2f} "
              f"{f'{min(r[chr(110)+chr(95)+chr(100)+chr(114)+chr(99)] for r in rs)}-{n_now}':>14} "
              f"{cut*100:10.0f}% {factor:10.1f}x {int(n_now*(factor-1)):13,}")

    if summary:
        fs = [v["factor"] for v in summary.values()]
        print(f"\n  required data factor: {min(fs):.1f}x to {max(fs):.1f}x "
              f"(median {np.median(fs):.1f}x)")
        print(f"  measured over only a {max(FRACTIONS)/min(FRACTIONS):.1f}x range of n, so any "
              "factor past ~10x is")
        print("  an extrapolation well outside the data and should be read as 'more than we can see'.")
        print("\n  UPPER BOUND, not an estimate: this is the return to IN-DISTRIBUTION data.")
        print("  External sources are a different assay and a different chemical distribution --")
        print("  10.5k Veith inactives made this model WORSE on 3/3 seeds. So a required factor")
        print("  that is already implausible in-distribution is decisive against the data lever.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
