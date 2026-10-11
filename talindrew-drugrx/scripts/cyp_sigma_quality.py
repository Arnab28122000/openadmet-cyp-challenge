"""Does our per-item sigma-hat track actual error at all? The Borda null depends on the answer.

`cyp_borda_decode.py` found the expected-rank decode worth -0.0003 macro rho, and the ceiling over
ALL functions of (p, sigma-hat) negative at every resolution. That bounds the 2-D family -- but
only for the sigma-hat it was given, which was the **raw spread of an 8-member ensemble**: nu ~ 7
replicates per item, which the variance-estimation literature says is squarely in the hazard zone
(E[1/sigma-hat^2] has no mean for nu <= 2, no variance for nu <= 4; estimated-IVW is 1.2-5x worse
than not weighting, and the damage grows with the number of items).

So there are two very different readings of that null and they call for opposite conclusions:

  (a) heteroscedasticity carries no rank information here -> the lever is dead, done;
  (b) the ensemble spread is a poor estimate of predictive uncertainty -> the null is about our
      sigma-hat, not about the idea, and a fitted variance function could revive it.

This script separates them with the only question that matters: **does sigma-hat predict |residual|?**
If it does not, (b) is live and the Borda null is uninformative about the lever. If it does, and
Borda still fails, (a) is established.

Three estimators, scored by Spearman against |OOF residual| on held-out compounds:

* **ens**     -- raw 8-member ensemble spread. What the Borda run used.
* **varfn**   -- a fitted variance function: a GBM trained to predict squared OOF residual from
                 the same features, which Carroll-Wu-Ruppert say is asymptotically free where
                 per-item estimates are first-order harmful.
* **const**   -- a constant. The null reference; anything not beating this carries nothing.

    python scripts/cyp_sigma_quality.py --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_borda_decode import N_MEMBERS, borda  # noqa: E402
from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/sigma_quality.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[sigma] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[sigma] seed {seed}", flush=True)
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

            # Ensemble on the validation fold -> p and the raw spread.
            preds, oof_parts = [], []
            for m in range(N_MEMBERS):
                bs = rng.choice(idx, size=len(idx), replace=True)
                mdl = gbm(seed * 100 + m).fit(X[bs], yy[bs], sample_weight=w[bs])
                preds.append(mdl.predict(X[va])[okv])
            p = np.vstack(preds).mean(axis=0)
            s_ens = np.vstack(preds).std(axis=0, ddof=1)
            yv = y[va][okv]
            resid = np.abs(yv - p)

            # Variance FUNCTION: 5-fold OOF squared residuals on TRAIN, then a GBM on them.
            # Fitted entirely inside the training fold, so nothing about the scored compounds
            # enters it -- the comparison stays honest.
            itr = idx[np.isfinite(yy[idx])]
            fold = rng.integers(0, 5, size=len(itr))
            oof = np.empty(len(itr), float)
            for k in range(5):
                a, b = fold != k, fold == k
                if b.sum() == 0 or a.sum() < 30:
                    oof[b] = yy[itr][b].mean() if b.sum() else 0.0
                    continue
                oof[b] = gbm(seed).fit(X[itr][a], yy[itr][a],
                                       sample_weight=w[itr][a]).predict(X[itr][b])
            sq = (yy[itr] - oof) ** 2
            vf = gbm(seed + 7).fit(X[itr], np.log1p(sq))
            s_fit = np.sqrt(np.maximum(np.expm1(vf.predict(X[va])[okv]), 1e-6))

            r_ens = float(spearmanr(s_ens, resid).statistic)
            r_fit = float(spearmanr(s_fit, resid).statistic)
            rho_raw = float(spearmanr(p, yv).statistic)
            rho_b_ens = float(spearmanr(borda(p, s_ens), yv).statistic)
            rho_b_fit = float(spearmanr(borda(p, s_fit), yv).statistic)
            rec = dict(seed=seed, iso=c, n=int(okv.sum()),
                       sp_ens_vs_abserr=r_ens, sp_fit_vs_abserr=r_fit,
                       cv_ens=float(s_ens.std(ddof=1) / s_ens.mean()),
                       cv_fit=float(s_fit.std(ddof=1) / max(s_fit.mean(), 1e-9)),
                       rho_raw=rho_raw, d_borda_ens=rho_b_ens - rho_raw,
                       d_borda_fit=rho_b_fit - rho_raw, secs=round(time.time() - t0, 1))
            rows.append(rec)
            print(f"   {short(c)}: sigma-vs-|err| ens {r_ens:+.3f}  varfn {r_fit:+.3f}   "
                  f"borda d: ens {rec['d_borda_ens']:+.4f}  varfn {rec['d_borda_fit']:+.4f}",
                  flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1

    print("\n" + "=" * 88)
    print("=== Does sigma-hat predict |residual|? (Spearman; 0 means it carries nothing) ===")
    print(f"{'iso':8} {'ens':>9} {'varfn':>9} {'cv ens':>8} {'cv fit':>8} "
          f"{'borda d ens':>12} {'borda d varfn':>14}")
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if not rs:
            continue
        print(f"{short(c):8} {np.mean([r['sp_ens_vs_abserr'] for r in rs]):+9.3f} "
              f"{np.mean([r['sp_fit_vs_abserr'] for r in rs]):+9.3f} "
              f"{np.mean([r['cv_ens'] for r in rs]):8.2f} "
              f"{np.mean([r['cv_fit'] for r in rs]):8.2f} "
              f"{np.mean([r['d_borda_ens'] for r in rs]):+12.4f} "
              f"{np.mean([r['d_borda_fit'] for r in rs]):+14.4f}")
    e = np.array([r["sp_ens_vs_abserr"] for r in rows])
    f = np.array([r["sp_fit_vs_abserr"] for r in rows])
    de = np.array([r["d_borda_ens"] for r in rows])
    df = np.array([r["d_borda_fit"] for r in rows])
    print(f"{'MACRO':8} {e.mean():+9.3f} {f.mean():+9.3f} {'':8} {'':8} "
          f"{de.mean():+12.4f} {df.mean():+14.4f}")
    print(f"\n  ensemble spread vs |error|: mean {e.mean():+.3f}, positive on {sum(e>0)}/{len(e)}")
    print(f"  variance function vs |error|: mean {f.mean():+.3f}, positive on {sum(f>0)}/{len(f)}")
    print("\n  Reading: if BOTH are near zero, heteroscedasticity in our predictions is not")
    print("  recoverable from the features, and the Borda null is a verdict on the LEVER.")
    print("  If the variance function is clearly better AND its Borda delta is still ~0, the")
    print("  lever is dead for a stronger reason. If varfn is better and Borda improves, the")
    print("  original null was an artefact of estimating sigma from 8 replicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
