"""Treat "screen-inactive" as LEFT-CENSORED (y <= 2.211) instead of exact (y = 2.211). Tobit EM.

`findings/06` established that the screen pseudo-label point mass is causal: `screen_labels` pins
every screen-inactive compound at the constant 2.211 with `PSEUDO_W["inactive"] = 1.0`, that is
61% of CYP2D6's labelled signal, and it makes the ensemble members spuriously AGREE where it
dominates. Dropping the mass costs CYP2D6 **-0.0516 rho**, so the information is real. The defect
is not the information, it is the false precision: a screen says the compound is inactive, which
means `y <= 2.211`, not `y = 2.211`.

That is a textbook left-censored regression, so use the textbook estimator. Under a Gaussian
latent model the EM step for a censored observation is its conditional expectation,

    E[y* | y* <= c]  =  mu - sigma * phi(alpha) / Phi(alpha),      alpha = (c - mu) / sigma

the inverse Mills ratio. When the model already believes the compound is well below threshold
(mu << c) the imputation returns mu and the row stops fighting the model; when the model wants to
put it above threshold (mu >> c) the imputation pulls it back to c and the row still carries its
constraint. Either way the 2648 rows inform the low end WITHOUT being pinned to one value.

Three arms, identical except in how the inactive rows enter:

* **full**   -- y = 2.211, weight 1.0. The status quo.
* **down**   -- y = 2.211, weight 0.2. Carried over from `cyp_sigma_anomaly.py`, where it gained
                rho on all four isoforms but only at t = 1.92, 10/16 -- short of our bar. The extra
                seeds here settle it rather than leaving a loose candidate lying around.
* **tobit**  -- y = the EM imputation above, weight 1.0, `--em-iters` passes.

PRE-REGISTERED, written before the run:

1. **The artefact must be repaired.** r(sigma, p) on CYP2D6 is +0.324 under `full` and positive on
   4/4 seeds -- the only such cell in the 4x3 grid. Under `tobit` it must fall to <= 0. If the
   point-mass diagnosis in finding 06 is right this is nearly forced; if it does not happen,
   finding 06 is wrong and says so.
2. **The deployment bar: macro drho >= +0.005 AND positive on >= 5 of 6 seeds.** Below that it is
   a correctness fix with no board value and will not be submitted, however clean the theory.
3. **Expected to be small, and said so first.** The floor mass is largest where our headroom is
   smallest (CYP2D6: 61% floor, rank 28, needs 5.4%) and smallest where headroom is largest
   (CYP3A4: 2% floor, rank 134, needs 24.7%). This method helps the isoform we least need helped.
   Priced in places, not in metric units (`cyp-challenge-leaderboard-ui`).

Note the `down`/`none` arms of finding 06 do NOT bound this one: they change the point mass's
WEIGHT or presence, while this changes its TARGET. Nothing measured so far constrains the result.

    python scripts/cyp_censored_floor.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_borda_decode import N_MEMBERS  # noqa: E402
from cyp_decide_vs_affine import PSEUDO_W, gbm, short  # noqa: E402
from cyp_sigma_anomaly import partial_spearman  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

#: The value screen_labels assigns every inactive compound. Read, not assumed -- asserted below.
FLOOR = 2.211
ARMS = ("full", "down", "tobit")


def mills(alpha: np.ndarray) -> np.ndarray:
    """phi(a)/Phi(a), computed in logs so the far-left tail does not underflow to 0/0."""
    return np.exp(norm.logpdf(alpha) - log_ndtr(alpha))


def tobit_impute(mu: np.ndarray, sigma: float, c: float = FLOOR) -> np.ndarray:
    """E[y* | y* <= c] under y* ~ N(mu, sigma^2). Never returns a value above c."""
    sigma = max(float(sigma), 1e-3)
    a = (c - mu) / sigma
    return np.minimum(mu - sigma * mills(a), c)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--em-iters", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/censored_floor.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[cens] features {X.shape}  EM iters {args.em_iters}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[cens] seed {seed}", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            base = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            idx = np.flatnonzero(drc | ps)
            okv = np.isfinite(y[va])
            yv = y[va][okv]
            if len(idx) < 50 or okv.sum() < 60:
                continue
            inact = (kind == "inactive") & ps
            assert np.allclose(base[inact], FLOOR, atol=1e-3), "floor value moved; re-read it"

            for arm in ARMS:
                t0 = time.time()
                yy = base.copy()
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                if arm == "down":
                    w = np.where(inact, 0.2, w)
                elif arm == "tobit":
                    # EM: fit, impute the censored rows from the current fit, refit.
                    ii = np.flatnonzero(inact)
                    dd = np.flatnonzero(drc)
                    for _ in range(args.em_iters):
                        m = gbm(seed).fit(X[idx], yy[idx], sample_weight=w[idx])
                        sig = float(np.std(y[dd] - m.predict(X[dd]), ddof=1)) if len(dd) > 5 else 1.0
                        yy[ii] = tobit_impute(m.predict(X[ii]), sig)

                rng = np.random.default_rng(1000 + seed)
                preds = []
                for mi in range(N_MEMBERS):
                    bs = rng.choice(idx, size=len(idx), replace=True)
                    preds.append(gbm(seed * 100 + mi).fit(
                        X[bs], yy[bs], sample_weight=w[bs]).predict(X[va])[okv])
                P = np.vstack(preds)
                p, s = P.mean(axis=0), P.std(axis=0, ddof=1)
                err = np.abs(yv - p)
                rec = dict(seed=seed, iso=c, arm=arm, n=int(okv.sum()),
                           rho=float(spearmanr(p, yv).statistic),
                           r_sig_p=float(spearmanr(s, p).statistic),
                           r_sig_err=float(spearmanr(s, err).statistic),
                           r_partial=partial_spearman(s, err, p),
                           imputed_mean=float(yy[inact].mean()),
                           imputed_sd=float(yy[inact].std(ddof=1)) if inact.sum() > 1 else 0.0,
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {short(c)} {arm:5}: rho {rec['rho']:.4f}  r(sig,p) {rec['r_sig_p']:+.3f}"
                      f"  r(sig,|err|) {rec['r_sig_err']:+.3f}  imputed {rec['imputed_mean']:.3f}"
                      f" +/- {rec['imputed_sd']:.3f}", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1

    seeds = sorted({r["seed"] for r in rows})

    def cell(iso, arm, seed, key):
        v = [r[key] for r in rows if r["iso"] == iso and r["arm"] == arm and r["seed"] == seed]
        return v[0] if v else np.nan

    def paired(arm, key="rho"):
        """Per-isoform (mean delta vs full, t, wins) and the pooled macro delta."""
        out, parts = {}, []
        for iso in PRIMARY:
            d = np.array([cell(iso, arm, s, key) - cell(iso, "full", s, key) for s in seeds])
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std(ddof=1) else 0.0
            out[iso] = (d.mean(), t, int((d > 0).sum()), len(d))
            parts.append(d)
        macro = np.array([np.mean([x[i] for x in parts]) for i in range(len(parts[0]))])
        return out, macro

    print("\n" + "=" * 86)
    print("=== CLAIM 1 (pre-registered): does the Tobit arm repair the artefact? ===")
    print(f"{'iso':8} {'r(sig,p) full':>14} {'r(sig,p) tobit':>15} {'pos seeds full':>15} "
          f"{'pos seeds tobit':>16}")
    for iso in PRIMARY:
        f = np.array([cell(iso, "full", s, "r_sig_p") for s in seeds])
        tb = np.array([cell(iso, "tobit", s, "r_sig_p") for s in seeds])
        print(f"{short(iso):8} {np.nanmean(f):14.3f} {np.nanmean(tb):15.3f} "
              f"{int(np.nansum(f > 0)):10d}/{len(seeds):<4d} {int(np.nansum(tb > 0)):11d}/{len(seeds):<4d}")
    print("  PASS if CYP2D6 falls from ~+0.32 (positive on every seed) to <= 0.")
    print("  A FAIL here contradicts finding 06 and must be reported as such.")

    print("\n=== CLAIM 2 (pre-registered): the deployment bar, macro drho >= +0.005 and >= 5/6 ===")
    for arm in ("down", "tobit"):
        per, macro = paired(arm)
        print(f"\n  arm = {arm}   (delta vs full, paired by seed)")
        print(f"  {'iso':8} {'d rho':>9} {'t':>7} {'wins':>7}")
        for iso, (m, t, wins, n) in per.items():
            print(f"  {short(iso):8} {m:+9.4f} {t:7.2f} {wins:3d}/{n:<3d}")
        mt = macro.mean() / (macro.std(ddof=1) / np.sqrt(len(macro))) if macro.std(ddof=1) else 0.0
        verdict = ("DEPLOY" if macro.mean() >= 0.005 and (macro > 0).sum() >= 5
                   else "BELOW BAR -- do not submit")
        print(f"  {'MACRO':8} {macro.mean():+9.4f} {mt:7.2f} {int((macro>0).sum()):3d}/{len(macro):<3d}"
              f"   -> {verdict}")
    print("\n  The bar was fixed before the run. A near miss is a miss")
    print("  (feedback-a-magnitude-bar-is-not-a-significance-bar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
