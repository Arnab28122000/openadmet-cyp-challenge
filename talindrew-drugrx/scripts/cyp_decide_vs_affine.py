"""Does the metric's own optimal point beat the affine placement we actually submit?

`drugrx/cyp/decide.py` computes the prediction that minimises expected ST-RAE: the balance point
where `P(lo > p) == P(hi < p)`. It is implemented, verified exact against a 4,001-point grid on
400 random cases -- and **nothing imports it.** Every submission v4-v7 instead uses
`drugrx/cyp/placement.py`, a monotone affine map per isoform, chosen that way because a monotone
map cannot disturb rho or tau.

That constraint may be self-imposed. **The board sorts by ST-RAE alone**
(`SORT_REGRESSION_LEADERBOARD_BY = "ST-RAE"`), so rho and tau are reported, not ranked on. If a
non-monotone, per-compound rule lowers ST-RAE further, it is better for rank even though it costs
rho -- and the affine map is only the best *linear* approximation to the per-compound optimum.

The magnitude is right for the gap this programme cannot otherwise close: rank 30 sits at ST-RAE
0.4737 against our 0.5925, so ~0.12 of ST-RAE is worth roughly 30 board places, where every
*model* lever ever measured here was worth ~1% of absolute accuracy.

### The comparison is deliberately rigged AGAINST the balance point

| arm | what it is | information it uses |
|---|---|---|
| `raw` | the model's predictions, unplaced | train only |
| `affine_nested` | centre/spread fitted on OOF training predictions | train only -- a fair comparison |
| `affine_oracle` | centre/spread fitted on the VALIDATION labels being scored | **validation labels** -- cheating, on purpose |
| `decide` | per-compound balance point | train only |

`affine_oracle` is the strongest version of the incumbent that could possibly exist: it is handed
the answers. `decide` is given only training information. A win against `affine_nested` is the
honest result; a win against `affine_oracle` would mean the affine *form* is the binding
constraint rather than how it is tuned.

rho is reported for every arm so the ranking cost is visible rather than hidden.

### The predictive distribution, and its one weak point

The balance point needs a distribution over `y`, not a point. This uses a residual bootstrap:
5-fold out-of-fold residuals on the training set, resampled onto each validation prediction. That
is honest and cheap, but it is **homoscedastic** -- every compound gets the same residual spread,
when in truth a likely-inactive compound should carry a wider, floor-shaped belief. `decide.py`'s
own docstring is about exactly that mixture case (`mixture_draws`), so this version understates
what the rule can do. Read a positive result as a lower bound and a null as "not with this
posterior", not as "the theory is wrong".

    python scripts/cyp_decide_vs_affine.py --seeds 0 1 2 3
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_per_endpoint_decode import bound_lookup  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.decide import decide_many  # noqa: E402
from drugrx.cyp.metric import st_rae  # noqa: E402
from drugrx.cyp.screen import allr_floor, screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
ARMS = ("raw", "affine_nested", "affine_oracle", "decide", "affine_then_decide")
N_DRAWS = 200
CENTRES = np.arange(2.6, 6.21, 0.04)
SPREADS = np.arange(0.20, 1.61, 0.02)


def short(c: str) -> str:
    return c.split("_")[0]


def gbm(seed: int, n_estimators: int = 800):
    import lightgbm as lgb
    return lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
        reg_lambda=1.0, random_state=seed, n_jobs=10, verbose=-1)


def affine(p: np.ndarray, centre: float, spread: float) -> np.ndarray:
    """centre + spread * z(p). Monotone in p for spread > 0, so rho is preserved."""
    sd = p.std()
    z = (p - p.mean()) / (sd if sd > 0 else 1.0)
    return centre + spread * z


def best_affine(p: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray
                ) -> tuple[float, float]:
    """Grid-search the (centre, spread) minimising ST-RAE against the supplied labels."""
    sd = p.std()
    z = (p - p.mean()) / (sd if sd > 0 else 1.0)
    base = np.clip(lo - y.mean(), 0, None).sum() + np.clip(y.mean() - hi, 0, None).sum()
    if base <= 0:
        return float(y.mean()), 1.0
    best, arg = np.inf, (float(y.mean()), 1.0)
    for s in SPREADS:
        q = s * z
        for c in CENTRES:
            pp = c + q
            err = np.clip(lo - pp, 0, None).sum() + np.clip(pp - hi, 0, None).sum()
            if err < best:
                best, arg = err, (float(c), float(s))
    return arg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/decide_vs_affine.json"))
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.model_selection import KFold

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, _ = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[decide] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        rng = np.random.default_rng(seed)
        print(f"\n[decide] seed {seed}: {len(tr)} train / {len(va)} val", flush=True)

        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            lo = frame[f"{c}_conf_low"].to_numpy(float)
            hi = frame[f"{c}_conf_high"].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            ok_va = np.isfinite(y[va]) & np.isfinite(lo[va]) & np.isfinite(hi[va])
            if len(idx) < 50 or ok_va.sum() < 30:
                continue
            t0 = time.time()

            m = gbm(seed).fit(X[idx], yy[idx], sample_weight=w[idx])
            p_va = m.predict(X[va])[ok_va]
            yv, lv, hv = y[va][ok_va], lo[va][ok_va], hi[va][ok_va]

            # OOF residuals on the TRAINING labels -> the predictive distribution
            tr_drc = np.flatnonzero(drc)
            resid = []
            for a, b in KFold(n_splits=5, shuffle=True, random_state=seed).split(idx):
                sub = idx[a]
                mm = gbm(seed).fit(X[sub], yy[sub], sample_weight=w[sub])
                held = np.intersect1d(idx[b], tr_drc)
                if len(held):
                    resid.append(y[held] - mm.predict(X[held]))
            resid = np.concatenate(resid) if resid else np.zeros(1)
            draws = p_va[:, None] + rng.choice(resid, size=(len(p_va), N_DRAWS), replace=True)

            bounds = bound_lookup(y[tr_drc], lo[tr_drc], hi[tr_drc])
            p_dec = decide_many(draws, bounds)

            # affine fitted WITHOUT validation labels: on the OOF training predictions
            oof_p, oof_y, oof_lo, oof_hi = [], [], [], []
            for a, b in KFold(n_splits=5, shuffle=True, random_state=seed + 7).split(idx):
                sub = idx[a]
                mm = gbm(seed).fit(X[sub], yy[sub], sample_weight=w[sub])
                held = np.intersect1d(idx[b], tr_drc)
                if len(held):
                    oof_p.append(mm.predict(X[held])); oof_y.append(y[held])
                    oof_lo.append(lo[held]); oof_hi.append(hi[held])
            cn, sn = best_affine(np.concatenate(oof_p), np.concatenate(oof_y),
                                 np.concatenate(oof_lo), np.concatenate(oof_hi))
            co, so = best_affine(p_va, yv, lv, hv)

            # The composition the per-isoform pattern implies: the balance point is a
            # DECISION RULE, not a calibration, so it inherits whatever mis-scaling the draws
            # carry. Calibrate first, then decide on the calibrated predictive distribution --
            # whose spread must be re-estimated from OOF residuals AFTER the same affine map,
            # not reused from the raw scale.
            oof_pc, oof_yc = np.concatenate(oof_p), np.concatenate(oof_y)
            resid_cal = oof_yc - affine(oof_pc, cn, sn)
            draws_cal = (affine(p_va, cn, sn)[:, None]
                         + rng.choice(resid_cal, size=(len(p_va), N_DRAWS), replace=True))
            p_dec_cal = decide_many(draws_cal, bounds)

            preds = {"raw": p_va, "affine_nested": affine(p_va, cn, sn),
                     "affine_oracle": affine(p_va, co, so), "decide": p_dec,
                     "affine_then_decide": p_dec_cal}
            rec = dict(seed=seed, iso=c, n=int(ok_va.sum()),
                       centre_nested=cn, spread_nested=sn, centre_oracle=co, spread_oracle=so,
                       resid_sd=float(resid.std()), secs=round(time.time() - t0, 1))
            for a, pp in preds.items():
                rec[f"strae_{a}"] = float(st_rae(pp, yv, lv, hv))
                rec[f"rho_{a}"] = float(spearmanr(pp, yv).statistic)
            rows.append(rec)
            print("   " + f"{short(c)}: n={rec['n']:4d}  " + "  ".join(
                f"{a} {rec[f'strae_{a}']:.4f}/r{rec[f'rho_{a}']:.3f}" for a in ARMS)
                + f"  [{time.time()-t0:.0f}s]", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== ST-RAE per arm, mean over seeds (LOWER is better) ===")
    print(f"{'iso':8} " + " ".join(f"{a:>15}" for a in ARMS))
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if rs:
            print(f"{short(c):8} " + " ".join(
                f"{np.mean([r[f'strae_{a}'] for r in rs]):15.4f}" for a in ARMS))
    macro = {a: np.mean([np.mean([r[f"strae_{a}"] for r in rows if r["iso"] == c])
                         for c in PRIMARY if any(r["iso"] == c for r in rows)]) for a in ARMS}
    print(f"{'MACRO':8} " + " ".join(f"{macro[a]:15.4f}" for a in ARMS))

    print("\n=== rho per arm (the ranking cost of breaking monotonicity) ===")
    print(f"{'iso':8} " + " ".join(f"{a:>15}" for a in ARMS))
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if rs:
            print(f"{short(c):8} " + " ".join(
                f"{np.mean([r[f'rho_{a}'] for r in rs]):15.4f}" for a in ARMS))

    print("\n=== paired, per seed: decide minus each affine arm (NEGATIVE favours decide) ===")
    seeds = sorted({r["seed"] for r in rows})
    for cand, ref in (("decide", "affine_nested"), ("decide", "affine_oracle"),
                      ("affine_then_decide", "affine_nested"),
                      ("affine_then_decide", "affine_oracle")):
        per = []
        for s in seeds:
            d = [r[f"strae_{cand}"] - r[f"strae_{ref}"] for r in rows if r["seed"] == s]
            per.append(float(np.mean(d)))
        per = np.array(per)
        se = per.std(ddof=1) / np.sqrt(len(per)) if len(per) > 1 else float("nan")
        print(f"  {cand:18} - {ref:14}: {per.mean():+.4f}  se {se:.4f}  "
              f"t {per.mean()/se if se else float('nan'):+.2f}  "
              f"better on {int((per < 0).sum())}/{len(per)} seeds   per-seed "
              f"{np.array2string(per, precision=3, floatmode='fixed')}")
    print("\n  NOTE: 4 seeds estimate a sd on 3 df. Treat any t here as provisional and confirm")
    print("  on fresh seeds before acting -- a 3-seed t of 8.67 died at 12 seeds on 2026-10-09.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
