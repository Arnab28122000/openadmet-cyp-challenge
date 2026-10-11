"""Re-measure the ensemble lever properly: same compounds, and in the test's similarity regime.

Two defects in how the ensemble's value has been measured, both fixed here.

**1. It was never actually paired.** `cyp_validation_lift.py` scores the production net on its
own saved validation predictions and compares against a `LGBM` dict of constants carried over
from a different run on a different split. Its docstring claims the two are "scored on the SAME
compounds"; they are not. Here the GBM is trained on the exact complement of the net's saved
`va_index` for each seed and scored on exactly those indices, so the difference is a paired
one.

**2. It was measured in the wrong regime.** Our validation is 65% in NN 0.3-0.5, a band that is
4.8% of the blinded test set, while 85% of the test sits in 0.5-0.7. `matched_spearman`
re-weights to the measured test histogram. On the docking arms that correction was worth about
+0.009 and flipped the sign, so a lever recorded at +0.0154 "1-2 se" is exactly the kind that
could change character under it.

The ensemble is the **largest** of the closed levers, which is why it is the one re-opened
first. This does not re-open it as a *decision* -- the production net already exists and is
what generated the board predictions -- it re-opens the **number**, because that number is
what every later trade-off was priced against.

    python scripts/cyp_ensemble_matched.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.matched import matched_spearman
from drugrx.cyp.screen import allr_floor, screen_labels


def paired_boot(pa, pb, y, draws: int = 2000, seed: int = 7):
    """95% CI for rho(a) - rho(b), resampling COMPOUNDS with replacement, paired."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n = len(y)
    d = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 3:
            continue
        ra = spearmanr(pa[i], y[i]).statistic
        rb = spearmanr(pb[i], y[i]).statistic
        if np.isfinite(ra) and np.isfinite(rb):
            d.append(ra - rb)
    if not d:
        return (float("nan"),) * 3
    d = np.array(d)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

NETS = ROOT / "data/cyp_cache/final_v4screen/nets"
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
BOARD = {"CYP1A2": 0.7597, "CYP2C9": 0.8084, "CYP2D6": 0.5004, "CYP3A4": 0.7685}


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
            for s in smiles]


def nn_sim(query, ref):
    from rdkit import DataStructs
    ref = [f for f in ref if f is not None]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else 0.0
                     for f in query])


def main() -> int:
    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    n = len(frame)
    labs, floor = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    allfp = fps(smiles)

    out = {c: {"net": [], "gbm": [], "net_m": [], "gbm_m": []} for c in PRIMARY}
    boot: dict[str, list] = {}
    for s in (0, 1, 2):
        f = NETS / f"final-net_s{s}.npz"
        if not f.exists():
            print(f"[ens] missing {f}"); continue
        z = np.load(f, allow_pickle=True)
        val, idx = z["val"], z["va_index"]
        tr = np.setdiff1d(np.arange(n), idx)          # the net's own training fold
        sim = nn_sim([allfp[i] for i in idx], [allfp[i] for i in tr])
        print(f"\n[ens] seed {s}: {len(tr)} train, {len(idx)} validation, "
              f"median NN {np.median(sim):.3f}", flush=True)

        for k, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            kind = labs[c]["kind"].to_numpy()
            yy = np.where(np.isfinite(y), y, labs[c]["y"].to_numpy(float))
            w0 = np.where(np.isfinite(y), 1.0, [PSEUDO_W.get(k_, 0.0) for k_ in kind])
            use = np.zeros(n, bool); use[tr] = True
            fit = np.flatnonzero(use & (np.isfinite(y) |
                                        np.isin(kind, ("inactive", "weak", "active"))))

            m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.3, reg_lambda=1.0, random_state=s,
                                  n_jobs=10, verbose=-1)
            m.fit(X[fit], yy[fit], sample_weight=w0[fit])
            pg = m.predict(X[idx])
            pn = val[:, k]

            ok = np.isfinite(y[idx])
            if ok.sum() < 30:
                continue
            yv, sv = y[idx][ok], sim[ok]
            rn = float(spearmanr(pn[ok], yv).statistic)
            rg = float(spearmanr(pg[ok], yv).statistic)
            mn, _, ess = matched_spearman(pn[ok], yv, sv)
            mg, _, _ = matched_spearman(pg[ok], yv, sv)
            # The three "seeds" share ONE validation split (va_index is byte-identical
            # across s0/s1/s2), so the spread across seeds measures network initialisation
            # and nothing else. The uncertainty that actually dominates is which compounds
            # are in the split, and a paired bootstrap over compounds is the only honest
            # handle on it available without retraining the net on new splits.
            boot.setdefault(c, []).append(paired_boot(pn[ok], pg[ok], yv))
            out[c]["net"].append(rn); out[c]["gbm"].append(rg)
            out[c]["net_m"].append(mn); out[c]["gbm_m"].append(mg)
            print(f"   {c[:6]:8} net {rn:.4f} / gbm {rg:.4f}  (d {rn-rg:+.4f})   "
                  f"matched net {mn:.4f} / gbm {mg:.4f}  (d {mn-mg:+.4f})  ess {ess}",
                  flush=True)

    print(f"\n{'iso':8} {'d rho (net-gbm)':>17} {'d rho MATCHED':>16} {'board':>8}")
    print("-" * 54)
    macro_o, macro_m = [], []
    for c in PRIMARY:
        if not out[c]["net"]:
            continue
        d = np.array(out[c]["net"]) - np.array(out[c]["gbm"])
        dm = np.array(out[c]["net_m"]) - np.array(out[c]["gbm_m"])
        macro_o.append(d.mean()); macro_m.append(dm.mean())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        sem = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else float("nan")
        bm, blo, bhi = boot[c][0] if c in boot else (float("nan"),)*3
        crosses = "" if (blo > 0 or bhi < 0) else "   CI crosses 0"
        print(f"{c[:6]:8} {d.mean():+10.4f}+-{se:.4f} {dm.mean():+9.4f}+-{sem:.4f} "
              f"{BOARD[c[:6]]:8.4f}   boot95 [{blo:+.4f}, {bhi:+.4f}]{crosses}")
    print(f"{'MACRO':8} {np.mean(macro_o):+10.4f}        {np.mean(macro_m):+9.4f}")

    (ROOT / "data/cyp_cache/ensemble_matched.json").write_text(
        json.dumps(out, indent=1, default=float))
    print("\nThe ensemble was recorded as +0.0154 macro on the uncorrected metric. What")
    print("matters is the MATCHED column -- that is the regime the board scores in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
