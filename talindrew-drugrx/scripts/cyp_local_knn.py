"""The test set is an analog expansion. Does a local model exploit that?

Two measurements point here together. The blinded test set sits at median nearest-neighbour
ECFP4 Tanimoto **0.581** to training with 96.9 % of it above 0.4, against 0.439 and 68 % for
our scaffold validation -- it is an analog expansion, and every test compound has a close
relative by construction. And the validation-to-board **lift** is +0.175 / +0.135 / +0.016 /
-0.005: on CYP1A2 and CYP2C9 most of our board Spearman is that easiness, not our model.

A global tree ensemble is not the natural model for that regime. When a query sits inside a
dense analog series, its neighbours' measured potencies are strong evidence that a model
averaging over the whole chemical space dilutes. So:

    gbm     the incumbent LightGBM on drc + screen labels
    knn     Tanimoto-weighted k-nearest-neighbour regression on dose-response labels alone
    blend   w*gbm + (1-w)*knn, w swept

and the diagnostic that matters more than the headline: **the same scores split by the
validation compound's own nearest-neighbour similarity**. If the blend helps, and the help is
concentrated in the high-similarity tercile, that generalises to a test set which is *entirely*
high-similarity -- and the lift says exactly how much it would be worth.

Reported on Spearman as well as ST-RAE, because placement is monotone and only a rho gain
survives it.

    python scripts/cyp_local_knn.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
#: board rho - validation rho, per isoform (scripts/cyp_validation_lift.py, 2026-10-09)
LIFT = {"CYP1A2": 0.1749, "CYP2C9": 0.1347, "CYP2D6": 0.0164, "CYP3A4": -0.0052}


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(gen.GetFingerprint(m) if m is not None else None)
    return out


def knn_predict(qfp, rfp, ry, k: int, power: float):
    """Similarity-weighted kNN. Weight = Tanimoto**power over the k nearest labelled refs."""
    from rdkit import DataStructs
    out = np.zeros(len(qfp))
    nn = np.zeros(len(qfp))
    for i, f in enumerate(qfp):
        if f is None:
            out[i] = ry.mean(); continue
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(f, rfp))
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        w = np.clip(s[top], 0, None) ** power
        out[i] = float(np.average(ry[top], weights=w)) if w.sum() > 0 else float(ry.mean())
        nn[i] = float(s.max())
    return out, nn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--power", type=float, default=4.0)
    ap.add_argument("--weights", type=float, nargs="*", default=[0.9, 0.8, 0.7, 0.6])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/local_knn.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    x = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    allfp = fps(smiles)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[knn] seed {seed}: {len(tr)} train / {len(va)} val", flush=True)

        for c in PRIMARY:
            iso = c[:6]
            y = frame[c].to_numpy(float)
            lo = frame[f"{c}_conf_low"].to_numpy(float)
            hi = frame[f"{c}_conf_high"].to_numpy(float)
            kind = labs[c]["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            v1 = np.isfinite(y[va])
            if v1.sum() < 40:
                continue

            # GBM: the incumbent arm.
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, labs[c]["y"].to_numpy(float))
            w0 = np.where(drc, 1.0, [PSEUDO_W.get(z, 0.0) for z in kind])
            idx = np.flatnonzero(drc | ps)
            t0 = time.time()
            m = lgb.LGBMRegressor(
                n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
                random_state=seed, n_jobs=10, verbose=-1)
            m.fit(x[idx], yy[idx], sample_weight=w0[idx])
            p_gbm = m.predict(x[va])

            # kNN on dose-response labels in the training fold only.
            ref = np.flatnonzero(drc)
            p_knn, nn = knn_predict([allfp[i] for i in va], [allfp[i] for i in ref],
                                    y[ref], args.k, args.power)

            inact = kind[va] == "inactive"
            yv = np.where(v1, y[va], floor["y"])
            lv = np.where(v1, lo[va], floor["lo"])
            hv = np.where(v1, hi[va], floor["hi"])
            k2 = v1 | inact

            def sc(p):
                return {"rho": float(spearmanr(p[v1], y[va][v1]).statistic),
                        "v2": float(st_rae(p[k2], yv[k2], lv[k2], hv[k2]))}

            row = {"seed": seed, "iso": iso, "n": int(v1.sum()),
                   "nn_med": float(np.median(nn)), "secs": round(time.time() - t0, 1),
                   "gbm": sc(p_gbm), "knn": sc(p_knn)}
            for w in args.weights:
                row[f"blend{w}"] = sc(w * p_gbm + (1 - w) * p_knn)

            # The diagnostic: the same contrast inside the test's own similarity regime.
            hi_m = v1 & (nn >= 0.50)
            if hi_m.sum() > 30:
                best_w = min(args.weights,
                             key=lambda w: abs(row[f"blend{w}"]["rho"] - row["gbm"]["rho"]) * 0)
                row["hi_nn"] = {
                    "n": int(hi_m.sum()),
                    "gbm_rho": float(spearmanr(p_gbm[hi_m], y[va][hi_m]).statistic),
                    "knn_rho": float(spearmanr(p_knn[hi_m], y[va][hi_m]).statistic),
                    **{f"blend{w}_rho": float(spearmanr(
                        (w * p_gbm + (1 - w) * p_knn)[hi_m], y[va][hi_m]).statistic)
                       for w in args.weights}}
            rows.append(row)
            print(f"   {iso}  gbm rho {row['gbm']['rho']:.4f}  knn rho {row['knn']['rho']:.4f}"
                  + "".join(f"  b{w} {row[f'blend{w}']['rho']:.4f}" for w in args.weights)
                  + f"   [{time.time()-t0:.0f}s]", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    arms = ["knn"] + [f"blend{w}" for w in args.weights]
    print(f"\n=== paired against the GBM, over seeds ===")
    print(f"{'iso':8} {'arm':10} {'d Spearman':>21} {'t':>6} | {'d V2':>21} {'t':>6}"
          f" | {'d rho, NN>=0.5':>15}")
    for c in PRIMARY:
        iso = c[:6]
        rs = [r for r in rows if r["iso"] == iso]
        if len(rs) < 2:
            continue
        for arm in arms:
            line = f"{iso:8} {arm:10}"
            for key in ("rho", "v2"):
                d = np.array([r[arm][key] - r["gbm"][key] for r in rs])
                se = d.std(ddof=1) / np.sqrt(len(d))
                line += f" {d.mean():+12.4f} +-{se:.4f} {d.mean()/se if se else 0:6.2f} |"
            hs = [r for r in rs if "hi_nn" in r]
            if hs and f"{arm}_rho" in hs[0]["hi_nn"]:
                dh = np.mean([r["hi_nn"][f"{arm}_rho"] - r["hi_nn"]["gbm_rho"] for r in hs])
                line += f" {dh:+15.4f}"
            print(line)
        print(f"{'':8} {'(lift ' + f'{LIFT[iso]:+.3f}, so a rho gain here moves the board 1:1)':<60}")
    print("\nA gain concentrated in the NN>=0.5 column is the one that matters: the blinded")
    print("test set is 96.9% above 0.4 and has a median of 0.581, so it is ALL that regime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
