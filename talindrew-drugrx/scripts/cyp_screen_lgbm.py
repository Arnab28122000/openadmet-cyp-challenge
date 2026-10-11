"""Does training on screen-derived labels undo the DRC selection bias? CPU, LightGBM.

The question is directional and cheap, so it is asked with the stubborn baseline
(LightGBM on the incumbent's features) before any GPU time: three arms, the same
scaffold splits, scored two ways.

* **V1 — the programme's validation.** DRC-labelled held-out compounds only. Every one
  of them was *promoted*, so V1 cannot see over-prediction on unselected compounds; a
  fix for selection bias is expected to be neutral-to-slightly-worse here.
* **V2 — the population validation.** V1 plus every held-out compound whose 50 µM
  readout shows no inhibition (`kind == "inactive"`), labelled with the floor window.
  This is the population the blinded set is drawn from, for CYP2D6 especially.

Arms: ``drc`` (incumbent's data), ``drc+inactive`` (adds the floor labels),
``drc+screen`` (adds every screen-derived label, weak/active ones at half weight).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae
from drugrx.cyp.screen import allr_floor, screen_labels, summary
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"drc": (), "drc+inactive": ("inactive",), "drc+screen": ("inactive", "weak", "active")}
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/screen_lgbm.json"))
    args = ap.parse_args()

    data = load(workers=8)
    frame, test = data.frame.reset_index(drop=True), data.test
    labels = screen_labels(frame)
    floor = allr_floor(frame)
    print("[screen] label kinds per isoform:\n" + summary(labels).to_string())
    print(f"[screen] floor label: y={floor['y']:.2f} [{floor['lo']:.2f}, {floor['hi']:.2f}] "
          f"from n={floor['n']} DRC'd compounds with no inhibition at 50 uM")

    opts = dict(counts=True, desc_mode="rdkit2d", extra_fps=True)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)
    xt = featurise(test["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)

    results = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        for arm, kinds in ARMS.items():
            t0 = time.time()
            for c in PRIMARY:
                lab = labels[c]
                y = frame[c].to_numpy(float)
                lo = frame[f"{c}_conf_low"].to_numpy(float)
                hi = frame[f"{c}_conf_high"].to_numpy(float)
                kind = lab["kind"].to_numpy()
                # Training rows: DRC labels, plus the selected pseudo-label kinds.
                use = np.zeros(len(frame), bool)
                use[tr] = True
                drc = use & np.isfinite(y)
                ps = use & np.isin(kind, kinds)
                yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                rows = np.flatnonzero(drc | ps)
                m = lgb.LGBMRegressor(
                    n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                    n_jobs=10, verbose=-1)
                m.fit(x[rows], yy[rows], sample_weight=w[rows])
                pv, pt = m.predict(x[va]), m.predict(xt)
                # V1: DRC-labelled held-out compounds.
                v1 = np.isfinite(y[va])
                s1 = st_rae(pv[v1], y[va][v1], lo[va][v1], hi[va][v1])
                # V2: plus held-out compounds with no inhibition at 50 uM.
                inact = kind[va] == "inactive"
                yv = np.where(v1, y[va], floor["y"])
                lv = np.where(v1, lo[va], floor["lo"])
                hv = np.where(v1, hi[va], floor["hi"])
                k2 = v1 | inact
                s2 = st_rae(pv[k2], yv[k2], lv[k2], hv[k2])
                results.append({
                    "seed": seed, "arm": arm, "endpoint": c, "v1": s1, "v2": s2,
                    "n_v1": int(v1.sum()), "n_v2": int(k2.sum()), "n_train": int(len(rows)),
                    "test_mean": float(pt.mean()), "test_sd": float(pt.std()),
                    "test_frac_lt_3p6": float((pt < 3.6).mean()),
                    "inactive_val_mean": float(pv[inact].mean()) if inact.any() else None,
                    "metric_id": METRIC_ID})
            print(f"[lgbm] seed {seed} {arm:13s} done in {time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(results)
    Path(args.out).write_text(json.dumps(results, indent=1))
    pd.set_option("display.width", 200)
    print("\n=== per endpoint, mean over seeds ===")
    print(df.groupby(["endpoint", "arm"])[["v1", "v2", "test_mean", "test_frac_lt_3p6",
                                           "inactive_val_mean", "n_train"]].mean().round(4))
    mac = df.groupby(["seed", "arm"])[["v1", "v2"]].mean().reset_index()
    print("\n=== macro (mean over endpoints), per seed ===")
    print(mac.pivot(index="seed", columns="arm", values=["v1", "v2"]).round(4))
    base = mac[mac.arm == "drc"].set_index("seed")
    for arm in ARMS:
        if arm == "drc":
            continue
        a = mac[mac.arm == arm].set_index("seed")
        for v in ("v1", "v2"):
            d = (a[v] - base[v]).to_numpy()
            print(f"  {arm:13s} - drc  {v}: {d.mean():+.4f} (per seed {np.round(d, 4)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
