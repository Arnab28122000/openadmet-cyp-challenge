"""Veith's inactives as left-censored bounds on the primary head. CPU, LightGBM, V1 and V2.

The programme already holds PubChem AID 1851 (Veith) as *binary auxiliary heads*, and recorded
the gain as a null. **That null was measured on V1** — the DRC-only validation surface, every row
of which is a promoted compound — so it could not see the thing an inactive row teaches. This
re-asks it on V2, where the question is whether the model has learned what a non-inhibitor looks
like.

A Veith 0 is not a pIC50; it is a statement that the compound did not inhibit up to ~57 µM. It
therefore enters as a **bound**, `y <= ACTIVE_CUT`, expressed as the interval `[FLOOR_LO,
ACTIVE_CUT]` that the metric already scores as free — never as a point estimate. (The published
reports of three higher-ranked entries agree on exactly this distinction: as a point pseudo-label
the same rows *cost* them score; as a censored bound they helped.)

**Only CYP1A2 and CYP2D6.** Measured on the 111 compounds we hold in common, a Veith 0 implies our
screen called the compound inactive on 26/30 (1A2) and 32/34 (2D6), but only 16/50 (2C9) and 3/30
(3A4) — Veith's assay does not resolve those two the same way. The counts are small, so this is a
weak check; it is used because it independently reproduces the isoform pair an earlier arm
comparison had already selected on different evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise

ROOT = Path(__file__).resolve().parents[1]
BORROW = ("CYP1A2", "CYP2D6")
ACTIVE_CUT = 4.0
ARMS = {"screen": 0.0, "screen+veith0": 0.3}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--external", default=str(ROOT / "data/cyp_cache/external.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/veith_floor.json"))
    args = ap.parse_args()

    data = load(workers=8)
    frame, test = data.frame.reset_index(drop=True), data.test.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)
    ext = pd.read_parquet(args.external)
    ext = ext[~ext.key.isin(set(frame.key))].reset_index(drop=True)

    opts = dict(counts=True, desc_mode="rdkit2d", extra_fps=True)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)
    xt = featurise(test["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)
    xe = featurise(ext["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)

    res = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        for arm, vw in ARMS.items():
            for c in PRIMARY:
                iso = c[:6]
                y = frame[c].to_numpy(float)
                kd = labs[c]["kind"].to_numpy()
                yy = np.where(np.isfinite(y), y, labs[c]["y"].to_numpy(float))
                w = np.where(np.isfinite(y), 1.0, np.where(kd == "inactive", 1.0, 0.5))
                keep = np.isfinite(y) | np.isin(kd, ["inactive", "weak", "active"])
                rows = tr[keep[tr]]
                X, Y, W = x[rows], yy[rows], w[rows]
                n_v = 0
                if vw > 0 and iso in BORROW:
                    v0 = np.flatnonzero(ext[f"veith_{iso}"].to_numpy() == 0.0)
                    n_v = len(v0)
                    # The bound's own midpoint is the point a squared-error learner should aim
                    # at; the interval is what the metric scores and what V2 checks.
                    X = np.concatenate([X, xe[v0]])
                    Y = np.concatenate([Y, np.full(n_v, (floor["lo"] + ACTIVE_CUT) / 2)])
                    W = np.concatenate([W, np.full(n_v, vw)])
                m = lgb.LGBMRegressor(n_estimators=args.n_estimators, learning_rate=0.03,
                                      num_leaves=31, min_child_samples=20, subsample=0.8,
                                      subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
                                      random_state=seed, n_jobs=10, verbose=-1)
                m.fit(X, Y, sample_weight=W)
                pv, pt = m.predict(x[va]), m.predict(xt)
                yv = y[va]
                v1 = np.isfinite(yv)
                lo = frame[f"{c}_conf_low"].to_numpy(float)[va]
                hi = frame[f"{c}_conf_high"].to_numpy(float)[va]
                inact = kd[va] == "inactive"
                yv2 = np.where(v1, yv, np.where(inact, floor["y"], np.nan))
                lo2 = np.where(v1, lo, np.where(inact, floor["lo"], np.nan))
                hi2 = np.where(v1, hi, np.where(inact, floor["hi"], np.nan))
                k2 = np.isfinite(yv2)
                res.append({"seed": seed, "arm": arm, "endpoint": iso, "n_veith0": n_v,
                            "v1": st_rae(pv[v1], yv[v1], lo[v1], hi[v1]),
                            "v2": st_rae(pv[k2], yv2[k2], lo2[k2], hi2[k2]),
                            "rho_v2": float(spearmanr(pv[k2], yv2[k2]).statistic),
                            "test_mean": float(pt.mean()), "test_sd": float(pt.std()),
                            "metric_id": METRIC_ID})
        print(f"[veith0] seed {seed} done", flush=True)

    df = pd.DataFrame(res)
    Path(args.out).write_text(json.dumps(res, indent=1))
    pd.set_option("display.width", 220)
    print(df.groupby(["endpoint", "arm"])[["v1", "v2", "rho_v2", "test_mean", "n_veith0"]]
          .mean().round(4).to_string())
    mac = df.groupby(["seed", "arm"])[["v1", "v2", "rho_v2"]].mean().reset_index()
    base = mac[mac.arm == "screen"].set_index("seed")
    a = mac[mac.arm == "screen+veith0"].set_index("seed")
    print("\nscreen+veith0 - screen, macro:")
    for v in ("v1", "v2", "rho_v2"):
        d = (a[v] - base[v]).to_numpy()
        print(f"  {v}: {d.mean():+.4f} (per seed {np.round(d, 4)})")
    print("\nper endpoint, V2 delta:")
    for iso in df.endpoint.unique():
        s = df[df.endpoint == iso].pivot_table(index="seed", columns="arm", values="v2")
        r = df[df.endpoint == iso].pivot_table(index="seed", columns="arm", values="rho_v2")
        print(f"  {iso}: v2 {(s['screen+veith0']-s['screen']).mean():+.4f}  "
              f"rho {(r['screen+veith0']-r['screen']).mean():+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
