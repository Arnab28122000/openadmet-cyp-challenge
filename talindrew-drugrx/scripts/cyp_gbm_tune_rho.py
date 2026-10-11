"""The last cheap stone: the GBM's hyperparameters have never been tuned.

Every comparison in this programme has used one LightGBM configuration -- `num_leaves=31`,
`learning_rate=0.03`, `n_estimators=800`, `colsample_bytree=0.3`, `min_child_samples=20`,
`reg_lambda=1.0` -- inherited from the first baseline and never revisited. With 3,091 features
against roughly 4,000 training rows per isoform, `colsample_bytree` and `num_leaves` are the
two that plausibly matter, and nobody has looked.

This is worth exactly one run because the alternative is not: every *model-side* candidate for
ranking is now measured and empty. CheMeleon 0.0000, the interval hinge -0.0094, the release's
20 auxiliary heads +0.0010, ensembling +0.0154 at one to two standard errors, local/kNN
blending -0.0616 in the test's own regime. Only the screen-derived interval labels ever
delivered, at +0.0417, and that is a *data* fix. The feature stack is already the MapLight
recipe (ECFP4 counts + rdkit2d + Avalon + ErG), so there is no obvious missing descriptor
either. Before concluding that 2D features are saturated, the default hyperparameters should
be ruled out too -- they are the one thing never checked.

Scored on **Spearman** first. Placement is monotone and cannot change rho, so only a rho gain
survives to the board; an ST-RAE gain at flat rho is one the affine step already reaches.

    python scripts/cyp_gbm_tune_rho.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import itertools
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
#: The configuration every previous comparison used.
INCUMBENT = dict(num_leaves=31, colsample_bytree=0.3, min_child_samples=20)
GRID = {"num_leaves": [15, 31, 63], "colsample_bytree": [0.1, 0.3, 0.6],
        "min_child_samples": [20]}
#: Confirmation grid. The 9-config sweep on seeds 0-2 chose (15, 0.6), so re-testing it on
#: FRESH seeds is the only way to tell a real effect from having picked the luckiest of nine.
CONFIRM = {"num_leaves": [15, 31], "colsample_bytree": [0.3, 0.6],
           "min_child_samples": [20]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--confirm", action="store_true",
                    help="use the 4-config confirmation grid instead of the full sweep")
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/gbm_tune_rho.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)

    g = CONFIRM if args.confirm else GRID
    combos = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    print(f"[tune] {len(combos)} configurations x {len(PRIMARY)} isoforms "
          f"x {len(args.seeds)} seeds", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[tune] seed {seed}", flush=True)
        for c in PRIMARY:
            y = frame[c].to_numpy(float)
            lo = frame[f"{c}_conf_low"].to_numpy(float)
            hi = frame[f"{c}_conf_high"].to_numpy(float)
            kind = labs[c]["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, labs[c]["y"].to_numpy(float))
            w0 = np.where(drc, 1.0, [PSEUDO_W.get(z, 0.0) for z in kind])
            idx = np.flatnonzero(drc | ps)
            v1 = np.isfinite(y[va])
            inact = kind[va] == "inactive"
            yv = np.where(v1, y[va], floor["y"])
            lv = np.where(v1, lo[va], floor["lo"])
            hv = np.where(v1, hi[va], floor["hi"])
            k2 = v1 | inact
            if v1.sum() < 40:
                continue
            for cfg in combos:
                t0 = time.time()
                m = lgb.LGBMRegressor(n_estimators=args.n_estimators, learning_rate=args.lr,
                                      subsample=0.8, subsample_freq=1, reg_lambda=1.0,
                                      random_state=seed, n_jobs=10, verbose=-1, **cfg)
                m.fit(x[idx], yy[idx], sample_weight=w0[idx])
                p = m.predict(x[va])
                rows.append({"seed": seed, "iso": c[:6], **cfg,
                             "rho": float(spearmanr(p[v1], y[va][v1]).statistic),
                             "v2": float(st_rae(p[k2], yv[k2], lv[k2], hv[k2])),
                             "secs": round(time.time() - t0, 1)})
            best = min([r for r in rows if r["seed"] == seed and r["iso"] == c[:6]],
                       key=lambda r: -r["rho"])
            print(f"   {c[:6]}  best rho {best['rho']:.4f} at "
                  f"leaves={best['num_leaves']} colsample={best['colsample_bytree']}",
                  flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== mean over seeds, paired against the incumbent config ===")
    for c in PRIMARY:
        iso = c[:6]
        rs = [r for r in rows if r["iso"] == iso]
        if not rs:
            continue
        base = [r for r in rs if all(r[k] == v for k, v in INCUMBENT.items())]
        bmap = {r["seed"]: r for r in base}
        print(f"\n{iso}  (incumbent rho {np.mean([r['rho'] for r in base]):.4f})")
        print(f"   {'leaves':>7} {'colsample':>10} {'d rho':>19} {'t':>6} {'d V2':>10}")
        out = []
        for cfg in combos:
            sel = [r for r in rs if all(r[k] == v for k, v in cfg.items())]
            d = np.array([r["rho"] - bmap[r["seed"]]["rho"] for r in sel if r["seed"] in bmap])
            dv = np.array([r["v2"] - bmap[r["seed"]]["v2"] for r in sel if r["seed"] in bmap])
            if len(d) < 2:
                continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            out.append((d.mean(), cfg, se, dv.mean()))
        for dm, cfg, se, dv in sorted(out, reverse=True):
            star = "  <- incumbent" if all(cfg[k] == v for k, v in INCUMBENT.items()) else ""
            print(f"   {cfg['num_leaves']:7} {cfg['colsample_bytree']:10} "
                  f"{dm:+12.4f} +-{se:.4f} {dm/se if se else 0:6.2f} {dv:+10.4f}{star}")
    print("\nA configuration beating the incumbent on rho by more than ~2 se on several")
    print("isoforms is worth adopting; anything smaller is noise and the default stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
