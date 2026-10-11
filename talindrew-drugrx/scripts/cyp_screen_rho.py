"""The screen fix was adopted on ST-RAE. What did it do to the *ranking*?

This programme's largest single gain was treating non-promoted compounds as interval labels:
macro V2 ST-RAE **-0.243**, and per isoform -0.2341 / -0.0954 / -0.6329 / -0.0081. Every one of
those numbers is ST-RAE.

**ST-RAE is not the durable part.** Placement is a monotone affine map, so it cannot change
Spearman -- which means any ST-RAE gain at fixed rho is something the placement step can reach
on its own, with no new labels. And placement is now optimised hard: v5 alone bought -0.0371 of
macro ST-RAE at *exactly* unchanged rho. So the screen fix's ST-RAE column is partly redundant
with a lever we are already pulling, and the question nobody has asked is what it did to rho.

It matters most where the two could disagree. CYP3A4's screen fix is worth -0.0081 of ST-RAE --
nothing -- while CYP3A4 is simultaneously our only below-median isoform for rho (41st
percentile, 0.7685 against a field best of 0.8741). If the screen-derived labels are *hurting*
CYP3A4's ranking while leaving its ST-RAE alone, dropping them for that isoform is free score,
and nothing measured so far would have revealed it.

Three arms, the incumbent's splits and hyperparameters, paired across seeds:

    drc            dose-response labels only
    drc+inactive   plus the screen's clear non-inhibitors at the floor window
    drc+screen     plus every screen-derived label, weak/active at half weight (the incumbent)

Scored two ways on the same held-out compounds: **Spearman on the held-out rows that carry a
real DRC label** -- which is what the board's rho is computed on, and what survives placement --
and V2 ST-RAE, for continuity with every earlier comparison.

    python scripts/cyp_screen_rho.py --seeds 0 1 2 3 4 5 6 7
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
ARMS = {"drc": (), "drc+inactive": ("inactive",),
        "drc+screen": ("inactive", "weak", "active")}
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
BOARD_RHO = {"CYP1A2": 0.7597, "CYP2C9": 0.8084, "CYP2D6": 0.5004, "CYP3A4": 0.7685}
FIELD_BEST_RHO = {"CYP1A2": 0.8561, "CYP2C9": 0.8808, "CYP2D6": 0.5700, "CYP3A4": 0.8741}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(8)))
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/screen_rho.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        print(f"\n[rho] seed {seed}: {len(tr)} train / {len(va)} val", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            lo = frame[f"{c}_conf_low"].to_numpy(float)
            hi = frame[f"{c}_conf_high"].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool)
            use[tr] = True
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))

            # Scored on exactly the same held-out compounds for every arm.
            v1 = np.isfinite(y[va])
            inact = kind[va] == "inactive"
            yv = np.where(v1, y[va], floor["y"])
            lv = np.where(v1, lo[va], floor["lo"])
            hv = np.where(v1, hi[va], floor["hi"])
            k2 = v1 | inact
            if v1.sum() < 40:
                continue

            row = {"seed": seed, "iso": c[:6], "n_rho": int(v1.sum()), "n_v2": int(k2.sum())}
            for arm, kinds in ARMS.items():
                t0 = time.time()
                drc = use & np.isfinite(y)
                ps = use & np.isin(kind, kinds)
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                idx = np.flatnonzero(drc | ps)
                m = lgb.LGBMRegressor(
                    n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=31,
                    min_child_samples=20, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                    n_jobs=10, verbose=-1)
                m.fit(x[idx], yy[idx], sample_weight=w[idx])
                p = m.predict(x[va])
                row[arm] = {"rho": float(spearmanr(p[v1], y[va][v1]).statistic),
                            "v2": float(st_rae(p[k2], yv[k2], lv[k2], hv[k2])),
                            "n_train": int(len(idx)), "secs": round(time.time() - t0, 1)}
            rows.append(row)
            print("   " + c[:6] + "  " + "  ".join(
                f"{a}: rho {row[a]['rho']:.4f} V2 {row[a]['v2']:.4f}" for a in ARMS), flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== paired against the `drc` baseline, over seeds ===")
    print(f"{'iso':8} {'arm':14} {'d rho':>20} {'t':>6} | {'d V2 ST-RAE':>20} {'t':>6}")
    for c in PRIMARY:
        iso = c[:6]
        rs = [r for r in rows if r["iso"] == iso]
        if len(rs) < 2:
            continue
        for arm in ("drc+inactive", "drc+screen"):
            line = f"{iso:8} {arm:14}"
            for key in ("rho", "v2"):
                d = np.array([r[arm][key] - r["drc"][key] for r in rs])
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = d.mean() / se if se > 0 else float("inf")
                line += f" {d.mean():+11.4f} +-{se:.4f} {t:6.2f} |"
            print(line.rstrip("|"))
        inc = [r["drc+screen"]["rho"] for r in rs]
        print(f"{'':8} {'-> incumbent rho':14} {np.mean(inc):.4f}   board {BOARD_RHO[iso]:.4f}"
              f"   field best {FIELD_BEST_RHO[iso]:.4f}"
              f"   gap {FIELD_BEST_RHO[iso]-BOARD_RHO[iso]:+.4f}")

    print("\nA negative d rho means the screen-derived labels COST ranking on that isoform.")
    print("Because placement cannot change rho, that cost is not recoverable downstream --")
    print("whereas a positive d V2 at flat d rho is something placement already reaches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
