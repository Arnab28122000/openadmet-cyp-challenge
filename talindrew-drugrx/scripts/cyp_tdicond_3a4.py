"""1,249 extra CYP3A4 labels are sitting in the TDI file. Do they move the ranking?

CYP3A4 is this programme's weakest isoform relative to the field -- 41st percentile for
Spearman against 69th/63rd/88th elsewhere -- and the reason is now established from three
directions: the screen-label fix that carries every other isoform is worth **-0.0081** there,
and a plain LightGBM reproduces the board's CYP3A4 rho to 0.008, meaning the ensemble and
everything in it adds nothing. The CYP3A4 model is still the untouched baseline.

So it needs *data*, and there is some. Measured on the release:

    isoform   direct   TDI-cond   both   TDI-only   pearson r   slope   resid sd
    CYP1A2      1412       1413   1412          1      0.9863   0.964      0.170
    CYP2C9      1285       1285   1285          0      0.9623   0.940      0.213
    CYP2D6      1493       1497   1493          4      0.8964   1.058      0.406
    CYP3A4      2335       3583   2334       1249      0.9471   0.915      0.351

**The opportunity is CYP3A4-only and it is large**: +1,249 compounds, +53% on 2,335 direct
labels, at a correlation of 0.947 to the target. The other three isoforms have nothing to gain
because their two files cover the same compounds -- which is why this was invisible in every
macro-level comparison, and why it is being asked per isoform.

The TDI-condition assay is a different condition, so its pIC50 is mapped onto the direct scale
by a linear fit, `y ~ a*t + b`, **fit on the training fold only**. Fitting it on all compounds
that carry both would leak held-out labels into the pseudo-labels, which is the shape of
mistake this programme has already paid for twice.

Arms, identical except for the extra rows:

    base          drc + screen-derived labels -- the incumbent
    +tdicond@w    the same, plus the 1,249 mapped TDI-condition compounds at weight w

Reported on **both** ST-RAE and Spearman. Placement is monotone and cannot change rho, so rho
is the part of a gain here that survives to the board as ranking; ST-RAE additionally moves
with where the extra labels put the predictions, which placement can undo. A gain in ST-RAE
with flat rho is a placement artefact and should not be banked.

    python scripts/cyp_tdicond_3a4.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import statistics as st
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
ISO = "CYP3A4"
COL = f"{ISO}_pIC50_direct_inhibition"
TDI = f"{ISO}_pIC50_TDI_condition"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--weights", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/tdicond_3a4.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)

    y = frame[COL].to_numpy(float)
    lo = frame[f"{COL}_conf_low"].to_numpy(float)
    hi = frame[f"{COL}_conf_high"].to_numpy(float)
    t = frame[TDI].to_numpy(float)
    kind = labs[COL]["kind"].to_numpy()
    y_screen = labs[COL]["y"].to_numpy(float)
    print(f"[3a4] {np.isfinite(y).sum()} direct, {np.isfinite(t).sum()} TDI-condition, "
          f"{int((np.isfinite(t) & ~np.isfinite(y)).sum())} TDI-only", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        use = np.zeros(len(frame), bool)
        use[tr] = True

        # The incumbent's training rows.
        drc = use & np.isfinite(y)
        ps = use & np.isin(kind, ("inactive", "weak", "active"))
        yy = np.where(np.isfinite(y), y, y_screen)
        w0 = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
        base_rows = np.flatnonzero(drc | ps)

        # Map TDI-condition onto the direct scale, FIT ON THE TRAINING FOLD ONLY.
        fitm = use & np.isfinite(y) & np.isfinite(t)
        a, b = np.polyfit(t[fitm], y[fitm], 1)
        resid = float(np.std(y[fitm] - (a * t[fitm] + b)))
        extra = np.flatnonzero(use & np.isfinite(t) & ~np.isfinite(y))
        print(f"\n[3a4] seed {seed}: map y = {a:.3f}*t + {b:.3f} on {int(fitm.sum())} "
              f"train compounds (resid sd {resid:.3f}); {len(extra)} extra rows available",
              flush=True)

        # V2 scoring surface: held-out DRC rows plus held-out screen-inactives at the floor.
        v1 = np.isfinite(y[va])
        inact = kind[va] == "inactive"
        yv = np.where(v1, y[va], floor["y"])
        lv = np.where(v1, lo[va], floor["lo"])
        hv = np.where(v1, hi[va], floor["hi"])
        k2 = v1 | inact

        row = {"seed": seed, "slope": float(a), "intercept": float(b),
               "resid_sd": resid, "n_extra": int(len(extra)), "n_base": int(len(base_rows))}
        for tag, wx in [("base", None)] + [(f"tdicond@{w}", w) for w in args.weights]:
            t0 = time.time()
            if wx is None:
                ridx, ry, rw = base_rows, yy[base_rows], w0[base_rows]
            else:
                ridx = np.concatenate([base_rows, extra])
                ry = np.concatenate([yy[base_rows], a * t[extra] + b])
                rw = np.concatenate([w0[base_rows], np.full(len(extra), wx)])
            m = lgb.LGBMRegressor(
                n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                n_jobs=10, verbose=-1)
            m.fit(x[ridx], ry, sample_weight=rw)
            p = m.predict(x[va])
            s2 = float(st_rae(p[k2], yv[k2], lv[k2], hv[k2]))
            rho = float(spearmanr(p[v1], y[va][v1]).statistic)
            row[tag] = {"v2": s2, "rho": rho, "n": int(len(ridx)),
                        "secs": round(time.time() - t0, 1)}
            print(f"   {tag:14} n={len(ridx):5d}  V2 ST-RAE {s2:.4f}  rho {rho:.4f}  "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        rows.append(row)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    tags = ["base"] + [f"tdicond@{w}" for w in args.weights]
    print(f"\n=== CYP3A4 over {len(rows)} seeds; paired deltas vs base ===")
    print(f"{'arm':14} {'V2 ST-RAE':>18} {'Spearman':>18}   {'dV2':>9} {'drho':>9}")
    base_v2 = [r["base"]["v2"] for r in rows]
    base_rho = [r["base"]["rho"] for r in rows]
    for tag in tags:
        v2 = [r[tag]["v2"] for r in rows]
        rh = [r[tag]["rho"] for r in rows]
        sd2 = f"+-{st.stdev(v2):.4f}" if len(v2) > 1 else ""
        sdr = f"+-{st.stdev(rh):.4f}" if len(rh) > 1 else ""
        d2 = [a - b for a, b in zip(v2, base_v2)]
        dr = [a - b for a, b in zip(rh, base_rho)]
        extra = ""
        if tag != "base" and len(rows) > 1:
            n_better = sum(1 for v in d2 if v < 0)
            extra = f"   {n_better}/{len(rows)} seeds better on ST-RAE"
        print(f"{tag:14} {st.mean(v2):8.4f} {sd2:>9} {st.mean(rh):8.4f} {sdr:>9}   "
              f"{st.mean(d2):+9.4f} {st.mean(dr):+9.4f}{extra}")

    print("\nRead rho first: placement is monotone and cannot change it, so a rho gain is the")
    print("part that survives to the board. An ST-RAE gain with flat rho is placement and")
    print("should not be banked -- the affine step can reach it without any new data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
