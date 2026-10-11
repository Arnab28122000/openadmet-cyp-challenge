r"""Does the model degrade inside analog clusters? Banded on the axis that defines the test set.

The earlier banding experiment (`cyp_analog_bands.py`) split validation by **similarity to
the training set** and concluded the model does not degrade on analogs. That was the wrong
axis, and the conclusion was wrong with it. What distinguishes the test set is its
**internal** clustering -- within-set nearest-neighbour similarity 0.665, 98.8% above 0.4 --
against validation's 0.354 and 23.9%. Banding on NN-to-train never varied that.

Reading the old table correctly also undermines its own conclusion. The n-weighted validation
mean is 0.7064; Q3 (sim 0.471) scored 0.6627, i.e. 0.044 **better** than average, and Q4
(sim 0.609) scored 0.7051, i.e. exactly average. The curve has a minimum at Q3 and is
**rising** at Q4 by +0.0424. Degradation at high similarity was already visible.

This bands validation by each compound's nearest-neighbour similarity **within validation**,
which is the analog-cluster axis, and reports the label spread inside each band. An activity
cliff is structural tightness with label dispersion, so if the tight bands carry *more* label
spread and *worse* error, that is the mechanism, measured on the axis it lives on.
"""
from __future__ import annotations

import glob

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error, st_rae

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def main() -> int:
    d = load()
    fr = d.frame
    runs = sorted(glob.glob("data/cyp_cache/cyp_h12/h12-longdeep_s*.npz"))
    idx = np.load(runs[0])["va_index"]
    pred = np.mean([np.load(r)["val"] for r in runs], axis=0)
    smi = fr["smiles"].to_numpy()[idx]

    fps = []
    for s in smi:
        m = Chem.MolFromSmiles(s)
        fps.append(GEN.GetFingerprint(m) if m is not None else None)
    ok = [i for i, f in enumerate(fps) if f is not None]

    # Internal nearest-neighbour similarity: the analog-cluster axis.
    inner = np.full(len(smi), np.nan)
    for i in ok:
        others = [fps[j] for j in ok if j != i]
        inner[i] = max(DataStructs.BulkTanimotoSimilarity(fps[i], others))

    # Local label dispersion: among a compound's close structural neighbours, how far
    # apart are the labels? This is the cliff quantity itself.
    cliff = np.full(len(smi), np.nan)
    prim = np.column_stack([fr[c].to_numpy(float)[idx] for c in PRIMARY])
    for i in ok:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in ok]))
        near = [ok[j] for j in np.where(sims >= 0.4)[0] if ok[j] != i]
        if not near:
            continue
        diffs = []
        for c in range(len(PRIMARY)):
            a = prim[i, c]
            if not np.isfinite(a):
                continue
            b = prim[near, c]
            b = b[np.isfinite(b)]
            if len(b):
                diffs.append(np.abs(b - a).max())
        if diffs:
            cliff[i] = max(diffs)

    qs = np.nanquantile(inner, [0.25, 0.5, 0.75])
    bands = [("isolated (Q1)", inner <= qs[0]),
             ("         (Q2)", (inner > qs[0]) & (inner <= qs[1])),
             ("         (Q3)", (inner > qs[1]) & (inner <= qs[2])),
             ("CLUSTERED(Q4)", inner > qs[2])]

    print(f"validation internal NN: mean {np.nanmean(inner):.3f}   "
          f"(TEST SET's is 0.665)\n")
    print(f"{'band':14s} {'n':>4s} {'innerNN':>8s} {'macro':>8s} {'err/cmpd':>9s} "
          f"{'sd(y)':>7s} {'maxΔy to':>9s}")
    print(f"{'':14s} {'':>4s} {'':>8s} {'':>8s} {'':>9s} {'':>7s} {'neighbour':>9s}")
    for name, m in bands:
        sc, errs = [], []
        for i, c in enumerate(PRIMARY):
            y = fr[c].to_numpy(float)[idx]
            lo = fr[f"{c}_conf_low"].to_numpy(float)[idx]
            hi = fr[f"{c}_conf_high"].to_numpy(float)[idx]
            k = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & m
            if k.sum() < 20:
                continue
            sc.append(st_rae(pred[k, i], y[k], lo[k], hi[k]))
            errs.append(soft_threshold_error(pred[k, i], lo[k], hi[k]).mean())
        yall = np.concatenate([prim[m & np.isfinite(prim[:, c]), c]
                               for c in range(len(PRIMARY))])
        print(f"{name:14s} {int(m.sum()):4d} {np.nanmean(inner[m]):8.3f} "
              f"{np.nanmean(sc):8.4f} {np.nanmean(errs):9.4f} {yall.std():7.3f} "
              f"{np.nanmean(cliff[m]):9.3f}")

    v = cliff[np.isfinite(cliff)]
    print(f"\nlocal cliff magnitude (max |Δy| to a neighbour at Tanimoto>=0.4), "
          f"n={len(v)}:")
    print(f"  median {np.median(v):.2f}   mean {v.mean():.2f}   "
          f">1.0 log unit {100*(v>1).mean():.1f}%   >2.0 {100*(v>2).mean():.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
