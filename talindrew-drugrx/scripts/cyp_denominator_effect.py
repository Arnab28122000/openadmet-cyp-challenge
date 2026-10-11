r"""Is the generalisation gap a model failure, or the metric's denominator moving?

ST-RAE divides the model's soft-threshold error by the **same error from a constant
predictor at mean(y) of the evaluation set**. That denominator is a property of the
evaluation set's label spread, not of the model. A set whose labels are concentrated --
analog series around a few hits, which is exactly how this challenge's test set was built --
makes the mean predictor look good, shrinks the denominator, and **inflates ST-RAE for an
unchanged absolute error**.

Two findings make this the live hypothesis. The composition shift is real (`cyp_shift_audit`)
but the model does not degrade on close analogs (`cyp_analog_bands`: the CLOSE band scores
0.705 against the FAR band's 0.781), so a +0.2114 gap cannot be a competence failure on
analog chemistry. And the early log had already noted that ST-RAE "is not directly comparable"
across evaluation sets before the point was dropped.

Numerator and denominator are therefore reported **separately** per similarity band. If the
ratio rises while the numerator is flat or falling, the metric is moving, not the model.
"""
from __future__ import annotations

import glob

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def main() -> int:
    d = load()
    fr = d.frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}
    smi = fr["smiles"].to_numpy()
    runs = sorted(glob.glob("data/cyp_cache/cyp_h12/h12-longdeep_s*.npz"))
    idx = np.load(runs[0])["va_index"]

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return GEN.GetFingerprint(m) if m else None

    tm = np.ones(len(fr), bool); tm[idx] = False
    train_fp = [f for f in (fp(s) for s in smi[tm]) if f]
    sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), train_fp))
                     for s in smi[idx]])
    qs = np.quantile(sims, [0.25, 0.5, 0.75])
    bands = [("far   (Q1)", sims <= qs[0]), ("      (Q2)", (sims > qs[0]) & (sims <= qs[1])),
             ("      (Q3)", (sims > qs[1]) & (sims <= qs[2])), ("CLOSE (Q4)", sims > qs[2])]

    print(f"{'band':12s} {'n':>4s} {'meanNN':>7s} {'NUMER':>9s} {'DENOM':>9s} "
          f"{'ratio':>8s} {'sd(y)':>7s}")
    base = None
    for name, m in bands:
        num = den = 0.0
        spreads = []
        for i, c in enumerate(PRIMARY):
            pa = np.mean([np.load(r)["val"][:, i] for r in runs], axis=0)
            yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
            k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) & m
            if k.sum() < 20:
                continue
            num += soft_threshold_error(pa[k], ll[k], hh[k]).sum()
            # The official denominator: a constant predictor at this subset's own mean.
            den += soft_threshold_error(np.full(k.sum(), yy[k].mean()), ll[k], hh[k]).sum()
            spreads.append(yy[k].std())
        r = num / den
        if base is None:
            base = (num, den, r)
        print(f"{name:12s} {int(m.sum()):4d} {sims[m].mean():7.3f} {num:9.2f} {den:9.2f} "
              f"{r:8.4f} {np.mean(spreads):7.3f}")

    print("\nsame numerator, different denominator — what the metric does on its own:")
    nums, dens = [], []
    for name, m in bands:
        num = den = 0.0
        for i, c in enumerate(PRIMARY):
            pa = np.mean([np.load(r)["val"][:, i] for r in runs], axis=0)
            yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
            k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) & m
            if k.sum() < 20:
                continue
            num += soft_threshold_error(pa[k], ll[k], hh[k]).sum()
            den += soft_threshold_error(np.full(k.sum(), yy[k].mean()), ll[k], hh[k]).sum()
        nums.append(num / max(int(m.sum()), 1)); dens.append(den / max(int(m.sum()), 1))
    print(f"  per-compound numerator across bands: " + " ".join(f"{x:.4f}" for x in nums))
    print(f"  per-compound denominator across bands: " + " ".join(f"{x:.4f}" for x in dens))
    print(f"  numerator far->close: {nums[-1]/nums[0]:.3f}x    "
          f"denominator far->close: {dens[-1]/dens[0]:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
