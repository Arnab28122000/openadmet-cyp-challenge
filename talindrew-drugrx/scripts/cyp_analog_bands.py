r"""Does the model degrade on close analogs? The activity-cliff account of the gap.

`cyp_shift_audit.py` showed the blinded test set is analog-expanded around training
actives (84.3% within Tanimoto 0.4 of one, against 15.1% of validation) and internally
tight (within-set NN 0.665 against 0.354). `cyp_testlike_offsets.py` then ruled out the
per-endpoint offsets as the channel: re-fitting them on validation's most test-like
quartile moves them 0.035 pIC50 and gains 0.0020.

That leaves the other candidate for a +0.2114 generalisation gap: the model may simply be
worse on tight analog series. Fingerprint-based models are known to fail on activity
cliffs, where near-identical structures differ sharply in potency -- and an analog
expansion around the strongest hits is exactly a cliff-rich set.

Validation is banded by each compound's nearest-neighbour similarity to the training set,
which is the axis along which test differs (test 0.562, validation 0.455). If performance
collapses in the CLOSE band, the gap has a mechanism we can act on; if it is flat, the
composition shift is not what is costing us and the search moves elsewhere.
"""
import glob, sys
sys.path.insert(0, ".")
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae, soft_threshold_error

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def main():
    d = load(); fr = d.frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}
    smi = fr["smiles"].to_numpy()
    runs = sorted(glob.glob("data/cyp_cache/cyp_h12/h12-longdeep_s*.npz"))
    idx = np.load(runs[0])["va_index"]
    def fp(s):
        m = Chem.MolFromSmiles(s); return GEN.GetFingerprint(m) if m else None
    tm = np.ones(len(fr), bool); tm[idx] = False
    train_fp = [f for f in (fp(s) for s in smi[tm]) if f]
    sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), train_fp)) for s in smi[idx]])
    print(f"validation NN-similarity to train: mean {sims.mean():.3f} "
          f"(test set's is 0.562)\n")
    # Split validation by how close each compound is to training -- the axis on which
    # test differs. If the model degrades on CLOSE analogs, that is the cliff problem.
    qs = np.quantile(sims, [0.25, 0.5, 0.75])
    bands = [("far   (Q1)", sims <= qs[0]), ("      (Q2)", (sims>qs[0])&(sims<=qs[1])),
             ("      (Q3)", (sims>qs[1])&(sims<=qs[2])), ("CLOSE (Q4)", sims > qs[2])]
    print(f"{'band':12s} {'n':>4s} {'meanNN':>7s} {'macro':>8s} {'mean|err|':>10s} {'meanY':>7s}")
    for name, m in bands:
        sc, errs = [], []
        for i, c in enumerate(PRIMARY):
            pa = np.mean([np.load(r)["val"][:, i] for r in runs], axis=0)
            yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
            k = np.isfinite(yy)&np.isfinite(ll)&np.isfinite(hh)&m
            if k.sum() < 20: continue
            sc.append(st_rae(pa[k], yy[k], ll[k], hh[k]))
            errs.append(soft_threshold_error(pa[k], ll[k], hh[k]).mean())
        yall = np.concatenate([y[c][idx][m & np.isfinite(y[c][idx])] for c in PRIMARY])
        print(f"{name:12s} {int(m.sum()):4d} {sims[m].mean():7.3f} {np.nanmean(sc):8.4f} "
              f"{np.nanmean(errs):10.4f} {yall.mean():7.3f}")
if __name__ == "__main__":
    main()
