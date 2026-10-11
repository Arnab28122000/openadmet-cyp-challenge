r"""Is Delta easier to predict than y? The expected-effect check before any GPU.

If a query's label were independent of its anchor's, sd(Delta) = sd(y)*sqrt(2). Anchoring is
informative only to the extent sd(Delta) falls below that. The ratio
    R = sd(Delta) / (sd(y) * sqrt(2))
is therefore the whole case for delta learning: R near 1 means the neighbour tells you
nothing and the reformulation has replaced one hard task with an equally hard one while
inheriting the anchor's label noise.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from drugrx.cyp.data import PRIMARY, load

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def main():
    fr = load().frame
    smi = fr["smiles"].to_numpy()
    fps, keep = [], []
    for i, s in enumerate(smi):
        m = Chem.MolFromSmiles(s)
        if m is not None: fps.append(GEN.GetFingerprint(m)); keep.append(i)
    keep = np.array(keep); n = len(fps)
    lab = {c: fr[c].to_numpy(float)[keep] for c in PRIMARY}
    THR = (0.3, 0.4, 0.5, 0.6)
    acc = {t: {c: [] for c in PRIMARY} for t in THR}
    for i in range(n - 1):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]))
        for t in THR:
            h = np.where(sims >= t)[0]
            if not len(h): continue
            j = h + i + 1
            for c in PRIMARY:
                a, b = lab[c][i], lab[c][j]
                m = np.isfinite(b) & np.isfinite(a)
                if m.any(): acc[t][c].extend((b[m] - a).tolist())
    print(f"{'thr':>5s} {'endpoint':9s} {'n pairs':>8s} {'sd(dy)':>8s} {'sd(y)*r2':>9s} "
          f"{'R':>6s}  verdict")
    for t in THR:
        Rs = []
        for c in PRIMARY:
            dv = np.array(acc[t][c])
            if len(dv) < 15: 
                print(f"{t:5.1f} {c.split('_')[0]:9s} {len(dv):8d}   too few"); continue
            sdy = np.nanstd(lab[c]); ind = sdy * np.sqrt(2); R = dv.std() / ind
            Rs.append(R)
            print(f"{t:5.1f} {c.split('_')[0]:9s} {len(dv):8d} {dv.std():8.3f} "
                  f"{ind:9.3f} {R:6.3f}  {'informative' if R < 0.85 else 'NOT informative'}")
        if Rs: print(f"{'':5s} {'mean R':9s} {'':8s} {'':8s} {'':9s} {np.mean(Rs):6.3f}\n")
if __name__ == "__main__":
    main()
