r"""Can pair space supply the analog-series geometry that molecule space cannot?

The structural bound established earlier: our validation set has within-set nearest-neighbour
similarity 0.354 against the test set's 0.665, and no re-splitting fixes it, because the
organisers **purchased** analogs that are not in our library at any density. Every quantity
tuned on validation has therefore been tuned on the wrong distribution, which is the
structural reason leaderboard transfer was 17%.

The claim under test, raised in review: that bound is a fact about **molecules**, not about
**pairs**. A pairwise model predicts Delta(pIC50) between a query and a structurally near
anchor whose label is known, and from ~6.1k labelled compounds there are ~19M pairs, of which
the high-similarity ones can simply be **selected**. If enough pairs exist at Tanimoto >= 0.6
-- the test set's own internal similarity -- then a pairwise model can be validated on the
geometry it will actually be scored against, which no experiment in this programme has
managed.

This is the feasibility check that decides whether the idea is worth pre-registering. It
counts, per endpoint, how many pairs exist at each similarity threshold with **both** labels
present, and how much label signal those pairs carry (a pair whose Delta is always ~0 teaches
nothing). It also reports the fraction of test compounds that have a labelled training anchor
at each threshold, since that is what the method needs at inference.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import PRIMARY, load

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
THRESHOLDS = (0.4, 0.5, 0.6, 0.7)


def main() -> int:
    d = load()
    fr = d.frame
    smi = fr["smiles"].to_numpy()
    fps, keep = [], []
    for i, s in enumerate(smi):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(GEN.GetFingerprint(m)); keep.append(i)
    keep = np.array(keep)
    n = len(fps)
    print(f"{n} fingerprintable training compounds -> {n*(n-1)//2:,} unordered pairs\n")

    lab = {c: fr[c].to_numpy(float)[keep] for c in PRIMARY}

    counts = {t: {c: 0 for c in PRIMARY} for t in THRESHOLDS}
    deltas = {t: {c: [] for c in PRIMARY} for t in THRESHOLDS}
    for i in range(n - 1):
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        for t in THRESHOLDS:
            hit = np.where(sims >= t)[0]
            if not len(hit):
                continue
            j = hit + i + 1
            for c in PRIMARY:
                a, b = lab[c][i], lab[c][j]
                m = np.isfinite(b) & np.isfinite(a)
                if m.any():
                    counts[t][c] += int(m.sum())
                    deltas[t][c].extend(np.abs(b[m] - a).tolist())

    print(f"{'threshold':>10s} {'endpoint':10s} {'labelled pairs':>15s} "
          f"{'median |dy|':>12s} {'% >1 log':>9s}")
    for t in THRESHOLDS:
        for c in PRIMARY:
            dv = np.array(deltas[t][c])
            med = np.median(dv) if len(dv) else float("nan")
            frac = 100 * (dv > 1).mean() if len(dv) else float("nan")
            print(f"{t:10.1f} {c.split('_')[0]:10s} {counts[t][c]:15,d} "
                  f"{med:12.3f} {frac:9.1f}")
        tot = sum(counts[t].values())
        print(f"{'':10s} {'TOTAL':10s} {tot:15,d}\n")

    # Inference-side: does each test compound have a labelled anchor close enough?
    test_fp = []
    for s in d.test["SMILES"].astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            test_fp.append(GEN.GetFingerprint(m))
    print("test compounds with at least one labelled training anchor:")
    best = np.array([max(DataStructs.BulkTanimotoSimilarity(f, fps)) for f in test_fp])
    for t in THRESHOLDS:
        print(f"  Tanimoto >= {t:.1f}: {100*(best>=t).mean():5.1f}%  "
              f"({int((best>=t).sum())} of {len(test_fp)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
