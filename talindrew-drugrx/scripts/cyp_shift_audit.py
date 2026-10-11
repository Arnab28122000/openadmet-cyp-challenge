r"""Is the blinded test set drawn from the training distribution? Measured, not assumed.

The leaderboard says our generalisation gap is large and **growing**: +0.1382 for the
previous submission, +0.2114 for the current one, while validation improved. Selection on
the 922 validation rows explains part of that. This checks the other candidate, which the
challenge states outright in its own description:

> "Analog Expansion Test Set: Hit expansion was performed on the top 25 hits from CYP1A2,
>  CYP2C9, and CYP3A4 by purchasing the top analogs."

Training comes from Enamine DDS10 (a diversity set) plus FDA-approved compounds. **The test
set is analogs of the strongest hits.** If so it is enriched in actives by construction, and
every quantity we fit on validation -- the per-endpoint offsets, the blend weight, the
best epoch -- is centred on the wrong distribution. That matters especially here because the
metric's mass sits in the active tail (43.9% of the ST-RAE numerator) and our predictions are
deliberately shrunk toward the mean, which is optimal only for the distribution they were
fitted on.

No labels are needed to detect it: if the test compounds are analogs of training actives,
they will sit far closer to the training actives than validation compounds do. Tanimoto over
Morgan fingerprints, which is what "top analogs" was selected by in the first place.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import PRIMARY, load
from drugrx.data.splits import scaffold_split

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fps(smis):
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        out.append(GEN.GetFingerprint(m) if m is not None else None)
    return out


def nn_sim(query, ref):
    """Max Tanimoto of each query against the reference set."""
    best = []
    for q in query:
        if q is None:
            best.append(np.nan); continue
        sims = DataStructs.BulkTanimotoSimilarity(q, ref)
        best.append(max(sims) if sims else np.nan)
    return np.array(best, float)


def describe(name, v):
    v = v[np.isfinite(v)]
    print(f"  {name:34s} n={len(v):4d}  mean {v.mean():.3f}  median {np.median(v):.3f}  "
          f">0.4 {100*(v>0.4).mean():5.1f}%  >0.6 {100*(v>0.6).mean():5.1f}%")


def main() -> int:
    d = load()
    fr, test = d.frame, d.test
    tr, va, rest = scaffold_split(fr["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=0)
    va = np.concatenate([va, rest]) if len(rest) else va

    train_fp = fps(fr["smiles"].to_numpy()[tr])
    valid_fp = fps(fr["smiles"].to_numpy()[va])
    test_fp = fps(test["SMILES"].astype(str).tolist())
    train_fp_ok = [f for f in train_fp if f is not None]

    # Training actives: any primary head at pIC50 >= 5.5, the tail threshold this
    # programme already uses.
    act = np.zeros(len(fr), bool)
    for c in PRIMARY:
        y = fr[c].to_numpy(float)
        act |= np.nan_to_num(y, nan=-9) >= 5.5
    act_idx = np.intersect1d(np.where(act)[0], tr)
    active_fp = [f for f in fps(fr["smiles"].to_numpy()[act_idx]) if f is not None]
    print(f"train {len(train_fp_ok)}  (actives pIC50>=5.5: {len(active_fp)})  "
          f"valid {len(valid_fp)}  test {len(test_fp)}\n")

    print("nearest-neighbour Tanimoto to ALL TRAINING compounds")
    describe("validation -> train", nn_sim(valid_fp, train_fp_ok))
    describe("TEST -> train", nn_sim(test_fp, train_fp_ok))

    print("\nnearest-neighbour Tanimoto to TRAINING ACTIVES only")
    describe("validation -> train actives", nn_sim(valid_fp, active_fp))
    describe("TEST -> train actives", nn_sim(test_fp, active_fp))

    print("\ninternal diversity (nearest neighbour WITHIN each set)")
    for name, f in (("validation", valid_fp), ("TEST", test_fp)):
        ok = [x for x in f if x is not None]
        best = []
        for i, q in enumerate(ok):
            sims = DataStructs.BulkTanimotoSimilarity(q, ok[:i] + ok[i+1:])
            best.append(max(sims) if sims else np.nan)
        describe(f"{name} -> itself", np.array(best, float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
