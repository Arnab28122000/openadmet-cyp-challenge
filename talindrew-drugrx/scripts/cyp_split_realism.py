"""Does our validation split look like the blinded test set, in the only way that matters?

Model selection has been done on a **scaffold** split throughout this programme -- deliberately
the hardest generalisation. The blinded test set is described by the organisers as analogs of
CYP1A2/2C9/3A4 hits, i.e. an *analog expansion*, which is the easiest. If those two differ in
how far a held-out compound sits from its nearest training neighbour, then every model
comparison we have run has been scored on a surface that rewards something the test does not,
and the arm that wins on our validation need not be the arm that wins on the board.

That is checkable without a label. Take the ECFP4 nearest-neighbour Tanimoto from each held-out
compound to the training set, and compare three distributions:

    train -> TEST            what the board actually asks for
    train -> scaffold-val    what we have been selecting on
    train -> random-val      the easy end of the range, for scale

If `train -> TEST` sits near the scaffold curve, our surface is honest and this line closes
cheaply. If it sits near the random curve, model selection has been mis-specified and the fix
is to re-weight or re-split so the validation's neighbour profile matches the test's.

This is deliberately the cheap decisive measurement before any GPU time -- it needs no labels,
no training and no model, and it can only come out one of two ways.

    python scripts/cyp_split_realism.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import load
from drugrx.data.splits import scaffold_split


def fingerprints(smiles: list[str]):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        out.append(gen.GetFingerprint(m))
        ok.append(i)
    return out, np.array(ok)


def nn_tanimoto(query, ref, block: int = 256) -> np.ndarray:
    """Max Tanimoto from each query fingerprint to the reference set."""
    from rdkit import DataStructs
    best = np.zeros(len(query))
    for i, fp in enumerate(query):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref)
        best[i] = max(sims) if sims else 0.0
    return best


def describe(name: str, v: np.ndarray) -> None:
    qs = np.percentile(v, [10, 25, 50, 75, 90])
    print(f"{name:22} n={len(v):5d}  mean {v.mean():.3f}  "
          f"p10 {qs[0]:.3f}  p25 {qs[1]:.3f}  med {qs[2]:.3f}  p75 {qs[3]:.3f}  p90 {qs[4]:.3f}  "
          f"frac>=0.4 {np.mean(v >= 0.4):.3f}  frac>=0.7 {np.mean(v >= 0.7):.3f}")


def main() -> int:
    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    train_smiles = frame["smiles"].tolist()
    test_smiles = data.test["SMILES"].tolist()
    print(f"[split] {len(train_smiles)} train compounds, {len(test_smiles)} blinded test\n")

    tr_fp, _ = fingerprints(train_smiles)
    te_fp, _ = fingerprints(test_smiles)

    # 1. the real question
    v_test = nn_tanimoto(te_fp, tr_fp)

    # 2. what we select on -- scaffold split, same proportions as the programme uses
    rows = []
    for seed in (0, 1, 2):
        tr, va, rest = scaffold_split(train_smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        ref = [tr_fp[i] for i in tr]
        rows.append(nn_tanimoto([tr_fp[i] for i in va], ref))
    v_scaf = np.concatenate(rows)

    # 3. the easy end, for scale
    rng = np.random.default_rng(0)
    rows = []
    for seed in (0, 1, 2):
        idx = rng.permutation(len(tr_fp))
        cut = int(0.85 * len(idx))
        tr, va = idx[:cut], idx[cut:]
        ref = [tr_fp[i] for i in tr]
        rows.append(nn_tanimoto([tr_fp[i] for i in va], ref))
    v_rand = np.concatenate(rows)

    print("Nearest-neighbour ECFP4 Tanimoto from each held-out compound to its training set:\n")
    describe("train -> TEST", v_test)
    describe("train -> scaffold-val", v_scaf)
    describe("train -> random-val", v_rand)

    print("\nVerdict:")
    mt, ms, mr = (float(np.median(x)) for x in (v_test, v_scaf, v_rand))
    print(f"  median NN Tanimoto: TEST {mt:.3f}  scaffold-val {ms:.3f}  random-val {mr:.3f}")
    if mt > max(ms, mr):
        # The interesting case, and the one that actually obtains here: the test is not
        # bracketed by our two splits, it is EASIER than both. An analog expansion of
        # training hits gives every test compound a close relative by construction.
        print(f"  the TEST is EASIER THAN BOTH -- +{mt-mr:.3f} over even a random split.")
        print(f"  near neighbours (>=0.4): TEST {np.mean(v_test>=0.4):.3f} vs "
              f"scaffold {np.mean(v_scaf>=0.4):.3f}, and the test has almost no far tail "
              f"(p10 {np.percentile(v_test,10):.3f} vs {np.percentile(v_scaf,10):.3f}).")
        print("  -> every arm comparison we have run was dominated by held-out compounds")
        print("     harder than anything the board asks about. Re-weight the validation to")
        print("     the test's neighbour profile and re-rank the arms before trusting them.")
    else:
        which = "SCAFFOLD" if abs(mt - ms) < abs(mt - mr) else "RANDOM"
        print(f"  the TEST median sits closer to the {which} split.")
        if which == "SCAFFOLD":
            print("  -> our validation surface is already the right difficulty; this line closes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
