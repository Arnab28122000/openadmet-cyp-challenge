r"""An analog-expansion validation split, built the way the organisers built the test set.

Today's measurements, in order:

* the blinded test set is analog-expanded around training actives -- 84.3% of its compounds
  sit within Tanimoto 0.4 of one, against 15.1% of our scaffold-split validation, and its
  internal nearest-neighbour similarity is 0.665 against validation's 0.354
  (`cyp_shift_audit.py`);
* the per-endpoint offsets are robust to that shift -- re-fitting on validation's most
  test-like quartile moves them 0.035 pIC50 (`cyp_testlike_offsets.py`);
* the model does **not** degrade on close analogs -- the CLOSE band scores 0.705 against the
  FAR band's 0.781 (`cyp_analog_bands.py`);
* and the metric's denominator does not inflate in tight bands -- flat at 1.016x far-to-close
  while the numerator falls to 0.769x (`cyp_denominator_effect.py`).

So no mechanism available *within* validation explains a +0.2114 gap, and the reason is that
**validation has no test-like subset at all**. A scaffold split of a diversity library cannot
produce analog series; it is built to do the opposite. That is why every quantity tuned on it
-- blend weight, offsets, best epoch, architecture -- has returned about 17% on test.

This builds the split the challenge actually evaluates on: take the strongest hits per
isoform, hold out their nearest analogs, and train on the rest. Selection then happens on a
set whose composition matches the thing being scored.

Deliberately *not* a scaffold split: analogs usually share a scaffold, so a scaffold split
would place them all on one side. That is the correct behaviour for estimating performance on
novel chemistry and the wrong one for estimating performance on an analog expansion. Both are
legitimate; they answer different questions, and this programme has only ever asked the first.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _fps(smiles):
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        out.append(_GEN.GetFingerprint(m) if m is not None else None)
    return out


def analog_expansion_split(smiles, labels, n_hits: int = 25, n_analogs: int = 10,
                           seed: int = 0):
    """Hold out the nearest analogs of the strongest hits, mirroring the test set.

    `labels` is a 2-D array (compounds x endpoints) with NaN for missing. Hits are the
    `n_hits` highest-potency compounds per endpoint; for each, its `n_analogs` nearest
    neighbours by Tanimoto go to validation. The hits themselves stay in training, which
    is what the organisers did -- they assayed *purchased analogs* of hits they already had.

    Returns (train_idx, valid_idx).
    """
    smiles = list(smiles)
    fps = _fps(smiles)
    ok = np.array([f is not None for f in fps])
    valid: set[int] = set()
    hits: set[int] = set()

    for j in range(labels.shape[1]):
        y = labels[:, j].astype(float)
        order = np.argsort(np.where(np.isnan(y), -np.inf, y))[::-1]
        taken = 0
        for i in order:
            if taken >= n_hits:
                break
            if not ok[i] or np.isnan(y[i]):
                continue
            hits.add(int(i))
            taken += 1
            sims = np.full(len(smiles), -1.0)
            idxs = [k for k in range(len(smiles)) if ok[k] and k != i]
            vals = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[k] for k in idxs])
            for k, v in zip(idxs, vals):
                sims[k] = v
            for k in np.argsort(sims)[::-1][:n_analogs]:
                if sims[k] > 0 and int(k) not in hits:
                    valid.add(int(k))

    valid -= hits                      # a hit used as an anchor must not also be scored
    va = np.array(sorted(valid), int)
    tr = np.array([i for i in range(len(smiles)) if i not in valid], int)
    rng = np.random.default_rng(seed)
    rng.shuffle(tr)
    return tr, va


def main() -> int:
    from drugrx.cyp.data import PRIMARY, load
    from drugrx.data.splits import scaffold_split

    d = load()
    fr = d.frame
    smi = fr["smiles"].tolist()
    lab = np.column_stack([fr[c].to_numpy(float) for c in PRIMARY])

    tr_a, va_a = analog_expansion_split(smi, lab)
    tr_s, va_s, rest = scaffold_split(smi, frac_train=0.85, frac_valid=0.10, seed=0)
    va_s = np.concatenate([va_s, rest]) if len(rest) else va_s

    fps = _fps(smi)
    test_fp = [f for f in _fps(d.test["SMILES"].astype(str).tolist()) if f]

    def profile(name, va, tr):
        vf = [fps[i] for i in va if fps[i] is not None]
        tf = [fps[i] for i in tr if fps[i] is not None]
        to_train = [max(DataStructs.BulkTanimotoSimilarity(f, tf)) for f in vf]
        within = []
        for i, f in enumerate(vf):
            o = vf[:i] + vf[i+1:]
            within.append(max(DataStructs.BulkTanimotoSimilarity(f, o)) if o else np.nan)
        ya = lab[va]
        print(f"  {name:24s} n={len(va):4d}  NN->train {np.mean(to_train):.3f}  "
              f"within-set {np.nanmean(within):.3f}  mean y {np.nanmean(ya):.3f}")

    print("split composition, against the test set's own profile:")
    profile("scaffold (incumbent)", va_s, tr_s)
    profile("analog-expansion", va_a, tr_a)
    tw = []
    for i, f in enumerate(test_fp):
        o = test_fp[:i] + test_fp[i+1:]
        tw.append(max(DataStructs.BulkTanimotoSimilarity(f, o)))
    all_fp = [f for f in fps if f is not None]
    print(f"  {'TEST SET (target)':24s} n={len(test_fp):4d}  "
          f"NN->train {np.mean([max(DataStructs.BulkTanimotoSimilarity(f, all_fp)) for f in test_fp]):.3f}  "
          f"within-set {np.mean(tw):.3f}  mean y    (blinded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
