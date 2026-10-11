"""Protonation-state features, for CYP2D6 specifically.

CYP2D6 is not noise-limited — measurement noise is 6.7% of its label variance
(max achievable R² 0.933, the second cleanest of the four) — yet it scores 0.6013
against a 0.6116 floor, the worst endpoint by far. CYP3A4, the *noisiest*
endpoint at 10.2%, scores the best at 0.2550. The ranking is nearly inverted, so
this is a representation failure rather than a ceiling.

The mechanism is specific and well known: **CYP2D6 binds basic amines**, through
a protonated nitrogen interacting with Glu216/Asp301 plus an aromatic ring at a
roughly fixed distance. That makes protonation state the governing property —
and **none of ECFP4, RDKit2D or Avalon encodes it**. ErG approximates it with a
positive-ionisable pharmacophore point, which is the closest anything in the
current 3,091 features comes.

So these are the features the representation is missing, not more capacity:
basic-nitrogen counts by class, Gasteiger charge on nitrogen, and the topological
distance from the most basic nitrogen to the nearest aromatic ring — the
pharmacophore the enzyme actually recognises.

RDKit has no pKa predictor, so basicity is approximated structurally: aliphatic
amines, guanidines and amidines are treated as protonated at pH 7.4, while
amides, anilines and aromatic nitrogens are not. That is a coarse rule and it is
the right coarseness for a 1,493-label endpoint.
"""
from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdmolops

RDLogger.DisableLog("rdApp.*")

#: Nitrogen environments that carry a positive charge at physiological pH.
_BASIC = [
    Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[a]);!$(N=*);!$([N+])]"),  # aliphatic amine
    Chem.MolFromSmarts("[NX3][CX3]=[NX2]"),          # amidine
    Chem.MolFromSmarts("[NX3][CX3](=[NX2])[NX3]"),   # guanidine
]
_ACIDIC = [
    Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),          # carboxylic acid
    Chem.MolFromSmarts("[SX4](=O)(=O)[OX2H1]"),      # sulfonic acid
    Chem.MolFromSmarts("c1nnn[nH]1"),                # tetrazole
]

NAMES = [
    "n_basic_centres", "n_acidic_centres", "net_charge_ph74",
    "has_basic_amine", "max_gasteiger_N", "min_gasteiger_N",
    "dist_basicN_to_aromatic", "n_aromatic_rings", "n_aliphatic_N", "frac_N_basic",
]


def basicity_features(smiles: str) -> np.ndarray:
    out = np.zeros(len(NAMES), dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    n_basic = sum(len(mol.GetSubstructMatches(p)) for p in _BASIC if p is not None)
    n_acid = sum(len(mol.GetSubstructMatches(p)) for p in _ACIDIC if p is not None)
    out[0], out[1] = n_basic, n_acid
    out[2] = n_basic - n_acid            # crude net charge at pH 7.4
    out[3] = 1.0 if n_basic else 0.0

    try:
        from rdkit.Chem import AllChem

        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(a.GetDoubleProp("_GasteigerCharge"))
                   for a in mol.GetAtoms() if a.GetSymbol() == "N"
                   and np.isfinite(a.GetDoubleProp("_GasteigerCharge"))]
        if charges:
            out[4], out[5] = max(charges), min(charges)
    except Exception:  # noqa: BLE001
        pass

    # The pharmacophore distance CYP2D6 recognises: basic nitrogen to aromatic ring.
    arom = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    basic_idx = [m[0] for p in _BASIC if p is not None for m in mol.GetSubstructMatches(p)]
    if arom and basic_idx:
        try:
            d = rdmolops.GetDistanceMatrix(mol)
            out[6] = float(min(d[b][a] for b in basic_idx for a in arom))
        except Exception:  # noqa: BLE001
            pass

    out[7] = rdMolDescriptors.CalcNumAromaticRings(mol)
    n_total = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
    out[8] = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N" and not a.GetIsAromatic())
    out[9] = (n_basic / n_total) if n_total else 0.0
    return out


def _worker(chunk: list[str]) -> np.ndarray:
    return np.stack([basicity_features(s) for s in chunk])


def basicity_matrix(smiles: list[str], workers: int | None = None) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor

    if len(smiles) < 500 or workers == 1:
        return np.stack([basicity_features(s) for s in smiles])
    chunk = max(256, len(smiles) // (workers or 8) // 4)
    chunks = [smiles[i:i + chunk] for i in range(0, len(smiles), chunk)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return np.concatenate(list(pool.map(_worker, chunks)))
