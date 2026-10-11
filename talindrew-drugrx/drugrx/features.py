"""Featurisation, computed once and cached — deliberately never on the GPU box.

The L4 Colab runtime is a `g2-standard-4`: **four vCPU**. RDKit featurisation of
a hundred thousand molecules on four cores will starve the GPU for most of the
run, and we would be paying compute units for a CPU-bound job. So featurisation
happens on the laptop (many cores, free) and the result ships to the VM as a
single `.npz`.

That also makes a resumed run cheap: a fresh VM loads one array file instead of
re-deriving everything.

`SMILES augmentation` is the exception that cannot be precomputed per-epoch, so
randomised SMILES are generated as a fixed pool of K variants per molecule and
sampled during training. A pool is not quite the same as fresh randomisation
every epoch, but it is close enough at K≥10 and keeps the GPU fed.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

#: A small, standard descriptor block. Not Mordred's 1,613 — those are for
#: CheMeleon's pretraining, which we download rather than repeat.
DESCRIPTORS: tuple[str, ...] = (
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHAcceptors", "NumHDonors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3",
    "HeavyAtomCount", "NHOHCount", "NOCount", "NumHeteroatoms",
    "NumSaturatedRings", "NumAliphaticRings", "qed", "BertzCT",
    "BalabanJ", "HallKierAlpha", "Kappa1", "Kappa2", "Kappa3", "Chi0v", "Chi1v",
)

#: Descriptors excluded from the full "RDKit2D" block. `Ipc` grows factorially
#: and overflows to inf on ordinary drug-sized molecules — it is the classic
#: poisoner of a descriptor matrix, and it takes the standardisation of every
#: other column down with it.
DESCRIPTOR_BLOCKLIST: frozenset[str] = frozenset({"Ipc"})

#: Widths of the two MapLight fingerprints appended when `extra_fps=True`.
AVALON_BITS = 512
ERG_BITS = 315


def _descriptor_fns(mode: str = "small"):
    """`"small"` is the hand-picked block above; `"rdkit2d"` is everything RDKit
    computes in 2D (~210 descriptors), which is the block the literature means
    by "RDKit2D" and which beat learned representations in Deng 2023."""
    table = dict(Descriptors.descList)
    if mode == "rdkit2d":
        return [(n, f) for n, f in Descriptors.descList if n not in DESCRIPTOR_BLOCKLIST]
    missing = [n for n in DESCRIPTORS if n not in table]
    if missing:
        raise RuntimeError(f"RDKit has no descriptor(s): {missing}")
    return [(n, table[n]) for n in DESCRIPTORS]


def _avalon(mol, n_bits: int = 512) -> np.ndarray:
    """Avalon fingerprint — a different substructure enumeration from Morgan's.

    Part of MapLight's recipe, which is one of only three TDC ADMET
    architectures that survived an independent reproduction audit. Our feature
    set was MapLight's minus this and ErG, so adding them costs no GPU and
    closes a gap against a method that is known to reproduce.
    """
    from rdkit.Avalon import pyAvalonTools

    out = np.zeros(n_bits, dtype=np.float32)
    try:
        fp = pyAvalonTools.GetAvalonCountFP(mol, nBits=n_bits)
        for idx, count in fp.GetNonzeroElements().items():
            out[idx] = count
    except Exception:  # noqa: BLE001
        pass
    return out


def _erg(mol) -> np.ndarray:
    """Extended reduced graph — pharmacophore-type features, 315 floats.

    Describes *what a molecule does* (donor/acceptor/charge/hydrophobe
    topology) rather than which substructures it contains, so it carries
    information a Morgan fingerprint structurally cannot.
    """
    from rdkit.Chem import rdReducedGraphs

    try:
        return np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    except Exception:  # noqa: BLE001
        return np.zeros(315, dtype=np.float32)


def featurise_one(
    smiles: str,
    n_bits: int = 2048,
    radius: int = 2,
    counts: bool = True,
    desc_mode: str = "rdkit2d",
    extra_fps: bool = False,
) -> np.ndarray:
    """ECFP4 concatenated with the descriptor block.

    `counts=True` keeps substructure *multiplicity* rather than mere presence —
    three amide bonds and one are different molecules, and the binary form
    cannot say so. Counts plus the full RDKit2D block is the configuration the
    literature reports topping or tying half of TDC's ADMET tasks and beating
    both MolFormer and MiniMol on MoleculeACE.

    An unparseable molecule returns all zeros rather than raising: the caller's
    indices must stay aligned with its label table, and a silently dropped row
    is far more dangerous than a featureless one.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fns = _descriptor_fns(desc_mode)
    width = n_bits + len(fns) + (AVALON_BITS + ERG_BITS if extra_fps else 0)
    out = np.zeros(width, dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out

    fp = (gen.GetCountFingerprintAsNumPy(mol) if counts
          else gen.GetFingerprintAsNumPy(mol))
    out[:n_bits] = fp.astype(np.float32)
    for i, (_, fn) in enumerate(fns):
        try:
            value = float(fn(mol))
        except Exception:  # noqa: BLE001 — RDKit descriptors raise several types
            value = 0.0
        # A few descriptors go infinite on pathological molecules and would
        # poison the standardisation for every other row.
        out[n_bits + i] = value if np.isfinite(value) else 0.0

    if extra_fps:
        base = n_bits + len(fns)
        out[base:base + AVALON_BITS] = _avalon(mol, AVALON_BITS)
        out[base + AVALON_BITS:] = _erg(mol)
    return out


def _worker(payload: tuple[list[str], dict]) -> np.ndarray:
    chunk, opts = payload
    return np.stack([featurise_one(s, **opts) for s in chunk])


def featurise(
    smiles: list[str],
    cache: str | Path | None = None,
    workers: int | None = None,
    chunk: int = 512,
    **opts,
) -> np.ndarray:
    """Featurise a list of SMILES, using and filling a cache keyed by content.

    The cache key is a hash of the SMILES list, so a changed dataset cannot
    silently reuse the wrong features — the failure mode that produces a model
    that trains fine and predicts nonsense.
    """
    # The key covers the featurisation options too: switching binary bits for
    # counts must not silently reuse the old matrix.
    spec = repr(sorted(opts.items()))
    key = hashlib.sha256(("\n".join(smiles) + spec).encode()).hexdigest()[:16]

    if cache is not None:
        cache = Path(cache)
        path = cache if cache.suffix == ".npz" else cache / f"feat-{key}.npz"
        if path.exists():
            with np.load(path) as handle:
                if handle.get("key", np.array(key)).item() != key:
                    print(f"[feat] cache {path} is for different SMILES; recomputing")
                else:
                    print(f"[feat] loaded {path}")
                    return handle["x"]
    else:
        path = None

    with ProcessPoolExecutor(max_workers=workers) as pool:
        chunks = [(smiles[i:i + chunk], opts) for i in range(0, len(smiles), chunk)]
        parts = list(pool.map(_worker, chunks))
    x = np.concatenate(parts).astype(np.float32)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, x=x, key=np.array(key))
        print(f"[feat] cached {x.shape} to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    return x


def descriptor_span(width: int, n_bits: int = 2048, desc_mode: str = "rdkit2d") -> slice:
    """Where the descriptor block sits, given the total feature width.

    The layout is [morgan | descriptors | avalon | ErG], so "everything after the
    fingerprint" stopped being the descriptor block the moment Avalon and ErG
    were appended. Assuming it did silently standardised 1,043 columns against
    216 statistics — it only surfaced as a broadcast error because the two
    matrices had different widths; with matching widths it would have z-scored
    count fingerprints against descriptor statistics and trained on quietly
    corrupted features.
    """
    n_desc = len(_descriptor_fns(desc_mode))
    if width == n_bits + n_desc:
        return slice(n_bits, width)                      # no extra fingerprints
    if width == n_bits + n_desc + AVALON_BITS + ERG_BITS:
        return slice(n_bits, n_bits + n_desc)            # Avalon/ErG follow
    raise ValueError(
        f"unrecognised feature width {width}: expected {n_bits + n_desc} or "
        f"{n_bits + n_desc + AVALON_BITS + ERG_BITS} for desc_mode={desc_mode!r}"
    )


def standardise_descriptors(
    x: np.ndarray, n_bits: int = 2048, stats: tuple[np.ndarray, np.ndarray] | None = None,
    desc_mode: str = "rdkit2d",
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Z-score the descriptor block, leaving every fingerprint block alone.

    Statistics come from the training split only; pass them back in for valid
    and test. Standardising over everything leaks the test distribution.
    """
    x = x.copy()
    span = descriptor_span(x.shape[1], n_bits, desc_mode)
    block = x[:, span]
    if stats is None:
        mean = block.mean(axis=0)
        std = block.std(axis=0)
        std[std < 1e-8] = 1.0
    else:
        mean, std = stats
    x[:, span] = (block - mean) / std
    return x, (mean, std)


def randomised_smiles(smiles: str, k: int = 10, seed: int = 0) -> list[str]:
    """K randomised (non-canonical) SMILES for the same molecule.

    Augmentation that is free at training time because the pool is precomputed.
    Returns fewer than k, or `[smiles]`, when RDKit cannot oblige.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    for _ in range(k * 3):  # oversample: randomisation collides on small molecules
        order = rng.permutation(mol.GetNumAtoms()).tolist()
        try:
            variant = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        except Exception:  # noqa: BLE001
            continue
        seen.add(variant)
        if len(seen) >= k:
            break
    return sorted(seen) or [smiles]
