"""Extract CheMeleon's frozen 2048-d molecular fingerprint.

Research step 2, and the best gain per GPU-hour on the list. The idea is not to
fine-tune anything: run every molecule through CheMeleon's pretrained D-MPNN
encoder once, keep the 2048-d output as extra columns, and hand it to the same
LightGBM that already beats our neural net. That is MapLight+GNN with a better
GNN, and MapLight's own published ablation moved it from top-1 on 6/22 endpoints
to 11/22 by adding a far weaker (300-column GIN) fingerprint.

Why frozen rather than fine-tuned: CheMeleon's released fine-tuning ablation,
ranked by minimum validation loss, has **every TDC classification endpoint
preferring a frozen or delayed-unfreeze encoder** — none prefers the end-to-end
fine-tune its headline numbers used. Frozen is also minutes of GPU instead of
hours, and it cannot overfit, which matters at our sample sizes.

CheMeleon is MIT (weights Zenodo 10.5281/zenodo.15426600, pretraining data
CC-BY-4.0, PubChem public domain), so this is clean for commercial use — unlike
ChemFM (CC BY-NC despite an HF tag saying MIT), MolE (non-commercial) or
ChemBERTa (no declared licence at all).

    python scripts/make_driver.py --detach chemeleon_fp.py --data /content/data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")

#: The foundation weights Chemprop downloads for `--from-foundation CheMeleon`.
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def load_encoder(device: str):
    """Load CheMeleon's message-passing encoder with its pretrained weights."""
    import torch
    from chemprop.nn import BondMessagePassing

    cache = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request

        print(f"[chemeleon] downloading weights to {cache}", flush=True)
        urllib.request.urlretrieve(CHEMELEON_URL, cache)

    blob = torch.load(cache, weights_only=True, map_location=device)
    # The checkpoint carries the architecture it was trained with; constructing
    # the module from our own guesses would load the weights into a differently
    # shaped network and silently produce nonsense.
    encoder = BondMessagePassing(**blob["hyper_parameters"])
    encoder.load_state_dict(blob["state_dict"])
    encoder.eval().to(device)
    print(f"[chemeleon] encoder loaded: output dim {encoder.output_dim}", flush=True)
    return encoder


def featurise(smiles: list[str], device: str, batch_size: int = 256) -> np.ndarray:
    import torch
    from chemprop.data import BatchMolGraph, MoleculeDatapoint, MoleculeDataset
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

    from chemprop.nn import MeanAggregation

    encoder = load_encoder(device)
    # Mean over a molecule's atoms: CheMeleon's own recipe, and the only pooling that
    # keeps the vector comparable across molecules of different size (sum would make
    # the embedding scale with heavy-atom count, which is mostly what ECFP+descriptors
    # already encode).
    agg = MeanAggregation()
    featurizer = SimpleMoleculeMolGraphFeaturizer()

    out = np.zeros((len(smiles), encoder.output_dim), dtype=np.float32)
    failures = 0

    for start in range(0, len(smiles), batch_size):
        chunk = smiles[start:start + batch_size]
        graphs, keep = [], []
        for i, smi in enumerate(chunk):
            try:
                point = MoleculeDatapoint.from_smi(smi)
                graphs.append(featurizer(point.mol))
                keep.append(i)
            except Exception:  # noqa: BLE001
                # Leave the row as zeros rather than dropping it: the caller's
                # indices must stay aligned with its label table.
                failures += 1
        if not graphs:
            continue
        batch = BatchMolGraph(graphs)
        batch.to(device)
        with torch.no_grad():
            # `BondMessagePassing` returns ATOM-level representations: for 6 molecules
            # holding 43 atoms between them it returns (43, 2048), not (6, 2048). The
            # molecule-level vector only exists after aggregating over `batch.batch`,
            # which maps each atom row to its molecule.
            #
            # Without this, `vectors[j]` indexed ATOMS while `out[start + i]` indexed
            # MOLECULES, so every "fingerprint" was one arbitrary atom's embedding
            # belonging to some other molecule. It failed silently and looked healthy:
            # right shape, no NaNs, no zero rows. What gave it away was that a
            # molecule's nearest neighbour in the embedding had **Tanimoto 0.132 --
            # exactly the random-pair baseline** (and 93.7% of entries were zero).
            # Any future change here must keep `test_chemeleon_embeds_similar_molecules
            # _together` passing; a shape check cannot catch this.
            vectors = agg(encoder(batch), batch.batch).cpu().numpy()
        assert vectors.shape[0] == len(keep), (
            f"aggregation returned {vectors.shape[0]} rows for {len(keep)} molecules")
        for j, i in enumerate(keep):
            out[start + i] = vectors[j]

        if start % (batch_size * 20) == 0:
            print(f"[chemeleon] {start + len(chunk)}/{len(smiles)}", flush=True)

    if failures:
        print(f"[chemeleon] {failures} molecules failed; rows left as zeros", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--out", default="/content/data/chemeleon.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    frame = pd.read_parquet(Path(args.data) / "admet_multitask.parquet")
    smiles = frame["smiles"].tolist()
    print(f"[chemeleon] {len(smiles)} molecules", flush=True)

    vectors = featurise(smiles, args.device, args.batch_size)
    nonzero = int((vectors.any(axis=1)).sum())
    print(f"[chemeleon] produced {vectors.shape}, {nonzero} non-empty rows", flush=True)

    np.savez_compressed(args.out, x=vectors, smiles=np.array(smiles, dtype=object))
    print(f"[chemeleon] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
