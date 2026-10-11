"""What ExpansionRx actually contains, per endpoint — and how clean it is per representation.

Three questions the earlier verification left open, each of which changes the plan:

1. **7,608 molecules against a quoted 30,369 labels is ~4 labels per molecule.** That is consistent
   with a multi-endpoint panel, but the gross figure hides the per-endpoint picture, and the
   per-endpoint picture is what the acquisition ordering depends on. Above all: **does it carry
   half_life?** If not, H7 — the pre-registered test of the programme's central claim — can only run
   on the ChEMBL pull, which is the gated one. Ordering the clean source first would then leave the
   decisive experiment dependent on the source we deferred.

2. **Near-neighbour similarity was computed in ECFP space only.** That understates leakage for the
   feature set H5 actually selected: **ErG is a reduced-graph pharmacophore descriptor built to find
   scaffold hops** — pairs deliberately distant in ECFP and close in ErG are its design intent, not
   its failure mode. An audit run in ECFP certifies safety for a model that does not see ECFP.
   Computed here in Morgan, Avalon and ErG, with **the worst case governing**.

3. **The ChEMBL and ExpansionRx comparisons were not like-for-like** — ChEMBL was reported as
   scaffold recovery, ExpansionRx as nearest-neighbour. Test-scaffold recovery is computed here under
   the same `MurckoScaffoldSmiles` definition so the two can be read in one table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdReducedGraphs

from drugrx.data.splits import murcko_scaffold
from drugrx.data.standardise import standardise_many

REPOS = [
    "openadmet/openadmet-expansionrx-challenge-data",
    "openadmet/openadmet-expansionrx-challenge-train-data",
]

#: ExpansionRx assay column → our endpoint. Anything absent here is an endpoint
#: ExpansionRx cannot help with, which is the point of printing it.
COLUMN_TO_ENDPOINT = {
    "LogD": "lipophilicity",
    "KSOL": "solubility",
    "HLM CLint": "cl_micro",
    "MLM CLint": "cl_micro",
    "RLM CLint": "cl_micro",
    "Caco-2 Permeability Papp A>B": "caco2",
    "Caco-2 Permeability Efflux": "pgp",
    "MPPB": "ppbr",
    "MBPB": "ppbr",
    "MGMB": "ppbr",
}


def morgan(mol):
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(mol)


def avalon(mol):
    from rdkit.Avalon import pyAvalonTools

    return pyAvalonTools.GetAvalonFP(mol, nBits=1024)


def erg_vector(mol) -> np.ndarray:
    return np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)


def nn_bitwise(query: list[str], ref: list[str], fp_fn) -> np.ndarray:
    """Max Tanimoto from each query to any reference, for a bit/count fingerprint."""
    def fps(items):
        out = []
        for s in items:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                try:
                    out.append(fp_fn(m))
                except Exception:  # noqa: BLE001
                    pass
        return out

    r = fps(ref)
    if not r:
        return np.zeros(0)
    return np.array([max(DataStructs.BulkTanimotoSimilarity(f, r)) for f in fps(query)])


def nn_erg(query: list[str], ref: list[str]) -> np.ndarray:
    """ErG is a float vector, so similarity is cosine rather than Tanimoto.

    Reported on the same 0–1 scale so it can sit in the same table, but it is a
    different measure and a direct numeric comparison with Tanimoto is loose.
    """
    def vecs(items):
        out = []
        for s in items:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            v = erg_vector(m)
            n = np.linalg.norm(v)
            if n > 0:
                out.append(v / n)
        return np.array(out) if out else np.zeros((0, 315), dtype=np.float32)

    q, r = vecs(query), vecs(ref)
    if len(q) == 0 or len(r) == 0:
        return np.zeros(0)
    return (q @ r.T).max(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data/admet_multitask.parquet")
    ap.add_argument("--out", default="/content/out/expansionrx_profile.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sample", type=int, default=800)
    args = ap.parse_args()

    from datasets import load_dataset

    frames = []
    for repo in REPOS:
        try:
            ds = load_dataset(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] {repo}: {exc}", flush=True)
            continue
        for split, part in ds.items():
            df = part.to_pandas()
            df["__repo"], df["__split"] = repo, split
            frames.append(df)
    if not frames:
        print("[profile] nothing loaded")
        return 1
    erx = pd.concat(frames, ignore_index=True)

    # ── per-endpoint label counts ────────────────────────────────────
    counts: dict[str, int] = {}
    per_column = {}
    for col, endpoint in COLUMN_TO_ENDPOINT.items():
        if col not in erx.columns:
            continue
        n = int(erx[col].notna().sum())
        per_column[col] = n
        counts[endpoint] = counts.get(endpoint, 0) + n

    print("[profile] label counts by ExpansionRx column:")
    for col, n in sorted(per_column.items(), key=lambda kv: -kv[1]):
        print(f"    {col:32s} {n:7d}  → {COLUMN_TO_ENDPOINT[col]}", flush=True)
    print("[profile] totals by our endpoint:")
    for ep, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {ep:16s} {n:7d}", flush=True)
    total = sum(counts.values())
    print(f"[profile] {len(erx)} rows, {total} labels across {len(counts)} endpoints", flush=True)

    frame = pd.read_parquet(args.data)
    our_endpoints = {c[: -len("__fold")] for c in frame.columns if c.endswith("__fold")}
    missing = sorted(our_endpoints - set(counts))
    print(f"\n[profile] endpoints ExpansionRx CANNOT help with ({len(missing)}): {missing}",
          flush=True)
    carries_half_life = "half_life" in counts
    print(f"[profile] carries half_life: {carries_half_life}", flush=True)

    # ── cleanliness, per representation, per endpoint ────────────────
    smiles_col = "SMILES" if "SMILES" in erx.columns else None
    erx_smiles = [s for s in erx[smiles_col].tolist() if isinstance(s, str)] if smiles_col else []
    std = standardise_many(erx_smiles, workers=args.workers)
    erx_std = [s.smiles for s in std if s.ok]

    held = standardise_many(frame["smiles"].tolist(), workers=args.workers)
    frame = frame.assign(key=[s.inchikey if s.ok else "" for s in held])
    good = frame[frame["key"] != ""]

    rng = np.random.default_rng(0)
    sample = erx_std
    if len(sample) > args.sample:
        sample = [sample[i] for i in rng.choice(len(sample), args.sample, replace=False)]

    rows = []
    for endpoint in sorted(our_endpoints):
        col = f"{endpoint}__fold"
        test_smiles = good.loc[good[col] == "test", "smiles"].tolist()
        if len(test_smiles) < 20:
            continue

        raw_test = [murcko_scaffold(s) for s in test_smiles]
        raw_erx = [murcko_scaffold(s) for s in erx_std]
        test_scaffolds = {s for s in raw_test if s}
        erx_scaffolds = {s for s in raw_erx if s}
        covered = test_scaffolds & erx_scaffolds

        nn_m = nn_bitwise(sample, test_smiles, morgan)
        nn_a = nn_bitwise(sample, test_smiles, avalon)
        nn_e = nn_erg(sample, test_smiles)

        def frac(v, t):
            return round(float((v > t).mean()), 4) if len(v) else None

        row = {
            "endpoint": endpoint,
            "test_size": len(test_smiles),
            "test_scaffolds": len(test_scaffolds),
            "test_scaffolds_recovered": len(covered),
            "frac_test_scaffolds_recovered": round(len(covered) / max(1, len(test_scaffolds)), 4),
            "morgan_gt_0.4": frac(nn_m, 0.4),
            "avalon_gt_0.4": frac(nn_a, 0.4),
            "erg_cos_gt_0.9": frac(nn_e, 0.9),
            "morgan_mean": round(float(nn_m.mean()), 4) if len(nn_m) else None,
            "avalon_mean": round(float(nn_a.mean()), 4) if len(nn_a) else None,
            "erg_mean": round(float(nn_e.mean()), 4) if len(nn_e) else None,
        }
        rows.append(row)
        print(f"{endpoint:16s} scaffold recovery {row['frac_test_scaffolds_recovered']:6.1%} | "
              f"morgan>0.4 {row['morgan_gt_0.4']:6.1%} | avalon>0.4 {row['avalon_gt_0.4']:6.1%} | "
              f"erg cos>0.9 {row['erg_cos_gt_0.9']:6.1%}", flush=True)

    Path(args.out).write_text(json.dumps(
        {"label_counts": counts, "per_column": per_column, "rows": len(erx),
         "carries_half_life": carries_half_life, "cannot_help": missing,
         "cleanliness": rows}, indent=2))
    print(f"\n[profile] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
