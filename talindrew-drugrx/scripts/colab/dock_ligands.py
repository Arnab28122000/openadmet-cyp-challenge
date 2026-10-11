"""Stage 1: conformers + PDBQT for every compound, and the 3D descriptor block for free.

6,895 molecules (6,145 train + 750 blinded test). Two outputs:

* `/content/dock/ligands/*.pdbqt` — one 3D conformer per molecule, ready for Vina.
* `/content/dock/desc3d.npz` — RDKit 3D descriptors (PMI/NPR/radius of gyration/asphericity/
  eccentricity/inertial shape factor/spherocity/PBF), computed from that same conformer.

The descriptor block is the **internal control** the plan calls for. It is ligand-only, so it
is identical across the four isoforms and cannot carry isoform-specific information. If it
explains as much of any later gain as the docking scores do, then the gain is 3D shape and the
isoform-specificity argument — the whole reason this build was authorised — is wrong.

Conformer generation is ETKDGv3 + MMFF, one conformer, fixed seed. One conformer is a real
approximation: a flexible ligand's bioactive conformer is not necessarily its lowest-energy
one. Vina samples torsions itself during docking, so the input conformer mainly sets bond
lengths and angles, which is the part Vina holds rigid. Recorded as a known limitation rather
than hidden.

Molecules that fail to embed are written to `failed.txt` and carry NaN downstream — never
silently dropped, because a silent drop would misalign the feature matrix against the label
matrix, which is the defect class this programme keeps finding.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT = "/content/dock/ligands"
DESC_FNS = None


def _descriptors(mol):
    """3D shape descriptors from the embedded conformer."""
    from rdkit.Chem import Descriptors3D, rdMolDescriptors
    f = {}
    for name in ("PMI1", "PMI2", "PMI3", "NPR1", "NPR2", "RadiusOfGyration",
                 "Asphericity", "Eccentricity", "InertialShapeFactor",
                 "SpherocityIndex"):
        try:
            f[name] = float(getattr(Descriptors3D, name)(mol))
        except Exception:
            f[name] = float("nan")
    try:
        f["PBF"] = float(rdMolDescriptors.CalcPBF(mol))
    except Exception:
        f["PBF"] = float("nan")
    return f


def one(arg):
    idx, smi = arg
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    path = f"{OUT}/{idx:05d}.pdbqt"
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return idx, None, "parse"
        m = Chem.AddHs(m)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 0xC0FFEE
        ps.useSmallRingTorsions = True
        if AllChem.EmbedMolecule(m, ps) != 0:
            ps.useRandomCoords = True
            if AllChem.EmbedMolecule(m, ps) != 0:
                return idx, None, "embed"
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        d = _descriptors(m)
        sdf = f"/tmp/{idx:05d}.sdf"
        Chem.MolToMolFile(m, sdf)
        r = subprocess.run(f"obabel {sdf} -O {path} 2>/dev/null",
                           shell=True, capture_output=True)
        os.unlink(sdf)
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            return idx, None, "pdbqt"
        return idx, d, None
    except Exception as e:                                        # noqa: BLE001
        return idx, None, f"{type(e).__name__}"


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    smiles = json.load(open("/content/dock/smiles.json"))
    n = len(smiles)
    print(f"[lig] {n} molecules, {os.cpu_count()} cores", flush=True)

    t0 = time.time()
    keys, rows, failed = None, [], []
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count())) as ex:
        for idx, d, err in ex.map(one, list(enumerate(smiles)), chunksize=32):
            done += 1
            if err:
                failed.append((idx, err))
                rows.append((idx, None))
            else:
                if keys is None:
                    keys = sorted(d)
                rows.append((idx, [d[k] for k in keys]))
            if done % 500 == 0:
                el = time.time() - t0
                print(f"[lig] {done}/{n}  {el:.0f}s  eta {el/done*(n-done):.0f}s  "
                      f"failed {len(failed)}", flush=True)

    rows.sort()
    width = len(keys) if keys else 0
    X = np.full((n, width), np.nan, dtype=np.float32)
    for idx, v in rows:
        if v is not None:
            X[idx] = v
    np.savez_compressed("/content/dock/desc3d.npz", x=X, names=np.array(keys or []))
    with open("/content/dock/failed.txt", "w") as fh:
        for idx, err in failed:
            fh.write(f"{idx}\t{err}\n")

    have = len([f for f in os.listdir(OUT) if f.endswith(".pdbqt")])
    print(f"\n[lig] {have}/{n} pdbqt written, {len(failed)} failed "
          f"({time.time()-t0:.0f}s)", flush=True)
    if failed:
        from collections import Counter
        print("[lig] failure modes:", Counter(e for _, e in failed).most_common(), flush=True)
    print(f"[lig] descriptor block {X.shape}, "
          f"{int(np.isfinite(X).all(1).sum())} rows fully finite", flush=True)
    print("LIGANDS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
