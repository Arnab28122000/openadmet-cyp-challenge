"""Fix the map strategy and prove the receptor with a known inhibitor.

`--write_maps` failed because Vina 1.2.x needs a `--ligand` to know *which atom types* to
build grids for, and a shared map set is only valid for ligands whose types it covers. Our
6,895 compounds span C/N/O/S/P/F/Cl/Br/I, so a map set built from one ligand would silently
not cover the rest -- the dangerous failure, not the loud one.

So this measures the alternative honestly: run Vina with `--receptor` directly, which rebuilds
the grid per ligand, and time it. If a single docking costs ~10-20 s then 6,895 of them over
12 cores is 2-3 hours, which is affordable and removes a whole class of correctness risk. The
decision is made on the measured number, not on the assumption that caching must be faster.

The positive control is **ketoconazole**, a canonical potent CYP3A4 inhibitor that coordinates
the heme iron. Two independent things must both hold or the receptor/box is wrong:
a strong score, and a best pose that actually reaches the iron.
"""
from __future__ import annotations

import subprocess
import textwrap

D = "/content/dock"


def sh(cmd: str, tail: int = 2000) -> str:
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-tail:], flush=True)
    return out


def main() -> int:
    ctrl = textwrap.dedent('''
        import subprocess, time
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        D = "/content/dock"
        fe = np.array([float(x) for x in open(f"{D}/fe.txt").read().split()])
        box = 24

        # Ketoconazole, plus two negatives for contrast: caffeine (not a 3A4 inhibitor,
        # small) and ritonavir (a very potent 3A4 inhibitor, large). A receptor that is
        # working should rank ritonavir and ketoconazole well ahead of caffeine.
        LIGS = {
          "ketoconazole": "CC(=O)N1CCN(CC1)c1ccc(OC[C@@H]2CO[C@](Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1",
          "ritonavir": ("CC(C)c1nc(CN(C)C(=O)N[C@@H](CCN)C(=O)N[C@@H](Cc2ccccc2)C[C@H](O)"
                        "[C@H](Cc2ccccc2)NC(=O)OCc2cncs2)cs1"),
          "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        }
        for name, smi in LIGS.items():
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            ps = AllChem.ETKDGv3(); ps.randomSeed = 0xf00d
            if AllChem.EmbedMolecule(m, ps) != 0:
                ps.useRandomCoords = True; AllChem.EmbedMolecule(m, ps)
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            Chem.MolToMolFile(m, f"{D}/{name}.sdf")
            subprocess.run(f"obabel {D}/{name}.sdf -O {D}/{name}.pdbqt 2>/dev/null", shell=True)

            t0 = time.time()
            r = subprocess.run(
                f"cd {D} && ./vina --receptor receptor.pdbqt --ligand {name}.pdbqt "
                f"--out {name}_out.pdbqt --center_x {fe[0]} --center_y {fe[1]} "
                f"--center_z {fe[2]} --size_x {box} --size_y {box} --size_z {box} "
                f"--exhaustiveness 8 --num_modes 9 --cpu 1 --seed 42",
                shell=True, capture_output=True, text=True)
            el = time.time() - t0
            if r.returncode != 0:
                print(f"{name}: VINA FAILED rc={r.returncode}\\n{(r.stderr or r.stdout)[-600:]}")
                continue

            scores, xyz, model = [], [], 0
            for line in open(f"{D}/{name}_out.pdbqt"):
                if line.startswith("MODEL"): model += 1
                elif line.startswith("REMARK VINA RESULT"): scores.append(float(line.split()[3]))
                elif model == 1 and line.startswith(("ATOM", "HETATM")):
                    el_ = (line[77:79].strip() or "C")
                    if el_ != "H":
                        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            d = float(np.linalg.norm(np.array(xyz) - fe, axis=1).min()) if xyz else float("nan")
            print(f"{name:14} best {scores[0]:6.2f}  n_poses {len(scores)}  "
                  f"Fe dist {d:5.2f} A  heavy {len(xyz):3d}  [{el:.1f}s per docking]")

        print("\\nDecision input: if a single docking is ~10-20 s, 6,895 over 12 cores is")
        print("2-3 h with NO shared-map correctness risk, which is the right trade.")
    ''')
    open(f"{D}/ctrl2.py", "w").write(ctrl)
    sh(f"cd {D} && python ctrl2.py", tail=4000)
    print("\nCONTROL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
