"""Stage 2 groundwork: receptors and positive controls for the other three CYPs.

This does **not** commit to stage 2 — `DOCKING-PLAN.md` says to decide that on the measured
CYP3A4 number, not on enthusiasm. It removes the latency, and more importantly it answers now,
while there is time to react, whether each receptor is even usable. A box that is wrong is far
cheaper to discover against a known inhibitor than after 6,895 dockings, which is the lesson
the `--maps` failure already charged once.

Each isoform gets the same treatment CYP3A4 got, and the control runs the **production
command** — `--receptor` with an explicit box at exhaustiveness 4 — because a control using a
different command from the real run controls nothing.

| isoform | PDB | why this structure | positive control | why this ligand |
|---|---|---|---|---|
| CYP1A2 | 2HI4 | the only human CYP1A2 crystal structure, with α-naphthoflavone bound | α-naphthoflavone | the co-crystallised ligand, so a correct box must recover it |
| CYP2C9 | 1OG5 | CYP2C9 with warfarin; 1R9O is the flurbiprofen form | sulfaphenazole | the canonical *selective* CYP2C9 inhibitor |
| CYP2D6 | 3QM4 | CYP2D6 with prinomastat, a liganded form | quinidine | the canonical *selective* CYP2D6 inhibitor |

α-naphthoflavone is the strongest of the three tests because it is 2HI4's own co-crystallised
ligand: re-docking it is a redocking control, and failing it means the receptor or box is
wrong rather than the chemistry being hard.

The bar is deliberately looser than CYP3A4's. Ketoconazole gave −8.77 kcal/mol at 3.82 Å from
the heme Fe, but these three ligands are smaller and the selective inhibitors are not all
heme-coordinating, so a pass is **score < −6.5 and closest heavy atom < 10 Å of the Fe** —
enough to show the ligand reaches the active site, which is all a box check can establish.
Note that Vina has no metal term (ketoconazole's imidazole docked 13.4 Å from the CYP3A4 iron),
so distance-to-Fe here is a *box* check, not evidence of coordination.

    colab exec -s <session> -f scripts/colab/dock_setup_iso.py
"""
from __future__ import annotations

import os
import subprocess
import textwrap

ISO = {
    "CYP1A2": {"pdb": "2HI4", "ctrl": "O=c1cc(-c2ccccc2)oc2c1ccc1ccccc12",
               "ctrl_name": "alpha-naphthoflavone"},
    "CYP2C9": {"pdb": "1OG5", "ctrl": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)n(-c2ccccc2)n1",
               "ctrl_name": "sulfaphenazole"},
    "CYP2D6": {"pdb": "3QM4", "ctrl": "C=C[C@H]1CN2CC[C@H]1C[C@@H]2[C@H](O)c1ccnc2ccc(OC)cc12",
               "ctrl_name": "quinidine"},
}
BOX = 20
EXH = 4
#: Heme can be deposited under several residue names. 1OG5 (CYP2C9) uses **HEC**, not HEM,
#: and filtering on "HEM" alone silently produced "no heme FE found -- unusable" for a
#: structure whose iron is plainly there. The positive control is what surfaced it.
HEME = ("HEM", "HEC", "HEA", "HEB", "HEV")


def sh(cmd: str, check: bool = False, tail: int = 1200) -> str:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise SystemExit(f"FAILED: {cmd}\n{out[-tail:]}")
    return out


def main() -> int:
    for iso, spec in ISO.items():
        pdb = spec["pdb"]
        d = f"/content/dock/{iso}"
        os.makedirs(d, exist_ok=True)
        print(f"\n=== {iso}: {pdb} ===", flush=True)
        sh(f"cd {d} && wget -q https://files.rcsb.org/download/{pdb}.pdb")
        if not os.path.exists(f"{d}/{pdb}.pdb"):
            print(f"  {iso}: DOWNLOAD FAILED"); continue

        prep = textwrap.dedent(f'''
            keep, fe = [], None
            seen = set()
            for line in open("{d}/{pdb}.pdb"):
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                res, chain = line[17:20].strip(), line[21]
                # Several of these crystals have multiple chains; keep the first only, or
                # the box is centred between two copies of the active site.
                seen.add(chain)
                if chain != sorted(seen)[0]:
                    continue
                if res == "HOH" or (line.startswith("HETATM") and res not in {HEME!r}):
                    continue
                keep.append(line)
                if res in {HEME!r} and line[12:16].strip() == "FE":
                    fe = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if fe is None:
                print("  NO HEME FE FOUND -- unusable")
            else:
                open("{d}/receptor.pdb", "w").writelines(keep)
                open("{d}/fe.txt", "w").write(" ".join(str(c) for c in fe))
                n_hem = sum(1 for l in keep if l[17:20].strip() in {HEME!r})
                print(f"  kept {{len(keep)}} atoms, {{n_hem}} heme, chains {{sorted(seen)}}, FE {{fe}}")
        ''')
        open(f"{d}/prep.py", "w").write(prep)
        print(sh(f"python {d}/prep.py").strip(), flush=True)
        if not os.path.exists(f"{d}/fe.txt"):
            continue

        sh(f"cd {d} && obabel receptor.pdb -O receptor.pdbqt -xr -p 7.4 2>&1 | tail -1")
        fe = open(f"{d}/fe.txt").read().split()

        ctrl = textwrap.dedent(f'''
            import subprocess
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import AllChem
            fe = np.array([float(x) for x in open("{d}/fe.txt").read().split()])
            m = Chem.AddHs(Chem.MolFromSmiles({spec["ctrl"]!r}))
            ps = AllChem.ETKDGv3(); ps.randomSeed = 0xC0FFEE
            assert AllChem.EmbedMolecule(m, ps) == 0, "control failed to embed"
            AllChem.MMFFOptimizeMolecule(m)
            Chem.MolToMolFile(m, "{d}/ctrl.sdf")
            subprocess.run("obabel {d}/ctrl.sdf -O {d}/ctrl.pdbqt 2>/dev/null", shell=True)
            r = subprocess.run(
                "cd /content/dock && ./vina --receptor {d}/receptor.pdbqt "
                "--ligand {d}/ctrl.pdbqt --out {d}/ctrl_out.pdbqt "
                "--center_x {fe[0]} --center_y {fe[1]} --center_z {fe[2]} "
                "--size_x {BOX} --size_y {BOX} --size_z {BOX} "
                "--exhaustiveness {EXH} --num_modes 9 --cpu 1 --seed 42",
                shell=True, capture_output=True, text=True)
            scores, xyz, model = [], [], 0
            try:
                for line in open("{d}/ctrl_out.pdbqt"):
                    if line.startswith("MODEL"): model += 1
                    elif line.startswith("REMARK VINA RESULT"): scores.append(float(line.split()[3]))
                    elif model == 1 and line.startswith(("ATOM","HETATM")):
                        xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            except FileNotFoundError:
                print("  no output:", (r.stderr or r.stdout)[-200:]); raise SystemExit
            d_fe = float(np.linalg.norm(np.array(xyz) - fe, axis=1).min())
            ok = scores and scores[0] < -6.5 and d_fe < 10.0
            print(f"  {spec["ctrl_name"]}: best {{scores[0]:.2f}} kcal/mol, "
                  f"closest heavy atom {{d_fe:.2f}} A from Fe  -> "
                  + ("PASS" if ok else "FAIL -- receptor or box wrong"))
        ''')
        open(f"{d}/ctrl.py", "w").write(ctrl)
        print(sh(f"python {d}/ctrl.py").strip(), flush=True)

    print("\nISO SETUP DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
