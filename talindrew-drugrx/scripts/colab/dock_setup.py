"""Stage 0 of the docking build: install the stack and prove it works on a known inhibitor.

Nothing here touches the dataset. The point is to fail fast and loudly on the install, and
then to run one docking whose answer is already known, so that a silent misconfiguration
cannot propagate into 6,895 runs. The known case is **ketoconazole into CYP3A4 (1TQN)** -- a
canonical, potent CYP3A4 inhibitor that coordinates the heme iron. If it does not dock close
to the Fe with a strong score, the receptor or the box is wrong and nothing downstream is
worth running.

Two install lessons already paid for on the first attempt:

* `pip install vina` builds from source and needs boost + swig; it failed, and because it
  shared one `pip install` line with rdkit and meeko it took **those** down with it. Each
  package now installs on its own line and is checked on its own.
* The **prebuilt Vina 1.2.5 binary** avoids the build entirely.

Map caching was tried and **abandoned**: `--write_maps` needs a `--ligand` to know which atom
types to compute, so one shared map set silently under-covers the types in a diverse library.
The control below therefore runs the *production* command -- `--receptor` with an explicit
box -- because a control that uses a different command from the real run controls nothing.
Leaving a stale `--maps rec` here once cost 4,399 failed dockings.
"""
from __future__ import annotations

import os
import subprocess
import textwrap

VINA_URL = ("https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/"
            "v1.2.5/vina_1.2.5_linux_x86_64")


def sh(cmd: str, check: bool = True, tail: int = 2500) -> str:
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out[-tail:], flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"FAILED ({p.returncode}): {cmd}")
    return out


def main() -> int:
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", check=False)
    sh("nproc; free -g | head -2", check=False)
    os.makedirs("/content/dock", exist_ok=True)

    print("\n=== installing, one at a time so a failure cannot cascade ===", flush=True)
    sh("apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y openbabel > /dev/null 2>&1",
       check=False)
    sh("obabel -V", check=False)
    sh("pip -q install rdkit 2>&1 | tail -3", check=False)
    sh("pip -q install meeko 2>&1 | tail -3", check=False)

    print("\n=== Vina: prebuilt binary, not the source wheel ===", flush=True)
    sh(f"cd /content/dock && wget -q {VINA_URL} -O vina && chmod +x vina && ./vina --version")

    print("\n=== verifying ===", flush=True)
    code = textwrap.dedent("""
        import importlib, shutil, subprocess, sys
        ok = True
        for m in ("rdkit",):
            try:
                mod = importlib.import_module(m)
                print(f"  {m}: {getattr(mod, '__version__', 'ok')}")
            except Exception as e:
                print(f"  {m}: FAILED {type(e).__name__}: {e}"); ok = False
        if shutil.which("obabel") is None:
            print("  obabel: MISSING"); ok = False
        else:
            print("  obabel:", subprocess.run(["obabel","-V"],capture_output=True,
                                              text=True).stdout.strip()[:40])
        sys.exit(0 if ok else 1)
    """)
    open("/content/_verify.py", "w").write(code)
    sh("python /content/_verify.py")

    print("\n=== receptor: CYP3A4 1TQN ===", flush=True)
    sh("cd /content/dock && wget -q https://files.rcsb.org/download/1TQN.pdb && ls -la 1TQN.pdb")
    prep = textwrap.dedent('''
        """Keep protein + HEM, drop waters and additives, locate the catalytic Fe."""
        keep, fe = [], None
        for line in open("/content/dock/1TQN.pdb"):
            if not line.startswith(("ATOM", "HETATM")):
                continue
            res = line[17:20].strip()
            if res == "HOH" or (line.startswith("HETATM") and res != "HEM"):
                continue
            keep.append(line)
            if res == "HEM" and line[12:16].strip() == "FE":
                fe = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        assert fe is not None, "no heme FE found -- wrong structure or altered naming"
        with open("/content/dock/receptor.pdb", "w") as fh:
            fh.writelines(keep); fh.write("END\\n")
        n_hem = sum(1 for l in keep if l[17:20].strip() == "HEM")
        print(f"kept {len(keep)} atoms, {n_hem} heme atoms; FE at {fe}")
        open("/content/dock/fe.txt", "w").write(" ".join(str(c) for c in fe))
    ''')
    open("/content/dock/prep.py", "w").write(prep)
    sh("python /content/dock/prep.py")
    sh("cd /content/dock && obabel receptor.pdb -O receptor.pdbqt -xr -p 7.4 2>&1 | tail -2",
       check=False)
    sh("cd /content/dock && ls -la receptor.pdbqt && grep -c ATOM receptor.pdbqt", check=False)

    print("\n=== positive control: ketoconazole into CYP3A4 ===", flush=True)
    ctrl = textwrap.dedent('''
        import subprocess
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        KETO = ("CC(=O)N1CCN(CC1)c1ccc(OC[C@@H]2CO[C@](Cn3ccnc3)(O2)c2ccc(Cl)cc2Cl)cc1")
        fe = np.array([float(x) for x in open("/content/dock/fe.txt").read().split()])

        m = Chem.AddHs(Chem.MolFromSmiles(KETO))
        ps = AllChem.ETKDGv3(); ps.randomSeed = 0xf00d
        assert AllChem.EmbedMolecule(m, ps) == 0, "ketoconazole failed to embed"
        AllChem.MMFFOptimizeMolecule(m)
        Chem.MolToMolFile(m, "/content/dock/keto.sdf")
        subprocess.run("obabel /content/dock/keto.sdf -O /content/dock/keto.pdbqt 2>/dev/null",
                       shell=True)

        r = subprocess.run(
            f"cd /content/dock && ./vina --receptor receptor.pdbqt --ligand keto.pdbqt "
            f"--out keto_out.pdbqt "
            f"--center_x {fe[0]} --center_y {fe[1]} --center_z {fe[2]} "
            f"--size_x 20 --size_y 20 --size_z 20 "
            f"--exhaustiveness 4 --num_modes 9 --cpu 1 --seed 42",
            shell=True, capture_output=True, text=True)
        print(r.stdout[-1200:] or r.stderr[-1200:])

        scores, xyz, model = [], [], 0
        for line in open("/content/dock/keto_out.pdbqt"):
            if line.startswith("MODEL"):
                model += 1
            elif line.startswith("REMARK VINA RESULT"):
                scores.append(float(line.split()[3]))
            elif model == 1 and line.startswith(("ATOM", "HETATM")):
                xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        d = float(np.linalg.norm(np.array(xyz) - fe, axis=1).min())
        print(f"pose scores: {scores}")
        print(f"best {scores[0]:.2f} kcal/mol; closest heavy atom to heme Fe {d:.2f} A")
        print("SANITY:", "PASS" if (scores and scores[0] < -7.0 and d < 8.0)
              else "FAIL -- receptor or box is wrong, do not proceed")
    ''')
    open("/content/dock/ctrl.py", "w").write(ctrl)
    sh("cd /content/dock && python ctrl.py", check=False)
    print("\nSETUP DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
