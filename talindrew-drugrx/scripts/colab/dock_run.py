"""Stage 2: dock every ligand into CYP3A4 and extract more than one number per run.

Throughput comes from the runs being embarrassingly parallel: they are spread over all vCPUs
with one Vina process each at `--cpu 1`, which is more efficient than fewer processes each
threaded. Map caching is **not** used -- `--write_maps` only builds grids for the atom types
of the ligand it is given, so one shared map set silently under-covers a diverse library.
Every run passes `--receptor` and an explicit box and rebuilds its own grid.

**Checkpointing is not optional, and the checkpoint must leave the VM.** Results append to a
JSONL after every ligand and a restart skips what is already there -- but that only survives a
dead *process*. On 2026-10-09 the VM itself was reaped mid-run and took the JSONL with it,
losing 4.6 h, because the kernel goes idle the moment `colab exec` returns and no keepalive was
running. A long run needs all three: the append-only JSONL, `scripts/colab_keepalive.sh`, and
`scripts/dock_pull.sh` copying the JSONL back to the laptop every few minutes.

## What gets extracted, and why not just the top score

A single number from a noisy scoring function is the weakest possible use of a docking run, and
Vina's absolute scores are not affinities -- particularly here, since **Vina scores Fe
coordination poorly** and a CYP's catalytic site is defined by that iron. So each ligand yields:

* `best`, `mean_top3`, `spread` over the retained poses -- the spread is a crude confidence:
  nine poses that agree is a different situation from nine scattered across a large pocket.
* `fe_dist_best`, `fe_dist_min` -- how close the ligand's nearest heavy atom gets to the heme
  iron. Mechanistically direct for a CYP, where inhibition frequently means coordinating or
  occluding the iron, and it is information the score itself handles badly.
* `n_poses`, `n_heavy`, and `ligand_efficiency = best / n_heavy` -- raw score correlates
  strongly with size, so a size-normalised version is usually the more informative feature and
  the pair lets the model use either.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

D = "/content/dock"
LIG = f"{D}/ligands"
OUTJL = f"{D}/dock_cyp3a4.jsonl"
EXH = int(os.environ.get("DOCK_EXH", "4"))
#: Per-ligand wall clock cap. The set is median 4 rotatable bonds and p99 = 11, but one
#: compound has 34 -- and a 22-torsion ritonavir stalled a control for 25 minutes. The cap
#: stops the flexible tail from holding a worker hostage; capped ligands record an error and
#: carry NaN, which the evaluation handles, rather than being silently dropped.
TIMEOUT = int(os.environ.get("DOCK_TIMEOUT", "300"))
#: Settings chosen by measurement, not assumption (scripts/colab timing run, 6 ligands
#: spanning the real size range, timed AND rank-compared against a box 24 / exh 8
#: reference):
#:   24/8  48.9 s  7.8 h   reference
#:   20/4  28.9 s  4.6 h   rank rho 0.657, mean |dscore| 0.27  <- chosen
#:   18/4  32.1 s  5.1 h   rank rho -0.257, |dscore| 0.80 -- SLOWER AND WRONG; an 18 A box
#:                         truncates CYP3A4's pocket, so the box cannot be shrunk further
#:   20/2  15.1 s  2.4 h   ties 20/4 on rank but carries more stochastic noise, and
#:                         attenuating a signal we are trying to detect at +0.02-0.05 to
#:                         save two hours is the wrong trade.
BOX = 20
FE = [float(x) for x in open(f"{D}/fe.txt").read().split()]


def parse(path: str, fe: np.ndarray) -> dict:
    scores, models, cur = [], [], []
    for line in open(path):
        if line.startswith("MODEL"):
            cur = []
        elif line.startswith("REMARK VINA RESULT"):
            scores.append(float(line.split()[3]))
        elif line.startswith(("ATOM", "HETATM")):
            el = line[77:79].strip() or line[12:16].strip()[:1]
            if el != "H":
                cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL"):
            models.append(np.asarray(cur, dtype=float))
    if not scores or not models:
        return {}
    d_each = [float(np.linalg.norm(m - fe, axis=1).min()) for m in models if len(m)]
    top3 = scores[:3]
    return {
        "best": scores[0],
        "mean_top3": float(np.mean(top3)),
        "spread": float(np.max(scores) - np.min(scores)) if len(scores) > 1 else 0.0,
        "n_poses": len(scores),
        "fe_dist_best": d_each[0] if d_each else float("nan"),
        "fe_dist_min": float(np.min(d_each)) if d_each else float("nan"),
        "n_heavy": int(len(models[0])),
        "lig_eff": scores[0] / max(len(models[0]), 1),
    }


def one(idx: int) -> dict:
    lig = f"{LIG}/{idx:05d}.pdbqt"
    if not os.path.exists(lig):
        return {"i": idx, "err": "no_ligand"}
    out = f"/tmp/o{idx:05d}.pdbqt"
    # NOT --maps: map caching was abandoned because `--write_maps` only builds grids for the
    # atom types of the ligand that generated them, and this set spans C/N/O/S/P/F/Cl/Br/I.
    # `--receptor` rebuilds the grid per ligand, which costs more and cannot be silently wrong.
    cmd = (f"cd {D} && ./vina --receptor receptor.pdbqt --ligand {lig} --out {out} "
           f"--center_x {FE[0]} --center_y {FE[1]} --center_z {FE[2]} "
           f"--size_x {BOX} --size_y {BOX} --size_z {BOX} "
           f"--exhaustiveness {EXH} --num_modes 9 --cpu 1 --seed 42")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        return {"i": idx, "err": f"timeout_{TIMEOUT}s"}
    if r.returncode != 0 or not os.path.exists(out):
        return {"i": idx, "err": (r.stderr or r.stdout or "vina")[-180:]}
    fe = np.array([float(x) for x in open(f"{D}/fe.txt").read().split()])
    try:
        rec = parse(out, fe)
    except Exception as e:                                        # noqa: BLE001
        rec = {}
        r = {"err": f"{type(e).__name__}"}
    finally:
        if os.path.exists(out):
            os.unlink(out)
    if not rec:
        return {"i": idx, "err": "parse"}
    rec["i"] = idx
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def main() -> int:
    """usage: dock_run.py [n] [shard] [n_shards]

    Sharding is by **stride**, not by contiguous block: shard k takes every index with
    `i % n_shards == k`. That matters for how the job degrades. One VM dying out of a
    contiguous split leaves a systematic hole -- with train-then-test ordering it could remove
    the entire blinded test set, or one whole region of the chemical series. One VM dying out
    of a stride split leaves a uniform 1-in-K thinning, which is a smaller, unbiased sample
    rather than a biased one. Each shard writes its own JSONL, so the files merge by
    concatenation and no two workers contend for the same handle.
    """
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(json.load(open(f"{D}/smiles.json")))
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n_shards = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    out = (f"{D}/dock_cyp3a4_s{shard}of{n_shards}.jsonl" if n_shards > 1 else OUTJL)

    # Count as done anything ANY shard has already written, not just this one's file. The
    # first 108 ligands were docked before sharding existed, and a shard that re-did them
    # would waste real time; this also lets a restarted shard inherit a dead shard's work.
    done = set()
    for f in sorted(glob.glob(f"{D}/dock_cyp3a4*.jsonl")):
        for line in open(f):
            try:
                done.add(json.loads(line)["i"])
            except Exception:
                pass
    todo = [i for i in range(n) if i % n_shards == shard and i not in done]
    cores = max(1, os.cpu_count())
    print(f"[dock] {n} ligands, shard {shard}/{n_shards}, {len(done)} already done, "
          f"{len(todo)} to go, {cores} cores, exhaustiveness {EXH} -> {out}", flush=True)
    if not todo:
        print("DOCK DONE", flush=True)
        return 0

    t0, k, errs = time.time(), 0, 0
    with open(out, "a") as fh, ProcessPoolExecutor(max_workers=cores) as ex:
        for rec in ex.map(one, todo, chunksize=4):
            fh.write(json.dumps(rec) + "\n")
            # Flush EVERY record, not every 100. The JSONL is only a durable checkpoint to
            # the extent it is actually on disk: buffered at 100 records and ~11 s per
            # ligand, up to 19 minutes of work sits in Python's buffer where neither a VM
            # death nor `scripts/dock_pull.sh` can see it, which quietly defeats the whole
            # pull-it-off-the-machine design. One write syscall per ligand is free here.
            fh.flush()
            k += 1
            errs += 1 if "err" in rec else 0
            if k % 100 == 0:
                el = time.time() - t0
                print(f"[dock] {k}/{len(todo)}  {el/60:.1f}m  "
                      f"eta {el/k*(len(todo)-k)/60:.0f}m  errors {errs}", flush=True)
    print(f"\n[dock] finished {k} in {(time.time()-t0)/60:.1f}m, {errs} errors", flush=True)
    print("DOCK DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
