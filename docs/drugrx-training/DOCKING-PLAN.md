# Docking features for the CYP challenge — plan

**Authorised 2026-10-09** after every ligand-only modelling lever came back empty. A100s
approved. `PROGRESS.md` has the measurements that led here; this file is the build.

## Why this and not another descriptor

Every feature the programme has is **ligand-only** — ECFP4 counts, rdkit2d, Avalon, ErG, 3,091
dims, already the MapLight recipe. A ligand-only feature vector is **identical for all four
isoforms**, so our four models can differ only in their learned weights, never in their input.
Nothing we feed the model knows that CYP2D6 has a small anionic pocket and CYP3A4 a large
promiscuous one.

A docking score against a specific CYP's pocket is the first feature that is **isoform-specific
by construction**. That is the argument for trying it, and it is different in kind from "a
bigger fingerprint" — which is why the saturation result in `AGENTS.md` does not apply.

Correcting a claim made while scoping: the challenge **does not ship protein structures**.
`cyp-challenge-TEST-BLINDED_structures.csv` is 20 *ligands* for the pose-prediction track. The
receptors come from the PDB.

## What success looks like, stated before building

The target is **Spearman**, not ST-RAE: placement is monotone and cannot change ρ, so only a ρ
gain survives to the board. The bar, from `AGENTS.md`:

- **Fail:** Δρ < +0.01, i.e. the scale at which the hyperparameter sweep already produced a
  t = 2.96 result that did not replicate.
- **Interesting:** Δρ +0.02 to +0.05.
- **What we actually need:** +0.07 to +0.11 per isoform to reach the field's best.

Every number goes through the established discipline, which has caught three false positives
in one day:

1. paired across seeds, Δρ reported beside ΔST-RAE;
2. the **NN ≥ 0.5 restricted contrast** beside the headline (a kNN blend scored +0.0186 at
   t = 10.7 overall and −0.0616 in the test's own regime);
3. any selected winner **confirmed on seeds never looked at** (slices share a resample;
   only fresh seeds redraw the noise).

## Staging — one isoform first

27,580 dockings (6,895 compounds × 4 receptors) is the full job. It is not the first job.

**Stage 1 is CYP3A4 only, all compounds.** Reasons, in order:

- Largest gap to the field best (**+0.106 ρ**), so the most to win.
- Most labels: 2,335 dose-response plus 1,249 TDI-condition-only.
- **Its validation→board lift is −0.005**, i.e. ~zero. Alone among the four, a Δρ measured on
  our validation transfers to the board one-for-one. CYP1A2 and CYP2C9 carry lifts of +0.175
  and +0.135, which would make a validation gain much harder to read.

Stage 2 (the other three) happens only if Stage 1 clears the "interesting" bar.

## Receptor

CYP3A4, **PDB 1TQN** — 2.05 Å, substrate-free, the standard choice for docking a diverse set
precisely because it carries no induced fit toward one ligand class. Kept: protein + **heme**
(the catalytic site sits directly above the Fe; removing it would dock ligands into a cavity
that does not exist). Removed: waters, crystallisation additives.

Box: centred on the heme Fe, ~24 Å cube — CYP3A4's pocket is large and promiscuous, and a tight
box around a single sub-pocket would bias against the bulky ligands that matter.

Known weakness to record now rather than discover later: **Vina scores Fe coordination poorly.**
Published CYP docking works around this rather than solving it, so the absolute scores are not
to be read as affinities. What the model needs is a *ranking-useful* signal, and that is what
gets measured.

## Features extracted per ligand

Not just the top score — a single number from a noisy scoring function is the weakest possible
use of a docking run:

- best score, mean of the top 3, spread across the 9 retained poses;
- the pose set's RMSD spread (pose-consensus: a ligand with 9 poses that agree is a more
  trustworthy score than one scattered across the pocket);
- distance from the ligand's closest heavy atom to the heme Fe in the best pose — directly
  mechanistic for a CYP, where inhibition often means coordinating or occluding the iron;
- 3D conformer descriptors (WHIM, GETAWAY, RDF, MORSE, PMI, NPR) as a free by-product, since
  conformers must be generated anyway. These are ligand-only and so *not* isoform-specific, but
  they are orthogonal to 2D fingerprints and cost nothing extra.

The last bullet gives a clean internal control: **if the 3D ligand descriptors help as much as
the docking features, the isoform-specificity argument is wrong** and the gain is just 3D shape.

## Compute

Vina is CPU-bound, so the A100 session is chosen for its **12 vCPU and 89.6 GB RAM** (against a
CPU session's 2 vCPU) as much as for the card — the same reasoning `AGENTS.md` already records
for featurisation. Estimate: ~8–16 CPU-s per docking at exhaustiveness 8, 6,895 ligands,
≈ 2 hours wall-clock on 12 cores for Stage 1.

Checkpoint every 250 ligands to Drive-free local storage and `colab download`, because a lost
session is otherwise a lost two hours.

## Order of work

1. Provision A100 session `cyp-dock`.  ← in progress
2. Install vina + meeko + openbabel; verify on one known ligand (ketoconazole, a canonical
   CYP3A4 inhibitor) and sanity-check it docks near the heme.
3. Prepare 1TQN receptor.
4. Generate conformers + PDBQT for all 6,895 compounds; emit the 3D descriptor block.
5. Dock all vs CYP3A4; checkpoint.
6. Pull features back; evaluate with the three-part discipline above.
7. Decide Stage 2 on the measured number, not on enthusiasm.


## Operational facts learned while building (2026-10-09)

- **`pip install vina` builds from source** and needs boost + swig. It failed, and because it
  shared one `pip install` line with rdkit and meeko it took those down too. Install each
  package on its own line and check each one. Use the **prebuilt Vina 1.2.5 binary** instead.
- **`--write_maps` needs a `--ligand`** to know which atom types to build grids for, so a
  cached map set is only valid for ligands whose types it covers. Our compounds span
  C/N/O/S/P/F/Cl/Br/I, so a map set built from one ligand would silently not cover the rest.
  Map caching is therefore **abandoned**, not patched: `--receptor` per ligand rebuilds the
  grid each time and costs more, but it cannot be silently wrong.
- **`colab exec` is serial on the kernel.** A probe issued while a job runs queues behind it
  and times out, so a multi-hour docking run must be launched **detached on the VM**
  (`nohup ... &`) writing to its own log, and monitored by separate short `colab exec` calls
  that tail that log. Running it in the foreground blocks the kernel and makes progress,
  checkpoints and failures all invisible until the end.
- **`colab exec` has no `-c`**; it takes `-f <local file>` only (already in `AGENTS.md`).
- `subprocess.run(capture_output=True)` in a driver script hides all progress until the child
  exits — fine for a 30 s install, wrong for anything long. Stream to a log instead.
