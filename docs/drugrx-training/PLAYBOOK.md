# Playbook

Rules this programme has learned, each marked **[verified]** (we tested it here, on our data) or
**[assumed]** (read in a paper or a leaderboard write-up, not yet confirmed by us). The distinction
is the point: an assumed rule may be acted on, but it may not be reported as a result, and it gets
promoted or deleted once we have evidence either way.

The format is borrowed from the harnesses that currently top ARC-AGI-3 and Terminal-Bench, which
share one shape: a thin harness with few tools, durable state in files rather than chat history, a
written prediction before each action, stuck-detection with escalation, and failures converted into
procedure. See `research/01-reasoning-models.md`.

---

## Operating

- **[assumed]** Write the expected metric down before a run starts; compare after. Prevents
  retrofitting a story onto whatever happened.
- **[assumed]** When stuck, escalate rather than retrying: change one variable, or stop. Repeating a
  failing action is the dominant failure mode in agent harnesses.
- **[verified]** `colab auth` and `colab drivemount` need a TTY and hang when driven by an agent.
  Checkpoints therefore go to the Hugging Face Hub, not Drive.
- **[assumed]** An unrecognised `--gpu` value silently allocates an A100. `scripts/smoke_l4.py`
  asserts the device name rather than printing it, so this fails in seconds rather than at billing.

## Modelling

- **[assumed]** Recursive tiny models (TRM, 7M params) reach 87.4% on Sudoku-Extreme and 44.6% on
  ARC-AGI-1, but arXiv 2512.11847 reports **0% without the puzzle-ID embedding**, ~11pp of the score
  coming from voting, and most accuracy arriving at the first recursion step. ARC Prize's own
  verification of HRM came in at 32% against a claimed 55%.
  **So:** recursion-at-fixed-budget is a Phase 2 hypothesis to be tested against a non-recursive
  control, not a technique to build on. A negative result is expected and acceptable.
- **[assumed]** **The recursion architecture is not the win; two pieces inside it are.** TRM gives
  one clean fixed-budget result (2 layers + more loops at 5M beating 4 layers at 10M, 87.4% vs
  79.5%), but Mixture-of-Recursions *underperforms* vanilla at 135M and ARC Prize found a same-size
  vanilla transformer within ~5 points of HRM. Take **deep supervision / iterative refinement**
  (13 points in ARC's ablation) and **EMA** (7.5 points in TRM's) — neither needs an architecture
  change — and treat the architecture itself as optional.
- **[assumed]** Those puzzle results depend on training against the evaluation tasks. The equivalent
  in our domain is leakage, so it cannot be copied — which is one more reason headline numbers here
  are scaffold-split with a held-out target set.
- **[assumed]** Recursion is more likely to pay off on *pose or structure refinement* (an iterative
  object) than on predicting a single property number.
- **[assumed]** A verifier head plus physical checks (PoseBusters-style) is the cheapest place to
  spend test-time compute: sample several, keep what the verifier trusts.
- **[refuted, 2026-10-04]** ~~Use mixture-of-experts at the molecule or assay level.~~ **Dropped.**
  All experts stay resident, so we pay full memory for sparse FLOPs, and MoEs have been reported
  *underperforming dense* at 10–20M active params until ~20× more data. If MoE is ever revisited,
  DeepSeek's auxiliary-loss-free balancing (u=0.001) is the right mechanism.
- **[assumed]** Evolution belongs in the outer loop only — a search over configs, then merging
  fine-tuned siblings — never inside a single 24h L4 run.

## Evaluation

- **[assumed]** Deep models frequently lose to ECFP4 + gradient boosting on scaffold splits. Until
  our own baseline says otherwise, assume the baseline wins and make the deep model earn it.

## Chemistry reality

- **[verified]** Blind prospective ML virtual screening lands around a **3.7% hit rate at
  K_D < 150 µM** (CACHE #1, 23 teams, 1,955 compounds → 73 binders), and **0.7%** on CACHE #2's
  Nsp13 helicase. **Only 7 of 23 teams found anything on #1 and 5 of 23 on #2**, so the modal outcome
  of a blind screen is *zero hits*, and anything the playground says must be consistent with that.
  Sources in `research/06-prospective-screening-reality.md`.
- **[verified]** 150 µM is a weak binder. CACHE #1's best confirmed result was **8 binders at
  3–41 µM from 84 compounds over two rounds**, SPR plus ¹⁹F-NMR. Do not let "hit" read as "drug".
- **[verified]** **Difficulty is the target's, not the method's** — the same teams and methods moved
  3.7% → 0.7% by changing target. A per-target expectation is required; a per-product one is wrong.
- **[verified]** Our `rescore` ranker is uncorrelated where it is used: Boltz-2 shows Pearson
  r = 0.24–0.45 against physics-based reference calculations globally and **≈ 0 among the top 100**.
  It separates binders from decoys; it does not order the survivors. Treat it as an enrichment filter.
- **[verified]** **37 of CACHE #1's 73 hits had poor solubility despite computational prediction.**
  The documented failure mode of a real campaign is a property we predict and cannot show beats a
  fingerprint baseline — the strongest argument for annotating rather than filtering.
- **[assumed]** The median TDC ADMET endpoint has ~1,000 labels: these tasks are **data-bound, not
  compute-bound**. Spend the budget on pretraining, augmentation, evaluation and calibration, not on
  more gradient steps.
- **[verified by the research brief, not by us]** PDB **1HSG is HIV-1 protease** despite a deposited
  title saying "HIV II protease" (SIFTS → P03367, 100% identity, both chains). **2Q7M (4.25 Å)** and
  **5U1C (3.9 Å)** are not docking-grade.
- **[assumed]** Boltz-2 is a **reranker for the top ~200**, not a screen. Uni-Dock does the screen
  (>37,000 dockings/GPU-hour).
- **[assumed]** With ~40 targets, freeze ESM-2 and cache embeddings once; the DTA model then trains
  in under an hour, which leaves the budget for honest evaluation.

## Licensing

- **[assumed]** ChEMBL is CC BY-SA and the share-alike may reach model weights. Resolve before
  anything trained on it ships. BELKA has no clear open licence — out unless Leash grants one.

## Pretraining

- **[assumed]** **Start from CheMeleon.** A 12.9M Chemprop D-MPNN pretrained on 1M PubChem molecules
  by regressing 1,613 Mordred descriptors beats MoLFormer, which saw 1.1B molecules: 97% win rate on
  30 MoleculeACE assays and **100% on the activity-cliff subset**, against RF at 50% and scratch
  Chemprop at 0%. Ships in Chemprop ≥2.2.0. Descriptor-prediction pretraining is what works at this
  scale — not contrastive, not masked-LM.
- **[assumed]** **Never pretrain a backbone ourselves.** MoLFormer-XL ~3,300 V100-hours; GROVER
  ~15,000–24,000. Three orders of magnitude past a 24h L4.
- **[assumed]** **Descriptor/fingerprint pretraining is negative for binding affinity**, though
  strong for ADMET — a 2026 study over 40 tasks with Butina OOD splits found a pretrained GIN losing
  to classical fingerprints on DRD2, Factor Xa and MUV. The affinity half goes to Uni-Mol (protein-
  pocket branch) or distillation instead.
- **[assumed]** Published pretraining gains are largely weak baselines: a tuned supervised D-MPNN
  ties MolCLR on Lipo and beats it on ESOL and QM8, and ChemBERTa-2 gains nothing from 5M → 77M
  pretraining molecules. Tune the simple thing hard before believing a pretraining delta.

## Multi-task

- **[assumed]** Standardise every regression target first — most of the reported benefit of fancy
  multi-task balancing is just scale. Plain summation scores Δm% 177.6 on QM9's 11 targets against
  FAMO's 58.5.
- **[assumed]** If a task is still crushed after standardisation, use **FAMO** (O(1) per step).
  **Skip PCGrad / CAGrad / Nash-MTL**: at 21 tasks they cost ~21× backward per step, and both PCGrad
  and CAGrad have been measured *worse* than plain uncertainty weighting.

## Colab mechanics

- **[assumed]** Pro+ is *"up to 24 hours"* — a **ceiling, not a guarantee**; the FAQ says the limits
  are unpublished and resources "not guaranteed and not unlimited". `colabtools` issue #5950 reports
  Pro+ runtimes dying after **40–60 minutes**. Resume logic is mandatory, not a nicety.
- **[assumed]** **The L4 runtime is a `g2-standard-4` with 4 vCPU.** RDKit/Mordred featurisation
  starves the GPU unless cached ahead of time. This is a harness requirement, not an optimisation.
- **[assumed]** **Use WSD or schedule-free, never cosine.** You can cool down from any checkpoint; a
  cosine schedule cannot survive a session that dies at hour 9.
- **[assumed]** Write checkpoints to local disk, then copy **one** file out — Drive's quota and
  10k-item warnings bite otherwise. Atomic save via `os.replace`.

## Evaluation (continued)

- **[assumed]** **The benchmarks leak badly enough to invalidate published deltas.** An audit of 51
  configurations found DTI benchmarks averaging **0.779 train/test nearest-neighbour similarity**,
  and changing the familiar-compound mix **flipped which model ranked first**. Under time-based ADME
  splits, four different models win six endpoints. Fixing the evaluation outranks every modelling
  idea on the list.

## Splits (measured here, 2026-10-04)

- **[verified]** On caco2, the fraction of test molecules with a training neighbour at
  Tanimoto > 0.4 is **47.25% on TDC's own official split**, 42.86% on our scaffold split, and
  **9.89% on a Butina split** (mean NN-similarity 0.429 / 0.450 / 0.294). The official benchmark is
  substantially leaky, so **a leaderboard-comparable number overstates what the model would do on
  genuinely new chemistry**. Report both, always, and never let the official number travel alone.
- **[assumed]** Across 60 datasets with the same random forest, top-100 hit rate falls
  **93% → 91% → 84% → 43%** for random → scaffold → Butina → UMAP splits, and series-holdout drives
  R² negative on 5–6 of 8 ADME endpoints. Triage operates in the series-holdout regime.

## Where compute runs (owner's instruction, 2026-10-04)

- **[verified — owner's instruction]** **No model runs on the local machine's CPU.** All compute,
  CPU and GPU alike, runs on Colab. The laptop is for writing code, orchestrating, reading logs and
  holding secrets — not for fitting models.
  **This supersedes the earlier "featurise on the laptop" rule**, which was adopted because the L4
  runtime has only four vCPU. The constraint is still real, so the resolution is: featurise and run
  CPU baselines on a **separate CPU-only Colab session** (`colab new -s <n>` with no `--gpu`), cache
  the result, and let the L4 session consume the cache. A CPU session is far cheaper per hour than
  an L4, so this is also the efficient arrangement, not merely the compliant one.
- **[assumed]** Keep the two sessions separate and stop each one the moment it is idle: an idle VM
  bills. `colab run` self-cleans even on error and is the right shape for one-shot CPU jobs.

## Sessions and orphans (verified 2026-10-04)

- **[verified]** `colab stop -s <name>` resolves by **local** session name and **cannot stop an
  orphan** — a runtime started from the Colab web UI has no local record and lists as `[?]`.
  Terminating one needs the browser: Runtime → Manage sessions → the trash icon **on that row**.
- **[verified]** In Manage sessions, a CLI-created session shows as **"Unknown notebook"**. So
  **"Terminate other sessions" kills the CLI training run**, not the stray notebook. Always use the
  per-row control.
- **[verified]** Terminating a runtime while its notebook tab is open makes the tab **immediately
  re-allocate a new GPU** ("Connecting to T4…"). Close the tab, or the unit you just saved comes
  straight back.
- **[verified]** Removing one idle T4 took the account burn from **2.61/hr to 1.54/hr** — so an L4
  session is ~1.54/hr, and 2000 units is on the order of 1,300 L4-hours. Compute is not this
  programme's constraint; data and splits are.

## The ceiling is generalisation, not label noise (revised 2026-10-04)

- **[refuted, same day it was written]** ~~Our solubility MAE 0.804 is at the aleatoric floor.~~
  **Withdrawn.** The Second Solubility Challenge gave entrants a test set with reproducibility
  SD 0.17 log and the best model still only reached RMSE 0.86 — ~0.7 log of headroom — and the
  organisers state plainly that it is *"the deficiency of QSPR methods … which is the limiting
  factor"*. Kolmar & Grulke (>1,800 models) show training noise inflates *measured* error, not true
  error. Removing suspicious compounds does not improve external predictivity.
- **[assumed]** **Most "assay noise" is curation debt, not physics.** Same-lab reproducibility is
  σ = 0.17–0.22; pooled public data is 0.66–0.68 — ~3× the SD and 9–16× the variance, and the
  difference comes from **pooling across sources**, which is in principle removable.
- **[assumed]** **The generalisation gap dwarfs the noise floor.** Solubility models claiming
  0.35–0.63 log on their own validation score **1.74–2.17 on a fresh external set**; ADMET-AI and
  ADMETlab 3.0 give **negative R² for Caco-2 and HLM clearance** on a real programme's molecules.
  So: fix the split and the features before buying compute.
- **[assumed]** **Our scaffold split cannot settle the model question.** 14 of 22 TDC sets are under
  2,000 molecules and eight have test folds of 95–175, so differences of ~0.01 are inside split
  noise. Scaffold is only Δ0.01–0.04 harder than random at our scale, while Lo-Hi/UMAP/DataSAIL drop
  0.08–0.19 and ID→OOD rank correlation falls from r≈0.94 to 0.19–0.26.
- **[assumed]** **Semi-supervised learning does nothing here.** On real assay labels the measured
  gains are +0.0000/+0.0001/+0.0007 AUROC; the −19%/−20% gains live entirely on labels that are
  deterministic functions of structure (computed logP, DFT). **No verified instance of a
  semi-supervised method beating a tuned descriptor GBM on a real ADMET endpoint under scaffold
  split.**
- **[verified by reasoning]** **SMILES enumeration is a no-op for us** — count-ECFP4 and RDKit2D are
  canonicalisation-invariant, so the augmentation cannot change the features. Do not implement it.
- **[assumed]** Per-endpoint reproducibility now has real numbers: hERG σ = 0.18 (5-lab ring trial;
  values within 5× "should not be considered different"); hepatocyte CLint ~3× interlab; Caco-2 log
  Papp 0.23–0.93; **P-gp inhibition IC50 spans 20- to 796-fold across 23 labs — effectively
  unpoolable**, which should change how Pgp_Broccatelli is read.

## Tooling hazards

- **[verified by a subagent]** **WebFetch's summarising model fabricated a results table** — asked
  for a specific arXiv paper it returned numbers and a split type that the PDF does not contain, with
  every delta inverted relative to the real result. Anything load-bearing must be read from the PDF
  (`pdftotext`) or the released CSV, not from a fetch summary.

## Acquisition (measured 2026-10-04)

- **[verified]** **Two leakage mechanisms, two exclusion rules.** *A*: the source document **is** the
  benchmark — TDC built `lipophilicity_astrazeneca`, `ppbr_az`, `clearance_microsome_az` from
  AstraZeneca ChEMBL deposits whole, so pulling them re-imports **99.5%/100%/100% of those test folds
  for 3/1/0 new molecules**. Drop the document. *B*: the source independently indexed the same
  upstream literature — hERG/half_life/caco2 residuals are **14,406/11,930/13,327**, and removing the
  53 colliding molecules from a 15,934 hERG pull leaves 90% intact. Molecule-level exclusion. **Never
  let an A result argue away a B source.**
- **[verified]** **The gross number is never the decision; the residual is.** "33k ChEMBL molecules"
  was gross; for the three AZ documents the exclusive residual is 4.
- **[assumed]** **Identity exclusion is necessary and not sufficient under a scaffold split.** The
  metric that signals inflation is **test-scaffold recovery** — what fraction of test scaffolds,
  absent from train by construction, the residual puts back — not the fraction of the residual that
  shares a scaffold. Use the split's own definition (`MurckoScaffoldSmiles`, non-generic, chirality
  off); a *generic framework* inflates the rate against nothing the split did. **Acyclic molecules
  return an empty scaffold and collapse into one bucket** — segregate and report them, or a
  half-life pull full of small aliphatic drugs manufactures a collision spike.
- **[assumed]** **Scaffold leakage can bias an effect size, not just an absolute number.** If
  contamination rewards near-neighbour memorisation, it favours representations that support it —
  and Avalon/ErG and ECFP do not afford that equally. So H5's *margin* is exposed to this, not only
  its levels.
- **[assumed]** **Label heterogeneity is the bigger risk in mechanism-B pulls, and it masquerades as
  a result.** TDC's hERG is binary at one threshold from one curated source; ChEMBL's is patch-clamp
  IC50 plus binding displacement across cell lines. Obach's half-life is ~670 *human IV terminal*
  values; ChEMBL `T1/2` spans species, routes and oral/IV. Merge them and "more data did not help"
  measures **our filtering**, which we would then misread as confirming the data-bound conclusion.
  Filter on `standard_type`, `assay_organism`, `confidence_score`, BAO format,
  `data_validity_comment`, and drop `potential_duplicate` — **the post-filter residual is the number
  that goes in the table.**
- **[assumed]** **A reported "zero overlap" must be recomputed on the InChIKey join.** A broken
  SMILES join returns *exactly zero*, which reads as good news and is never questioned; the same
  artefact gave 21/420 on lipophilicity and was caught only because it was absurdly low rather than
  absurdly good.

## The audit layer is the least-verified part of the stack

- **[verified]** **Two audit bugs in one session, both failing toward reassurance.** The statistics
  joined on the SMILES string (reported a clean comparison while matching 21 of lipophilicity's 420),
  and the similarity verdict used `abs(shift)` (printed "real proximity" for sets that were *further*
  than background). Numbers right, interpretation inverted, in both cases. **Neither was caught by a
  test** — only by eye.
- **[verified]** Code that *checks* the work needs its own checks more than the code that does it,
  because nobody audits the auditor, and its failures look like results rather than crashes.
  `tests/test_audit_fixtures.py` holds known-answer fixtures: a deliberately spiked pull **must**
  come back hot (including when the spike is a salt form or a re-spelled SMILES), a shuffled control
  **must** come back cold, and a negative shift **must not** be labelled proximity.
- **[verified]** **A test that re-implements the logic it tests guards nothing.** The first draft of
  the verdict fixture defined its own copy of the rule and would have passed while the shipped script
  kept the inverted label. The rule now lives in `drugrx/data/splits.proximity_verdict` and both the
  script and the test import it.
- **[verified]** Both fixtures were **mutation-tested**: reintroducing `abs(shift)` fails
  `test_negative_shift_is_not_labelled_proximity`, and reintroducing `SetRemoveSp3Stereo(True)` fails
  `test_enantiomers_stay_distinct`. A fixture nobody has seen fail is not known to work.

---

## Promoted / deleted

<!-- When a rule is verified or refuted here, move it with the date and the run id that settled it. -->
