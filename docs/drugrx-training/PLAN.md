# DrugRx training — plan

Phases are **expanded one at a time**: the detail for phase N is written when phase N starts, after
the research for it is in `research/`. Everything below phase 1 is deliberately a sketch.

Read `AGENTS.md` for the rules that bind every phase (L4 only, baselines first, scaffold splits,
kill rules, honesty).

---

## Phase 0 — Harness and ground truth *(expanded)*

Nothing is trained until a run can survive being killed.

**P0.1 Colab CLI operable headless.** Installed (`uv tool install google-colab-cli`). Auth is an
interactive Google consent flow — a human runs `colab sessions` once and pastes the code. Verify
with `colab usage` (balance) and a `--gpu L4` smoke session that prints `nvidia-smi`.

**P0.2 The run harness.** One Python package, uploaded to the VM per run:
- config as a dataclass, hashed into the run id;
- checkpoint every epoch (weights, optimizer, scheduler, RNG, epoch, best metric, config hash);
- resume automatically from the newest checkpoint;
- metrics appended as JSONL, pulled by the local monitor;
- a heartbeat file so the monitor can tell "slow" from "dead".

**P0.3 Durable checkpoints without a TTY.** Hugging Face Hub private repo is the default sink
(`colab drivemount` cannot be driven by an agent). Drive stays available for the owner's existing
notebooks. **Acceptance: kill a run at epoch 2, restart it, and the loss curve continues.**

**P0.4 Data cache.** Download, clean and split the datasets once; push the processed tensors to the
Hub so a fresh VM is training within minutes instead of re-downloading.

**P0.5 Baselines.** ECFP4 + LightGBM/RF on every task, scaffold-split, numbers committed. Plus the
existing Vina score as the docking baseline. **No deep model is reported until it beats these.**

**P0.6 The monitor.** A local loop that polls `colab status` / `colab log`, pulls metrics, applies
the kill rules, and appends to `PROGRESS.md`. This is the "thin harness + lab notebook" pattern from
the ARC-AGI-3 and Terminal-Bench winners (`research/01-reasoning-models.md`): minimal scaffolding,
an explicit written record, and cheap restarts.

---

## Phase 1 — Evaluation first, then the first real model *(sketch)*

Revised after `research/03-training-techniques.md`. The shortlist there puts **fixing the
evaluation above every modelling idea**, because an audit of 51 configurations found DTI benchmarks
averaging 0.779 train/test nearest-neighbour similarity, and changing the familiar-compound mix was
enough to flip which model ranked first.

**P1.1 Leakage audit.** Nearest-neighbour similarity between train and test on every endpoint, plus
a time split where assay dates allow. Report it beside every number from here on.

**P1.2 Harder baseline.** A properly tuned D-MPNN and RF alongside the committed ECFP4+LightGBM. A
tuned supervised D-MPNN ties MolCLR on Lipo and beats it on ESOL and QM8, so this is the real bar,
not a formality.

**P1.3 CheMeleon.** Start from the 12.9M Chemprop D-MPNN pretrained on Mordred-descriptor regression
(ships in Chemprop ≥2.2.0), multi-task across all 21 ADMET endpoints with standardised targets.
**We pretrain nothing ourselves** — the cheapest comparable backbone cost ~3,300 V100-hours.

**P1.4 Calibration.** β-NLL head, a 5-member ensemble via `stack_module_state` + `vmap`, and
scaffold-calibrated conformal intervals. The playground ranks candidates, so a confident wrong score
is worse than an honest wide one.

## Phase 2 — Technique bake-off *(sketch)*

Controlled experiments against the Phase 1 model at fixed budget, one variable at a time, each with
a written prediction in `HYPOTHESES.md` *before* it runs.

In: randomised-SMILES augmentation + test-time augmentation; FAMO (only if a task is still crushed
after standardisation); deep supervision / iterative refinement; EMA; WSD schedule; active learning
(MolPAL reached 87.9% of the top-50k from 2.4% of 100M); LoRA on ESM-2; distillation from a *cheap*
oracle (Vina/smina — not Boltz-2 at this scale).

Out, with reasons recorded in `PLAYBOOK.md`: **MoE** (full memory for sparse FLOPs; reported worse
than dense at 10–20M active params), **PCGrad/CAGrad/Nash-MTL** (~21× backward per step at 21 tasks,
and measured worse than plain uncertainty weighting), and **self-pretraining** (three orders of
magnitude out of budget). The TRM-style recursive architecture stays as a *control arm only* — the
transferable parts of it, deep supervision and EMA, are already in the list above.

## Phase 3 — Binding affinity *(sketch)*

A separate track, because descriptor/fingerprint pretraining is **explicitly negative** for affinity
even though it is strong for ADMET. Uni-Mol (it has a protein-pocket branch) or distillation from
docking scores, trained on PDBbind-family data, evaluated on CASF-2016 scoring/ranking/screening
power with the LP-PDBbind split and a held-out target set. This is what would augment the `rescore`
stage. PDBbind licensing is the blocker to resolve first.

## Phase 4 — Serve the winners *(expanded 2026-10-05)*

Winning checkpoints into `talindrew-foldrx-models/` on Modal, registered as `ModelImpl`s with
caveats, wired into FoldRx and `/ai-workspace`. Backend tests, e2e specs, no perf regression.

**The thing this phase can get wrong is not the weights, it is everything the weights do not carry.**
Sub-resolution per endpoint; gradient boosting as good and cheaper on most endpoints; six ExpansionRx
heads on a measurement scale that must never be merged into a TDC target; 18% domain coverage. A
`ModelImpl` that exposes a predict surface and nothing else hands the first caller a number with none
of that attached, and FoldRx's ranking will consume it. So the registry wiring carries three things
beyond the tensor file, designed in rather than retrofitted: **per-endpoint routing**, **applicability
domain on every prediction**, and a **separate namespace for the ExpansionRx heads**.

### P4.0 — Decisions and invariants, recorded before any code

**D1 — Annotate, never filter. This is already true of the code and is now pinned.** FoldRx's `admet`
stage consumes `LIGAND SET` and `feeds_from=("ligands",)`, so it runs beside docking's inputs and
cannot shrink the docked set (`registry.py:92-94`); no guard in `engine._check()` gates on an admet
result; and `_rules_stage` labels rather than removes. More to the point, **the only admet value that
reaches the composite score is `drug_likeness` (QED)** — a deterministic RDKit computation, not a
model prediction (`local_stages.py:191-194`). So a DrugRx endpoint contributes **zero** to ranking,
and the paste's conditional ("if it filters, the 3.7% framing has to propagate into the ranking
logic") does not arm. **A test pins it**, because this is an invariant someone will otherwise
"improve" by wiring the 21 endpoints into the weight vector.

**D2 — Out of domain is visible, not inferable.** Every endpoint value carries `in_domain` and the
nearest-neighbour similarity that decided it. The value is still returned (owner's decision,
2026-10-05): withholding it would empty a screen of novel chemistry, which is arguably true and
certainly unusable, and a flagged number stays auditable.

**D3 — Endpoint names are admet-ai's TDC-canonical names; ExpansionRx is a separate namespace.**
Emitting `Caco2_Wang` rather than `caco2` means the two admet models speak one vocabulary, the
existing `AdmetPanel` renders DrugRx with no frontend change, and a user can compare them. The six
ExpansionRx heads are prefixed **`ExpansionRx_`** and never occupy a TDC name. The case that matters:
`Clearance_Microsome_AZ` (TDC/AstraZeneca) and `ExpansionRx_CLint` are different measurements whose
two panels **share zero molecules**, so no cross-lab offset is even estimable — the names must make
comparing them obviously wrong, not merely discouraged.

**D4 — The routing table is data, generated from the measurement.** Prose in `runs/ROUTING.md` cannot
be the source of truth for a dispatch decision; a `routing.json` generated *from* the bootstrap output
is, and a test asserts the two agree.

### P4.1 — The arm selection, and ROUTING v2 *(do first: it can move the table)*

**`ROUTING.md` compares the ensemble against the `count-rdkit2d` GBM — but H5 showed the `maplight`
GBM (plus Avalon + ErG) beats `count-rdkit2d` on 14 of 21 endpoints.** So the routing table does not
describe a comparison against the GBM we would actually choose to serve, and on the two endpoints it
awards to the ensemble the better arm changes the picture in opposite directions:

| endpoint | ensemble − count_gbm | maplight − count_gbm | consequence |
|---|---|---|---|
| `pgp` | **+0.0255** [+0.0017, +0.0499] | **+0.0117** | maplight closes ~46% of the ensemble's margin; the CI very likely crosses zero |
| `half_life` | **+0.1620** [+0.0001, +0.3229] | **−0.1333** | maplight is much worse; the ensemble's verdict should survive, and `count-rdkit2d` is the right GBM here |

So the GBM arm is **per endpoint**, and the ensemble-vs-GBM comparison must be re-run against the
chosen arm before anything is hardcoded.

**The selection rule, fixed before the run (this is the part that can be got wrong).** Choosing the
better of two GBM arms per endpoint *on the test fold* is test-set selection and would inflate GBM by
construction — the same class of error as T6's uncalibrated threshold. So: **the arm is chosen on the
validation fold; the ensemble-vs-GBM paired bootstrap is then run on the test fold against whichever
arm validation picked.** Registered as **H11** with the prediction written first.

Acceptance: `runs/ROUTING.md` v2 plus `runs/routing.json`, each endpoint carrying its chosen arm, the
diff, the CI, and a verdict; `undecided` never routes to the ensemble.

### P4.2 — The served artifact, and the proof it is the measured one

`ship_artifact.py` saves the ensemble. **Nothing saves a fitted GBM** — `baseline.py` fits, scores and
throws the booster away — so 19 of 21 endpoints currently have no servable model at all. A
`ship_router.py` produces one bundle:

- the 21 LightGBM boosters, each on the arm P4.1 chose, with its feature spec and label norm;
- the ensemble checkpoint (`arnab2812/drugrx-admet-ensemble`);
- `routing.json`;
- an **applicability-domain pack**: per-endpoint packed Morgan r=2/2048 training fingerprints, plus
  the random-background similarity distribution, so the 0.3 cut is calibrated against a background
  rather than fixed (T6 applies here exactly as it did to the leakage claim).

**Acceptance is reproduction, not a smoke test:** load the bundle cold, predict each endpoint's test
fold, and reproduce the committed metric within tolerance. An artifact that cannot reproduce the
number on its own card is not the model that was measured.

### P4.3 — The Modal endpoint

`talindrew-foldrx-models/`: a new image + function + one `handlers` entry at **`/admet/drugrx`**.
Request is the admet shape already built by `local_stages.build_request` (`{compounds, params}`);
response is admet-ai's shape — `{compounds:[{id, name, smiles, drug_likeness, endpoints, predicted}],
count, model}` — **plus** a parallel `endpoint_meta` keyed identically, carrying `in_domain`,
`nn_similarity`, `served_by` and the interval. Parallel rather than nested because `AdmetPanel` reads
`endpoints[key]` as a number; nesting it would break admet-ai's panel for a field only we have.
Featurisation lives in the same container as both arms, so a GBM and the ensemble cannot drift onto
different features.

### P4.4 — Registry wiring

One `ModelImpl("drugrx-admet", "admet", …, provider="oss-modal", path="/admet/drugrx")` carrying a
`caveat` that states the sub-resolution finding and the domain limit in one sentence. **`admet-ai`
stays the default** — it covers 41 endpoints against our 21 and has no claim to defend.
Also close the invariant the survey found unguarded: **no test asserts that every `oss-modal` impl's
`path` has a matching gateway handler.** Adding a model is exactly when that bites, so the test lands
in this phase.

### P4.5 — Frontend

`endpoint_meta` surfaced on the DrugRx rows (out-of-domain badge, nearest-neighbour similarity, which
arm served it); `Clearance_Microsome_AZ` added to `ADMET_GROUPS`' Excretion group; the ExpansionRx
heads rendered in their own group so the namespace is visible, not merely encoded.

### P4.6 — Verify

Backend unit tests, Playwright e2e against a stubbed gateway and then the real one, Chrome extension
pass over `/foldrx` and `/ai-workspace`, logs read, screenshots verified **and deleted**, CHANGELOG
and memory updated. **Colab sessions stopped** (owner's instruction, 2026-10-05: use the GPUs, back
the data up, close the sessions when done).

## Phase 5 — The playground *(expanded 2026-10-05)*

Disease → target → screen → ranked hypotheses, inside `/ai-workspace`, built from the existing node
graph and templates. Per-disease templates for the four diseases, each honest about which of its
targets are small-molecule-tractable and which are biologics-only. Uncertainty shown, not hidden.

**The research that sets the bar is `research/06-prospective-screening-reality.md`**, and it changes
what this phase may put on screen. The programme has been quoting CACHE #1's **3.7%** as "what a good
result looks like"; that is true and it is the optimistic end. Beside it:

- **the median blind team finds nothing** — 7 of 23 groups on CACHE #1, 5 of 23 on CACHE #2;
- **3.7% is measured at K_D < 150 µM**; the best confirmed work was 8 binders at 3–41 µM from 84
  compounds across two rounds;
- **the same teams got 0.7% on a different target**, so difficulty belongs to the target, not the
  method — which is why `TARGETS.md`'s tiering has to reach the UI rather than stay in a doc;
- **37 of 73 CACHE #1 hits had poor solubility despite computational prediction** — our own endpoint,
  on which we cannot distinguish our model from a fingerprint baseline. This is the strongest
  argument yet for P4.0 D1 (annotate, never filter);
- **our `rescore` stage's ranker is uncorrelated where it is used** — Boltz-2 shows r = 0.24–0.45
  globally against physics-based reference calculations and **≈ 0 among the top 100**, which is
  exactly the shortlist a screen operates on.

### P5.0 — What the playground may claim, decided before any of it is built

**D1 — The output is a shortlist for ordering wet-lab work, and the page says so without being asked.**
Not a prediction of activity, not a candidate list. The honest sentence is "these are worth testing
first", and the number that calibrates it is **"most blind screening teams find nothing"**, not 3.7%.

**D2 — Target tractability is served from data and the UI refuses rather than omits.** A biologics-only
target (IL-17A, IL-5, TSLP, IL-23, IL-4Rα …) must be **listed and declined with its reason**, because
omitting it makes the disease look covered. `TARGETS.md` already says this is the point of recording
them; Phase 5 is where it becomes code.

**D3 — Expectations are per target, not per product.** A Tier 1 target with 16,623 pChEMBL points and
162 structures and a Tier 3 target with 57 points cannot carry the same framing. The tier travels with
the target into the UI.

**D4 — The pipeline ships with a negative control, and it is ours rather than a citation.** See P5.4.

**D5 — Nothing in this phase states a therapeutic claim for any disease**, and `AGENTS.md`'s existing
rule ("it does not discover, design or validate a treatment … any UI, export or notebook that states
otherwise is a bug") is enforced by a test, not by care.

### P5.1 — The target catalogue, as data *(expand and implement first)*

`TARGETS.md` is prose with verified accessions. Phase 5 needs it as a typed, served catalogue: disease
→ targets, each with tier, UniProt, ChEMBL, a docking-grade PDB, point count, structure count, whether
an approved drug exists, and — for the refused ones — the reason.

**The corrections in `TARGETS.md` are the part that must not be lost in translation**, because each one
is a wrong answer someone would otherwise ship: IL-23 is *refuted* in axial disease rather than
intractable (so ankylosing spondylitis means TYK2/JAK1/RORγt); GLP-1R is tractable only since
orforglipron and must dock against 7LCI/6ORV and not the peptide-bound 4ZGM; 1HSG is HIV-1 protease
despite its title; 2Q7M and 5U1C are **not docking-grade** at 4.25 Å and 3.9 Å; Boltz-2 is weak on
GPCRs, transporters and ion channels, which covers five of the shortlist.

Acceptance: every accession and PDB id in the catalogue is checked live against UniProt and RCSB by a
test that is **skipped, not passed, when offline**; resolution is asserted for every docking target so
a 4.25 Å structure cannot be added silently.

### P5.2 — Disease templates on the node graph

One template per disease, producing an `/ai-workspace` document: target → fold/provided structure →
pockets → ligands → dock → admet → rescore → rank. Pinned library, pinned models, params that match
the measured guidance (our `drugrx-admet` where it is enabled, `admet-ai` otherwise).

### P5.3 — The ranked-hypothesis view

The candidate table gains what Phase 4 produced: the applicability-domain percentile per compound, the
ensemble interval where one exists, and the per-target expectation from P5.1. The yardstick sentence
is rendered from the catalogue, not hard-coded, so a Tier 3 target cannot inherit a Tier 1 framing.

### P5.4 — The decoy-target control *(the distinctive piece)*

**A screen that would rank the same compounds the same way against the wrong protein is not telling
you about your protein.** So the playground can run the identical compound set against a decoy target
— pocket-scrambled, or an unrelated protein — and report the rank correlation between the two.

This is the same discipline the programme already applies to itself: a null-vs-null arm that must
centre on 0.5 (T3), and a spike-recovery positive control (T4). It also answers an unverified external
claim (that Boltz-2 retains enrichment under binding-site mutation) **by measuring it** rather than
citing it. A high rank correlation is a finding and gets written down either way.

### P5.5 — Verify

E2E against stubs for the document and template layer; Chrome/Playwright over the real page; logs
read; screenshots verified **and deleted**; `CHANGELOG.md`, `AGENTS.md` and memory updated.
**Local runs cannot reach a GPU** — no local backend has `FOLDRX_OSS_BASE_URL` (`docs/ai-workspace/PROGRESS.md`),
so every real-compute claim in this phase is stubbed and must be labelled as such.

## Phase 6 — Verify and document *(sketch)*

E2E with Playwright/Chrome, screenshots verified then deleted (screenshots are scratch), logs
checked, `CHANGELOG.md` and memory updated.

---

---

## Testing methods this programme now requires *(established 2026-10-04/05)*

Four defects this session all lived in code that **interprets or prepares** rather than computes, and
all failed toward a readable-looking answer: a SMILES join that reported clean comparisons while
matching 21 of 420 molecules; an `abs(shift)` label that printed "real proximity" for sets further
than background; a descriptor span that would have z-scored count fingerprints against descriptor
statistics; and a batch size that differed between the arms of a controlled experiment. None was
caught by a test. **The verification layer is the part whose failures look like results**, so it
carries its own obligations now.

**T1 — Known-answer fixtures for every audit.** `tests/test_audit_fixtures.py`. A spiked input must
come back hot *including when the spike is a salt form or a re-spelled SMILES*; a shuffled control
must come back cold; a negative shift must never be labelled proximity; acyclic molecules must not
share a scaffold bucket. **A test that re-implements the logic it tests guards nothing** — the rule
lives in the library and both the script and the test import it.

**T2 — Mutation-test the fixtures.** A fixture nobody has seen fail is not known to work.
Reintroducing `abs(shift)` must fail the proximity test; reintroducing `SetRemoveSp3Stereo(True)`
must fail the enantiomer test. Both verified.

**T3 — Negative control (null-vs-null).** Two permutation arms against each other must centre on
**0.5**. This certifies the whole comparison path and is what would have caught the batch-size
defect — a **broken experimental control** rather than a bug, since the arms differed in a
hyper-parameter that was not the variable under test. Measured: 0.512, PASS.

**T4 — Positive control (spike recovery).** A detector certified only against the null cannot
distinguish "no effect" from "blind". Spike a clean source with known test molecules at levels
bracketing what matters and confirm exact recovery. Measured: cold at 0%, exact at 1/5/10/25%,
resolving 2 molecules in 221.

**T5 — Effects are reported against a permutation null, never as win counts.** One permuted draw is a
*sample* from the null, not the null; at effects of 0.01–0.03 its own sampling variance is the size of
the effect. Use ≥20 draws, run **concurrently** as vmapped members each carrying their own permuted
labels — that is the legitimate claim on spare VRAM, since independent draws are the binding resource
for a small effect while a larger batch buys gradient noise reduction this dataset has no use for.
**Replicate the real arm too**: a point against a distribution charges all of the real arm's
initialisation variance to the effect.

**T6 — Thresholds are calibrated against a background, never fixed.** A raw "fraction above Tanimoto
0.4" depends on fingerprint density: Avalon's background is 0.546 against Morgan's 0.265, which made
a 99.8% figure look like contamination when it was baseline. The decisive comparison is
**query-vs-target against query-vs-random-background**. This retired the Avalon row and it forced the
re-derivation of our own split-leakage claim.

**T7 — Decision rules and effect sizes are fixed before the run.** Register the threshold *and* the
minimum detectable effect, measured by simulation at the real fold size. H7's first threshold was
unpassable as written (≥0.08 *and* a CI excluding zero at n=134, where the interval is ±0.10). Where
the design cannot resolve a plausible effect, **register the question as undecidable rather than
running it** — a null nobody can read is worse than a stated limitation.

**T8 — Audit on standardised identity, per endpoint, at two levels.** Identity exclusion on InChIKey
is necessary and **not sufficient** under a scaffold split; measure **test-scaffold recovery** (what
the residual puts *back*) using the split's own scaffold definition, and segregate acyclic molecules,
which return an empty scaffold and would otherwise collapse into one bucket.

## Open questions

- Does any of this beat ECFP4+LightGBM on scaffold splits? Honest answer may be no for some tasks.
- Is recursion-at-fixed-budget (TRM-style) actually useful outside the puzzle domains it was shown
  on? Phase 2 is where we find out, and a negative result is a result.
- PDBbind licensing — may force BindingNet/Papyrus instead. Phase 3 blocker, resolved in research.
