# DrugRx CYP — endgame plan (opened 2026-10-06)

The last round of the OpenADMET CYP challenge, planned backwards from what the evidence says is
still live. **Read `PROGRESS.md` (newest last) and `AGENTS.md` first; this file is the phase list
for the remaining window only.**

## Where this starts

| | validation | leaderboard |
|---|---|---|
| v2 (L3/32/300ep) | 0.737 | 0.8752, rank 163 |
| v3deep (L6/16/3000ep) | 0.6484 | **0.8598, rank 159** |

Leader 0.3772. Mean predictor 1.0. True deficit ≈ **0.31** once the metric's tailwind on
wide-label sets is removed (`PROGRESS.md`, 2026-10-06 review entry).

**Nothing in this plan closes 0.31.** The leaders are a different predictor, not a tuning
distance, and nine hypotheses have established that the task is bound by data and
representation rather than by anything adjustable in what we hold. Every phase below is
scoped as *non-zero expected value*, with its prior stated, and the **report is the
deliverable** — the ρ-curve, the metric error, the shrinkage finding and §8's rule about the
range a negative was tested over are usable by other people regardless of rank.

## Standing rules this round inherits

1. **Effect size over noise band, in the same sentence as the proposal.** Band sd is 0.0095 on
   validation macro ST-RAE. Below ratio 1, the run does not happen
   (`feedback-effect-size-before-gpu`).
2. **A negative is only as strong as the range it was tested over.** State the range the
   variable took and check the target regime is *inside* it, or the verdict is "untested"
   (`feedback-negatives-need-range`).
3. **A flagged unknown blocks every decision downstream of it.** Resolve from the authoritative
   source or do not make the decision (`feedback-flagged-unknown-blocks`).
4. **Never pool runs across a metric change.** `metric_id` on every artifact; this round adds
   `config_id` for the same reason (Phase 0).
5. **Discount validation gains by ~17%** and prefer levers that do not route through the
   validation score at all (`feedback-validation-gains-transfer-at-17pc`).
6. **Compare the validation set to the evaluation set before tuning on it.**

## Phases

Expanded one at a time. Each lands its own section in `PROGRESS.md` and a `CHANGELOG.md` row.

### Phase 0 — Guardrails and cheap infrastructure

Enabling work, no score claim. Three items, all of which remove a known defect source rather
than chase a gain.

- **0a. Featurisation cache.** ECFP + RDKit2D + Avalon + ErG over 6,525 molecules is
  deterministic and is the actual CPU bottleneck — it is why a 12-vCPU A100 beat a 4-vCPU L4,
  not the GPU. Compute once, content-hash the inputs, write `.npy`, push to the Hub beside the
  checkpoints, mount on every later run. After this the L4's vCPU count stops mattering.
- **0b. `config_id` on every artifact.** The ensembling gain came in at −0.0168 against a
  −0.0102 curve measured at 16 members/400 epochs instead of 32/300. An effect measured at an
  experiment-sized config is not an effect for the shipped model. Either state the config with
  every number or prove config invariance once; this phase makes the former mechanical.
- **0c. Seed-parallel L4 harness.** Seeds are the binding resource for *measurement* — n=10 vs
  n=4 is what killed the loss ablation's inflated estimate, and Phase 4's pre-registration
  needs |t| > 2.5. Seeds are embarrassingly parallel; N L4s run N seeds at a fraction of the
  A100 rate. Reserve the A100 for the 32-member build that genuinely needs 37.1 GB.
  Boundary rule: **vmap anything under a few minutes, isolate anything over an hour** — 24
  permutation draws vmapped in 31 s is right; the A100 that died twice mid-experiment is why
  long work gets process isolation plus Hub persistence.

### Phase 1 — Shrinkage, re-derived under the test label spread

**CPU-only, runnable on the laptop, and the one place a standing clean negative was tested
over too narrow a range.**

`match_sd → 1.0142` (as bad as predicting the mean) was measured on validation at label sd
**0.944**. The test spread is **~1.29**. Optimal shrinkage is a function of label spread: at
the same predictive correlation, wider labels move the optimum toward wider predictions. Our
prediction sd is ~0.478, a ratio of 0.51 to validation's labels; the same ratio at the test
spread wants ≈0.66. So the conclusion "shrinkage is correct, do not fix it" was reached on a
distribution **27% narrower** than the one we are scored on — exactly rule 2 above, sitting in
the one place this programme has not looked.

Done **without test labels**: resample validation to the 1.29 spread *preserving the
prediction–label correlation*, re-optimise the global scale under the corrected ST-RAE, and
check whether the optimum moves. `cyp_metric_vs_spread.py` already built most of the
machinery. The interval half-widths are a property of assay quality rather than of label
spread, so the resampling needs a measured width-vs-y model — that measurement is part of this
phase and is a blocker for it, not a caveat.

Pre-registered before any submission. Prior: **moderate** — the mechanism is arithmetic rather
than empirical, but its size at the realised interval widths is unknown until measured.

### Phase 2 — Data completeness, and the Octant public CYP screen

Two parts, cheapest first.

- **2a. Are we training on everything the release contains?** We hold 6,525 primary labels; the
  announcement describes mapping inhibition for >10k molecules. Audit the repo file list
  against what `drugrx/cyp/data.py` loads. This has already paid once — a plausible-looking
  column filter silently dropped 6 heads carrying 9,223 labels.
- **2b. Screen Octant's public CYP set.** ~51,400 rows covering CYP3A4 and CYP2J2, well-level
  readouts and quality annotations. **Same lab, same platform, same assay conditions as the
  challenge data**, which removes the cross-lab offset that killed the ExpansionRx `cl_micro`
  merge — the cleanest acquisition candidate this programme has had. Screen it the way Veith
  was screened, at a **pre-registered coverage threshold** against the 750 test SMILES.
  It touches only 3A4 of four isoforms, so the macro effect is capped near a quarter; it is
  the only source without the structural objection.

### Phase 3 — Nested CV for the blend weight and the per-endpoint offsets

**Not a score improvement — a stop to the thing that produced 17% transfer.** The blend weight
and the offsets are currently fitted and read on splits that are not nested, so their
contribution is an optimistic estimate of itself. Necessary regardless of what else runs, and
it ships with whatever ships.

### Phase 4 — The anchored pairwise/delta model

Already pre-registered in `PROGRESS.md` (2026-10-06): ŷ = y_anchor + Δ̂, scored on pair-space
validation at Tanimoto ≥ 0.5 **and** on the molecule-level analog-expansion split, paired
across seeds, |t| > 2.5, **reported per endpoint**.

The scarcity numbers are the binding constraint and they are already measured: 596 pairs at
≥0.6, of which **520 are CYP3A4 and 16 CYP2C9**. That makes this a CYP3A4 experiment wearing a
four-isoform label. Report per endpoint as pre-registered and do not let a 3A4 gain carry a
macro claim. CYP2C9 is a null by construction (R > 1 on 16 pairs).

CPU-feasible — 596 training examples does not need an A100.

### Phase 5 — 3D interaction features *(conditional: only if 1 and 2 both null early)*

The one orthogonal axis left, and the only workload in this plan that genuinely saturates an
A100: co-fold the 750 test compounds against the four CYP structures and derive interaction
features. For analog series where 2D fingerprints are near-identical and activity is not,
docking-derived features are the standard answer, and it is the only information source not
yet tried that is not a rearrangement of the same 2D input.

Honest prior: **low**. CYP3A4's pocket is large and promiscuous, docking scores correlate
weakly with potency, and it is expensive. Genuinely orthogonal, low probability — spend
accordingly.

### Phase 6 — Report and ship

The deliverable. Method report with every null and its measured reason; the trained artifact
served into FoldRx as a `ModelImpl` with its `caveat`; the candidate-ranking playground.

## Sequencing

Phases 1 and 2 **this week** — both cheap, both test something the evidence says is live.
Phase 3 folded into whatever ships. Phase 4 as the pre-registered experiment. Phase 5 only on
an early double null in 1 and 2. Phase 6 throughout, not at the end.

Compute posture: **A100s stopped**, features cached, seeds on parallel L4s, burn at 0.00/hr
between experiments. The constraint here was never compute — 1,817 of 2,000 units remain.

---

## Re-scope, 2026-10-06, after Phase 1 and the per-endpoint read

The plan above was written before the per-isoform leaderboard columns were found. What they
showed changes the ordering, so the original phase list is kept as the record of what was planned
and this is what is actually being done.

### Settled

- **Phase 1 — done, and it split in two.** The *scale* lever is real but second-order
  (−0.081 macro at the test-side correlation) and the *level* is the story. Its magnitude cannot
  be modelled — the reconstruction decoder was refuted on 4/4 endpoints — so it is being
  **measured** with rescale probes, whose identification self-tests to 0.0006 on `s*`. Three
  verified files are built and waiting on the owner to submit.
- **Phase 2a — done.** There is no large unused pile of released labels. The ">10k molecules"
  premise was a different post's figure; the announcement says "~1,500 DRCs per CYP". One
  candidate survives: 1,240 compounds with a TDI-arm pIC50 and no direct label.
- **Phase 2b — dropped.** Octant's public CYP set is 1,340 inhibition rows (CYP3A4 only, not
  51,400), and 1,250 of those compounds are already in training (r = 0.961 against our
  `CYP3A4_pIC50_TDI_condition`). ~90 new compounds, on the endpoint we are already good at.
  One blinded test compound (`OCNT-0493952`) is in it and would have to be dropped.

### Phase 2D6 — the broken endpoint *(new, takes Phase 2b's slot)*

CYP2D6 at **1.4577** against a per-endpoint leader at 0.4135. Worth **−0.261 macro** on its own
if closed, against −0.081 for the global scale — and merely reaching the mean predictor's 1.0 is
worth −0.114, roughly 50 ranks.

Two distinguishable failures, and both are measurable on data we hold:

- **Level.** R² −0.9509 with MAE 1.7446: we are off-centre. Mechanism is the activity-promotion
  shift between training and test labels. The probe measures the size; the direction is settled.
- **Ranking.** ρ 0.3302 on test, and 0.3913 on validation — CYP2D6 is our *weakest* endpoint on
  both, so part of this is not a decoding problem at all. CYP2D6 inhibition has a specific
  pharmacophore (a basic nitrogen at the right distance from an aromatic ring, to Asp301), which
  is the one place a mechanistic descriptor should pay.

**~~First experiment: re-examine the protonation/basicity descriptors per endpoint.~~
WITHDRAWN before any compute was spent — the premise was false.** The reasoning was that +0.0071
is a *macro* number for a CYP2D6-specific mechanism, so a macro mean had divided a
single-endpoint effect by four. It had not: `cyp_basicity_test.py` is per endpoint by
construction, **+0.0071 is CYP2D6's own figure** against a 0.013–0.021 fold sd, and nested
selection then returned **−0.0002 [−0.0113, +0.0110]** on that endpoint. Tested twice, per
endpoint, refuted twice. Checking the log before launching is what the log is for.

**The actual first experiment was free and is already done: decompose the gap.** CYP2D6 is
**74.3%** of the entire validation→test gap; CYP2C9 is **−3.8%** (it scores *better* on the
blinded set than on our own split); and ρ **rose** on CYP1A2 and CYP2C9. CYP2D6's ρ moved −0.061
while its R² moved **−1.095**, so its failure is **level, not ranking** — a decoding error, which
is exactly what the rescale probes measure. See `PROGRESS.md`.

Realistic target, stated so it is not over-sold: decoding cannot fix CYP2D6's *ranking*
(ρ 0.33 on both splits), so leg 1B's curve at r ≈ 0.33 puts the achievable score near
**0.9–1.0**, not the leader's 0.4135. Macro would land near **0.73** — worth ~50 ranks, and not a
contender.

### Phase 3 — nested CV *(reduced)*

Largely overtaken: the probe base vector **drops** the blend and the offsets, which were what
needed nesting. Keep the rule — anything fitted on validation gets nested before it ships — and
re-apply it if a fitted component comes back.

### Phase 4 — the anchored pairwise model *(unchanged, and now better aimed)*

Still pre-registered, still weak-prior. One update: it was flagged as "a CYP3A4 experiment
wearing a four-isoform label" (520 of 596 pairs at ≥0.6 are CYP3A4). That is now a *point in its
favour on 3A4* and irrelevant to CYP2D6, which has 16 pairs and no measurable anchor signal
(R > 1). It cannot help the endpoint that is actually broken. **Report per endpoint; it will not
move the macro.**

### Phase 5 — 3D interaction features *(still conditional, and now better motivated for one endpoint)*

Low prior overall, but CYP2D6 is the one isoform where the binding determinant is specific and
well characterised, and where a 2D fingerprint plausibly misses it. If Phase 2D6's cheap
descriptor route nulls per endpoint, this is the next thing with a mechanism behind it — scoped
to **CYP2D6 only**, not all four.

### Free, and currently misrepresenting the work

Our board row shows **Model Report Link: "Not submitted"** and **Open Code: "No"**, while the
programme has an honest method report and published artifacts (`arnab2812/drugrx-admet{,-ensemble,
-router}`, the `arnab2812/drugrx-cyp-runs` dataset). Both flags are free to set and both are
true. Do it with the final submission.
