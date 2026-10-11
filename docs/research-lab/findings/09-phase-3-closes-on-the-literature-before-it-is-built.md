# 09 — Phase 3 (SIGReg on a learned molecular encoder) closes before it is built, on four independent lines

**Status: closed on price, by the lab's own corollary** — *compute the ceiling over the whole
method family before building any member of it.* Here the ceiling was computed by other groups
with 10²–10⁴× our compute, which makes this the cheapest ceiling calculation this lab has done.
Literature sweep run 2026-10-11 by a Fable shell; the two load-bearing claims were then verified
against the primary sources by hand, verbatim.

## The decisive paper: someone has already run Phase 3

**Self-Supervised Pretraining of Molecular Graph Encoders with LeJEPA** — Michał Kulczykowski,
Rafał Łabędzki. arXiv [2609.04261](https://arxiv.org/abs/2609.04261), 2 Sep 2026. **Verified
against the arXiv abstract directly, not taken from the shell's report.**

They adapt **LeJEPA — the exact object, SIGReg and all** — to molecular graphs, on **GPS and
Chemprop-style D-MPNN** encoders, with a multi-seed bootstrap protocol. Verbatim from the
abstract:

> "Pretraining improves learned representations but does not robustly improve finetuning. A
> frozen probe on pretrained embeddings exceeds random initialisation on both tasks (ogbg-molhiv
> ROC-AUC 0.788 vs 0.665; +0.123), reaching the published self-supervised band, but this does not
> translate into finetuning gains. On the antibiotic scaffold split, a canonical partition is
> significant (delta AUPRC +0.041, p = 0.010), but the effect vanishes across five partitions
> (pooled +0.013, p = 0.095). Finetuning is null on the random split, ogbg-molhiv, and D-MPNN."

Note **which** encoder the finetuning null includes: **D-MPNN**, which is Chemprop, which is
`scripts/cyp_dmpnn.py` — the architecture already in this repo. And note the shape of the result:
a single partition significant at p = 0.010 that **does not survive four more partitions**. That
is this lab's trap #2 (`feedback-slices-share-noise-resamples-dont`) measured in someone else's
experiment, on the precise method we were about to build.

Unrefereed, two unknown authors, 0 citations — but the protocol is stricter than most refereed
work in this area, and it is the experiment we would have run.

## Three more lines, all pointing the same way

**1. Balestriero's own molecular JEPA loses to descriptors + a tabular model.**
**Mol-JEPA** (Rottach, Schieferdecker, Rudman, **Balestriero**, Eickhoff — Boehringer Ingelheim +
Brown; arXiv [2608.22642](https://arxiv.org/abs/2608.22642)) is the same senior author as LeWAM.
402,755 molecules, eight modalities, 313 GB, 23 ADME/potency datasets on Taylor–Butina cluster
splits. Average MAE puts Mol-JEPA (Best) at 0.402 ahead of TabICLv2+AlvaDesc at 0.453 — **but the
paper's own paired Wilcoxon signed-rank test, FDR-corrected, gives Mol-JEPA a 38% win rate
against 47% for TabICLv2 + AlvaDesc.** The authors say so plainly and narrow their claim to
low-data and out-of-distribution settings. Two flags worth carrying: the headline column is a
**max over probe and embedding variants taken on the test split** (the honest column is the
Transformer head at 0.427), and LoRA finetuning *underperformed* most other methods.

**Our pipeline is already "descriptors + gradient-boosted trees", which is the family that won
that significance test.** CheMeleon finetuned (0.471) and LGBM+AlvaDesc (0.468) are tied in the
same table.

**2. The strongest molecular JEPA to date gets exactly zero on bioactivity.**
**Atom-JEPA** (Petersen, Tirsgaard, Cornet, Jordahn, Schmidt — DTU; arXiv
[2610.08400](https://arxiv.org/abs/2610.08400), 6 Oct 2026) pretrains on **~19M molecules + 1.7M
crystals** and wins decisively where the physics is clean: lowest MAE on 8/12 QM9 targets, **36%
mean relative MAE reduction vs EquiformerV2**, 6/8 Matbench tasks, 10.1–14.5% from pretraining on
Matbench regression. And then, in their own ablation: **no improvement at all from pretraining on
ChEMBL-MT** — the bioactivity benchmark.

**That split is the whole finding.** CYP direct inhibition is bioactivity, not quantum chemistry.

**3. The frame that predicts the split, and the parity test that prices CheMeleon.**
Grisoni, *How "Foundational" Are Current Molecular Foundation Models?* (arXiv
[2609.37550](https://arxiv.org/abs/2609.37550)) finds none of the examined approaches satisfies
all three of generality, transferability and out-of-distribution generalisation, and that success
concentrates **"in domains where target properties are consistently defined and training data are
abundant, with low label noise relative to physically meaningful variation."** CYP inhibition from
a selected, screen-pseudo-labelled 6145-compound set is none of those three.

And, **verified directly**: *Interpretable-by-Design Descriptor Portfolios Match a
2048-Dimensional Foundation Embedding on Low-Data Molecular Assays* (Yao, Duran-Frigola; arXiv
[2609.30789](https://arxiv.org/abs/2609.30789)) reports, across nine ADME/Tox assays and 50
evaluation cells, portfolio **0.762** vs **CheMeleon 0.764** vs Mordred 0.756 — pooled gap
**+0.003 AUC, task-bootstrap 95% CI [−0.020, +0.030]**, satisfying their **pre-declared pooled
parity gate** (though *not* their per-assay threshold — a nuance the shell's summary dropped and
worth keeping).

## The price, in this lab's units

| source | pretraining corpus | gain on *bioactivity* |
|---|---|---|
| molecular LeJEPA | large unlabelled corpus | **frozen-probe +0.123, finetuning null**, effect gone across 5 partitions |
| Mol-JEPA | 402,755 molecules, 8 modalities, 313 GB | **38% win rate vs 47%** for descriptors+TabICL |
| Atom-JEPA | 19M molecules + 1.7M crystals | **none** |

Our own encoder would pretrain on ~6k labelled compounds, or ~10⁵–10⁶ unlabelled if we pulled
ChEMBL — still **20–200× less** than Atom-JEPA, which got nothing here. The measured family
ceiling on bioactivity is **zero**, against a **13.6%** requirement. There is no version of this
that gets built.

**`PLAN.md` priced Phase 3 as "in the live zone *if* an encoder can be trained at all on 6k", and
named the most likely failure mode in advance: "no uncontested molecular augmentation family".
Mol-JEPA agrees — it replaced augmentations with modality masking specifically because molecular
augmentations are chemically invalid — and it still lost its own significance test.** The
pre-registered failure mode was the right one.

## What the sweep opened rather than closed

**A live threat to this lab's validation instrument.** *Clever Hans in Chemistry: Chemist Style
Signals Confound Activity Prediction on Public Benchmarks* (Blevins & Quigley, arXiv
[2512.20924](https://arxiv.org/abs/2512.20924)) trains an **1,815-class author classifier from
molecular fingerprints** and reports **60% top-5 accuracy under scaffold-based splitting**, then
shows an activity model given only a protein identifier and a structure-derived author-probability
vector matches a structure-based baseline. **Scaffold splits do not break the leak**; they propose
author-disjoint splits.

Every arm comparison in this lab uses `scaffold_split`, and we already know
`cyp-set-is-82pc-scaffold-singletons` — our scaffold split is **near-random**, which makes the
leak *more* available, not less. This is cheap to test on our own data (we hold ChEMBL provenance
via `scripts/cyp_chembl_build.py` and `scripts/chembl_audit.py`) and it is now the most valuable
open question the sweep produced. **Not claimed either way here.**

**The unoccupied hole, for the sister programme.** Learned surrogates that gate expensive
*structure prediction* — predicting whether an AlphaFold/ESMFold/Boltz run is worth paying for —
are **essentially unpublished**, while the docking equivalent is crowded (Deep Docking 100×,
DDU 45× over that, MolPAL 94.8% of top-50k at 2.4% docked, Thompson Sampling >half the top-100 at
1%). That is FoldRx's territory, not this lab's, and it is recorded in `PLAN.md` Phase 4.

**And the caveat that reframes every docking-surrogate number:** *Understanding active learning of
molecular docking and its applications* (Kim, Nam, Ryu — HITS; arXiv
[2406.12919](https://arxiv.org/abs/2406.12919)) finds **"surrogate models tend to memorize
structural patterns prevalent in high docking scored compounds obtained during acquisition
steps."** The surrogate learns the chemotype the loop has been enriching, not the docking
function. Good for ranking, not for calibrated scores. Combined with our own
`vina-has-no-metal-coordination-term`, the naive "learn a Vina surrogate" idea is dead twice over
for CYP.

## Two literature results that corroborate our own negatives

Worth recording, because independent agreement on a negative is the cheapest validation there is:

- **Active learning accelerates *search*, not *model quality*.** Barrett & White (*JCIM* 2021):
  "We find neither active learning method tested to be better than random choice." Ghosh et al.
  (*J. Chem. Phys.* 2025): for dataset compilation "AL did not outperform the randomized
  selection of data points", while targeted search saved up to 64%. Kashafutdinova et al.
  (*Mol. Inf.* 2025) reproduce the same split independently.
- **No UQ method ranks per-molecule errors reliably.** Hirschfeld, Swanson, Yang, Barzilay, Coley
  (*JCIM* 2020, 341 cites): "none of the methods we tested is unequivocally superior to all
  others, and none produces a particularly reliable ranking of errors across multiple data sets."
  That is findings 04–05 — our closed σ̂ 2-D decode family — arrived at independently, and it says
  ensemble spread is fine for prioritisation and not fine for calibrated gating.

## The methodological note

This finding cost no GPU and settled a phase that `PLAN.md` had priced into the live zone. The
lab's corollary says to compute the family ceiling before building a member; it does not say the
ceiling has to be computed *here*. **Three groups had already spent 10²–10⁴× our compute on it,
and one had run our exact experiment.** Checking that first should be the default step of any
phase that is about to be expanded, and it is now in `AGENTS.md`.

---

## ADDENDUM 2026-10-12 — **THIS CLOSURE WAS WRONG, AND PHASE 3 REOPENS**

A citation audit went to all seven primary sources. **No citation was fabricated and almost every
number I quoted was transcribed correctly.** The failure was in the inference, and it is a clean
example of reading the one result that agrees with you.

### The error: I closed a low-data programme on a high-data null

My argument was *"Atom-JEPA pretrains on 19M molecules and gets no improvement on bioactivity,
therefore JEPA pretraining does nothing for bioactivity."* The null is real and it is a main-text
result, not an ablation — Table 4, `No pretraining 0.4543` vs `Atom-JEPA 0.4567`, **Δ = −0.5%,
improved on 12/25 endpoints, p = 0.57**. But:

| benchmark | n | pretraining effect |
|---|---|---|
| Biogen ADME | **3,521** | **−9.6% MAE, p = 0.011, 6/6 endpoints** |
| ExpansionRx | **7,608** | **−8.5% MAE, p < 0.001, 9/9 endpoints** |
| ChEMBL-MT | **114,112** | −0.5%, p = 0.57 — the null I quoted |

**The null sits at 15–30× the data scale of the two wins.** Our per-isoform real-label counts are
**1285–2335** — the Biogen/ExpansionRx scale, *not* the ChEMBL-MT scale. So the pattern is at least
as consistent with *"pretraining stops paying once labels are plentiful"*, which is the **opposite
operational conclusion for a programme with ~1.3–2.3k labels per endpoint.**

And **the authors explicitly disown the null**: *"ChEMBL-MT does not see an improvement, but we
suspect that it is confounded by hyperparameters not generalizing well from Biogen to the much
larger dataset size."* They reused CKERMT's hyperparameters and zeroed encoder-freezing and L2SP
for ChEMBL-MT rather than tuning it. I quoted the number and omitted the authors' own stated
confound.

**Two further corrections to the same paper, both mine:**
- **The "36% mean relative MAE reduction vs EquiformerV2" is not evidence about pretraining at
  all.** EquiformerV2 is, in the paper's words, *"the predecessor of our EquiformerV3 backbone"* —
  so that figure bundles an architecture upgrade with pretraining. The pretraining effect in this
  paper is the 8.5–14.5% range from the random-init comparison. **Deleted from my reasoning.**
- "Matbench lowest error on 6/8" dropped its qualifier: *"Among pretrained models with at most 4M
  pretraining structures."* Immediately afterwards, **JMP-L beats Atom-JEPA on all seven
  regression tasks.**
- Disclosed but omitted by me: *"For most ADMET endpoints, Unimol contains somewhere around
  35%−75% of the structures of molecules seen in the endpoints."* That contamination applies to the
  ADMET results.

### Three independent lines now point the *same* way, and it is not my way

- **Atom-JEPA**: pretraining wins at 3.5k and 7.6k labels, null at 114k.
- **Mol-JEPA**: the advantage is data-size dependent — TabICLv2 wins via the *larger* ExpansionRx
  sets, Mol-JEPA on the *smaller* ASAP/Biogen ones, and the paper says it *"may offer particular
  benefits in low-data settings."*
- **Molecular LeJEPA (C1)**: see below — its positive result is real and better controlled than I
  gave it credit for.

**Phase 3 therefore reopens**, in one specific form (below). `PLAN.md` is updated.

### The result I under-weighted, and what it actually claims

I treated molecular LeJEPA (arXiv 2609.04261) as purely negative. Its concatenation result is
**positive and unusually well controlled**, and it claims something different from accuracy:

- Truncated pretrained embedding **⊕ a 1024-bit Morgan fingerprint**: **0.803 → 0.830** (+0.027,
  95% CI [+0.003, +0.054], p = 0.014). *(I previously quoted 0.805 → 0.832 from the abstract; the
  paper's own Table 6 and conclusion say 0.803 → 0.830. Δ, CI and p are identical.)*
- **An untrained encoder through the identical pipeline gives −0.003** (p = 0.59), and *"across all
  twelve cells the pretrained embedding beats its untrained twin in every one."*
- **An internal replication I missed**: at 512 bits, fingerprint alone 0.790 → +LeJEPA PCA-32
  **0.821**, with the random-init control actively *harmful* at 0.773.

**And here is the part that matters most for us.** The gain is **zero on validation** — 0.847 vs
0.845, which the paper calls *"well within noise"* — and appears only on shifted test scaffolds.
Verbatim: *"The embedding is therefore not adding signal that the validation chemotypes reward; it
is adding signal that survives the change of chemotype."*

**That is exactly our situation.** This programme's central measurement problem is that our
validation is a diversity scaffold split while the board's test set is a **purchased analog
expansion** — test within-set similarity 0.665 against validation's 0.354, and no re-splitting of
our data can emulate it (`feedback-check-the-split-resembles-the-test`). A method whose benefit is
**invisible on validation and appears under chemotype shift** is precisely the method our
validation would reject and the board might reward. **I closed this family using the instrument
that is structurally blind to its benefit.**

Honest limits, which the paper states itself: one benchmark, one split, **130 test actives**; the
truncation dimension is a fitted hyperparameter and the k=64 dilution control collapses the gain to
+0.008; the CI lower bound is +0.003, barely clear of zero; and *"because the gain is invisible on
validation, it cannot be selected for."* So it is a hypothesis to test on our data, not a banked
result — but it is a *live* hypothesis, which is more than I previously credited.

### Clever Hans: my alarm was overstated

The numbers are verbatim correct (1,815 classes, Top-5 **0.60** against a dummy at 0.0015;
author-only activity model 0.652 AUROC vs ECFP+protein 0.656). But it does **not** undermine every
scaffold-split number here:

1. **Both models are weak.** The "structure-based baseline" it matches is AUROC **0.656**. The
   finding is "author style matches a 0.65-AUROC baseline", not that it explains good SAR models.
2. **That table is validation, not test** — *"mean validation AUROC … over five random scaffold
   splits."*
3. **Error bars swamp the gap**: 0.652 ± 0.064 vs 0.656 ± 0.067 on a 0.004 difference, and ECFP is
   *better* on the other two metrics (AP 0.430 vs 0.399, P@1%R 0.700 vs 0.636).
4. **Heavily curated slice** — 1,815 *prolific* authors, exactly the regime where lab-specific
   series repeat. The paper limits itself twice over to *"this curated ChEMBL slice."*

So Phase 6 drops from "the highest-value open question" to "a cheap check worth running once."

### Two more of my framings were wrong

- **The active-learning "negatives" are not all negatives.** Kashafutdinova et al. is a *split*
  result — AL failed for model accuracy but *"were successful in selecting molecules with desired
  properties using minimal number of tests"* — and Ghosh et al. report **64% savings for targeted
  search**. So **two of my four AL citations argue that AL works for candidate selection**, which
  is the use case a FoldRx acquisition loop would actually have. And **Hirschfeld et al. is a
  uncertainty-quantification paper, not an AL paper** — filing it under "AL negatives" over-reads
  it.
- **"LeJEPA has no peer review anywhere on OpenReview" is unverifiable** and I should not have
  stated it as fact. arXiv shows no `journal_ref` and no comment field, and an OpenReview search
  surfaces only a derivative submission — that is absence of evidence from a fuzzy endpoint that
  cannot see non-public submissions. The defensible wording is **"no venue declared and no public
  OpenReview record found."**

### Evidence grade, stated plainly

**None of C1–C7 has a declared venue; all are unreviewed preprints, mostly n = 1 benchmark.** C1 is
deepsense.ai + SGH Warsaw and carries a generative-AI-use declaration; C5 lists no institutional
affiliation; C2 is the only one with substantial institutional backing (Boehringer Ingelheim +
Brown). **A phase should not have been closed on this evidence grade in either direction** — and
the symmetric point is that it should not be reopened on it either. What reopens Phase 3 is the
*scale argument*, which is arithmetic about our own label counts, not a preprint's claim.
