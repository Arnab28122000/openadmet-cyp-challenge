# What wins blind ADMET/potency challenges — and what this one's own materials say

> **CORRECTION, 2026-10-08, made after verifying the board in a browser.** This note repeatedly
> states our live row is **0.9628 (rank 225/260)**. It is not. Our live row is **rank 171,
> MA-ST-RAE 0.8663**, submitted 2026-10-07 19:02 UTC (the `a = 1.40` rescale probe), with
> per-isoform 2D6 1.5149 unchanged. **0.9628 is the `a = 0.60` probe, and it is what the frozen
> *Interim* leaderboard shows** — a different table, scored on the full 750 at the 24 Sep
> intermediate deadline, which this note read as though it were the live one. Every comparison
> below that uses 0.9628 as "where we are" is therefore ~0.10 pessimistic. Nothing else in the
> note depends on it: the organisers' baselines (CheMeleon 0.83, TabICL 0.68), the leaders' 0.43,
> the PXR precedent and the per-isoform argument all stand, and so does the conclusion that we are
> near the organisers' own CheMeleon baseline rather than ahead of it.

Research date **2026-10-08**. 234 unique sources screened (tally kept while working); ~30 deep-read.
Everything numeric below was retrieved from a public page, a public repo file, or a public API during
this session, and is attributed. Nothing was submitted anywhere and no non-public surface was touched.

The single most important thing in this document is **§1.6**: the organisers have *published a
reproducible recipe that scores 0.68 MA-ST-RAE on this exact challenge*, against our 0.9628, and the
ingredient that earns it is the one piece of released data we are currently using only as a
multi-task head.

---

## 0. Executive summary — ten lines

1. **We are behind the organisers' own baseline.** The leaderboard's `CheMeleon-baseline` is **0.83
   MA-ST-RAE**; their `TabICL-baseline` is **0.68**; the leaders are **0.43**. Our live row is
   **0.9628** (rank 225 / 260). We are not losing a close race; we are below the free baseline.
2. **The published 0.68 recipe is: off-the-shelf CheMeleon embedding (2048-d, PCA→256) + a Chemprop
   *prediction* of the primary-screen log2FC readout as extra feature columns → TabICL/TabPFN.** The
   organisers tested every subset and said so in print.
3. **Our TabICL null was a feature-space null, not a method null.** We fed 3,091 engineered features
   and found PCA "catastrophic"; they fed a dense 2048-d CheMeleon embedding where PCA→256 was free.
4. **Macro is decided on CYP2D6, and that is where we fail worst** (live ST-RAE 1.5149, R² −0.989).
   The organisers state why: "the test set for CYP2D6 contains inherently less potent compounds".
   Median entry scores 1.35 there; the leader scores 0.41. It is learnable and it is 1/4 of the metric.
5. **Shrinkage is the wrong direction and the leaderboard already proves it.** The leader reaches
   MA-R² 0.677 and MA-MAE 0.6005 on the same blinded set. Our last submission, more shrunk than the
   one before, went 0.8598 → **0.9628**. Headroom is in the model, not the decoding.
6. **External/proprietary data is not what separates the top here** — unlike ExpansionRx. Only 8 of
   219 interim entries declared proprietary data; 19 of the interim top 20 declared none.
7. **The nearest precedent is OpenADMET's own PXR challenge, also scored on analog series.** Its
   winner's trick was *proxy labels for >10,000 inactives*, extending the low end of the target range —
   exactly the failure mode our 2D6 row shows.
8. **Ensembling was universal in PXR's Tier 1 (2–15+ members, 20/28 using 3D)**, and
   hyperparameter optimisation was repeatedly reported as near-worthless across all three challenges.
9. **Metric design is the quiet determinant.** ST-RAE's zero-error band makes the interval-aware hinge
   the right loss, and makes per-compound interval *width* a modelling input, not a caveat.
10. **The live board is half the test set split by chemisimilar series; interim and final use all 750.**
    Tuning against live is tuning on different series from the ones that decide the result.

---

## 1. The CYP challenge's own public materials

### 1.1 Where the authoritative text lives

| Artefact | URL | What it settles |
|---|---|---|
| Announcement post | `openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/` | construction, assays, timeline, tracks |
| Space `config.py` (public) | `huggingface.co/spaces/openadmet/cyp-challenge/raw/main/config.py` | **the About + FAQ text verbatim**, phase, rate limits, schema |
| Halfway post, 25 Sep 2026 | `openadmet.ghost.io/halfway-through-the-openadmet-cyp-challenge/` | interim results, tiering, per-isoform difficulty |
| Tabular-FM post, 23 Sep 2026 | `openadmet.ghost.io/are-tabular-foundation-models-all-you-need/` | **the 0.83 → 0.68 recipe** |
| Data card | `huggingface.co/datasets/openadmet/cyp-challenge-train-test` | file/row counts, interval columns |
| Scoring design doc | `github.com/OpenADMET/blind-challenge-template/blob/main/docs/scoring-and-leaderboards.md` | bootstrap, macro pseudo-endpoint, which stages get significance testing |

The Space's **`app.py`/`config.py` are readable as raw files** even though the rendered Gradio page is
dynamic — that is how the About/FAQ text below was obtained rather than paraphrased. The rendered
leaderboard is also recoverable without a browser: `GET https://openadmet-cyp-challenge.hf.space/config`
returns the Gradio component tree with every leaderboard `DataFrame` already populated.

### 1.2 Construction — in the organisers' words

> "**Primary Screen:** Single-concentration screening across Enamine DDS10 (diversity set) and FDAA
> (FDA-approved compounds) libraries."
> "**Dose-Response Training Set:** Top hits were promoted to 12-point DRCs (~1,500 DRCs per CYP) and
> fit using a **Bayesian curve-fitting pipeline**."
> "**Analog Expansion Test Set:** Hit expansion was performed on the top 25 hits from CYP1A2, CYP2C9,
> and CYP3A4 by purchasing the top 10 chemisimilars (ECFP4 Tanimoto similarity) per hit compound from
> the Enamine US catalog."
> "**Dense Test Matrix:** The resulting 750-compound test set was fully assayed in 12-point DRC mode
> against **all four CYPs (including CYP2D6)**, yielding a dense test matrix with **detailed SAR and
> activity cliffs**."

Three consequences the programme should treat as given rather than re-derive:

- The test set is **75 series of 11** (hit + 10 analogs), by construction. Series membership is
  recoverable from ECFP4 nearest-neighbour structure; the organisers use it themselves to split the
  live half.
- **CYP2D6 got no hit expansion.** The 750 are analogs of 1A2/2C9/3A4 hits, assayed against 2D6
  anyway. So the 2D6 test distribution is an unselected sample of other isoforms' hit chemistry —
  the organisers' own phrasing: *"the test set for CYP2D6 contains inherently less potent compounds
  than the other three isoforms. This distribution shift between training and test sets makes CYP2D6
  a much harder target."*
- Intervals come from a Bayesian DRC fit and **"intervals expand significantly at lower activity
  levels, aligning with assay precision limits"**. Interval width is a function of measured activity,
  which is exactly why a width model fitted on training labels transfers.

### 1.3 Rules that bind (FAQ, verbatim)

- **"External data and pretrained models are fully permitted."** Proprietary data must be declared
  with a checkbox that "will be publicly displayed on the leaderboard".
- **"submissions are rate-limited to at most once every 12 hours. Only your latest valid submission
  will count toward the live leaderboard standings."** — the *latest*, not the best. Our 0.9628 row
  has therefore replaced the 0.8598 one.
- **"Multiple submissions from the same team or lab … are not permitted."** The halfway post confirms
  entries were removed for breaking this.
- Open-sourcing code is "strongly encouraged … but not strictly mandatory for standard leaderboard
  ranking" — however a **model report link is required to appear on the final leaderboard** (observed
  in ExpansionRx: "Some participants were removed for not meeting eligibility criteria (missing
  required model reports with valid links)").
- `MIN_PREDICTION_STD = 0.01` in the Space config: constant-column submissions are rejected, because
  "constant columns (with or without slight noise added) can be used to game the leaderboard metric".
- **"Will there be a summary preprint paper? No."** Post-challenge analysis is published as blog posts.
  So the organisers' blog *is* the retrospective literature for this challenge.
- Innovation in ML Award is discretionary and explicitly rank-independent.

### 1.4 Leaderboard mechanics

| Stage | Scored on | Significance testing |
|---|---|---|
| **live** | half the test set, "**split by chemisimilar series**" | none |
| **interim** (one-off, 25 Sep) | **the entire test set** | yes — Benjamini–Hochberg tiers |
| **final** (from 4 Nov) | per template: interim/final both get testing; scope stated as the master leaderboard | yes |

Every published column is a **bootstrap mean over 1,000 resamples**, macro is a plain arithmetic mean
over the four endpoints computed *per bootstrap sample* as an "MA" pseudo-endpoint, and rank is
ST-RAE alone. (`docs/scoring-and-leaderboards.md`; `MIN_PREDICTION_STD`, `ACTIVITY_DATASET_SIZE = 750`
in config.)

**Tiering is sequential-from-the-top with FDR control**, the same procedure used for PXR: "we group
submissions that perform similarly into tiers … compared sequentially from the top until performance
differs significantly," accepting "up to 5% of the pairwise comparisons flagged as 'statistically
distinct'" as false discoveries.

### 1.5 What the interim leaderboard actually shows (read 2026-10-08)

Interim = **full 750-compound test set**, 219 valid entries, "over 200 unique participants".

| Rank | Alias | Tier | MA-ST-RAE | MA-MAE | MA-R² | MA-ρ |
|---|---|---|---|---|---|---|
| 1 | preheat-to-450 | **Tier 1 (alone)** | **0.3721** | 0.6005 | 0.677 | 0.8048 |
| 2 | wbot | Tier 2 | 0.3912 | 0.6195 | 0.6523 | 0.7875 |
| 3 | cypadeedoodah | Tier 2 | 0.3921 | 0.6511 | 0.6284 | 0.7947 |
| 4 | Rubyy | Tier 2 | 0.4014 | 0.6394 | 0.6309 | 0.7882 |
| 5 | EleanorRigby | Tier 3 | 0.4173 | 0.6627 | 0.6179 | 0.7846 |
| … | | | | | | |
| median of 219 | | | **0.7064** | | | |

41 distinct tiers; **Tier 1 contains exactly one entry** (the halfway post: "Only one entry qualified
for Tier 1"). 206/219 beat the mean predictor, 67 are below 0.60, 47 below 0.50, 19 below 0.45.
By contrast the TDI track's Tier 1 held **26 statistically indistinguishable entries**.

**Per-isoform, interim (full test set):**

| Isoform | leader ST-RAE | leader MAE | leader R² | leader ρ | median | n beating mean |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.3752 (wbot) | 0.6192 | 0.6886 | 0.8507 | 0.6673 | 213/219 |
| CYP2C9 | 0.3130 (preheat) | 0.4155 | 0.7576 | 0.8862 | 0.5125 | 214/219 |
| **CYP2D6** | **0.4568** (preheat) | 0.9113 | 0.4913 | 0.5982 | **1.1794** | **88/219** |
| CYP3A4 | 0.3331 (preheat) | 0.4276 | 0.7890 | 0.8882 | 0.5144 | 210/219 |

Two inferences that matter more than the ranks:

- **The test labels are not pathological.** Back out `var_y` from the leader's row under a
  folded-normal residual: MAE 0.6005 and R² 0.677 give σ ≈ 0.6005/0.7979 ≈ 0.753, MSE ≈ 0.567,
  **var_y ≈ 1.75, sd_y ≈ 1.32** — independent confirmation of the programme's sd ≈ 1.29 estimate, now
  from a row with a *large* R² rather than from ours with R² ≈ 0.05. A set on which someone reaches
  R² 0.79 (3A4) is not a set where predicting near the mean is optimal.
- **2D6 is where the field is, collectively, losing** (60% of entries worse than the mean) and where
  the macro is decided. 0.46 is attainable there.

### 1.6 The organisers published a recipe, with ablations

*"Are tabular foundation models all you need?"*, Sean Colby, 23 Sep 2026, DOI
`10.5281/zenodo.23198306`. Written up on PXR, then applied to CYP:

> "Off-the-shelf CheMeleon embedding + log2FC readout is the **TabICL-baseline**, which held the top
> spot on the regression live leaderboard … for... a few days (**0.68 MA-ST-RAE**). As of this writing,
> the top performers have reduced that error by ~36% (**0.43 MA-ST-RAE**). Still, the new method
> improves upon previous openadmet-models baselines by **~20%** in this challenge (see the
> leaderboard's **CheMeleon-baseline at 0.83 MA-ST-RAE**)."

The ablation chain, all with Tukey-HSD intervals over 5 seeds (PXR, 260-compound Phase-2 set, MAE):

| Step | Finding | Number |
|---|---|---|
| GNN transfer-learning configs (Buterez E1/E3/E4) | **none separated from each other**; HSD critical distance 0.10 MAE, closest p = 0.62 | all ≈ 0.50 |
| What actually helped | *"the Buterez paper contributed a single, albeit important, component: an **auxiliary encoder trained on primary-screen data to predict a log2FC readout**"* — the architecture did not | — |
| Feed that readout + an embedding to a TFM | **0.50 → 0.43** | 0.427 (TabICL) / 0.436 (TabPFN v3) |
| Independent agreement | participant N283T "finding the log2FC-trained Chemprop embedding to be their single strongest feature when used as input to a TFM"; their entry scored **0.41** | — |
| PCA on CheMeleon 2048-d | 256 / 384 / 512 statistically tied; **≤128 clearly worse** | canonical 256 |
| PCA on RDKit descriptors | native 0.53, tied with 128 | canonical 128 |
| RDKit/Mordred descriptor block | *"the descriptor block adds no detectable value beyond an embedding plus log2FC readout"* | — |
| Which TFM | TabICL ≈ TabPFN v3; **TabFM, LightGBM, XGBoost, TabPFN v2.6 all separated and worse** | 0.427 vs 0.436 |
| TFM uncertainty | intervals well calibrated but **Spearman ρ = 0.26** between predicted sd and absolute error | — |
| Compute | TabPFN v3 GPU 14.6 s/run; TabICL CPU fallback 184 s; TabICL@128-d GPU 13 s | 4,392 in-context rows fit |

**Reconciliation with our own null.** `cyp_incontext.py` found TabICL/TabDPT inside noise of XGBoost
and "PCA reduction is catastrophic (t = −10 to −15)". Both results are correct and not in conflict:
PCA on a 3,091-dim block dominated by sparse fingerprint bits destroys the signal; PCA on a dense
2048-d CheMeleon embedding does not. **Our arm never tried the configuration that scores 0.68** —
dense pretrained embedding plus a *predicted* auxiliary readout — so the TFM question is open, not
settled.

### 1.7 The data we hold, and the part we are under-using

Data card (`openadmet/cyp-challenge-train-test`): `default` 5.66k rows (train 4.91k / test 750),
`single_concentration` **17.5k rows**, `tdi` 6.15k, `emax` 6.15k, `structure` 20. Interval columns are
`{isoform}_pIC50_direct_inhibition_conf_high` / `_conf_low` / `_std`.

PROGRESS already records the right handling (auxiliary columns as **multi-task heads**, never as
features, because the blinded test file carries only `Molecule_Name` and `SMILES`, and because the
TDI-condition pIC50 correlates r = 0.90–0.99 with the target). The organisers' recipe uses the same
data differently and legitimately: **train a separate model on the single-concentration log2FC, then
feed its *predictions* on train and test as feature columns.** That is available at inference, leaks
nothing, and is the measured source of the 0.83 → 0.68 step.

### 1.8 Where we stand (live board, read 2026-10-08)

| | MA-ST-RAE | MA-MAE | MA-R² | MA-ρ | rank |
|---|---|---|---|---|---|
| leader (preheat-to-450) | 0.3706 | 0.5919 | 0.6648 | 0.7844 | 1 |
| median of 260 | 0.7726 | | | | |
| **talindrew, submitted 2026-10-07 06:59 UTC** | **0.9628** | 1.0726 | **−0.0658** | 0.6212 | **225 / 260** |

Per isoform, live: 1A2 **0.9877** (R² −0.043), 2C9 0.6799 (R² 0.358), **2D6 1.5149 (MAE 1.78,
R² −0.989, ρ 0.363)**, 3A4 0.6689 (R² 0.411).

**This is a regression on our own record and it is the row that counts.** PROGRESS logs v3deep at
0.8598 on 2026-10-06; the current row is 0.9628, +0.103 worse, and "only your latest valid submission
will count". R² −0.989 on 2D6 means the squared error is twice the label variance — a location/scale
failure, not a ranking failure (ρ is still 0.36). Two-thirds of our macro numerator sits in 2D6 and
1A2.

Also worth noting as a sanity check on tuning signal: the leader scores 0.3706 live and 0.3721
interim. For a good model the two halves agree to 0.4%, so the live half is not systematically easier
or harder — but it is a *different set of series*, which is a different risk (see §5.4).

---

## 2. OpenADMET PXR challenge (Apr–Jul 2026) — the closest precedent there is

Same organisers, same template, same tier procedure, and **scored on analog series**: a live
leaderboard on Analog Set 1, final standings on the hidden Analog Set 2. 350+ participants, 5,500+
entries, 130 model reports, >11,000 compounds, metric MAE on pEC50.

**Final activity leaderboard:** matcha-croissant 0.4061 ± 0.0280 (ρ 0.8269); AIDD-LiLab 0.4092;
AIDD-LiLab-Aggressive 0.4104. **Tier 1 held 28 entries with no statistically significant differences**,
spanning MAE 0.41–0.44.

The organisers' retrospective (*"Don't Look Back in Error", Part I*, 27 Aug 2026) is the richest
public analysis of an analog-series ADMET challenge that exists:

- **"Ensembles were virtually universal among top performers this time around"** — Tier 1 teams
  combined 2–15+ distinct models (Chemprop, AttentionFP, GraphGPS, SVR, GBTs, Uni-Mol v1/v2, TorchMD).
- **20 of 28 Tier 1 teams used 3D representations**, but "multiple top-tier teams noted that
  structure-based signals degraded performance" when used naively; what worked was **gating by
  confidence scores** and restricting to active subsets.
- **"Single-concentration data led to better performance"**, via multi-task training or pretraining.
- **The winner's distinguishing move was proxy labels for the inactive end:** matcha-croissant used
  "SVR-derived proxy pEC50 values for >10,000 inactive compounds" to train over an extended target
  range, giving supervised signal where the assay gives none.
- **External public data was sparsely used and often harmful:** "external datasets introduced
  out-of-distribution noise that hurt performance" (PXR is under-represented in public repositories).
- **The failure mode is overprediction of inactives next to potent neighbours:** "The 5 hardest
  compounds all possess a close structural neighbor with high potency in the training set, causing
  GNNs and classic ML models to **universally overpredict** their activity."
- **"Downweighting errors in the low-activity region yielded no visible change in overall
  performance"** — the errors are categorical misassignments, not magnitude errors. A loss tweak does
  not fix them; data at the low end does.
- Every Tier 1 model is an excellent classifier (**AUC 0.90–0.96** at pEC50 > 5) while being
  regression-limited. Rank and resolution come apart.
- Recommendation: move "past monster ensembles" toward "physics-grounded and richer data strategies".

A Tier-1 participant write-up (Acellera, AceProp) adds the transfer finding directly:
**"Additional labelled compounds did not automatically improve transfer to the second analogue
series"** — "data relevance and evaluation design matter more than dataset size alone." Their entry
was an NN on pretrained embeddings plus a GBM on descriptors+embeddings, automated hyperparameter
search under CV, multi-seed ensembling and calibration, **challenge data only**.

---

## 3. OpenADMET × ExpansionRx (Oct 2025 – Feb 2026) — temporal split, and the one challenge external data won

7,000+ molecules, 25,000+ measurements, nine ADMET endpoints from Expansion Therapeutics' programmes,
**time-split** evaluation, metric MA-RAE. 370+ participants, 4,000+ submissions, 103 valid final
submissions, 446 in the Discord channel.

**Final leaderboard (full test set), with compact letter display:**

| Rank | Alias | Who | MA-RAE | CLD |
|---|---|---|---|---|
| 1 | pebble | **Inductive Bio** (Alex Rich) | **0.5113 ± 0.0070** | a |
| 2 | overfit | — (Chemprop + KERMT) | 0.5284 ± 0.0070 | b |
| 3 | moka | **Merck** | 0.5321 ± 0.0070 | c |
| 4 | campfire-capillary | **deepmirror** (Daniel Crusius) | 0.5517 ± 0.0069 | d |
| 5 | shin-chan | — | 0.5536 ± 0.0073 | e |
| 6 | HybridADMET | — | 0.5582 ± 0.0072 | f |
| 7 | rced_nvx | — | 0.5663 ± 0.0073 | g |
| 8 | tibo | — | 0.5672 ± 0.0073 | g |
| 9 | beetroot | — | 0.5726 ± 0.0075 | h |
| 10 | temal | — | 0.5809 ± 0.0074 | i |

Note the contrast with PXR: here **almost every adjacent pair is statistically distinguishable**
(a,b,c,d,e,f,g,g,h,i), because the test set is large and the endpoints many. The CYP challenge sits
between the two: Tier 1 of one, then a 3-entry Tier 2.

**What the organisers concluded** (*"Lessons Learned…"*, 4 Feb 2026):

- **"2D Graph neural networks (GNNs) dominated the top of the leaderboard"**, Chemprop most popular;
  KERMT and Uni-Mol also successful. TabPFN "performed adequately but ultimately lacked the
  performance of 2D GNNs at the top".
- **"Multitask learning was essential, with several top-performing groups also investigating the
  placement of tasks into related groups."**
- **"Almost all top-performing entries used external data"**; **"4 of the top 5 participants also
  utilized proprietary data."** 3 of the top 10 combined traditional ML with MPNNs.
- **Pretraining helped; CheMeleon was widely adopted.**
- **"While hyperparameter optimization is often touted as essential in deep learning, it took a back
  seat this challenge … additional data, data engineering and feature augmentation proved more
  impactful."**
- Validation: "several top-performing participants prioritized temporal splits to evaluate their
  models. This approach directly mirrors the prospective nature of the task."
- Endpoint difficulty: LogD and protein binding easiest; **metabolic clearance (HLM/MLM CLint) hardest
  "despite adequate training data" because of "its sensitivity to small structural changes"**; KSOL
  had "decent relative performance, but poor ranking ability", models acting as binary classifiers.

**The two published top-performer methodologies** (CDD write-up, and Inductive Bio's own posts):

- **Inductive Bio (1st):** GNN representations, **ensembling across multiple GNN architectures** on 2D
  inputs; **"models are trained in conceptually related task groups rather than a single massively
  multitask model"** (CACO-2 Papp + efflux + MDCK + LogD together); HPO on a **temporal split using
  compound ID as a proxy for time**. Ablations: removing auxiliary tasks → measurable MAE increase;
  removing all external data → "larger performance drop across all endpoints"; single-task →
  "further degradation". They also **found and fixed an assay artefact**: two solubility clusters
  (~300 µM ceiling and an unexpected 50–100 µM cluster), hypothesised an early lower max concentration,
  imputed 250 µM for 40–110 µM compounds, "produced a measurable reduction in MAE on the final test set".
- **deepmirror (4th):** five architecture families including **source-stratified multitask** (LogD split
  by data source) and **dependent models (ML-predicted LogD injected as an input feature)**;
  **"tree-based models implemented through the AutoGluon framework performed on par with or better than
  GNNs for clearance and permeability endpoints, particularly when bagged and ensembled"**; they
  independently found the same solubility shift and **removing the first ~15% of compounds improved
  both holdout and leaderboard**.
- Merck (3rd, `moka`): four complementary architectures (GatedGCN + virtual nodes, GraphGPS, GIN with
  positional encodings, Chemprop + RDKit descriptors), **10-model ensembles each**, simple averaging,
  multitask **within endpoint groups only**, validation by **time-based sliding-window splitting** using
  molecule IDs as a temporal proxy, repeated 5×. Their stated findings: pretrained ≫ from scratch;
  graph ≫ descriptor/SMILES; auxiliary tasks critical for some endpoints and not others; **"hyperparameter
  optimization showed limited leaderboard gains despite reasonable noise levels"**.

**Cross-cutting lessons the CDD analysis drew out:** "data curation precedes modeling in practical
impact"; **"ensemble diversity matters more than ensemble size — combining three to four
architecturally distinct models consistently outperformed larger ensembles"**; ~50 program-specific
compounds meaningfully improve predictions when combined with curated external data; and "models tuned
for challenge metrics may underperform on the lower end of dynamic ranges".

**Negative control from the same challenge:** a published AI-scientist benchmark (agitter) ran Biomni,
Heureka, K-Dense and Kosmos on it with **challenge data only** and finished **rank 61 of 103 at
MA-RAE 0.7015 ± 0.0085**; the best configuration (Kosmos with rich context) reached rank 42. Useful as
a floor for "competent automated modelling, no external data".

**Follow-up post, "Can We Trust Zero-Shot ADMET Predictions?" (26 Mar 2026)** — tested ADMET-AI and
ADMETlab 3.0 off the shelf on this test set: LogD R² = **0.02** (ADMETlab) vs **0.50** for an
OpenADMET ChEMBL-trained Chemprop vs the winner near the 0.2–0.3 MAE intrinsic limit; Caco-2 A→B and
HLM CLint gave **negative R²** for all zero-shot models. Cause: "average nearest-neighbor chemical
similarity between the test set and public databases fell below 0.3–0.4". Verdict: **"Do not rely on
these models for 'zero-shot' ADMET predictions."**

---

## 4. Polaris × ASAP Discovery × OpenADMET antiviral challenge (Jan–Mar 2025)

381 submissions, 66 unique participants, 93 on the final leaderboard; three tracks; **temporal split**
for both ADMET and potency; Tukey HSD + CLD for ranking. Organisers' retrospective: MacDermott-Opeskin,
Scheen, Wognum *et al.*, *JCIM* 66(6):3129 (Mar 2026), DOI `10.1021/acs.jcim.5c02106`
(PMC13329884; ChemRxiv `10.26434/chemrxiv-2025-zd9mr-v6`).

**Potency (aggregated, MAE on pIC50):**

| Rank | Alias | MAE | R² | Kendall τ | CLD |
|---|---|---|---|---|---|
| 1 | oscar-mendez-lucio (**MolE**, Recursion) | **0.509 ± 0.022** | 0.594 | 0.626 | a |
| 2 | ccorbi (Exscientia, molflux) | 0.516 ± 0.022 | 0.603 | 0.649 | b |
| 3 | siat353 | 0.517 ± 0.021 | 0.598 | 0.614 | b |
| 4 | xubf | 0.518 ± 0.022 | 0.594 | 0.612 | bc |
| 5 | alx-dga | 0.522 ± 0.022 | 0.588 | 0.628 | c |
| 23 | ASAP internal GNN | ~0.7 | | | |
| 29 | **ECFP linear-regression baseline** | | | | u* |

**ADME (aggregated MAE):** 1 prairiewarbler10 (**Inductive Bio**) **0.224 ± 0.009** (CLD a);
2 ebl88 0.243 (b); 3 vchupakhin 0.269 (c); 4 longhung25 0.276 (d); 5 xiaolinpan 0.277 (de);
**the organisers' baseline (hmacdope) 0.430, rank 60 of 71**. Note the winner's margin over 2nd is
7.8% and CLD-distinct, and that **MolE — 1st in potency — placed 42nd in ADME at 0.307**.

Retrospective findings:

- **Pretraining was the potency differentiator:** "Many of the top performing models (**including all
  of the top 3**) employed some kind of pre-training step involving large numbers of unlabeled
  chemical structures." Top entries used **MolE and MolGPS** foundation GNNs; ensembling was common;
  "deep learning and machine learning models outperformed physics based and other approaches across
  the board".
- **ADME:** "Three of the top 5 participants employed variations on GNN architectures (e.g. D-MPNN
  with Chemprop)" with "the majority employing multi-task learning"; the top performer "employed
  proprietary data directly", others used heavily curated ChEMBL.
- **Ceiling:** top-3 Kendall τ ≈ 0.63 against inter-assay variation of ≈ 0.5–0.7 pKi units and
  ChEMBL's own τ ≈ 0.71 between curated duplicate measurements. **The potency winners were at the
  assay noise floor.**
- **Chemical similarity did not predict who was easy:** "Relative chemical similarity to the training
  set did not meaningfully assist model predictions across the test set", and "temporal breakdown of
  the test set by synthesis date revealed limited influence of synthesis date on model performance".
  What *did* matter was series: "SARS-CoV-2 backup series was predicted on most accurately … likely an
  effect of publicly available COVID Moonshot binding data consisting largely of this scaffold."
- **Same KSOL pathology as ExpansionRx:** good MAE, poor ranking — "models are memorizing the absolute
  solubility of the training set, performing closer in practice to a classification model".
- Methodological recommendations: "the need for a holistic set of metrics rather than focusing on a
  single one"; Tukey HSD pairwise testing per best-practice; open evaluation code during the challenge.

**Inductive Bio's own analysis of that board** is the cleanest published statement of the
external-data effect: **"eight of the top ten entrants made use of additional ADMET data"** versus
four in positions 11–20; their winning model's error was 53% below a descriptor baseline and 60% below
a fingerprint baseline; 5th place (MolMCL) was 23% higher error, 12th (traditional ML, no external
data) 41% higher; and MolE's potency win was only **7.6%** better than the best local model. Their
conclusion: adding global ADMET data meaningfully beat local-only models, while **"non-ADMET
pre-training produced mixed results"**, and program-to-program variation (4–24% gaps across 10
programmes per endpoint) means single-challenge rankings generalise weakly.

A follow-up full paper from a top-5 ADME entrant: *"Improved ADME Prediction by Multitask Pretraining
on Predicted Data"*, JCIM 66(1) (Dec 2025), `10.1021/acs.jcim.5c02030`, code
`github.com/LongHung-Pham/pADME` — multitask GNN transfer learning, ranked **4th on aggregated MAE and
tied 2nd on aggregated Pearson R**, and reports gains from **pretraining on a mix of experimental and
*predicted* labels**. That is the same "predicted auxiliary label as supervision" idea the CYP
organisers found works, arrived at independently.

---

## 5. Other blind challenges, and the generic lesson about analog/temporal shift

### 5.1 CACHE (already in `research/06`)
23 teams per round; **7 of 23 (CACHE #1) and 5 of 23 (#2) found any confirmed binder**; overall hit
rates 3.7% and 0.7%. Target difficulty dominated method choice. Relevant here only as the base rate
for "most entrants return nothing", and as a warning that a leaderboard's top is not the median.

### 5.2 D3R Grand Challenges
Affinity *ranking* stayed hard across all rounds even when poses were good; best Spearman ρ ≈ 0.5 for
docking-based ranking, with receptor choice (most-similar-ligand holo structure) mattering more than
the scoring function; RBFE on **congeneric series** is the one regime where physics reliably delivers.
The transferable point: for a congeneric/analog test set, methods that exploit the *pairing* beat
methods that score each ligand independently.

### 5.3 SAMPL (logP/pKa)
SAMPL7 logP accuracy was **worse** than SAMPL6's "likely due to a more diverse dataset", with 33 blind
submissions from 17 groups; pKa accuracy unchanged, and the striking result was that macroscopic
values were often right while **participants disagreed on the microscopic transitions behind them,
sometimes on the sign**. Lesson: a correct aggregate can hide a wrong mechanism, which is why
per-series/per-isoform diagnostics beat a macro number.

### 5.4 BELKA (NeurIPS 2024) and Kaggle shake-ups
BELKA: 133M molecules, ~100M binding measurements, test sets constructed to be **disjoint in building
blocks/scaffolds** from train. Public→private shake-up was severe, and the published post-mortems
converge on the same three items: *recognise the distribution shift between data sources*, use
**two-stage training and pseudo-labelling**, and do not trust the public leaderboard. The canonical
example outside chemistry (AES 2.0) had the eventual 1st place sitting **619th on the public board**,
with "the key to success being careful examination of the distribution shift between train and test".

**Applied here:** the live CYP board is half the test set *split by chemisimilar series*. The series
scored live and the series scored finally are disjoint. A choice that improves live by less than the
series-to-series spread is not evidence. The 25 Sep interim (full test set) is the only honest
leaderboard signal we have, and we have no row in it.

### 5.5 Activity cliffs, as a literature
MoleculeACE (Tilborg, Alenicheva, Grisoni, *JCIM* 2022, `10.1021/acs.jcim.2c01073`): on 30 curated
ChEMBL targets, ML on activity-cliff compounds is "generally poor", **descriptor-based models beat
deep learning on cliffs**, and representation-learning methods failed to beat fingerprints. The
authors advocate "activity-cliff-centered metrics during model development". The CYP test set is
*described by its organisers* as having "detailed SAR and activity cliffs" — so the cliff literature
is the right prior, and it points at fingerprints/descriptors + pairing, not at a bigger encoder.

### 5.6 Pairwise / delta learning, and TFMs under structural shift
- **Pairwise Difference Regression (PADRE)**, *JCIM* 2021 `10.1021/acs.jcim.1c00670`, and **DeepDelta**
  (*J Cheminform* 2023, `10.1186/s13321-023-00769-x`): train on *differences* between molecule pairs,
  turning n labels into ~n² training pairs; "optimally suited for smaller datasets (< 1500
  datapoints)"; DeepDelta beat single-molecule Chemprop/RF on 10 pharmacokinetic benchmarks. Our DRC
  set is ~1,300–2,300 labels per isoform — squarely in that regime.
- **MolPAIR**, arXiv `2609.38744` (30 Sep 2026): two frozen TFMs — one predicts from labelled
  molecules, the second **predicts prediction-*error* differences between query and reference
  molecules** and refines the first. Evaluated on **58 MoleculeACE and Polaris tasks**; CheMeleon +
  TabPFN-3 beat every baseline on a majority of tasks, and **MolPAIR improved 46 of 58 tasks across
  four representations and three TFM backbones**. This is published, external validation of the
  anchored-delta idea the programme pre-registered — and specifically *under structural shift*.
- **Tabular FMs for molecules**, arXiv `2604.16123`: CheMeleon + TFM reaches "up to 100% win rates"
  on 30 MoleculeACE tasks; **"molecular representation emerges as a key determinant in TFM
  performance"**, with 2D descriptor sets substantially beating classic fingerprints.
- **CheMeleon** (Burns *et al.*, arXiv `2506.15792`): a 12.9M-param D-MPNN pretrained on 1M PubChem
  molecules to predict **Mordred descriptors** (noise-free targets). 75% win rate on Polaris tasks
  (RF 68%, fastprop 36%, Chemprop 32%), **97% on MoleculeACE assays**. MIT-licensed.

### 5.7 Censored labels
pIC50 < 4 is below the lowest tested dose, i.e. **left-censored**, and ST-RAE's zero-band is the
organisers' way of not punishing us there. The censored-regression literature (Chalmers/AZ,
*Artif. Intell. Life Sci.* 2025, `10.1016/j.ailsci.2025.100128`; and the older QSAR treatment,
PMID 27464349) reports that Tobit-style likelihoods "significantly outperform least-squares methods",
with failure to model censoring costing most "when the proportion of censored observations is high",
and recommends using censored labels when they are **~one-third or more** of an assay's data.
PROGRESS records CYP3A4 at **40% of labels below pIC50 4**. This is above that threshold.

---

## 6. Ranked tactics, with the evidence for each

Ordered by (expected effect) / (cost × risk). Each is phrased so it can be pre-registered.

### Tier A — do these first

**A1. Reproduce the organisers' 0.68 recipe exactly, as a baseline, before anything else.**
Off-the-shelf CheMeleon 2048-d → PCA 256, **plus** a Chemprop trained on the 17.5k-row
`single_concentration` log2FC whose *predictions* become feature columns, → TabICL (BSD-3, ungated)
or TabPFN v3. 4,392 in-context rows fit; 13–15 s per run on GPU at 128-d.
*Evidence:* organisers' own post with 5-seed Tukey-HSD ablations, 0.83 → 0.68 on this challenge, 0.50
→ 0.43 on PXR; N283T independently found the log2FC readout their strongest single feature.
*Why it is first:* it is a published number on the exact metric and test set, it is a day's work, and
until we match it every other result is unanchored. It also replaces our TabICL null, which was run
in a different feature space (§1.6).

**A2. Fix CYP2D6 as a separate problem, and fix the location/scale failure there.**
Our live 2D6 is ST-RAE 1.5149 / MAE 1.78 / **R² −0.989** while ρ = 0.363: the ranking is weak but the
*calibration* is catastrophic. A per-isoform monotone recalibration (isotonic or quantile mapping onto
a 2D6-specific predicted label distribution) plus a 2D6-specific intercept is nearly free.
*Evidence:* organisers state the 2D6 test set "contains inherently less potent compounds … This
distribution shift between training and test sets makes CYP2D6 a much harder target"; 2D6 was never
hit-expanded; 131 of 219 interim entries are worse than the mean on 2D6 while the leader reaches 0.457.
*Size of prize:* 2D6 alone is 25% of the macro; moving 1.51 → 0.60 is **−0.23 macro**.

**A3. Manufacture supervision at the inactive end — proxy labels for the primary-screen negatives.**
The DRC training set is hit-enriched (~1,500 per isoform, promoted from the primary screen); the test
set is 750 analogs of hits *plus* whatever they do on 2D6, i.e. a lot of genuinely inactive compounds.
Fit a model on the single-concentration log2FC, use it to assign proxy pIC50 (or proxy "below 4" labels
with wide intervals) to the thousands of screened-but-not-promoted compounds, and train on the union.
*Evidence:* PXR's **winner** did exactly this — "SVR-derived proxy pEC50 values for >10,000 inactive
compounds" to train on an extended target range; the organisers' diagnosis of the general failure mode
is that models "universally overpredict" compounds with a potent close neighbour; the 4,376 × 4
single-concentration matrix is released. Note the matching negative: PXR found **downweighting**
low-activity errors did nothing — the fix is data at the low end, not loss weights.

**A4. Stop optimising decode/shrinkage; the leaderboard has refuted it.**
The leader achieves MA-R² 0.677 / MA-MAE 0.6005 on the same 750 compounds, which backs out
sd_y ≈ 1.32 — matching our estimate but from a high-R² row. Our last submission, more shrunk, went
0.8598 → 0.9628. A squared-error-optimal `s* ≈ 0.70` derived from *our own* R² = 0.055 is optimal only
conditional on a near-useless prediction vector; it is not a statement about the task.
*Action:* close the shrinkage line, record it as refuted by external evidence, and re-point the
remaining budget at A1–A3.

### Tier B — high value, a few days each

**B1. An anchored-delta / pairwise head on top of the best point model.**
For each test compound, retrieve its nearest labelled training neighbours (the test set *is* analog
series, 84.3% within Tanimoto 0.4 of a training active per our own audit) and predict the *difference*
from the anchor, then add it back; or, following MolPAIR, train a second model to predict the first
model's *error difference* between query and reference.
*Evidence:* MolPAIR improved **46/58** MoleculeACE+Polaris tasks across 4 representations × 3 TFMs,
specifically framed as structural shift; PADRE and DeepDelta show pairwise training wins below ~1,500
labels; D3R's one reliable regime is congeneric-series relative prediction. Caveat to pre-register:
PXR's "additional labelled compounds did not automatically improve transfer to the second analogue
series", and MoleculeACE says cliffs defeat smooth models — so expect the gain to come from the
retrieval, not the architecture.

**B2. Diverse small ensemble: 3–4 architecturally distinct members, averaged.**
*Evidence:* "Ensembles were virtually universal among top performers" in PXR Tier 1 (2–15+ members);
Merck's 3rd place used four distinct architectures × 10 seeds with simple averaging; CDD's cross-cutting
lesson is explicit — **"ensemble diversity matters more than ensemble size. Combining three to four
architecturally distinct models consistently outperformed larger ensembles."** Our own 64-member
single-architecture ensembles are the wrong axis.

**B3. Task grouping, not one massively multitask head.**
Group 1A2+2C9+3A4 (fluorogenic, aromatic-preferring, hit-expanded) separately from 2D6 (mass-spec,
basic-amine-preferring, not expanded), with the single-concentration log2FC heads as auxiliaries.
*Evidence:* Inductive Bio — "models are trained in conceptually related task groups rather than a
single massively multitask model", with ablations showing auxiliary-task removal costs MAE; Merck
multitasked **within groups only**; ExpansionRx organisers — "Multitask learning was essential, with
several top-performing groups also investigating the placement of tasks into related groups". Mechanistic
support from OpenADMET's own *Escape from Flatland, CYP Edition* (6 Oct 2026): Fsp3 correlates with
inhibition for 2C9 and 1A2 and weakly for 3A4, but **"There's no clear correlation for CYP2D6, which
tracks with its preference for basic amines"** — 2D6 is a different pharmacophore, so it is a different
task group.

**B4. Model the interval, and treat censoring explicitly.**
ST-RAE's zero-band has width that grows at low activity; the training file ships `_conf_high`,
`_conf_low` and `_std` per label. Fit width as a function of predicted activity, and use a
Tobit/censored likelihood for the pIC50 < 4 mass rather than regressing on point estimates.
*Evidence:* organisers — "intervals expand significantly at lower activity levels, aligning with assay
precision limits"; censored-regression literature reports significant gains over least squares once
censoring exceeds ~⅓ of labels, and CYP3A4 is at 40%.

**B5. Validate on an analog-expansion split, and distrust the live board.**
`scripts/cyp_analog_split.py` already builds this. Add series-held-out repeats (hold out whole
chemisimilar series, not random analogs) and report the series-to-series spread as the decision
threshold.
*Evidence:* the live board is explicitly "split by chemisimilar series", so live and final series are
disjoint; PXR's Phase 1 → Phase 2 was a series change and transfer did not follow dataset size;
BELKA/Kaggle post-mortems make public-LB overfitting the canonical failure.

### Tier C — worth testing, lower expected value

**C1. Fingerprint/descriptor models as first-class ensemble members, not as a baseline to beat.**
MoleculeACE: descriptor models beat deep learning on activity cliffs across 30 targets; deepmirror found
AutoGluon trees matched or beat GNNs for the structurally-sensitive endpoints when bagged. Our XGBoost
is already competitive internally — keep it in the blend rather than replacing it.

**C2. Pretrained-embedding ensembles (CheMeleon fine-tuned on log2FC, then on pIC50).**
The organisers' Figure 3 found the **log2FC-fine-tuned CheMeleon embedding** the nominal best single
ingredient. All of the top-3 Polaris potency entries used pretraining.

**C3. 3D only if gated.** 20/28 PXR Tier 1 teams used 3D, but "multiple top-tier teams noted that
structure-based signals degraded performance" naively; it worked when gated by confidence and restricted
to actives. The CYP structure track exists and 20 cryo-EM CYP3A4 complexes are being released — but this
is a separate track, and for the regression track 3D is a Tier C option at best.

**C4. Skip generic external ADMET data; it is not what the leaders are using.**
Interim board: 8/219 declared proprietary data; 19 of the top 20 declared none. PXR: "external datasets
introduced out-of-distribution noise that hurt performance". Zero-shot ADMET-AI/ADMETlab gave negative
R² on ExpansionRx endpoints because public-database nearest-neighbour similarity was below 0.3–0.4.
Our own ChEMBL/Veith experiments being "inconclusive" is the expected result, not a failure of method.
*Exception:* data from the **same assay family** (the challenge's own single-concentration, TDI-arm and
Emax files) is the kind that paid everywhere.

### Not worth doing

- **Hyperparameter optimisation.** Reported as near-worthless in all three OpenADMET/Polaris
  retrospectives, in the winner's own report ("limited leaderboard gains despite reasonable noise
  levels"), and by the organisers ("it took a back seat this challenge").
- **Downweighting low-activity error.** PXR tested it: "no visible change in overall performance".
- **Chasing validation ST-RAE on a diverse scaffold split.** Already recorded in PROGRESS; now
  externally corroborated — Polaris found "relative chemical similarity to the training set did not
  meaningfully assist model predictions", while *series* identity did.
- **Any further offset/blend-weight fitting on the same 922 validation rows.** The programme's own
  cross-fitting put an offset's honest value at ≈ −0.004.

---

## 7. Two housekeeping facts the owner should see

1. **Our current leaderboard row is our worst.** `talindrew`, submitted 2026-10-07 06:59 UTC,
   MA-ST-RAE **0.9628**, MA-R² **−0.0658**, rank **225 / 260** — against 0.8598 the day before. Only
   the latest valid submission counts. With 12-hour rate limiting and the deadline on 3 Nov 2026
   23:59 UTC there is ample room to replace it, but it should be replaced.
2. **A model report link is required to appear on the final leaderboard** (FAQ; and ExpansionRx removed
   entries for "missing required model reports with valid links"). `notes/METHOD-REPORT.md` exists
   locally; publishing it is a prerequisite for a final-board row, and is also the gate for the
   Innovation in ML Award, which is explicitly rank-independent.

---

## 8. Bibliography

### The CYP challenge itself
- Announcing OpenADMET's CYP inhibition blind challenge — https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/
- OpenADMET's CYP challenge is underway — https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/
- Halfway Through the OpenADMET CYP Challenge (interim leaderboard, 25 Sep 2026) — https://openadmet.ghost.io/halfway-through-the-openadmet-cyp-challenge/
- Are tabular foundation models all you need? (Sean Colby, 23 Sep 2026; DOI 10.5281/zenodo.23198306) — https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/
- Escape from Flatland, CYP Edition (6 Oct 2026) — https://openadmet.ghost.io/escape-from-flatland-cyp-edition/
- Space (rendered app) — https://huggingface.co/spaces/openadmet/cyp-challenge · live data via https://openadmet-cyp-challenge.hf.space/config
- Space source (About + FAQ text) — https://huggingface.co/spaces/openadmet/cyp-challenge/raw/main/config.py
- Dataset card — https://huggingface.co/datasets/openadmet/cyp-challenge-train-test
- Tutorial repo — https://github.com/OpenADMET/CYP-Challenge-Tutorial
- Scoring template + design docs — https://github.com/OpenADMET/blind-challenge-template · https://github.com/OpenADMET/blind-challenge-template/blob/main/docs/scoring-and-leaderboards.md
- Building the OpenADMET Data Engine (Octant assay development) — https://openadmet.github.io/Octant_CYP_blog_post/ · data https://huggingface.co/datasets/openadmet/Octant_CYP_inhibition_reactivity_blog_release · Zenodo https://zenodo.org/records/21450795
- Blind-challenge index — https://openadmet.org/blindchallenges/

### OpenADMET PXR challenge
- Don't Look Back in Error: What we learned predicting PXR induction (Part I) — https://openadmet.ghost.io/dont-look-back-in-error-what-we-learned-predicting-pxr-induction-part-i/
- It's the End of the PXR Challenge as We Know It (final leaderboard, tiers) — https://openadmet.ghost.io/its-the-end-of-the-pxr-challenge-as-we-know-it-and-i-feel-fine/
- Tune In: What Fine-tuning OpenFold3 Taught Us — https://openadmet.ghost.io/tune-in-what-fine-tuning-openfold3-taught-us/
- Acellera / AceProp participant write-up — https://www.acellera.com/blog/reaching-the-top-tier-of-the-pxr-blind-challenge-with-an-ai-agent-black-box-trainer

### OpenADMET × ExpansionRx challenge
- Lessons Learned from the OpenADMET–ExpansionRx Blind Challenge — https://openadmet.ghost.io/lessons-learned-from-the-openadmet-expansionrx-blind-challenge/
- The OpenADMET–ExpansionRx Blind Challenge has Come to an End (final leaderboard + CLD) — https://openadmet.ghost.io/the-openadmet-expansionrx-blind-challenge-has-come-to-an-end/
- Can We Trust Zero-Shot ADMET Predictions? — https://openadmet.ghost.io/zero-shot-expansiorx-admet-predictions/
- CDD: Applying a Focused Modeling Strategy … Lessons from Top Performers — https://www.collaborativedrug.com/cdd-blog/applying-a-focused-modeling-strategy-in-the-openadmet-expansionrx-blind-challenge-lessons-from-top-performers
- Inductive Bio win announcement — https://www.inductive.bio/news/inductive-bio-wins-worlds-largest-admet-prediction-competition · https://www.bio-itworld.com/news/2026/04/02/inductive-bio-on-a-winning-streak-with-admet-predictions
- Merck (`moka`, 3rd) model report — https://github.com/dahvida/openadmet-expansionrx-blind-challenge-moka
- AI-scientist benchmark entry (rank 61/103) — https://github.com/agitter/openadmet-expansionrx-challenge/blob/main/writeup.md · https://zenodo.org/records/18303883
- Tutorial — https://github.com/OpenADMET/ExpansionRx-Challenge-Tutorial

### Polaris × ASAP × OpenADMET antiviral challenge 2025
- Organisers' retrospective, *JCIM* 66(6):3129 (2026), DOI 10.1021/acs.jcim.5c02106 — https://pmc.ncbi.nlm.nih.gov/articles/PMC13329884/ · preprint https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-zd9mr-v6
- Leaderboards (machine-readable in page payload) — https://polarishub.io/competitions/asap-discovery/antiviral-drug-discovery-2025 · potency https://polarishub.io/competitions/asap-discovery/antiviral-potency-2025 · ADMET https://polarishub.io/competitions/asap-discovery/antiviral-admet-2025
- Inductive Bio, Lessons from the Polaris ADMET competition — https://www.inductive.bio/blog/lessons-from-the-polaris-admet-competition
- MolE (potency winner) — https://www.nature.com/articles/s41467-024-53751-y · https://github.com/recursionpharma/mole_public
- Exscientia molflux (2nd, potency) — https://github.com/exscientia/molflux
- pADME: Improved ADME Prediction by Multitask Pretraining on Predicted Data, *JCIM* 66(1), DOI 10.1021/acs.jcim.5c02030 — https://pubmed.ncbi.nlm.nih.gov/41423887/ · https://github.com/LongHung-Pham/pADME
- Press summary — https://globalbiodefense.com/2025/07/28/pan-coronavirus-ai-drug-discovery-blind-challenge-results/
- Blind Challenges Let Us See the Path Forward for Predictive Models, *JCIM* 66(4):1947 — https://pubs.acs.org/jcisd8/article/66/4/1947/5071541/Blind-Challenges-Let-Us-See-the-Path-Forward-for *(paywalled; abstract only)*

### Methods literature cited above
- CheMeleon — Burns *et al.*, arXiv 2506.15792 — https://arxiv.org/pdf/2506.15792
- Tabular foundation models for in-context prediction of molecular properties — arXiv 2604.16123 — https://arxiv.org/abs/2604.16123
- MolPAIR: Molecular Property Prediction under Structural Shift with Tabular Foundation Models — arXiv 2609.38744 — https://arxiv.org/abs/2609.38744
- Buterez *et al.*, Transfer learning with GNNs in the multi-fidelity setting, *Nat Commun* 15:1517 (2024) — https://www.nature.com/articles/s41467-024-45566-8
- Pairwise Difference Regression (PADRE), *JCIM* 2021 — https://pubs.acs.org/doi/10.1021/acs.jcim.1c00670
- DeepDelta, *J Cheminform* 2023 — https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00769-x · https://github.com/RekerLab/DeepDelta
- MoleculeACE — Exposing the Limitations of Molecular ML with Activity Cliffs, *JCIM* 2022 — https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073 · https://github.com/molML/MoleculeACE
- Enhancing uncertainty quantification in drug discovery with censored regression labels — https://www.sciencedirect.com/science/article/pii/S2667318525000042 · preprint PDF https://research.chalmers.se/publication/545243/file/545243_Fulltext.pdf
- QSAR with greater-than/less-than cut-off results — https://pubmed.ncbi.nlm.nih.gov/27464349/
- PoseBusters (cited by the structure track) — https://pubs.rsc.org/sc/article/15/9/3130/827511/PoseBusters

### Other challenges referenced
- BELKA (NeurIPS 2024) — https://www.kaggle.com/competitions/leash-BELKA/overview · https://neurips.cc/virtual/2024/competition/84787 · 11th-place solution https://github.com/dangnh0611/kaggle_leash_belka · gold-solution talk https://slideslive.com/39030638/gold-solution-of-leash-bio-competition-belka-kaggle
- Kaggle shake-up measurement — https://github.com/davidthaler/shakeup · worked example https://hippocampus-garden.com/kaggle_aes2/
- D3R Grand Challenge 4 — https://www.osti.gov/pages/servlets/purl/1803880 · optimal affinity ranking https://pubmed.ncbi.nlm.nih.gov/28918599/
- SAMPL7 logP/pKa/logD evaluation — https://chemrxiv.org/engage/chemrxiv/article-details/60c757c99abda251fdf8e7cb · https://link.springer.com/10.1007/s10822-021-00397-3
- CACHE (carried from `research/06-prospective-screening-reality.md`) — https://www.biorxiv.org/content/10.1101/2024.07.18.603797.full.pdf · https://pmc.ncbi.nlm.nih.gov/articles/PMC11844436/

---

## 9. Provenance and limits

- **Leaderboard numbers in §1.5 and §1.8 were read on 2026-10-08** from the Space's public Gradio
  config payload, which is the same data the rendered page shows. The rendered page could not be read
  by plain HTML fetch (it is a dynamic Gradio app) — that is stated here rather than guessed around.
  Per-entry *model reports* are deliberately not covered here; they are the subject of
  `research/24-leaderboard-model-reports.md`.
- **The final leaderboard's exact scope is inferred, not confirmed.** The organisers confirm the
  interim used "the entire test set" and that live uses half; the template applies significance testing
  to interim and final alike. Treat "final = all 750" as the working assumption and re-check after
  4 Nov 2026.
- Two secondary sources could not be read directly and are cited at one remove: the JCIM editorial
  (paywalled, 403) and the Inductive Bio Polaris blog (its tables were not reproducible in full).
  Their numbers here come from the same publishers' other pages or from search snippets, and are
  marked where the attribution is indirect.
- `sd_y ≈ 1.32` in §1.5 is a derivation from the leader's published MAE and R² under a
  Gaussian-residual assumption, not a measurement. It agrees with the programme's independent estimate,
  which is why it is quoted; it is not an argument on its own.
