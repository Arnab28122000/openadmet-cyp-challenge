# 26 — What is actually state of the art for molecular property regression at n ≈ 1k–5k (October 2026)

Scope: the question asked of this screen was *"what genuinely wins on 1,000–5,000-molecule property
regression as of 2026, with measured effect sizes from head-to-head comparisons."* Everything below
is sourced; every number is marked **[verified]** (read in the source), **[reported]** (quoted from an
abstract/blog I read but could not open the table for), or **[assumed]** (my inference).

This file does not re-derive what `research/21`, `research/20`, `research/23` and `research/24`
already pin. It supersedes `21 §3` and `21 §5` on the numbers, because two 2026 papers published
after that screen (Monroe, and the organisers' own blog post) give the controlled comparison those
sections had to infer.

---

## 0. The answer first

1. **One architecture has separated from the field on exactly our task shape, and it is not a
   molecular model.** It is *frozen molecular-foundation-model embeddings read by a tabular
   in-context learner*. On MoleculeACE — 30 ChEMBL potency-regression tasks, the closest public
   analogue to our problem — a tuned GBM gets RMSE **0.680**, CheMeleon with its native fine-tuned
   head **0.662**, and **CheMeleon + a TabPFN head 0.641**; the 2026 SOTA encoder Monroe gets
   **0.629**. Same official splits, same seeds, Tukey HSD. [verified, Monroe Table 1]
2. **The head is worth more than the encoder.** Holding Monroe's encoder fixed and changing only the
   downstream adaptation: MLP decoder **0.723**, LoRA→TabPFN **0.653**, frozen→TabPFN **0.637**.
   That is **0.086 RMSE** for the head choice against **0.051** for the whole encoder progression
   from GBM to Monroe. [verified, Monroe Table 4]
3. **Our current stack is the two worst cells of the published grid.** Count-ECFP/Morgan features into
   an MLP. TabPFN-Morgan is a **22.4%** win rate against TabPFN-CheMeleon's **86.2%** on the same 58
   tasks; the MLP-on-frozen-embeddings head is the **0.723** row above. [verified]
3a. **The in-context arm we already ran (`cyp_incontext.py`, null) does not close this**, and §4a is
   the paragraph to read before anything else. It changed the *model* while keeping count-ECFP, then
   PCA'd it to 500 — the 22.4% cell, reduced in the one way the winning protocol forbids, under
   TabPFN-v2's stale 500-feature budget, without ever running TabPFN. It still **beat xgb on 3 of 4
   endpoints at half the variance.** No arm in this programme has fed CheMeleon to anything.
4. **The organisers published their 0.68 recipe with hyperparameters and we have not run it.**
   CheMeleon 2048 → PCA 256, plus two columns of a Chemprop-predicted screen log2FC, into
   `TabICLRegressor`. On the PXR challenge that stack measured **0.427 MAE / R² 0.557 / Spearman
   0.796** on a 260-compound blind set; on the CYP board it scores **0.68 MA-ST-RAE** against their
   older CheMeleon-only baseline's **0.83**. We are at **0.8663**. [verified]
5. **Chemistry foundation models that are *not* descriptor- or QC-target pretrained are dead.** On 25
   datasets × 25 models with frozen embeddings, **only CLAMP** beat count-ECFP; Uni-Mol2, GROVER,
   GEM, Mol2Vec, MoLFormer, ChemFM-3B and 15 others lost, several at posterior probability 1.00.
   [verified]
6. **LLMs are settled negatively** and `21 §4` already covers it. Nothing in this screen changes it.
7. **Censored labels: use the one-sided squared loss, not the Tobit likelihood.** On AstraZeneca's
   three CYP pIC50 assays at 58–63% left-censoring, the one-sided squared loss improved MSE by
   **~20%** and NLL by **~52%**; the full Tobit/Gaussian NLL improved NLL by 34–56% but **lost ~30%
   MSE**. [verified]
8. **Pairwise/delta learning is a 0.008-RMSE effect on top of a TFM, not a 0.05 effect.** MolPAIR's
   own measured gain over its global TFM baseline is **0.00819 RMSE [0.00235, 0.01388]**. DeepDelta's
   published advantage is statistically indistinguishable from RF/Chemprop *within the same scaffold*,
   which is our entire test set. [verified]
9. **Ensembling buys ~0.014 RMSE** and only from error diversity, not member quality. [verified]
10. **The metric will re-rank the models.** ScoringBench: across 97 regression datasets "models that
    excel on point-estimate metrics can rank poorly on probabilistic ones." ST-RAE is an interval
    score, not MAE. Select on ST-RAE. [verified]

---

## 1. The one table that governs the decision

**Monroe, Table 1** (arXiv [2608.18982](https://arxiv.org/abs/2608.18982), Banaszewski & Fitzgibbon,
Graphcore, 19 Aug 2026). MoleculeACE = 30 ChEMBL potency-regression assays, official
expert-curated splits, mean ± sd across seeds, pairwise winner = Tukey HSD + Benjamini–Hochberg.
Polaris = 28 ADMET/potency tasks. [verified, read in full]

| Model | Polaris Win % | Polaris Rank ↓ | MoleculeACE Win % | MoleculeACE RMSE ↓ | Cliff RMSE ↓ |
|---|---|---|---|---|---|
| MolFormer | 10.7 | 17.57 | 0.0 | 0.748 ± 0.018 | 0.854 ± 0.027 |
| MiniMol | 42.9 | 11.39 | 10.0 | 0.703 ± 0.018 | 0.794 ± 0.031 |
| **GBM** | 46.4 | 10.04 | 16.7 | **0.680 ± 0.020** | 0.769 ± 0.023 |
| CheMeleon (native) | 46.4 | 13.25 | 36.7 | 0.662 ± 0.016 | 0.760 ± 0.026 |
| MiniMol**PFN** | 75.0 | 5.82 | 70.0 | 0.642 ± 0.004 | 0.760 ± 0.008 |
| CheMeleon**PFN** | 42.9 | 11.46 | 80.0 | **0.641 ± 0.013** | 0.741 ± 0.020 |
| **Monroe** | **100.0** | **3.71** | **86.7** | **0.629 ± 0.004** | **0.737 ± 0.006** |

Read the deltas against the GBM, because the GBM is the honest baseline for our problem:

| Change | Δ RMSE vs GBM | Controlled? |
|---|---|---|
| GBM → CheMeleon native head | **−0.018** | yes, same splits |
| GBM → CheMeleon + PFN head | **−0.039** | yes |
| GBM → Monroe (PM6-pretrained + PFN) | **−0.051** | yes |
| MLP head → PFN head, encoder fixed | **−0.086** | yes (Table 4) |

The last row is the headline. **Changing the head is twice the effect of changing the encoder.**

Two cautions that come from the same papers and must not be dropped:

- **Neither TFMs nor CheMeleon solve activity cliffs.** TabPFN-CheMeleonFP still shows
  RMSE_cliff > RMSE_noncliff on **28 of 30** tasks (CheMeleon: 26 of 30). [verified, TFM paper Fig. S3]
  Our test set is an analog expansion, i.e. cliff-rich. The gains above are *average* gains; they do
  not come with cliff immunity.
- **Polaris and MoleculeACE disagree about CheMeleon.** CheMeleonPFN is *worse* than CheMeleon
  native on Polaris rank (11.46 vs 13.25 is better, but win% 42.9 vs 46.4 is worse) and much better
  on MoleculeACE (80.0 vs 36.7). Potency regression is the MoleculeACE shape. Ours is potency
  regression.

---

## 2. Molecular foundation models and pretrained embeddings

### 2.1 The negative result that rules out most of the list

*Benchmarking Pretrained Molecular Embedding Models for Molecular Representation Learning* —
Praski, Adamczyk & Czech, arXiv [2508.06199](https://arxiv.org/abs/2508.06199) v4.
**25 models × 25 datasets**, frozen embeddings + {RF, logistic regression, kNN} with identical
hyperparameter grids, scaffold split, AUROC, hierarchical Bayesian Bradley–Terry, ROPE 35–65%.
[verified, full text read]

| Model | Mean rank ↓ | Mean AUROC ↑ | | Model | Mean rank ↓ | Mean AUROC ↑ |
|---|---|---|---|---|---|---|
| **CLAMP** | **5.40** | **82.55%** | | TT | 12.04 | 78.12% |
| R-MAT | 6.08 | 80.83% | | ChemFM (1B/3B) | 12.60 | 78.25% |
| MolBERT | 6.92 | 80.51% | | **Uni-Mol2** | 13.32 | 78.11% |
| ChemBERTa | 7.32 | 79.99% | | MolR | 13.80 | 77.84% |
| **ECFP (count)** | **7.52** | **79.89%** | | COATI | 13.92 | 78.02% |
| CDDD | 7.64 | 80.60% | | GraphMVP | 14.64 | 77.18% |
| Atom Pair | 7.88 | 79.48% | | Chemformer | 15.00 | 77.44% |
| MAT | 8.02 | 80.18% | | Uni-Mol | 16.64 | 76.85% |
| **MoLFormer** | 9.50 | 79.80% | | **GROVER** | 17.08 | 75.68% |
| **Mol2Vec** | 10.36 | 79.55% | | GIN | 17.28 | 75.14% |
| | | | | SimSon / ChemGPT | 19.12 / 19.32 | 73.56% / 74.01% |
| | | | | **GEM** | 19.36 | 74.03% |
| | | | | SELFormer | 20.04 | 73.18% |
| | | | | GraphFP | 24.20 | 59.67% |

Bayesian verdict against the ECFP baseline: **better — CLAMP alone.** *Practically equivalent* —
R-MAT, MolBERT, ChemBERTa, CDDD, Atom Pair, MAT. **Worse — every other model.** Posterior
probability that ECFP beats the named model: MoLFormer 0.60 (p=1.00), Mol2Vec 0.64, ChemFM 0.74,
**Uni-Mol2 0.76 (p=1.00)**, MolR 0.76, COATI 0.77. On **5 of 25 datasets no model at all** beat ECFP;
on 8 more only one or two did. [verified]

That is the direct answer to "which of MolFormer / ChemBERTa-2 / GROVER / Mol2Vec / MAT / GEM /
Uni-Mol / Uni-Mol2 / SMI-TED beat plain descriptors+GBM on small ADMET sets?" — **none of them, and
ChemFM-3B loses to a hashed fingerprint.** Stop researching this family.

**What the benchmark does not settle**, and must be said: it is **classification AUROC** on 25 sets
(the five CYP panels in it are the TDC *Veith* classification sets at 12,092–13,130 compounds — not
regression); it used **frozen embeddings + shallow heads only**; and it predates CheMeleon (Jun
2025), GRASP, Monroe and every TFM head. The two things it never tested are the two things that win
in §1 and §3.

### 2.2 CheMeleon — exact numbers

arXiv [2506.15792](https://arxiv.org/abs/2506.15792) (Burns et al., MIT ChemE + BASF). [verified]

- Pretraining: **1,613 Mordred descriptors** over **1M PubChem molecules**, D-MPNN (Chemprop),
  message dimension **2048**, depth **6**, mean aggregation, **8.7M** message-passing params /
  **12.9M** total. **85% of descriptor targets randomly masked** in the loss; MSE on the unmasked.
  Pretraining test RMSE **0.14** after rescaling to zero mean / unit variance with Winsorisation at 6σ.
- Weights `doi.org/10.5281/ZENODO.15426600`; MIT code / CC0 weights; in Chemprop ≥ 2.2.0 via
  `--from-foundation CheMeleon`, frozen vectors via the `CheMeleonFingerprint` class.

| Benchmark | CheMeleon | minimol | Random Forest | MoLFormer | fastprop | Chemprop | MolCLR |
|---|---|---|---|---|---|---|---|
| Polaris (28), wins | 21 (**75%**) | 20 (71%) | 19 (68%) | 11 (39%) | 10 (36%) | 9 (32%) | 4 (14%) |
| MoleculeACE (30), wins | 29 (**97%**) | 12 (40%) | 15 (50%) | 5 (17%) | 5 (17%) | 0 (0%) | 0 (0%) |
| MoleculeACE cliff subset | 30 (**100%**) | 20 (67%) | 20 (67%) | 10 (33%) | 12 (40%) | 4 (13%) | 2 (7%) |

**The honest reading of CheMeleon's own paper:** on Polaris it beats Random Forest **21 to 19 of 28**
— a coin-flip. The separation is entirely on **MoleculeACE (29 vs 15)**, i.e. on ChEMBL potency
regression with activity-cliff splits. That is our task, which is why CheMeleon is the right
encoder — but anyone quoting "CheMeleon beats RF" from the Polaris column is quoting noise.

### 2.3 Monroe — the 2026 successor, and why it matters even if we never run it

arXiv [2608.18982](https://arxiv.org/abs/2608.18982), 19 Aug 2026, `github.com/blazejba/monroe`,
CC-BY-4.0. [verified]

- Pretrained on **81M molecules from PM6** (semi-empirical QC) + **1.56M** from PCBA, **T = 1,152
  tasks** (62 PM6 graph-level targets + 1,089 PCBA assays), uncertainty-weighted multitask
  scalarisation, conformer denoising, embedding decorrelation, stereochemistry rewiring.
- **Downstream predictor is TabPFNv3.** The paper's own framing: "replacing standard fine-tuning
  with downstream in-context prediction using a prior-data-fitted model."

The result we should actually use is **Table 2**, the within-model adaptation ablation on 28 Polaris
tasks — because it is the only published head-to-head on *how to attach a head to CheMeleon*:

| Model | frozen → PFN | fine-tuned → PFN | Native head |
|---|---|---|---|
| MiniMol | **92.9%** | 85.7% | 57.1% |
| **CheMeleon** | 75.0% | **96.4%** | 92.9% |
| MolFormer | 39.3% | 71.4% | **92.9%** |

**This contradicts `21 §5.1`'s instruction "do not fine-tune CheMeleon."** The TFM paper compared
*frozen + TabPFN* (86.2%) against *fine-tuned + native MLP head* (41.4%) — not against *fine-tuned +
TabPFN*. Monroe ran that missing cell and **fine-tuned → PFN wins for CheMeleon specifically**
(96.4% > 92.9% native > 75.0% frozen). `CheMeleonPFN` in Table 1 (RMSE 0.641) **is** the fine-tuned
variant. For Monroe's own encoder the opposite holds (Table 4: frozen 0.637 < LoRA 0.653). So:
**frozen-vs-fine-tuned is encoder-dependent and is a cheap two-arm ablation, not a settled question.**

Table 6 — performance vs labelled support-set size, CheMeleonPFN on MoleculeACE RMSE:

| support | 1% | 5% | 10% | 25% | 50% | 100% |
|---|---|---|---|---|---|---|
| CheMeleonPFN | 1.247 | 1.116 | 1.022 | 0.854 | 0.747 | **0.641** |
| Monroe | 1.173 | 1.014 | 0.914 | 0.786 | 0.707 | **0.629** |

Going 50% → 100% of labels is worth **0.106 RMSE**. Below ~100 labelled molecules per task the
encoders are indistinguishable. At our n (~4,376 across 4 isoforms, ~1,000–2,000 per endpoint) we
are in the regime where the encoder *does* separate, and where **more usable labels beat a better
model** — which is the argument for the censored screen compounds in §7.

Table 7 — ensembling: Monroe 0.629 → ENS(4) 0.620 → ENS(8) 0.621 (no gain) → ENS(10) 0.615.
**Total ensembling gain 0.014 RMSE**, and the last two members that helped were the *individually
much weaker* MiniMolPFN and CheMeleonPFN. The paper's own conclusion: "the gain is driven by error
diversity rather than member quality." [verified]

### 2.4 Candidate table, 2026 status

| Model | 2026 verdict for our problem | Evidence |
|---|---|---|
| **CheMeleon** | **Use it.** 2048-dim, MIT/CC0, 12.9M params, descriptor-target pretraining, the organisers' own baseline encoder | 97% MoleculeACE; 0.641 with PFN head |
| **Monroe** | Best published encoder; code CC-BY, but PM6-scale and untested outside the paper | 0.629 MoleculeACE, 100% Polaris |
| **MiniMol** | Competitive *with* a PFN head (0.642), weak without (0.703). Earlier leakage concerns stand | Monroe Table 1 |
| CLAMP | Only model of 25 to beat count-ECFP; but in the TFM grid TabPFN-CLAMP is only 37.9% | both papers |
| GRASP | 0.374 mean MAE on 23 OpenADMET endpoints (best of four tested) — not in any TFM head-to-head | `21 §3.2` |
| MolE | Beats TxGemma-27B on 9/9 ADMET regression; 10th at Polaris | `21 §4.1` |
| **Uni-Mol / Uni-Mol2** | **No.** Rank 16.6 / 13.3 of 25; ECFP beats Uni-Mol2 at posterior 0.76, p=1.00 | §2.1 |
| **MoLFormer / MoLFormer-XL** | **No.** Rank 9.5, loses to ECFP at 0.60/p=1.00; 0.0% MoleculeACE win | §2.1, §1 |
| **ChemBERTa-2** | Practically equivalent to ECFP. MTR pretraining > MLM, but the ceiling is ECFP | §2.1 |
| **GROVER / GEM / Mol2Vec / MAT / GraphMVP / MolCLR** | **No.** Ranks 14.6–19.4; MolCLR 1.7% win rate | §2.1, §3.1 |
| **SMI-TED** | **No.** TabPFN-SMI-TED is 5.2% win rate, rank 24.45 of 25 | §3.1 |
| ChemFM (1B/3B) | **No.** Loses to count-ECFP at posterior 0.74 | §2.1 |
| GLACIER (KDD 2026) | Student–teacher distillation of MiniMol + MolFormer over 100k molecules, Finsler-geometry fusion. No head-to-head against a PFN head found | [arXiv 2606.11382](https://arxiv.org/abs/2606.11382) [reported] |

---

## 3. Tabular in-context-learning models — the live lead

### 3.1 The molecular evidence

*Tabular foundation models for in-context prediction of molecular properties* — Ben Hicham, Rittig,
Grohe & Mitsos (RWTH Aachen), arXiv [2604.16123](https://arxiv.org/abs/2604.16123) v2, 20 Apr 2026.
Code: `git.rwth-aachen.de/avt-svt/public/tabpfn-molprop`. **Table 1, all 58 tasks**, win = best or
statistically indistinguishable from best by Tukey HSD over 5 seeds, original expert-curated splits:
[verified, full text read]

| Model | Wins /58 | Win rate | Avg rank ↓ | | Model | Wins | Win rate | Avg rank ↓ |
|---|---|---|---|---|---|---|---|---|
| **TabPFN-CheMeleonFP** | **50** | **86.2%** | **4.52** | | TabPFN-Morgan | 13 | 22.4% | 12.98 |
| TabICL-CheMeleonFP | 43 | 74.1% | 6.31 | | RF Morgan | 10 | 17.2% | 15.12 |
| TabPFN-Mordred | 39 | 67.2% | 5.71 | | fastprop | 5 | 8.6% | 15.98 |
| TabPFN-RDKit2d | 33 | 56.9% | 5.31 | | MLP PLR Pretrained | 6 | 10.3% | 16.60 |
| TabICL-RDKit2d | 32 | 55.2% | 5.90 | | RF Mordred | 6 | 10.3% | 17.50 |
| TabICL-CLAMP | 25 | 43.1% | 8.86 | | TabICL-Morgan | 8 | 13.8% | 17.64 |
| **CheMeleon (fine-tuned)** | 24 | 41.4% | 8.84 | | MoLFormer | 6 | 10.3% | 18.36 |
| TabICL-Mordred | 22 | 37.9% | 8.35 | | PCA MLP Prefitted | 5 | 8.6% | 18.74 |
| TabPFN-CLAMP | 22 | 37.9% | 8.91 | | XGBoost-Mordred | 6 | 10.3% | 19.53 |
| minimol | 19 | 32.8% | 11.43 | | **Chemprop** | **2** | **3.4%** | **21.74** |
| CatBoost-Mordred | 11 | 19.0% | 12.05 | | TabICL/TabPFN-SMI-TED | 4 / 3 | 6.9% / 5.2% | 24.09 / 24.45 |
| | | | | | MolCLR | 1 | 1.7% | 25.81 |

**MoleculeACE only (30 potency-regression tasks)** — the subset that matches us:
TabPFN-CheMeleonFP **100.0% / rank 2.10**; TabICL-CheMeleonFP 76.7% / 4.00; TabPFN-Mordred 66.7% /
4.73; TabPFN-RDKit2d 50.0% / 5.20; TabICL-RDKit2d 53.3% / 5.30; CheMeleon 36.7% / 7.53. **Twelve of
the 24 methods record zero wins.** [verified]

**Polaris only (28 tasks)** — note the ordering *changes*: by win rate TabPFN- and
TabICL-CheMeleonFP tie at 20/28 (71.4%), TabPFN-Mordred 19 (67.9%), TabPFN-RDKit2d 18 (64.3%); but
**by average rank the compact descriptor sets lead** — TabPFN-RDKit2d **5.43**, TabICL-RDKit2d 6.54,
TabPFN-Mordred 6.75, TabPFN-CheMeleonFP 7.11, CheMeleon 10.25, minimol 11.00. [verified]

Four conclusions with the numbers attached:

1. **The featuriser matters more than the tabular model.** TabPFN-Morgan **22.4%** vs
   TabPFN-CheMeleon **86.2%** — same model, same splits. Feeding our existing count-ECFP4 into a TFM
   is the single worst featurisation in the published grid. **RF-Morgan (17.2%) and
   XGBoost-Mordred (10.3%) are both below TabPFN-Morgan**, so a TFM helps even on bad features — it
   just cannot rescue them.
2. **TabPFN > TabICL for the same representation**, "most clearly for CheMeleonFP, RDKit2d and
   Mordred." CheMeleon: 86.2% vs 74.1%. [verified]
3. **Frozen embeddings + TFM beats task-specific fine-tuning** (86.2% vs 41.4%) and is **18.3–46.0×
   faster on GPU, 4.8–27.3× on CPU** (H100, vs fine-tuned CheMeleon: n_train 111 → 46.0×, 728 →
   ~30×, 5,907 → 18.3×). See §2.3 for the caveat that the *fine-tuned → PFN* cell was not run here.
4. **Compact descriptors are a real fallback.** RDKit2d is 217 dims and Mordred 1,613; they reach
   56.9% / 67.2% combined and *lead Polaris on rank*. They scale where 2048-dim embeddings do not —
   the paper notes memory binds at ~21,000 samples with high-dimensional embeddings (~50 GB/fold with
   disk offload) while "compact RDKit2d continued to run without issues." [verified]

**Protocol, verbatim where it matters** [verified]: **no dimensionality reduction**;
standard-scale with train-set statistics; **clamp features to [−6, 6]**; out-of-the-box TabPFN-2.6
and TabICLv2 with **no hyperparameter tuning**. Featuriser dims: CheMeleonFP 2048, Mordred 1613,
Morgan(r=2) 2048, RDKit2d-MACCS 383, RDKit2d 217, CLAMP 768, SMI-TED 768.

**Table S1 — the default inference hyperparameters**, and the answer to the 2048-feature problem:

| Hyperparameter | TabICLv2 | TabPFN-2.6 |
|---|---|---|
| n_estimators | 8 | 8 |
| outlier threshold | 4.0 | None |
| softmax temperature | 0.9 | 0.9 |
| prediction threshold | 0.5 | 0.5 |
| **max features per estimator** | **all unique** | **680 / 500** |
| feat shuffle method | latin | shuffle |

**This is the fact that resolves `21 §5.2`.** TabPFN's nominal 2,000-feature ceiling is not a hard
wall for a 2048-dim embedding: the ensemble **subsamples features per estimator** (680 or 500 of
them, 8 estimators, shuffled), so the full 2048 is covered across the ensemble without PCA.
TabPFN-3 goes further — "features are subsampled in a round-robin fashion" so **no feature is
excluded from the ensemble.** [verified, TabPFN-3 report]

### 3.2 The TabPFN family — limits, as published

TabPFN-3 Technical Report, arXiv [2605.13986](https://arxiv.org/abs/2605.13986), dated **12 May 2026**.
[verified, full text read]

| Release | Rows | Features | Params (clf / reg) |
|---|---|---|---|
| TabPFN-v1 | 1,000 | 100 | 26M / — (classification only) |
| TabPFN-v2 | 10,000 | 500 | 7M / 11M |
| TabPFN-2.5 | 100,000 | 2,000 | 11M / 10M |
| TabPFN-2.6 | 100,000 | 2,000 | 11M / 13M |
| **TabPFN-3** | **1,000,000 @ 200 feats**, or **100,000 @ 2,000**, or **1,000 @ 20,000** | | 53M / 58M |

The numbers are "benchmark-validated regimes where public and internal evaluations demonstrate SOTA
performance; larger or different row–feature configurations may be feasible, but are outside the
validated SOTA envelope." TabPFN-3 is benchmarked along a **cell-budget frontier** — so
**4,376 rows × 2,048 features is comfortably inside the 100k × 2,000 cell**, and the architecture
adds a reduced KV cache, 128 inducing points and row-chunking expressly to make this cheap.
[verified]

**TabArena-medium (10k–100k rows, 15 datasets / 135 tasks) pairwise win rates** for TabPFN-3
(default, single forward pass) as winner: vs AutoGluon 1.5 **73%**, TabICLv2 **78%**, TabPFN-2.6
**87%**, RealTabPFN-2.5 (tuned+ensembled) **76%**, CatBoost (T+E) **75%**, **TabDPT (T+E) 76%**,
LightGBM (T+E) **91%**, TabM (T+E) **77%**, XGBoost (T+E) **77%**. TabPFN-3-Thinking beats plain
TabPFN-3 **69%**. TabPFN-3 "dominates the Pareto frontier" of accuracy vs wall-clock.
[verified, Figures 2–3]

Earlier generation, for context: TabPFN-2.5 "scales to 50,000 samples (5× TabPFNv2) and 2,000
features (4×)", is the leading method on TabArena, and has an **85% win rate on regression** up to
100K × 2K; the regression model "benefits much more from tuning and outperforms AutoGluon 1.4 after
being tuned for 60 configurations." [reported, arXiv 2511.08667]

**Regression handling — and this is the part that fits ST-RAE.** TabPFN uses a **bar-distribution
(piecewise-constant Riemann) head**: the model emits logits for probability mass per bucket, the
discrete CDF is formed, and **arbitrary quantiles are decoded analytically by inverting it with
linear interpolation between bucket borders — from a single forward pass, no per-quantile
retraining.** The API exposes `output_type="main"` → {mean, median, mode, default quantiles} and
`output_type="full"` → + raw logits and the criterion object for custom losses. [verified/reported,
`docs.priorlabs.ai/capabilities/regression`]

ST-RAE is an L1-style hinge with a dead zone, so **its risk-minimising point estimate is a median,
not a mean** (`research/23 §0.3` derives this in-tree). TabPFN hands us the median and any quantile
directly, and the `full` logits let us minimise the *actual* ST-RAE over the predictive distribution
instead of reconstructing a spread from an ensemble.

**Fine-tuning:** TabPFN-2.x is inference-only; **TabPFN-3 does support fine-tuning**
(`docs.priorlabs.ai`). [verified]

**Licence.** TabPFN-2.5/2.6 weights are **non-commercial**; TabPFN-2 is Prior Labs Licence
(Apache-2.0 + attribution). **TabPFN-3 is released under the TABPFN-3.0 License v1.0, "designed to be
permissive for academic use, research, and evaluation in commercial settings. The license explicitly
allows testing, evaluation, and internal benchmarking, so an organization can download the model and
run preliminary assessments on its own datasets without a commercial agreement."** Full text at
`huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE`; commercial deployment →
`sales@priorlabs.ai`. **TabPFN-3-Plus (Thinking)** is API/enterprise only. [verified]

So for the challenge, TabPFN-3 is clean. For a DrugRx `ModelImpl` served to paying users, it still
needs a commercial agreement — **unchanged from `21 §5.2`'s conclusion, but TabPFN-3's licence is
materially more permissive than 2.5/2.6's and should be re-read before assuming otherwise.**

### 3.3 TabICL, TabDPT, CARTE

- **TabICL** (Qu et al., ICML 2025, arXiv 2502.05564). Two-stage **column-then-row attention** →
  fixed-dimension row embeddings → ICL transformer. Pretrained on synthetic tables up to 60K
  samples; handles **500K samples × 500 features**, ~10× faster than TabPFNv2 on large tables.
  `TabICLClassifier` and **`TabICLRegressor`** both ship (`pip install tabicl`). [verified/reported]
  **This is what the organisers used.** It loses to TabPFN for the same representation (74.1% vs
  86.2%, §3.1) and to TabPFN-3 head-to-head (78%, §3.2) — so **the organisers' own recipe is running
  the weaker of the two tabular models**, and that is free headroom.
- **TabDPT** (arXiv 2410.18164, NeurIPS 2025; TabDPT-Turbo arXiv 2608.01400). Pretrained on *real*
  data; strong on CTR23 regression. In the TabPFN-3 matrix, TabPFN-3 beats TabDPT (T+E) **76%**. No
  molecular result found in this screen.
- **CARTE.** Named in the TFM literature as a modern TFM; **no molecular-regression result found in
  two screens** (this one and `21 §5.4`). Treat as unevidenced for our purpose.
- **Chemistry-specific usage reports** beyond the three papers above: Monroe uses TabPFNv3 as its
  production head; MolPAIR uses **TabPFN-3, TabICLv2 and Causilo**; the OpenADMET organisers use
  TabICL in `openadmet-models`. All four independent groups landed on the same architecture pattern
  inside eight months.

### 3.4 How to ensemble them

- **Within a TFM:** `n_estimators = 8` is the default both papers used, with feature shuffling. This
  is already an ensemble; it is not the same as ensembling across models.
- **Across models:** Monroe Table 7 — **0.014 RMSE for 10 members**, and the gain came from adding
  *weaker but differently-wrong* members. Diminishing at k=4.
- **With a GBM — the evidence is specific and positive.** **PFN-Boost** (arXiv
  [2502.02672](https://www.alphaxiv.org/abs/2502.02672)) replaces GBDT's constant initialisation
  with the TFM's score and lets the trees boost the residual:
  `pred₀ = pred₁ + s · SCORE_transformer + C`, with `s` a tunable scale (`s=0` recovers plain GBDT).
  Across 16 UCI/TabLLM datasets at n ∈ {10, 25, 50, 100, 200, 500, full}, **PFN-Boost had the best
  average performance at all but the very smallest sizes, beating standalone XGBoost, standalone
  TabPFN, Stacking and Selection.** [reported — mechanism verified, per-dataset effect sizes not
  extractable]
  This is strictly better-evidenced than late-fusion averaging and is ~20 lines of code, so it is
  the ensembling design to prefer. Note the TFM molecular paper **explicitly did not test** TFM+GBM
  ensembles or feature concatenations and flags them as future work "mostly at the cost of extra
  compute." [verified]
- **The caution that outranks all of the above: ScoringBench** (arXiv
  [2603.29928](https://arxiv.org/abs/2603.29928), 31 Mar 2026). 97 regression datasets; CRPS, CRLS,
  **interval score**, energy score, weighted CRPS; in-context learners, fine-tuned FMs, GBDTs, MLPs.
  Finding: **"model rankings shift substantially depending on the scoring rule: models that excel on
  point-estimate metrics can rank poorly on probabilistic ones, and the top-performing model under
  one proper scoring rule may rank noticeably lower under another … the choice of evaluation metric
  is not a technicality but a modelling decision."** [verified, abstract]
  **ST-RAE is an interval score.** Every ensemble weight, every model selection and every ablation
  gate in this programme must be scored on ST-RAE, not MAE — otherwise we will select the wrong
  member. This is also the formal reason the organisers' MAE-selected recipe is not automatically
  the ST-RAE-optimal recipe.

---

## 4. The organisers' own recipe, with its numbers

Two primary sources, both read: the OpenADMET post
[*Are tabular foundation models all you need?*](https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/)
and `OpenADMET/optimus-prime` [PR #21](https://github.com/OpenADMET/optimus-prime/pull/21),
"Add two-stage log2FC transfer-learning recipe" (smcolby). [verified]

**Attribution matters and is easy to get wrong: the MAE numbers below were measured on the *PXR*
challenge (Apr–Jul 2026, 350+ participants), not on the CYP challenge.** The CYP numbers are
MA-ST-RAE from the live board.

**The pipeline.**

- *Stage 1* — multitask **Chemprop** on primary-screen **log2 fold change**, two tasks = the two
  concentration arms (**8.25 µM** and **33 µM**), on the **10,830-compound** single-concentration
  pool. `noam` scheduler, `warmup = 2`, `max_epochs = 5`, early stopping **disabled**, no eval block.
  Output is used only as frozen features.
- *Stage 2* — **`TabICLRegressorModel`** on dose-response p(E/I)C50. Features = `CheMeleonEmbeddingFeaturizer`
  **2048 → PCA 256** (mean imputation) **+ 2 log2FC readout columns** (passed through unreduced) =
  **258 columns**. `batch_size = 1`, **device = CPU** (explicitly required: 4,392 × 258 exceeds
  24 GB GPU), KV cache **off**, offload **CPU**, `n_estimators = 8`, `random_state = 42`.
  Fixed split, no shuffle: **4,392 train / 260 test.**

**Measured, on the 260-compound blinded PXR phase-2 set:**

| Metric | PR #21 single seed | Published 5-seed mean |
|---|---|---|
| MAE | 0.4343 | **0.4269** |
| R² | 0.5296 | **0.5566** |
| Spearman ρ | 0.7800 | **0.7959** |
| Kendall τ | 0.5935 | **0.6054** |
| RAE | 0.6016 | **0.5914** |

Their leaderboard entry reached **0.41 MAE**. The single-seed vs 5-seed gap of ~0.007 MAE is "within
about one seed's noise" — **that is the seed noise band, and it is the floor for calling any
ablation real.**

**Model panel, same split, 5 seeds, Tukey HSD:** TabICL **0.427** (nominal winner), **TabPFN v3
0.436** (statistically tied), and TabPFN-2.6, TabFM (capped at 500 rows by memory), LightGBM and
XGBoost all **separated from the leader**. The best GNN arm — Chemprop initialised from the log2FC
checkpoint — was **statistically tied with the CheMeleon baseline**; "nothing on the panel is
separated from the best row." [verified/reported]

**Dimensionality ablations** (this is the part worth copying exactly):
- **RDKit descriptors:** native dimension led at **0.53 MAE**, statistically indistinguishable from
  PCA-128.
- **CheMeleon embedding:** PCA to **256, 384 and 512 were statistically indistinguishable**; **128
  and below were clearly separated and worse** → they chose **256**.
- **Best feature set measured:** CheMeleon + Chemprop log2FC readout **+ RDKit 128**. They shipped
  the simpler **CheMeleon + log2FC** because descriptors added negligible value.
- The log2FC-trained Chemprop embedding is described as **"the single strongest feature"** into the
  TFM. [reported]

**An untested cell, and it is the one the other literature says to run.** The organisers never
reported **native 2048-dim CheMeleon** against PCA-256 — they bracketed 128–512. The RWTH paper's
winning protocol used **no dimensionality reduction at all** on the same 2048-dim vector (§3.1), and
TabPFN-3's round-robin feature subsampling plus row-chunking removes the memory reason the
organisers had to reduce (they were forced onto CPU at *258* columns with TabICL). **CheMeleon
native 2048 → TabPFN-3 is a one-line experiment with published evidence on both sides and nobody's
measurement.**

**Uncertainty, and the gap it leaves.** Their intervals have a **miscalibration area of just 0.04**
— the right *width* — but the **Spearman ρ between predicted sd and absolute residual is 0.26.**
Their words: "tabular foundation models give us well-calibrated intervals of roughly the right
width, but not a reliable ranking of which compounds are uncertain / hard." [verified]
For a metric that pays off per-compound interval placement, a 0.26 difficulty ranking is a real
deficiency and an opening: **the width is free, the ranking is not.**

**Where this puts us.** TabICL-baseline **0.68 MA-ST-RAE** on the CYP board; their older
CheMeleon-only baseline **0.83**; board leaders **0.43** ("~36% better"). Our shipped macro is
**0.8663** (`PROGRESS.md`). **The documented, hyperparameter-complete, publicly-reproducible recipe
is 0.19 ST-RAE better than what we are shipping, and we have not run it.**

---

## 4a. We already ran an in-context arm and got a null — and why that null does not transfer

This is the most important paragraph in the file, because without it §3 reads as a recommendation to
redo a finished experiment. `scripts/cyp_incontext.py`, 5 scaffold repeats, paired, re-confirmed on
side2 (`PROGRESS.md`): [verified, in-tree]

| arm | macro ST-RAE | vs xgb (3,091 features) | verdict |
|---|---|---|---|
| xgb (full features) | 0.4565 (sd 0.0095) | — | — |
| **icl_top** (500 Spearman-selected) | 0.4597 (sd **0.0043**) | −0.0032 ± 0.0051, 2/5 | inside noise |
| icl_desc (216 RDKit2D) | 0.4650 | −0.0084 ± 0.0016, 0/5 | WORSE |
| dpt_desc | 0.4687 | −0.0121 ± 0.0010 | WORSE |
| dpt_top | 0.4861 | −0.0295 ± 0.0028 | WORSE |
| **icl_pca / dpt_pca** (500 PCs) | 0.5112 / 0.5663 | **−0.0547 / −0.1098** | **WORSE** |

**Four confounds, each of which the 2026 literature names explicitly as the losing choice:**

1. **The featuriser was count-ECFP, not CheMeleon.** This is the `TabPFN-Morgan` cell:
   **22.4%** win rate against `TabPFN-CheMeleonFP`'s **86.2%** on the same 58 tasks (§3.1). The arm
   measured the worst featurisation in the published grid. *The script's own docstring cites the
   25×25 benchmark as the reason not to expect a learned representation to help — but that benchmark
   never tested CheMeleon or a TFM head (§2.1), and it is classification AUROC.*
2. **It reduced dimensionality, and the winning protocol forbids exactly that.** The RWTH protocol is
   verbatim **"no dimensionality reduction"**, standard-scale on train statistics, **clamp to
   [−6, 6]** (§3.1). Our catastrophic arm is `*_pca` at **−0.0547 / −0.1098** — and the in-tree
   conclusion drawn from it ("the reduction, not the model, was doing the damage") is *the same
   conclusion the literature had already published*. The organisers' PCA-256 works only because it
   is applied to a 2048-dim CheMeleon embedding whose variance is dense; 500 PCs of a sparse
   count-fingerprint is a different operation.
3. **The ≤500-feature budget was already two generations stale.** 500 features is **TabPFN-v2's**
   limit. TabPFN-2.5/2.6 take **2,000**, and TabPFN-3 takes **2,000 at 100k rows** (§3.2). At
   4,376 × 2,048 we are comfortably inside the validated envelope, so **the reduction that destroyed
   the arm was never necessary.**
4. **TabPFN itself was never run** — the arm used TabICL and TabDPT because TabPFN's weights sit
   behind a licence acceptance, correctly treated as the owner's decision rather than an agent's.
   But TabPFN beats TabICL for the same representation (**86.2% vs 74.1%**), beats TabICLv2 **78%**
   head-to-head on TabArena, and beats TabDPT (T+E) **76%** (§3.1–3.2). TabDPT, the worst performer
   in our arm, is also the weakest of the three in the published matrix — **the in-tree ordering
   TabICL > TabDPT reproduces the literature's ordering exactly**, which is a good sign the arm was
   run correctly and a bad sign for concluding anything about the approach from it.
   On the licence: **TabPFN-3's TABPFN-3.0 License v1.0 "explicitly allows testing, evaluation, and
   internal benchmarking"** without a commercial agreement (§3.2). That is still the owner's
   acceptance to give, but it is a materially lower bar than the non-commercial 2.5/2.6 terms the
   earlier decision was taken against.

**The live signal the arm already found, which §1 predicts.** `icl_top` **beat xgb on 3 of 4
endpoints** (CYP1A2 +0.0062, CYP2C9 +0.0101, CYP3A4 +0.0116) and lost CYP2D6 by −0.0175, at **half
the across-split variance** (sd 0.0043 vs 0.0095). A crippled TFM on the worst available features
already ties a tuned XGBoost on 3,091 features and is twice as stable. That is the floor of this
approach, not its ceiling — and the single failing endpoint is the one §6.3 says is failing for an
*information* reason (fewest actives, most censoring), which is what §7 and A2 address.

**So the correct statement of the open question is not "do TFMs help here" — it is "does
CheMeleon-at-native-dimension help here", and no arm in this programme has ever fed CheMeleon to
anything.** `scripts/chemeleon_fp.py` and `drugrx/features.py` already extract the frozen 2048-d
vector, and neither is wired into the CYP pipeline (`drugrx/cyp/*.py` contains no reference to it).
[verified, in-tree]

---

## 5. Chemprop v2 (D-MPNN) — current best practice, and its real role here

Chemprop v2: *An Efficient, Modular Machine Learning Package for Chemical Property Prediction*,
JCIM 2026, [10.1021/acs.jcim.5c02332](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02332); v1 paper
JCIM 2024 [10.1021/acs.jcim.3c01250](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01250).
[reported]

- **Hyperparameters that matter**, in the order the docs and practitioner guides give them:
  `depth` (message-passing iterations), `message_hidden_dim`, `ffn_num_layers`, `ffn_hidden_dim`,
  `dropout`, activation. HPO is **Ray Tune** in v2, parallelisable across instances. v2 adds
  multi-GPU training.
- **Pretraining / transfer:** `--from-foundation CheMeleon` (≥ 2.2.0) initialises the encoder from
  CheMeleon; the `TrainedModelFeaturizer` path (used in PR #21) lets a trained Chemprop be read as
  frozen features by a downstream model. Checkpoint-initialisation from a related task is the arm the
  organisers measured as tied-with-best.
- **Multitask:** constrained multitask D-MPNN supported. Epoch budgets in the literature are
  **60 epochs single-task / 40 multitask**; the organisers used **5 epochs** for the log2FC encoder,
  which says this stage is a feature extractor, not a fitted model.
- **Uncertainty:** MVE (mean-variance estimation) and ensembles for estimation; **four calibration
  methods for regression — z-scaling, t-scaling, Zelikman's CRUDE, and MVE weighting** — plus two for
  classification. Given §4's finding that *width is calibrated but ranking is not*, CRUDE and MVE
  weighting are the two to try, because they recalibrate the *conditional* spread rather than a
  global scale.

**But do not plan on Chemprop as the final regressor.** In the only head-to-head at our scale it is
**3.4% win rate, average rank 21.74 of 25** [verified, §3.1], 0% of MoleculeACE tasks in CheMeleon's
own table, and **1.046 RMSE vs MolPAIR's 0.885** under structural shift [verified, §8]. Its value in
2026 is as (a) the CheMeleon architecture, and (b) **the log2FC feature generator** — which is
exactly how the organisers use it.

---

## 6. CYP inhibition specifically

### 6.1 Benchmarks, and what they are not

- **TDC CYP** is the **Veith classification** suite: CYP1A2 12,579, CYP2C19 12,665, CYP2C9 12,092,
  **CYP2D6 13,130**, CYP3A4 12,328 compounds, with positive-class shares **46.34 / 45.95 / 33.45 /
  19.15 / 41.45 %**. [verified, benchmark Table 4] There is **no pIC50-regression CYP task in TDC**,
  so the leaderboard is not evidence for us, and `21 §2.1` already records why the leaderboard
  itself should be discounted.
- **Polaris** has no CYP potency-regression task either; the OpenADMET challenges *are* the benchmark.
- **MoleculeACE** (30 ChEMBL targets, 48.7K measurements / 35.6K unique molecules; cliffs defined as
  pairs > 90% similarity with > 10-fold activity difference) is the closest public proxy and is what
  every number in §1–§3 is measured on. [verified/reported]

### 6.2 Typical achievable accuracy — and the numbers not to believe

Published CYP *regression* R² values span 0.69–0.99 (PLSR on 6–15 descriptors; G/PLS molecular field
analysis reaching external R²pred = 0.912, RMSEP 0.363) — but these are **tiny congeneric series**
(e.g. aryloxypropanolamines) where a handful of descriptors span the whole SAR. **They do not
transfer and must not be used as targets.** [reported]
The defensible figures for a diverse set at our n are the ones in §1 and §4: **RMSE ≈ 0.63–0.68 log
units**, **MAE ≈ 0.43**, **R² ≈ 0.53–0.56**, **Spearman ≈ 0.78–0.80**. AstraZeneca's internal CYP
assays carry a **control-compound sd of 0.11–0.12** [verified, §7], so the irreducible assay floor is
~0.12 and we are an order of magnitude above it — the ceiling is not the assay.

### 6.3 Is CYP2D6 the hard one? Yes, for three separable reasons

1. **Fewest actives.** 19.15% positives in Veith against 33–46% for the others — the lowest hit rate
   of the panel, so the fewest uncensored pIC50 labels and the most left-censoring. This is an
   information argument, not a chemistry one, and it is the dominant one for us (our CYP2D6 ST-RAE is
   **1.4577**, the worst endpoint, `research/24`).
2. **The narrowest pharmacophore.** CYP2D6 inhibition "typically requires a protonatable basic
   nitrogen and a hydrophobic aromatic or aliphatic region to anchor the molecule"; **Glu216 and
   Asp301 favour basic substrates and discriminate against acidic ones.** A model must learn a
   charge-state-dependent rule, and descriptor sets represent protonation state poorly. [reported]
3. **Substrate-dependent inhibition.** "Inhibitors with structurally constrained amino moieties
   exhibited different inhibition potencies depending on the substrate used," with atypical
   inhibition kinetics for various inhibitor–substrate pairings. The *label itself* is probe-dependent,
   so cross-source CYP2D6 data is noisier than the other isoforms. [reported]
   General: "CYP inhibition is more difficult to predict than metabolism owing to the lack of
   reactive sites on inhibitors that can be used as anchor positions," and inhibitors may chelate
   heme, bind entrance/exit channels, or distal pockets. [reported]

### 6.4 Multitask across the four isoforms

"The five CYP isoform datasets are highly correlated, and the multi-task model can capture relevant
information among subtasks" — the generic reported figure is **5–15%** improvement for data-sparse
endpoints via representation transfer from data-rich ones; specific published gains are classification
AUCs (MTGL-ADMET CYP1A2 0.952). [reported] No controlled multitask-regression effect size for CYP
pIC50 was found in this screen. **[assumed]** a 4-task shared trunk is worth most on CYP2D6 (the
sparsest) and ~nothing on CYP1A2/3A4, which matches our board pattern. Note TFMs are **single-target**
engines: multitask has to enter as either a shared Chemprop/CheMeleon trunk upstream, or as
*other isoforms' predictions used as feature columns* — the latter is the TFM-native way and is the
same trick as the log2FC column.

---

## 7. Censored and interval-valued labels

### 7.1 The one paper that is our exact problem

Svensson, Bjerrum et al. (AstraZeneca), *Enhancing uncertainty quantification in drug discovery with
censored regression labels*, **Artificial Intelligence in the Life Sciences 7 (2025) 100128**,
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667318525000042) /
[Chalmers full text](https://research.chalmers.se/publication/545243/file/545243_Fulltext.pdf).
15 assays, **three temporal splits**, test sets always retaining censored labels. [verified, read in full]

Three of the assays are **CYP pIC50 with majority left-censoring** — the closest published match to
our labels that exists:

| Assay | End-point | n | Left-censored | Right-censored | Control sd |
|---|---|---|---|---|---|
| ADME-T CYP 1 (CYP3A4) | pIC50 | 12,875 | **61%** | 0% | 0.11 |
| ADME-T CYP 2 (CYP2C9a) | pIC50 | 12,876 | **63%** | 0% | 0.11 |
| ADME-T CYP 3 (CYP2C9b) | pIC50 | 14,062 | **58%** | 0% | 0.12 |
| ADME-T hERG | pIC50 | 67,687 | 42% | 0% | 0.19 |
| Target 1–7 | pIC50/pEC50 | 5,082–14,605 | 0–43% | 0–2% | 0.13–0.36 |

**Two loss families, and they behave oppositely:**

| Loss family | Models | Effect on NLL (assays > 35% censored) | Effect on MSE |
|---|---|---|---|
| **One-sided squared loss** | Ensemble, MC-Dropout, Bayes-by-Backprop | **+52%** (Ensemble), **+48%** (MC-Dropout), +24% (BbB, already from >13%) | **+~20%** |
| **Tobit / Gaussian NLL** | Gaussian, Gaussian Ensemble | +34% (aleatoric), **+56%** (epistemic) | **−~30%** |

Verbatim: *"the predictive accuracy of the Gaussian models trained with the Tobit loss is mostly not
improved by including the censored labels … The proportional losses in MSE … consist of around −30%
on average for all assays. This is an unfortunate drawback of our proposed adaption."* And the
authors' own suggested remedy: *"it is possible that a weighting scheme included in the Tobit loss
functions could make the Gaussian models trained with censored labels favorable also in terms of
predictive accuracy."* [verified]

**The actionable rule: use the one-sided squared (or absolute) loss for point accuracy; reach for the
full Tobit likelihood only if you want calibration and are willing to pay ~30% MSE.** `research/20`'s
Rank-1 recommendation (left-censored labels with a masked one-sided loss) is the arm this paper
supports; its Rank-3 (`XGBoost survival:aft`) is the Tobit-family arm and should be expected to trade
accuracy for calibration.

**The finding that reframes the whole question** — verbatim: *"Models trained without censored labels
do not perform well on test sets with censored labels, and models trained on full datasets do not
perform well on test sets without censored labels."* On observed-only test labels, censored training
"is mostly not improved" and for Gaussian models "almost always negatively impacted." [verified]

Our blinded 750 is an analog expansion of *screening hits*, so it contains a large inactive fraction
(our own minimax scan assumes 0.55–0.81). **That means our test set is the "with censored labels"
case, and the paper says a model trained only on the ~4,376 dose-response compounds is the wrong
model for it.** This is the same conclusion `research/20 §1` reached from decision theory; here it is
measured on three CYP assays.

### 7.2 Methods inventory

- **Tobit likelihood** — the integral of the density over the censored region. Standard, and what
  `survival:aft` implements.
- **One-sided squared / absolute loss** (Arany et al.; "censored MSE"/"censored MAE") — penalise only
  violations on the wrong side of the threshold. Simpler to implement, **and the better choice for
  MSE** per §7.1.
- **`XGBoost survival:aft`** — native **left, right and interval** censoring since 1.2.0
  ([docs](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html), arXiv
  [2006.04920](https://arxiv.org/abs/2006.04920)). Applied to hERG QSAR to accept the qualified
  values "often reported in protein inhibition studies," reaching comparable blocker/non-blocker
  discrimination at a 10 µM threshold ([J Comput Aided Mol Des 2022](https://doi.org/10.1007/s10822-022-00483-0)).
  [reported]
- **L2Boost-CUT / L2Boost-IMP** — Bian, Yi & He, **ICLR 2025**, arXiv
  [2601.17973](https://arxiv.org/abs/2601.17973). Boosting for **interval-censored** data via a
  censoring-unbiased transformation (CUT) either in the loss (CUT) or by imputing transformed
  responses (IMP); functional gradient descent; **proved minimax-optimal MSE rates.** [reported]
  This is the most directly applicable published method for *interval* labels specifically, which is
  what our credible intervals are.

### 7.3 TFMs cannot eat censored labels — and the three 2026 workarounds

This is a hard architectural constraint and the main friction between §3 and §7: **a TFM is a
point-regression in-context engine; "standard TFMs cannot handle natively censored data, leading to
biased and inaccurate predictions."** Three published routes, all 2026:

1. **Discretise and treat censoring as missing labels.** Kim, Lai & Zhang (Imperial),
   *Tabular Foundation Models Can Do Survival Analysis*, arXiv
   [2601.22259](https://arxiv.org/html/2601.22259v2) v2, 18 Jul 2026. Bin the axis into K bins, make
   binary labels `Y_{i,k} = 1(T_i ≤ t_k)`, and **evaluate the loss only for pairs where `t_k < C_i`**
   — a censored point simply contributes nothing beyond its threshold. MITRA and TabPFN-2.5 on 4,343
   static SurvSet datasets (median n ≈ 461, IQR 228–1,494; median censoring **56.4%**):
   MITRA-CF C-index **0.674 ± 0.017** / IBS **0.164 ± 0.007**, against XGBoost-CF 0.688/0.166,
   CoxPH 0.682/0.169, RSF 0.666/0.167, DeepSurv 0.650/0.179. **Modest at the median (~2–5%), > 20%
   on the top decile of datasets.** Right-censoring only; left-censored regression "would require
   fundamentally different assumptions." [verified]
   *Our case is left-censoring, which is right-censoring with the sign flipped, so the mechanism
   transfers even though the paper's guarantees do not.* **[assumed]**
2. **Buckley–James iterative imputation.** *Staying Alive: Uncensored Survival Analysis with Tabular
   Foundation Models*, arXiv [2606.03689](https://arxiv.org/html/2606.03689v1), 2 Jun 2026.
   TabSA-BJ: initialise censored pseudo-targets from Kaplan–Meier jackknife; then iterate —
   TFM predicts the mean, **sample imputed values from the truncated conditional `p(y | y > C)`**,
   update a single scalar σ by MLE; **10 iterations to convergence.** TabPFN and TabICL on 5 datasets
   (n = 500–8,873, censoring 31.9–72.5%): C-index comparable to Cox/AFT, **IBS worse** (Cox/AFT
   better calibrated). It **collapses the TFM's predictive distribution to point estimates during
   iteration.** [verified]
   This is the cheapest of the three to implement on top of an existing TabPFN call and the only one
   that needs no change to the TFM's interface — **it is a label-imputation loop, not a new model.**
3. **Pseudo-observation targets** — convert the censored outcome into jackknife pseudo-observations
   for a restricted mean, letting any tabular backbone do horizon-specific regression with no
   survival-specific fine-tuning (arXiv [2607.09577](https://arxiv.org/abs/2607.09577); see also
   SurvPFN arXiv 2606.04564). [reported]

**Practical synthesis for us:** the TFM takes the ~4,376 dose-response compounds as point labels, and
the ~17k screened-but-not-promoted compounds enter **either** as a Buckley–James imputation loop
(route 2) **or**, far more cheaply and with the organisers' own evidence behind it, **as the
Chemprop-predicted log2FC feature column** (§4) — which sidesteps censoring entirely by moving the
screen from the label side to the feature side. `research/20 §Rank 7` already warns against
Hill-inversion of log2FC into a pIC50 point label; using it as a *feature* is the sanctioned form, and
the organisers call it their single strongest feature.

---

## 8. Delta / pairwise learning

Our test set is an analog expansion, so this family deserves a fair hearing. The measured answer is:
**real, small, and smallest exactly where our geometry sits.**

### 8.1 MolPAIR — the strongest and most recent evidence, and it is a 0.008 effect

*Molecular Property Prediction under Structural Shift with Tabular Foundation Models*, Lee, Lee,
Jeong & Yoo (Nums AI), arXiv [2609.38744](https://arxiv.org/html/2609.38744), **30 Sep 2026**,
`github.com/nums-ai/MolPAIR`. Two frozen TFMs: a global predictor, plus a **relation predictor that
regresses the signed error difference between query and anchor**; ≤ 32 anchors at inference
(exact MMPs first, then Morgan similarity); similarity-adaptive **reliability gate** at τ = 0.20.
58 tasks (30 MoleculeACE + 28 Polaris; 47 regression, 11 classification), Bemis–Murcko separation
**plus a Tanimoto ≤ 0.60 cutoff**, 5 seeds = 286 splits, Elo with 10,000-resample bootstrap CIs and
Holm-adjusted sign tests. [verified]

| Method | MoleculeACE RMSE | Elo | Top-1 | vs global |
|---|---|---|---|---|
| **MolPAIR** | **0.885 ± 0.100** | **1360** [1289, 1455] | **29** | **46 / 58** |
| Global TFM baseline | 0.893 ± 0.097 | 1246 [1196, 1310] | 7 | — |
| MiniMol (tuned head) | 0.931 ± 0.107 | 1122 | 14 | — |
| ExtraTrees | 0.921 ± 0.102 | 1062 | 2 | — |
| Chemprop | 1.046 ± 0.107 | 602 | 0 | — |
| iMoLD | 1.080 ± 0.139 | 522 | 0 | — |

**The pairwise machinery's own contribution is 0.00819 RMSE [0.00235, 0.01388]** — significant,
+114 Elo, 46/58 wins, and **an eighth of the 0.065 that the global TFM already has over
ExtraTrees.** Robust across cutoffs (0.40: 38/14; 0.50: 41/13; 0.60: 46/12; 0.70: 42/16). Gains hold
across all 12 representation × backbone cells (ΔElo +97.7 for RDKit2D+Causilo to +314.1 for
MiniMol+TabICLv2; CheMeleon+TabPFN-3 **+290.8**). **The largest single component is the reliability
gate**: removing it costs **0.03443 RMSE [0.01947, 0.04956]** — i.e. *gating the correction matters
four times more than the correction itself.* An ungated pairwise model is worse than no pairwise
model. [verified]

### 8.2 DeepDelta and the older literature

*DeepDelta: predicting ADMET improvements of molecular derivatives with deep learning*, Fralish,
Reker et al., **J Cheminform 15, 101 (2023)**,
[PMC10605784](https://pmc.ncbi.nlm.nih.gov/articles/PMC10605784/). 10 ADMET datasets,
**253–1,321 compounds** (fraction unbound brain 253 → half-life 1,321), 5 × 10-fold CV + external
test sets. **Pairs are formed *after* splitting** — "every molecule will only be present in pairs
made from either the training or the test set but not both." [verified]

- CV Pearson r **median 0.72** vs Chemprop **0.60** and RF **0.63**; wins **90%** of benchmarks on r,
  average improvement **+0.04 (p = 0.006)**; MAE wins **60%**, average **+0.13 (p = 0.04)**.
  External test sets: beat both on r, MAE and RMSE in all cases.
- **Three free invariance diagnostics:** `D(x,x) = 0` → MAE **0.127 ± 0.042**;
  `D(x,y) = −D(y,x)` → r **−0.947 ± 0.044**; additivity `D(x,y)+D(y,z) = D(x,z)` → MAE
  **0.127 ± 0.043**. The same-molecule MAE correlates with CV performance at **r = 0.916** — a
  **label-free predictor of model quality, computable on the blinded 750.** Steal this regardless of
  whether a delta model ships.
- **Failure modes, which are ours:** highest error on the **lowest 10% of |Δ|** (p = 0.002); loses
  outright on hepatic clearance; all three clearance sets are r < 0.5 for every model.
  `research/23 §1.2` adds the decisive one: DeepDelta is only *statistically indistinguishable* from
  RF/Chemprop **within the same scaffold** (p > 0.3 for r, p > 0.1 for MAE) — **and same-scaffold is
  our entire test set.**

### 8.3 The review, and the negative result

*Pairwise learning for molecular property prediction and optimization*, Fralish & Reker,
**Front. Drug Discov., 17 Jun 2026**,
[10.3389/fddsv.2026.1859068](https://www.frontiersin.org/journals/drug-discovery/articles/10.3389/fddsv.2026.1859068/full).
[verified] Pooled across **274 benchmark datasets**: error reductions "up to 60%", but the per-study
ranges **straddle zero** (`research/23` transcribes them: Wetzel −1.6→+60.2%, Zadorozhny
−11.6→+43.9%, Belaid −0.1→+27.7%, Tynes +3.3→+24.8%). PADRE: max 33%, **average 9%** MAE reduction on
4 datasets of 548–12,726. Pairs encode symmetry, additivity and identity for free. Data augmentation
is the real mechanism — DyAb turned "500 datapoints into nearly 125,000 pairs". Active-learning gains
are the largest reported effects (ABCG2: 7 → 30 hits; Chemprop: 61% vs 45% of the top decile
recovered) — **but we are not running an active-learning loop.**

**SQRL** (arXiv [2501.09103](https://arxiv.org/pdf/2501.09103)) is the negative result that should
set our prior. Similarity-thresholded pairing (α = 0.7), Spearman ρ on 30 MoleculeACE tasks:
pairing **hurts every tree model** (XGBoost 0.79 → 0.76; RF 0.80 → 0.77) and **rescues the MLP from
0.32 → 0.73** — still short of RF's 0.80. The authors' own sentence: **"None of the neural networks
were able to outperform tree-based baseline models without SQRL pairing."** Pairwise learning is a
*fix for a weak head*, not an improvement on a strong one. Given §3 says our head should become a
TFM, pairing's expected value drops accordingly — and MolPAIR, which measured pairing *on top of* a
TFM, got 0.008.

### 8.4 The additivity ceiling

`research/23 §1.3` holds the numbers: nonadditivity in **57.8%** of AZ in-house and **30.3%** of
public assays; nonadditive MMP subsets cost **1.7–2.3× the RMSE** of additive ones (log D 0.23 →
0.51; permeability 0.28 → 0.64), **and deep learning is no exception.** Our own geometry — 46% of
T ≥ 0.4 pairs differing by > 1 log — is nonadditivity-rich. This is the structural reason to prefer
**gated** corrections (MolPAIR's τ) over unconditional delta prediction.

---

## 9. Ranked by expected gain on *our* problem

Effect sizes are MA-ST-RAE unless stated. Our baseline: **0.8663 shipped**; organisers' documented
recipe **0.68**; board leaders **0.43**. The seed-noise floor is **~0.007 MAE** (§4), so nothing
below ~0.01 is measurable in one run.

### Tier A — do this first

**A1. Feed CheMeleon's native 2048-d vector to a TFM. Expected 0.05–0.15.**
**This is not a re-run of `cyp_incontext.py`.** That arm changed the *model* while keeping the worst
featurisation in the published grid and then reducing it (§4a). A1 changes the **featuriser**, which
is the variable all three controlled comparisons say dominates:
- featuriser Morgan → CheMeleon, *same TFM, same splits*: **22.4% → 86.2%** win rate (§3.1);
- head MLP → PFN, *same encoder*: **0.723 → 0.637 RMSE** (§2.3);
- encoder + head vs GBM on 30 potency tasks: **0.680 → 0.641 RMSE** (§1).

Order of operations, cheapest-first, each a gate:
1. **CheMeleon 2048 → TabICL, no reduction**, standard-scaled on train statistics and **clamped to
   [−6, 6]** (the RWTH protocol verbatim). This is one featuriser swap away from the arm that already
   ran, reuses `scripts/chemeleon_fp.py`, needs no new licence, and isolates the featuriser against a
   known in-tree baseline (`icl_top` 0.4597). If CheMeleon is the variable that matters, this moves
   on its own.
2. **Then the organisers' exact recipe** — CheMeleon → PCA 256 + 2 log2FC columns → `TabICLRegressor`,
   PR #21's hyperparameters verbatim — to land their documented **0.68** and confirm we can reproduce
   a published number before trying to beat it.
3. **Then TabPFN-3** in place of TabICL, if the owner accepts the licence (§3.2: it explicitly permits
   evaluation and internal benchmarking). It wins the same-representation comparison 86.2% vs 74.1%,
   beats TabICLv2 78% on TabArena, and its row-chunking removes the GPU-memory wall that forced the
   organisers onto CPU at only 258 columns.

Take the **median** from the bar-distribution head, not the mean — ST-RAE's optimum is a median
(`research/23 §0.3`). Expect the gain to concentrate on CYP1A2/2C9/3A4, where `icl_top` already won,
and **not** on CYP2D6, which is an information problem (A2, B1) rather than a representation one.

**A2. Add the screen as a predicted feature column. Expected 0.03–0.08.**
The organisers call the Chemprop log2FC readout **"the single strongest feature"**, and it is the
difference between their 0.83 CheMeleon-only baseline and their 0.68 TabICL baseline. Their stage-1
config is published in full (two concentration arms, 10,830 compounds, noam/warmup 2/5 epochs).
`research/22 §1.7` already flags this as the data we are under-using. Feature side only — **never as
a point pIC50 label** (`research/20 Rank 7`).

**A3. Score every gate on ST-RAE, not MAE. Expected 0.01–0.04, and it protects A1/A2.**
ScoringBench: rankings "shift substantially depending on the scoring rule" across 97 regression
datasets. Our metric is an interval score; the organisers' recipe was selected on MAE. Two concrete
consequences: the ensemble weights that minimise MAE are not the ones that minimise ST-RAE, and
TabPFN's `output_type="full"` logits let us minimise ST-RAE **directly over the predictive
distribution** instead of post-hoc shrinking a point estimate. This is the cheapest item here and
the one most likely to make A1 look smaller than it is if skipped.

### Tier B — worth a day each

**B1. Censored screen compounds via a one-sided squared loss. Expected 0.02–0.06, concentrated on CYP2D6.**
AZ's three CYP pIC50 assays at 58–63% left-censoring: **+~20% MSE and +~52% NLL** from the one-sided
loss. **Do not use the full Tobit likelihood for point accuracy — it cost ~30% MSE.** Because TFMs
cannot eat censored labels, this arm has to be either a separate model (the MLP/GBM ensemble we
already have, retrained with the masked one-sided loss) blended into the TFM, or a Buckley–James
imputation loop (§7.3 route 2, ~10 iterations, no interface change). `research/20 Rank 1` already
specifies the implementation; §7.1 is the measurement that justifies choosing its loss over `survival:aft`.

**B2. PFN-Boost the residual. Expected 0.01–0.03.**
`pred₀ = pred₁ + s·SCORE_TabPFN + C`, trees learn the rest. Best average performance across 16
datasets at all but the smallest n, beating standalone XGBoost, standalone TabPFN, Stacking and
Selection. ~20 lines, and better evidenced than late-fusion averaging. Note the molecular TFM paper
never tested TFM+GBM — this is genuinely open, with positive non-molecular evidence.

**B3. Fine-tuned-CheMeleon → TabPFN as a second arm. Expected 0.00–0.03.**
The one place the two best papers disagree, and the disagreement is resolvable in two runs. Monroe:
for CheMeleon specifically, **FT→PFN 96.4% > native 92.9% > frozen→PFN 75.0%** (Polaris). RWTH:
frozen→PFN 86.2% ≫ FT→native 41.4% (58 tasks) — but they never ran FT→PFN. Supersedes
`21 §5.1`'s "do not fine-tune CheMeleon," which rested on the missing cell.

**B4. Diversity ensembling, deliberately. Expected ~0.014.**
Monroe Table 7: 0.629 → 0.615 at k=10, flat between k=4 and k=8, and the members that helped were
the *individually weaker* ones. So ensemble **across featurisers** (CheMeleon + Mordred 1613 +
RDKit2d 217 — the 67.2% and 56.9% configurations) rather than across seeds of the same stack. Weight
on ST-RAE (A3).

### Tier C — real but small, or already covered

- **C1. Per-isoform / multitask trunk for CYP2D6.** No controlled regression effect size exists;
  **[assumed]** 0.01–0.03 and only on CYP2D6. The TFM-native form is *other isoforms' predictions as
  feature columns* — the same trick as A2.
- **C2. Gated pairwise correction (MolPAIR-style).** Measured at **0.008 RMSE** on top of a TFM, with
  the **gate worth 4× the correction (0.034)**. If built at all, build the gate. DeepDelta's
  advantage is ~zero within-scaffold, which is our whole test set.
- **C3. DeepDelta's invariance diagnostics.** Not a gain, a **free, label-free quality probe** on the
  blinded 750: same-molecule MAE correlates with CV performance at **r = 0.916**. Cheapest
  diagnostic in this document.
- **C4. Interval-censored boosting (L2Boost-CUT/IMP, `survival:aft`).** The right tool for *interval*
  labels with minimax guarantees, but §7.1 predicts a calibration-for-accuracy trade. Use for the
  interval arm of an ensemble, not as the point predictor.
- **C5. Chemprop v2 uncertainty calibration (CRUDE, MVE weighting).** Targets the real gap the
  organisers left — intervals of the right *width* (miscalibration area 0.04) but a **0.26** Spearman
  difficulty ranking. Fixing the ranking is worth more than fixing the width under a per-compound
  interval metric. **[assumed]** 0.01–0.03.

### Do not do

- **Any structure-only SSL encoder.** Uni-Mol/Uni-Mol2 (rank 16.6/13.3 of 25), MoLFormer-XL (9.5,
  0% MoleculeACE), GROVER (17.1), GEM (19.4), Mol2Vec (10.4), MolCLR (1.7% win), SMI-TED (5.2%,
  rank 24.5), ChemFM-3B (loses to ECFP at posterior 0.74). §2.1 settles this on 25 datasets with
  Bayesian multiple-comparison control.
- **LLMs**, fine-tuned or in-context. `21 §4`: TxGemma-27B loses **9 of 9** ADMET regression endpoints
  to MolE by 4–43%; 60-shot ICL underperforms 0-shot in 19 of 27 cases.
- **Feeding raw count-ECFP to a TFM.** 22.4% win rate — the worst configuration published.
- **Plain Chemprop as the final regressor.** 3.4% win rate, rank 21.74 of 25; 1.046 RMSE under
  structural shift. Use it as a feature generator (A2) only.
- **Ungated all-pairs delta learning.** SQRL hurts every tree model; MolPAIR's ablation says the
  ungated variant is worse than no pairing.
- **Trusting the old CYP congeneric-series R² of 0.9+** as a target. §6.2.

---

## 10. The three places this file corrects the in-tree record

0. **`PROGRESS.md` — "in-context tabular models: a null."** The null is real but it is a null about
   *reduced count-ECFP*, not about in-context learning (§4a). Four confounds, each a documented
   losing choice: the 22.4%-win-rate featuriser, a reduction the winning protocol forbids, a
   500-feature budget two TabPFN generations stale, and no TabPFN. The arm's own surviving signal —
   3 of 4 endpoints won at half the variance — points the other way. The line in `PROGRESS.md` that
   *"every one of the six fashionable methodologies is a model-class change, and this task has
   answered model-class changes with nulls six times now"* is correct and is **the argument for A1**,
   because A1 is a *featuriser* change, not a model-class change.

1. **`21 §5.1` — "Do not fine-tune CheMeleon."** The comparison behind that was frozen→TabPFN vs
   fine-tuned→*native MLP head*. Monroe ran the missing cell and found **fine-tuned→TabPFN is the
   best adaptation for CheMeleon specifically** (96.4% within-model on Polaris). Downgrade to "test
   both" (B3).
2. **`21 §5.2` — "Our ~3,600-dim vector exceeds TabPFN's 2,000-feature ceiling, so PCA or select."**
   The ceiling is softer than it reads: TabPFN ensembles **subsample features per estimator**
   (`max_features_per_estimator = 680/500`, 8 estimators), and TabPFN-3 subsamples **round-robin so
   no feature is excluded from the ensemble**. The 2048-dim CheMeleon vector needs **no** reduction —
   which is what the winning RWTH protocol did ("no dimensionality reduction"), against the
   organisers' PCA-256. Both are defensible; the head-to-head has not been run (A1).

---

## 11. Bibliography

**The decisive head-to-heads**
- Monroe — [arXiv 2608.18982](https://arxiv.org/abs/2608.18982), 19 Aug 2026, `github.com/blazejba/monroe`, CC-BY-4.0. Tables 1, 2, 4, 6, 7.
- TFMs for molecular properties (RWTH) — [arXiv 2604.16123](https://arxiv.org/abs/2604.16123) v2, 20 Apr 2026, `git.rwth-aachen.de/avt-svt/public/tabpfn-molprop`. Tables 1, S1, S2, Fig. S3.
- Benchmarking pretrained molecular embeddings — [arXiv 2508.06199](https://arxiv.org/abs/2508.06199) v4, Praski/Adamczyk/Czech, `github.com/scikit-fingerprints/benchmarking_molecular_models`. Tables 1–4, 6.
- CheMeleon — [arXiv 2506.15792](https://arxiv.org/abs/2506.15792); weights `doi.org/10.5281/ZENODO.15426600`.
- MolPAIR — [arXiv 2609.38744](https://arxiv.org/html/2609.38744), 30 Sep 2026, `github.com/nums-ai/MolPAIR`.

**Tabular foundation models**
- TabPFN-3 Technical Report — [arXiv 2605.13986](https://arxiv.org/abs/2605.13986), 12 May 2026; licence `huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE`; docs `docs.priorlabs.ai`.
- TabPFN-2.5 — [arXiv 2511.08667](https://arxiv.org/pdf/2511.08667).
- TabPFN v2 (Nature) — [s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6).
- TabICL — [arXiv 2502.05564](https://huggingface.co/papers/2502.05564), ICML 2025, [PMLR v267](https://proceedings.mlr.press/v267/qu25d.html), docs `tabicl.readthedocs.io`.
- TabDPT — [arXiv 2410.18164](https://arxiv.org/abs/2410.18164); TabDPT-Turbo [arXiv 2608.01400](https://arxiv.org/pdf/2608.01400).
- ScoringBench — [arXiv 2603.29928](https://arxiv.org/abs/2603.29928), 31 Mar 2026.
- PFN-Boost / LLM-Boost — [arXiv 2502.02672](https://www.alphaxiv.org/abs/2502.02672).
- TabPFN regression API — `docs.priorlabs.ai/capabilities/regression`.

**The challenge's own materials**
- *Are tabular foundation models all you need?* — [openadmet.ghost.io](https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/).
- `OpenADMET/optimus-prime` [PR #21](https://github.com/OpenADMET/optimus-prime/pull/21) — the two-stage log2FC recipe, full hyperparameters.
- CheMeleon CYP baseline card — [huggingface.co/openadmet/cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline](https://huggingface.co/openadmet/cyp1a2-cyp2d6-cyp3a4-cyp3c9-chemeleon-baseline) (multitask CheMeleon on ChEMBL pIC50; the card itself warns "the model performance is quite poor").

**Censored and interval labels**
- Svensson et al., *Enhancing uncertainty quantification in drug discovery with censored regression labels* — Artif. Intell. Life Sci. 7 (2025) 100128, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667318525000042), [full text](https://research.chalmers.se/publication/545243/file/545243_Fulltext.pdf); data `github.com/MolecularAI/`.
- Tabular Foundation Models Can Do Survival Analysis — [arXiv 2601.22259](https://arxiv.org/html/2601.22259v2) v2, 18 Jul 2026.
- Staying Alive (TabSA-BJ) — [arXiv 2606.03689](https://arxiv.org/html/2606.03689v1), 2 Jun 2026.
- Censoring-aware target interface — [arXiv 2607.09577](https://arxiv.org/abs/2607.09577); SurvPFN [arXiv 2606.04564](https://arxiv.org/html/2606.04564v2).
- L2Boost-CUT / L2Boost-IMP — [arXiv 2601.17973](https://arxiv.org/abs/2601.17973), ICLR 2025.
- XGBoost AFT — [arXiv 2006.04920](https://arxiv.org/abs/2006.04920), [docs](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html).
- hERG QSAR with qualified data — [J Comput Aided Mol Des 2022](https://doi.org/10.1007/s10822-022-00483-0).

**Pairwise / delta**
- DeepDelta — [J Cheminform 15, 101 (2023)](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00769-x), [PMC10605784](https://pmc.ncbi.nlm.nih.gov/articles/PMC10605784/), `github.com/RekerLab/DeepDelta`.
- Fralish & Reker, *Pairwise learning for molecular property prediction and optimization* — [Front. Drug Discov. 2026](https://www.frontiersin.org/journals/drug-discovery/articles/10.3389/fddsv.2026.1859068/full), 17 Jun 2026.
- SQRL — [arXiv 2501.09103](https://arxiv.org/pdf/2501.09103).

**Chemprop and CYP**
- Chemprop v2 — [JCIM 2026, 10.1021/acs.jcim.5c02332](https://pubs.acs.org/doi/10.1021/acs.jcim.5c02332); v1 [JCIM 2024](https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01250).
- Advancing ADMET prediction for major CYP450 isoforms — [BioMed Eng OnLine 2025](https://link.springer.com/article/10.1186/s12938-025-01412-6).
- MTGL-ADMET (multitask, adaptive auxiliary task selection) — [PMC10654589](https://pmc.ncbi.nlm.nih.gov/articles/PMC10654589/).
- CYP2D6 modelling / substrate-dependent inhibition — [PMC2268062](https://pmc.ncbi.nlm.nih.gov/articles/PMC2268062/); [Understanding CYP2D6 interactions](https://www.sciencedirect.com/science/article/abs/pii/S1359644609002311).
- Computational prediction of CYP inhibitors/inducers — [Molecules 27:5875](https://doi.org/10.3390/molecules27185875); QSAR review [Molecules 17:9283](https://doi.org/10.3390/molecules17089283).
- MoleculeACE — [JCIM 2022, 10.1021/acs.jcim.2c01073](https://pubs.acs.org/doi/10.1021/acs.jcim.2c01073).
- GLACIER — [arXiv 2606.11382](https://arxiv.org/abs/2606.11382), KDD 2026, `github.com/eemokey/glacier`.

---

## 12. Provenance

Screened 2026-10-08. Every number marked **[verified]** was read in the source (PDFs extracted
locally with `pdftotext -layout`: arXiv 2604.16123 v2, 2508.06199 v4, 2605.13986, 2608.18982, and
the Chalmers full text of AI in the Life Sciences 7:100128). **[reported]** means quoted from an
abstract, blog or search result I read but whose table I could not open. **[assumed]** is my
inference and is never load-bearing for a Tier-A item.

In-tree facts checked directly against the repo rather than taken from `PROGRESS.md`:
`scripts/cyp_incontext.py` (the featuriser, the reduction, the ≤500 budget and the "Not TabPFN"
docstring), `scripts/chemeleon_fp.py` and `drugrx/features.py` (CheMeleon is extracted but unwired),
and `drugrx/cyp/*.py` (no CheMeleon reference). [verified]

Not resolved by this screen, and worth a follow-up: per-dataset effect sizes for PFN-Boost; whether
CARTE has any molecular result at all; GLACIER against a PFN head; and whether a left-censored
adaptation of TabSA-BJ actually recovers the screen information better than the log2FC feature column
(A2 vs B1 is currently an **[assumed]** ordering).

The expected-gain bands in §9 are **[assumed]** translations from RMSE/MAE/win-rate onto MA-ST-RAE.
They are anchored on two real numbers — the organisers' own 0.83 → 0.68 move from adding a TFM head
and the log2FC feature, and our shipped 0.8663 — but ST-RAE is an interval score and §3.4's
ScoringBench result says point-metric gains need not transfer one-for-one. Treat the bands as
ordering, not as forecasts.
