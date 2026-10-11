# 21 — Foundation models, LLMs and tabular FMs for CYP pIC50 regression

Research read 2026-10-08. Claims are marked **[verified]** (read in the paper, model card, repo or
report cited) or **[assumed]** (my inference, not read). Benchmark numbers are transcribed from the
source tables, not paraphrased.

**Question:** which pretrained molecular representation, chemistry foundation model, LLM or tabular
foundation model is most likely to beat count-ECFP4 + RDKit2D + Avalon + ErG on CYP inhibition
**potency regression** at 1–2k labels per isoform, with ~17k single-point readouts and ~30k PubChem
AID 1851 rows available as auxiliary data, on a Colab L4/A100?

**Screen:** 476 unique abstracts with full text screened via the Europe PMC REST API across 12
queries (793 raw hits, 484 unique records), plus ~140 titles/snippets across 14 web searches and the
arXiv listing pages — ≈600 records considered. 22 papers, model cards and repos deep-read. The arXiv
API 429'd this IP after six calls and was abandoned in favour of Europe PMC; nothing was fetched
behind a paywall (ACS and Springer 403s were left alone and the open abstract or the authors' own
PDF used instead).

---

## 0. The answer first — ten lines

1. **Nothing here is a 0.1 MA-ST-RAE move.** Our seed band is 0.0095 macro; the honest ceiling for
   any representation swap on *potency* is 1.7–6% relative error, i.e. **0.015–0.05 MA-ST-RAE**.
2. **#1 — TabPFN(-2.5/2.6) on frozen CheMeleon embeddings.** 100% win rate on 30 MoleculeACE ChEMBL
   potency-regression tasks (≤6k compounds), avg rank 2.10, vs 36.7% for *fine-tuned* CheMeleon. Our
   task is a MoleculeACE task in all but name. **Cost: ~2–4 h on an L4. Expected: +0.02 to −0.05.**
3. **#2 — frozen CheMeleon (or GRASP) embeddings as an extra feature block in the existing MLP
   ensemble.** On 23 cluster-held-out **OpenADMET** endpoints: GRASP 0.374 / CheMeleon 0.383 /
   ECFP4-LightGBM 0.408 / Chemprop 0.440 mean MAE — 6.1% and 8.3% under our baseline class.
   **Cost: ~3 h. Expected: +0.01 to −0.03.** Both MIT/CC0, both trivial on an L4.
4. **#3 — TabPFN's native predictive distribution fed to the existing shrinkage/interval machinery.**
   ST-RAE is an interval-aware absolute-error metric; TabPFN emits calibrated quantiles for free, and
   this programme's measured wins have all come from calibration, not fit. **Cost: ~2 h.**
5. **#4 — activity-cliff-aware loss (ACA) or ECFP-target pretraining**, both cheap retrofits to the
   existing heads. Test set is 10 nearest purchasable analogs of top hits — a cliff generator.
6. **#5 — pretrain-on-predicted-labels (pADME trick)**, 4th on MAE / tied 2nd Pearson at Polaris.
   Ranked last because we **already measured a null** for 49,965 real Veith labels (+0.0004/+0.0031).
7. **Do not LoRA an LLM.** TxGemma-27B-Predict, instruction-tuned on TDC itself, **loses 9 of 9 TDC
   ADMET regression endpoints to MolE**, by 17–43% relative. A 2B on an L4 will be worse.
8. **Do not read the TDC CYP leaderboard as a ranking.** A 2026 audit reproduced only 3 of 22
   top-ranked models and found leakage in MiniMol, GradientBoost and XGBoost. The three that survive
   include **MapLight = ECFP + Avalon + ErG + RDKit2D + CatBoost — our feature set**.
9. **Do not train another D-MPNN.** Chemprop 0.440 vs ECFP4-LightGBM 0.408 on OpenADMET endpoints
   independently reproduces our in-house result.
10. **The CYP2D6 gap is a structural-frontier problem, not a representation problem.** Median error
    rises 87–130% outside the training frontier, and no model family recalls more than 13% of
    activity-cliff pairs. Expect a representation swap to shave the macro, not to fix 2D6.

---

## 1. Sizing the prize before choosing a model

From `PROGRESS.md`: leaderboard **MA-ST-RAE 0.8598**, macro seed band **0.0095**, CYP2D6 at test
ST-RAE 1.4577 against validation 0.9400 and carrying ~74% of the gap [verified, in-repo].

Two numbers from the literature set the scale of what a representation can buy:

- **Potency is where foundation models help least.** In the ASAP-Polaris-OpenADMET Antiviral
  Challenge 2025 — same organisers, same `ST-RAE` family of blinded evaluation — the *potency* track
  was won by MolE and MolGPS, and the organiser-side write-up records the winning margin as
  **7.6% over the best local ML model and 1.7% over the fingerprint baseline** [verified, Inductive
  Bio]. 1.7% of 0.8598 = **0.015 MA-ST-RAE**, i.e. ~1.6× our seed band.
- **ADME/absolute-error endpoints are where they help most.** On 23 cluster-held-out OpenADMET
  endpoints, CheMeleon is 6.1% and GRASP 8.3% under ECFP4-LightGBM in mean MAE [verified, GRASP
  §Results]. 6.1% of 0.8598 = **0.052**.

Our endpoints are potency measured as pIC50 and scored by an absolute-error metric, so the honest
interval is **0.015–0.05 MA-ST-RAE**, centred ~0.025. That is 2–5× the band: detectable in one
paired-seed run, not transformative. Every recommendation below is sized against that, not against
the authors' win rates.

---

## 2. Benchmarks of CYP inhibition specifically

### 2.1 TDC CYP Veith — and why the leaderboard is not evidence

| Leaderboard | Rank 1 | AUPRC | Rank 2 | Rank 3 |
|---|---|---|---|---|
| CYP3A4_Veith | MapLight + GNN | 0.916 ± 0.000 | ContextPred 0.904 | AttrMasking 0.902 |
| CYP2C9_Veith | MapLight + GNN | 0.859 ± 0.001 | ContextPred 0.839 | AttrMasking 0.829 |

[verified, tdcommons.ai]. **MapLight** is ECFP (1024) + Avalon (1024) + ErG (315) + RDKit2D (~210)
→ 2,573-dim → CatBoost [verified]. That is, allowing for the MLP-vs-CatBoost head, **our feature
vector is the reproducible state of the art on CYP inhibition.**

The qualifier matters. *An End-User Audit of Reproducibility, Data Leakage and Overfitting of the
Top-Ranked ADMET Prediction Models in TDC Leaderboards* (JCIM 2026) re-ran the top 3 of all 22 TDC
ADMET leaderboards under one protocol: **only CaliciBoost, MapLight and MapLight + GNN passed every
stage and reproduced their reported performance**; the rest failed on unavailable code,
non-reproducible environments or methodological flaws, and **direct or indirect data leakage was
traced in MiniMol, GradientBoost and XGBoost** [verified, abstract]. The authors then showed with
deliberately overfitted Mol2Vec variants that tuning on the public test set inflates both metric and
rank. **Consequence: MiniMol's reported rank-1 TDC results (Lipophilicity MAE 0.456, AqSolDB 0.741,
mean rank 3.3 across 22) cannot be used as evidence here** [verified, both sources].

Note also that all four TDC CYP leaderboards are **classification (AUPRC)**, not pIC50 regression.
Nothing on them transfers directly to our metric.

### 2.2 Direct CYP modelling studies

*Evaluation of machine learning models for cytochrome P450 3A4, 2D6 and 2C9 inhibition* (J Appl
Toxicol 2024) swept algorithms × representations × splits: **XGBoost and CatBoost on combined
fingerprint + physicochemical descriptors reached AUC 0.92, and deep learning models were generally
inferior at average AUC 0.89 on the same test sets**; data volume and sampling strategy had only a
minor effect [verified, abstract].

*Enhancing CYP3A4 Inhibition Prediction Using a Hybrid GNN-ML Model with Data Augmentation*
(Pharmaceuticals 2026), 23,713 compounds, predicting CYP3A4 inhibition **as a continuous %**: weighted
ML ensemble (LightGBM/XGBoost/CatBoost) **RMSE 19.1031, PCC 0.7566**; best GNN (O-GNN + contrastive +
manifold mixup) **RMSE 20.1002, PCC 0.7265**; hybrid of both **RMSE 19.0784, PCC 0.7570** [verified,
abstract]. At 24k labels — 5× our per-isoform count — the GNN still loses to the vector models, and
blending them buys **0.0025 RMSE (0.013%)**. This is the single most directly comparable published
experiment to ours, and its answer is "the ensemble of the two is not worth the second model".

### 2.3 MoleculeACE — the closest analogue to our task

MoleculeACE (van Tilborg, Alenicheva & Grisoni, JCIM 2022) is 30 ChEMBL targets, >35,000 molecules,
**pKi/pIC50 regression**, each task a few hundred to ~6,000 compounds, scored overall and on
activity-cliff compounds. Original finding: **SVM on ECFP was the best overall method**, and deep
learning was conspicuously poor on cliff compounds [verified]. The J Cheminform 2026 re-analysis
(§2.4) reproduces the shape: KPGT R² 0.678 overall but **0.46 on cliff molecules against 0.51 for
XGBoost and MorganCount**, and **every model recalled <13% of activity-cliff pairs** (MolMCL 0.130,
KPGT 0.128) [verified].

Our 750-compound test set is the 10 nearest purchasable analogs of each top-25 hit. That is a
deliberately constructed activity-cliff set. **[assumed]** This is why 2D6 inverts between validation
and test, and no representation in this document claims to fix it.

### 2.4 The 2026 ADMET-under-real-world-challenges benchmark

*Revisiting ADMET prediction reliability under real-world challenges in the foundation model era*
(J Cheminform 18:95, 2026; open access on PMC) benchmarked molecular FMs (**KPGT, Uni-Mol**),
tabular FMs (**TabPFNv2**) and classical baselines over 13 datasets under four stress tests:
few-shot/OOD, class imbalance, bRo5 space and activity cliffs [verified, PMC13371653].

Headline: **in few-shot and OOD scenarios TabPFNv2's generalisation "often outperforms" the
customised molecular foundation models**; KPGT only wins where there is plenty of data and genuinely
unusual topology (cyclic peptides, R² 0.627 vs GNNAK 0.605); and on small linear peptides TabPFNv2
ties AutoGluon at ROC AUC 0.972 against XGBoost 0.970 [verified]. TabPFNv2 was fed **RDKit 2D
descriptors** and the paper records a hard constraint: **"fewer than 500 dimensions" required**,
which prevented high-dimensional feature sets [verified]. (TabPFN-2.5 lifts this to 2,000 — §5.)

This is the paper that most directly answers our question, and its answer is: **the tabular
foundation model, not the molecular one, is the thing that wins in our regime.**

### 2.5 Polaris: the same organisers, one year earlier

The ASAP-Polaris-OpenADMET Antiviral ADMET Challenge 2025, 39 submissions, is the best available
dress rehearsal. The Inductive Bio post-mortem gives a clean ablation ladder against their winning
entry [verified]:

| Entry | Rank | Error vs winner |
|---|---|---|
| Inductive (external ADMET data + their models) | 1 | — |
| Best pretrained DL, **no** extra ADMET data (MolMCL) | 5 | **+23%** |
| MolE (supervised pretraining) | 10 | **+37%** |
| Traditional ML, no extra data | 12 | **+41%** |
| Descriptor baseline | — | **+53%** |
| Fingerprint baseline | — | **+60%** |
| Fine-tuned LLM (Gemini) | — | statistical tie with best local ML |

"8 of the top 10 used additional ADMET datasets; only 4 of 11th–20th did." And the verdict in the
authors' words: **"Large pre-trained models are starting to show benefit, but not in a clear and
consistent way"**; MolGPS had higher error *despite* using external ADMET data [verified].

The podium itself is worth reading as a menu of what wins a blinded OpenADMET challenge:

- **1st — more ADMET data** (proprietary).
- **2nd — multitask Chemprop D-MPNN on >55 curated public tasks**, public data only [verified].
- **3rd — descriptor-first + TabPFN.** Chupakhin & DiBella, JCIM 2026: mechanistic + physicochemical
  + fragment + site-of-metabolism features, then **"a pretrained foundation model for tabular data
  (TabPFN), used in single-task regression, consistently outperformed or matched a strong gradient
  boosting baseline (CatBoost), yielding up to 44% mean absolute error reduction across end
  points"** [verified, abstract]. Caveat: their descriptors came from **ADMET Predictor**
  (Simulations Plus, commercial), so the 44% is TabPFN-vs-CatBoost *on those features*, not a claim
  that TabPFN fixes weak features.
- **4th on MAE / tied 2nd on Pearson — pretrain on *predicted* labels.** Pham et al., JCIM 2025:
  multitask GNN pretrained on labels **predicted by industry teams' in-house models** over public
  structures, then fine-tuned on experimental data. Code: `github.com/LongHung-Pham/pADME`
  [verified].

**The one transferable lesson is that data beat architecture by a wide margin.** We have already
tested the cheap version of that and got a null (§7).

---

## 3. Chemistry foundation models with usable weights

### 3.1 The benchmark our earlier notes cited — and what it did not test

*Benchmarking Pretrained Molecular Embedding Models for Molecular Representation Learning*
(arXiv 2508.06199): **25 pretrained models × 25 datasets** (7 MoleculeNet, 18 TDC), **frozen
embeddings** + RF / logistic regression / kNN, AUROC, Bayesian Bradley-Terry with ROPE [0.25, 0.75].
**Only CLAMP was significantly better than count-ECFP**; ChemBERTa, R-MAT and MolBERT were
practically equivalent; the other 21 were significantly worse. ECFP ranked 5th (mean rank 7.52,
79.89% AUROC) against CLAMP's 5.40 / 82.55%. GROVER, Uni-Mol and GraphMVP ranked 16th–17th.
**The suite includes CYP1A2, CYP2C19, CYP2C9, CYP2D6 and CYP3A4** [verified].

That is a strong result and our earlier notes were right to weight it. But it does not close the
question, for four reasons, all checkable against the paper:

1. **It is classification AUROC**, not potency regression under an absolute-error metric.
2. **Frozen embeddings + shallow heads only** — no fine-tuning, and no tabular FM as the head.
3. **CheMeleon is not in it** (CheMeleon posted June 2025; the benchmark August 2025), nor GRASP,
   nor MiniMol, nor MolE.
4. CLAMP, the one winner, is itself "a fusion of molecular fingerprints" with a shallow MLP — which
   is the paper's own point, not a counterexample to it.

So the correct updated statement is: **structure-only self-supervised encoders do not beat
count-ECFP; descriptor-target pretraining and tabular in-context heads were never tested.** Both of
the untested things are the two recommendations in §0.

### 3.2 Candidate table

L4 column assumes 24 GB and our ~4.9k labelled + ~17k screened compounds.

| Model | Weights | Licence | Params / dim | Input | How to use | ADMET/CYP evidence | L4 |
|---|---|---|---|---|---|---|---|
| **CheMeleon** | `doi.org/10.5281/ZENODO.15426600`; in Chemprop ≥2.2.0 | **MIT** (code), **CC0** (paper/weights) | ~10M, **2048-dim** | SMILES → D-MPNN | `CheMeleonFingerprint` class for frozen vectors, or `chemprop --from-foundation CheMeleon` | 58 tasks: **75% win on Polaris, 97% on MoleculeACE**, vs RF 68%/50%, Chemprop 32%. **0.383 mean MAE on 23 OpenADMET endpoints vs 0.408 ECFP4-LightGBM** | **Trivial.** Pretrained on 8× 2080 Ti; authors fine-tuned on a laptop Quadro RTX 4000 |
| **GRASP** | HF `caithmac/GRASP` | **MIT** | 93.5M, **768-dim** | SMILES → graph transformer, ≤511 atoms | `GRASPEncoder.from_pretrained(...).encode([...])`, or `python -m grasp finetune` | **0.374 mean MAE on 23 OpenADMET endpoints** (best of four); beats CheMeleon on 13/22 TDC ADMET | **Yes**, repo documents Colab GPU |
| **TabPFN-2.5 / 2.6** | `pip install tabpfn` (auto-download) | **Non-commercial** (TabPFN-2.5 Licence v1.0); **TabPFN-2 is Prior Labs Licence = Apache-2.0 + attribution** | 50k rows × **2,000 features** | tabular | `TabPFNRegressor().fit(X,y).predict(X')`. **No fine-tuning** — inference engine only | See §5. **100% win on 30 MoleculeACE tasks with CheMeleon features**; 19/22 vs XGBoost on TDC ADMET | **Yes.** 8 GB VRAM works, 16 GB for large sets; chunk test sets at 1,000 rows |
| **MiniMol** | `pip install minimol` | **MIT** | 10M, **512-dim** | SMILES → graph | `Minimol()(smiles_list)` → 512-d tensors | Claims 22-task TDC mean rank 3.3 — **leakage traced by the 2026 audit; discount** | Trivial |
| **MolE** | `github.com/rolayoalarcon/MolE` | Open repo (**[assumed]** permissive; licence not read) | GIN-based | SMILES → graph | `model.pth` into `ckpt/...` | Beats TxGemma-27B on **9/9** TDC ADMET regression endpoints (§4); 10th at Polaris, +37% error | Trivial |
| **CLAMP** | `github.com/ml-jku/clamp` | Open repo (**[assumed]**) | fingerprint + text encoders | SMILES + assay text | pretrained checkpoint in repo | **Only model of 25 to significantly beat count-ECFP** on 25 datasets incl. 5 CYP panels; 50× enrichment zero-shot vs KV-PLM | Trivial |
| **Uni-Mol / Uni-Mol2** | `github.com/deepmodeling/Uni-Mol`, `unimol_tools` | **MIT** | 84M–1.1B, **768-dim** (42/84/164M) | 3D conformers | `unimol_tools` representation API | 16th–17th of 25 on the frozen-embedding benchmark; loses to TabPFNv2 few-shot/OOD in J Cheminform 2026 | Embedding yes; 1.1B fine-tune no |
| **KPGT** | figshare `s/d488f30c23946cf6898f` | **Apache-2.0** | graph transformer | SMILES → graph + descriptors | drop into `KPGT/models/` | Best FM in J Cheminform 2026 **only** on bRo5 cyclic peptides with ample data; **R² falls 0.678 → 0.46 on cliffs** | Yes |
| **SMI-TED** | HF `ibm-research/materials.smi-ted` | **Apache-2.0** | 289M / 8×289M | SMILES | `load_smi_ted(...)`, `model.encode(df.SMILES)` | Strong on MoleculeNet classification; in the TabPFN study it is beaten by CheMeleon features | Yes (289M) |
| **MolFormer-XL** | HF `ibm-research/MoLFormer-XL-both-10pct` | **Apache-2.0** | 46.8M | SMILES | `AutoModel.from_pretrained(..., trust_remote_code=True)` → `pooler_output` | 9th of 25 (79.80% AUROC), "relatively easy to set up"; CheMeleon beats it 32%→75% Polaris, 10%→97% MoleculeACE | Trivial |
| **ChemBERTa-2 / -3** | HF `DeepChem/ChemBERTa-*`; ChemBERTa-3 framework | **MIT** (DeepChem) | 5M–77M | SMILES | HF encoder → CLS/mean pool | Practically equivalent to ECFP in the 25×25 study. **MTR pretraining beat MLM by ~9% AUROC at 77M** | Trivial |
| **GROVER / GraphMVP / MolCLR / Graphormer** | public repos | mixed | — | graph | — | 16th–24th of 25; MolCLR 21% Polaris win vs CheMeleon 79% | — |
| **Contrastive KERMT** | NGC `nvidia/clara/kermt-contrastive` | NVIDIA NGC terms | graph transformer | graph | NIM/NGC | +7.6% Biogen, +9.9% ExpansionRX, +9.5% ChEMBL-MT over KERMT baseline | **[assumed]** yes |

### 3.3 Why CheMeleon is the one to try

CheMeleon's pretraining target is **1,613 Mordred descriptors over ~1M PubChem molecules**
[verified] — it is a *learned, smoothed, 2048-dimensional descriptor set*, not a structure-only
self-supervised encoder. That is exactly the class the 25×25 benchmark never tested, and it is the
only published representation that beats ECFP-class baselines on **OpenADMET's own endpoints** under
cluster-held-out splits. It is also the representation that the TabPFN study found works best as
TabPFN input (§5). MIT/CC0, ~10M params, and the authors fine-tuned it on a laptop GPU.

---

## 4. LLMs for molecular property prediction — settled, negatively

### 4.1 TxGemma vs MolE, transcribed from the TxGemma report, Table 1

TxGemma (Google DeepMind, Apr 2025) is 2B/9B/27B fine-tuned from Gemma-2 on ~7M TDC examples
(3.3M regression/generation, 3.7M binary) for 67B tokens on **256 TPUv4 chips** [verified]. Its
training mixture **includes the TDC CYP tasks**. Regression is emitted as **"an integer between 0 and
1000", a binned transformation of the label** [verified].

| Task | Metric | MolE | TxGemma-27B-Predict | Δ |
|---|---|---|---|---|
| Caco2 Wang | MAE ↓ | **0.329** | 0.401 | +21.9% worse |
| Lipophilicity AZ | MAE ↓ | **0.406** | 0.538 | +32.5% worse |
| Solubility AqSolDB | MAE ↓ | **0.776** | 0.907 | +16.9% worse |
| PPBR AZ | MAE ↓ | **7.229** | 9.048 | +25.2% worse |
| LD50 Zhu | MAE ↓ | **0.602** | 0.627 | +4.2% worse |
| VDss Lombardo | Spearman ↑ | **0.644** | 0.559 | −13.2% rel |
| Half Life Obach | Spearman ↑ | **0.578** | 0.458 | −20.8% rel |
| Clearance Microsome AZ | Spearman ↑ | **0.632** | 0.462 | −26.9% rel |
| Clearance Hepatocyte AZ | Spearman ↑ | **0.456** | 0.260 | −43.0% rel |
| CYP3A4 Veith | AUPRC ↑ | **0.876** | 0.854 | loses |
| CYP2C9 Veith | AUPRC ↑ | 0.782 | **0.798** | wins |
| CYP2D6 Veith | AUPRC ↑ | 0.679 | **0.683** | ties |

**9 of 9 ADMET regression endpoints lost, by 4–43%.** The classification picture is mixed (it wins
hERG, DILI, Pgp, HIA, bioavailability). The report's own sentence: **"Regression tasks experience the
greatest performance decline from the general-purpose training"** [verified]. The 2B is reported as
weaker still than the 9B and 27B.

**That is the whole answer to question 3.** A 27B model, instruction-tuned on *this exact benchmark
family* with 256 TPUs, loses every ADMET regression endpoint to MolE — a 2022 graph foundation model
that itself only reached 10th at Polaris. A QLoRA-tuned 2B–9B on 2,000 of our labels cannot plausibly
beat our current MLP ensemble. The Polaris result agrees independently: a fine-tuned Gemini was a
**statistical tie with the best local ML model** [verified].

### 4.2 In-context learning is worse than that

*In-Context Molecular Property Prediction with LLMs: A Blinding Study* (arXiv 2603.25857) tested nine
models (GPT-4.1 nano/mini/full, GPT-5 nano/mini/full, Gemini 2.5 flash-lite/flash/pro) at 0/60/1000
shot on ESOL, Lipophilicity, QM7 [verified]:

- **60-shot underperformed 0-shot in 19 of 27 model × dataset cases (sign test p = 0.026)** — prior
  knowledge overwhelms sparse in-context examples.
- Under full blinding, the LLMs were **never clearly better than ridge regression on bag-of-characters
  SMILES features** fitted to the same in-context examples.
- No verbatim memorisation (3-sig-fig match ~0.16–0.33%, at chance), so the finding is about
  capability, not leakage. Lipophilicity degraded 72% → 35% correlation under blinding, i.e. its
  apparent skill was pretrained knowledge, not in-context learning.

Our test set is blinded Enamine chemistry with almost no public measured data. This is the regime the
paper says LLM ICL fails in.

### 4.3 Hidden states as features, and the rest of the family

No paper found in this screen reports LLM hidden states beating count-ECFP on an ADMET regression
endpoint. The nearest evidence is *Regression with Large Language Models for Materials and Molecular
Property Prediction* (arXiv 2409.06080): fine-tuned LLaMA-3 on SMILES is "competitive with, although
typically somewhat worse than, conventional random forest and fully-connected DNN models", and
**significantly underperforms SOTA DNNs on QM9** [verified]. *Benchmarking Large Language Models for
Molecule Prediction Tasks* (arXiv 2403.05075): "LLMs are not potent experts for molecule prediction
tasks … consistently underperform compared to machine learning models" [verified]. Tx-LLM is
TxGemma's weaker predecessor (TxGemma-27B beats Tx-LLM M on 45/66). LlaSMol-Mistral 7B is beaten by
TxGemma-27B on 4 of 5 shared tasks. ChemDFM, nach0, MolecularGPT and Galactica returned no ADMET
regression result in this screen that competes with a specialist.

### 4.4 Colab-L4 feasibility, for completeness

QLoRA 4-bit on Gemma-2 9B fits a 24 GB L4 at **~18 GB peak VRAM** (rank-16 adapters on q/k/v/o,
NF4 + double quant, gradient checkpointing, batch 4 × grad-accum 8) [verified, practitioner guides].
So it is **feasible and still not worth doing.** TxGemma licence is Google's *Health AI Developer
Foundations* terms, not Apache-2.0 — another reason not to build a product path on it.

---

## 5. Tabular foundation models — the live option

### 5.1 The evidence, strongest first

**(a) TabPFN on frozen CheMeleon embeddings, on 30 MoleculeACE potency-regression tasks.**
*Tabular foundation models for in-context prediction of molecular properties* (arXiv 2604.16123),
TabPFN-2.6 and TabICLv2 [verified]:

| Configuration | MoleculeACE (30) win rate | Polaris + MoleculeACE (58): wins / rate / avg rank |
|---|---|---|
| **TabPFN–CheMeleonFP** | **100.0%** (avg rank 2.10) | 50 / **86.2%** / 4.52 |
| TabICL–CheMeleonFP | — | 43 / 74.1% / 6.31 |
| TabPFN–Mordred | — | 39 / 67.2% / 5.71 |
| TabPFN–RDKit2d | — | 33 / 56.9% / 5.31 |
| TabPFN–Morgan | — | — / **22.4%** / — |
| **CheMeleon, fine-tuned** | **36.7%** (avg rank 7.53) | 24 / 41.4% / 8.84 |

Three things to carry out of that table:

1. **Frozen + TabPFN beats fine-tuned, by a lot** (86.2% vs 41.4%), and is **18.3–46.0× faster on
   GPU, 4.8–27.3× on CPU** [verified]. Do not fine-tune CheMeleon.
2. **The featuriser matters more than the tabular model**: TabPFN–Morgan 22.4% vs TabPFN–CheMeleon
   86.2%. Feeding TabPFN our existing count-ECFP would be the *worst* of the configurations tested.
3. **MoleculeACE is ChEMBL potency regression at ≤6,000 samples per task.** Our task is that task.

Protocol, verbatim where it matters: **no dimensionality reduction** on the 2048-dim CheMeleon
vectors; standard-scale with train-set statistics; **clamp features to [−6, 6]**; out-of-the-box
TabPFN-2.6 with **no hyperparameter tuning** [verified]. Runtime case study on an H100 against
fine-tuned CheMeleon: n_train 111 → 46.0×, 728 → ~30×, 5,907 → 18.3×. Memory becomes the binding
constraint only at ~21,000 samples with high-dimensional embeddings (~50 GB per fold with disk
offload); **compact RDKit2d "continued to run without issues"** [verified].

**(b) TabPFNv2 wins in few-shot and OOD specifically.** J Cheminform 18:95 (§2.4): across 13
datasets, TabPFNv2 "ranked first across multiple few-shot datasets" and its generalisation "often
outperforms" KPGT and Uni-Mol in few-shot/OOD, while end-to-end deep models "showed marked
deterioration as sample sizes decreased" [verified].

**(c) TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery** (JCIM 2026):
"TabPFN performs **on par with XGBoost in classification but demonstrates clear and stable advantages
in regression**, with its strongest gains on small and medium data sets and under OOD evaluations";
feature and data ablations from 10–90% show it "degrades gracefully … minimal sensitivity compared
with tree ensembles"; it loses to tree ensembles on the larger QM8 [verified, abstract].

**(d) TDC ADMET, 217 RDKit descriptors, TabPFNv2** (Ipas et al., BSB 2025, 2-page abstract):
**beats XGBoost on 19 of 22 datasets, surpasses MiniMol on 9 of 22, top rank on 3 tasks**, "despite
not relying on any prior, drug-specific knowledge" [verified, read in full].

**(e) Polaris 3rd place** (§2.5): TabPFN "consistently outperformed or matched" CatBoost, **up to 44%
MAE reduction**, "while simplifying deployment by eliminating an extensive hyperparameter search"
[verified].

### 5.2 Constraints you will hit

| Constraint | Value | Source |
|---|---|---|
| Rows × features, TabPFN-2.5/2.6 | **50,000 × 2,000** (20× v2) | PriorLabs release [verified] |
| Rows × features, TabPFN**v2** | ~10,000 × **<500** | J Cheminform 2026 records the <500 limit [verified] |
| Fine-tuning | **Not supported** — inference engine only | PriorLabs repo [verified] |
| Test-set batching | split into chunks of **1,000 samples** | PriorLabs repo [verified] |
| VRAM | 8 GB works, 16 GB for large sets | PriorLabs repo [verified] |
| Licence | **2.5/2.6/3/3.5 weights are non-commercial**; **TabPFN-2 = Apache-2.0 + attribution** | PriorLabs repo [verified] |

**Our feature vector is ~3,600-dim** (count-ECFP4 2048 + Avalon 1024 + RDKit2D ~210 + ErG 315) and
**exceeds TabPFN-2.5's 2,000-feature ceiling.** Options, in order of what the literature supports:
CheMeleon 2048 alone (the configuration that won); RDKit2D + Mordred (the "compact" configuration,
67.2%/56.9% win rates, and the one that scales); or a supervised feature selection down to ≤2,000.
Do **not** feed raw count-ECFP alone — that is the 22.4% configuration.

**The licence is the real blocker for anything beyond the challenge.** TabPFN-2.5/2.6 weights are
non-commercial. A challenge submission is fine; a DrugRx `ModelImpl` served to paying users is not,
without a commercial licence from Prior Labs, or falling back to TabPFN-2 (Apache-2.0 + attribution)
with the <500-feature limit.

### 5.3 The metric-fit argument

ST-RAE charges L1 error **outside** per-compound credible intervals and normalises by the mean
predictor. This programme's measured leaderboard movement has come from shrinkage and calibration,
not from fit (`PROGRESS.md`: parabola optimum at a ≈ 1.252 → predicted ST-RAE ≈ 0.8611; shrinking to
0.60 cost +0.0865). **TabPFN returns a full predictive distribution natively** [verified], so its
quantiles can feed the existing shrinkage machinery directly instead of being reconstructed from an
ensemble spread. **[assumed]** this is worth as much as the point-estimate gain, and it is the
cheapest thing in this document to test.

### 5.4 Others

TabICLv2 is close behind TabPFN (74.1% vs 86.2% with CheMeleon features) and note 13 already records
it at a 47% Wilcoxon win rate against Mol-JEPA's 38% on 23 ADMET endpoints. CARTE returned nothing on
molecular regression in this screen. LimiX-2M (arXiv 2606.04485) exists but has no chemistry result.
Real-TabPFN (arXiv 2507.03971) continues TabPFN pretraining on real-world tables — **[assumed]**
irrelevant to us, since it is not chemistry-specific.

---

## 6. Ensembling and stacking

Direct evidence is thinner than the question deserves, and what exists is mostly "yes, but small":

- **Concatenating ECFP with other descriptors consistently improves AUROC and AUPRC over either
  alone** [verified, arXiv 2605.10722]; the same paper's main result is that **GNNs pretrained to
  predict ECFP significantly outperform both from-scratch GNNs and classical fingerprinting on a
  diverse set of ADMET tasks** — a cheap retrofit that needs no external weights.
- The CYP3A4 hybrid study (§2.2) is the cleanest ML+GNN stack on our exact enzyme at 5× our data, and
  it bought **0.0025 RMSE on 19.10 (0.013%)** over the ML ensemble alone [verified]. That is the
  realistic size of a "blend the two model families" gain.
- MapLight **+ GNN** outranks plain MapLight on CYP2C9/2D6/3A4 Veith [verified, TDC], so a GNN blend
  does add something on CYP classification — but both are in the audit's reproducible set and the
  margin is not broken out.
- The TabPFN study **did not test** concatenations or TFM+GBM ensembles, and explicitly flags them as
  future work that "could yield further gains, mostly at the cost of extra compute" [verified].
- CheMeleon tested a `Chemprop-Mordred` concatenation baseline but did not report an
  ECFP-concatenation ablation [verified].

**Practical read:** adding a CheMeleon/GRASP block to the existing 2,573-dim vector is the
*concatenation* experiment, and it is better supported than a late-fusion ensemble of two separately
trained models. **[assumed]** Expected 0.01–0.03 MA-ST-RAE, strictly less than the TabPFN swap,
because the head stays the same.

---

## 7. What NOT to do

1. **Do not LoRA or fine-tune an LLM (TxGemma, Tx-LLM, LlaSMol, ChemDFM, nach0, MolecularGPT) on
   ~2k labels.** TxGemma-27B loses 9/9 ADMET regression endpoints to MolE (§4.1). Feasible on an L4,
   still wrong.
2. **Do not use LLM in-context learning for the blinded set.** 60-shot was worse than 0-shot in 19/27
   cases, and blinded LLMs did not beat ridge-on-characters (§4.2).
3. **Do not read TDC CYP leaderboard positions as a model ranking.** Only CaliciBoost, MapLight and
   MapLight+GNN reproduced; **MiniMol, GradientBoost and XGBoost had traced leakage** (§2.1). In
   particular, do not adopt MiniMol on the strength of its TDC ranks.
4. **Do not train another D-MPNN/Chemprop from scratch.** Chemprop is **last** of four on the 23
   OpenADMET endpoints (0.440 vs ECFP4-LightGBM 0.408) and 32% vs CheMeleon's 75% on Polaris. This
   independently reproduces our in-house result; stop re-litigating it.
5. **Do not fine-tune CheMeleon end-to-end.** Frozen + TabPFN wins 86.2% vs 41.4% and is 18–46×
   faster on GPU (§5.1).
6. **Do not feed raw count-ECFP to TabPFN.** TabPFN–Morgan is the worst configuration tested
   (22.4%), and 3,600 features exceeds the 2,000 ceiling anyway.
7. **Do not spend the budget on more auxiliary data first.** Polaris says data beats architecture,
   but we have already measured our version of it: **49,965 real PubChem Veith CYP labels moved
   MA-ST-RAE by 0.0004 and 0.0031 across two seeds against a band of ~0.0095–0.03** (`PROGRESS.md`,
   2026-10-05). The Polaris winners' extra data was *proprietary in-house ADME on the target assays*,
   which is not a thing we can obtain.
8. **Do not expect any of this to fix CYP2D6.** The gap is structural-frontier: median error rises
   **87.0%** (equally weighted) to **130.3%** (skew-adjusted, bootstrap 52.1–246.0%) outside the
   training frontier [verified, arXiv 2607.10729], and every model family recalls **<13%** of
   activity-cliff pairs (§2.3). Representation buys macro, not 2D6.
9. **Do not ship TabPFN-2.5/2.6 weights in Talindrew.** Non-commercial licence (§5.2).
10. **Do not chase CLAMP's headline.** Its win is zero-shot/classification AUROC on frozen
    embeddings, it is fingerprint-based internally, and nothing in this screen shows it doing
    potency regression better than ECFP.

---

## 8. The experiment to run, in order

Each is sized against the 0.0095 macro seed band and stated as a paired-seed comparison.

1. **CheMeleon featurisation** (≈1 h L4). `pip install 'chemprop>=2.2.0'`, pull weights from Zenodo,
   `CheMeleonFingerprint` over all ~22k structures (4.9k labelled + 17k screened + test). Cache.
   No model change yet — this is pure infrastructure and is reused by steps 2–4.
2. **TabPFN-2.6 on frozen CheMeleon, per isoform** (≈2 h). Standard-scale, clamp to [−6, 6], no
   tuning, chunk prediction at 1,000 rows. Compare MA-ST-RAE against the current ensemble on the
   existing cluster validation, paired across the same seeds. **Expected 0.015–0.05; decision rule
   |Δ| > 0.02 macro, reported per endpoint.**
3. **TabPFN quantiles → the existing shrinkage fit** (≈2 h). Replace the ensemble-spread interval
   with TabPFN's native predictive distribution and refit the per-endpoint `(shift, scale)`.
   **[assumed]** this is where the metric-specific points are.
4. **CheMeleon block concatenated into the current MLP ensemble** (≈3 h), as the fallback if (2)
   fails — it keeps the interval-hinge loss, which TabPFN cannot use. **Expected 0.01–0.03.**
5. **Only if 2–4 all fail:** ACA-style activity-cliff triplet loss on the existing head, or
   ECFP-target pretraining of a GNN. Both are retrofits, neither needs external weights.

GRASP is the drop-in alternative to CheMeleon at step 1 (`caithmac/GRASP`, MIT, 768-dim, better mean
MAE on the OpenADMET endpoints) and is worth running as a second featuriser in the same harness if
step 2 shows signal.

---

## 9. Bibliography

**Benchmarks and audits**

- Praski et al., *Benchmarking Pretrained Molecular Embedding Models for Molecular Representation
  Learning*, arXiv:2508.06199 — https://arxiv.org/html/2508.06199v1
- *An End-User Audit of Reproducibility, Data Leakage, and Overfitting of the Top-Ranked ADMET
  Prediction Models in TDC Leaderboards*, JCIM 2026 — https://doi.org/10.1021/acs.jcim.6c00819
  (preprint: https://doi.org/10.64898/2026.02.26.708193)
- *Revisiting ADMET prediction reliability under real-world challenges in the foundation model era*,
  J Cheminform 18:95 (2026) — https://pmc.ncbi.nlm.nih.gov/articles/PMC13371653/
- van Tilborg, Alenicheva & Grisoni, *Exposing the Limitations of Molecular Machine Learning with
  Activity Cliffs*, JCIM 2022 — https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.2c01073
- *Beyond Scaffold Splits: Structural-Frontier Evaluation Reveals Hidden Failures in ADMET Models*,
  arXiv:2607.10729 — https://arxiv.org/pdf/2607.10729
- TDC CYP leaderboards — https://tdcommons.ai/benchmark/admet_group/12cyp3a4i/ ,
  https://tdcommons.ai/benchmark/admet_group/10cyp2c9i/

**CYP-specific**

- *Evaluation of machine learning models for cytochrome P450 3A4, 2D6, and 2C9 inhibition*,
  J Appl Toxicol 2024 — https://doi.org/10.1002/jat.4601
- *Enhancing CYP3A4 Inhibition Prediction Using a Hybrid GNN-ML Model with Data Augmentation*,
  Pharmaceuticals 2026 — https://doi.org/10.3390/ph19020258
- PubChem AID 1851 (Veith qHTS, 17,143 compounds, 5 isoforms, 15-point dose–response) —
  https://pubchem.ncbi.nlm.nih.gov/bioassay/1851
- OpenADMET CYP blind challenge announcement —
  https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/

**Polaris / ASAP-OpenADMET 2025**

- Inductive Bio, *Lessons from the Polaris ADMET competition* —
  https://www.inductive.bio/blog/lessons-from-the-polaris-admet-competition
- Chupakhin & DiBella, *Descriptor-First Approach for ADMET Prediction in the PolarisHub Antiviral
  Challenge* (3rd place, TabPFN), JCIM 2026 — https://doi.org/10.1021/acs.jcim.5c02094
- Pham et al., *Improved ADME Prediction by Multitask Pretraining on Predicted Data*, JCIM 2025 —
  https://doi.org/10.1021/acs.jcim.5c02030 ; code https://github.com/LongHung-Pham/pADME
- *ChemProp multi-task models for predicting ADME properties in the Polaris challenge* (2nd place) —
  https://chemrxiv.org/doi/10.26434/chemrxiv-2025-q12vh

**Chemistry foundation models**

- Burns & Green, *CheMeleon / Descriptor-based Foundation Models for Molecular Property Prediction*,
  arXiv:2506.15792 — https://arxiv.org/html/2506.15792v1 ; code
  https://github.com/JacksonBurns/chemeleon ; weights https://doi.org/10.5281/ZENODO.15426600
- *GRASP: Graph Representation Learning with Assay Supervision for Molecular Properties* (2026) —
  https://doi.org/10.64898/2026.09.28.755019 ; code https://github.com/caithmac/GRASP ; weights
  https://huggingface.co/caithmac/GRASP
- MiniMol — https://github.com/graphcore-research/minimol (MIT, 512-dim), arXiv:2404.14986
- MolE — https://github.com/rolayoalarcon/MolE
- CLAMP — https://github.com/ml-jku/clamp ; https://proceedings.mlr.press/v202/seidl23a/seidl23a.pdf
- Uni-Mol / Uni-Mol2 — https://github.com/deepmodeling/Uni-Mol (MIT); arXiv:2406.14969
- KPGT — https://github.com/lihan97/KPGT (Apache-2.0); weights
  https://figshare.com/s/d488f30c23946cf6898f
- SMI-TED — https://huggingface.co/ibm-research/materials.smi-ted (Apache-2.0);
  https://www.nature.com/articles/s42004-025-01585-0
- MolFormer-XL — https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct (Apache-2.0)
- ChemBERTa-3 — https://chemrxiv.org/doi/pdf/10.26434/chemrxiv-2025-4glrl-v2
- *Probabilistic Contrastive Pretraining for Multi-task ADME Property Prediction* (Contrastive
  KERMT, NVIDIA), arXiv:2606.11508 — https://arxiv.org/abs/2606.11508
- *On Improving Graph Neural Networks for QSAR by Pre-training on Extended-Connectivity
  Fingerprints*, arXiv:2605.10722 — https://arxiv.org/html/2605.10722v1
- *Activity-cliff awareness enables robust graph learning for molecular property prediction*,
  Nat Commun 2026 — https://doi.org/10.1038/s41467-026-75713-2

**LLMs**

- Wang, Schmidgall et al., *TxGemma: Efficient and Agentic LLMs for Therapeutics*, arXiv:2504.06196 —
  report PDF https://storage.googleapis.com/research-media/txgemma/txgemma-report.pdf (Table 1 is the
  MolE comparison transcribed in §4.1); model card
  https://developers.google.com/health-ai-developer-foundations/txgemma/model-card
- *In-Context Molecular Property Prediction with LLMs: A Blinding Study on Memorization and Knowledge
  Conflicts*, arXiv:2603.25857 — https://arxiv.org/html/2603.25857
- *Regression with Large Language Models for Materials and Molecular Property Prediction*,
  arXiv:2409.06080 — https://arxiv.org/pdf/2409.06080
- *Benchmarking Large Language Models for Molecule Prediction Tasks*, arXiv:2403.05075 —
  https://arxiv.org/pdf/2403.05075
- MolecularGPT, arXiv:2406.12950 — https://arxiv.org/abs/2406.12950

**Tabular foundation models**

- *Tabular foundation models for in-context prediction of molecular properties*, arXiv:2604.16123 —
  https://arxiv.org/html/2604.16123v1
- Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature
  637:319–326 (2025) — https://doi.org/10.1038/s41586-024-08328-6
- *TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models*, arXiv:2511.08667 —
  https://arxiv.org/pdf/2511.08667
- *TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery*, JCIM 2026 —
  https://doi.org/10.1021/acs.jcim.5c02823
- Ipas et al., *Exploring TabPFNv2 as a Novel Baseline for ADMET Prediction in Drug Discovery*,
  BSB 2025 —
  http://bsb.sbc.org.br/2025/wp-content/uploads/sites/9/2025/09/Exploring-TabPFNv2-as-a-Novel-Baseline-for-ADMET-Prediction-in-Drug-Discovery.pdf
- PriorLabs TabPFN repo (limits, licences, API) — https://github.com/PriorLabs/TabPFN

**In-repo cross-references**

- `research/13-jepa-and-molecular-pretraining.md` — the 25×25 benchmark, Mol-JEPA, and the measured
  null on 49,965 PubChem Veith labels.
- `PROGRESS.md` — leaderboard 0.8598, macro seed band 0.0095, the CYP2D6 validation→test inversion.
