# 13 — JEPA, and self-supervised pretraining for molecular property prediction

Research read 2026-10-05. Every claim is marked **[verified]** (read in the paper, repo or data file
cited) or **[assumed]** (inference, not read). Benchmark numbers are transcribed from the source
tables, not paraphrased.

The question this document was written to answer: **the programme's standing conclusion is that we
are bound by data and representation, so pretraining on a large unlabeled corpus is the most
promising remaining direction — does the literature support that, or refute it?**

---

## 0. The answer first

**The literature refutes the specific plan and supports a different one.**

Refuted: *pretrain an encoder on unlabeled molecular structures, freeze or fine-tune it, beat
ECFP+GBM.* The single most relevant paper — 25 pretrained models × 25 datasets, frozen embeddings,
scaffold splits, **five of the datasets being CYP inhibition panels** — finds that **only one of 25
models is statistically better than a count-ECFP fingerprint, and that one model is itself built on
fingerprints** [verified, Praski et al. 2025, §4.1]. Four independent audits spanning 2021–2026 reach
the same verdict with different methods.

Supported: *pretrain by predicting things you can compute or have already measured, across many
modalities, and use the result as one feature source among several.* Mol-JEPA (Boehringer Ingelheim,
Aug 2026) does exactly this — 14 modalities, 4.66M molecules — and reaches **0.402 mean MAE against
0.559 for RF+ECFP4 and 0.468 for LightGBM+AlvaDesc** on 23 held-out ADMET/potency endpoints
[verified, §5.1]. But its own Wilcoxon analysis gives it a **38% win rate against TabICLv2's 47%**, a
tabular in-context model fed ordinary descriptors — and its leave-one-modality-out analysis finds
**ECFP and MOE descriptors among the most important modalities** [verified, §5.1]. The gain is from
*more kinds of supervision*, not from self-supervision on structure.

**Our own programme has already run the cheaper version of this experiment and got a null.** Adding
49,965 *real labelled* PubChem Veith CYP measurements moved MA-ST-RAE by 0.0004 and 0.0031 across two
seeds, against a bootstrap scale of ~0.03 (`PROGRESS.md`, 2026-10-05). If 50k genuine labels on the
same four enzymes are unmeasurable, unlabeled structure pretraining is a strictly worse bet, because
it must pass through an even longer causal chain to reach the same four numbers.

Expected value of a structure-only JEPA on our 2,500-dim feature vectors: **0.00 ± 0.03 MA-ST-RAE**,
i.e. indistinguishable from noise at our fold resolution, for 10–30 A100-hours. §8 says what to build
instead, and §9 gives the two-hour control that would decide it cheaply if we build it anyway.

---

## 1. I-JEPA — the exact mechanism

**Paper.** Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas, *Self-Supervised
Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023.
<https://arxiv.org/abs/2301.08243>. Code: <https://github.com/facebookresearch/ijepa>.

### 1.1 Three networks

| Component | Symbol | What it sees | How it is trained |
|---|---|---|---|
| **Context encoder** | `f_θ` | the context block only (masked ViT) | gradient descent |
| **Target encoder** | `f_θ̄` | the **whole, unmasked** input | EMA of `θ`, no gradients |
| **Predictor** | `g_φ` | context embeddings + a mask token per target patch | gradient descent |

The asymmetry is the point: the target encoder sees everything and is never optimised; the context
encoder sees a fragment and must be optimised until the predictor can recover what the target encoder
would have said. [verified]

### 1.2 The loss — and the discrepancy between paper and code

The paper's equation is an average L2 in representation space [verified, §3 of the paper]:

```
L  =  (1/M) Σ_{i=1..M}  Σ_{j∈B_i}  ‖ ŝ_y(j) − s_y(j) ‖²₂
```

where `B_i` is the i-th target block, `s_y` the target-encoder patch representations and `ŝ_y` the
predictor's output. `M = 4` target blocks.

**The released implementation differs in two ways that matter more than the equation does**
[verified, `src/train.py` lines 296–313 of `facebookresearch/ijepa@main`]:

```python
def forward_target():
    with torch.no_grad():                      # (a) hard stop-gradient
        h = target_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),))     # (b) LayerNorm over the FEATURE dim
        h = apply_masks(h, masks_pred)
        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
        return h

def loss_fn(z, h):
    loss = F.smooth_l1_loss(z, h)              # (c) Huber, not L2
    return AllReduce.apply(loss)
```

- **(a)** `torch.no_grad()` plus a separate module — not a `detach()` on a shared graph. [verified]
- **(b)** `F.layer_norm` on the *target* removes the per-patch mean and fixes the per-patch norm.
  **This is an anti-collapse mechanism that the paper's loss equation does not contain**: a constant
  target vector is not a fixed point of the objective once the target is re-normalised per sample,
  because LayerNorm of a constant vector is undefined/zero-variance and the gradient path through it
  is cut by `no_grad`. Reimplementations that drop this line are the most common way to get a JEPA
  that trains to a beautifully low loss and has learned nothing. [verified that the line exists;
  **[assumed]** for the causal claim about reimplementations — it is my reading of the mechanism, not
  a quoted result]
- **(c)** `smooth_l1_loss` (Huber, β=1 by default) not squared L2. Bounded gradients on outlier
  patches. [verified]

### 1.3 The EMA update — exact

```python
# momentum schedule: linear from ema[0] to ema[1] over the whole run
momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                      for i in range(int(ipe*num_epochs*ipe_scale)+1))
...
with torch.no_grad():
    m = next(momentum_scheduler)
    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
```

i.e. `θ̄ ← m·θ̄ + (1−m)·θ`, with **`m: 0.996 → 1.000` linearly over the whole run** [verified,
`configs/in1k_vith14_ep300.yaml`: `ema: [0.996, 1.0]`]. Note `m → 1.0` means the target encoder
**freezes completely by the end of training**. That is deliberate: a frozen target is a fixed
regression problem, so late training is pure fitting rather than a moving target.

### 1.4 Masking — exact

From `configs/in1k_vith14_ep300.yaml` [verified]:

| Parameter | Value |
|---|---|
| `num_enc_masks` (context blocks) | 1 |
| `enc_mask_scale` | `[0.85, 1.0]` of the image |
| `num_pred_masks` (target blocks) | 4 |
| `pred_mask_scale` | `[0.15, 0.2]` each |
| `aspect_ratio` (targets) | `[0.75, 1.5]` |
| `allow_overlap` | `false` — overlapping regions are **removed from the context block** |
| `min_keep` | 10 patches |

Two design facts the paper argues for explicitly [verified]: targets must be **large and semantic**
(0.15–0.2 of the image, not single patches), and the context must be **informative and spatially
distributed** (0.85–1.0, nearly the whole image) *minus* the target regions. Predicting a single
masked patch from its immediate neighbours is solvable by low-level interpolation and teaches nothing.

### 1.5 Predictor and optimisation

- Predictor is a **narrow ViT**: embedding dim fixed at **384**; depth 6 (ViT-B/16), **12** (ViT-L/16,
  H/16, H/14), 16 (ViT-G/16). [verified] Mask tokens are **one shared learnable vector + positional
  embedding** per target patch. [verified]
- AdamW; `lr` 2e-4 → 1e-3 over 40 warmup epochs → cosine to 1e-6; weight decay **0.04 → 0.4 linearly**;
  batch 2048 (128 × 16 GPUs); 300 epochs; bfloat16. [verified, same config]
- ViT-H/14 on ImageNet-1k: **16 A100s, under 72 hours**. [verified, abstract]

### 1.6 Why it avoids collapse — what the paper actually claims

The paper's stated mechanism is **architectural asymmetry**, not a regulariser: *"an asymmetric
architecture between the x- and y-encoders to avoid representation collapse"*, which it says *"has
proven essential for training JEAs with Vision Transformers"* [verified]. There is **no contrastive
term, no negative sampling, no variance/covariance penalty, and no handcrafted augmentation** — the
only "augmentation" is the mask itself, and a random resized crop at `crop_scale: [0.3, 1.0]` with
colour jitter, blur and flips all **disabled** (`use_color_distortion: false`,
`use_gaussian_blur: false`, `use_horizontal_flip: false`) [verified, config].

So the load-bearing set is, in my reading: **(i) stop-gradient on the target path, (ii) EMA with
m→1 so the target lags and then freezes, (iii) the asymmetric predictor, (iv) the per-sample
LayerNorm on targets, (v) large non-overlapping targets that make the task non-trivial.** Items (i)
and (iii) are the paper's claim [verified]; (ii), (iv) and (v) are read off the code and config
[verified that they exist, **[assumed]** for the weighting between them].

**This is not a proof.** BYOL-style EMA+predictor collapse-avoidance is an empirical regularity with
contested theory; §2 covers the methods that replaced it with something provable.

---

## 2. The JEPA family, only as far as it changes implementation

- **V-JEPA** — Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations
  from Video*, <https://arxiv.org/abs/2404.08471>. Same recipe, video tubelets instead of image
  blocks. [assumed — not read in full for this brief]
- **V-JEPA 2** — Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding,
  Prediction and Planning*, <https://arxiv.org/abs/2506.09985>. Pretrained on **over 1 million hours
  of internet video**; `V-JEPA 2-AC` is post-trained into a **latent action-conditioned** world model
  on **under 62 hours of unlabeled robot video**, then deployed zero-shot on Franka arms. [verified,
  abstract]
- **LeCun's world-model framing.** The JEPA argument is that predicting in *representation* space
  lets the model discard unpredictable detail, which a pixel-space generative model cannot. The only
  implementation consequence for us: **the encoder is allowed — required — to throw information away,
  which means the embedding is not a reconstruction of the input and must not be evaluated as one.**
  [assumed — this is the standard reading of LeCun 2022, not a quoted sentence]
- **LeJEPA** — Balestriero & LeCun, *LeJEPA: Provable and Scalable Self-Supervised Learning Without
  the Heuristics*, <https://arxiv.org/abs/2511.08544>. **This is the one that changes what we would
  build.** It identifies the **isotropic Gaussian as the embedding distribution that minimises
  downstream prediction risk**, and enforces it with **SIGReg** (Sketched Isotropic Gaussian
  Regularization). Claimed properties: *"(i) single trade-off hyperparameter, (ii) linear time and
  memory complexity, (iii) stability across hyper-parameters, architectures (ResNets, ViTs, ConvNets)
  and domains, (iv) heuristics-free, e.g., no stop-gradient, no teacher-student, no hyper-parameter
  schedulers"* [verified, abstract].

  **If we build a JEPA, build the LeJEPA variant, not the I-JEPA variant.** No EMA, no momentum
  schedule, no stop-gradient — one λ. SIGReg as instantiated by Mol-JEPA [verified, §5.1]:

  ```
  L_SIGReg  =  (1/L) Σ_{l=1..L}  T({ aₗᵀ sᵢ }_{i=1..B})
  ```

  project the batch of embeddings onto **L = 1024** random unit directions `aₗ ∼ U(S^{K−1})`, apply
  the **Epps–Pulley** goodness-of-fit statistic `T` for 1-D standard normality to each slice, average.
  Mol-JEPA's settings: `L = 1024`, `T_max = 3.0`, 17 evaluation points. [verified]

---

## 3. Collapse — what is load-bearing, and how to detect it

### 3.1 The two collapses

1. **Complete collapse** — the encoder outputs a constant. Loss → 0. Trivially detected by a std
   monitor, and almost never what actually happens in practice.
2. **Dimensional collapse** — embeddings span a low-dimensional subspace of the available space.
   Jing, Vincent, LeCun, Tian, *Understanding Dimensional Collapse in Contrastive Self-supervised
   Learning*, ICLR 2022, <https://arxiv.org/abs/2110.09348>: non-contrastive methods suffer it, and
   **contrastive methods with negatives suffer it too** — negatives are not a cure. [verified]

**Dimensional collapse is the one that silently ruins a run**, because it coexists with a low and
smoothly decreasing loss, and because the usual training curves cannot see it.

### 3.2 The four families of prevention, with exact objectives

**(a) Asymmetry + stop-gradient + EMA** (BYOL, SimSiam, I-JEPA). Empirical. No guarantee. §1.6.

**(b) Explicit variance + decorrelation — VICReg.** Bardes, Ponce, LeCun,
<https://arxiv.org/abs/2105.04906>. Exact terms [verified from the full text]:

```
variance:    v(Z)    = (1/d) Σ_{j=1..d} max(0, γ − S(z^j, ε)),   S(x,ε) = sqrt(Var(x) + ε)
invariance:  s(Z,Z') = (1/n) Σ_i ‖z_i − z'_i‖²₂
covariance:  c(Z)    = (1/d) Σ_{i≠j} [C(Z)]²_{i,j},
                       C(Z) = (1/(n−1)) Σ_i (z_i − z̄)(z_i − z̄)ᵀ
total:       ℓ(Z,Z') = λ·s(Z,Z') + μ·[v(Z)+v(Z')] + ν·[c(Z)+c(Z')]
```

Published values: **λ = 25, μ = 25, ν = 1, γ = 1, ε = 1e-4**; expander output **8192-d** over a
2048-d encoder representation. [verified]

**(c) Cross-correlation identity — Barlow Twins.** Zbontar et al.,
<https://arxiv.org/abs/2103.03230>. [verified from full text]

```
L_BT = Σ_i (1 − C_ii)²  +  λ Σ_i Σ_{j≠i} C²_ij
C_ij = Σ_b z^A_{b,i} z^B_{b,j} / ( sqrt(Σ_b (z^A_{b,i})²) · sqrt(Σ_b (z^B_{b,j})²) )
```

**λ = 5e-3**; projector 3 × 8192 units. The paper states it *"does not require large batches nor
asymmetry between the network twins such as a predictor network, gradient stopping, or a moving
average on the weight updates"*, and that it **"strongly benefits from the use of very
high-dimensional embeddings"** with performance still improving as dimension grows. [verified]

**(d) Distribution matching — SIGReg / LeJEPA.** §2. Strictly stronger than (b): VICReg fixes the
first two moments per dimension; SIGReg tests the whole 1-D marginal against `N(0,1)` along 1024
random directions, so it also penalises multi-modality and heavy tails. [assumed — the comparison is
mine; the formulation is verified]

### 3.3 For a JEPA over our fixed feature vectors — what actually prevents collapse

Ranked by how much I would rely on each, for a ~512-d embedding over a ~2,500-d input:

1. **SIGReg on the *target* embeddings.** Mol-JEPA ablated exactly this and the result is decisive:
   *"We observe training collapse when jointly optimizing the prediction objective and the SIGReg
   regularization on the predicted embeddings. In contrast, applying the regularization to the
   target embeddings yields stable training and the best downstream performance."* [verified,
   Mol-JEPA Appendix F]. Numbers: SIGReg-on-targets **0.436 / 0.393** (linear / nonlinear probe MAE)
   vs SIGReg-on-CLS **0.445 / 0.406**. SIGReg on predictions **collapses**.
2. **Per-sample LayerNorm on the target** before the loss (§1.2(b)). Cheap, one line.
3. **Asymmetric predictor + stop-gradient + EMA** if you go the I-JEPA route rather than LeJEPA.
4. **No projection head.** Mol-JEPA: removing it improved both probes, **0.436 vs 0.468** (linear)
   and **0.393 vs 0.488** (nonlinear). *"unlike in contrastive learning, introducing a separate
   projection space is not advantageous in our JEPA-based setting"* [verified].
5. **Embedding dimension 512.** Mol-JEPA's sweep: 256 → **0.483/0.465**, 512 → **0.436/0.393**,
   1024 → **0.435/0.397**. 512 is the knee; 1024 buys nothing. [verified]
6. **A hard non-triviality constraint on the masking**, see §8.3 — this is the molecular-specific
   risk and it has no analogue in vision. Our feature vector is **massively internally redundant**:
   Avalon bits and ECFP4 bits fire on overlapping substructures, and RDKit2D's `MolWt`,
   `HeavyAtomCount`, `ExactMolWt` and `NumValenceElectrons` are near-deterministic functions of each
   other. A JEPA that masks a random 30% of *scalars* learns to interpolate within a descriptor
   family and nothing else. **[assumed]** — this follows from the construction of those descriptor
   sets, and it is the single most likely way this project fails quietly.

### 3.4 Collapse diagnostics, with thresholds

Compute every `N` steps on a **fixed held-out set of ≥10,000 molecules** (not the training batch).
Let `Z ∈ R^{N×d}` be the embeddings, `d = 512`.

| # | Diagnostic | Formula | Healthy | Warning | Collapsed |
|---|---|---|---|---|---|
| D1 | **RankMe effective rank** | `exp(−Σ_k p_k log p_k)`, `p_k = σ_k(Z)/‖σ(Z)‖₁ + ε`, `ε = 1e-7` | `> 0.4·d` (≈ >205) | `0.1–0.4·d` | `< 0.1·d` (< 51), **or falling for >20% of the run** |
| D2 | **Per-dim std** | `s_j = std(Z[:,j])` | `min s_j > 0.1 · median s_j` | 0.02–0.1 | `>25%` of dims with `s_j < 0.02 · median s_j` |
| D3 | **Off-diagonal correlation** | `mean_{i≠j} |corr(Z_i, Z_j)|` | `< 0.15` | 0.15–0.4 | `> 0.4` |
| D4 | **Singular-value decay** | `#{k : σ_k > 1e-3·σ_1}` | `> 0.5·d` | — | `< 0.05·d` |
| D5 | **Trivial-predictor control** | `L_jepa / L_meanpred`, where `L_meanpred` is the same loss with `ŝ` replaced by the **batch mean target** | `< 0.7` | 0.7–0.95 | **`≥ 0.95`** |
| D6 | **Online linear probe** | ridge on 500 held-out labelled molecules, every 10 epochs | rising | flat | falling (overfit) |

**Thresholds for D1–D4 are [assumed]** — I calibrated them against the published anchors below;
nobody publishes a universal number, and they must be re-anchored against an untrained random
encoder on day one (see §8.5).

Published anchors:
- **RankMe** — Garrido, Balestriero, Najman, LeCun, ICML 2023, <https://arxiv.org/abs/2210.02885>.
  Exact metric as in D1 [verified from full text]. Estimated on **25,600 samples**; **10,000 samples
  reach >95% of converged value** [verified]. Pearson correlation **>0.99** between rank measured on
  source and on target datasets; *"having a high rank is a necessary condition for good downstream
  performance"* [verified]. Values materially below the embedding dim (2048 in their runs) indicate
  dimensional collapse.
- **LiDAR** — Thilak et al., *LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL
  Architectures*, <https://arxiv.org/abs/2312.04000>. Rank of the **LDA matrix of the surrogate SSL
  task**, which *"addresses several shortcomings of recent approaches based on feature covariance rank
  by discriminating between informative and uninformative features"* and *"significantly surpasses
  naive rank based approaches in its predictive power of optimal hyperparameters"* [verified]. It is
  **JEPA-specific and strictly better than RankMe for hyperparameter selection** — worth the extra
  implementation if we run a sweep, skippable for a single run.
- **Mol-JEPA** monitors **effective rank per modality** throughout training and reports it rising
  monotonically while per-modality prediction loss oscillates [verified, Mol-JEPA Fig. 5] — i.e.
  **rank was informative where the loss was not**, which is the whole argument for D1 over watching
  the loss.
- **T-JEPA** (tabular) reports a different trio: KL-divergence `9.3e-4 → 9.38e-2`, a uniformity
  metric `3.12 → 11.38`, and mean pairwise Euclidean distance `5.83 → 70.0` over training; and states
  that *"appending a regularization token [REG] to both target and context representations appeared
  critical to escape the initial representation collapse"* [verified, §5.2].

**D5 is the most important one in this table and nobody publishes it.** It is the direct answer to
"collapse that still trains to a low loss": if your predictor does no better than emitting the mean
target, the loss curve looks perfect and the model is worthless. It costs one extra forward pass.
[assumed — the construction is mine]

---

## 4. Molecular self-supervised pretraining — the honest track record

### 4.1 The methods

| Method | Year | Pretraining corpus | Objective | Source |
|---|---|---|---|---|
| **Mol2Vec** | 2018 | ChEMBL/ZINC | word2vec over integer ECFP4 substructure identifiers treated as words | Jaeger et al. |
| **AttrMask / ContextPred** | 2020 | **2M ZINC15** [assumed] | attribute masking; context-subgraph prediction, + graph-level supervised | Hu et al., ICLR 2020, <https://arxiv.org/abs/1905.12265> |
| **MolBERT** | 2020 | not stated on the abstract page | MLM + SMILES-equivalence + **physchem property prediction** | Fabian et al., <https://arxiv.org/abs/2011.13230> |
| **GROVER** | 2020 | **10M unlabeled**, **100M params** | node/edge/graph-level self-supervision on a message-passing transformer | Rong et al., <https://arxiv.org/abs/2007.02835> |
| **ChemBERTa** | 2020 | **77M PubChem SMILES** released; 10M used | MLM on SMILES | Chithrananda et al., <https://arxiv.org/abs/2010.09885> |
| **MolCLR** | 2022 | **~10M PubChem** | contrastive; atom masking, bond deletion, subgraph removal | Wang et al., *Nat. Mach. Intell.*, <https://arxiv.org/abs/2102.10056> |
| **ChemBERTa-2** | 2022 | **77M PubChem** | MLM **vs multi-task regression (MTR)**, compared directly | Ahmad et al., <https://arxiv.org/abs/2209.01712> |
| **Uni-Mol** | 2023 | 3D conformations + protein pockets | 3D position denoising + masked atom | Zhou et al., ICLR 2023 |
| **MolE** | 2024 | **~842M structures** self-supervised, **then** multi-task supervised on bioactivity | two-stage; disentangled attention | Méndez-Lucio, Nicolaou, Earnshaw, *Nat. Commun.* 15, <https://doi.org/10.1038/s41467-024-53751-y>, PMC11557931 |
| **CheMeleon** | 2026 | descriptor prediction, 1,613 tasks | supervised on computed descriptors | already covered in `research/05-scaling-and-pretraining.md` §1 |

All rows [verified] from the cited abstract or paper except where marked.

Headline claims, for calibration against §4.2: Hu et al. report **"up to 9.4% absolute
improvements"** [verified, abstract] but also that naive single-level pretraining *"give[s] limited
improvement and can even lead to negative transfer on many downstream tasks"* [verified]. GROVER
claims **"more than 6% on average"** over 11 MoleculeNet benchmarks [verified]. MolE
**"outperformed previously published benchmarks on 10 of 22 ADMET prediction tasks in the
Therapeutic Data Commons leaderboard as of September 2023"** [verified] — 10 of 22, not 22 of 22,
and that is the *best* of these claims because it is measured against a public leaderboard rather
than against the authors' own baseline.

### 4.2 The audits — this is the part that decides the question

**(1) Praski, Adamczyk, Czech (2025) — the decisive one.**
*Benchmarking Pretrained Molecular Embedding Models For Molecular Representation Learning*,
<https://arxiv.org/abs/2508.06199>.

Protocol [verified from full text]: **25 models × 25 datasets** (7 MoleculeNet + 18 TDC),
classification, **scaffold splits**, **frozen embeddings** with the *same* downstream heads for every
model (Random Forest, logistic regression, kNN) — *"inspired by approaches like DINO in computer
vision, ensures a fair comparison by removing confounding factors such as downstream classifier
architecture, parameter count, or finetuning strategy"*. Baseline is the **count-ECFP fingerprint**.
Statistics: hierarchical **Bayesian Bradley–Terry** with a conservative **35–65% ROPE**.

**The datasets include `CYP1A2_Veith` (12,579), `CYP2C19_Veith` (12,665), `CYP2C9_Veith` (12,092),
`CYP2D6_Veith` (13,130), `CYP3A4_Veith` (12,328)** [verified, their dataset table] — i.e. **five of
the 25 datasets are CYP inhibition panels on the same four-plus enzymes as our target task.** This is
the closest thing to a direct experiment on our question that exists.

Results table, verbatim [verified]:

| Model | Mean rank ↓ | Mean AUROC ↑ |
|---|---|---|
| CLAMP | 5.40 | 82.55% |
| R-MAT | 6.08 | 80.83% |
| MolBERT | 6.92 | 80.51% |
| ChemBERTa | 7.32 | 79.99% |
| **ECFP** | **7.52** | **79.89%** |
| CDDD | 7.64 | 80.60% |
| Atom Pair | 7.88 | 79.48% |
| MAT | 8.02 | 80.18% |
| MoLFormer | 9.50 | 79.80% |
| Mol2Vec | 10.36 | 79.55% |
| TT | 12.04 | 78.12% |
| ChemFM | 12.60 | 78.25% |
| Uni-Mol2 | 13.32 | 78.11% |
| MolR | 13.80 | 77.84% |
| COATI | 13.92 | 78.02% |
| GraphMVP | 14.64 | 77.18% |
| Chemformer | 15.00 | 77.44% |
| Uni-Mol | 16.64 | 76.85% |
| GROVER | 17.08 | 75.68% |
| GIN | 17.28 | 75.14% |
| SimSon | 19.12 | 73.56% |
| ChemGPT | 19.32 | 74.01% |
| GEM | 19.36 | 74.03% |
| SELFormer | 20.04 | 73.18% |
| GraphFP | 24.20 | 59.67% |

Bayesian verdict against ECFP [verified]:
- **Better: CLAMP only.**
- **Practically equivalent: R-MAT, MolBERT, ChemBERTa, CDDD, Atom Pair, MAT** — *"practically
  equivalent to ECFP, but coming with significantly higher computational cost."*
- **Worse: every other model**, including GROVER, Uni-Mol, Uni-Mol2, GIN, GEM, GraphMVP, ChemGPT,
  MoLFormer, Mol2Vec, Chemformer, SELFormer, SimSon, MolR, COATI, ChemFM, GraphFP.

Abstract, verbatim: *"we arrive at a surprising result: nearly all neural models show negligible or
no improvement over the baseline ECFP molecular fingerprint. Only the CLAMP model, which is also
based on molecular fingerprints, performs statistically significantly better than the alternatives.
These findings raise concerns about the evaluation rigor in existing studies."* [verified]

Per-dataset: *"on 5 datasets no model outperforms ECFP, and on 8 more, only one or two models do"*
[verified]. Win/near-win counts: CLAMP 14, Atom Pair 7, R-MAT 6, ChemBERTa 5, **ECFP 5** [verified].

Their recommendation #1, verbatim: *"Always include the ECFP count fingerprint as a baseline, paired
with a tree-based classifier. Ideally, test other fingerprints like Atom Pair."* [verified]

**Caveats I will not hide.** (i) Frozen embeddings, not fine-tuning — they justify it (*"TDC datasets
include several small ones that reflect real-world ML-assisted drug discovery scenarios, where
fine-tuning results in catastrophic overfitting"*) but it is a real restriction, and it happens to be
*exactly* the way we would use a pretrained encoder. (ii) Classification only, so our regression
target is extrapolation. (iii) The ECFP baseline uses default settings and is not itself tuned — which
cuts *against* the pretrained models, not for them.

**(2) Jiang et al. (2021).** *Could graph neural networks learn better molecular representation for
drug discovery? A comparison study of descriptor-based and graph-based models*, *J. Cheminform.* 13,
<https://doi.org/10.1186/s13321-020-00479-8>, PMID 33597034, PMC7888189.

Descriptor models: SVM, XGBoost, RF, DNN over **206 MOE 1-D/2-D descriptors + 881 PubChem
fingerprints + 307 substructure fingerprints**. Graph models: GCN, GAT, MPNN, Attentive FP. **11
datasets, 50 independent runs each.** [verified]

| Dataset | best descriptor model | best graph model |
|---|---|---|
| ESOL (RMSE ↓) | **SVM 0.569 ± 0.052** | Attentive FP 0.587 ± 0.065 |
| FreeSolv (RMSE ↓) | **SVM 0.852 ± 0.171** | Attentive FP 1.091 ± 0.191 |
| Lipop (RMSE ↓) | XGBoost 0.574 ± 0.034 | **Attentive FP 0.553 ± 0.035** |
| BBBP (AUC ↑) | **RF 0.927 ± 0.025** | — |
| HIV (AUC ↑) | — | **GCN 0.834 ± 0.025** |
| BACE (AUC ↑) | — | **GCN 0.898 ± 0.019** |
| ClinTox (AUC ↑) | **XGBoost 0.911 ± 0.036** | — |
| SIDER (AUC ↑) | **RF 0.646 ± 0.022** | — |
| Tox21 (AUC ↑) | — | **Attentive FP 0.852 ± 0.012** |
| ToxCast (AUC ↑) | — | **Attentive FP 0.794 ± 0.017** |
| MUV (AUC ↑) | **SVM 0.112 ± 0.045** | — |

Conclusion, verbatim: *"we demonstrated that on average the descriptor-based models outperform the
graph-based models in the predictions of a variety of molecular properties in terms of predictive
accuracy and computational efficiency. SVM generally gives the best predictions to regression tasks.
Both XGBoost and RF can give reliable predictions to classification tasks, and graph-based methods,
such as GCN and Attentive FP, can offer outstanding performance on a fraction of larger or
multi-task datasets."* [verified]

Cost, same paper: HIV (40,748 compounds) wall-clock — **XGBoost 23.7 s, RF 14.1 s, SVM 852 s,
Attentive FP 678 s, GAT 1,122 s**. [verified]

**(3) Deng et al. (2023).** *Unraveling Key Elements Underlying Molecular Property Prediction: A
Systematic Study*, <https://arxiv.org/abs/2209.13492> (v4, Sept 2023). **62,820 models trained**
(50,220 on fixed representations, 4,200 on SMILES, 8,400 on graphs). [verified]

- Verbatim: *"representation learning models exhibit limited performance in molecular property
  prediction in most datasets"* and *"dataset size is essential for representation learning models to
  excel."* [verified]
- Verbatim: *"RF achieves the best performance in BACE, BBBP, HIV, Lipop and all opioids-related
  datasets"* under scaffold split, with **RDKit2D descriptors**. [verified]
- **GROVER underperformed on HIV, Lipop and most opioid datasets; concatenating RDKit2D to GROVER
  ("GROVER_RDKit") significantly improved it** — i.e. the learned representation was the weaker half
  of the pair. [verified]
- Dataset-size threshold, on their descriptor-prediction probe: fixed representations + RF/XGBoost and
  pretrained GROVER best **below 1K**; *"RNN achieves the best performance when dataset size exceeds
  10K"*; **pretrained GROVER "does not improve with increasing dataset size."** [verified]
- **Activity cliffs:** AC scaffolds are ~7–13% of scaffolds but hold **25–46% of molecules** across
  the opioid targets; *"RMSE values tend to be higher as the proportions of AC molecules increase"*;
  and MolBERT, GROVER and GROVER_RDKit showed *"average pIC50 values appear to be imputed as the
  predicted values for the AC molecules"*. Verdict: *"Activity cliffs are a key factor contributing
  to limited prediction performance."* [verified]

**Direct bearing on us:** our 6,525 primary labels sit in the band Deng et al. identify as **too small
for representation learning and well-served by fixed representations**, and CYP pIC50 is an
activity-cliff-rich endpoint.

**(4) Sun, Dai, Yu (2022).** *Does GNN Pretraining Help Molecular Representation?*,
<https://arxiv.org/abs/2207.06010>. Ablations over pretraining objective, splits, input features,
dataset scale, architecture. Verbatim findings [verified]:
- *"self-supervised graph pretraining do not always have statistically significant advantages over
  non-pretraining methods in many settings"*
- supervised pretraining helps, but *"the improvement may diminish with richer features or more
  balanced data splits"*
- *"hyper-parameters could have larger impacts on accuracy of downstream tasks than the choice of
  pretraining tasks, especially when the scales of downstream tasks are small"*

The middle bullet is the one that bites: **we already have richer features than any of these papers'
baselines** (count-ECFP4 + RDKit2D + Avalon + ErG, ~2,500-d). The regime where pretraining helps is
the regime where the baseline features are poor.

**(5) Wang, Kaddour, Liu, Tang, Lasenby, Liu (2023).** *Evaluating Self-Supervised Learning for
Molecular Graph Embeddings* (MOLGRAPHEVAL), NeurIPS 2023 Datasets & Benchmarks,
<https://arxiv.org/abs/2206.08005>. Probes representations on generic graph properties, substructure
features and embedding-space characteristics, and finds *"significant inconsistencies between
inferences drawn solely from existing datasets and those derived from more nuanced probing"* and that
*"current evaluation methodologies fail to capture the entirety of the landscape."* [verified]
Implication for us: a downstream win or loss on one benchmark is weak evidence about the
representation either way — which cuts both ways and is a reason to use the §7 protocol.

**(6) van Tilborg, Alenicheva, Grisoni (2022).** *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs*, *JCIM*, <https://doi.org/10.1021/acs.jcim.2c01073>, PMC9749029. **24
ML/DL approaches, 30 macromolecular targets.** All methods struggle on activity cliffs, and
*"machine learning approaches based on molecular descriptors outperformed more complex deep learning
methods."* MoleculeACE released as the benchmark. [verified]

**(7) Guo & Ding (2026).** *Do Larger Models Really Win in Drug Discovery? A Benchmark Assessment of
Model Scaling in AI-Driven Molecular Property and Activity Prediction*,
<https://arxiv.org/abs/2604.26498>. **26 endpoints, 165,541 compound records**, random / Murcko
scaffold / structure-separated protocols. Share of best-performing entries: **classical ML 47.4%,
molecular sequence models 28.8%, GNNs 21.8%, LLM-based 1.9%**. Conclusion: *"compact specialized
models remain highly effective"*, and fit depends on task-model-validation pairing rather than scale.
[verified]

**(8) The one result that goes the other way — and what it is actually about.**
Fischer, Southiratn, Triki, Cedeno (2025), *Deep Learning vs Classical Methods in Potency and ADME
Prediction: Insights from a Computational Blind Challenge*, *JCIM* **65**(24) 13115–13131,
<https://doi.org/10.1021/acs.jcim.5c01982>. From the 2025 ASAP–Polaris–OpenADMET Antiviral Challenge
(65+ teams); the authors placed **1st in SARS-CoV-2 Mpro pIC50** and **4th in aggregated ADME**.
Verbatim: *"while classical methods remain highly competitive for predicting potency, modern deep
learning algorithms significantly outperformed traditional machine learning in ADME prediction."*
They emphasise **data curation** and **feature augmentation using public datasets**. [verified]

Read carefully, this is not a counterexample to §4.2 — it says DL wins on **ADME**, where labels are
plentiful and the endpoint is a bulk physical property, and **classical methods stay competitive on
potency**, which is what our CYP pIC50 task is.

**(9) A clean 2026 control on feature engineering generally.** Cheng, Jin, Zhang (2026), *When
Three-Dimensional Conformer Ensembles Improve Molecular Property Prediction Beyond Two-Dimensional
Fingerprints*, <https://arxiv.org/abs/2606.08825>. ~1,000 experiments, 13 model configurations, 14
regression + 2 classification targets over MoleculeNet, QM9, MARCEL. Four-tier hierarchy: **end-to-end
3D GNNs (SchNet, PaiNN) 21–42% better than fingerprints** on solvation-type properties; engineered
physchem descriptors next; **Morgan+XGBoost next**; **all neural conformer-ensemble methods last**.
Statistically significant RMSE reductions only on solvation-dependent properties (ESOL −11.0%,
p < 1e-9; FreeSolv −13.5%, p < 3e-5) and **no benefit for electronic or steric tasks**. [verified]
Lesson: added representation helps when it encodes **physics the baseline cannot express**, and is
worthless otherwise. That is the shape of H11's result in this programme too.

### 4.3 Honest verdict on §4

**The claim "pretrained molecular models beat well-tuned ECFP+GBM" is not supported.** Across 2021,
2022, 2023, 2025 and 2026, with independent authors, different statistics (50-run means, 62,820
models, hierarchical Bayesian Bradley–Terry, win-share over 26 endpoints) and different task mixes,
the finding replicates: **descriptor/fingerprint baselines with a tree model are at or above the
median of the pretrained field, and the pretrained models that do match them are the ones built on
fingerprints.**

Where pretraining *does* pay, in the same literature:
- **below ~1,000 labels** (Deng et al.; polymer-JEPA §5.2) — not our regime at 6,525;
- when the **baseline features are weak** (Sun et al.: gains *"diminish with richer features"*) — not
  our regime;
- on **ADME bulk properties** with plentiful labels (Fischer et al.) — not our endpoint;
- when pretraining supplies **supervision the baseline cannot compute**: MolE's stage-2 bioactivity
  multi-task, CheMeleon's descriptor prediction, Mol-JEPA's 14 modalities. **This is the live
  direction**, and it is not self-supervision on structure.

---

## 5. JEPA applied to molecules and to tabular data — what exists

### 5.1 Mol-JEPA (the most relevant paper in existence for this question)

Rottach, Schieferdecker, Rudman, Balestriero, Eickhoff — University of Tübingen + **Boehringer
Ingelheim** + UT Austin + Brown. *Mol-JEPA: A multimodal Joint Embedding Predictive Architecture for
Molecules*, arXiv:2608.22642v4, 1 Sep 2026. <https://arxiv.org/abs/2608.22642>. Code:
`github.com/Boehringer-Ingelheim/mol-jepa`. **All of the following is [verified] from the full PDF.**

**The key design decision, and it is the opposite of what we were considering.** They mask **entire
modalities**, never the molecule:

> *"Rather than applying augmentations to the molecular structure, we mask entire modalities and
> predict the corresponding latent representations."*

Their stated reason is exactly the activity-cliff argument: *"molecular property landscapes are often
highly discontinuous, with small structural modifications leading to substantial changes in molecular
properties, as demonstrated by property cliffs… This misalignment between augmentation bias and
desired inductive invariance has been shown to harm SSL performance and offers an explanation why SSL
approaches on molecular data have not been particularly successful."*

And they say outright that JEPAs applied to augmented molecular views *"avoid negative samples, but
suffer from the same augmentation bias as the previous methods"* — citing Mizera et al. 2024, Piccoli
et al. 2026, Orester 2025, Iyer & Sabar 2026.

**Pretraining corpus: 4.66M unique compounds**, merged and deduplicated **by InChI**:

| Dataset | Domain | # raw | # processed | Avg MW |
|---|---|---|---|---|
| ChEMBL | bioactivity | 280,271 | 268,206 | 399.73 |
| TDC | ADMET | 512,303 | 402,881 | 365.86 |
| PCBA-1328 | bioactivity | 1,563,664 | 1,124,981 | 378.82 |
| ∇²DFT | quantum chemistry | 1,936,931 | 1,270,806 | 307.46 |
| Enamine REAL | unlabeled | 2,014,554 | 2,012,628 | 307.13 |

**14 modalities** — note that **ECFP (2048) and MOE descriptors (227) are modalities, i.e. our kind of
feature vector is an input here**: UMA (128), ChemGPT (2048), CLOOME (512), BioXMol (1024), Boltz-2
(6912), Boltz-2 affinity (4), graph transformer (512), MOE (227), ECFP (2048), GFN2-xTB (7), ∇²DFT
(17), ChEMBL assay vector (306), PCBA (1328), TDC ADMET (672).

**Loss** — prediction + SIGReg, no stop-gradient, no teacher-student:

```
z_i^m ~ Bernoulli(r · a_i^m)                              # mask modalities, a = availability
L_pred^m   = (1/B) Σ_i ‖ ŝ_i^m − ŝ_i,pred^m ‖²₂
L_SIGReg^m = (1/L) Σ_l T({ a_lᵀ s_i^m }_{i=1..B})          # L=1024, Epps–Pulley, T_max=3.0, 17 pts
L_JEPA     = Σ_m ( L_pred^m + λ · L_SIGReg^m ),  λ > 0
```

**Hyperparameters (Optuna, 300 configs on a 100k subset, selected on online probe):** lr 5e-4, weight
decay 0.01, hidden dim **512**, **masking ratio 0.3**, **λ = 0.7**, dropout 0.1, graph encoder 3
layers / max pooling / 8 heads, batch 128 (aggregated 1024, 2 accumulation steps). ~**50M params**.
Compute: **12 × RTX-6000ADA, converges at ~150 epochs / 3 days**, then overfits.

**Results — mean absolute error, 3 Taylor–Butina cluster splits (Tanimoto 0.65 on 1024-bit ECFP4):**

| | Mol-JEPA (best) | Mol-JEPA (transformer) | CLAMP | CheMeleon | Chemprop | TabICLv2 + AlvaDesc | RF + ECFP4 | LGBM + AlvaDesc |
|---|---|---|---|---|---|---|---|---|
| **Average MAE over 23 datasets** | **0.402** | 0.427 | 0.576 | 0.471 | 0.519 | 0.453 | 0.559 | 0.468 |

Selected rows: ExpansionRx LogD **0.33** vs RF 0.73 / LGBM 0.43 / TabICL 0.36. PXR activity **0.55**
vs RF 0.81 / LGBM 0.87 / TabICL 0.87. ASAP MERS potency 0.59 vs **Chemprop 0.52** / RF 0.58.
Biogen solubility **0.30** vs RF 0.43 / LGBM 0.33.

**And now the honest part, which the authors state themselves:**

> *"We find that Mol-JEPA has the highest win rates when using classical metrics and is close to the
> highest win rate when using wilcoxon rank tests. Specifically, it achieves superior performance in
> **38%** of the comparisons, compared to **47% for TabICL with AlvaDesc descriptors**. Further
> analysis indicates that TabICLv2 derives much of its advantage from the larger ExpansionRx datasets,
> while Mol-JEPA tends to achieve more significant wins on the smaller ASAP and Biogen benchmarks."*

So: **best on mean MAE, second on paired statistical wins, behind a tabular in-context model fed
ordinary descriptors.** Its advantage is concentrated in **small datasets and out-of-distribution
(low Tanimoto-to-train) molecules**.

**Three ablations that directly constrain our design:**

| Ablation | Linear probe MAE | Nonlinear probe MAE |
|---|---|---|
| All 14 modalities | **0.436** | **0.393** |
| Same data as **semi-supervised labels** instead of JEPA latents | 0.467 | 0.417 |
| Experimental modalities removed entirely | 0.449 | 0.407 |
| SIGReg on targets | **0.436** | **0.393** |
| SIGReg on CLS | 0.445 | 0.406 |
| SIGReg on **predictions** | **training collapse** | **training collapse** |
| No projection head | **0.436** | **0.393** |
| With projection head | 0.468 | 0.488 |
| Embedding dim 256 / **512** / 1024 | 0.483 / **0.436** / 0.435 | 0.465 / **0.393** / 0.397 |

**Leave-one-modality-out:** the **graph** modality contributes most (*"possibly because it is the only
modality with a trainable GNN backbone, whereas the remaining modalities rely on frozen
representations"*), and **"the molecular fingerprint modalities MOE and ECFP4 rank among the most
important modalities, indicating that their expert-designed features contain valuable information for
downstream tasks."** Removing one modality at *inference* changes little (redundancy). **CLOOME
sometimes decreased downstream performance.**

**Pretraining data scaling:** performance improves steadily with corpus size up to 4.66M, for both
probes; they expect further gains. **Masking ratio:** *"no consistent relationship between masking
ratio and downstream performance could be observed."* **λ:** *"performance deteriorates for small
values of λ"* — i.e. the anti-collapse term is load-bearing, not decoration.

### 5.2 Polymer-JEPA — the honest low-data comparison

Piccoli, Vogel, Weber, *Joint Embedding Predictive Architecture for self-supervised pretraining on
polymer molecular graphs*, <https://arxiv.org/abs/2506.18194>, 22 Jun 2025. [verified from full PDF]

Setup: conjugated-copolymer dataset split **40% pretrain (17,186) / 40% finetune / 20% test**; METIS
graph patches with 1-hop expansion; EMA target encoder; wD-MPNN backbone.

Ablations worth copying: **context subgraph 60% of the graph** is optimal (R² 0.65 ± 0.03 vs 0.56 at
20%, 0.61 at 95%); **target subgraph 10%** optimal (R² 0.66 ± 0.02 vs 0.61 at 5%, 0.63 at 20%);
**one target** better than many. Recommended ranges: context **50–75%**, target **10–20%**.

Findings, in their words and order:
- Pretraining helps **up to 4% labelled data (1,728 polymers)**; *"beyond this threshold, the benefits
  of pretraining plateau. This suggest that the available labeled data is sufficient for supervised
  learning, rendering the transferred pretraining knowledge redundant."*
- Against an input-space SSL baseline (Gao et al.): *"the performance increase using input space
  self-supervised pretraining and embedding space self-supervised pretraining (JEPA, our method) is
  comparable"* — JEPA slightly better in the very-low-data regime, the input-space method better with
  more labels. **This is the single most important sentence in §5 for our purposes: a JEPA was not
  better than plain masked-input pretraining.**
- **Against RF on count-ECFP (2048 bits, radius 2):** *"the random forest model outperforms the
  pretrained wD-MPNN in the very low labeled data regimes"* (0.4% and 0.8%); *"only for a small regime
  around 4.0% finetune data, we observe that the pretrained model outperforms both the random forest
  and the wD-MPNN without pretraining."* On the smaller Diblock dataset, *"the random forest model
  outperforms our model throughout all data scenarios."*
- Their own recommendation: *"we advise to consider also the application of simpler models for small
  labeled datasets with relatively simple structure to property relationships."*

### 5.3 M-JEPA — molecular graphs, scaffold shift

Iyer & Sabar, *M-JEPA: Predictive Self-Supervised Learning for Molecular Graphs with Scaffold-Shift
Evaluation on Tox21*, *JCIM* 2026, <https://doi.org/10.1021/acs.jcim.6c00828>. Connected-subgraph
masking + EMA teacher; Bemis–Murcko scaffold splits. Reported: ESOL RMSE **2.12 vs 3.75** for a
compute-matched **InfoNCE** baseline; Tox21 hybrid fine-tuning mean ROC-AUC **0.561 → 0.609** and mean
calibration error **0.279 → 0.064** vs supervised baselines. [verified, abstract]

**Note what the comparisons are against: a contrastive SSL baseline and a supervised-from-scratch GNN.
There is no fingerprint+GBM baseline**, and an ESOL RMSE of 2.12 is far worse than the 0.569 SVM in
§4.2(2). The calibration improvement (0.279 → 0.064) is the interesting number, and it is the one
thing in this corner of the literature that would matter to our deep-ensemble uncertainty story.

### 5.4 Graph-JEPA — graph-level, no molecular property benchmarks

Skenderi, Li, Tang, Cristani, *Graph-level Representation Learning with Joint-Embedding Predictive
Architectures*, <https://arxiv.org/abs/2309.16014>. [verified from full text] METIS partitioning into
non-overlapping patches + 1-hop expansion for connectivity; one context patch, several targets;
**Smooth L1** loss; targets expressed as **hyperbolic-angle coordinates on the 2D unit hyperbola**
rather than raw embeddings; stop-gradient + EMA target following BYOL. Datasets: PROTEINS, MUTAG, DD,
REDDIT-BINARY/MULTI-5K, IMDB-BINARY/MULTI, **ZINC** (regression), Peptides-struct/func, EXP. SOTA on
5/8 classification sets; MUTAG 91.25%, DD 78.64%, REDDIT-B 91.99%, **ZINC 0.434 MSE**. They provide a
theoretical argument (their Eqs. 7–9) that simpler predictor parameterisation + separate target
updates prevent collapse, and acknowledge *"variance in results"* from trivial context selection and
collapse risk — **without reporting any collapse diagnostic.**

### 5.5 T-JEPA — tabular, and the direct GBDT comparison

Thimonier, De Melo Costa, Popineau, Rimmel, Doan, *T-JEPA: Augmentation-Free Self-Supervised Learning
for Tabular Data*, <https://arxiv.org/abs/2410.05016>. [verified from full text]

Mechanism: context encoder (transformer) over masked features, EMA target encoder over unmasked,
transformer predictor; **context and target masks may not overlap** (inter-overlap forbidden,
intra-overlap allowed); masked features either dropped or zeroed, concatenated with mask indicators.
Loss:

```
L(x; M_ctx, M_tgt) = (1/|M_tgt|)(1/|M_ctx|) Σ ‖ g_φ(h_ctx^m, m_k) − h_tgt^{m_k} ‖²₂
```

**Collapse:** *"Appending a regularization token [REG] to both target and context representations
appeared critical to escape the initial representation collapse."* Diagnostics in §3.4.

Results (7 datasets, 80/10/10; MLP+ResNet averaged):

| Method | Adult ↑ | Helena ↑ | Jannis ↑ | ALOI ↑ | CA ↓ | Higgs ↑ | MNIST ↑ |
|---|---|---|---|---|---|---|---|
| MLP baseline | 0.827 | 0.353 | 0.672 | 0.916 | 0.511 | 0.681 | 0.978 |
| MLP + T-JEPA | 0.866 | 0.400 | 0.728 | 0.961 | 0.468 | **0.517** | 0.983 |
| ResNet + T-JEPA | 0.865 | 0.401 | 0.718 | 0.964 | 0.441 | 0.705 | 0.983 |
| XGBoost | **0.872** | 0.375 | 0.721 | 0.951 | **0.433** | **0.729** | 0.980 |
| CatBoost | 0.873 | 0.381 | 0.721 | 0.946 | 0.430 | 0.726 | 0.972 |

Honest reading: **T-JEPA beats the MLP it pretrains, and ties GBDT — it does not beat it.** XGBoost
and CatBoost still win Adult, CA (California Housing, regression) and Higgs; T-JEPA wins Helena,
Jannis, ALOI, MNIST — the high-cardinality-class / image-like sets. On MLP+Higgs it **collapses from
0.681 to 0.517**. The paper's own framing is careful: *"T-JEPA enables some methods to consistently
outperform or match the performance of traditional methods likes Gradient Boosted Decision Trees."*
Against other tabular SSL, T-JEPA+ResNet averages rank **2.6** vs PTaRL 6.1, SubTab 6.3, VIME 7.6.

Hyperparameter ranges (Bayesian search): heads 2–8, hidden 2–128, layers 1–16, FFN 64–1024, dropout
0–0.01, lr 1e-5–1e-3, **context mask share 0.07–0.9**, **target mask share 0.05–0.9**, predictor
layers 2–32, batch 512, **EMA 0.996–1.0**, AdamW + cosine annealing.

### 5.6 The tabular prior that explains §5.5

Grinsztajn, Oyallon, Varoquaux, *Why do tree-based models still outperform deep learning on tabular
data?*, NeurIPS 2022, <https://arxiv.org/abs/2207.08815>. **45 datasets, ~20,000 compute-hours of
hyperparameter search per learner.** Tree models remain SOTA on medium-sized (~10k row) tabular data.
Three inductive-bias gaps for NNs [verified]: **(i) robustness to uninformative features, (ii)
preserving the orientation of the data (no rotational invariance), (iii) learning irregular,
non-smooth functions.**

**All three describe our situation exactly.** A ~2,500-d count-ECFP4+RDKit2D+Avalon+ErG vector at
~6,500 rows is mostly uninformative features; its axes are individually meaningful (bit *k* = a
specific substructure environment), so any rotation-invariant encoder destroys information a tree
reads directly; and CYP pIC50 is activity-cliff-ridden, i.e. non-smooth. This is the mechanistic
reason the §4.2 audits keep replicating, and it is why **a JEPA over our feature vector inherits the
problem rather than solving it.**

### 5.7 Also in the neighbourhood

From an arXiv sweep of JEPA × molecular [verified that these exist; contents not read]:
`2610.01947` Latent JEPA (latent reasoning in chemistry), `2609.20800` JEPA-Anything (molecular
dynamics rollouts), `2606.29723` ScaleAware-JEPA. And the tabular-foundation-model alternative that
Mol-JEPA's own baseline table makes hard to ignore: **TabPFN v2** — Hollmann et al., *Accurate
predictions on small data with a tabular foundation model*, *Nature* 2025,
<https://doi.org/10.1038/s41586-024-08328-6> — *"In 2.8 s, TabPFN outperforms an ensemble of the
strongest baselines tuned for 4 h in a classification setting"*, on datasets **up to 10,000 samples**
[verified]. Our CYP task is 4,905 compounds / 6,525 labels. **This is inside TabPFN v2's stated
envelope and is a far cheaper experiment than any JEPA.**

---

## 6. Unlabeled molecular data at scale — verified URLs and sizes

### ChEMBL 37 [verified 2026-10-05, directory listing and release notes read]

Root: <https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/>

| File | Size | Date |
|---|---|---|
| `chembl_37_chemreps.txt.gz` | **279 MB** | 2026-05-29 |
| `chembl_37.sdf.gz` | 888 MB | 2026-05-29 |
| `chembl_37_sqlite.tar.gz` | 5.4 GB | 2026-05-29 |
| `chembl_37_postgresql.tar.gz` | 1.9 GB | 2026-05-29 |

Release-notes statistics, verbatim [verified,
<https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_release_notes.txt>]: **2,921,148
distinct compounds** (2,897,819 with mol files), **3,824,604 compound records**, **24,527,044
activities**, **1,970,438 assays**, **18,552 targets**, **101,100 documents**. Prepared 1 May 2026.

`chembl_*_chemreps.txt.gz` is the right file for a SMILES corpus. Header and first row, read live by
range-request on 2026-10-05 [verified]:

```
chembl_id	canonical_smiles	standard_inchi	standard_inchi_key
CHEMBL153534	Cc1cc(-c2csc(N=C(N)N)n2)cn1C	InChI=1S/C10H13N5S/c1-6-3-7(4-15(6)2)8-5-...	MFRNFCWYPYSFQQ-UHFFFAOYSA-N
```

**It already ships `standard_inchi_key`**, which is exactly the key §7.2 needs — so the dedupe
against the blinded set needs only the *test* side standardised and keyed, not the corpus side.
Tab-separated, one header line.

```bash
curl -O https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_37_chemreps.txt.gz
# a 200 kB range request is enough to inspect the header without the 279 MB download:
curl -s -r 0-200000 .../chembl_37_chemreps.txt.gz | gunzip -c 2>/dev/null | head -2
```

### PubChem [verified 2026-10-05, directory listing read]

Root: <https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/>

| File | Size | Date |
|---|---|---|
| **`CID-SMILES.gz`** | **1.4 GB** | 2026-09-27 |
| `CID-InChI-Key.gz` | 6.9 GB | 2026-09-27 |
| `CID-Mass.gz` | 1.3 GB | 2026-09-27 |
| `CID-Title.gz` | 1.8 GB | 2026-09-27 |
| `CID-Identifiers.tsv.gz` | 94 MB | 2026-09-27 |
| `CID-Parent.gz` | 520 MB | 2026-09-27 |

Each has a `.md5`. `CID-SMILES.gz` is `CID<TAB>SMILES`, one line per compound. The compound count is
roughly **120M CIDs** [**[assumed]** — the statistics page returned no readable figures; the file size
is consistent with ~10⁸ lines. Get the exact number from `zcat CID-SMILES.gz | wc -l`.] The full
structure archive is under `.../Compound/CURRENT-Full/SDF/`. PUG REST (`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<cid>/JSON`)
works but is per-compound and **must not** be used for bulk — use the FTP files. [verified that the
REST endpoint responds]

### ZINC

Root: <https://files.docking.org/> — `/2D` ("2D files organized by property for chemoinformatics"),
`/3D`, and `/zinc22` with `2d/` and `2d-all/` subdirectories organised into generations `a`–`z`
(*"The biggest generation (by far) is n, followed by x, followed by f"*; `g` = "ZINC20 in stock").
[verified, directory text]

**Two practical obstacles, both verified:** `zinc.docking.org/tranches/home/` **redirects to a
captcha**, and `files.docking.org/zinc22/2d/` returned **HTTP 401** — some generations are
password-protected *"to preserve patentability"*. ZINC20 is Irwin et al., *JCIM* 2020,
<https://doi.org/10.1021/acs.jcim.0c00675>; its abstract reports an **88-fold increase in molecules**
and **16-fold increase in Bemis–Murcko scaffolds** from make-on-demand expansion, and that *"over 97%
of the core Bemis-Murcko scaffolds in make-on-demand libraries are unavailable"* from in-stock
collections [verified]. Absolute totals were not readable from the sources I could reach.

**Recommendation: do not use ZINC for this.** ChEMBL (drug-like, bioactivity-adjacent, 279 MB, one
curl) plus PubChem (scale, 1.4 GB, one curl) covers the need without a captcha or a password, and
Mol-JEPA's own corpus is ChEMBL + TDC + PCBA + ∇²DFT + Enamine REAL — **no ZINC**.

### If we wanted Mol-JEPA's actual recipe

Its three cheap-to-reproduce modalities are **MOE descriptors** (licensed; substitute RDKit2D),
**ECFP** (we have it) and **GFN2-xTB** (7 scalars, free: <https://github.com/grimme-lab/xtb>). Its
expensive ones are Boltz-2 co-folding (they used 8 × RTX A6000 for 100k molecules) and CLOOME /
BioXMol cell-painting embeddings. **ChemGPT embeddings are free and 2048-d**, and ∇²DFT is a public
dataset. A 4-modality subset (ECFP, RDKit2D, xTB, ChemGPT) is reachable on our budget.

---

## 7. A fair pretrain-then-finetune test

### 7.1 Held fixed

Everything except the encoder. Specifically:

1. **The same feature pipeline.** If the baseline GBM eats raw count-ECFP4+RDKit2D+Avalon+ErG, the
   JEPA encoder must eat **the identical arrays from the identical featuriser call**, not a
   re-featurised copy with different RDKit settings or a different descriptor-NaN policy.
2. **The same standardisation, fitted on the same rows.** See §10.3 — this is the most common way a
   pretraining comparison becomes meaningless.
3. **The same splits.** Our 5 scaffold folds, same seed, same fold assignment file.
4. **The same metric and the same bootstrap.** Macro ST-RAE, paired bootstrap over molecules, 95% CI.
5. **The same hyperparameter budget.** Sun et al.'s finding — *"hyper-parameters could have larger
   impacts on accuracy of downstream tasks than the choice of pretraining tasks"* — means an untuned
   baseline against a tuned pretrained model measures tuning. Equal Optuna trial counts, or none.
6. **The same seed count.** `T5` in this programme already requires replicating the reference arm; a
   pretrained arm at 5 seeds against a baseline at 1 seed is not a comparison. The programme has
   already been bitten: `h5/results.json` and `baseline/results.json` are the same arm and disagree by
   **up to 0.094** because `baseline.py` sets no `random_state`.
7. **No pretraining-time model selection on the downstream test fold.** Pick the pretraining
   checkpoint by RankMe or by an **online probe on the validation fold only** (Mol-JEPA selected on
   online-probe performance and found overfitting after ~150 epochs — so a checkpoint *must* be
   selected, and it must not be selected on test).

### 7.2 Leakage control — the 750 blinded test molecules

The pretraining corpus is ChEMBL and/or PubChem. **The OpenADMET CYP challenge's 750 blinded test
compounds are drawn from chemistry that may well be in ChEMBL or PubChem already.** If a test molecule
is in the pretraining set, the encoder has seen its structure — which is not label leakage, but it is
exactly the confound that makes "the pretrained model generalises better" unfalsifiable.

Procedure, in order:

1. **Standardise both sides with the same pipeline.** Use `chembl_structure_pipeline`
   (<https://github.com/chembl/ChEMBL_Structure_Pipeline>, `pip install chembl_structure_pipeline`)
   [verified]: `standardizer.standardize_molblock` then `standardizer.get_parent_molblock` (salt
   strip), and `checker` to drop structures with a high penalty score. Then RDKit
   `Chem.MolToInchiKey`.
2. **Block on the full 27-character standard InChIKey** for exact duplicates, **and on the
   14-character skeleton prefix** (the connectivity-layer block, before the first hyphen) to catch
   stereoisomers, tautomer variants and salt forms of the same connectivity. Drop both from the
   pretraining corpus.
3. **Also block the neighbourhood, not just the identity.** Our own H10 run already measured
   Morgan-0.3 coverage of test folds and used it as a decision criterion; do the same here. Remove
   pretraining molecules with **Tanimoto ≥ 0.9** (count-ECFP4) to any test molecule, and **report the
   remaining similarity distribution**. Mol-JEPA's §4 out-of-distribution figure is binned by
   Tanimoto-to-train for exactly this reason — the headline gain was concentrated in the low-similarity
   bins, which is only a meaningful statement if the high-similarity bins are honest.
4. **Record the counts.** "N removed by exact InChIKey, M by skeleton, K by Tanimoto ≥ 0.9, from a
   corpus of X" belongs in `PROGRESS.md`. An unreported dedupe is an unverified dedupe.
5. **The blinded set has only `Molecule_Name` and `SMILES`** (`PROGRESS.md`, 2026-10-05) — so step 1
   can be run on it without touching anything we are not allowed to see. Do it before pretraining,
   not after.

### 7.3 How to report

One table, four rows, same folds, same seeds, with CIs:

| arm | MA-ST-RAE (mean ± across-seed sd) | Δ vs baseline, paired-bootstrap 95% CI |
|---|---|---|
| XGBoost on raw features (current) | | — |
| MLP deep ensemble on raw features (current) | | |
| **+ pretrained embedding concatenated to raw features** | | |
| **pretrained embedding alone** | | |

Plus: the §3.4 diagnostics at the selected checkpoint; the §7.2 dedupe counts; and **the per-endpoint
breakdown with the fold sd beside it**, because this programme has repeatedly found per-endpoint
effects (0.003–0.007) smaller than the resolution (0.013–0.03). A verdict is only claimable when the
CI excludes zero.

**The two extra arms are not optional.** "Embedding alone" tests whether the encoder learned anything;
"embedding concatenated" tests whether it learned anything *new*. Deng et al.'s GROVER_RDKit result —
concatenating RDKit2D to GROVER *"significantly improved performance"* — is the warning: a pretrained
representation can be strictly worse than the hand-made features and still look good when the two are
glued together.

---

## 8. What to actually implement here

### 8.1 Honest framing

Build this only as a **pre-registered hypothesis with a kill rule**, in the shape `HYPOTHESES.md`
already uses, and only after the two cheaper experiments in §8.6. My prior is a null. The value of
running it is that it is the last untested item on the programme's "bound by representation" thesis,
and a clean null closes it.

### 8.2 Architecture — LeJEPA, not I-JEPA

No EMA, no momentum schedule, no stop-gradient, no projection head. One λ. Justified by §2 (LeJEPA's
own claim) and §5.1 (Mol-JEPA's ablations: no projection, SIGReg on targets, 512-d).

```
x ∈ R^D          D ≈ 2,500   (count-ECFP4 | RDKit2D | Avalon | ErG, concatenated)
blocks b_1..b_K  a fixed partition of the D feature indices into K ≈ 24 blocks  (§8.3)

enc_θ    : per-block MLP → block token ∈ R^512, + learned block-identity embedding
backbone : 4-layer TransformerEncoder, d_model=512, 8 heads, FFN 1024, dropout 0.1, + CLS token
pred_φ   : 2-layer TransformerEncoder, d_model=512  (narrow, asymmetric)
```

### 8.3 Masking — the part that decides whether this works at all

**Do not mask random scalars.** Our vector is internally redundant (§3.3 item 6), and random-scalar
masking is solvable by within-family interpolation. Mask **whole semantic blocks**:

```
K ≈ 24 blocks:
  ECFP4 count bits (2048)  → 16 blocks of 128 CONTIGUOUS bit indices    # bit index is a hash, so
                                                                       # contiguity is arbitrary —
                                                                       # see the hard rule below
  Avalon (512)             → 4 blocks of 128
  ErG (315)                → 2 blocks
  RDKit2D (~200)           → 2 blocks, split by descriptor FAMILY
                             (constitutional/graph-topological | physchem/electronic)

per step:
  mask_ratio r = 0.3                                   # Mol-JEPA's tuned value
  target blocks: sample ceil(r*K) ≈ 7 blocks
  context      : ALL remaining blocks                  # cf. I-JEPA 0.85-1.0, polymer-JEPA 50-75%
  constraint   : context ∩ targets = ∅                 # T-JEPA: inter-overlap forbidden
```

**The hard rule, and it is the one that makes or breaks this.** *Every target block must be
non-trivially predictable from the context, and not deterministically so.* Two things violate it:

- **Within-family redundancy.** `MolWt`, `ExactMolWt`, `HeavyAtomCount` and `NumValenceElectrons`
  are near-deterministic functions of each other. If they land in different blocks, the model learns
  a linear identity and the loss goes to zero with no representation learned. **Mitigation: put
  mutually-redundant RDKit2D descriptors in the SAME block**, and compute the pairwise `|r|` matrix
  over the pretraining corpus first, clustering at `|r| > 0.95` before defining blocks.
- **Hash collisions and bit correlation in ECFP4.** Contiguous bit ranges of a hashed fingerprint are
  arbitrary groupings, which is fine, but *co-occurring* bits (a substructure and its superstructure)
  must not be split the same way. **Mitigation: define the 16 ECFP blocks by clustering bits on
  co-occurrence over the pretraining corpus, not by index.** [assumed — this is design, not a result]

**Pre-flight check, before any training:** for each block, fit a **ridge regression from the other
blocks' raw features to that block's raw features** on 100k corpus molecules and record the R².
**Any block with R² > 0.95 is a trivial target and must be merged or dropped.** This is 30 minutes of
CPU and it is the difference between a real experiment and a month of beautiful loss curves.

### 8.4 Loss

```python
def jepa_step(x, avail_mask):
    tgt_blocks, ctx_blocks = sample_masks(K, r=0.30)

    # targets: FULL input through the SAME encoder (no EMA under LeJEPA), then per-token LayerNorm
    h = backbone(enc(x))                                     # (B, K+1, 512)
    h = F.layer_norm(h, (h.size(-1),))                       # §1.2(b) -- keep this line
    h_tgt = h[:, tgt_blocks, :]

    # context: masked input; target positions become a shared learnable mask token + block embedding
    z = backbone(enc(x, keep=ctx_blocks))
    z_pred = pred(z, query_positions=tgt_blocks)             # (B, |tgt|, 512)

    L_pred   = F.smooth_l1_loss(z_pred, h_tgt)               # Huber, as in the I-JEPA code
    L_sigreg = sigreg(h_tgt, L=1024, T_max=3.0, n_pts=17)    # SIGReg on TARGETS (Mol-JEPA ablation)
    return L_pred + 0.7 * L_sigreg                           # lambda = 0.7 (Mol-JEPA tuned value)
```

If implementing LeJEPA properly is too much, the I-JEPA fallback is: separate `target_encoder` under
`torch.no_grad()`, EMA `0.996 → 1.0` linear over the run, keep the LayerNorm, **and add VICReg's
variance+covariance terms (μ=25, ν=1, γ=1, ε=1e-4) on the target embeddings** rather than trusting
asymmetry alone. Do **not** ship asymmetry-only; §3.1 and the whole of §3.2 say it is the one
mechanism with no guarantee.

### 8.5 Hyperparameters — all taken from a measured source

| Knob | Value | Source |
|---|---|---|
| Embedding dim | **512** | Mol-JEPA sweep: 256 → 0.483/0.465, 512 → 0.436/0.393, 1024 → 0.435/0.397 [verified] |
| Projection head | **none** | Mol-JEPA: 0.436/0.393 without vs 0.468/0.488 with [verified] |
| Mask ratio | **0.30** | Mol-JEPA best value; *"no consistent relationship"* across 0.1–0.9, so do not sweep it [verified] |
| λ (SIGReg weight) | **0.7** | Mol-JEPA best; *"performance deteriorates for small values of λ"* [verified] |
| SIGReg placement | **targets** | Mol-JEPA: on predictions → **collapse** [verified] |
| SIGReg L / T_max / pts | **1024 / 3.0 / 17** | Mol-JEPA [verified] |
| Optimiser | AdamW, lr **5e-4**, wd **0.01** | Mol-JEPA [verified] |
| Batch | 128, grad-accum 2 → effective **1024** | Mol-JEPA [verified] |
| Backbone / predictor depth | 4 / 2, predictor narrower | I-JEPA uses a narrow 384-d predictor against a 1280-d encoder [verified]; depths scaled down for our size [assumed] |
| Epochs | **~150**, early-stop on online probe | Mol-JEPA: *"converges after 3 days at around 150 epochs. After this point, we observe overfitting and degradation of online probe performance"* [verified] |
| Corpus | ChEMBL 37 (2.92M), deduped per §7.2 | §6 [verified] |
| Precision | fp32 or bf16 | Mol-JEPA used fp32 [verified] |

**Day-one calibration, before trusting any diagnostic:** run §3.4 on a **randomly initialised,
untrained** encoder and record D1–D4. A random encoder has near-full rank, so it sets the *ceiling*,
not the floor. Then run them on **PCA of the raw features to 512-d**, which sets the floor that any
useful embedding must beat on D6 (the online probe). If the trained encoder's probe does not beat
512-d PCA of the input, the pretraining did nothing — and **that comparison, not the loss, is the
result**.

### 8.6 Cost

Mol-JEPA: 50M params, 14 modalities, 4.66M molecules, 12 × RTX-6000ADA, 3 days. Ours is one modality
family, ~5–15M params, 2.9M molecules. On one A100-40GB: **[assumed] 10–20 hours** for 150 epochs, +
~2 h for featurising 2.9M ChEMBL SMILES (which is the real bottleneck; `AGENTS.md` already says
featurisation dominates and the A100's 12 vCPU / 89.6 GB is why it is the cheaper choice). Budget
**~25 A100-hours including the dedupe and diagnostics**, against ~2 A100-hours for the §8.7
alternative and ~0.2 for the §8.8 controls.

### 8.7 The cheaper alternative — and whether it is just as good

**Plain masked-feature denoising autoencoding**: same block masking, but reconstruct the **raw masked
features** instead of their embedding, with a per-family loss (Poisson/count NLL for ECFP4 counts,
BCE for Avalon bits, Huber on standardised RDKit2D/ErG). No target encoder, no EMA, no SIGReg, no
collapse risk — a constant output is a bad reconstruction, so the degenerate solution is penalised by
the task itself.

**Is it as good? The evidence says yes, probably indistinguishably so.**

- Polymer-JEPA ran the comparison directly: *"the performance increase using input space
  self-supervised pretraining and embedding space self-supervised pretraining (JEPA, our method) is
  comparable"* — JEPA marginally ahead in the very-low-data regime, input-space SSL ahead with more
  labels. [verified, §5.2]
- T-JEPA beats SubTab (rank 6.3) and VIME (7.6), which are input-space tabular SSL — but T-JEPA still
  only **ties** GBDT [verified, §5.5]. So the JEPA-vs-denoising gap is smaller than the
  everything-vs-GBDT gap, which is the gap that actually matters to us.
- JEPA's theoretical advantage — discarding unpredictable detail — is strongest where the input has
  large irreducibly-unpredictable content (pixels, video). **Our input is a deterministic function of
  the SMILES.** There is no sensor noise to discard. [assumed — this is the argument, not a result,
  but it is the argument LeCun makes for JEPA and it runs backwards here]

**Therefore: build the denoising autoencoder first.** Same masking machinery, same corpus, same
dedupe, same probe, **a quarter of the implementation and none of the collapse risk**. If it produces
nothing, a JEPA on the same features almost certainly produces nothing either, and we will have
learned that for ~5 A100-hours instead of 25. If it produces something, *then* the JEPA is worth
trying as the refinement.

### 8.8 Two controls that should run before either

1. **TabPFN v2 / TabICLv2 on our existing features.** TabICLv2+AlvaDesc beat Mol-JEPA on paired
   Wilcoxon wins (47% vs 38%) in Mol-JEPA's own paper [verified, §5.1], and TabPFN v2's stated
   envelope is datasets up to 10,000 samples — ours is 4,905 compounds [verified, §5.7]. **Hours, not
   days, and no pretraining.** Feature count needs reducing (TabPFN v2 has a feature ceiling; use
   SelectKBest as Mol-JEPA's baselines did, or PCA) [assumed].
2. **CLAMP.** It is the **only** model of 25 that beat ECFP in Praski et al., winning or near-winning
   **14 of 25 datasets** [verified, §4.2(1)], it is fingerprint-based, and Praski's recommendation #2
   is to try it *"especially when using embeddings without fine-tuning neural networks (e.g., with
   very small datasets)"*. Seidl et al. 2023. It is also in Mol-JEPA's baseline table, where it was
   the *worst* deep baseline at 0.576 mean MAE — so the two sources disagree, which itself is worth
   one afternoon to resolve on our own folds.

---

## 9. Honest expected value

**Do I expect pretraining to beat our ECFP+GBM / MLP-ensemble baseline on 6,525 labels? No.**

The reasoning, in the order that matters:

1. **The closest published experiment to ours says no, with CYP datasets in it.** Praski et al.: 25
   models, frozen embeddings, scaffold splits, five CYP Veith panels among the 25 datasets, and only
   CLAMP — a fingerprint-based model — beats count-ECFP. ECFP's mean rank is 7.52 of 25.
2. **Our label count is in the dead zone.** Deng et al.: representation learning needs scale; fixed
   representations win below ~1K and tie up to ~10K; GROVER *"does not improve with increasing dataset
   size."* Polymer-JEPA: gains plateau past ~1,700 labels. We are at 6,525.
3. **Our baseline features are already good**, which is precisely where Sun et al. found the gains
   *"diminish with richer features."*
4. **The tabular prior is against us.** Grinsztajn et al.'s three gaps — uninformative features,
   rotational non-invariance, irregular targets — all hold for a 2,500-d fingerprint vector at 6.5k
   rows, and a JEPA over that vector inherits all three.
5. **CYP pIC50 is activity-cliff territory**, where van Tilborg et al. found descriptor models beat
   deep learning and Deng et al. found pretrained models imputing the mean.
6. **The programme has already run the stronger version of the experiment and got a null.** 49,965
   real labelled PubChem Veith CYP measurements: +0.0004 and −0.0031 across two seeds, against a
   ~0.03 bootstrap scale. Unlabeled structures are a weaker intervention than labels on the same
   enzymes.
7. **And the effect would be unmeasurable even if real.** Our validation fold is ~240 scorable primary
   labels per endpoint; H12 already fixed that *"a per-endpoint ST-RAE difference under about 0.02 is
   not resolvable."* Every pretraining effect in §4–§5 that survived scrutiny was smaller than that.

**Point estimate: 0.00 ± 0.03 MA-ST-RAE. Probability it clears a 95% CI excluding zero on macro
ST-RAE: ~15%.** [assumed — my calibration, from the seven points above]

**What I would expect to work instead, in descending expected value:**

1. **TabPFN v2 / TabICLv2 on the features we already have.** Hours. Beat Mol-JEPA on paired wins in
   Mol-JEPA's own table. Nothing to pretrain.
2. **More feature modalities we can compute.** GFN2-xTB (7 scalars, free, and Mol-JEPA used it);
   ChemGPT embeddings (free, 2048-d); protonation-state features (H11 moved CYP2D6 by +0.0071 — a
   third of what was predicted, but in the right direction, and the mechanism was right). This is the
   §4.2(9) lesson: added representation pays when it encodes physics the baseline cannot express.
3. **Blending**, which this programme has already measured as the one thing that works: xgb 0.4432 →
   blend 0.4185, **+0.0283, 95% CI [+0.0160, +0.0427]**. Adding a third *differently-wrong* model is
   worth more than making one model better.
4. **CheMeleon / CLAMP as additional frozen feature sources** concatenated to our vector — cheap,
   already-trained, and the GROVER_RDKit result says concatenation is where pretrained
   representations earn their keep.
5. **The JEPA, last**, and only in the §8.7 denoising-autoencoder form first.

**And the direction the literature genuinely endorses, if we ever have the budget: Mol-JEPA's.** Not
self-supervision on structure — **latent prediction across many kinds of supervision**, with
fingerprints as one of the modalities rather than as the thing being replaced. Its 0.402 vs 0.559
(RF+ECFP4) is the only large margin in this entire document, it came from 14 modalities including
Boltz-2 affinities and cell-painting embeddings, and its own leave-one-out found **ECFP and MOE among
the most important of them.**

---

## 10. Pitfalls that make a naive implementation silently wrong

**10.1 Collapse that still trains to a low loss.** The headline trap. A JEPA whose encoder outputs a
near-constant has a *lower* loss than a working one, because the predictor's job becomes trivial.
Nothing in the training curve shows it. **Detectors:** D5 (trivial-predictor control — if
`L_jepa / L_meanpred ≥ 0.95` the model learned nothing), D1 (RankMe `< 0.1·d` or falling), D2
(dimension stds). Mol-JEPA's own experience: SIGReg on the *predicted* embeddings **collapses**, on
the *targets* it does not — a one-line difference between a working run and a worthless one, and the
loss does not tell you which you have.

**10.2 Trivially-predictable target blocks.** Our feature vector is internally redundant.
`ExactMolWt` from `MolWt` is a linear identity; Avalon bits and ECFP4 bits fire on overlapping
substructures. Mask those apart and the loss collapses to ~0 with no representation learned — and
this looks like *success*, not collapse, because D1–D4 can stay healthy while the model has only
learned arithmetic. **Detector:** the §8.3 pre-flight ridge R² per block; merge or drop any block
above 0.95. **This has no analogue in I-JEPA** (image patches are not deterministic functions of
their neighbours), so it is not in any paper's pitfall list.

**10.3 Test molecules in the pretraining corpus.** ChEMBL 2.92M and PubChem ~120M plausibly contain
the OpenADMET blinded compounds. §7.2 is the procedure. Three specific ways it goes wrong:
- **Deduping on non-standardised SMILES.** A salt, a different tautomer or a drawn-differently
  stereocentre gives a different canonical SMILES and the same molecule. Standardise with
  `chembl_structure_pipeline` on both sides, then InChIKey.
- **Deduping on the full InChIKey only.** The 27-char key is stereo- and protonation-sensitive. Block
  on the **14-char skeleton prefix** too.
- **Deduping identity but not neighbourhood.** A corpus molecule at Tanimoto 0.97 to a test molecule
  is, for pretraining purposes, that molecule. H10 already established Morgan-0.3 coverage as this
  programme's instrument for exactly this question; reuse it, and report the similarity histogram.

**10.4 Different feature standardisation for the pretrained arm than the baseline.** The quietest and
most likely error. An encoder trained on z-scored features whose scaler was fitted on **2.9M ChEMBL
molecules** and a GBM trained on raw counts are not two models on the same data — and the encoder's
scaler has seen the test molecules' *distribution*, which is a mild but real leak. **Rules:** one
featuriser call, one scaler, **fitted on the training fold only**, serialised, reused byte-identically
by every arm; and the baseline GBM must be refit under that same scaler so the comparison is of
encoders, not of preprocessing. Trees are scale-invariant so this costs the GBM nothing — which is
exactly why it is so easy to forget.

**10.5 Selecting the pretraining checkpoint on the test fold.** Mol-JEPA overfits after ~150 epochs,
so a checkpoint *must* be chosen. Choose it on RankMe (needs no labels at all) or on a validation-fold
online probe. H11's design note in `HYPOTHESES.md` already names this error shape: *"choosing the
better arm on the test fold is test-set selection and would inflate GBM by construction."*

**10.6 Comparing one fit against one fit.** `h5/results.json` vs `baseline/results.json` — same arm,
same split — disagree by **up to 0.094** because `baseline.py` sets no `random_state`. Any pretraining
verdict must be distribution-against-distribution at ≥5 seeds per arm, or it is measuring the RNG.

**10.7 Taking the headline number and not the paired test.** Mol-JEPA's mean MAE is the best in its
table (0.402) and its paired-Wilcoxon win rate is **second** (38% vs TabICLv2's 47%). GROVER claims
*"more than 6% on average"* and ranks **17.08 of 25** under Praski's Bayesian test. Both are in this
document because both are true; only the second of each pair survives multiple-comparison correction.
If we publish a number for our own arm, it goes in with a paired bootstrap CI beside it.

**10.8 Evaluating the embedding as a reconstruction.** A JEPA embedding is *supposed* to discard
information. Low reconstruction fidelity of the input is not a failure and reconstruction loss is not
a quality signal. The only valid quality signals are §3.4's D1/D6 and the downstream metric.

**10.9 Masking-ratio sweeps.** Mol-JEPA found *"no consistent relationship between masking ratio and
downstream performance"* across 0.1–0.9. T-JEPA searched 0.07–0.9 for context and 0.05–0.9 for target.
Do not spend GPU-hours here; take r = 0.3 and spend the budget on the block definition (§8.3), which
is where the actual signal is.

---

## 11. Sources

**JEPA core**
- I-JEPA — <https://arxiv.org/abs/2301.08243>; code <https://github.com/facebookresearch/ijepa>
- V-JEPA — <https://arxiv.org/abs/2404.08471>
- V-JEPA 2 — <https://arxiv.org/abs/2506.09985>
- LeJEPA / SIGReg — <https://arxiv.org/abs/2511.08544>

**Collapse and diagnostics**
- VICReg — <https://arxiv.org/abs/2105.04906>
- Barlow Twins — <https://arxiv.org/abs/2103.03230>
- Dimensional collapse — <https://arxiv.org/abs/2110.09348>
- RankMe — <https://arxiv.org/abs/2210.02885>
- LiDAR — <https://arxiv.org/abs/2312.04000>

**Molecular pretraining**
- Hu et al. — <https://arxiv.org/abs/1905.12265>
- GROVER — <https://arxiv.org/abs/2007.02835>
- MolBERT — <https://arxiv.org/abs/2011.13230>
- ChemBERTa — <https://arxiv.org/abs/2010.09885>; ChemBERTa-2 — <https://arxiv.org/abs/2209.01712>
- MolCLR — <https://arxiv.org/abs/2102.10056>
- MolE — <https://doi.org/10.1038/s41467-024-53751-y> (PMC11557931)

**The audits**
- Praski, Adamczyk, Czech 2025 — <https://arxiv.org/abs/2508.06199>
- Jiang et al. 2021 — <https://doi.org/10.1186/s13321-020-00479-8> (PMC7888189)
- Deng et al. 2023 — <https://arxiv.org/abs/2209.13492>
- Sun, Dai, Yu 2022 — <https://arxiv.org/abs/2207.06010>
- MOLGRAPHEVAL — <https://arxiv.org/abs/2206.08005>
- van Tilborg et al. 2022 — <https://doi.org/10.1021/acs.jcim.2c01073> (PMC9749029)
- Guo & Ding 2026 — <https://arxiv.org/abs/2604.26498>
- Fischer et al. 2025 — <https://doi.org/10.1021/acs.jcim.5c01982>
- Cheng, Jin, Zhang 2026 — <https://arxiv.org/abs/2606.08825>

**JEPA on molecules / tabular**
- Mol-JEPA — <https://arxiv.org/abs/2608.22642>; code `github.com/Boehringer-Ingelheim/mol-jepa`
- Polymer-JEPA — <https://arxiv.org/abs/2506.18194>
- M-JEPA — <https://doi.org/10.1021/acs.jcim.6c00828>
- Graph-JEPA — <https://arxiv.org/abs/2309.16014>
- T-JEPA — <https://arxiv.org/abs/2410.05016>
- Grinsztajn et al. — <https://arxiv.org/abs/2207.08815>
- TabPFN v2 — <https://doi.org/10.1038/s41586-024-08328-6>

**Data**
- ChEMBL 37 — <https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/>
- PubChem bulk — <https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/>
- ZINC — <https://files.docking.org/>; ZINC20 paper <https://doi.org/10.1021/acs.jcim.0c00675>
- ChEMBL Structure Pipeline — <https://github.com/chembl/ChEMBL_Structure_Pipeline>
- xtb (GFN2-xTB) — <https://github.com/grimme-lab/xtb>

**Method note.** The WebSearch budget for this session was exhausted before research began, so
everything above was gathered by direct fetch of arXiv, PMC/Europe PMC, publisher, FTP and GitHub
URLs, plus local `pdftotext` extraction of the Mol-JEPA, polymer-JEPA and Praski PDFs. The three
results tables transcribed in full (Mol-JEPA Tables 3/10, Praski Tables 1–3, polymer-JEPA Tables 1–2)
were read from the extracted text, not from an abstract.
