# 23 — Analog-series prediction: pairwise/delta learning, local models, activity cliffs

**Question.** How do we best predict pIC50 for close analogs of known hits (hit expansion / SAR-by-catalog),
given a global multitask MLP ensemble, ~4.9k dose-response compounds + a 4,376-compound single-point screen,
a test set of 750 Enamine analogs organised as 75 series × 10, and an L1-outside-credible-interval metric
(ST-RAE) normalised so that 1.0 = predicting the mean.

**Method.** 503 unique abstracts screened (70 arXiv API, 413 Europe PMC REST across 22 title/abstract queries,
20 Semantic Scholar) plus ~25 targeted web searches; 22 papers/repos deep-read (full text where open access).
Date: 2026-10-08.

---

## Executive summary — ranked by expected effect per unit of work

1. **Measure the neighbour *residual* correlation ρ_r(T) of your current ensemble before building anything.**
   Everything below is a monotone function of it. Expected effect: none directly; it is the number that tells you
   whether items 2–4 are worth 15% or 1%. Validate: out-of-fold residuals, binned by Tanimoto, Pearson ρ per bin. **½ day.**
2. **Series-level residual offset with empirical-Bayes shrinkage** (anchor the *series*, not the molecule).
   Expected effect **3–12% ST-RAE** at ρ_r ≈ 0.28, up to ~26% if ρ_r ≥ 0.5 in the T ≥ 0.6 bin. This is the
   single best-evidenced mechanism (MolPAIR's reliability gate is worth 4× its own pairwise learning, §2.2).
   Validate on a hit-expansion split (§5). **2 days.**
3. **Switch the training loss and the ensemble aggregator to L1/median.** ST-RAE is L1-with-a-dead-zone; the
   optimum is a median-like balance point, not a mean (§0.3). ACA found MAE > MSE as the base loss on
   narrow-scaffold series; median aggregation is free. Expected **2–6%**, near-zero risk. **½ day.**
4. **Tanimoto-kernel GP (or GP on the ensemble's residuals) for calibrated per-compound intervals.**
   GAUCHE: Tanimoto-fragprints GP has the best NLPD of all GP/deep-probabilistic variants tested, and exact GP
   is trivial at n ≈ 5k. Calibration is what a dead-zone metric pays for. Expected **2–5%** plus a usable
   confidence gate. **1–2 days.**
5. **Correct the selection bias in the anchors.** The parents are top-25 *selected* hits, so their labels carry a
   winner's curse (≈0.24 log at 0.3-log assay noise) and their analogs regress to the mean. Any un-shrunk
   anchor over-predicts. Expected **1–4%**, and it removes a systematic per-series bias — exactly what L1 punishes. **½ day.**
6. **A tuned tree baseline (LightGBM/RF on ECFP+descriptors) as a blend partner.** MoleculeACE, SQRL and the
   Fralish/Reker review all land on the same place: trees win or tie on 1–5k-compound bioactivity tasks. If your
   MLP ensemble has never been blended with one, this is likely the largest single accuracy gain available. Expected **3–10%.** **1 day.**
7. **Pairwise/delta training (DeepDelta/PADRE/SQRL style) — do it, but as an auxiliary, not a replacement.**
   Evidence says it rescues weak neural nets and helps most where |Δ| is *large* (cliffs), while being *worse*
   on small |Δ| — and small |Δ| is most of an analog series. Expected **0–4%** on ST-RAE; **negative** if it replaces the global model. **3–4 days.**
8. **Build the hit-expansion validation split first (SIMPD-style, matched on test geometry).** Not a gain, a
   prerequisite: your scaffold split (NN sim 0.354) is a different kind of set from the test (0.665), which is
   precisely the failure already recorded in this programme. **1 day. Do it before 2–7.**
9. **Activity-cliff-aware losses (ACA triplet / ACtriplet).** Reported 4.6–7.2% RMSE reduction — but over GNN
   baselines, not over tuned ECFP trees. Worth a try only after 2–6. Expected **0–5%.** **3 days.**
10. **Do NOT reach for few-shot / in-context / meta-learning (FS-Mol, MHNfs, CLAMP, TabPFN-as-few-shot).**
    Wrong regime: those are 16–128 labels on a *new* task; you have 4.9k labels on these four. Expected **~0%.**

---

## §0 The arithmetic that governs this problem

Three identities decide how much any local/anchor method can possibly earn here. They are worth more than any
paper in the bibliography, because they run on numbers you already measured.

### 0.1 Your pairwise prior already pins the neighbour correlation

You measured `r := sd(Δ) / (sd(y)·√2) = 0.846` at Tanimoto ≥ 0.5 (n ≈ 2,183 pairs). Assume paired compounds
have equal marginal variance σ². Then `sd(Δ)² = 2σ²(1−ρ)`, so

```
ρ = 1 − r²
```

**r = 0.846 → ρ = 0.284.** At T ≥ 0.5, two analogs' potencies correlate at only ~0.28. Consequences:

| quantity | formula | at r = 0.846 |
|---|---|---|
| "copy the anchor" (Δ̂ = 0) error | `r·√2·σ` | **1.196 σ — 20% WORSE than predicting the mean** |
| best single-anchor linear estimator | `σ·√(1−ρ²)` | 0.959 σ (4.1% better than the mean) |
| break-even for anchor-copying | `sd(Δ) < σ ⟺ ρ > 0.5 ⟺ r < 0.707` | **not met at T ≥ 0.5** |

So there is a crisp gate that falls straight out of your own statistic: **only let an anchor carry weight in a
similarity bin where the measured r in that bin is below 0.707.** Above it, un-shrunk anchoring is actively
harmful. Recompute r per bin (T ∈ [0.4,0.5), [0.5,0.6), [0.6,0.7), [0.7,0.8), ≥0.8) — the test set's internal
mean NN similarity of 0.665 means the bins that matter are the ones you have not yet characterised. CYP2C9
having almost no pairs is itself a finding: it should get a *smaller* anchor weight, not a borrowed one.

### 0.2 What matters is residual correlation, not label correlation

The anchor can only tell you what the global model missed. With out-of-fold residuals `e = y − ŷ_global`, let
`ρ_r = corr(e_query, e_anchor)` in a similarity bin. For k exchangeable anchors, the optimal correction is
`ŷ = ŷ_global + β·ē` with

```
β   = k·ρ_r / (1 + (k−1)·ρ_r)
RMSE reduction = 1 − sqrt( 1 − k·ρ_r² / (1 + (k−1)·ρ_r) )
```

| ρ_r | k=1 | k=3 | k=5 | k=10 | k=30 | k→∞ |
|---|---|---|---|---|---|---|
| 0.15 | 1.1% | 2.6% | 3.6% | 4.9% | 6.5% | 7.8% |
| 0.20 | 2.0% | 4.4% | 5.7% | 7.4% | 9.3% | 10.6% |
| **0.284** | **4.1%** | **8.0%** | **9.9%** | **12.1%** | 14.1% | 15.4% |
| 0.40 | 8.3% | 14.4% | 16.8% | 19.2% | 21.3% | 22.5% |
| 0.50 | 13.4% | 20.9% | 23.6% | 26.1% | 28.2% | 29.3% |
| 0.60 | 20.0% | 28.6% | 31.4% | 33.9% | 35.7% | 36.8% |

ρ_r is normally **lower** than the label ρ (a trained model has already eaten the structural signal), so treat
ρ = 0.284 as a *ceiling* for ρ_r at T ≥ 0.5 and read the table as a bound. This table is why the published
effect sizes are what they are: MolPAIR's whole advantage over its global predictor is 0.00819 RMSE on a
mean of 0.893 — **0.9%** — which is what k≈1–3 anchors at ρ_r ≈ 0.1 buys. If your binned ρ_r at T ≥ 0.6
comes back at 0.4–0.5 — plausible, given 84% of the test has a near-neighbour active — the same machinery is
worth 15–25% instead, and item 2 becomes the whole programme.

### 0.3 ST-RAE is a hinge loss with a dead zone, so the optimum is a median, not a mean

Per compound the loss is `max(0, p − hi) + max(0, lo − p)`. Differentiating in p,
`∂/∂p = P(hi < p) − P(lo > p)`, so the minimiser p\* satisfies

```
P(interval entirely below p*)  =  P(interval entirely above p*)
```

a **median-like balance point of the interval distribution**, not a conditional mean. Three consequences:

- Train with **L1/Huber or pinball at τ=0.5**, not MSE. ACA independently found MAE beats MSE as the base loss
  on low-sample narrow-scaffold series, which is this setting.
- **Aggregate the ensemble by median**, not mean. Heavy-tailed cliff residuals are exactly what breaks a mean.
- A wide credible interval is *free accuracy*. Predictions landing inside a wide CI cost nothing, so effort
  should be concentrated on compounds whose intervals are narrow — i.e. weight the training/selection loss by
  `1/width` if you want the optimiser to care about what the metric cares about.

### 0.4 The anchors are selected top hits, so they are biased upward

The 75 parents are the top-25 hits per CYP — a selection on a noisy measurement. With between-compound
sd σ ≈ 1 log and assay sd s, the parent's own label over-states its true potency by a factor
`1 − σ²/(σ²+s²)` of its selection lift (≈2.9σ at the top 0.5%):

| assay sd | reliability σ²/(σ²+s²) | winner's curse on the parent label |
|---|---|---|
| 0.2 log | 0.962 | 0.11 log |
| **0.3 log (Kramer's homogeneous-data value)** | **0.917** | **0.24 log** |
| 0.5 log (heterogeneous) | 0.800 | 0.58 log |

On top of that, the analog regresses toward the mean by `(1−ρ)` of the parent's lift. Both effects push the
same way: **an un-shrunk anchor over-predicts every one of the 75 series**, and a systematic per-series offset
is the most expensive possible error under an L1 metric. Shrink the anchor offset by (reliability × ρ_bin). You
have the credible intervals, so you can estimate s per compound rather than assuming it.

---

## §1 Pairwise / delta learning

### 1.1 What the methods are

| method | pairing | input | inference |
|---|---|---|---|
| **MMP analysis** (Hussain–Rea fragmentation; `mmpdb`) | exact single-substituent transforms | transform identity | median Δ of that transform, applied to the anchor |
| **Free–Wilson** | — | substituent indicator matrix | additive substituent contributions |
| **PADRE** (Tynes 2021) | all pairs | `[f(x_i), f(x_j), f(x_i)−f(x_j)]` | average Δ̂ over all anchors; variance of Δ̂ = uncertainty |
| **DeepDelta** (Fralish 2023) | all pairs (cross-merged *after* the CV split) | two molecules into a shared D-MPNN | Δ̂ directly; invariance tests as a free diagnostic |
| **Siamese + similarity pairing** (Zhang 2023) | pairs above a similarity cutoff 0.3–0.5 | twin subnetworks, shared weights | O(n) instead of O(n²) pairs |
| **SQRL** (Zadorozhny 2025, Genentech) | only pairs within a distance threshold α | `g(x_i) − g(x_new)` | `ŷ = (1/n)Σ_{i∈NN_n}[y_i + f(g(x_i)−g(x_new))]`, default **n = 1** |
| **MolPAIR** (2026) | MMP-first, then Morgan-similarity anchors | frozen TabPFN over paired features | global prediction + **gated** anchor-weighted error correction |

### 1.2 The evidence, with numbers

**DeepDelta** (10 ADMET sets, 253–1,321 compounds, 5×10-fold CV, 10 datasets + external test sets):
beats ChemProp and RF-ECFP on 70%/82% of benchmarks by Pearson r and 60%/73% by MAE, and on *all* external
test sets. Baseline ChemProp/RF median pairwise r is only 0.60/0.63.

The two numbers that matter for us are the subgroup results:

- **Top 10% of |Δ| — DeepDelta has the lowest error on 10/10 datasets** (p = 0.002 vs both baselines). Delta
  models genuinely own activity cliffs.
- **Bottom 10% of |Δ| — DeepDelta has the *highest* error** (p = 0.002). Its error-vs-|Δ| correlation is
  r ≈ 0.3 against 0.5 (ChemProp) and 0.6 (RF): it is flatter, which means it is relatively worse where |Δ| is small.

In a 10-member analog series most pairs have small |Δ|. A pure delta model therefore trades the common case
for the rare one. Under ST-RAE (per-compound L1) that trade is probably not profitable; under a
"find-the-cliff" objective it would be.

Also: scaffold-hopping pairs are predicted *as well as* same-scaffold pairs (p = 0.11), beating RF on 9/10 and
ChemProp on 8/10 — and crucially DeepDelta is only *statistically indistinguishable* from RF/ChemProp within
the same scaffold (p > 0.3 for r, p > 0.1 for MAE). **Same-scaffold is our entire test set.** Honest reading:
on our geometry, DeepDelta's published advantage is ~zero.

DeepDelta's three free invariance diagnostics are worth stealing regardless of whether you ship a delta model:
`D(x,x)=0` (MAE 0.127 ± 0.042), `D(x,y)=−D(y,x)` (r = −0.947 ± 0.044), `D(x,y)+D(y,z)=D(x,z)`
(MAE 0.127 ± 0.043). The same-molecule MAE correlates with CV MAE at **r = 0.916** — an unsupervised,
label-free predictor of model quality, computable on the 750 test compounds.

**PADRE** (Tynes 2021): 3.3–24.8% error reduction; the mechanism claimed is cancellation of systematic
method/assay error within a pair. The Fralish & Reker 2026 review pools the pairwise literature: across 274
benchmark datasets, error reductions "up to 60%" but with ranges that straddle zero —
Wetzel −1.6 to +60.2%, Zadorozhny −11.6 to +43.9%, Belaid −0.1 to +27.7%, Tynes +3.3 to +24.8%. On the 10
ADMET sets, paired beats unpaired on 80% (r) / 70% (MAE), and for *divergent scaffolds* the average gain is
"0.04 in Pearson's r" — moderate.

**SQRL** is the most important negative result in this literature for us. Spearman ρ, 30 MoleculeACE tasks,
Tanimoto distance threshold α = 0.7, Standard → SQRL:

| model | MoleculeACE | MoleculeACE-Cliff | Internal (5 industry targets) |
|---|---|---|---|
| XGBoost | 0.79 → **0.76** | 0.72 → **0.64** | 0.73 → **0.65** |
| RF | **0.80** → 0.77 | 0.72 → 0.68 | 0.70 → 0.68 |
| KNN | 0.66 → 0.67 | 0.57 → 0.59 | 0.52 → 0.52 |
| MLP | 0.32 → **0.73** | 0.23 → **0.62** | 0.39 → **0.59** |

Pairwise training **hurts every tree model** and **rescues the MLP from 0.32 to 0.73** — and still does not
reach RF's 0.80. The authors say it plainly: "most modern deep learning methods do not outperform conventional
baselines (with or without SQRL)". Their own stated conclusion is that *none* of the neural nets beat the
tree baselines. If our MLP ensemble is in the 0.32-like regime, SQRL is a large win; if it is already
competitive with a tuned LightGBM, it is not.

**Siamese with similarity-based pairing** (Zhang 2023, AstraZeneca), RMSE:

| model | Lipophilicity | FreeSolv | ESOL |
|---|---|---|---|
| MLP | 0.75 | 1.60 | 0.84 |
| MLP-ΔFP | 0.74 | 1.60 | 0.81 |
| MLP-SNN | **0.72** | **1.50** | **0.79** |
| RF-FP | 0.77 | 1.91 | 0.92 |
| RF-ΔFP | 0.74 | 1.62 | 0.83 |
| Chemformer (pretrained) | **0.58** | **1.07** | **0.58** |

Same shape again: pairing buys an MLP 4%, and the pretrained single-molecule model beats everything by 20%+
while Siamese *hurts* it. Similarity-based pairing also cuts pair construction from O(n²) to O(n), which is the
practical reason to prefer it: all-pairs on 4.9k compounds is 12M pairs.

### 1.3 The additivity ceiling

Delta/MMP/Free–Wilson all assume additivity. Two papers bound it:

- **Nonadditivity in public and in-house data** (Gogishvili 2021): significant nonadditivity events in **57.8%
  of AZ in-house assays** and **30.3% of public assays**; **9.4%** (AZ) / **5.1%** (public) of all compounds show
  significant additivity shifts. Experimental uncertainty is **0.3 log** (homogeneous) / **0.5 log**
  (heterogeneous); a double-transformation cycle counts as nonadditive above 2× that. And: *adding nonadditive
  data to training did not increase predictivity.*
- **Implications of Additivity and Nonadditivity for ML/DL** (ACS Omega 2022, AZ, 17k–220k compounds). Accuracy
  ranking is **invariant**: additive MMPs > all MMPs > all data > nonadditive MMPs. RMSE, best model per row:

| property | additive MMPs | nonadditive MMPs | ratio |
|---|---|---|---|
| log D | 0.23 | 0.51 | 2.2× |
| solubility | 0.42 | 0.77 | 1.8× |
| permeability | 0.28 | 0.64 | 2.3× |
| clearance | 0.25 | 0.42 | 1.7× |

Nonadditive subsets cost **1.7–2.3× the RMSE**, and deep learning is no exception. Your 46% of T ≥ 0.4 pairs
differing by > 1 log says you are in a nonadditivity-rich regime, which caps additive delta reasoning and
argues for the *gated* designs of §2 rather than unconditional delta prediction.

### 1.4 Implementation sketch — delta as an auxiliary head, with the leakage rule

DeepDelta's leakage rule is non-negotiable and is the most common way this gets silently broken:
**split molecules into folds first, then form pairs inside each fold.** A molecule must never appear in both a
training pair and a test pair.

```python
# Similarity-thresholded pairs, O(n·k) not O(n²) — Zhang 2023 / SQRL
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit import DataStructs
import numpy as np

def pairs_within(mols, y, fold_id, tmin=0.5, k=20):
    gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    out = []
    for i in range(len(fps)):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        sims[i] = -1
        cand = np.where(sims >= tmin)[0]
        for j in cand[np.argsort(-sims[cand])[:k]]:
            if fold_id[i] != fold_id[j]:      # <-- the leakage rule
                continue
            out.append((i, int(j), y[i] - y[j], float(sims[j])))
    return out
```

Train a second head on `Δy` from the *shared* trunk (multi-task with the absolute head), weight the Δ loss by
similarity, and at inference use it **only inside the gate of §2.2**. Keep the absolute head as the prediction
of record. Run DeepDelta's three invariance checks on the 750 test compounds as a label-free sanity gate.

---

## §2 Local / series-aware models

### 2.1 The classical result: error is a function of distance to the training set

Sheridan 2004 established, and the error-bar literature has repeatedly confirmed, that the two best predictors
of per-compound error are **the similarity to the nearest training molecule** and **the number of training
neighbours above a cutoff** — and that this holds largely *independently of descriptor and ML method*. This is
the justification for a similarity-gated correction and for similarity-binned validation. It is also the reason
your scaffold-split error estimate does not transfer: it is an estimate taken at a different NN similarity
(0.354) from the one you will be scored at (0.665).

### 2.2 MolPAIR (2026) — the closest published system to what we should build, and its ablation is the lesson

Architecture: a frozen tabular-foundation-model "global predictor" (CheMeleon features + TabPFN-3), plus a
"relation predictor" that predicts *error differences* between molecule pairs, used to correct the global
prediction. Paired features include signed and absolute feature differences, Morgan similarity, a shared-scaffold
indicator, a **one-cut MMP indicator**, the global prediction, and uncertainty summaries. Anchors are
MMP-first, then by Morgan similarity; ≤32 anchors per query, relation context ≤8,192 rows.

Anchor weights and the gate:

```
w̃_aq = max(s_aq, 1e-4)² / max(u_aq, 1e-4)          # similarity² / relation uncertainty
λ_q  = clip( (max_a s_aq − τ) / (1 − τ), 0, 1 ),  τ = 0.20
ŷ_q  = μ_q + λ_q · Σ_a w_aq · c_aq                  # gated correction on the global prediction
```

Evaluation: 58 tasks (30 MoleculeACE + 28 Polaris), structural-generalisation splits — hold out whole
Bemis–Murcko scaffold groups, **then additionally drop test molecules with max Morgan Tanimoto > 0.60**.
Headline: pooled Elo 1360 vs global 1246; wins 46/58 tasks; MoleculeACE mean RMSE 0.893 → **0.885**.

Table 3 (MolPAIR minus each control; positive favours MolPAIR) is the part to read:

| control | ΔElo [95% CI] | W/L | RMSE reduction [95% CI] |
|---|---|---|---|
| Global baseline | 295.7 [209.6, 393.9] | 46/12 | 0.00819 [0.00235, 0.01388] |
| Two-global-TFM ensemble | 209.6 [111.1, 321.3] | 44/14 | 0.00547 [−0.00032, 0.01093] |
| **kNN residual** | 112.0 [37.9, 191.7] | 41/17 | **0.00402 [−0.00110, 0.00930]** |
| Single-molecule residual | 160.9 [78.7, 253.9] | 43/15 | 0.00768 [0.00251, 0.01295] |
| **Shuffled relation labels** | 88.9 [14.0, 168.8] | 41/17 | **0.00405 [−0.00100, 0.00947]** |
| **No reliability gate** | **325.2 [235.3, 439.8]** | 45/13 | **0.03443 [0.01947, 0.04956]** |

Read it carefully:

- The **whole method is worth 0.008 RMSE (0.9%)** over its global predictor. Small.
- Against a plain **kNN residual correction**, the RMSE interval **includes zero**. The learned pairwise relation
  model does not reliably beat a weighted average of neighbours' residuals.
- Against **shuffled pair labels**, the RMSE interval **includes zero** too — so most of the benefit is not coming
  from the pair labels being correct.
- Removing the **reliability gate costs 0.0344 RMSE — four times the method's entire advantage.**

**The dominant ingredient is the shrinkage/gating, not the pairwise learning.** That is the same conclusion §0.1
reaches from your own `r = 0.846`, and it is consistent with this programme's earlier finding that shrinkage is
correct. Build the gate; treat the learned relation model as optional garnish.

They also report the kNN-residual control's own anchor logic ("weighted average of stored prediction errors from
similar labeled molecules"), which is exactly item 2 on the summary list and gets you within noise of the
full system.

### 2.3 Gaussian processes with Tanimoto kernels

GAUCHE (NeurIPS 2023) provides batch-GP-compatible implementations of the Tanimoto kernel and 12 other
bit/count-vector kernels:

```
k_Tanimoto(x, x') = σ_f² · ⟨x,x'⟩ / (‖x‖² + ‖x'‖² − ⟨x,x'⟩)
```

RMSE / NLPD, mean ± s.e. over 20 random 80/20 splits:

| kernel | repr. | Photoswitch | ESOL | FreeSolv | Lipophilicity |
|---|---|---|---|---|---|
| Tanimoto | **fragprints** | **20.9±0.7 / 0.22±0.03** | **0.71±0.01 / 0.33±0.01** | **1.31±0.06 / 0.28±0.02** | **0.67±0.01 / 0.71±0.01** |
| Tanimoto | fingerprints | 23.4±0.8 / 0.33±0.03 | 1.01±0.01 / 0.71±0.01 | 1.93±0.09 / 0.58±0.03 | 0.76±0.01 / 0.85±0.01 |
| Scalar product | fragprints | 22.5±0.7 / 0.23±0.03 | 0.88±0.01 / 0.53±0.01 | 1.27±0.02 / 0.25±0.02 | 0.77±0.01 / 0.92±0.01 |
| Scalar product | fingerprints | 24.8±0.8 / 0.33±0.03 | 1.17±0.01 / 0.84±0.01 | 1.93±0.07 / 0.64±0.03 | 0.84±0.01 / 1.03±0.01 |

Findings: Tanimoto > scalar product on both RMSE and NLPD; **fragprints** (fingerprints ⊕ fragment counts) is
the representation to use; deep probabilistic models match on RMSE but their **uncertainty is consistently less
reliable**. For small-to-mid datasets the authors recommend Tanimoto + fragprints outright. Exact GP inference is
fine at our n ≈ 4,900 (and fitting on residuals keeps the mean function in the ensemble).

Relevance to ST-RAE: the metric pays for calibration, since anything inside the credible interval is free. A GP
with good NLPD gives per-compound predictive variance that doubles as the gate's confidence term.

### 2.4 Implementation sketch — series-level residual offset (the recommended build)

The test set is 75 series × 10, each series an ECFP4 neighbourhood of a parent that is in our training data.
Decompose `y = (series level) + (within-series deviation)`. The parent's label pins the series level; the model
ranks within it. Fit the shrinkage empirically rather than guessing it.

```python
import numpy as np

def fit_offset_params(e_oof, series_id):
    """Variance components of out-of-fold residuals grouped into analogue series."""
    groups = [e_oof[series_id == s] for s in np.unique(series_id)]
    groups = [g for g in groups if len(g) >= 2]
    sw2 = np.mean([g.var(ddof=1) for g in groups])            # within-series
    means = np.array([g.mean() for g in groups])
    tau2 = max(means.var(ddof=1) - sw2 / np.mean([len(g) for g in groups]), 1e-6)
    return tau2, sw2

def series_offset(e_anchors, tau2, sw2, reliability=1.0, sim_gate=1.0):
    """Empirical-Bayes shrunk offset from n labelled anchors in the same series."""
    n = len(e_anchors)
    if n == 0:
        return 0.0
    lam = n * tau2 / (n * tau2 + sw2)            # James-Stein / EB weight
    return sim_gate * reliability * lam * float(np.mean(e_anchors))

# y_hat = y_global + series_offset(...)
```

`λ` for representative variance ratios (λ = n·τ²/(n·τ² + σ_w²)):

| τ² / σ_w² | n=1 | n=2 | n=5 |
|---|---|---|---|
| 0.10 / 0.90 | 0.100 | 0.182 | 0.357 |
| 0.25 / 0.75 | 0.250 | 0.400 | 0.625 |
| 0.40 / 0.60 | 0.400 | 0.571 | 0.769 |
| 0.50 / 0.50 | 0.500 | 0.667 | 0.833 |

Multiply by MolPAIR's similarity gate `λ_q = clip((max_a s − τ)/(1−τ), 0, 1)` with τ ≈ 0.2–0.4, and by the
§0.4 reliability factor `σ²/(σ²+s²)` computed from each anchor's credible-interval width. Three
shrinkages, each measured, none guessed. Clip the total offset (e.g. ±1 log) so that a single mis-measured
parent cannot move a whole series.

### 2.5 Transductive / test-time options

Legitimate and cheap: the 750 test structures are known, so you can (a) recover the 75-series block structure by
clustering and use it to *share* an offset across each block's members rather than estimating 750 independent
corrections — a 10× variance reduction on the offset; (b) use the test fingerprint distribution to reweight
validation folds so the measured ST-RAE is estimated on a matched covariate distribution. Published test-time
adaptation for molecular property prediction remains thin and unconvincing at this scale; the block-structure
and calibration uses above are standard and carry essentially no risk.

---

## §3 Activity cliff benchmarks

### 3.1 MoleculeACE (van Tilborg, Alenicheva & Grisoni, JCIM 2022)

24 ML/DL approaches × 30 ChEMBL targets, >35,000 molecules, Ki/EC50.

- **Cliff definition:** similarity ≥ 0.90 by at least one of three measures (ECFP Tanimoto; scaffold-ECFP
  Tanimoto; scaled SMILES Levenshtein) **and** > 10× potency difference. Note how much stricter this is than
  our T ≥ 0.4–0.5 working definition. They deliberately excluded MMP-based cliffs, which would have nearly
  doubled the cliff count while covering 86.6% of theirs.
- **Split:** spectral clustering on ECFP, then 80/20 **stratified by cliff label** per cluster. Cliff compounds
  have 2.7 ± 0.9 cliff partners; in 9.1 ± 5.3% of cases all of a compound's partners land in test, yet
  train/test NN similarity stays high (substructure 0.80 ± 0.03, scaffold 0.93 ± 0.02, SMILES 0.95 ± 0.01).
  **This is the right split recipe to copy for a cliff-aware fold** — it keeps similarity high while varying activity.
- **Results:** RMSE_cliff 0.62–1.60 log (classical ML) and 0.68–1.44 log (deep learning). **Descriptor-based ML
  beats deep learning**; SVM + ECFP is best on average; among neural nets, **MLP + ECFP beats all SMILES/graph
  models**. RMSE and RMSE_cliff correlate at r = 0.81 but diverge by up to **0.54 log** — so overall error is a
  decent but not sufficient proxy. SVM-ECFP's mean RMSE_cliff − RMSE gap is only 0.094 log, rising to 0.39 on JAK1.
- **Cliff prevalence across targets: 7% (JAK1) to 52% (OX2R).** Our 46% of close pairs crossing 1 log puts the
  CYP panel at the hard end of this range — useful calibration, not an anomaly.
- Their recommendation: report cliff-specific metrics separately during model selection. Do that: compute
  ST-RAE overall **and** ST-RAE restricted to cliff-involved validation compounds.

### 3.2 Follow-ups

- **ACA / ACANET** (Nature Communications 2026, 52 datasets): regression loss + soft-margin triplet contrastive
  loss, mining activity-cliff triplets on the fly. Test RMSE reductions vs the same backbone without ACA:
  **PNA 5.81%, GAT 5.48%, GCN 4.63%, GIN 6.62%**; average **7.16%** across 9 low-sample narrow-scaffold sets
  with MAE as the base loss; 7.54% and 21.6% over baselines on the high-sample mixed-scaffold sets; label
  incoherence 0.290 → 0.199. Two reusable findings: **MAE beats MSE as the base loss on narrow-scaffold series**,
  and gains scale with the *number of mined triplets* (RMSE strongly negatively correlated with M). Caveat: the
  comparisons against SVM/RF/GBM/MLP/KNN on ECFP include "n.s." cells.
- **ACtriplet** (2025): triplet loss + pretraining, 30 benchmarks; "significantly better than those DL models
  *without* pre-training" — note the comparator is other DL, not tuned trees.
- **Activity cliffs from bioactivity profiles** (Abe, Miyao & Bajorath, J Cheminform 2026) — the scepticism
  paper, and the one to quote internally. Reported cliff-prediction accuracies of "80% or above" in the
  literature largely reflect **compound memorisation and train/test leakage**; under partitioning schemes
  designed to control leakage, "AC prediction accuracy based on chemical structure is low or moderate at best",
  and the accurate subsets were traceable to **one or at most a few nearest-neighbour profile assays** that
  simple NN search would have found. Treat every published cliff-prediction headline as leakage until the split
  is shown.

**Net:** no method reliably "solves" cliffs. The achievable move is to (a) stop averaging them away (median/L1),
(b) widen intervals where cliff risk is high and let the dead zone absorb them, (c) use the cliff-stratified
metric during model selection so you do not pick a model that is good on the easy 54%.

---

## §4 Few-shot / in-context molecular property prediction

**Verdict: wrong regime for us.** FS-Mol-style few-shot is "16–128 labels on a *new* target"; we have ~4,900
dose-response labels plus a 4,376-compound screen on *these four* targets. The methods are nonetheless worth
knowing for what they reveal about anchors.

FS-Mol (5,000 tasks, 40 validation / 157 test tasks), ΔAUC-PR at support set size 16 → 128:

| method | \|S\|=16 | 32 | 64 | 128 |
|---|---|---|---|---|
| ADKF-IFT | 0.231±0.009 | 0.263±0.010 | 0.287±0.010 | 0.318±0.010 |
| Quadratic probe (plain fine-tuning) | 0.227±0.010 | 0.255±0.010 | 0.279±0.010 | 0.310±0.010 |
| Linear probe | 0.224±0.010 | 0.252±0.010 | 0.273±0.010 | 0.301±0.009 |
| ProtoNet | 0.206±0.009 | — | — | — |
| GNN-MAML | 0.160±0.009 | — | — | — |
| PAR | 0.130±0.009 | — | — | — |
| Random forest | 0.093±0.007 | — | — | — |
| **kNN** | **0.051±0.005** | — | — | — |

- **MHNfs** (Schimunek et al., ICLR 2023) set the FS-Mol SOTA at ΔAUC-PR **0.241 ± 0.009**, vs IterRefLSTM
  0.234 ± 0.010 and ADKF-IFT 0.234 ± 0.009 — i.e. the Hopfield context module buys ~0.007, inside a
  standard error of its rivals. Elaborate few-shot architecture, marginal gain.
- **"A Strong Baseline for Molecular Few-Shot Learning"** (TMLR 2025): simple linear/quadratic probes on a
  pretrained backbone, with shrinkage regularisation λ = 0.2, are within one standard error of meta-learning —
  and **beat it under domain shift and class imbalance** (DAVIS, imbalance 0.21: quadratic probe 39.2 ± 1.8 vs
  ADKF-IFT 36.5 ± 3.1). Meta-training is not necessary.
- **The number to carry away: kNN scores 0.051 where learned methods score 0.22–0.32.** Nearest-neighbour
  retrieval is a weak *primary* predictor even in the regime designed to flatter it. Anchors belong in a
  correction term, never as the prediction of record.

**TabPFN-style in-context learners** are the one strand that *is* relevant — but as the backbone of MolPAIR
(§2.2), not as few-shot learners. A frozen TabPFN over molecular embeddings, with pair features as extra
columns, beat tuned XGBoost/LightGBM/RF (≈0.92–0.94 RMSE, Elo 1000–1026), fine-tuned CheMeleon (0.930), tuned
MiniMol (0.931), tuned Chemprop (1.046) and iMoLD (1.080) on 58 structural-shift tasks — with positive Elo gains
in **all twelve** representation × backbone combinations (+97.7 to +314.1). Context caps: ≤2,000 features
globally, ≤8,192 relation rows. At n ≈ 5k this is directly runnable and is the cheapest route to a strong,
hyperparameter-free second opinion for blending.

---

## §5 Building a validation split that mimics analog expansion

### 5.1 Why the current split cannot answer the question

Test internal mean NN similarity **0.665** vs scaffold-split validation **0.354**, with 84% of test compounds
close analogs of a training active. Those are different kinds of set, and a number measured on one does not
transfer to the other. This is the generic failure SIMPD was written about: **random splits are systematically
optimistic, neighbour/scaffold splits systematically pessimistic.**

### 5.2 SIMPD (Landrum et al., J Cheminform 2023) — the method to copy

Derived from temporal splits of **138 Novartis lead-optimisation projects**; a NSGA-II multi-objective genetic
algorithm (pymoo) re-partitions a dataset to hit eight objectives:

| # | objective | target |
|---|---|---|
| 1 | Δ test−train median(SA_Score) | 0.28 |
| 2 | Δ test−train median(HeavyAtomCount) | 3.1 |
| 3 | Δ test−train median(TPSA) | 13.2 |
| 4 | Δ test−train median(fr_benzene / 1000 heavy atoms) | −8.8 |
| 5 | frac active (train) | from the data |
| 6 | frac active (test) | from the data |
| 7 | **ΣG − ΣF′ (spatial statistics)** | 10 < · < 30 |
| 8 | **ΣG (test-set internal clustering)** | > 70 |

Constraints: test = 20% of compounds; relative cluster-population entropy of the test set < 0.9. Objectives 7–8
are up-weighted, because "the relationship of the chemical similarities *within* the test set and *between*
training and test sets are an essential part of what makes temporal splits different". Late-project compounds
are larger, more polar, harder to make, and less benzene-rich. Random splits sit above the ΣG/ΣF′ diagonal,
neighbour and temporal splits below it. SIMPD splits remain mildly optimistic vs temporal but are significantly
closer than random. 99 public ChEMBL datasets released; code at
[github.com/rinikerlab/molecular_time_series](https://github.com/rinikerlab/molecular_time_series).

One aside from that paper worth using: they tested simply **averaging random-split and neighbour-split
performance** as a temporal-split estimate and found it "does indeed lead to error estimates which tend to be
closer to the temporal splits than either of the starting points". They declined to pursue it as *ad hoc*. For
us it is a free, five-minute sanity bracket: your true ST-RAE should sit between the two.

MolPAIR's protocol is the other template: **hold out whole Bemis–Murcko scaffold groups, then drop test
molecules whose max Morgan Tanimoto to train exceeds a cutoff** — and they verify robustness across cutoffs
0.40–0.70 and against DataSAIL C1 splits. We want the *inverse* of that filter.

### 5.3 The split to build: a hit-expansion split

Construct it to reproduce the test set's geometry, then *verify* the reproduction numerically.

```python
# Sketch: simulate "top-25 hits × 10 catalogue analogs" inside the training set.
# 1. Per CYP, take the top-N most potent training compounds as pseudo-parents.
# 2. For each pseudo-parent, take its 10 nearest ECFP4 training neighbours -> held-out series.
# 3. KEEP the pseudo-parent in training (the real parents are in our training set).
# 4. Verify the split's geometry against the real test set before using it:
#      a. mean within-held-out NN Tanimoto          ~= 0.665
#      b. fraction of held-out with a train NN >= t  ~= 0.84
#      c. fraction of held-out close pairs > 1 log   ~= 0.46
#      d. potency distribution skewed potent (analogs of actives)
#      e. SIMPD spatial statistics SumG, SumG - SumF'
# 5. Repeat with disjoint parent sets to get several folds; report ST-RAE mean and spread.
```

Report four numbers side by side every time: ST-RAE on (i) random split, (ii) scaffold split, (iii) hit-expansion
split, (iv) hit-expansion split restricted to cliff-involved compounds. **Only (iii) and (iv) are decision-relevant**;
(i) and (ii) bracket it and catch bugs.

**Expected gaps.** Taking the direction and rough magnitudes from SIMPD (random optimistic, neighbour pessimistic),
MoleculeACE (RMSE_cliff − RMSE up to 0.54 log, typically 0.1–0.4) and the additivity study (nonadditive subsets
1.7–2.3× additive-subset RMSE): expect the hit-expansion split to land **better than your scaffold split** (higher
NN similarity) but with **a much heavier tail** (cliff- and nonadditivity-rich). Do not be reassured by a lower
mean; look at the ST-RAE of the worst-decile series, because 75 series × 10 means a handful of blown series
dominates the total.

---

## What NOT to do

1. **Don't let an anchor carry un-shrunk weight.** Your own `r = 0.846` at T ≥ 0.5 implies copy-the-anchor is
   **20% worse than predicting the mean**. The break-even is `r < 0.707` (ρ > 0.5). Gate on measured per-bin r.
2. **Don't form pairs before splitting.** Split molecules into folds, then pair within folds (DeepDelta's rule).
   A molecule appearing in both a training pair and a test pair invalidates the entire comparison.
3. **Don't expect a pairwise/delta model to beat a tuned tree baseline.** SQRL *degraded* XGBoost (0.79→0.76) and
   RF (0.80→0.77) on MoleculeACE and all five industry targets; its own authors report no neural net beating the
   tree baselines. Siamese pairing *hurt* the pretrained Chemformer. Pairwise learning rescues weak models.
4. **Don't replace the global model with a delta model.** DeepDelta is *statistically indistinguishable* from
   RF/ChemProp within the same scaffold (p > 0.3 r, p > 0.1 MAE) — and same-scaffold is our entire test set —
   while being **worst on the bottom 10% of |Δ|**, which is most pairs in a 10-member series.
5. **Don't believe published activity-cliff prediction accuracies.** Bajorath's 2026 analysis attributes the
   ">80% accuracy" literature to compound memorisation and leakage; under leakage-controlled partitioning,
   structure-based cliff prediction is "low or moderate at best", and the accurate subsets were reproducible by
   plain nearest-neighbour search.
6. **Don't optimise or select on RMSE, and don't mean-aggregate the ensemble.** ST-RAE is L1 with a dead zone;
   the optimum is a median-like balance point (§0.3). Use L1/Huber/pinball-0.5 and a median aggregator. ACA's own
   ablation found MAE > MSE on narrow-scaffold series.
7. **Don't build a few-shot / meta-learning / in-context-as-few-shot pipeline.** Wrong regime, and
   meta-training is not even necessary within its own regime (plain probes match ADKF-IFT and beat it under
   domain shift). kNN scores 0.051 ΔAUC-PR on FS-Mol where learned methods score 0.22–0.32.
8. **Don't trust your scaffold-split ST-RAE as an estimate of the leaderboard.** 0.354 vs 0.665 mean NN
   similarity is a different kind of set. This has already cost this programme once.
9. **Don't dump the nonadditive pairs into training hoping the model absorbs them.** Gogishvili et al. tried
   exactly that across increasing NA fractions and got no improvement in predictivity.
10. **Don't use an all-pairs expansion on 4.9k compounds** (~12M pairs) when similarity-thresholded pairing is
    O(n·k), demonstrably better, and the SQRL ablation shows indiscriminate pairing is *worse* than thresholded.
11. **Don't borrow CYP2C9's anchor weight from the other isoforms.** You have almost no CYP2C9 pairs; an
    unmeasured ρ_r is not a shared ρ_r. Gate it to zero until measured, and let the global model carry it.
12. **Don't evaluate a pair model with pair-level metrics and infer single-compound gains.** Nearly every
    headline in this literature (Pearson r on Δ, % of datasets won) is a pair-level or rank-level claim. MolPAIR's
    295 Elo points over its global baseline are **0.9% RMSE**.

---

## Bibliography

**Pairwise / delta learning**
- Fralish, Chen, Skaluba & Reker. *DeepDelta: predicting ADMET improvements of molecular derivatives with deep learning.* J Cheminform 15:101 (2023). https://doi.org/10.1186/s13321-023-00769-x · code https://github.com/RekerLab/DeepDelta
- Fralish & Reker. *Pairwise learning for molecular property prediction and optimization.* Front Drug Discov (2026). https://doi.org/10.3389/fddsv.2026.1859068
- Tynes, Gao, Burrill, Batista, Perez, Yang & Lubbers. *Pairwise Difference Regression: a machine learning meta-algorithm for improved prediction and uncertainty quantification in chemical search.* J Chem Inf Model 61(8):3846–3857 (2021). https://doi.org/10.1021/acs.jcim.1c00670 · open version https://www.osti.gov/pages/biblio/1811954
- Zadorozhny, Chuang, Sathappan, Wallace, Sresht & Grambow (Prescient Design, Genentech). *Similarity-Quantized Relative Difference Learning for Improved Molecular Activity Prediction.* NeurIPS MLSB Workshop (2024). https://arxiv.org/abs/2501.09103
- Zhang, Menke, He, Nittinger, Tyrchan, Koch & Zhao (AstraZeneca). *Similarity-based pairing improves efficiency of siamese neural networks for regression tasks and uncertainty quantification.* J Cheminform 15:75 (2023). https://doi.org/10.1186/s13321-023-00744-6
- *Molecular Property Prediction under Structural Shift with Tabular Foundation Models* (MolPAIR). arXiv (2026). https://arxiv.org/abs/2609.38744
- McNutt & Koes. *Improving ΔΔG predictions with a multitask convolutional Siamese network.* J Chem Inf Model 62:1819–1829 (2022). https://doi.org/10.1021/acs.jcim.1c01497
- Turk, Merget, Rippmann & Fulle. *Coupling matched molecular pairs with machine learning for virtual compound optimization.* J Chem Inf Model 57(12):3079–3085 (2017). https://pubmed.ncbi.nlm.nih.gov/29131617/
- de la Vega de León & Bajorath. *Prediction of compound potency changes in matched molecular pairs using support vector regression.* J Chem Inf Model (2014). https://pubmed.ncbi.nlm.nih.gov/25191787/
- Hussain & Rea. *Computationally efficient algorithm to identify matched molecular pairs (MMPs) in large data sets.* J Chem Inf Model 50:339–348 (2010). https://doi.org/10.1021/ci900450m
- Note: no distinct "DeltaProp" paper surfaced in 503 screened abstracts; "delta learning" in this sense is
  covered by DeepDelta / PADRE / SQRL above, and Δ-ML in the quantum-chemistry sense (Ramakrishnan et al.,
  https://arxiv.org/abs/1503.04987) is a different construction (baseline + learned correction, same molecule).

**Additivity ceiling**
- Gogishvili, Nittinger, Margreitter & Tyrchan. *Nonadditivity in public and inhouse data: implications for drug design.* J Cheminform 13:47 (2021). https://doi.org/10.1186/s13321-021-00525-z
- Nittinger, Tyrchan et al. *Implications of Additivity and Nonadditivity for Machine Learning and Deep Learning Models in Drug Design.* ACS Omega 7(31) (2022). https://doi.org/10.1021/acsomega.2c02738 · PMC9352238
- Kramer et al. — experimental-uncertainty framework (0.3 log homogeneous / 0.5 log heterogeneous) underpinning nonadditivity analysis; see refs within the two above. Nonadditivity code: https://github.com/KramerChristian/NonadditivityAnalysis

**Activity cliffs**
- van Tilborg, Alenicheva & Grisoni. *Exposing the Limitations of Molecular Machine Learning with Activity Cliffs.* J Chem Inf Model 62(23):5938–5951 (2022). https://doi.org/10.1021/acs.jcim.2c01073 · open PDF https://pure.tue.nl/ws/files/257498223/acs.jcim.2c01073.pdf · code https://github.com/molML/MoleculeACE
- *Activity-cliff awareness enables robust graph learning for molecular property prediction* (ACA / ACANET). Nature Communications (2026). PMC13534492 · https://pmc.ncbi.nlm.nih.gov/articles/PMC13534492/
- *ACtriplet: an improved deep learning model for activity cliffs prediction by integrating triplet loss and pre-training* (2025). PMC12398830 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12398830/
- Abe, Miyao & Bajorath. *Accurate prediction of activity cliff compounds based on bioactivity profiles depends on assay nearest neighbor relationships.* J Cheminform 18 (2026). https://doi.org/10.1186/s13321-026-01210-9 · PMC13231687
- Vogt, Bajorath et al. *Strong nonadditivity as a key structure–activity relationship feature: distinguishing structural changes from assay artifacts.* J Chem Inf Model (2015). https://doi.org/10.1021/acs.jcim.5b00018

**Local models, GPs, applicability domain**
- Griffiths et al. *GAUCHE: a library for Gaussian processes in chemistry.* NeurIPS (2023). https://arxiv.org/abs/2212.04450 · proceedings PDF https://proceedings.neurips.cc/paper_files/paper/2023/file/f2b1b2e974fa5ea622dd87f22815f423-Paper-Conference.pdf · code https://github.com/leojklarner/gauche
- Sheridan, Feuston, Maiorov & Kearsley. *Similarity to molecules in the training set is a good discriminator for prediction accuracy in QSAR.* J Chem Inf Comput Sci 44(6) (2004). https://www.semanticscholar.org/paper/1bf3e53dd6ecb62c6132ef62b9a987a8d5eadb6f
- Liu & Wallqvist. *General approach to estimate error bars for QSAR predictions.* (2018). https://bhsai.org/pubs/Liu_2018_General_Approach_to_Estimate_Error.pdf
- Sahigara, Mansouri, Ballabio, Mauri, Consonni & Todeschini. *Comparison of different approaches to define the applicability domain of QSAR models.* Molecules 17(5):4791–4810 (2012). https://doi.org/10.3390/molecules17054791 · PMC6268288
- Hu, Bajorath et al. *Context-dependent similarity analysis of analogue series for structure–activity relationship transfer.* J Cheminform (2025). PMC11734349 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11734349/

**Splits**
- Landrum, Beckers, Lanini, Schneider, Stiefl & Berthold. *SIMPD: an algorithm for generating simulated time splits for validating machine learning approaches.* J Cheminform 15:119 (2023). https://doi.org/10.1186/s13321-023-00787-9 · PMC10712068 · code/data https://github.com/rinikerlab/molecular_time_series
- `splito` SIMPD implementation (Datamol): https://splito-docs.datamol.io/stable/api/simpd.html
- Sheridan. *Time-split cross-validation as a method for estimating the goodness of prospective prediction.* J Chem Inf Model 53(4) (2013). https://doi.org/10.1021/ci400084k

**Few-shot / in-context**
- Stanley et al. *FS-Mol: a few-shot learning dataset of molecules.* NeurIPS Datasets & Benchmarks (2021). https://github.com/microsoft/FS-Mol
- Schimunek, Seidl, Friedrich, Kuhn, Rippmann, Hochreiter & Klambauer. *Context-enriched molecule representations improve few-shot drug discovery* (MHNfs). ICLR (2023). https://openreview.net/pdf?id=XrMWUuEevr · code https://github.com/ml-jku/MHNfs
- Schimunek et al. *MHNfs: prompting in-context bioactivity predictions for low-data drug discovery.* J Chem Inf Model (2025). https://doi.org/10.1021/acs.jcim.4c02373
- Formont et al. *A Strong Baseline for Molecular Few-Shot Learning* / *Is Meta-training Really Necessary for Molecular Few-Shot Learning?* TMLR (2025). https://arxiv.org/abs/2404.02314
- Seidl et al. *Enhancing activity prediction models in drug discovery with the ability to understand human language* (CLAMP). ICML (2023). https://arxiv.org/abs/2303.03363 · code https://github.com/ml-jku/clamp
- Chen et al. *Meta-learning adaptive deep kernel Gaussian processes for molecular property prediction* (ADKF-IFT). ICLR (2023). https://arxiv.org/abs/2205.02708

**CYP context**
- *Advancing ADMET prediction for major CYP450 isoforms: graph-based models, limitations, and future directions.* (2025). PMC12288271 · https://pmc.ncbi.nlm.nih.gov/articles/PMC12288271/
