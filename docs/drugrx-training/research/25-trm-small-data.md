# 25 — TRM / HRM in the extreme-small-data regime, and what actually wins on 4,376 molecules

Compiled 2026-10-08. Companion to `research/10-recursive-and-adaptive-depth.md`, which already has the
exact mechanisms (TRM/HRM algorithm boxes, repo source, ACT, PonderNet, Tab-TRM, the ARC Prize
ablation). **This file does not repeat that.** It answers the three things file 10 left open:

1. **The epoch/augmentation crux.** What TRM's "60,000 / 100,000 epochs" actually is in optimiser
   steps and in presentations-per-distinct-input, and the apples-to-apples conversion to 4,376
   molecules. (§1 — this is the part that changes a decision.)
2. **A fresh, exhaustive check** that nobody has applied recursive / adaptive-depth latent
   computation to molecular property regression (§2), plus the 2025-12 → 2026-10 TRM follow-ups
   that audit *where* its performance comes from (§3).
3. **Ranked measured effect sizes** for the levers that do work at n ≈ 1k–10k (§4), and the
   concrete arm list (§5).

Verification key: **[V]** = read this session in the paper/abstract/repo source and quoted;
**[E]** = my arithmetic or estimate, derivation shown; **[C10]** = carried from `research/10`,
verified there; **[S]** = search-index summary only, *not* confirmed in the paper text — treat as a
lead, not a number.

---

## 0. The three sentences

- **TRM's "3,000 / 60,000 / 100,000 epochs" is not a long-training result.** An "epoch" in that
  codebase is **1.3 optimiser steps on Sudoku and ~4.1 on ARC**, and the thing that is large is not
  the number of passes per distinct input (**3.7–100**, the same order as our current 20–65) but the
  **number of distinct inputs: 1,000 exact label-preserving views per base example**. We have **one**
  view per molecule. The entire TRM budget story is an augmentation story. [V + E, §1]
- **Nobody has applied this family to molecular property regression.** arXiv `all:"tiny recursive
  model"` returns **26 papers**, of which **exactly one** is tabular (insurance pricing) and **zero**
  are chemical/biological. The intersection query (recursive reasoning | latent reasoning | adaptive
  depth | deep supervision) ∧ (molecular property | QSAR | ADMET | bioactivity | IC50) returns
  **`totalResults = 0`**. [V, §2]
- **The two biggest measured effects found anywhere in this sweep, at our exact data scale, are
  ensembling-shaped, not reasoning-shaped**: a parameter-efficient 32-member implicit ensemble is the
  best tabular DL method over 46 datasets, and test-time retrieval fusion cuts RMSE by a **median
  19.4%** across 14 molecular regression sets of 340–8,745 compounds. Both are cheap. Spend the GPU
  there. [V, §4]

---

## 1. The epoch crux: what TRM's training budget really is, in our units

### 1.1 The three numbers the paper gives

From the paper text [V, `arxiv.org/html/2510.04871v1`], verbatim:

| Dataset | Base training data | Augmentation | Epochs | Batch | Compute |
|---|---|---|---|---|---|
| Sudoku-Extreme | "only 1K training samples are used to test small-sample learning" (test: 423K) | "1000 shuffling (done without breaking the Sudoku rules) augmentations per data example" | "we train for 60k epochs" | 768 | "1 L40S … generally less than 36 hours" |
| Maze-Hard | "both the training set and test set include 1000 mazes" | "8 dihedral transformations per data example" | 60k | 768 | "4 L40S … less than 24 hours" |
| ARC-AGI-1 / -2 | "ARC-AGI-1 contains 800 tasks, while ARC-AGI-2 contains 1120 tasks. We also augment our data with the 160 tasks from the closely related ConceptARC dataset" | "1000 data augmentations (color permutation, dihedral-group, and translations transformations) per data example" | "For ARC-AGI, we train for 100K epochs" | 768 | "around 3 days with 4 H100" |

Test-time: "at test-time, the most common answer out of the 1000 data augmentations is given as
answer." EMA: "TRM uses an Exponential Moving Average (EMA) of 0.999" (the committed
`cfg_pretrain.yaml` ships `ema: False`, `ema_rate: 0.999` — the ablation says no-EMA costs 87.4 →
79.9 on Sudoku) [V, both].

**So where does "3,000 epochs" come from?** Not from this paper. The published figures are 60k
(Sudoku/Maze) and 100k (ARC). If a 3,000 figure is in circulation it is either a secondary summary,
a shorter reproduction, or the Sudoku *quickstart*. Either way the number is meaningless until
converted, because of §1.2.

### 1.2 An "epoch" in this codebase is 1–4 optimiser steps

`pretrain.py`, verbatim [V]:

```python
total_steps = int(config.epochs * train_metadata.total_groups *
                  train_metadata.mean_puzzle_examples / config.global_batch_size)
```

and the loader, verbatim [V, `puzzle_dataset.py`]:

```python
group_order = np.concatenate([rng.permutation(dataset["group_indices"].size - 1)
                              for _i in range(self.config.epochs_per_iter)])
...
puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
```

A **group** is one base example together with all of its augmentations; a **puzzle** is one augmented
copy; an **example** is one (input, output) pair inside it [V, `dataset/build_arc_dataset.py`:
`"num_aug: int = 1000"`, `group = [converted]` then augmentations appended,
`mean_puzzle_examples = total_examples / total_puzzles`].

**One epoch therefore iterates over base examples (groups), sampling one random augmentation from
each — not over the augmented corpus.** That single fact is the whole conversion.

| | Sudoku-Extreme | ARC-AGI-1 |
|---|---|---|
| `total_groups` | 1,000 [V] | ≈ 960 (400 train + 400 eval + 160 ConceptARC) [E] |
| `mean_puzzle_examples` | 1 (one grid pair per sudoku) [E] | ≈ 3.3 (mean ARC demonstration pairs/task) [E] |
| examples per epoch | 1,000 | ≈ 3,168 |
| **batches ("steps") per epoch** @ 768 | **1.30** | **4.13** |
| epochs | 60,000 [V] | 100,000 [V] |
| **total optimiser steps** | **78,125** | **≈ 412,500** |

Sanity check against the stated wall clock: 78,125 steps in <36 h on one L40S = 0.6 steps/s at batch
768 × seq 81 with 42 effective layer-applications; 412,500 steps in 72 h on 4×H100 = 1.6 steps/s at
batch 768 × seq ≈ 900. Both plausible, so the arithmetic is not off by an order of magnitude [E].

### 1.3 Deep supervision does *not* multiply optimiser steps the way file 10 feared — it divides data

File 10 §9.5 warned that `opt.step()` sits inside the `N_sup` loop, giving the recursive arm up to
16× the updates at equal epochs. The *paper's* pseudocode does read that way, but **the shipped
trainer does one optimiser step per supervision segment and persists the carry across loader
batches** [V, `pretrain.py`: "The optimizer step occurs outside any `halt_max_steps` loop"], with
this in the model's forward [V, `models/recursive_reasoning/trm.py`]:

```python
new_inner_carry  = self.inner.reset_carry(carry.halted, carry.inner_carry)
new_steps        = torch.where(carry.halted, 0, carry.steps)
new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,)*(batch[k].ndim-1)), batch[k], v)
                    for k, v in carry.current_data.items()}
```

Read it carefully: **for a batch slot that has not halted, the freshly loaded example is thrown
away** and the slot keeps refining its old example. So one optimiser step = one segment, and the
model's real data consumption is `total_steps × 768 / E[segments]`, with `E[segments] ∈ [1, 16]`
(`halt_max_steps = 16`). The confound is still real but it points the other way: a deep-supervised
model takes **E[segments]× more optimiser steps per distinct example consumed**. Any comparison must
be matched on **optimiser steps**, not epochs, and not examples. [V + E]

### 1.4 The conversion table — this is the answer to the question

Define **presentations per distinct input** = (optimiser steps × batch) ÷ (number of distinct
label-preserving input vectors), then divide by `E[segments]` for the deep-supervised arms.

| | TRM Sudoku | TRM ARC-AGI-1 | **CYP, current recipe** | **CYP, "3,000 epochs"** |
|---|---|---|---|---|
| base training examples | 1,000 | ≈ 960 tasks / ≈ 3,168 pairs | **4,376** | 4,376 |
| exact label-preserving views each | **1,001** | **1,001** | **1** | **1** |
| distinct training inputs | ≈ 1.00 M | ≈ 3.17 M | **4,376** | 4,376 |
| batch | 768 | 768 | 256 | 256 |
| steps / epoch | 1.30 | 4.13 | **17.1** | 17.1 |
| epochs | 60,000 | 100,000 | **20–65** (early stop) | 3,000 |
| **optimiser steps** | **78,125** | **≈ 412,500** | **342–1,112** | **51,300** |
| slot-presentations (steps × batch) | 60.0 M | 316.8 M | 0.088–0.28 M | 13.1 M |
| **presentations per base example** | 60,000 | ≈ 330,000 | **20–65** | **3,000** |
| ÷ E[segments] (1–16) | 3,750–60,000 | 20,600–330,000 | 20–65 | 3,000 |
| **presentations per *distinct input*** | **3.7 – 60** | **6 – 100** | **20 – 65** | **3,000** |

**The crux, stated plainly.** On the axis that controls overfitting — how many times the optimiser
sees the *same* input vector — **TRM and our current recipe are already in the same regime (single
digits to ~100).** TRM looks like it trains 1,000× longer only because it holds 1,000× more *distinct*
inputs, manufactured by an exact symmetry group (sudoku digit/row/column permutation; ARC colour
permutation + dihedral + translation) that leaves the label provably unchanged.

Consequences:

- **"Copy TRM's epoch count" is not a transferable hyperparameter.** Transplanting 3,000 epochs onto
  4,376 molecules with one view each puts us at **3,000 presentations per distinct input — 30–800×
  past TRM's own ratio** (3,000 ÷ 3.7–100). That is not a long-training regime; it is a memorisation budget. [E]
- **The supportable epoch count scales with the view multiplier, not with model cleverness.** If we
  manufacture `V` exact label-preserving views per molecule, the TRM-equivalent schedule is roughly
  `20–65 × V` epochs. `V = 1` → we are already there. `V = 10` → 200–650. `V = 100` → 2k–6.5k. [E]
- **Compute is not the blocker; distinct data is.** TRM's Sudoku run is 78k steps of a 7M model — on
  our 2,500→256 MLP at batch 256 that is minutes on an L4. We could run TRM's exact schedule today.
  It would overfit, and more GPU would not fix it. [E]
- **Our feature vector is the problem, and it is a *choice*.** count-ECFP4 + RDKit2D + Avalon + ErG
  is computed from the canonical molecular graph, so it is **invariant** to SMILES randomisation,
  atom renumbering and conformer resampling. The single most powerful augmentation axis in molecular
  ML is *identically zero* for our representation. Any plan that wants TRM-scale training has to
  either change the input representation (so randomised SMILES / conformers actually change the
  tensor) or manufacture views in feature/label space. [E — this follows from how the descriptors are
  defined, not from a paper]

### 1.5 The one exact view multiplier this task *does* have

ST-RAE is **zero inside the credible interval**. So for a compound with interval `[lo, hi]`, *every*
value in `[lo, hi]` is an equally admissible target under the competition metric. Resampling the
regression target uniformly (or by the interval's posterior) from `[lo, hi]` **every epoch** is
therefore an *exactly metric-preserving* augmentation — the structural analogue of TRM's symmetry
group, in label space instead of input space.

This is my inference, **not a measured result** [E], and it is the single cheapest thing in this
document to test. It is also the only mechanism I can find that would *license* a longer schedule
rather than merely tolerate one. Caveat: it augments the *target*, not the input, so it reduces
label-noise overfitting but does nothing about input-space overfitting — the honest expected effect
is "a better-calibrated loss and a flatter early-stopping curve", not a 1,000× view multiplier.

---

## 2. Does anyone apply recursive / iterative-refinement latent reasoning to molecular or tabular
regression? — re-checked, and no

Two arXiv API queries run this session [V]:

- `all:"tiny recursive model"` → **`totalResults = 26`**. Full list with dates/titles below. Of the
  26: **one** tabular (Tab-TRM, insurance pricing, 2601.07675), **zero** chemistry, biology, drug
  discovery, ADMET, QSAR or protein.
- `(abs:"recursive reasoning" OR abs:"latent reasoning" OR abs:"adaptive depth" OR abs:"deep
  supervision") AND (abs:"molecular property" OR abs:QSAR OR abs:ADMET OR abs:bioactivity OR
  abs:IC50)` → **`totalResults = 0`**.

The complete TRM application surface, 2025-10 → 2026-10 [V]:

| arXiv | Date | Title | Domain |
|---|---|---|---|
| 2510.04871 | 2025-10-06 | Less is More: Recursive Reasoning with Tiny Networks | puzzles (origin) |
| 2511.02886 | 2025-11-04 | Test-time Adaptation of Tiny Recursive Models | ARC |
| 2511.08653 | 2025-11-11 | Accelerating Training Speed of TRMs with Curriculum Guided Adaptive Recursion | ARC, training cost |
| 2511.16886 | 2025-11-21 | Latent Reasoning in TRMs is Secretly a Policy Improvement Operator | theory |
| 2512.11847 | 2025-12-04 | TRMs on ARC-AGI-1: Inductive Biases, Identity Conditioning, Test-Time Compute | ARC audit |
| 2601.07675 | 2026-01-12 | **Tab-TRM: Tiny Recursive Model for Insurance Pricing on Tabular Data** | **tabular** |
| 2601.19085 | 2026-01-27 | Speed is Confidence | — |
| 2603.02193 | 2026-03-02 | Symbol-Equivariant Recurrent Reasoning Models | puzzles |
| 2603.05234 | 2026-03-05 | Recursive Inference Machines for Neural Reasoning | puzzles + UCI denoising |
| 2603.08082 | 2026-03-09 | Tiny Autoregressive Recursive Models | sequences |
| 2603.14593 | 2026-03-15 | Parameter-Efficient Quality Estimation via Frozen Recursive Models | MT QE |
| 2603.15641 | 2026-03-03 | Form Follows Function: Recursive Stem Model | — |
| 2603.19503 | 2026-03-19 | Vision Tiny Recursion Model (ViTRM) | images |
| 2603.22871 | 2026-03-24 | Dynamical Systems Theory Behind a Hierarchical Reasoning Model | theory |
| 2604.17224 | 2026-04-19 | LASER: Low-Rank Activation SVD for Efficient Recursion | efficiency |
| 2604.18839 | 2026-04-20 | One Step Forward and K Steps Back: Denoising Recursion Models | puzzles |
| 2605.19943 | 2026-05-19 | Probabilistic Tiny Recursive Model | puzzles |
| 2606.00895 | 2026-05-30 | TRMs for Solving the J2-Perturbed Lambert Problem | orbital mechanics |
| 2607.27251 | 2026-07-28 | Recursive transformers for semiconductor thermo-mechanical reliability | PDE/FEA surrogate |
| 2608.01534 | 2026-08-02 | Recursive Vision Language Models for General Symbolic Reasoning | VLM |
| 2608.08113 | 2026-08-08 | Think Deep, Speak Once: Relit | LLM |
| 2608.29376 | 2026-08-29 | Evaluating TRMs Across Training for Code Generation | code |
| 2609.05988 | 2026-09-05 | DART: Distributional Adversarial Recurrent Training | algorithms |
| 2609.26487 | 2026-09-22 | When Recursive Models Finish Computing | halting |
| 2609.34585 | 2026-09-28 | Compute Time Scaling with Recursive Models for Combinatorial Optimization | combinatorics |
| 2610.09876 | 2026-10-07 | Think Before You Paint: Recursive Latent Reasoning for Diffusion Models | diffusion |

**The two "nearest misses" are both not what they look like**, and neither is latent-depth-on-molecules:

- **RELISH — LLM Regression with a Latent Iterative State Head**, arXiv
  [2604.01206](https://arxiv.org/abs/2604.01206). This *is* iterative latent refinement for scalar
  regression, and it works: ~3.4–3.7M trainable params over frozen LLM backbones (0.01–0.04%
  overhead), beating autoregressive decoding, regression-aware inference and existing predictive
  heads across 6 datasets and 4 backbones [V, abstract]. But it is **text regression** — the latent
  state refines by *cross-attending over a token sequence*, which is why "the latent state requires
  at least one subsequent pass to re-attend to the token sequence" and convergence is at **L = 3**
  iterations [S]. A 2,500-dim fingerprint has no token sequence to re-attend to; the mechanism is
  absent, not merely untested.
- **VLSR — "Localize, Then Reason"**, arXiv [2608.13244](https://arxiv.org/abs/2608.13244). Latent
  structural reasoning for molecular properties and edits — but it is a *vision-language* model that
  localises substructures in a rendered molecule image and then reasons; the "iteration" is over
  structural localisation, not weight-tied latent depth, and I could not extract benchmark numbers
  from the PDF (6.6 MB, compressed streams) [V that it exists; numbers not obtained].

**Honest answer to question 3: nobody has done it.** Not on molecular property regression, not on
small high-dimensional tabular regression. The single tabular TRM result in existence is 9 features
over ~680k rows with a 0.33%-relative margin [C10].

---

## 3. Where TRM's and HRM's performance actually comes from — now with four independent audits

File 10 §5 has the ARC Prize ablation. Re-verified verbatim this session
[V, [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)]:

| Question | Measured |
|---|---|
| Verified HRM, ARC-AGI-1 / -2 | **32%** (9h 16m, $148.50, $1.48/task) / **2%** (12h 35m, $201) |
| Hierarchy? | "A regular transformer comes within **~5pp** of the HRM model without any hyperparameter optimization." Matched at a single outer loop; the gap "closes for higher numbers of outer loops." |
| Outer refinement loop? | "From no refinement (1 loop) to just 1 refinement, performance jumps by **+13pp**." 1 → 8 loops: "the Public Evaluation set performance **doubles**." |
| Train-time vs inference-time refinement? | "Training with more refinement improves the performance of predictions with a single refinement loop by **>15pp**." "Further refinement loops at inference don't have as big of an impact." |
| Transduction / puzzle IDs? | Model "can only be applied on puzzles with `puzzle_id`s it has seen at training time." Eval-tasks-only training: **31% pass@2** vs 41% original. |
| Augmentations? | "Only using **300** augmentations, instead of the 1,000 … achieves near max performance"; 30 augs "within 4% of max". |
| ACT halting? | "While using ACT does improve performance, the difference is **within a few percentage points** when compared to a fixed 16-loop run." |
| Overall framing | "fundamentally a **zero-pretraining test-time training** approach." |

The three post-hoc audits that have landed since, all pointing the same way:

- **2512.11847 (TRM on ARC-AGI-1 audit)** [V, abstract]: "Replacing the correct puzzle ID with a
  **blank or random token yields zero accuracy**." "The 1000-sample voting pipeline improves Pass@1
  by about **11 percentage points** over single-pass canonical inference." And, decisively for us:
  **"Most of the final accuracy is achieved at the first recursion step and performance saturates
  after few latent updates, indicating shallow effective recursion."**
- **2511.02886 (Test-time adaptation of TRM)** [V, abstract]: baseline TRM ~7.8% on ARC-AGI-2 public;
  pre-trained ~10% public; adapted **6.67% semi-private** via **12,500 gradient steps** of *full*
  fine-tuning on competition tasks — "not LoRA fine-tuning or fine-tuning of task embeddings alone."
  The original TRM run "required a level of compute far in excess of what is allowed during the
  competition."
- **2511.16886 (Policy improvement operator)** [V, abstract]: formalises latent recursive reasoning
  as a policy-improvement algorithm, and says the framing "explains when recursion contributes
  meaningfully versus when it produces **dead compute**"; applying RL/diffusion training schemes
  gives an **18× reduction in forward passes at maintained performance**. Read with 2512.11847: most
  of TRM's nominal recursion is dead compute.

**Causal decomposition of the ARC score, descending** (unchanged from file 10, now with a fourth
corroboration): per-task identity conditioning / test-time training (blank ID → 0%) ≫ augmentation +
voting (~11pp) ≳ the outer refinement loop as a *training-time* regulariser (+13pp, >15pp
train-vs-inference) > learned halting (a few pp, and *negative* in TRM's own ablation: 87.4 → 86.1)
> the hierarchy (0–5pp, 0 at one loop, and deleting it gained TRM 5pp).

**What of this can we have?** Identity conditioning: **no** — the test set is 750 blinded compounds
with `Molecule_Name` + `SMILES` and nothing to embed per-task. 1,000× exact augmentation: **no** (§1.4).
Augmentation voting: **only if** we create views. Outer refinement loop as training-time
regulariser: **yes, in principle**, and that is the one arm worth paying for — but note that the
mechanism as measured is *"the model re-predicts and is supervised again"*, which on a 4-output
regression head is an intra-model ensemble with a learned selector, not reasoning [C10].

---

## 4. What the small-data literature measures, ranked by effect size

Ordered by measured effect on tasks within an order of magnitude of ours (n ≈ 10² – 10⁴).
Where the split protocol matters I say so, because most of these are random splits and the blinded
750 will not be.

### Tier A — large, measured, at our data scale

**A1. Test-time retrieval fusion (kNN + GP over labelled neighbours).**
*Adapting Evidential Neural Networks to Test-Time Neighbor Fusion*, arXiv
[2607.11091](https://arxiv.org/abs/2607.11091) [V, HTML]. At inference, retrieve structurally similar
*training* molecules, treat their measured labels as noisy observations of the query's property, and
fuse with the network's prediction through a Gaussian process — **no retraining**.
Measured: RMSE reduced on **14 of 15** datasets, **median −19.4%**; best cases **CDK2 −42.4%**,
**BACE −38.9%**, **LD50 −38.2%**.
Dataset sizes, which are the point: **CDK2 2,060 · BACE 1,513 · ESOL 1,128 · Lipophilicity 4,200 ·
5-HT2A 5,388 · LD50 7,385 · hERG 8,362 · D2 8,433 · pKa 8,745 · HOPV 340** [V]. These are
*bioactivity regressions at exactly our n*. This is the single biggest measured effect in this sweep.
**Caveat, and it is a big one:** retrieval gains on bioactivity sets are heavily inflated by
analog-series structure under random splits. Discount hard for a diversity/scaffold split — see
`research/23-analog-series-local-models.md`.

**A2. Parameter-efficient implicit ensembling (TabM / BatchEnsemble head).**
*TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, arXiv
[2410.24210](https://arxiv.org/abs/2410.24210) [V, HTML v3]. An MLP with BatchEnsemble-style
per-submodel rank-1 adapters produces **k = 32** predictions per object from mostly-shared weights.
Benchmark: **46 datasets, 1.8K–723K training objects, 3–986 features, 28 regression / 18
classification, 9 with domain-aware splits**. Findings: TabM is "the best performance among tabular
DL models"; "MLP coupled with BatchEnsemble … right away outperforms popular attention-based models,
such as FT-Transformer"; "easily competes with GBDT"; retrieval/attention models are matched or
beaten "while being simpler and more efficient"; median ~2–3% relative over a plain MLP [V]. The
mechanism, Figure 5: **"the submodels look overfitted individually, while their collective prediction
exhibits substantially better generalization."** Performance rises in k to ≈32, and "too high values
of k can be detrimental."
**Why this is the headline recommendation:** it is *the same mechanism* the TRM audits credit (many
weak predictions, voted) with a tabular-validated implementation, at ~1× the training cost, and this
programme's only confirmed win to date is blending (+0.0283, 95% CI [+0.0160, +0.0427]) [C10].

**A3. Explicit deep ensembles, on bioactivity-scale data.**
Reported improvements of **18% / 29% / 16% / 24%** RMSE vs a single evidential model on CDK2 / hERG /
ESOL / Lipophilicity at 5× train+inference cost — **and a catastrophic failure on the smallest set,
HOPV (N ≈ 272–340): ensemble RMSE +69%**, because five members cannot each be reliably fit [S — this
is from the search index's summary of 2607.11091's Appendix C.2; I could not load that appendix, so
treat the per-dataset percentages as unverified]. Elsewhere: QM9 atomization energy "reasonably low
error achieved with an ensemble of M=5 and not much gained beyond that"; solubility RMSE 1.511 →
1.418 → 1.263 at ensemble sizes 4 → 8 → 10 [S]. The *direction* is solid and corroborated by A2; the
decision-relevant part is the shape: **big gains at n ≳ 1k, inversion at n ≈ 300.** Our per-isoform
subsets are 1,300–2,300, which is above the inversion but not comfortably.

**A4. Joint regularisation search ("regularisation cocktail").**
Kadra, Lindauer, Hutter, Grabocka, *Well-tuned Simple Nets Excel on Tabular Datasets*, NeurIPS 2021,
arXiv [2106.11189](https://arxiv.org/abs/2106.11189) [V]. Jointly search *which* of **13**
regularisers to apply and their hyperparameters, per dataset. On **40 datasets**, regularised plain
MLPs beat specialised NN architectures **and XGBoost**, and the cocktail "was the only method that
had statistically significant improvements compared to all other methods (**p ≤ 0.001 in all
cases**)." The 13 include mixup, dropout, weight decay, and more.
This is the highest-*confidence* lever on the list, and it is pure Optuna time.

### Tier B — real but smaller, or conditional

**B1. Feedback tokens / re-reading your own answer — NOT weight-tied looping.**
*Latent Chain-of-Thought Improves Structured-Data Transformers*, arXiv
[2605.11262](https://arxiv.org/abs/2605.11262) [V, abstract re-verified this session; table from C10].
36 datasets (27 tabular OpenML + 9 LTSF), TabICL-style architecture, depth- and parameter-matched
baselines:

| Domain | Variant | Wins | Average gain |
|---|---|---|---|
| Tabular | Deeper (2× params) | 14/27 | +0.66% ± 0.63% |
| **Tabular** | **CoT / feedback tokens** | **23/27** | **+3.25% ± 0.88%** |
| **Tabular** | **Looped (weight-tied)** | **14/27** | **−0.39% ± 0.80%** |
| Time series | CoT | 7/9 | +2.77% ± 1.05% (+12.63% headline) |

Compress the query-position hidden state into feedback tokens, append to the input, process again:
**+3.25%, 23/27**. Weight-tie a block and loop it: **−0.39% ± 0.80%, 14/27 — a null.** This is
exactly TRM decomposed, and it says the `y`-channel (re-reading the current answer) carries the
gain while the shared-weights recursion does not. Their limitations note that adaptive depth on
tabular data (PonderNet-style) was still undone as of 2026-05 [C10].

**B2. Input-space augmentation — only if the model can see it.**
Bjerrum, *SMILES Enumeration as Data Augmentation*, arXiv
[1703.07076](https://arxiv.org/abs/1703.07076) [S, consistent across several secondary sources]: on
DHFR with a ~130× augmented set, LSTM test **R² 0.56 → 0.66** and **RMS 0.62 → 0.55**; averaging
predictions over the enumerated SMILES at test time gives a further **R² 0.68 / RMS 0.52**. Later
work finds the effect **saturates at ~20× augmentation**, with "very limited" gains beyond [S].
Note the shape: roughly **two-thirds of the benefit is train-time augmentation and one-third is
test-time averaging** — the same train-dominates-inference split the ARC Prize found for refinement
loops. **But this is zero for us as currently configured**: ECFP/RDKit2D/Avalon/ErG are functions of
the canonical graph and do not change under SMILES randomisation. Buying this effect means adding a
sequence-reading arm (ChemBERTa / SMILES transformer), which is a representation decision, not an
augmentation decision.

**B3. C-Mixup for regression.** Yao et al., NeurIPS 2022, arXiv
[2210.05775](https://arxiv.org/abs/2210.05775) [V, abstract]: reweight mixup *pair sampling* by label
similarity. **+6.56% in-distribution generalisation, +4.76% task generalisation, +5.82% OOD
robustness** over the best prior approach, on Airfoil/NO2/Exchange-Rate/Electricity/Echo. Operates in
feature space, so it applies to fingerprints directly. Interacts with A4 (mixup is one of the 13).

**B4. Transductive use of the blinded inputs.** ARC Prize's own framing of HRM is "fundamentally a
zero-pretraining **test-time training** approach", and 2511.02886 gets its competition score from
**12,500 gradient steps of full fine-tuning on the competition tasks** [V]. We cannot copy that —
their test-time training uses the eval tasks' *demonstration labels*; our 750 are blinded. The
*input-only* version is available and legitimate: fit standardisation/whitening on train ∪ test
inputs, masked-feature self-supervision on the pooled inputs, or uncertainty-gated pseudo-labelling.
Measured effect in our domain: pseudo-labelling helps when the pseudo-labels are quality-filtered and
hurts when they are not (InstructMol, arXiv [2304.03906](https://arxiv.org/abs/2304.03906);
uncertainty-aware pseudo-labelling, Huang et al. PMLR v180) [S]. No clean effect size at our scale.

**B5. TabPFN v2 — regime-dependent, worth one cheap arm.** *Accurate predictions on small data with a
tabular foundation model*, Nature 2025 [V]: "outperforms all previous methods on datasets with up to
10,000 samples by a wide margin". Corroborated at the very small end — *Modern Neural Networks for
Small Tabular Datasets*, arXiv [2508.09888](https://arxiv.org/abs/2508.09888) [V], 31 field-scale
soil datasets of **30–460 samples**, concludes "TabPFN delivers the strongest overall performance …
the new default choice". **But** TabM's 46-dataset benchmark (1.8K–723K) found "the performance of
TabPFN on our benchmark was not competitive" [V]. Our n = 4,376 sits in the crossover, and TabPFN v2
caps out around a few hundred features vs our ~2,500 — so this is a *feature-selection + TabPFN* arm,
not a drop-in.

### Tier C — measured nulls. Do not spend the budget here.

| Lever | Measured | Source |
|---|---|---|
| **Weight-tied recursion on tabular** | **−0.39% ± 0.80%**, 14/27 wins | 2605.11262 [C10] |
| **TRM-style recursive encoder, this task** | gap **0.0017** (H2, "confirmed null") | this programme [C10] |
| **SSL graph pretraining, with rich features + scaffold splits** | **−0.578% to +0.128%**, inside baseline sd of **0% ± 1.626%** | Sun et al., *Does GNN Pretraining Help Molecular Representation?*, NeurIPS 2022, arXiv [2207.06010](https://arxiv.org/abs/2207.06010) [V] |
| **HRM's 1-step gradient approximation** | **−30.9pp** (87.4 → 56.5) in TRM's own ablation; and its fixed-point premise does not exist for a 4-output regression head | 2510.04871 [C10] |
| **Full ACT / Q-learning halting** | **−1.3pp** (87.4 → 86.1) at **2×** forward passes; ARC Prize: "within a few percentage points" of fixed 16 loops | 2510.04871, arcprize [V] |
| **The hierarchy itself** | **0–5pp**, 0pp at one outer loop; TRM deleted it and gained **+5.0pp** (87.4 vs 82.4) | arcprize, 2510.04871 [V] |
| **Deeper (2× params) on tabular** | +0.66% ± 0.63%, 14/27 — indistinguishable from zero | 2605.11262 [C10] |
| **Auxiliary heads, this task** | 0.4346 vs 0.4359 (twice) | this programme [C10] |

And the detectability argument, which should be in the register before any TRM arm is run:
Tab-TRM's entire margin over the previous best compact model is **0.33% relative ≈ 0.0014 ST-RAE**
here, against a paired across-seed sd of **±0.0098** and a single-fold bootstrap sd of **≈0.03**.
**If the true effect is the size of the only published positive result, this experiment cannot see
it** [C10].

---

## 5. The arm list, cheapest-first, with what each would prove

| # | Arm | Cost | Expected | Decides |
|---|---|---|---|---|
| 1 | **TabM head** (k=32 BatchEnsemble adapters) on the existing 2,500-dim encoder, matched optimiser steps vs the current MLP, 4 paired seeds | hours | **positive, moderate**; A2's mechanism + this programme's blending result point the same way | whether the blending win (+0.0283) is obtainable intra-model at 1× cost |
| 2 | **Regularisation-cocktail Optuna sweep** (13 regularisers, joint on/off + hyperparameters) | GPU-days of search, trivially parallel | **positive**; highest-confidence lever on the list (p ≤ 0.001 over 40 datasets) | the ceiling of the current representation |
| 3 | **Interval-resampled targets** — redraw y ~ U[lo, hi] per epoch, with a longer schedule and the same early-stopping rule | ~free | unknown; **my inference, unmeasured** | whether this task has *any* exact view multiplier, which is the precondition for every long-schedule idea |
| 4 | **Test-time retrieval fusion** (Tanimoto kNN over the training set + GP fusion, no retraining), evaluated under a *diversity* split, not a random one | hours | **large under random splits (median −19.4%), much smaller under diversity splits** | the biggest measured effect in the literature at our n, and how much of it is analog-series leakage |
| 5 | **Feedback-token arm**: append the current prediction (and its interval-hinge residual) back as extra input features, re-run, supervise both passes. **Not** weight-tied | hours | **small positive** (+3.25%, 23/27 on tabular) | TRM's `y`-channel, isolated from the component that is a measured null |
| 6 | **C-Mixup** on the fingerprint features, label-similarity-weighted pairing | hours | small positive | whether feature-space augmentation substitutes for the absent input-space augmentation |
| 7 | **Feature-selected TabPFN v2** (≤ 500 features by mutual information / PCA) as an ensemble member | hours | regime-dependent | whether n=4,376 is on the TabPFN side of the crossover |
| 8 | **SMILES-reading arm** (ChemBERTa or a small SMILES transformer) with 20× enumeration + test-time averaging, as an *ensemble member*, not a replacement | days | uncertain; it is the only way to buy a real view multiplier | whether representation diversity beats architecture search — which is the one thing that has ever moved this number [C10] |
| — | **TRM/HRM as an architecture** | days | **null at 70–80%** [C10], and the effect size of the only positive precedent is 7× below our MDE | nothing we do not already know from three independent nulls |

Protocol notes that will otherwise invalidate arms 1, 3 and 5:

- **Match on optimiser steps, never on epochs.** §1.3: a deep-supervised / multi-pass arm consumes
  `E[segments]×` more optimiser steps per distinct example. File 10 §9.5 flagged this as H12's
  confound in another guise.
- **An "epoch" is a meaningless unit across arms with different view multipliers.** Report
  presentations-per-distinct-input alongside it (§1.4).
- **The halt/selector head's target must be stop-gradded** and computed under `no_grad`, as in TRM's
  loss head (`seq_is_correct` inside a `torch.no_grad()` block) [C10].
- **Register arms 1 and 5 as *selection/ensembling* hypotheses, not reasoning hypotheses.** If they
  win, the conclusion is "a learned per-step selector over an intra-model ensemble helps", which
  generalises to the 21-endpoint ADMET panel and has nothing to do with recursion [C10].

---

## 6. Sources

**TRM / HRM primary**
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, arXiv [2510.04871](https://arxiv.org/abs/2510.04871) · [HTML](https://arxiv.org/html/2510.04871v1)
- Repo [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) — `pretrain.py`, `puzzle_dataset.py`, `dataset/build_arc_dataset.py`, `models/recursive_reasoning/trm.py`, `config/cfg_pretrain.yaml`
- Wang et al., *Hierarchical Reasoning Model*, arXiv [2506.21734](https://arxiv.org/abs/2506.21734)

**Audits of where the performance comes from**
- ARC Prize, *HRM analysis*, [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)
- *TRMs on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute*, arXiv [2512.11847](https://arxiv.org/abs/2512.11847)
- *Test-time Adaptation of Tiny Recursive Models*, arXiv [2511.02886](https://arxiv.org/abs/2511.02886)
- *Latent Reasoning in TRMs is Secretly a Policy Improvement Operator*, arXiv [2511.16886](https://arxiv.org/abs/2511.16886)
- *Accelerating Training Speed of TRMs with Curriculum Guided Adaptive Recursion*, arXiv [2511.08653](https://arxiv.org/abs/2511.08653)
- *Dynamical Systems Theory Behind a Hierarchical Reasoning Model*, arXiv [2603.22871](https://arxiv.org/abs/2603.22871)

**Recursion off grid puzzles**
- *Tab-TRM: Tiny Recursive Model for Insurance Pricing on Tabular Data*, arXiv [2601.07675](https://arxiv.org/abs/2601.07675)
- *Latent Chain-of-Thought Improves Structured-Data Transformers*, arXiv [2605.11262](https://arxiv.org/abs/2605.11262)
- *Recursive Inference Machines for Neural Reasoning*, arXiv [2603.05234](https://arxiv.org/abs/2603.05234)
- *RELISH: LLM Regression with a Latent Iterative State Head*, arXiv [2604.01206](https://arxiv.org/abs/2604.01206)
- *Localize, Then Reason: Visual Latent Structural Reasoning for Molecular Properties and Edits*, arXiv [2608.13244](https://arxiv.org/abs/2608.13244)

**Small-data levers with numbers**
- *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*, arXiv [2410.24210](https://arxiv.org/abs/2410.24210)
- *Adapting Evidential Neural Networks to Test-Time Neighbor Fusion Improves Molecular Property Prediction*, arXiv [2607.11091](https://arxiv.org/abs/2607.11091)
- Kadra et al., *Well-tuned Simple Nets Excel on Tabular Datasets*, arXiv [2106.11189](https://arxiv.org/abs/2106.11189)
- Yao et al., *C-Mixup: Improving Generalization in Regression*, arXiv [2210.05775](https://arxiv.org/abs/2210.05775)
- Sun et al., *Does GNN Pretraining Help Molecular Representation?*, arXiv [2207.06010](https://arxiv.org/abs/2207.06010)
- Bjerrum, *SMILES Enumeration as Data Augmentation*, arXiv [1703.07076](https://arxiv.org/abs/1703.07076)
- Hollmann et al., *Accurate predictions on small data with a tabular foundation model*, Nature 2025, [s41586-024-08328-6](https://www.nature.com/articles/s41586-024-08328-6)
- *Modern Neural Networks for Small Tabular Datasets*, arXiv [2508.09888](https://arxiv.org/abs/2508.09888)
- *Instructor-inspired Machine Learning for Robust Molecular Property Prediction* (InstructMol), arXiv [2304.03906](https://arxiv.org/abs/2304.03906)

**arXiv API queries run this session**
- `all:"tiny recursive model"` → `totalResults = 26`
- `(abs:"recursive reasoning" OR abs:"latent reasoning" OR abs:"adaptive depth" OR abs:"deep supervision") AND (abs:"molecular property" OR abs:QSAR OR abs:ADMET OR abs:bioactivity OR abs:IC50)` → `totalResults = 0`

**Companion files in this programme**
- `research/10-recursive-and-adaptive-depth.md` — mechanisms, algorithm boxes, ACT/PonderNet, pitfalls
- `research/23-analog-series-local-models.md` — why retrieval gains need a diversity split
- `research/03-training-techniques.md` §9, §15–17 — recursion, augmentation and test-time compute as catalogued earlier
