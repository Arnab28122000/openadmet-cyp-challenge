# 28 — Does compute help? Scaling, grokking, double descent, and the 30,000-epoch proposal

**Question put to this review.** *"Training much longer and much bigger solves problems — like it
did for LLMs and Stable Diffusion. We should train for 30,000 epochs on an A100 for 12–24 hours or
a week, and the score will improve."*

**Setting.** Supervised molecular property regression. ~4,376 molecules, ~1,300–2,335 labelled
values per task, 4 tasks (CYP1A2/2C9/2D6/3A4 inhibition pIC50), ~2,000-dimensional
descriptor+fingerprint features, ~6,525 primary labels in total. Incumbent models: LightGBM and a
32-member vmapped multitask MLP ensemble (d1024 × 3 layers) that converges in minutes.

**Verdict in one line.** The literature says no, and — decisively — **this programme has already
run the experiment twice and measured no**. But the honest answer is not a flat no on compute:
there is one real, replicated, *local* effect in which more compute does pay, and it is a
**depth × schedule interaction** worth ~0.023 on validation (~0.004 on test after the measured 17%
transfer rate), at **3,000** epochs with a 6-layer net — not 30,000 epochs with a 3-layer one. And
the constructive finding is sharper than "spend it elsewhere": **there is no good way to spend a week
of an A100 on this problem at all.** Every lever that works saturates in minutes to a few hours, and
the measured compute/error frontier for a dataset our size fits in **~20 CPU-minutes** (§7.0). The
A100's real use here is to run the sweep **wide** — many seeds × folds × configs in parallel — not
long. The ranked list is in §7.

**The sentence that does most of the work.** Every headline "we trained for 30,000 epochs and won"
result turns out, on inspection, to be counting something other than repeated passes over fixed
data — LLMs and Stable Diffusion did ~1 epoch over 10⁹ examples; the diffusion models that profit
from repetition do so through "implicit data augmentation"; TRM's 100,000 "epochs" are 1.3
optimiser steps each over 1,000 label-preserving views per example. **We have one view per molecule
and 6,525 labels.** See §1.4.

---

## 0. The arithmetic first: what 30,000 epochs actually costs and actually does here

The proposal's compute estimate is roughly right, and that is worth conceding up front. Measured
on this programme's own hardware (`PROGRESS.md`, 2026-10-05):

- 16 members × 80 epochs in **24 s** → **0.28 s/epoch** (d1024 L3, peak 18 GB of 42.4)
- 64 members → **1.5 s/epoch** (20.2 GB of 42.4)

So ~0.023 s/epoch/member, i.e. **~0.7 s/epoch** for the incumbent 32-member L3 ensemble and
~1.2–1.5 s/epoch at L6/16. 30,000 epochs is therefore **6–12 h per seed**, and a 5-seed result is
**40–60 A100-hours**. The "12–24 hours" figure is realistic. The proposal is feasible. The
question is only whether it buys anything, and that has been measured.

**It has already been run, at exactly the point where it stops mattering.** Two experiments:

| experiment | config | result |
|---|---|---|
| `v3sched` vs incumbent `v2` | L3 / 32 members / **3,000 ep, patience 400** vs 300 ep / patience 40 | **delta exactly −0.0000**, and *per endpoint identical to four decimal places* |
| `h12-longbase` / `h12-longdeep` | L3 and L6 / **3,000 ep, patience 400**, 5 seeds | best epochs **310, 365, 15, 20, 340** (L3) and **205, 235, 215, 190, 235** (L6); early stopping halts at **591–766 epochs. None ever reaches 3,000.** |

The validation trajectories are the whole argument (`PROGRESS.md`, 2026-10-06, "Therefore: more
epochs is the wrong lever, and the curves say so directly"; lower ST-RAE is better):

```
h12-longdeep s0   ep20 0.6859  ep100 0.7091  ep200 0.6855  ep400 0.7644  ep600 0.7535  ep700 0.7358   best 0.6823 @205
h12-longdeep s1   ep20 0.7045  ep100 0.7077  ep200 0.6900  ep400 0.7677  ep600 0.7755  ep700 0.7510   best 0.6656 @235
h12-longbase s0   ep20 0.7224  ep100 0.7619  ep200 0.7722  ep400 0.7767  ep600 0.7952  ep700 0.7975   best 0.7073 @310
```

Validation bottoms out at **epoch 190–365** and then **rises monotonically**. By epoch 700 every
run is **10–17% worse than its own best**. So the 30,000-epoch proposal collapses into a dilemma:

- **With early stopping**, it is a literal no-op — the schedule never gets there, and the
  measured delta of extending 300 → 3,000 epochs at L3/32 is **−0.0000**. 30,000 changes nothing
  that 3,000 did not already change.
- **Without early stopping**, the measured curves say the answer is 10–17% worse by epoch 700 and
  monotonically worse beyond.

And there is a multiplier on top: this programme measured that **validation gains transfer to the
leaderboard at ~17%**, while the generalisation gap *widens*. A lever that only moves validation
returns about a sixth of itself on test. Longer training is exactly such a lever.

**The one remaining fallback defence, closed.** Someone might argue that even if long training does
not raise the mean, it might *stabilise* the result — a tighter seed band would still be worth
something, given that our seed spread (0.013) exceeds every effect we have measured. Picard's
10,000-seed study answers this directly (<https://arxiv.org/abs/2109.08203>):

> *"the accuracy does not increase past epoch 25 … However, there is **no reduction in variation
> past convergence**, which indicates that **training for longer is not likely to reduce the
> variation**."*

So longer training buys neither the mean nor the variance. For calibration on how large seed noise
is in absolute terms, the same study found a **1.82 pp range** in CIFAR-10 accuracy across 10,000
seeds at fixed everything, and notes that *"a 0.5% difference in accuracy could be entirely
explained by just a seed difference"* — a spread that *"is widely considered as significant in the
community — to the point of being an argument for publication at very selective venues — whereas we
know here that this is just the effect of finding a lucky/cursed seed."*

---

## 1. Scaling laws: where they apply, and where they do not

### 1.1 The analogy to LLMs and Stable Diffusion runs the wrong way

This is the most important factual correction, because the premise of the claim is a
misremembering of how those models were trained.

- **Chinchilla** (Hoffmann et al., NeurIPS 2022) trained >400 models from 70 M to 16 B parameters
  on 5 B to 400 B tokens, and found compute-optimal training needs **~20 tokens per parameter**,
  with parameters and tokens scaled **1:1**. Chinchilla itself: 70 B parameters on **1.4 trillion
  tokens** — drawn from a ~2.35 T-token corpus, i.e. **less than one epoch**.
  <https://arxiv.org/abs/2203.15556> ·
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/c1e2faff6f588870935f114ebe04a3e5-Paper-Conference.pdf>
- **Kaplan et al. 2020** likewise fit power laws in the regime where fresh data is effectively
  unlimited, and explicitly recommend *stopping well short of convergence* on a fixed dataset.
  <https://arxiv.org/abs/2001.08361>
- **Stable Diffusion** was trained on subsets of **LAION-5B** (5.85 B image–text pairs, 2.32 B
  English) for **150,000 A100-hours**; the published training recipe is ~550 k + 850 k iterations
  at large batch over **790 M–2.3 B unique images** — again **on the order of one to a few
  epochs**. <https://arxiv.org/abs/2210.08402> ·
  <https://proceedings.neurips.cc/paper_files/paper/2022/file/a1859debfb3b59d094f3504d5ebb6c25-Paper-Datasets_and_Benchmarks.pdf> ·
  <https://www.databricks.com/blog/stable-diffusion-2>

**Neither LLMs nor Stable Diffusion were trained for many epochs.** They were trained for roughly
*one* epoch over a corpus 10⁵–10⁹ times larger than ours. The thing that made them work was
**unique data volume**, and compute was the instrument for consuming it. The proposal keeps the
compute and discards the mechanism. 6,525 labels × 30,000 epochs is not the Chinchilla regime at
all; it is 1.96 × 10⁸ *sample presentations* of 6,525 distinct facts.

### 1.2 The small, fixed-data regime has its own law, and it has a number

**Muennighoff et al., "Scaling Data-Constrained Language Models"** (NeurIPS 2023 / JMLR 2025) is
the direct answer. 400 training runs, up to 900 B tokens and 9 B parameters, varying data
repetition explicitly. <https://arxiv.org/abs/2305.16264> ·
<https://jmlr.org/papers/volume26/24-1000/24-1000.pdf> ·
<https://proceedings.neurips.cc/paper_files/paper/2023/file/9d89448b63ce1e2e8dc7af72c984c196-Paper-Conference.pdf>

From the abstract, verbatim:

> "We find that with constrained data for a fixed compute budget, **training with up to 4 epochs of
> repeated data yields negligible changes to loss** compared to having unique data. However, **with
> more repetition, the value of adding compute eventually decays to zero**."

The fitted constants (Appendix A): **R_N\* = 5.309743 and R_D\* = 15.387756**. R_D\* is *"the point
where repeated tokens have lost 1 − 1/e of their value"*, and their own summary of it:

> "**Meaningful gains from repeating data can be made up to around 16 epochs (R_D\*) beyond which
> returns diminish extremely fast.**"

And — critically for this question — repetition does not merely plateau:

> "**Adding parameters and epochs causes the loss to decrease and eventually increase again,
> suggesting that too much compute can hurt performance.**"

(The authors note their closed-form isoLoss contours "do not model the possibility that excess
epochs or parameters could hurt performance" — the degradation is an empirical observation outside
the fitted law.)

### 1.2b The scaling law that actually governs a fixed small dataset — and it has a plateau

Kaplan and Hoffmann answer "how big a model for how much data". The law for *our* question — "my
dataset is fixed at n; what does more model buy?" — is **Rosenfeld, Rosenfeld, Belinkov & Shavit,
"A Constructive Prediction of the Generalization Error Across Scales", ICLR 2020** —
<https://arxiv.org/abs/1909.12673>. Fitted across 6 vision + language benchmarks (WRN, DenseNet,
VGG, AWD-LSTM, Transformer-XL), SGD and Adam, as a joint function of model size *m* and data size
*n*:

> **Observation 1.1, verbatim:** *"For a given dataset size, scaling up the model results in an
> initial decrease in test error, which then **saturates to a level determined by the dataset
> size**."*

with the footnote that **at some point error *increase* ensues** — and they verify this is genuine
overfitting, since training error does not rise with it. Fit quality: relative errors *"under 2%
when fitting across all scales"* and *"under 5% when extrapolating from a slimmed-down model (1/16
of the parameters) on a fraction of the training data (1/8 of the examples)."*

This is the shape this programme measured independently: capacity moved the score by +0.0019
(p = 0.723) and width by exactly +0.0000. **With n fixed at ~4,376 and a model that already
converges in minutes, we are on the plateau `c_m(n)`, where the marginal return on model size — and
therefore on A100-hours spent training a bigger net — is ≈ 0 and turns negative beyond it.**

**And the paper hands us a cheap decisive test.** Fit
`ε(m, n) = a·n^−α + b·m^−β + c` on a handful of deliberately *small* (m, n) points of our own data,
then read off which term dominates. On a multitask MLP that trains in minutes, that is about an hour
of work, and it answers "are we capacity-limited or data-limited?" with a number instead of an
argument. **This is the single highest-value experiment this review can recommend**, because it
settles the disagreement on our own data rather than by citation — and the extrapolation accuracy
they report (<5% from 1/16 the parameters and 1/8 the data) says the cheap version is trustworthy.

### 1.3 Steelman: the 16-epoch number is architecture-dependent, and one paper beats it

For even-handedness, the strongest published challenge to R_D\* ≈ 16 is **"Diffusion Beats
Autoregressive in Data-Constrained Settings"** (2025), which finds diffusion language models keep
extracting value from repeated data for **far more epochs** than autoregressive ones, and
"significantly outperform AR models **when compute is abundant but data is scarce**" — the exact
regime the proposal invokes. They even derive "a closed-form expression for the critical compute
threshold at which diffusion begins to outperform AR".
<https://arxiv.org/abs/2507.15857> · <https://arxiv.org/pdf/2507.15857>
*(I could not verify the specific epoch counts from the abstract alone; cited for the mechanism and
direction, not for a number.)*

But read their **explanation**, because it relocates the credit:

> diffusion models excel because "their **randomized masking objective** implicitly trains over a
> rich distribution of token orderings, **acting as an implicit data augmentation** that AR's fixed
> left-to-right factorization lacks."

So the thing that let a model profit from many epochs of repeated data was **data augmentation
built into the objective** — not the epochs. That is an argument for adding augmentation
(SMILES randomisation, conformer/feature dropout, input noise) and *then* lengthening the
schedule, which is a different and much better-supported proposal than lengthening the schedule
alone. See also "Data-Constrained Language Model Pretraining: Improved Regularization and Scaling
Laws" <https://arxiv.org/abs/2606.06888>, whose title makes the same point: what unlocks repeated
data is **regularisation**.

### 1.4 Three independent routes, one answer: every "huge epoch count" success story is an augmentation story

This is the convergence that settles the question, and it is worth stating on its own because three
unrelated lines of evidence land on it.

1. **Chinchilla / Stable Diffusion** (§1.1): ~1 epoch over 10⁹-scale corpora. The lever was *unique
   data*.
2. **Diffusion vs AR in data-constrained settings** (§1.3): the architecture that profits from many
   epochs of repeated data does so via "**implicit data augmentation**" baked into its objective.
3. **TRM / HRM**, the headline "3,000 / 60,000 / 100,000 epochs" results — analysed in this
   programme's companion file `research/25-trm-small-data.md`, whose finding is blunt:
   > "**TRM's '3,000 / 60,000 / 100,000 epochs' is not a long-training result.** An 'epoch' in that
   > codebase is **1.3 optimiser steps on Sudoku and ~4.1 on ARC**, and the thing that is large is
   > not the number of passes per distinct input (**3.7–100**, the same order as our current
   > 20–65) but the **number of distinct inputs: 1,000 exact label-preserving views per base
   > example**. We have **one** view per molecule. **The entire TRM budget story is an augmentation
   > story.**"

**So the headline epoch numbers that motivate this proposal are, on inspection, all counting
something other than repeated passes over fixed data.** Where someone appears to have trained for
30,000 epochs and won, they had either 10⁹ unique examples, an objective that manufactures views,
or 1,000 label-preserving views per example. We have one view per molecule and 6,525 labels. The
correct translation of "they trained for 30,000 epochs" into our setting is **"get more views or
more labels"** — §7 rows 2, 4 and 7 — not "raise `--epochs`".

**Numbers for this setting.** The literature's "returns essentially gone" point is **~16 epochs**,
with ~4 epochs free. The proposal is **30,000 epochs — roughly 1,900× R_D\***. Our own measured
optimum is **epoch 15–25 at L3/32 members** and **190–365 under long schedules at lower member
counts**, which is 1–20× R_D\* and sits comfortably inside the literature's band. Nothing in the
data-constrained scaling literature supports an optimum three or four orders of magnitude beyond
it, and the one paper that measured the far tail reports the loss *rising*.

---

## 2. Grokking — the strongest case *for* the claim, examined rigorously

This is the honest best argument for the proposal, so it gets the most scrutiny.

### 2.1 What Power et al. actually showed

**Power, Burda, Edwards, Babuschkin, Misra, "Grokking: Generalization Beyond Overfitting on Small
Algorithmic Datasets"** (2022). <https://arxiv.org/abs/2201.02177> ·
<https://arxiv.org/pdf/2201.02177> · code <https://github.com/openai/grok>

Validation accuracy jumps from chance to ~100% *long after* training accuracy saturates. But read
the title: **small algorithmic datasets**. The tasks are modular arithmetic and binary operations
on finite groups — *S*₅ permutation composition, modular addition/multiplication/division. The
authors' own framing is that these datasets "provide a fertile ground for studying generalization
of overparametrized neural networks beyond memorization".

Three properties of that setting do not hold for molecular regression:

1. **The target is an exact, noiseless, closed-form function.** There is a finite circuit that gets
   100%. pIC50 is a noisy physical measurement with assay error; there is no exact rule to find.
2. **The task is classification into a small finite set**, so a representation is either right
   (100%) or wrong (chance). There is no analogue for a continuous regression target.
3. **The input space is the complete group table.** Train and test are drawn from the *same finite
   enumerable set*. Our test set is 75 analog series purchased after the fact, with 0.665 internal
   Tanimoto similarity against our validation set's 0.354 — a different population, not a held-out
   slice of the same one.

### 2.2 Has grokking ever been shown on a molecular task? Yes — and the answer still refutes the proposal

**Liu, Michaud, Tegmark, "Omnigrok: Grokking Beyond Algorithmic Data"** (ICLR 2023).
<https://arxiv.org/abs/2210.01117> · <https://ar5iv.labs.arxiv.org/html/2210.01117> ·
<https://github.com/KindXiaoming/Omnigrok>

This is the paper that matters most, and it does include molecules. Abstract, verbatim:

> "Grokking … has remained elusive. We aim to understand grokking by analyzing the loss landscapes
> of neural networks, identifying the mismatch between training and test losses as the cause for
> grokking. We refer to this as the **'LU mechanism'** … Guided by the intuitive picture, we are
> able to **induce** grokking on tasks involving images, language and molecules. In the reverse
> direction, we are able to **eliminate** grokking for algorithmic datasets. We attribute the
> dramatic nature of grokking for algorithmic datasets to **representation learning**."

The QM9 experiment: a GCNN predicting isotropic polarizability, 2 conv layers + linear, Adam at
1e-3, MSE, 50/50 train/test split. The decisive sentence:

> "with **200 training samples**, grokking is observed for **large initialization (α = 3)**, while
> **no grokking is observed for standard initializations (α = 1)**."

Read that carefully. On the one molecular dataset where grokking has been demonstrated:

- it required **200 training samples** (we have ~4,376 molecules — 20× more);
- it required the weight norm to be **inflated 3× above standard initialization**;
- **at standard initialization it does not happen at all.**

The same pattern holds for their MNIST (60 k → **1 k** samples, α > 1) and IMDb runs. Grokking on
real data in this paper is **manufactured by a deliberately pathological initialization**, and the
paper's framing is diagnostic — it is explaining a landscape mismatch, not proposing a recipe. The
word is *induce*.

The mechanism also tells you why it cannot be the lever here. Omnigrok attributes the *dramatic*
grokking to representation learning, where "the quality of representation determines whether test
accuracy is random guess (bad representation) or 100%". We are feeding the model **fixed
precomputed descriptors and fingerprints**. There is no representation to discover in the inputs;
the featurisation is frozen before the first gradient step.

### 2.3 Grokking's conditions are narrow, and the delay is removable

- **It needs a narrow weight-decay window.** Grokking occurs only within a narrow range of
  regularisation strength: too little and the model never escapes memorisation, too much and train
  and test fall together or nothing is learned. See **Liu et al., "Towards Understanding Grokking:
  An Effective Theory of Representation Learning"** (NeurIPS 2022) and its "Goldilocks zone".
  <https://arxiv.org/abs/2205.10343> ·
  <https://papers.neurips.cc/paper_files/paper/2022/file/dfc310e81992d2e4cedc09ac47eff13e-Paper-Conference.pdf>
- **It needs a small training fraction.** With larger training fractions, models generalise
  *without* the delay — the memorisation phase that grokking escapes from never forms.
  <https://arxiv.org/abs/2609.10657> · <https://arxiv.org/html/2603.25009v1>
- **The delay is an optimisation artefact, not a required payment.** **"Grokfast: Accelerated
  Grokking by Amplifying Slow Gradients"** accelerates grokking by **×50** by low-pass filtering
  the gradient — and tests on modular multiplication, MNIST MLPs, IMDb LSTMs **and QM9 molecular
  property prediction with Graph ConvNets**. Their own statement:
  > "Although the generalization accuracy, the training accuracy, the training loss, and the
  > validation loss at the final state are **similar in both the baseline and Grokfast**."

  So on a molecular task, the long-training delay buys **the same endpoint, ×50 faster**. Those
  epochs are paying for a slow optimiser, not for generalisation.
  <https://arxiv.org/abs/2405.20233> · <https://arxiv.org/html/2405.20233v1>
- **Train past grokking and it collapses.** "Late-Stage Generalization Collapse in Grokking:
  Detecting anti-grokking" identifies a **third phase** in which "test accuracy collapses back to
  chance while training accuracy remains perfect", at ~10⁷ steps — roughly **10× longer than the
  grokking phase itself** (~10⁶ steps), with the ℓ₂ norm growing without bound absent weight decay.
  Shown on an MLP on MNIST and a transformer on modular addition.
  <https://arxiv.org/html/2602.02859v1>

**Even-handed summary of §2.** Grokking is real, it has been shown on QM9, and that is the single
strongest piece of evidence for the proposal. But every demonstration outside algorithmic data
required an intervention we are not making (3× initialization inflation), a data size we do not
have (200 samples), and a task structure we do not have (classification with an exact rule); the
delay it describes is removable in ~1/50 the compute; and training 10× past it destroys
generalization outright. **Grokking has never been demonstrated on a real-world regression,
tabular, or molecular-potency task under standard initialization — and "real-world grokking signals
are usually less dramatic than for algorithmic datasets" is the field's own summary.**
<https://arxiv.org/abs/2405.12755>

---

## 3. Double descent — epoch-wise and model-wise

**Nakkiran, Kaplun, Bansal, Yang, Barak, Sutskever, "Deep Double Descent: Where Bigger Models and
More Data Hurt"** (ICLR 2020 / J. Stat. Mech. 2021). <https://arxiv.org/abs/1912.02292> ·
<https://ar5iv.labs.arxiv.org/html/1912.02292>

Abstract, verbatim:

> "We show that a variety of modern deep learning tasks exhibit a 'double-descent' phenomenon
> where, as we increase model size, performance first gets worse and then gets better. Moreover, we
> show that double descent occurs not just as a function of model size, but also as a function of
> the number of training epochs. We unify the above phenomena by defining a new complexity measure
> we call the **effective model complexity** … our notion of model complexity allows us to identify
> certain regimes where **increasing (even quadrupling) the number of train samples actually hurts
> test performance**."

Epoch-wise double descent is the part that bears on the proposal, and yes, it exists:

> "The test error of a large model first decreases (at the beginning of training), then increases
> (around the critical regime), then decreases once more (at the end of training) — that is,
> **training longer can correct overfitting**."

**The sharp question — does the second descent end up *below* the first (early-stopping) minimum?**
This is the crux for the proposal, and the paper is genuinely ambiguous on it. In their
label-noise CIFAR-10 settings the final error of a large model does end below the first local
minimum, which is what "training longer can correct overfitting" means. But their own §8 says the
phenomena largely vanish under optimal early stopping, and I could not extract a clean, quotable
statement that the converged error beats the *optimally* early-stopped error. **Flagging this as
unresolved from the source rather than asserting either reading.** What is not ambiguous is that
nobody has shown this on small tabular regression.

**Three caveats that the proposal needs to clear, and does not.**

1. **The paper's own conclusion disclaims the practical reading.** From §8:
   > "**Many of the phenomena that we highlight often do not occur with optimal early-stopping.**"

   An optimally early-stopped model does not display double descent, and the best early-stopped
   error is competitive with the post-second-descent error. If you are already doing proper early
   stopping — we are, with patience 400 — double descent is largely not a resource you can mine.
2. **It is driven by label noise and interpolating capacity.** "The test error peak is accentuated
   with label noise"; their headline CIFAR-10 demonstrations use **20% random label noise**. They
   do show peaks without label noise (ResNets and CNNs on CIFAR-100, Transformers on IWSLT'14), so
   noise is an amplifier rather than a strict requirement — but **Stephenson & Lee, "When and how
   epochwise double descent happens"** find a **critical noise level below which epoch-wise double
   descent does not occur at all**, and that a model too small to reach the interpolation threshold
   never shows it. <https://arxiv.org/abs/2108.12006> · <https://arxiv.org/pdf/2108.12006>
3. **Where it does appear, the right response is to fix the step sizes, not to pay the epochs.**
   **Heckel & Yilmaz, "Early Stopping in Deep Networks: Double Descent and How to Eliminate it"**
   show epoch-wise double descent is "a superposition of two or more bias-variance tradeoffs that
   arise because **different parts of the network are learned at different epochs**", and that
   **proper per-layer step-size scaling removes the phenomenon and *improves* the early-stopping
   performance** — i.e. you get a better model sooner, not the same model later.
   <https://arxiv.org/abs/2007.10099> · <https://arxiv.org/pdf/2007.10099>

**Datasets.** CIFAR-10, CIFAR-100, Fashion-MNIST, IWSLT'14, WMT'14, with ResNet18, 5-layer CNNs
and Transformers. **No tabular data and no small-sample regression anywhere in the double-descent
literature's core demonstrations.** There is no published evidence that riding through epoch-wise
double descent helps on small tabular regression.

**Where this programme's own curves land.** The `h12-longdeep` trajectory is *literally* a double
descent — "descend to ~20, rise to ~100, descend again to ~200, then diverge". So the phenomenon is
present in our data. And it was *measured*: the second descent is worth **−0.0234 ± 0.0048,
t = −4.92, 5/5 seeds** at L6 (see §6). It bottoms out at **epoch ~200**, and then **diverges**. The
constructive reading is that the second descent is real and already harvested at 3,000 epochs with
patience 400; the destructive reading of the proposal is that 30,000 epochs is 100× past the
bottom, in the divergent region the curves already measured at 10–17% worse.

---

## 4. Deep learning vs gradient boosting on tabular data

The inputs here are a ~2,000-dimensional descriptor/fingerprint vector. That is tabular data in
the sense the benchmark literature means, and it is tabular data of the *specific kind* that the
literature identifies as worst for neural networks.

### 4.1 Grinsztajn, Oyallon, Varoquaux (NeurIPS 2022)

"Why do tree-based models still outperform deep learning on tabular data?" 45 datasets, ~400
random-search iterations per model per dataset. <https://arxiv.org/abs/2207.08815> ·
<https://ar5iv.labs.arxiv.org/html/2207.08815> ·
<https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf>

- **Regime:** "medium-sized data (~10 K samples)", with a separate 50 K-sample analysis. **Our
  n ≈ 4,376 is below even their "medium".**
- **HPO budget:** "**Tree-based models are superior for every random search budget, and the
  performance gap stays wide even after a large number of random search iterations.**" You cannot
  tune your way out of it.
- **Uninformative features — the finding that matters most here:** "removing uninformative features
  reduces the performance gap between MLPs (ResNet) and the other models, while **adding
  uninformative features widens the gap**." GBTs are stable as uninformative features are added;
  MLP/ResNet degrade substantially; FT-Transformer is intermediate. **This is exactly our feature
  matrix.** `drugrx/features.py` builds ECFP4 at `n_bits = 2048` (Morgan radius 2) concatenated
  with a ~210-descriptor RDKit 2D block, optionally plus Avalon (512) and ErG — so ~2,258 columns,
  **~91% of them sparse binary fingerprint bits**, of which only a small minority carry signal for
  any one CYP isoform. We are in the precise regime where Grinsztajn measures the neural net being
  handicapped and the tree not being handicapped at all, and **adding capacity or epochs does not
  touch the cause.** Feature selection or a learned projection does.
- **Irregular targets:** NNs have a "bias toward low-frequency functions"; "smoothing the target
  function on the train set decreases markedly the accuracy of tree-based models, but barely
  impacts that of NNs" — i.e. the tree's advantage *is* its ability to fit irregular functions,
  which an activity cliff in a congeneric series very much is.
- **Conclusion, verbatim:** "On such data, tree-based models more easily yield good predictions,
  with much less computational cost."
- **More data narrows it:** "increasing the train set size reduces the gap between neural networks
  and tree-based models" — which again points at data, not epochs.

### 4.2 Shwartz-Ziv & Armon (Information Fusion 2022)

"Tabular Data: Deep Learning is Not All You Need". <https://arxiv.org/abs/2106.03253> ·
<https://doi.org/10.1016/j.inffus.2021.11.011> ·
<https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360>

- XGBoost generally outperformed the deep models on the 11 datasets tested.
- **Deep models required significantly more HPO iterations to converge**, and XGBoost was both
  faster and more robust to default hyperparameters.
- **The best single result was the *ensemble* of deep models and XGBoost** — best on 7 of 11
  datasets. This is the one pro-neural result in the paper, and it is an argument for **blending**,
  which this programme already does at w(gbm) = 0.30–0.35.

### 4.3 McElfresh et al. (NeurIPS 2023 D&B) — the most useful framing

"When Do Neural Nets Outperform Boosted Trees on Tabular Data?" 19 algorithms × 176 datasets × up
to 30 HP settings. <https://arxiv.org/abs/2305.02997> ·
<https://proceedings.neurips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf>

> "the 'NN vs. GBDT' debate is overemphasized: for a surprisingly high number of datasets, either
> the performance difference between GBDTs and NNs is negligible, or **light hyperparameter tuning
> on a GBDT is more important than choosing between NNs and GBDTs**."

They also find GBDTs are "much better than NNs at handling skewed or heavy-tailed feature
distributions and other forms of dataset irregularities". pIC50 with a censored floor at the
inactive window is exactly a skewed, irregular target.

### 4.4 2024–2026 updates and rebuttals — the state of play is genuinely mixed

The field has moved, and it is fair to say the tabular-DL case is stronger than it was in 2022:

- **TabPFN v2** — Hollmann et al., *"Accurate predictions on small data with a tabular foundation
  model"*, **Nature (2025)** — a transformer pre-trained on 130 M synthetic tabular tasks that does
  **in-context learning**: no gradient training on your data at all. It "outperforms traditional
  methods on datasets with up to **10,000 samples and 500 features**", and **"performs as well with
  half the data as the next best baseline (CatBoost) with all the data"**, at 2.8 s
  (classification) / 4.8 s (regression) against baselines tuned for 4 hours.
  <https://doi.org/10.1038/s41586-024-08328-6> (paywalled; the DOI resolves via Nature's IdP) ·
  v1 paper <https://arxiv.org/abs/2207.01848> · overview
  <https://en.wikipedia.org/wiki/TabPFN> · independent analysis of its strengths/limits
  <https://arxiv.org/abs/2502.17361>
  **This is the single most relevant positive result for our n, and note what it is: a model where
  the compute was spent on *pretraining over synthetic priors*, and inference is seconds. It is the
  opposite of the 30,000-epoch proposal.** The caveat for us is the **500-feature ceiling** —
  2,000-dim fingerprints exceed it and need reduction first.
- **TabM** (Gorishniy et al., ICLR 2025) — "Advancing Tabular Deep Learning with Parameter-Efficient
  Ensembling": an MLP with cheap implicit ensembling beats much fancier architectures. Again the
  lever is **ensembling**, not depth or epochs. <https://arxiv.org/abs/2410.24210>
- **TabR** (Gorishniy et al., ICLR 2024) — retrieval-augmented tabular DL; the lever is
  **neighbours**, i.e. bringing more *data* to bear at inference. <https://arxiv.org/abs/2307.14338>
- **"Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data"** (NeurIPS 2024) —
  the finding is that good *defaults* capture most of what tuning buys.
  <https://proceedings.neurips.cc/paper_files/paper/2024/file/2ee1c87245956e3eaa71aaba5f5753eb-Paper-Conference.pdf>
- **Realistic Evaluation of TabPFN v2 in Open Environments** (2025) — tempers the Nature claims
  under distribution shift and open-environment conditions. <https://arxiv.org/abs/2505.16226>

**What scale does in this literature.** Across all of it, the winning lever is never "train the
same model much longer". It is *ensembling* (TabM, deep-ensemble+XGBoost), *retrieval* (TabR),
*pretraining on a prior* (TabPFN), or *light HPO on a GBDT* (McElfresh). Every one of those is a
different way to spend compute, and three of the four spend it outside the training loop of the
final model.

---

## 5. Molecular property prediction specifically

The short version: on small molecular datasets, descriptors + GBM/RF is a strong and often winning
baseline; learned representations need either much more data or large-scale pretraining to beat it;
and reported deep-learning margins on MoleculeNet-scale benchmarks frequently fail to clear
statistical significance.

### 5.1 Pretraining — the cleanest published null, at our split protocol

**Sun, Dai, Kabir et al., "Does GNN Pretraining Help Molecular Representation?"** (NeurIPS 2022).
<https://arxiv.org/abs/2207.06010>

Self-supervised graph pretraining delivers **−0.578% to +0.128%** on downstream molecular tasks
once you use **rich input features and scaffold splits** — against a baseline standard deviation of
**0% ± 1.626%**. In other words: **inside the noise, and the sign is as often negative as
positive.** The two conditions that kill the apparent benefit are exactly our two conditions
(descriptor-rich inputs, scaffold splitting). This is the single most load-bearing negative result
for the "more pretraining compute" variant of the proposal.

### 5.2 Descriptors + trees vs learned representations: five large head-to-heads, all the same way

| Study | Scale | Result |
|---|---|---|
| **Jiang et al. 2021**, *J. Cheminform.* — <https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00479-8> | 11 datasets, 8 algorithms (SVM/XGB/RF/DNN vs GCN/GAT/MPNN/AttentiveFP), TPE-tuned | Descriptor models won **6/11** datasets and **24 of 33 top-3 slots (73%)**. Verbatim: *"on average the descriptor-based models outperform the graph-based models in terms of prediction accuracy and computational efficiency. SVM generally achieves the best predictions for the regression tasks."* **All three graph wins were the larger sets** (Lipophilicity 4.2 k, Tox21 7.8 k, HIV 40.7 k). On HIV, XGBoost took **23.65 s** vs AttentiveFP **677.5 s**. |
| **Xia et al., NeurIPS 2023** — <https://chengtan9907.github.io/assets/publications/nips23_ifm.pdf> | 12 models × 15 datasets × 50 seeds; non-deep features = **881 PubChem FP + 307 SubFP + 206 MOE descriptors** — almost exactly our block | *"(i) Deep models are generally unable to outperform non-deep ones; (ii) **The failure of deep models on MPP cannot be solely attributed to the small size of molecular datasets**; (iii) … XGB and RF that use molecular fingerprints as inputs tend to perform better."* Non-deep ranked top-1 on **11/15**; trees top-1 on **8/15**. A SMILES transformer **pretrained on 861,000 molecules** still lost to XGB/RF on **13/15**. |
| **Deng et al. 2023**, *Nat. Commun.* — <https://doi.org/10.1038/s41467-023-41948-6> | **62,820 models**, 30 repeated scaffold splits, pairwise tests | *"representation learning models exhibit limited performance in molecular property prediction in most datasets … dataset size is essential for representation learning models to excel."* On the **24 ChEMBL pIC50 activity datasets** — the closest published analogue to our task — *"MolBERT shows the highest RMSE (p < 0.05) in **21** activity datasets."* MolBERT becomes competitive only at **n = 100 K**. And on variance: *"representation learning models, presumably due to their large numbers of parameters, tend to show **greater metric variability**, which is further negatively correlated with metric mean values."* |
| **van Tilborg, Alenicheva & Grisoni 2022** (MoleculeACE), *JCIM* — <https://doi.org/10.1021/acs.jcim.2c01073> | **720 models, 24 approaches, 30 ChEMBL pIC50/pKi targets** | *"While all methods struggled in the presence of activity cliffs, machine learning approaches based on molecular descriptors outperformed more complex deep learning methods."* *"no statistical difference was found between SVM, GBM, or RF coupled with ECFPs"* — **our LightGBM is already in the statistical top tier.** Pretraining scale did nothing: transformers pretrained on 10 M SMILES *"did not perform better than LSTMs, which were pretrained on 36,281 molecules only."* |
| **Guo & Ding 2026**, "Do Larger Models Really Win in Drug Discovery?" — <https://arxiv.org/abs/2604.26498> | 26 ADME/tox/bioactivity endpoints, 156 fold comparisons, 1.2 k–49 k molecules | Classical ML wins **116**, GNNs **25**, pretrained sequence models **12**, LLM-SAR **3**. *"predictive performance depends on the fit among model, task and validation scenario, **not on scale alone**."* *(2026 preprint, not peer-reviewed.)* |

**The crossover is measured, three independent ways, and it is ~10× our dataset.** Kamuntavičius
et al. 2025 (*J. Cheminform.* 17:108, <https://doi.org/10.1186/s13321-025-01041-0>) ran 10-fold CV ×
25 datasets with Nemenyi post-hoc tests and tabulated MPNN vs CatBoost R² by dataset size: CatBoost
wins at **every** size from 662 to 7,323 (half_life 0.16 vs 0.25; caco2 0.52 vs 0.61; ppbr 0.37 vs
0.44; ld50 0.14 vs 0.25) and MPNN only overtakes on **nih_solubility at n = 36,238** (0.32 vs 0.27).
van Tilborg finds deep learning wins on *physicochemical* cliffs *"potentially due to the higher
number of training samples (approx. 20,000 molecules)"*. Deng puts MolBERT's crossover at 100 K.

Two further findings from the same Kamuntavičius paper that bear directly on us:
- *"catboost, svm and lightgbm all perform statistically similarly according to the Nemenyi
  post-hoc test"* — **LightGBM is not the thing to replace.**
- Best feature recipe for regression was `rdkit_desc + erg + ecfp4 + avalon`, and *"**Adding more
  features to these combinations decreases the model performance** across the datasets."* More input
  dimensionality is not free.

**The D-MPNN authors say it themselves.** Yang et al. 2019, the Chemprop paper
(<https://pubs.acs.org/doi/10.1021/acs.jcim.9b00237>):

> *"We also note that **on small datasets (up to 1000 training molecules) fingerprint models can
> outperform learned representations**, which are negatively impacted by data sparsity."*
>
> on the 200 RDKit global features: *"the benefit is much more pronounced on the extremely small
> PDBbind-C (core) dataset … **the features may help compensate for the lack of training data** …
> we hypothesize that the features may help to **regularize** a representation derived from a small
> dataset."*

And their own effect sizes for the two levers under discussion: *"most datasets experience a
moderate **2–5% improvement** in performance following hyperparameter optimization"*, while
ensembling *"only provides a small **1–5% benefit**."* Those are the magnitudes available — and
neither of them is "train 25× longer".

### 5.3 The ablation that should end the capacity argument

**fastprop** (Burns & Green, *J. Cheminform.* 2025;17:73,
<https://doi.org/10.1186/s13321-025-01013-4>) is architecturally *our model*: a feed-forward net on
1,613 mordred descriptors, two hidden layers of 1,800 units. It matches or exceeds Chemprop at ~10×
lower wall clock. Then, on **TDC Pgp (n = 1,275, scaffold split)**:

| model | AUROC |
|---|---|
| TDC leaderboard leader | 0.938 ± 0.0023 |
| Chemprop (+ molecular features) | 0.886 ± 0.016 |
| fastprop (2 × 1800 MLP) | 0.903 ± 0.033 |
| **a zero-hidden-layer, purely *linear* model on the same descriptors** | **0.917 ± 0.016** |

Verbatim: *"Remarkably, the **linear QSPR model outperforms both Chemprop and fastprop**, approaching
the performance of the current leader."* On a ~1.3 k-molecule descriptor task, **zero capacity beat
two neural networks.** (Honest caveat the authors supply: on their ARA dataset linear falls to
0.824, so the claim is "capacity is not the binding constraint, and it is dataset-dependent" — not
"linear always wins".)

### 5.4 Nobody in the published record trains anything like 30,000 epochs

| Source | Setting | Epochs |
|---|---|---|
| **Chemprop v2 CLI default** — <https://github.com/chemprop/chemprop> | the most-used MPP package; powers ADMET-AI and several Polaris top-5 entries | **50** |
| Xia et al., NeurIPS 2023 | all 9 deep models, 15 datasets | *"maximal epoch as 300"*, early stopping at patience 50 |
| van Tilborg et al. 2022 | 720 models, 30 pIC50 targets | **300** (patience 10); some 500 (patience 10); 100 for pretrain/finetune heads |
| fastprop | FNN on 1,613 descriptors | **30 s – 2 min** total wall clock, including featurisation |
| **Kearnes et al. 2016**, industrial multitask ADMET — <https://arxiv.org/abs/1606.08793> | the one multi-GPU-week budget in the literature | *"Models were trained for **1 M steps**"*; the side-information variant *"required several weeks with a single GPU (all models were trained for at least **40 M steps**)"* |

**And Kearnes is the decisive precedent, because that enormous budget did not pay.** Verbatim:
*"adding side information did not consistently improve performance … on average, **adding side
information either had no significant effect or damaged performance**."* They also note *"some
validation set AUC scores were still increasing when training was stopped"* — the loss kept falling
and the benefit never arrived. Their multitask effect overall: **+0.01 to +0.03 AUC, with only 2 of
10 comparisons statistically significant**, and bigger tasks benefiting *less* (r² ≈ 0.12).

For scale: 30,000 epochs × ~2,000 molecules at batch 64 ≈ **1 M optimiser steps** — i.e. Kearnes'
budget, spread over 1/100th of the data and 1/25th of the tasks.

### 5.5 There may be very little headroom left to buy

- **Landrum & Riniker 2024**, "Combining IC50 or Ki Values From Different Sources is a Source of
  Significant Noise" (<https://doi.org/10.26434/chemrxiv-2024-2smhk>): with minimal curation,
  *"almost **65% of the points differ by more than 0.3 log units, 27% differ by more than one log
  unit**, and … Kendall's τ is only **0.51**"*; maximal curation gets τ to 0.71.
- The **Polaris/ASAP** winning potency entry scored **MAE 0.509 ± 0.022 pIC50**, against the same
  paper's statement that *"inter-assay variation [is] approximately 1 kcal/mol (approx 0.5–0.7 pKi
  units)"*. The winner is sitting on the assay noise floor.
- TDC auditors: *"Given experimental uncertainty and inter-laboratory variability, the coefficient
  of determination R² for most biological endpoints **rarely exceeds 0.7–0.8**."*
  <https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full>

**And the hardest number: the inter-laboratory ceiling on pIC50 itself.** Kalliokoski, Kramer,
Vulpetti & Gedeck, *PLoS ONE* 8(4):e61007 (2013) —
<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061007> — took **9,465 pIC50
values / 16,844 independent measurement pairs on identical protein–ligand systems** in ChEMBL:

| quantity | value |
|---|---|
| **R² between two independent pIC50 measurements of the *same* system** | **0.40** raw; **0.53** after removing ΔpIC50 ≥ 2.5 annotation errors |
| σ of ΔpIC50 (pairs) | 0.87 log units → **σ ≈ 0.62 log units per measurement** |
| pKi equivalent | σ ≈ 0.49 |
| fraction of pairs with ΔpIC50 < 1 log unit | ~70% (max observed 7.7; unit errors most common) |

**If our four endpoints are pIC50-like, an inter-lab R² of 0.40–0.53 is the measurement ceiling, and
every intervention must be priced as distance-to-ceiling rather than against zero.** Corroborated
from the other side by Sultan et al. 2025 on **seven ADME regression endpoints of 1.8 K–4 K molecules
each — our exact size and task type** — under a Butina-cluster split: the best model *of any kind*
reached **R² ≈ 0.3** on four endpoints and < 0.6 on the other two; without domain adaptation R² was
**0.02 (HPPB), 0.06 (HLM), 0.08 (Solubility_AZ)**, best 0.33. <https://arxiv.org/abs/2503.03360>

### 5.5b The mechanism by which a noisy test set *punishes* capacity

This is the piece that converts the noise ceiling from a caveat into an argument, and it is the most
important single addition to this review.

**Northcutt, Athalye & Mueller, NeurIPS 2021 D&B** —
<https://arxiv.org/abs/2103.14749> — hand-validated label errors in ten canonical test sets
(ImageNet val **5.83%**, CIFAR-100 **5.85%**, QuickDraw 10.12%; "an average of at least 3.3% errors";
MTurk confirmed **51%** of flagged candidates as genuinely mislabelled). Then they re-ranked models
on the corrected sets. On the **correctable subset the rankings invert**:

| model | rank on original test set | rank on corrected |
|---|---|---|
| NASNet-large | **1 / 34** | **29 / 34** |
| Xception | 2 / 34 | 24 / 34 |
| ResNet-18 | **34 / 34** | **1 / 34** |
| ResNet-50 | 20 / 24 | 2 / 24 |

And they locate the threshold precisely: ImageNet's measured noise prevalence is N = 2.9%, and
*"The ResNet-50 and ResNet-18 benchmarks cross at noise prevalence **N = 9%**."* In other words:
**above roughly 9% label noise, the bigger model is ranked worse than the smaller one — and the
smaller one is actually better.**

A scaffold-split pIC50 target with inter-lab R² of 0.40–0.53 sits far above N = 9%. **So this
programme's repeated finding that capacity does not pay is exactly what the noise literature
predicts, and it supplies the mechanism: at our noise level the evaluation itself prefers the
lower-capacity model, and it is right to.** That reframes "capacity is a null" from a disappointing
result into a correctly-functioning one.

**The constructive corollary — cleaning beats capacity, measured.** Confident learning / cleanlab
(*JAIR* 70, <https://arxiv.org/abs/1911.00068>), CIFAR-10 ResNet-50 under asymmetric noise, test
accuracy: baseline **78.4 / 60.2 / 27.0** at 20/40/70% noise → best CL **91.1 / 87.1 / 41.1** —
**+12.7 / +26.9 / +14.1 points from cleaning the data alone, no model change.** Verbatim:
*"**Data-centric AI methods (CL, INCV, Mixup) outperform model-centric methods** (SCE-Loss,
MentorNet, Co-Teaching, S-Model)."* On Amazon Reviews with a *multinomial logistic regression*:
84.4 → **90.3** while **removing 25–49% of the training set**, and *"Baseline training begins to
overfit to noise with additional epochs, whereas CL test accuracy continues to increase."*

Read the pair honestly: on an *already clean* dataset (ImageNet) cleaning buys ~+0.5 to +1.2 points.
**The gain scales with the noise you actually have** — so the first move is to measure ours, not to
assume either number.

**One concrete check this implies, costing nothing.** Sultan et al. found *"~27% of the HLM data has
the same target value"* — censored labels silently entered as points, *"introducing noise to the
model"* — and removed them. **Look for a spike at a single value in each of the four endpoints
before provisioning any GPU hours.** (This programme is already ahead here: the floor window
`[1.07, 3.52]` and the interval loss exist precisely because the screen's inactives are censored.
That decision now has external support it did not have when it was made.)

### 5.5c The closest published match to our dataset shape, and what it says about selection

**Krstajic, Buturovic, Leahy & Thomas, *J. Cheminformatics* 6:10 (2014)** —
<https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-6-10> — studies seven QSAR datasets,
one of which is **Mutagen at n = 4,335 compounds × 1,283 descriptors**. That is essentially our
4,376 × ~2,258. Protocol: **50 repeats of 10-fold CV** for selection and repeated **nested** CV
(50×10 outer, 50×10 inner) for assessment.

Their Table 2 shows the distribution of the "optimal" hyperparameter chosen by 50 *independent
single* 10-fold CV runs. On Mutagen, ridge logistic regression's λ across 9 bins spanning ~3× came
out **7, 2, 1, 6, 5, 8, 4, 6, 7 — essentially uniform.** Verbatim: *"**It is obvious that the model
selected by single cross-validation may have high variance**"* and *"**No single cross-validation run
provided for reliable selection of the best model on those datasets.**"*

**But the good news is specific to our size, and it is worth stating plainly:** *"In case of Mutagen
dataset … where intervals of nested cross-validation errors are narrow and similar to
cross-validations', we can conclude that **if we randomly remove 10% of samples, the quality of
models remains almost the same**."* The catastrophic instability is in their tiny set (bbb2, n = 79,
misclassification swinging 0.13–0.23).

**So at our n and p: expect the selected hyperparameters to be near-random draws from a plateau,
while the achieved loss is stable.** Two consequences — don't trust a single-CV argmin (which is
precisely the extreme-value artefact §6 documents, with two seeds landing on the same score 335
epochs apart), and don't chase the last hyperparameter decimal, because there probably isn't much
there. Their own recommendation is to report **minimum and maximum** nested-CV error rather than the
mean, *"because the variability is such that reporting a single error value may be misleading."*

**The companion warning, which bites at exactly our p/n ratio.** Vabalas, Gowen, Poliakoff & Casson,
*PLoS ONE* 2019 —
<https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365> — generated **pure-noise
data with a true chance level of 50%** and ran K-fold CV with model selection done *inside* the folds
being reported. It returned *"accuracies considerably higher than the theoretical chance level of
50%"*, and *"**the difference was still evident even at the sample size of N = 1000**."* Nested CV and
a clean train/test split did not differ from chance. **With ~2,258 descriptors against 1,300–2,335
labels per endpoint, we are squarely in the regime where this produces signal from nothing** — which
is the formal version of what §7 row 1 is about.

### 5.6 What actually won the Polaris / ASAP ADMET blind challenge (2025) — data, not FLOPs

"A Computational Community Blind Challenge on Pan-Coronavirus Drug Discovery Data", *JCIM* 2026,
DOI 10.1021/acs.jcim.5c02106 — 381 submissions, 66 participants, **temporal** splits.
<https://www.repository.cam.ac.uk/bitstreams/27c99979-6475-4bc9-abbe-e7e1380ac73a/download>

Yes, deep learning won — and the reason is stated explicitly:

> *"For ADMET endpoint prediction, **multi-task GNNs proved to be the best approach overall,
> benefiting from the ability to incorporate auxiliary data**."*
>
> *"**the top-performing approaches across the three subchallenges leveraged either auxiliary data
> (public or proprietary) or pre-training beyond the datasets provided by ASAP.** This highlights
> the critical role of data availability in model performance."* — the top performer *"employed
> proprietary data directly."*

And one participant write-up cuts the other way entirely: *"**XGBoost models trained on a feature
space of 55 descriptors outperformed other tree-based models and GNNs on four out of the five
endpoints**"* (KSOL, LogD, HLM, MLM, MDR1-MDCKII).
<https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/68bab055728bf9025e90df8d/original/experiments-with-data-augmented-modeling-of-adme-and-potency-endpoints-in-the-asap-polaris-open-admet-antiviral-challenge.pdf>
Another placed 4th with a *simple* multitask GNN pretrained on industry-released **predicted**
labels — again extra data, *"without using any proprietary data or commercial featurization
methods"* (<https://doi.org/10.1021/acs.jcim.5c02030>).

**Read the whole challenge as one sentence: what separated the top of that leaderboard was access
to more labelled data, not a longer training run.**

### 5.7 Treat published ADMET leaderboard deltas with suspicion

The TDC critical assessment (<https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full>)
reports top-3 occupancy of **MapLight 17/22** and **MapLight+GNN 15/22** — MapLight being
fingerprints + descriptors into a GBDT — and identifies **direct or indirect data leakage in
MiniMol, GradientBoost and XGBoost entries**, with only 3 of the assessed methods passing every
reproducibility check. ADMETboost, a plain XGBoost on fingerprints + descriptors, was *"ranked first
in 18 [of 22] tasks"* (<https://arxiv.org/abs/2204.07532>). Kamuntavičius's whole methodological
point — 250 paired measurements and Nemenyi tests, because single hold-out differences are not
interpretable — applies to us directly: **the proposal's hoped-for gain is smaller than our own
measured seed band of 0.013.**

### 5.8 Where molecular scaling laws do exist, they are pretraining laws that plateau

Uni-Mol2 is the billion-parameter molecular pretraining scaling study, and its own finding is a
ceiling: "the final validation losses of scaling models start to deviate from the predicted scaling
law, suggesting that under the current data regime, further model scaling may lead to **diminishing
returns** as the loss approaches a plateau", with properties like HOMO/LUMO/gap/ZPVE **converging as
model size increases**.
<https://proceedings.neurips.cc/paper_files/paper/2024/file/53923bb44655a7defb31c7744c01b62b-Paper-Conference.pdf>
And "Diversity Beats Size Scaling for Chemical Language Models" makes the complementary point that
pretraining *volume* without diversity does not deliver.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC13439062/>
Crucially, **all of these are laws for the pretraining corpus, not for epochs on a 2,000-molecule
fine-tuning set.** They say "pretrain on more diverse molecules", which is a data argument again.

### 5.9 This programme's own capacity measurements

The most directly relevant evidence available, and it points the same way:

- **TDC capacity nulls** on 95–2,600-molecule folds: capacity failed to help.
- **H12** on >10 k molecules from one platform, 4 paired seeds, identical split and data order:
  depth (1024×3 → 1024×5) moved the score by **+0.0019 ± 0.0098, p = 0.723**, while moving the best
  epoch by **+116.2 ± 20.2, 4/4 seeds, p = 0.0014**. Width at matched member count: **+0.0000**.
- The programme's summary: "**Capacity doesn't help at n = 500 and doesn't help at n = 10,000
  either, it just converges slower.**"
- The representation half was tested directly and answered negatively (`fp-mlp` against 0.6622),
  after which "**architecture work stops here**" was recorded as an allocation decision.
- `research/25-trm-small-data.md` adds the field-coverage fact: the intersection of
  (recursive/latent reasoning, adaptive depth, deep supervision) with (molecular property, QSAR,
  ADMET, bioactivity, IC50) returns **`totalResults = 0`** on arXiv. Nobody has made long-training
  latent-reasoning models work on molecular regression, because nobody has tried.

**But H12's depth null was later overturned, and §6 is where that is reported honestly.** Read
§5.9 as "single-factor capacity is a null", not as "capacity can never pay".

This programme's own measurements are consistent, and are the most directly relevant evidence
available:

- **TDC capacity nulls** on 95–2,600-molecule folds: capacity failed to help.
- **H12** on >10 k molecules from one platform, 4 paired seeds, identical split and data order:
  depth (1024×3 → 1024×5) moved the score by **+0.0019 ± 0.0098, p = 0.723**, while moving the best
  epoch by **+116.2 ± 20.2, 4/4 seeds, p = 0.0014**. Width at matched member count: **+0.0000**.
- The programme's summary: "**Capacity doesn't help at n = 500 and doesn't help at n = 10,000
  either, it just converges slower.**"
- The representation half was tested directly and answered negatively (`fp-mlp` against 0.6622),
  after which "**architecture work stops here**" was recorded as an allocation decision.

---

## 6. The honest case FOR more compute: the depth × schedule interaction

This section exists because the evidence is not one-sided, and a review that omitted this would be
confirming a prior rather than settling a disagreement.

**On 2026-10-06 this programme reversed itself twice and landed on a real effect.** Five seeds per
arm, one shared split, every epoch judged by the corrected metric, `metric_id` stamped:

| arm | n | mean | sd | best epochs |
|---|---|---|---|---|
| h12-base (d1024 L3, 400 ep / pat 150) | 5 | 0.7055 | 0.0134 | 15 25 15 20 25 |
| h12-base2048 (same, batch 2048) | 5 | 0.6968 | 0.0098 | 10 10 15 10 10 |
| h12-deep (d1024 L6, 400 ep) | 5 | 0.6969 | 0.0056 | 20 40 15 15 30 |
| h12-wide (d2048 L4, 400 ep) | 5 | 0.6928 | 0.0107 | 15 10 10 10 15 |
| h12-longbase (d1024 L3, 3000 ep / pat 400) | 5 | 0.6998 | 0.0096 | 310 365 15 20 340 |
| **h12-longdeep (d1024 L6, 3000 ep / pat 400)** | 5 | **0.6764** | 0.0065 | 205 235 215 190 235 |

| comparison | effect | t | wins | verdict |
|---|---|---|---|---|
| **depth @ 3000 ep** | **−0.0234 ± 0.0048** | **−4.92** | **5/5** | **DECIDED** |
| depth @ 400 ep | −0.0085 ± 0.0070 | −1.22 | 4/5 | inside noise |
| width, matched at bs 2048 | −0.0040 ± 0.0089 | −0.45 | 3/5 | inside noise (157 seeds to resolve) |
| schedule, on L3 | −0.0057 ± 0.0028 | −2.02 | 4/5 | inside noise |
| **schedule, on L6** | **−0.0205 ± 0.0050** | **−4.07** | **5/5** | **DECIDED** |

> "**Depth pays only with a long schedule, and a long schedule pays only with depth.** … Neither
> factor is the effect; the combination is."
>
> "The mechanism is visible in the best epochs: `longdeep` peaks consistently at **190–235**, while
> the 400-epoch/patience-150 run stops the same architecture at **15–40**. The deeper net plateaus
> early, then improves again — and **patience 150 kills it before the second descent**."

So the pro-compute case, stated at full strength: **there is a measured second descent on this
exact problem; it is worth 0.023 on validation at t = −4.92 across 5/5 seeds; and every
single-factor sweep before it was structurally incapable of seeing it.** Anyone who says "compute
never helps here" is overclaiming, and this programme said so in its own log: "'Bound by data and
representation, not by model class or compute' is **too strong as stated**."

**But the same record bounds the effect precisely, and the bounds kill the 30,000-epoch version.**

1. **It is 3,000 epochs, not 30,000.** The optimum is epoch **190–235** at L6. Early stopping fires
   at 591–766. The curves rise monotonically after ~365 and are 10–17% worse by 700.
2. **It requires depth.** The schedule alone at L3/32 members is **−0.0000** — identical to four
   decimals per endpoint. The proposal as phrased ("train for 30,000 epochs") without changing the
   architecture is the arm that was already measured at exactly zero.
3. **It requires *fewer* ensemble members, and that is a trade.** L6 cannot hold 32 members (16 at
   L6 already peaks at 37.6 GB of 39.5). The shipped v3deep gain of **−0.0131** is "depth + member
   drop" combined, not depth alone.
4. **It transfers at ~17%.** Shipped: validation 0.6615 → 0.6484 (−0.0131); board 0.8752 → 0.8598
   (−0.0154 on MA-ST-RAE), rank 163 → 159. "**about a sixth of what validation promised.**" The
   remaining distance is the real story: **0.86 against a leader at 0.3772 and a mean predictor at
   1.0.**
5. **The band-min selection is partly an extreme-value artefact.** Two `long-base` seeds landed on
   0.4441 and 0.4448 — the same score — with best epochs **335 apart** (360 and 25). "An optimum
   that relocates by 335 epochs without moving the score is not an optimum." Some of the apparent
   late-epoch gain is the minimum of ~160 autocorrelated draws from a flat noisy band.

**Net of §6:** budget ~**0.004 of real test improvement** for the entire capacity+schedule axis,
already banked. Scaling it 10× further is not supported by any curve in the record.

---

## 7. What the A100-hours should buy instead — ranked

### 7.0 The measured compute frontier for a dataset our size fits in ~20 CPU-minutes

Before ranking anything, there is one paper that prices the whole question. **Holzmüller,
Grinsztajn & Steinwart, "Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular
Data", NeurIPS 2024** — <https://arxiv.org/abs/2407.04491> — benchmarks 118 meta-train + 90
meta-test datasets (1 K–500 K samples) and plots **normalised RMSE against CPU-seconds per 1,000
samples**. Selected points from their meta-test regression frontier (42 datasets):

| Method | nRMSE | ~CPU s / 1 K rows | vs LGBM default |
|---|---|---|---|
| LGBM-D (library defaults) | 0.446 | 0.025 | — |
| CatBoost-TD (meta-learned defaults) | 0.407 | 0.8 | −8.7% |
| LGBM-HPO (50-step random search) | 0.4045 | 17 | −9.3% |
| **Ensemble-D** (portfolio at defaults + greedy selection) | **0.3925** | **3.8** | **−12.0%** |
| RealMLP-TD | 0.378 | 6.5 | −15.2% |
| **Ensemble-TD** | **0.3595** | **7** | **−19.4%** |
| RealMLP-HPO | 0.359 | **250** | −19.5% |
| **Ensemble-HPO** (everything) | **0.344** | **270** | **−22.9%** |

Two readings that matter:

- **Ensemble-TD (0.3595 at ~7 s/1 K) ties RealMLP-HPO (0.359 at ~250 s/1 K) — the same score for
  ~36× less compute.** Authors, verbatim: *"Simply trying all default algorithms is faster and very
  often better than (naive) single-algorithm HPO."*
- **Scaled to our 4,376 rows, the far right-hand end of that frontier — a portfolio of 4–5
  algorithms, each HPO'd, bagged 5×, blended by greedy ensemble selection — is ≈ 270 × 4.4 ≈ 20
  CPU-minutes.** Not GPU. Not hours. **Minutes, on a CPU.**

That is the honest frame for "spend 12–24 A100-hours": **the entire measured tabular frontier for a
dataset this size is already affordable in under an hour of CPU, and the proposal would spend 40–60
A100-hours moving along an axis that is not on the frontier at all.** The defensible GPU uses are
narrow: (i) TabPFN-family inference, (ii) extracting frozen embeddings from a molecular foundation
model, (iii) running the sweep **wide** — many seeds × folds × configs in parallel — rather than
**long**.

### 7.1 The ordering of levers, with measured effect sizes

Across both literatures the ranking is consistent, and *capacity and epochs come last*:

| lever | measured effect (relative) | source |
|---|---|---|
| **Molecular representation** | **22.4% → 56.9% → 86.2% win rate** (same model, data, splits) | Ben Hicham et al., <https://arxiv.org/abs/2604.16123> |
| **Training recipe** (numerical feature embeddings, Adam β2 = 0.95 not 0.999, non-constant LR schedule, robust-scale + smooth-clip) | **+20.6%, +22.8%, +13.5%, +9.5%** regression error *each* if you get it wrong | Holzmüller Tables B.1/B.2 |
| **Ensembling / refitting** | **+4–6%** at 5×; **+4.1% at ~1× extra** for refit-on-all-data | Holzmüller Tables B.3/B.4 |
| **HPO** | **+0.13 norm. R²** for XGBoost, but **83% of it by iteration 100** | Grinsztajn Fig. 1 |
| **Architecture choice among tuned tabular models** | **~1–3% total spread across 15 models** | TabM Fig. 3, <https://arxiv.org/abs/2410.24210> |
| **More epochs / more capacity on a fixed small set** | **no measured benefit found** | — |

The single most striking number is the first. Ben Hicham et al. ran 58 Polaris + MoleculeACE tasks
(≤ ~6,000 samples, expert-curated splits, 5 seeds, Tukey HSD) and got, for **the same model on the
same splits**, win rates of **TabPFN-Morgan 22.4% → TabPFN-RDKit2d 56.9% → TabPFN-CheMeleon
(frozen embeddings) 86.2%** — *"the choice of molecular representation is a major driver of
performance … **Morgan fingerprints performing substantially worse**."* A ~4× swing in win rate from
the features alone, which is larger than everything else in the table combined. **Our primary
feature block is a 2,048-bit Morgan fingerprint.**

Also from that paper, and counter-intuitive: *"pairing TFMs with **frozen** molecular
representations not only matches but substantially exceeds the performance of task-specific
fine-tuning on these benchmarks"* — fine-tuned CheMeleon scored 41.4% against 86.2% for the same
model's frozen embeddings under an in-context learner. **Gradient-training the representation was
worse than not gradient-training it**, which is the opposite of what the proposal assumes.

### 7.2 The ranked table

Ranking is by **expected test-score movement per A100-hour**, using this programme's measured
transfer rate (~17% validation → test) to discount anything validated only in-sample. The gap to
close is **0.86 → 0.38**; the capacity axis has delivered ~0.015 of it.

| # | Intervention | Measured effect | Cost | Why it ranks here |
|---|---|---|---|---|
| **0** | **Fix the measurement before measuring anything.** Repeated (≥10×) CV for *selection*, nested or truly held-out for *reporting*, and decide with **P(A > B) ≥ 0.75 across seeds rather than Δmean**. | Bouthillier et al., MLSys 2021 (6.4 GPU-years, each variance source randomised 200×, <https://arxiv.org/abs/2103.03098>): single-point comparison has **≈10% false-positive and ≈75% false-negative** rates; `P(A>B) ≥ 0.75` gives **≈5% / ≈30%**. They calibrate the "meaningful difference" threshold against paperswithcode at **δ = 1.9952σ** — *the typical published SOTA gain is almost exactly 2σ of benchmark noise*. Their recommendations, verbatim: *"As many sources of variation as possible should be randomized"* and *"**Resampling techniques such as out-of-bootstrap should be favored instead of fixed held-out test sets.**"* | Hours, one-off | **Everything below is unmeasurable without this.** Krstajic et al. 2014 measured it on *our exact shape* — see §5.5c. |
| **1** | **Stop selecting on the 922 validation rows — and use greedy ensemble selection, not a fitted weight.** Nested CV for the blend weight + the four per-endpoint offsets. | Removes a **measured** optimistic bias: in-sample offset value **+0.0211** vs honest out-of-sample **~−0.004**; the submission realised **−0.0061**. ⚠ **And the method matters more than I first wrote.** Caruana et al., ICML 2004 (<https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf>) ran *almost exactly our setup* — ~2,000 models, a **1,000-point** hillclimbing set — and measured mean normalised score: **ensemble selection 0.969 · best single model 0.946 · uniform average of all models 0.890 · logistic-regression stacking 0.406.** Verbatim: *"**regression overfits dramatically when there are 2000 highly correlated input models and only 1k points in the validation set**"*; uniform averaging also loses, because it *"cannot overfit"* but cannot discriminate either. auto-sklearn reached the same conclusion: stacking and numerical optimisation were *"prone to overfitting and computationally costly"*, while *"ensemble selection was fast and robust."* | **CPU-only, ~0 GPU-hours** | Infinite benefit per GPU-hour, and the one error already shown to have cost a submission. **Use selection *with replacement*, *sorted initialisation* (adds 5–25 of the best models first), and *bagged* selection (20 bags at p = 0.5) — without the bagging guard Caruana's 8.7% gain drops to ~6.2%.** Our 922 rows fitting a blend weight plus four offsets is the small-validation-set regime these results describe. |
| **2** | **Change the representation.** Our primary block is a 2,048-bit Morgan fingerprint — the *worst* feature class measured. Swap/augment with RDKit-2D + Mordred + ErG + Avalon, and with **frozen** CheMeleon/MiniMol embeddings. | **22.4% → 56.9% → 86.2% win rate** on 58 Polaris/MoleculeACE tasks for the *same model on the same splits* (Morgan → RDKit2d → frozen CheMeleon). <https://arxiv.org/abs/2604.16123> Largest effect in the whole review. Sun et al. add: hand-crafted rich features bought **more than any self-supervised pretraining objective** (+1.3–3.1% vs +0.12 ± 1.5%). | One featurisation pass; embeddings are a forward pass | **Biggest measured lever, and it is cheap.** Note it is *frozen* embeddings that win — fine-tuned CheMeleon scored 41.4% against 86.2% frozen. Gradient-training the representation was *worse*. ⚠ **One genuine disagreement in the literature, and it lands on us:** van Tilborg (30 ChEMBL **bioactivity** targets, 615–3,657 molecules) finds *binary fingerprints beat physicochemical descriptors*, while Sultan (7 **ADME** endpoints) and Deng find the reverse. Task type is the likely discriminator — and CYP inhibition IC50 is bioactivity-flavoured. So run this as an **arm, not an assumption**: Morgan vs RDKit2d/Mordred vs frozen embeddings, on V1+V2, same splits. |
| **3** | **Fix the training recipe before touching capacity.** Numerical feature embeddings, Adam β2 = 0.95 (not 0.999), a non-constant LR schedule, robust-scale + smooth-clip preprocessing, and early-stopping **on the evaluation metric**. | Getting each of these wrong costs **+20.6%, +22.8%, +13.5%, +9.5%** relative regression error (Holzmüller Tables B.1/B.2, LR re-tuned per ablation). Stopping on the wrong metric costs ~2%. <https://arxiv.org/abs/2407.04491> | **Free** — these are config lines | An order of magnitude larger than the 1–3% that separates tuned tabular architectures. This is the strongest quantitative case in the literature that **recipe > capacity** on small tabular data. |
| **4** | **More labels on the broken endpoints (1A2, 2D6).** ChEMBL CYP2D6 `CHEMBL289` = **2,942** pChEMBL compounds vs the challenge's **1,493** (3×); Veith AID 1851 continuous AC50 (3.7×), as **rows with a learned source offset**. | "**More data — the one thing that moved anything in this programme.**" Every Polaris/ASAP top finisher won by bringing auxiliary or proprietary data. Grinsztajn: more data is also what narrows the NN↔tree gap. | Low: featurisation is the CPU bottleneck | The axis the whole scaling literature says binds. The Chinchilla/SD analogy, read correctly, points *here*. |
| **5** | **Refit the early-stopped model on all the data.** | **+4.1%** relative regression error for LightGBM (meta-test-reg, 95% CI [3.2, 4.9]) at **~1× extra training** — the best benefit-per-compute entry in Holzmüller's whole ablation. *"refitting is mostly better than bagging."* | ~1× one extra fit | Almost free, and almost nobody does it. Caveat: a refitted model has no validation score, so it cannot be used for selection. |
| **5b** | **Post-hoc ensemble the configs you have ALREADY trained** (greedy/Caruana weighting over every stored prediction vector in `runs/`, instead of picking the argmin). | **+82 Elo for LightGBM and +220 Elo for RealMLP at *zero extra training compute*** (TabArena, 51 datasets, ~25 M model fits, <https://arxiv.org/abs/2506.16791>). For MLPs, post-hoc ensembling of tuned configs is worth **more than twice** as much as picking the single best config. And it **reorders the leaderboard**: *"the top three models in our leaderboard (TabM, LightGBM, RealMLP) would all be worse than the actual fourth-best model (CatBoost) **without post-hoc ensembling**."* 15 GES iterations suffice; more give *"no measurable improvement"*. | **Zero** — the predictions already exist on disk | **The best benefit-per-compute item in this entire review.** We already store per-seed/per-arm prediction vectors; nothing new has to be trained. This also directly attacks the extreme-value selection bias documented in §6. |
| **5c** | **Label-distribution smoothing (LDS)** — convolve a kernel over the empirical label density and re-weight the loss by its inverse. **Works in LightGBM today via `sample_weight`.** | STS-B MSE **0.974 → 0.914 (−6.2%)**; medium-shot 1.520 → 1.027 (**−32%**) with FDS; IMDB-WIKI few-shot MAE 26.33 → 22.95 (**−12.8%**). Motivation is itself measured: smoothed label density correlates with the error distribution at **−0.83**, raw density does not. <https://arxiv.org/abs/2102.09554> | **~0** | Our label distribution is strongly bimodal (a censored inactive floor plus an active mode) — exactly the imbalanced-regression setting this was built for. ⚠ **And the same paper measures the obvious alternative as *negative*: SMOTER and SMOGN *degrade* performance on high-dimensional inputs** (IMDB-WIKI MAE 8.06 → 8.14; STS-B 0.974 → 0.990). Interpolating two fingerprints yields a point corresponding to no real molecule. |
| **6** | **A portfolio at good defaults + Caruana greedy ensemble selection**, rather than one tuned model. | **Ensemble-TD −19.4%** vs LGBM-default, and it **ties 50-step-HPO on the best NN at ~1/36 the compute**. Ensemble-D already −12.0%. Meta-learned defaults alone capture **~67% of everything XGBoost HPO can ever buy** (0.029 of 0.043 AUC, Probst et al. <https://arxiv.org/abs/1802.09596>). | **~30 CPU-seconds at our n** | *"Simply trying all default algorithms is faster and very often better than (naive) single-algorithm HPO."* |
| **7** | **Joint regularisation + hyperparameter search — the "regularisation cocktail."** Kadra et al., NeurIPS 2021: jointly search **which** of **13** regularisers to apply and their hyperparameters. Add **C-Mixup** (label-similarity-weighted mixup), which works in feature space so it applies to fingerprints directly. | Regularised plain MLPs beat specialised NN architectures **and XGBoost** over **40 datasets** (26/12/2, p = 6e-4), where *"none of the specialized previous deep learning methods (TabNet, NODE, AutoGluon Tabular) outperforms GBDT significantly"* <https://arxiv.org/abs/2106.11189>. C-Mixup: **−1.9% to −14.0% RMSE** (mean ≈ −6.3%) <https://arxiv.org/abs/2210.05775>. **Warning: *vanilla* mixup makes regression 28.6% worse**; Local Mixup +27.6%, MixRL +24.6%, Manifold +5.6% — only the label-similarity version works. | **2–4 GPU-hours, and then it stops.** Their own budget curve: 2 h → **27/1/11** wins vs XGBoost; 96 h → **30/1/9**. **48× the compute buys three datasets out of forty.** | Real and worth doing — but **two corrections to an earlier draft of this file.** (i) It is not "a good way to spend a week of A100"; it saturates by 2–4 hours. (ii) ⚠ **Do not expect it to beat LightGBM on data shaped like ours.** Their own Table 2, restricted to the high-dimensional moderate-n rows: christine (5,418 × **1,637**) XGB **74.81** vs cocktail 74.26; dilbert (10,000 × **2,001**) XGB **99.11** vs 99.05; fabert (8,237 × 801) XGB **70.10** vs 69.18; jasmine XGB **80.55** vs 79.22. **XGBoost wins all four.** The cocktail's big wins are on low-dimensional or very small sets (arrhythmia 452 × 280: **+12.68**). So the realistic payoff here is **a decorrelated second opinion worth blending**, not a new best single model. Their own caveats: classification only, *"not regression"*, balanced data, and *"a **single** train-val-test split."* Per-regulariser ranking at n ≤ 6,000: cocktail +3.23 · Shake-Drop +1.96 · Mix-Up +1.33 · Cut-Mix +0.96 · Dropout +0.57 · **weight decay −0.04 · Lookahead −0.08**. Our curves diverge after epoch ~365, which *is* a regularisation deficit; Heckel & Yilmaz show fixing step sizes **improves the early-stopping point** rather than deferring it. |
| **8** | **TabPFN v2 on ≤500 well-chosen features — or Ye et al.'s 4 × 500 random feature subsets, ensembled.** | **+0.051 norm. RMSE over default CatBoost** and **+0.048 over 4-hour-tuned CatBoost** (Nature 2025 Fig. 4a); matches CatBoost's accuracy with **half the training samples**. On TDC ADMET with 217 RDKit descriptors it beat XGBoost on **19 of 22** datasets with *no tuning*. **But at d ≥ 2,000 it falls *below logistic regression*** (avg rank 3.36 vs LR 2.27, Ye et al. Table 1) — so the subsample-and-ensemble recipe is mandatory, and it is 4 subsets of 500 for us. <https://arxiv.org/abs/2502.17361> §7.1 | **Seconds per fit** | Highest upside per second *conditional on* solving the feature ceiling — but **demoted by the one benchmark built to test our exact shape.** BeyondArena (142 datasets crossing IID/temporal/**grouped** × size × **low vs high dimensionality**, <https://arxiv.org/abs/2606.30410>) reports that *"the in-context learning performance of all TFMs fails to compete with traditional models on non-IID (temporal and **grouped**), large-scale, **high-dimensional**… datasets."* **Grouped + high-dimensional is precisely our intersection.** So this is a dimensionality-reduction project with a measured risk of landing in the losing regime. For regression the authors do note *"tuning hyperparameters is more important"* (+0.045). |
| **9** | **Prediction averaging over the post-warm-up band, instead of band-min selection.** | Directly attacks the measured extreme-value bias (two seeds, same score, best epochs **335 apart**). `cyp_curse.py` already tests exactly this. | Near-free — reuses stored predictions | Converts a selection artefact into a real gain. ⚠ **But not SWA, and this corrects an earlier draft of this file.** On 40 *tabular* datasets with each regulariser applied alone and independently tuned, **SWA is the worst of 13: mean −6.72 pp, 15 W / 24 L**, with catastrophic individual failures (skin-segmentation **99.97 → 2.57**; mfeat 97.75 → 69.25; cnae 87.50 → 50.00). On the 16 datasets with n ≤ 6,000 it is **−5.12**. *Snapshot ensembles* are the opposite (**+0.93, 30 W / 5 L**). SWA's +0.27–1.36 pp on vision does not transfer. **Average predictions, not weights — and if you try weight averaging, guard it with a validation accept/reject.** Relatedly, never weight-average *independent* inits: *"interpolating weights achieves no better than random accuracy"* (<https://arxiv.org/abs/2203.05482>). |
| **10** | **Test-time retrieval fusion** (kNN + GP over labelled neighbours, fused with the net's prediction — **no retraining**). | RMSE reduced on **14 of 15** datasets, **median −19.4%**; CDK2 −42.4%, BACE −38.9% — on bioactivity regressions of **340–8,745** compounds, i.e. exactly our n. <https://arxiv.org/abs/2607.11091> | Inference-time only | Large, but **discount hard**: retrieval gains on bioactivity sets are inflated by analog-series structure under random splits (`research/23-analog-series-local-models.md`). Cuts the other way here, though — the test set *is* 75 analog series. |
| **11** | **Seed/fold ensembling, and blending the GBM with the net.** Our base spans **0.4237–0.4363 across seeds — a 0.013 range, wider than the entire claimed depth gain.** | 5-member refit ensemble: **+5.5%** (LGBM) / **+5.7%** (MLP). Deep+XGBoost blend closes **~31%** of the gap to the per-dataset best (3.34% → 2.32%, p < 0.005), and **~85% of that is in the first 3 members**, chosen *by validation loss* not at random. TabM's weight-shared ensembling: **+2.15% at ~1× cost**. | Linear and parallel | Already partly done at w(gbm) = 0.30–0.35. **Three corrections that change how to spend the ensemble budget.** (i) A *traditional* 32-member deep ensemble of MLPs buys only **+0.96% ± 1.6%** — std exceeds mean — for **32× the compute**. (ii) ⚠ **On small MLPs, seed diversity buys ~nothing and hyperparameter diversity buys several times more at identical cost**: Wenzel et al., NeurIPS 2020 (<https://arxiv.org/abs/2006.13570>) measured a 3-seed deep ensemble on an fmnist MLP at **0.889 → 0.889 (zero)**, while 3 models with the *same* init and *different hyperparameters* gained **+0.4 to +1.9 pp**; and *"provided that the initial random search produces enough models, the stratification step may be bypassed"* — i.e. **ensemble the hyperparameter search you already ran, don't re-run seeds.** (iii) **M = 5 captures 87.7% of the M = 10 gain**, and since e(M) ≈ e∞ + b/M fits almost exactly, the marginal value of the M-th member is b/(M(M−1)): **member 2 is worth 6× member 4 and 90× member 10.** Diversity of the blend pays; size does not — and *cross-family* diversity pays most (next row). |
| **12** | **~50–100 iterations of HPO per model.** | XGBoost **+0.133 norm. R²** over defaults on Grinsztajn's regression suite — but **~41% of it by iteration 10 and ~83% by iteration 100**; the last 300 iterations buy ~17%. McElfresh: *"for roughly one-third of all datasets, light hyperparameter tuning … increases performance more than choosing among GBDTs and NNs."* | ~15 CPU-minutes at our n | Worth doing, bounded, and **logarithmic** — which is why swapping random search for a good BO library saves 16–22× of *iterations* but only a *"relatively small"* amount of score. |
| **13** | **Molecular foundation-model embeddings, fine-tuned.** | **Measured null at our protocol:** SSL graph pretraining gives **−0.578% to +0.128%** with rich features + scaffold splits, inside a baseline sd of **0% ± 1.626%** (Sun et al. <https://arxiv.org/abs/2207.06010>); pretraining corpora beyond **400–800 K molecules stop helping** on ADME (<https://arxiv.org/abs/2503.03360>); **tested negatively here** (`fp-mlp` vs 0.6622). | Moderate | Superseded by row 2: use these models' embeddings **frozen**, not fine-tuned. Fine-tuning is the arm that fails. |
| **14** | **The depth × schedule interaction, as already configured** (L6 / 3,000 ep / patience 400). | **−0.0234 ± 0.0048, t = −4.92, 5/5 seeds** on validation; ~0.004 on test. Real, replicated, and **already banked** in v3deep. | ~12 h/seed; **40–60 A100-h for 5 seeds** | The one place more compute genuinely paid. But it is spent, and the curves do not support extending it. |
| **—** | **30,000 epochs at the current architecture.** | **With early stopping: measured at exactly −0.0000.** Without it: **10–17% worse by epoch 700**, monotonically worse beyond. No published study of a small molecular set trains past ~300 epochs; the one multi-GPU-week budget in the literature (Kearnes, ≥40 M steps) *"had no significant effect or damaged performance."* | 40–60 A100-hours | Dominated by every row above, including by doing nothing. |

### 7.3 Measured negatives — things to cross off the list, not try

| ruled out | measurement |
|---|---|
| **Neural architecture search** | Improvement over *randomly sampled* architectures spans **−3.44 to +2.87 points and is frequently negative** across 8 NAS methods × 5 datasets (<https://arxiv.org/abs/1912.12522>); the same paper finds *"the **training protocol** is often far more important than the architecture"* (3 p.p. vs a best-ever 0.69 p.p. over random). On tabular specifically, **TabNAS**'s two best architectures differ by **0.0002 loss against a seed std of 0.0002** (<https://arxiv.org/abs/2204.07615>), and its contribution is latency-constrained search, not accuracy. Weight-sharing supernets rank their own top-1 pick at a true rank of **~15–19 of 64** (<https://arxiv.org/abs/2001.01431>) — a filter, not a selector. Random search + ASHA matches DARTS/ENAS at equal GPU-days (<https://arxiv.org/abs/1902.07638>). |
| **Self-supervised molecular pretraining from scratch** | +0.122 / +0.126 / +0.128 / **−0.578** points against a run-to-run σ of **1.2–1.7** (Sun et al.). A **1-billion-molecule** corpus (SAVI, 500× ZINC15) gave −0.106 / +0.546. GROVER's price for its true **+3.1-point** ablation was *"**250 Nvidia V100 GPUs** … 2.5 days and 4 days"* ≈ **15,000–24,000 V100-hours**. Hu et al. report negative transfer on 2 of 8 molecular datasets, and it is architecture-dependent to the point of catastrophe: **GIN +7.2, GCN +3.4, GraphSAGE +2.0, GAT −6.5**. |
| **SMILES enumeration / test-time SMILES averaging** | **Structurally inapplicable.** ECFP/Morgan, MACCS, atom-pair and RDKit descriptors are computed from the canonicalised graph, so a randomised SMILES yields a **bit-identical feature vector**. 70× enumeration = 70 exact duplicate rows. The literature confirms it by construction: van Tilborg applied SMILES augmentation *only* to SMILES-based models; Maxsmi left its fingerprint baselines un-augmented — and on its one bioactivity task (EGFR, 5,849 molecules) the **un-augmented RF on fingerprints beat the best augmented CNN**. |
| **SMOTER / SMOGN** | Measured *negative* on high-dimensional inputs — see row 5c. |
| **Vanilla / manifold / local mixup on a regression target** | **+28.6% / +5.6% / +27.6% worse** than no augmentation. Only label-similarity-weighted C-Mixup works. |
| **Swapping LightGBM for SVM or RF** | *"**no statistical difference was found between SVM, GBM, or RF coupled with ECFPs**"* across 30 bioactivity targets of 615–3,657 molecules (van Tilborg). Expected return ≈ 0. |
| **Tuning fingerprint radius / bit width** | Deng et al. swept radius {2,3} × bits {1024,2048} across 62,820 models: *"little difference when altering the sizes."* |

**One further free lever the supplement surfaced, worth its own line:** *numerical feature
embeddings*. Gorishniy et al. (NeurIPS 2022, <https://arxiv.org/abs/2203.05556>) changed **only** the
numerical-feature embedding and moved a plain MLP from **average rank 8.5 → 3.0 over 11 datasets —
past both CatBoost (3.6) and XGBoost (4.6)** — with no change to depth, width or capacity
(California Housing RMSE 0.495 → 0.439, −11.3%). Holzmüller measures removing them as **+20.6%**
relative regression error. The linear variant *"can be fused together with the first linear layer
after training, which completely removes the overhead."* ⚠ Counter-note for our shape: on **Epsilon
(2,000 numerical features — the closest published match to our width)** Gorishniy found
*"preprocessing to be detrimental to deep models' performance, so we use the raw features on this
dataset."*

### 7.4 Two free measurements that would settle more than any training run

**(a) The residual correlation between the GBM and the net.** No published paper reports
`corr(residual_GBM, residual_NN)` on a tabular regression task — so this number does not exist in the
literature and has to be measured here. It takes one pass over stored out-of-fold predictions. For
two predictors of equal variance the optimal blend's variance is `(1+ρ)/2` of a single one, so
**ρ = 0.6 → 20% variance reduction, ρ = 0.3 → 35%.** That single number prices our entire blending
strategy, and we already have the prediction vectors on disk.

Why it matters is itself measured. Gontijo-Lopes, Dauphin & Cubuk (ICLR 2022,
<https://arxiv.org/abs/2110.12899>) held every model inside a narrow 74–78% ImageNet band so the
ensemble gain could not come from one member simply being better, and found **error inconsistency is
*linearly* correlated with the ensemble improvement**. Their ladder of what to vary:

| pair differs in… | ensemble gain over the *better* member |
|---|---|
| **seed only (REINIT)** | **≈ +1.1%** |
| hyperparameters | ≈ +0.9% |
| architecture | ≈ +1.6% |
| framework | ≈ +1.9% |
| **training data / objective (DATASET)** | **≈ +4.9%** |

Headline, from their text: against a ResNet-50 at 76.5%, the most *differently*-trained model
(ALIGN-ZeroShot, 75.5% — a **worse** model) ensembles to **83.4% top-1, "a boost of nearly 7%."**
**A LightGBM + MLP pair is a framework/dataset-class difference, not a seed-class one** — which is
the mechanism behind TabRepo's 0.524 → 0.365 and behind this programme's only confirmed win being
the blend. Related free identity: for Brier score the Jensen gap *equals* the ensemble variance
(<https://arxiv.org/abs/2202.06985> §4.3) — **measure the disagreement among members and you have
measured the ensemble gain, with no test set required.**

Saturation is also measured, and it is hard: the Netflix Grand Prize blend of **454** predictors
scored RMSE 0.8603 against ~0.8777 for its best single member (~2% relative), and **cutting it to 75
forward-selected predictors cost 0.0003 RMSE — nothing**
(<https://www.asc.ohio-state.edu/statistics/dmsl/GrandPrize2009_BPC_BellKor.pdf>). TabRepo finds
*"a portfolio of size **3** is sufficient to outperform all AutoML systems except AutoGluon, and a
portfolio of size **15** is sufficient to outperform AutoGluon"*, with **15** greedy iterations enough
and more giving *"no measurable improvement."*

**(b) The Rosenfeld fit.** §1.2b: fit `ε(m,n) = a·n^−α + b·m^−β + c` on a few deliberately small
`(m, n)` points and read off whether the data term or the model term dominates. About an hour on a
model that trains in minutes, and it answers the actual question — capacity-limited or data-limited —
on our own data.

**One more that costs no GPU at all and outranks most of the table:** the measured composition
shift. **84.3%** of test compounds are close analogs of a training active (Tanimoto > 0.4) against
**15.1%** of validation's — a **5.6×** difference — and the test set's internal similarity is
**0.665** vs validation's **0.354**. Validation is a diversity sample; test is 75 analog series.
No amount of training longer on a diversity sample fixes a model that is scored on analog series.
Building a **hit-expansion re-split** so that validation resembles test is worth more than any
compute decision in this document, and it is CPU work.

---

## 8. Bottom line

1. **The LLM/Stable-Diffusion analogy argues against the proposal, not for it.** Both were trained
   for ≈1 epoch over corpora 10⁵–10⁹× larger than ours. The transferable lesson is *unique data*,
   which is item 2 in §7.
2. **The data-constrained scaling law has a number and it is ~16 epochs** (R_D\* = 15.39), with
   returns thereafter diminishing "extremely fast" and loss eventually *rising*. 30,000 epochs is
   ~1,900× that.
3. **Grokking is the best argument for the claim, and it does not reach.** The one molecular
   demonstration (Omnigrok, QM9) needed 200 samples and a **3× inflated initialization**, and
   **does not occur at standard initialization**; Grokfast reaches the same endpoint **×50 faster**
   on the same QM9 task; and anti-grokking destroys generalization ~10× past the grok.
4. **Double descent is real here and was already harvested.** The second descent bottoms at epoch
   ~200 and is worth −0.023 validation / ~0.004 test. Nakkiran's own conclusion: the phenomena
   "often do not occur with optimal early-stopping", which we are doing.
5. **The tabular literature's levers are ensembling, retrieval, priors and light GBDT tuning** —
   not longer training; and our 2,000 mostly-uninformative features are the exact regime that
   handicaps a neural net and does not handicap a tree.
6. **The decisive fact is local: the experiment has been run.** Extending 300 → 3,000 epochs at the
   current architecture moved the score by **−0.0000, identical per endpoint to four decimal
   places**. A 10× extension of a schedule that already never completes cannot do better than that.
7. **There is no good way to spend a week of an A100 here — and that is a measured claim, not a
   shrug.** Every lever saturates fast: Kadra's regularisation-cocktail search goes from 27/40 wins
   at **2 h** to 30/40 at **96 h** (48× compute, three datasets); AMLB's own verdict on budget is
   *"Performance generally does not improve much with more time"*, and auto-sklearn got **worse**
   with 4× time on **12 of 29** datasets; HPO's gain curve is logarithmic (83% of it by iteration
   100). "A week of A100" is a good instinct pointed at the wrong axis — and the right axis is
   **width, not length**: run many seeds × folds × configs in parallel. Where GPU genuinely earns
   its place is narrow and specific: frozen-embedding extraction, TabPFN inference, and avoiding the
   CPU early-stop truncation TabArena measured (**~16%** of TabM/ModernNCA runs hit the limit on CPU
   vs **<0.1%** on GPU).
8. **But the frontier is cheaper than the question assumes.** Scaled from Holzmüller et al.'s
   measured compute/error frontier, the *entire* right-hand end for a dataset our size — a
   portfolio of 4–5 algorithms, each HPO'd 50 steps, bagged 5×, blended by greedy ensemble
   selection — is about **20 CPU-minutes**. Nothing on the measured tabular frontier needs a GPU
   at all. The defensible GPU uses here are TabPFN-family inference, extracting **frozen**
   foundation-model embeddings, and running the sweep **wide** rather than **long**.
9. **The biggest single lever is the features, not the model.** Same model, same splits, 58
   molecular tasks: **Morgan 22.4% → RDKit2d 56.9% → frozen CheMeleon 86.2%** win rate. Our
   primary block is a 2,048-bit Morgan fingerprint. Fix that before spending an hour on anything
   else.
10. **The fitted law for our regime says "plateau".** Rosenfeld et al. (ICLR 2020): *"For a given
    dataset size, scaling up the model results in an initial decrease in test error, which then
    **saturates to a level determined by the dataset size**"* — and past that, error *increases*.
    They also give the cheap test: fit `ε(m,n) = a·n^−α + b·m^−β + c` on a few small (m, n) points
    and read off whether we are capacity-limited or data-limited. **An hour of work that would
    settle this argument on our own data.**
11. **And longer training does not even buy stability.** Picard, across 10,000 seeds: *"the accuracy
    does not increase past epoch 25 … there is **no reduction in variation past convergence**."*
    Neither the mean nor the variance.
12. **A detectability warning that bounds this whole document.** RealMLP's authors note their
    benchmark resolves ~1% relative effects aggregated over ~100 datasets × multiple splits, and
    that *"on a single dataset you should expect an order of magnitude worse."* On four endpoints
    we can resolve roughly **10% relative effects** — which means **most of the 1–3% interventions
    in this literature are not detectable here at all**, and anything smaller than ~2σ of our own
    repeated-CV spread (σ ≈ 0.0065, so ~0.013) is not a result. That is the same 0.013 seed band
    that already swallowed the depth gain.
13. **Order of work:** §7 rows 0–6 first (all CPU or near-free; rows 3 and 5 are config changes),
    then rows 7 and 8. Keep row 14 — it is already configured and already paid for.
    **Do not run row "—".** And before any of it, the two free measurements in §7.4.

---

## 9. What I could not verify (flagged, per the programme's standing rule)

- **Nakkiran's second descent vs the *optimal* early-stopping minimum.** Their §8 explicitly
  disclaims the phenomena under optimal early stopping; I could not extract a clean statement on
  whether the converged error beats an optimally early-stopped one. Treated as unresolved in §3,
  not as support for either side.
- **The exact epoch counts in "Diffusion Beats Autoregressive in Data-Constrained Settings."** The
  abstract gives the direction and the mechanism but not the numbers; cited for mechanism only
  (§1.3).
- **TabPFN v2's behaviour above its 500-feature ceiling** is a stated limit, not something I
  measured; the reduction route in §7 rows 2/8 follows Ye et al.'s published recipe, not a
  published benchmark on fingerprints.
- **Several PDFs would not yield text** to the fetch tool (Muennighoff's JMLR PDF, Nakkiran's
  J. Stat. Mech. version, Stephenson & Lee's PDF, Omnigrok's PDF). Where that happened I used
  arXiv abstract pages and the ar5iv HTML renderings instead, which is why some quotes are
  attributed to `ar5iv.labs.arxiv.org` rather than the publisher PDF. Every verbatim quotation in
  this document came from a page that rendered as text.
- **Figure-read values.** Several numbers in §7.0 and §7.1 (the Holzmüller frontier coordinates,
  the Grinsztajn budget curves, the TabM relative-improvement bars) were read off rendered figures
  rather than tables — ±~0.01 on normalised scales. The published raw-results CSVs
  (<https://github.com/LeoGrin/tabular-benchmark>) would settle them exactly if a decision turns on
  a small margin.
- **An unresolved tension on TabPFN and feature count.** Ben Hicham et al. ran TabPFN successfully
  on **2,048-dim CheMeleon embeddings** (their best configuration), while Ye et al. measure
  degradation *below logistic regression* at d ≥ 2,000. The plausible reconciliation is that TabPFN
  tolerates many *dense, informative, learned* features but not many *sparse, noisy* ones — which
  is consistent with TabPFN-Morgan scoring 22.4% in the very same table. **No study tests this
  directly**, so treat it as a hypothesis, and it is exactly why row 8 is conditional.
- **Abstract-only sources.** ChemFM's "67.48%", Suiren-1.0's "state-of-the-art", the
  ExpansionRx-OpenADMET 2026 challenge summary (ChemRxiv returned 403 on the full text), and the
  BSB 2025 TabPFN/ADMET abstract were not verified against their baselines. The ExpansionRx
  abstract is the strongest pro-deep-learning ADMET datapoint found and should be read in full
  before anyone leans on or dismisses it.
- **No study was found that varies epoch count alone on a fixed small molecular dataset and plots
  the test-error curve.** The closest substitutes are the field's standard early-stopping protocols
  (§5.4 — which are themselves the field's answer) and Kearnes' negative GPU-weeks result. **This
  programme's own 3,000-epoch curves in §0 are, as far as this review found, the best such
  measurement in existence for a CYP-scale dataset.**
- **Corrections made to this document during review.** Three of my own earlier rows were wrong and
  are now marked in place: (i) I called the regularisation cocktail "a good way to spend a week of
  A100" — it saturates by 2–4 hours, and on Kadra's own high-dimensional moderate-n rows XGBoost
  beats it 4/4; (ii) I recommended SWA, which is the **worst of 13 regularisers on tabular data**
  (−6.72 pp, 15 W/24 L, with 99.97 → 2.57 failures); (iii) I ranked seed ensembling above
  hyperparameter ensembling, which Wenzel et al. measure the other way round on small MLPs (3-seed
  MLP ensemble: 0.889 → 0.889, i.e. zero).
- **A genuine disagreement left open.** van Tilborg (30 ChEMBL **bioactivity** targets) finds binary
  fingerprints beat physicochemical descriptors; Sultan (7 **ADME** endpoints) and Deng find the
  reverse. Task type is the likely discriminator and CYP inhibition sits between them, so §7 row 2
  is written as an **arm to run, not a conclusion**.
- **A second methodological caveat on the benchmark literature.** Several of the largest effects
  cited here come from benchmarks whose authors chose the datasets, and at least two say so
  (Gorishniy 2022: *"our benchmark is biased towards GBDT-friendly problems"*; Gorishniy 2021:
  *"slightly biased towards 'DL-friendly' problems"*). **Grinsztajn reaches the opposite conclusion
  to Kadra on the same question**, using 400 random-search iterations and 20,000 compute-hours. The
  findings that survive across all of them — and the only ones this review would stake a decision on
  — are the **within-model ablations**: ensembling ≫ its compute cost, preprocessing/embeddings ≫
  architecture, HPO saturates by 2–4 hours, and the split and the labels set the ceiling.
- **Figure-read and single-sourced items from the supplementary sweeps**, not to be cited as
  measured: the Gontijo-Lopes error-inconsistency category averages (scatter-plot reads; only the
  83.4% / "nearly 7%" headline is in the text); Ovadia's per-M values (figure only); Chemprop's
  per-dataset ensemble deltas (figure; the "1–5% benefit" text is verbatim); Bouthillier's
  per-source variance magnitudes (bar chart; the verbatim claims are solid); Vabalas's bias in
  percentage points (figures only); Maxsmi's RMSEs including the load-bearing EGFR/RF comparison
  (ScienceDirect 403, single-sourced); MolCLR's "+7.2% from augmentation" (absolute vs relative
  unstated); Kadra's Fig. 5 off-diagonal counts (heatmap read — Table 11 was parsed exactly and is
  the reliable source). TabArena/BeyondArena Elo values are version-dependent: cite
  arXiv 2506.16791v2 and 2606.30410v1, not the live leaderboards.
- **Genuine literature gaps — nothing to cite, so these must be measured here.** No measured table
  exists for **MAE / Huber / quantile loss vs MSE on tabular regression**; no paper ablates
  **log/pIC50/Box-Cox target transforms** and reports the delta; no published
  **GBM-vs-NN residual correlation** on a tabular regression task (hence §7.4a); no measured
  **pseudo-labelling** effect size for bioactivity regression at 1–5 K molecules; and nothing at all
  on **conformer augmentation**. Sheridan 2013 and Ramsundar 2017 are paywalled (qualitative only).
- **Transfer rate.** The ~17% validation→test transfer figure is this programme's own measurement
  from one submission pair (validation −0.0131 → board −0.0154 on a different scale, rank
  163 → 159). It is the best estimate available but it is **n = 1**, and every per-GPU-hour ranking
  in §7 inherits that uncertainty.
