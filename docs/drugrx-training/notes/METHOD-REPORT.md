# Predicting CYP inhibition: what we measured, and what we got wrong

A method report for the OpenADMET CYP blind challenge. It is organised around the
things in it that are reusable, which are not the model. Our best validation score is
**0.6615** macro soft-threshold relative absolute error (ST-RAE) where 1.0 is "as good
as predicting the mean"; the leaderboard scored an earlier version **0.8752** at rank
163 against a leader at **0.3772**. The model is unremarkable. Five other things are
worth reading.

**Provenance.** Every number here resolves to a stored artifact, a re-derivation
performed from one, or an explicit "lost" label. The audit is `PROVENANCE.md`;
`PROGRESS.md` is the day-by-day log. Leaderboard figures are **external
observations** — the only numbers in this programme our own code did not produce.
Where a figure was computed under the superseded metric it is marked **[old metric]**
and is not comparable to anything unmarked.

---

## 1. The metric denominator was wrong, and it cost a factor of two

**This is the most expensive mistake in the programme and the easiest to repeat.**

The challenge scores a soft-threshold relative absolute error. The numerator is the
distance from the prediction to the nearer bound of a per-measurement credible
interval, zero inside it:

```python
err = sum(relu(lo - y_pred) + relu(y_pred - hi))
```

We implemented the denominator as `sum(|y - mean(y)|)` — the mean absolute deviation
of the labels. The organisers' actual denominator puts the **naive mean predictor
through the same soft-thresholding**:

```python
# OpenADMET/blind-challenge-template, backend/custom_scoring_functions.py
base = sum(relu(mean(y) - hi) + relu(lo - mean(y)))
```

So the baseline also scores zero wherever *it* lands inside an interval. The true
denominator is far smaller and the true score far larger.

**The single most useful fact about this metric, which our version could not
express: ST-RAE = 1.0 means "exactly as good as predicting the mean."**

### What it cost

| | our value **[old metric]** | truth |
|---|---|---|
| validation of the submitted model | 0.4346 (`runs/cyp/report.json`) | **0.737** |
| leaderboard, same model | — | **0.8752** (external) |
| train-mean sanity check | 0.6134 | **1.0106** |

We quoted ~0.44 internally for an entire phase while the leaderboard scored the same
model 0.8752. About **60% of the apparent gap was mis-measurement and 40% a real
generalisation gap**.

**The correction is not uniform per endpoint** — ratios 0.54 to 0.67 — which is the
part that propagates. Every macro-level choice made before the fix (blend weights,
endpoint trade-offs, the per-epoch best-epoch selection) optimised a **mis-weighted**
objective, and because the distortion varies per endpoint those choices cannot be
rescaled after the fact. They have to be re-fitted. Our current submission re-fits
them; nine stored result files predating the fix are quarantined in `PROVENANCE.md`.

### Why it survived so long

Our own docstring had flagged the denominator as a guess — *"not guaranteed to match
the leaderboard's"* — and we ran the whole first phase on it anyway. A score of 0.44
against a leader at 0.3772 looked like a competitive model. **The flag was in the
code, in writing, and it did not stop anything.** A caveat inside a measurement path
is a blocker, not a licence to proceed.

### The fix, pinned

`tests/test_official_metric.py` pins our implementation to a **verbatim transcription**
of the organisers' function across 40 random cases at 1e-12. It also asserts:

- the mean predictor scores **exactly 1.0**;
- the old denominator is **always optimistic** — provable, since `y ∈ [lo, hi]` implies
  `dist(mean, [lo, hi]) ≤ |mean − y|`;
- a genuine 0/0 exists in the official metric when the mean predictor is itself perfect.

The superseded function is retained as `st_rae_mad` **only** so old log entries stay
reproducible, never for scoring. Every run artifact now carries a `metric_id`; see §5
for why that was not enough on its own.

### Transferable

1. **Transcribe the scorer, do not reimplement it.** The organisers' code was public
   the whole time.
2. **Score the naive baseline first.** `train_mean → 1.0106` would have caught this on
   day one, in one line.
3. A relative metric whose denominator you guessed is not a relative metric.

---

## 2. Our predictions are under-dispersed, and that is correct

Our predictions have roughly **half the label spread** (sd 0.478 against 0.944) and the
leaderboard reported **R² = 0.045** with Spearman 0.62. That looks exactly like a
calibration bug. We tried to fix it. **It is not a bug, and fixing it is actively
harmful.**

`cyp_recalibrate.py`, 5 paired repeats, corrected metric, on `p' = m + α(p − m) + β`:

| arm | official | pred sd | vs raw |
|---|---|---|---|
| raw | 0.7496 | 0.478 | — |
| **shift** (β only) | **0.7284** | 0.478 | **+0.0211 ± 0.0015, 5/5, t=+14.3** |
| scale (α only) | 0.7528 | 0.460 | −0.0032 ± 0.0023 — inside noise |
| affine (α, β) | 0.7297 | 0.435 | +0.0199 — no better than shift |
| **match_sd** (α forced, 0 free params) | **1.0142** | 0.949 | **−0.2646 ± 0.0216, 0/5** |

**Forcing the prediction spread to match the labels scores 1.0142 — as bad as
predicting the mean.** And the *fitted* scales come out mostly **below** 1
(0.75–1.15): the optimiser wants *more* shrinkage, not less.

**The zero-free-parameter `match_sd` control is what settled it.** Without it,
"scale is inside noise" would have read as a tuning failure rather than a refutation.
A control with no free parameters cannot be accused of being badly tuned.

### Why shrinkage is right here

The loss charges **nothing** inside the credible interval. A predictor that hedges
toward the region where intervals are dense pays no penalty for being imprecise there.
Under a loss with a free zone, shrinkage is the optimal response, not a symptom.

So **the leaderboard's R², MAE, ρ and τ columns are traps**: rank is MA-ST-RAE alone
(verified monotonic), and optimising the diagnostic columns would lower it. Only the
additive per-endpoint shift helps.

### The one place theory paid

ST-RAE's risk is convex piecewise-linear per molecule, so the Bayes-optimal point
estimate is the **weighted median of the pooled `lo`/`hi` draws** (`P(lo>a) = P(hi<a)`)
— not the conditional mean and not the mode. For Gaussian bounds there is a closed
form, verified against a 4M-sample Monte-Carlo argmin (0.1609 vs 0.1610):

```
a* = (σ_hi·μ_lo + σ_lo·μ_hi) / (σ_lo + σ_hi)
```

Each bound weighted by *the other's* standard deviation. Because the metric
**decomposes per molecule**, a joint `p(y|x)` has **zero decision-theoretic value** —
which is why diffusion, flow matching and CARD were dropped on paper rather than tried
(§4).

And the richer version of that idea is a measured null: a monotone per-bin decode map
beats the one-parameter offset by nothing (4 of 8 bins inside noise) and at 16 bins is
significantly **worse** (−0.0197 free, −0.0061 isotonic). An offline study had
predicted −0.010 to −0.030; it did not replicate, because it had been fitted against a
*simulated over-dispersed* predictor — a predictor unlike ours in exactly the property
under test.

---

## 3. Eight hypotheses, each demoted for a measured reason

Effects are quoted with the spread that bounds them. "Inside noise" means the sample
cannot resolve the effect, **not** that the effect is zero — and the seeds-to-resolve
column is what makes that distinction actionable.

**The two metric families must not be mixed.** Hypotheses 1, 2, 5 and 6 are the
**21-endpoint TDC ADMET** work, scored per endpoint in TDC's own metrics (PR-AUC,
Spearman, MAE) on TDC's official splits. The ST-RAE denominator bug **does not touch
them**. Hypotheses 3, 4, 7, 8 and 9 are the **CYP challenge** under ST-RAE, and only
8 and 9 were measured after the fix.

| # | hypothesis | family | result | seeds to resolve |
|---|---|---|---|---|
| 1 | richer features help | ADMET | **moved it** — 18/21 endpoints | — |
| 2 | weight-tied recursion at fixed budget | ADMET | null, gap 0.0017 | — |
| 5 | richer features again (Avalon + ErG) | ADMET | **moved it** — 14/21 | — |
| 6 | neural net beats gradient boosting | ADMET | **tie** — 12/21, sign test p = 0.180 | — |
| 3 | deeper network | CYP | re-measured and **still unresolved**: −0.0085 ± 0.0070, t = −1.22, 4/5. (A re-scoring of stored predictions suggested −0.0225 at t = −4.45; it did not replicate — see below.) | ~75 |
| 4 | wider network | CYP | **re-measurement void** — every wide seed OOM'd to batch 2048 against base's 4096, so the comparison conflates capacity with batch size. Matched re-run pending. | — |
| 7 | richer decoding than a scalar offset | CYP | null at 8 bins; **worse** at 16 (−0.0197) **[old metric, re-measurement owed]** | — |
| 8 | the interval loss beats a point loss | CYP | **+0.0078 ± 0.0054, t = 1.44**, 6/10 — unresolved | ~30 |
| 9 | auxiliary heads help (Veith / ChEMBL) | CYP | all inside noise vs a true control | ~57 (macro) |

### The central finding survives a scare, and the scare is instructive

The programme's headline is: **this task is bound by data and representation, not by model
class or compute.**

It briefly looked refuted. Re-scoring the capacity runs from stored predictions under the
corrected metric turned depth and width from nulls into decisive effects (−0.0225 and
−0.0195, 5/5 seeds each) — because the original test had differenced two
differently-proportioned **metric mixtures**: `long-base` held two official-scale scores
and three old-scale ones, `long-deep` three and two, every file with `metric_id: None`.
That diagnosis was correct and the original test was indeed invalid.

**But the replacement did not replicate.** A clean end-to-end re-run, every epoch judged
by the corrected metric, puts depth at **−0.0085 ± 0.0070, t = −1.22** — inside noise. The
re-scored predictions came from 3000-epoch runs with best epochs reaching 310; the clean
400-epoch run finds 15–40. **The epoch-selection confound was the entire effect.** Width
could not be measured at all: every wide seed OOM'd to batch 2048 against base's 4096.

So: **the central claim stands**, and what survives is unchanged — **better features**
(both ADMET feature hypotheses, on a metric the bug never touched) and the **per-endpoint
additive offset** (+0.0211 ± 0.0015 in-sample; ≈ −0.004 honestly cross-fitted, §2).

**The instructive part is the sequence.** An invalid measurement was correctly identified,
its replacement looked decisive at t = −4.45 on 5 of 5 seeds, and it was still wrong,
because it carried a confound that had been named in writing at the time it was reported.
A t-statistic cannot see a confound. The only thing that caught it was running the clean
experiment anyway — and the only reason that was possible is that the result had been
labelled "strong evidence, not a result" rather than published.

### Two methodological points these produced

**"Best epoch" is largely an extreme-value artifact.** At 3,000 epochs the score
oscillates in a band of mean 0.4857, sd 0.0095 **[old metric]**. Two seeds scored
0.4441 and 0.4448 with best epochs **360 and 25**. Any score quoted as a
minimum-over-epochs *on the split used to select it* — including a challenge
submission — is optimistically biased.

**Underpowered estimates inflate, and we have the pair to prove it.** Hypothesis 8 was
first run at 4 seeds: −0.0121 ± 0.0052, t = 2.32, winning 4/4. At 10 seeds the same
comparison gives **+0.0078 ± 0.0054, t = 1.44, 6/10**. The effect fell by a third and
the t-statistic by 40% purely from adding seeds. **The 4-seed figure is retired and
should not be cited.** Its direction was right; its magnitude was an artifact of a
sample too small to resolve its own effect.

### The rule that follows

**Divide the expected effect by the noise band before provisioning a GPU.** We once
verified a real mechanism — the training loss Huber-smooths a pure-L1 metric, and
83–94% of molecules sit in the quadratic zone — and launched a 16-run sweep before
checking its size. Measured cost: **0.0003** against a band sd of **0.0095**. The
mechanism was real and the effect was 3% of the noise.

---

## 4. Six methods declined on the literature, before any compute

Each was costed against this task's measurement resolution (~0.009 macro band sd) and
its data scale (6,525 primary labels, 4,905 compounds). Research notes in
`research/10`–`13`.

| method | why not |
|---|---|
| **JEPA / self-supervised molecular pretraining** | The most relevant study — 25 pretrained models × 25 datasets, frozen embeddings, scaffold splits, **five of them CYP panels** — finds **1 of 25** beats a count-ECFP fingerprint, and that one is itself built on fingerprints. Four independent audits 2021–2026 agree. What *does* work (Mol-JEPA: 14 modalities, 4.66M molecules) gains from **more kinds of supervision**, not self-supervision on structure — and its own analysis ranks ECFP and MOE descriptors among its most important modalities, while losing on win-rate to a tabular in-context model fed ordinary descriptors. |
| **Sparse MoE (DeepSeek-style)** | Buys more *total* parameters at fixed *active* FLOPs — valuable when compute-bound on trillions of tokens. We are label-bound on 6,525 labels and the dense model trains in minutes. Nothing for sparsity to buy. Sparse top-k dispatch is also incompatible with `vmap`. |
| **Multi-gate MoE (MMoE / PLE)** | The right family to consider, and still declined: PLE's own gain is **+0.0027 AUC on a billion samples**; MMoE is negative on 2 of 3 benchmarks. At matched parameters a dense-gate MMoE layer is algebraically the dense layer plus one input-dependent scalar per hidden block — below our resolution. And the mechanism it fixes (task conflict, high across-seed variance) is **the one a 16–32 member deep ensemble already fixes**. |
| **TRM / HRM / adaptive depth** | ARC Prize's controlled ablation: a plain transformer of the same size lands **within ~5pp** of HRM, and 1→2 outer loops is worth **~13pp** — "not an effect of the HRM architecture". **Zero** arXiv results for recursive/adaptive-depth ∧ molecular-property/QSAR/ADMET. The one tabular result (Tab-TRM, 9 features, 680k rows against our 2,500 features and 6.5k rows) has a margin of 0.33% relative ≈ **0.0014 ST-RAE**, two orders below detectability. Looped transformers: **−0.39% ± 0.80%** over 27 tabular datasets. |
| **Diffusion / flow matching / CARD** | ST-RAE decomposes per molecule, so a joint predictive distribution has **zero decision-theoretic value** here (§2). The optimal decode has a closed form costing `O(S log S)`; CARD costs 1000 steps × 1000 samples for a quantity we can compute exactly. |
| **Pretraining a backbone ourselves** | MoLFormer-XL ≈ 3,300 V100-hours; GROVER ≈ 15,000–24,000. Three orders of magnitude beyond a 24-hour budget. |

**Still open, and the only two we would spend on:** in-context tabular models (TabPFN
v2 / TabICL — our 4,905 compounds fit the ≤10k envelope, and TabICLv2 out-win-rated
Mol-JEPA) and CLAMP, the one model in that 25-model audit that beat fingerprints.

---

## 5. The measurement layer, which is where the real defects lived

Most reports omit this section. It is the one that makes the rest credible, because
every number above passed through code that has been wrong before.

**Seven defects have lived in this programme's interpretation layer**, not in its
models. A representative set, each found by a different mechanism:

| defect | how it would have surfaced | how it was found |
|---|---|---|
| metric denominator (§1) | never — the score looked competitive | comparing our value to the leaderboard's |
| `.gitignore`'s bare `data/` hid `drugrx/data/` — including `splits.py`, **the scaffold split behind every headline number** | never; **no checkout could reproduce any result** | trying to `git checkout` a file |
| a column filter that silently dropped **6 usable auxiliary heads carrying 9,223 labels** while producing a head count that matched the docstring | never — the count looked right | re-deriving the count from the files |
| the mirror row-count guard, handed `n_rows=0`, compared every panel against 0 and **could never fire** | it printed `expected 0`, which reads as noise | reading a warning instead of ignoring it |
| a metric-provenance guard installed in 4 drivers by hand, **broken in 2** — one file did not parse, one would `NameError` only on the cached path a *resumed* run takes | the guard would fail exactly when needed | compiling all four |
| results lost because the per-seed upload existed in **1 of 3** drivers | two whole experiments vanished with a VM | the crash |
| the submission builder freed GPU memory with `del`+`empty_cache()` — the remedy the trainer documents as **tried three times and failed** | OOM at seed 1 of 3 | the OOM; then measurement showed one seed peaks at **37.1 of 39.5 GB**, so two could never fit |

### The pattern

**Six of the seven are a failure that presents as quiet success.** A vacuous guard
printing `expected 0`. A monitor whose glob aborted so its counter never changed. A
poller whose directory listing timed out and returned an empty list, so it pulled
nothing for thirty minutes while looking alive. A `DONE` marker echoed after the step
that failed, because the script lacked `set -e`. **We wrote that lesson down three
times in one day and implemented it three more times.** Writing a rule does not
install it; only a mechanism does.

**And the asymmetry is the tell.** When the same change is needed in four places and
made four times by hand, two of them will be wrong — and the two that are wrong will be
the ones nothing exercises. Every one of those became a shared callable or a test:
`drugrx/cyp/publish.py` (one push, four drivers), `tests/test_benchmark_group.py` (pins
the 21-endpoint group, after a fix of ours nearly added a 22nd and silently moved the
denominator of every "/21" macro and every checkpoint's head layout).

### The closing argument: every defect was in code that prepares, interprets or ships

Look at where the seven lived. A metric. A `.gitignore`. A column filter. A row-count
guard. A provenance stamp installed by hand in four drivers. A results upload present
in one driver of three. A submission assembler's memory handling.

**Not one of them was in code that computes.** The training loop, the `vmap`ped
ensemble, the scaffold split, the featurisation and the optimiser were reliable
throughout — thousands of runs with no defect traced to them. What repeatedly failed
was the layer that decides **what the compute means**: what is measured, which
artifacts are comparable, what gets kept, and what gets shipped.

The last one is the sharpest instance. `cyp_final.py` — the one script whose output
reaches the leaderboard — freed GPU memory with `del` + `empty_cache()`, the exact
remedy that `cyp_net_one.py`'s docstring records as having been tried three times and
failed, with the reason explained. **The fix existed, in the same repository, with its
rationale written down, and the shipping path did not have it.** Knowledge recorded in
one module does not propagate to another; only shared code does.

**The corollary is actionable, and it is not "document it better".** The fix for the
shipping-path OOM was never more documentation — the rationale was already written, in
the right place, in the module that had solved it. The fix is that `cyp_final.py` should
have **imported the trainer's process isolation instead of reimplementing it**.
Documentation is what you write when you cannot share the code, and it fails in exactly
the way this programme demonstrates: read once, not at the moment of reimplementation.
Every one of the seven was closed by a shared callable or a test, never by a better
comment.

**This is a more transferable claim than any of the nine demoted hypotheses.** Those
tell you about CYP inhibition under one metric. This tells you where to look: a
programme that measures carefully and computes carelessly will notice. A programme that
computes carefully and measures carelessly will produce confident numbers and not
notice, because the failures present as quiet success. Our compute budget was never the
binding constraint; our interpretation layer was, and it stayed invisible until a
leaderboard disagreed with it by a factor of two.

### What this implies for the numbers above

We assumed, when this section was first drafted, that re-measuring the old-metric
hypotheses was housekeeping — that "a non-uniform rescaling of a null is very probably
still a null". **That assumption was wrong, and it was wrong about the two hypotheses we
checked first.** Depth and width both reverse from null to decisive on re-scoring (§3).

The lesson generalises beyond this programme: **a measurement error does not merely add
noise to the conclusions drawn under it — it can invert them.** The old denominator
compressed the score and distorted it unevenly across endpoints, which was enough to
hide two of the largest effects available. And the specific mechanism was pooling:
`long-base` and `long-deep` each held a *mixture* of old- and official-metric scores, in
different proportions, so the paired test differenced two incomparable quantities. That
is why `METRIC_ID` exists, and why the drivers now **refuse** to pool across a metric
change instead of doing it silently.

**Still owed re-measurement under the corrected metric:** hypothesis 7 (the decode map)
and the Veith auxiliary result. Those are the last CYP conclusions resting on the broken
denominator, and on the evidence of depth and width they cannot be assumed to survive.

---

## 6. What we would tell someone starting this challenge

1. **Transcribe the scorer and score the mean predictor.** One line, day one.
2. **Expect to be beaten by a fingerprint baseline**, and make the deep model earn it
   on a scaffold split.
3. **Do not fight the shrinkage.** The free zone inside the intervals makes
   under-dispersion correct; the R² column is a trap.
4. **Fit a per-endpoint additive offset.** It is the one cheap, real, repeatable gain
   (+0.0211 ± 0.0015) and it takes minutes.
5. **Average several seeds.** −0.0168 on our submission, free, no label needed, and the
   gain tracks single-model variance exactly as ensembling theory says.
6. **Budget seeds by dividing the expected effect by the noise band** before touching a
   GPU. At a band sd of ~0.012, an effect of 0.003 needs ~57 seeds per arm; decide
   whether the answer is worth that *before* you start.
7. **Before reporting an oracle ceiling as headroom, compute its same-configuration
   null.** Ours was entirely a decorrelation artifact — see
   `oracle-ceilings-measure-decorrelation.md`, which is the separable result from this
   work and the one most likely to save someone a month.

---

## 7. What we do not know

- **A validation-to-leaderboard gap that is permanently unrecoverable, not merely
  unmeasured.** Our one observed gap (+0.138) came from a model whose validation was
  computed under the broken denominator. Since the correction is non-uniform per
  endpoint (0.54–0.67), that gap is a mis-weighted average of per-endpoint gaps, and
  decomposing it needs per-endpoint leaderboard scores. The board reports **macro
  columns only** (MA-ST-RAE, MA-MAE, MA-R², Spearman, Kendall). There is no later
  analysis that recovers this: the information was never emitted. Any expectation built
  on that gap is order-of-magnitude.

  **One request to the organisers, since this is a finding about the challenge's
  reporting rather than about our models:** per-endpoint leaderboard columns would let
  participants decompose their own generalisation gaps against their own per-endpoint
  validation scores, which every participant already has. Their absence cost this
  programme a calibrated prediction it had otherwise built the apparatus to make — the
  metric is transcribed and pinned, the validation is per-endpoint, and the only missing
  piece is on the scoring side.
- ~~Whether the current submission's validation gain transfers.~~ **Answered, and badly.**
  Leaderboard **0.8598, rank 159**: validation improved 0.0886 and the leaderboard improved
  0.0154, so **17% transferred** and the gap widened from +0.1382 to +0.2114. See §8.
- **Whether hypotheses 1–7 survive the corrected metric.** They have not been
  re-measured (§5).
- **Whether the ensembling gain saturates.** Measured to k=4; marginal returns had
  mostly collapsed, but k=4 is one combination per arm with no error bar.
- **An unexplained 0.013.** Two independent reruns of the loss ablation reproduce each
  other to ~1e-4 but sit 0.013 from the original run. Mirror drift, the `metric_id`
  code change and the external tables are all excluded; the original's configuration is
  unrecoverable. Those seeds are therefore never pooled with the new ones.


---

## 8. The generalisation gap, and why validation could not have predicted it

Added after the leaderboard scored the submission. It is the most useful section for anyone
else working this benchmark, and the most uncomfortable.

**Validation improved 0.0886; the leaderboard improved 0.0154.** Transfer was **17%**, and the
gap *widened* from +0.1382 to +0.2114.

### The evaluation set is a different kind of set, and the challenge says so

> *"Analog Expansion Test Set: Hit expansion was performed on the top 25 hits from CYP1A2,
> CYP2C9, and CYP3A4 by purchasing the top analogs."*

Training is a diversity library plus FDA-approved compounds. Measured by Morgan Tanimoto:

| | NN to train | NN to train **actives** | within itself |
|---|---|---|---|
| our validation | 0.455 | 0.329 (15.1% >0.4) | 0.354 (23.9% >0.4) |
| **test** | **0.562** | **0.536 (84.3% >0.4)** | **0.665 (98.8% >0.4)** |

**A scaffold split of a diversity library is built to produce dissimilar compounds. The test
set is analog series.** We spent the programme measuring novel-chemistry performance while
being scored on analog expansion.

And the bound is hard: an analog-expansion split of our own data reaches within-set 0.446
against the target 0.665, because the organisers **purchased** analogs we do not hold at any
density. **No re-splitting or re-weighting of our data can emulate the evaluation set.**

### The metric was helping, so the deficit is larger than the gap

ST-RAE divides by a constant predictor's error on the evaluation set, so it depends on that
set's label spread. Resampling validation to varying spread with the model fixed:

| sd(y) | 0.470 | 0.721 | **0.964** | 1.210 | **1.296** |
|---|---|---|---|---|---|
| ST-RAE | 1.0614 | 0.7604 | **0.6602** | 0.6063 | **0.5673** |

The ratio **falls** as labels spread. Test label sd, backed out of the organisers' own MAE and
R² columns, is **≈1.29** against validation's 0.944 — so the metric should have made test
**0.099 easier**. **True deficit ≈ 0.31, about 47% larger than the raw gap.**

That the test labels are *wider* while the structures are *tighter* is not a contradiction:
an analog expansion around potent hits is **bimodal** — analogs that keep activity sit high,
those that lose it sit low. That is the definition of an activity cliff. Measured on our own
data: among compounds with a neighbour at Tanimoto ≥0.4, **46.2% differ by more than 1 log
unit in potency and 17.3% by more than 2.** The leaderboard's own diagnostics agree —
**Spearman 0.64 with R² 0.045** is ranking roughly right and getting magnitudes badly wrong.

### The epistemic failure, which is the transferable part

We ran three experiments to explain the gap — offsets, activity cliffs, metric geometry — and
reported all three as refuted. **All three were run on validation, the set we had just proved
contains no test-like subset.** A set that cannot reproduce analog-series structure cannot
rule out mechanisms that only appear in it. The denominator experiment is the clearest case:
it found the denominator flat across similarity bands and concluded the metric was honest,
when a flat denominator means the bands never differed in label concentration — **the
experiment did not vary the quantity the mechanism runs on.**

**A negative result is only as strong as the range of the variable it was tested over**, and
ours were tested over a range the evaluation set lies outside. Three of the nine demotions in
§3 were produced the same way and are owed the same scepticism.

**What we would do differently:** check that the validation set resembles the evaluation set
*before* tuning anything on it — a five-minute fingerprint comparison that would have
reframed the whole programme, and that we ran only after the leaderboard forced it.