# 12 — A per-row pseudo-label is licensed by the readout and gated out on the features, and the thing to vary is the LABEL

**Status: Phase 8a closed; 8b and 8c running.** `scripts/cyp_weight_predictability.py`, raw numbers
in `data/cyp_cache/weight_predictability.json`. Pre-registered in the script's docstring before the
run, committed in `e55f84f`.

## The hypothesis, and why it was not priced away

Arnab, 2026-10-11: *stop using one scalar weight for every pseudo-labelled row; train a weight
**matrix**, a bank of such weightings, and combine them at inference, with dynamic learning rates
on L4s and A100s.*

`findings/11` had just closed the scalar-weight family at a measured **+0.0037 macro Δρ** ceiling.
That ceiling bounds **scalar** weightings only — one free parameter against 24,580 cells
(6145 × 4) — so it does not price this out. And there is a specific signal the pipeline
**discards**:

> `screen_labels` thresholds the continuous readout `r = 2^log2fc` at `INACTIVE_R = 0.9`.

A compound at r = 0.99 (no inhibition whatsoever) and one at r = 0.90 (barely over the line) get
the **same** pseudo-label 2.2111 and the **same** weight 1.0. The within-bin position is thrown
away.

**And it is testable without GPU**, because the 303 dual-labelled compounds — carrying both a
dose-response curve and a no-inhibition readout — are a **labelled sample of the pseudo-label
error itself**.

## Q1 — does the error depend on the continuous readout? PASS

| | |
|---|---|
| Spearman(r, \|error\|) | **−0.1869**, p = 0.0011 |
| Spearman(r, **signed** error) | **−0.3771**, p < 0.0001 |

| tercile of r | r range | **mean true y\*** | sd y\* | mean \|err\| |
|---|---|---|---|---|
| bottom | 0.9000–0.9380 | **2.5901** | 0.4834 | 0.4542 |
| middle | 0.9405–1.0096 | 2.3029 | 0.3601 | 0.2564 |
| top | 1.0103–6.5095 | **2.2535** | 0.4047 | 0.2644 |

**A 0.34 spread in the true mean across terciles — 77% of the floor population's own sd — and the
pipeline assigns all three the single constant 2.2111.** The signed correlation is twice the
absolute one, which is the substantive point: **this is a location effect, not a variance effect.
The thing to vary per row is the LABEL, not the weight.**

**And it holds within isoform**, so it is not a between-isoform artefact:

| isoform | n | Spearman(r, true y\*) | p |
|---|---|---|---|
| CYP1A2 | 42 | **−0.6140** | < 0.0001 |
| CYP2C9 | 11 | −0.5000 | 0.117 (underpowered) |
| CYP3A4 | 250 | **−0.2342** | 0.0002 |

Calibrations fitted on the pooled sample: log-linear `y* = 2.3886 − 1.3261·log10(r)`, **R² 0.0526**;
isotonic decreasing, R² 0.1773 in sample. **4683 rows would be relabelled**, mean shift **+0.0966**,
sd **0.1118**, range −1.635 to +0.238.

**The sharpest single number: the constant 2.2111 scores R² = −0.1492 against the floor
population.** The value in production is **worse than the trivial mean** of the distribution it is
meant to summarise — which is the same defect `findings/11` found from the other direction (median
under a squared-error loss).

## Q2 — is the error predictable from the 3091 features, out of fold? FAIL

Grouped by compound, so the same molecule cannot straddle a fold (a compound contributes up to
4 cells and ignoring that would leak).

| | |
|---|---|
| **oracle** (in-fold) Spearman | **+0.9949** |
| **cross-fitted** Spearman (pooled) | **+0.1274** |
| cross-fitted R² | **−0.0477** — worse than predicting the mean |
| per fold | +0.133, +0.056, +0.249, +0.121, +0.127 |
| folds at the pre-registered +0.20 bar | **1 / 5** |
| **oracle-to-cross-fitted gap** | **+0.8676** |

**A learned weight function of molecular structure would be fitting noise, with a +0.99 in-sample
story to show for it.** That is precisely the pattern that closed the decode family in
findings 04–05, and reporting oracle and cross-fitted together is what makes it visible.

## The design consequence, which is the whole value of the gate

**The per-row object is licensed as a function of `r` (1-D) and gated out as a function of `X`
(3091-D).** So the learned arm's weight-net reads only

    u_i = [ log10(r_i), one-hot(kind_i), CI width_i, detached per-row loss_i ]

and never the features. Without the gate the natural build — a weight-net on the fingerprint —
would have produced a convincing and worthless result.

## The nuisance, named in advance (`AGENTS.md` trap 8)

The dual-labelled sample is **250 CYP3A4, 42 CYP1A2, 11 CYP2C9, 0 CYP2D6.** CYP2D6 carries
**60.5%** of the pseudo-label mass and has **zero** compounds here, because its promotion rule
(`DRC iff log2FC < −0.6`) makes "has a curve" and "is screen-inactive" disjoint by construction. So
`g(r)` is fitted on a pooled sample dominated by CYP3A4 (1.9% mass) and applied to CYP2D6 across a
selection boundary. **A PASS on Q1 is permission to build, not evidence the lever reaches CYP2D6**,
and the per-isoform table is the thing to read in 8b/8c.

## What is being built on it

- **8b** (`cyp_rowwise_labels.py`, + `cyp_rowwise_shard.py` for parallelism): `base`,
  `meanfloor`, `rowlabel`, `rowiso`, `rowboth`, and **`bank`** — Arnab's bank done properly, where
  each ensemble member gets its own calibration from a bootstrap of the in-split dual sample, so
  the ensemble averages over **calibration uncertainty**. Unlike the draw-based arm of
  `findings/11`, this one has a downstream consumer: the point estimate itself.
  **`g(r)` is cross-fitted per split** — the 303 carry real DR labels and some land in validation,
  so fitting on all of them would leak.
- **8c** (`cyp_learned_weights_gpu.py`): learned `w_i = 2·sigmoid(h_φ(u_i))` and learned label
  `a + b·log10(r_i)`, with (φ, a, b) trained by **bilevel lookahead** against a **meta** split
  carved out of *train*, so validation is never tuned on. Dynamic learning rates on two
  timescales: OneCycle on the backbone, cosine on φ after a uniform-weight burn-in.

**The hypothesis for why learning could still beat deriving**, stated so the arm has a reason to
exist: the derived weight knows about `r` but not **which** rows are mislabelled, and `findings/11`
measured that **49.8% of screen-inactive compounds truly sit above 2.211**. Those rows carry high
loss, and a loss-dependent weight can discover them and stand down — an automatic robust loss the
derived form cannot express.

**Honest expectation, on record:** `derived` beats `uniform`; `learned` matches or loses to
`derived`; `bank` is the arm with the cleanest variance argument; and the macro +0.005 bar is
probably still missed, as it has been by every member of this family.

**Early, one seed, not a result:** on the differentiable backbone `derived` beat `uniform` by
**+0.0050 macro ρ** (0.5576 vs 0.5526), winning on 3 of 4 isoforms.

## Four bugs in my own code, all caught before they became results

Recorded because three of the four would have produced a *publishable-looking* number:

1. **Zero meta-gradient.** The first meta step reduced to `mloss.backward()`, and the meta loss has
   no path to φ. The `learned` arm would have been **silently identical to `uniform`** and reported
   a meaningless tie.
2. **Unbounded meta objective.** Maximising ⟨G, g_meta⟩ outright lets the weight-net inflate
   forever; the first smoke run returned NaN weights and macro ρ **−0.40**. Replaced with a bounded
   **cosine alignment**.
3. **No minibatching.** One full-batch step per epoch meant "30 epochs" = **30 gradient steps**,
   and the MLP looked catastrophically worse than the GBM (CYP3A4 0.290 vs 0.765). The
   pre-registered backbone-competitiveness check (item 4) fired exactly as written — **on my own
   bug**, which is why it is not reported as a finding about model class.
4. **`np.clip(NaN, 0.9, None)` is NaN.** A row with no screen readout gives `logr = NaN`. The
   forward pass survives — `torch.where` never selects it — but the **backward** for the
   label-calibration parameter sums **0 × NaN** over the unselected branch. Only the `learned` arm
   broke, because it is the only arm that differentiates through the label. Fixed at the readout,
   not at the gradient.

And one environment defect: **the Colab VM shipped lightgbm 4.6.0 against 4.7.0 locally.** Paired
within-seed Δρ would have survived that (all arms for a seed run on one machine); pooled absolute
ρ would not. Pinned the VM rather than reasoning around it, and `cyp_rowwise_shard.py --selftest`
now asserts the learner's 11 parameters against the production literal so a future edit to one and
not the other fails loudly.

---

## ADDENDUM 2026-10-12 — adversarial verification. **B7's SIGN is broken**, two headlines are tautologies, and a live production defect was found

A verifier re-derived everything from the release without reading these findings. **A1, A2, A3, A5,
B1, B2, C1 and C2 confirm exactly** — and B5 *strengthens* (R² **−0.1072** not −0.0477; **0/5**
folds at bar, not 1/5). Five things of mine do not survive.

### 1. B7's direction is a choice of loss I never named — and four of five defensible rules go the other way

`findings/11` establishes that the floor label is a **median**. `findings/12` then "improves" it by
fitting **OLS** — a conditional-**mean** estimator — and reports **+0.0966** as the correction.
Refit the identical log-linear model under other defensible losses:

| rule | mean shift | **CYP2D6 shift** (56.5% of the relabelled mass) | OOF MSE | OOF MAE |
|---|---|---|---|---|
| shipped constant | 0 | 0 | 0.22688 | 0.32632 |
| **B7's log-linear OLS** | **+0.0966** | **+0.0299** | 0.19259 | **0.34491 (worse)** |
| log-linear **median (L1)** | **−0.0665** | **−0.1252** | 0.21792 | 0.32056 |
| isotonic | +0.0108 | **−0.1165** | **0.17798** | 0.32042 |
| two-value split at r = 1.0 | **−0.0320** | **−0.0723** | 0.21694 | 0.32130 |
| empirical local median | **−0.0725** | **−0.1765** | — | — |

**Four of five move the label down. B7's OLS is the only one that moves it up, and the only one
worse than doing nothing under L1.** It also extrapolates unguarded — one CYP1A2 row at r = 23.27
is relabelled to pIC50 **0.576**, which is nonsense.

**And the loss matters concretely for us.** v9 is **0.30 × XGBoost (`reg:squarederror`, L2 on the
point) + 0.70 × net (interval hinge that reads only `lo`/`hi` and is exactly zero inside the
2.46-wide floor window)**. So for **70% of the prediction the 2.2111 point is never read at all**;
for the 30% L2 arm, the optimal point target for a coarsened observation genuinely *is* the
conditional **mean**, 2.3822. That makes an upward move defensible — **but for the opposite reason
to the one B7 gives**, and `allr_floor` simply returning `y.mean()` captures the full **+0.1710**
with no slope, no leverage problem and no extrapolation.

**B6's slope is also not estimable over the range it is applied on.** Kish effective n for the
slope is **5.7** against a nominal 303: CYP1A2 is 13.9% of rows and carries **85.7%** of the
leverage in log₁₀ r, and the 12 highest-leverage rows are *all* CYP1A2. Drop CYP1A2 → slope
**−3.8779**. Restrict to r ≤ 1.25 (100% of the applied rows for 2C9/2D6/3A4) → **−3.6473**. The
shipped −1.3261 is **2.8× flatter** than the slope over its own domain, and the fit is biased
**+0.17 locally over 82% of the applied rows** — i.e. essentially the whole of B7's shift is the
fit's own misspecification. B6's R²s were in-sample; honest CV gives **+0.0174** (log-linear, 3.0×
shrinkage) and +0.0920 (isotonic).

### 2. Two of my headline numbers are tautologies

- **A4, "P(y\* > 2.211) = 0.4983", is a property of the median.** Sweeping `INACTIVE_R` over
  0.70–1.00 gives 0.4996 / 0.4995 / 0.5000 / 0.4991 / 0.4983 / 0.4973 / 0.5000 — **it cannot
  flip**, and its content is readable off `.median()` on line 1 of `allr_floor`. The *substantive*
  claim survives — the pipeline uses a **median as an upper bound**, and that is a
  mis-specification — but I presented a tautology as a measurement.
- **A7, "R² = −0.1492", is exactly −(mean−median)²/MSE(mean)**: it restates the skew A3 already
  reported. Under L1 the sign flips to **+0.0782**, and against a median baseline it is exactly
  0.000. Whether the constant is "worse than its own mean" depends entirely on the loss, which A7
  never names.

### 3. B3 was mislabelled and B4 is confounded

The per-isoform figures I quoted (**−0.6140 / −0.5000 / −0.2342**) are the **signed**-error
correlations, not |error|. The |error| versions are **−0.3991 / −0.5909 / −0.1515**. The effect is
still real and not a CYP3A4 artefact — dropping CYP3A4 (82.5% of rows) *strengthens* it to
−0.3870/−0.6884 — but the labels were wrong.

**B4's tercile gradient is confounded with isoform**: tercile 1 is **100% CYP3A4**. Within CYP3A4
alone the sequence is 2.5194 / 2.2937 / **2.3105** — a one-step drop then a **reversal**, not a
gradient.

### 4. A live defect in production, which I did not look for and have now verified myself

`scripts/cyp_final.py:114` builds the XGBoost early-stopping set as
`b = va[isfinite(y[va]) | (ps[va] & (kd[va]=="inactive"))]`, with `eval_metric="mae"`. **MAE
against a constant is minimised by predicting that constant.** My own measurement of the share of
that selection set which *is* the constant 2.2111:

| isoform | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| **CYP2D6** | **61.8%** | **64.0%** | **65.1%** |
| CYP1A2 | 45.6% | 46.5% | 46.7% |
| CYP2C9 | 33.7% | 33.3% | 32.2% |
| CYP3A4 | 2.8% | 2.4% | 5.4% |

**That ordering is exactly `findings/06`'s anomaly ordering.** Production early stopping actively
rewards collapse onto 2.2111 precisely where the floor mass is largest — a concrete mechanism for
"the members spuriously agree", which `findings/06` attributed to the point mass in *training*
without noticing it is also in the *model-selection* signal. The same rows enter `truth_v2`, which
`select_on=v2` uses to pick the 0.30 blend weight and the offsets.

### 5. Five more things found that I never checked

- **"The weak/active q10–q90 windows are wider than 0.4435" is broken twice.** One live window is
  **narrower** (CYP1A2 bin 3, width **0.4002**, issued to **213 rows**) — and the comparison is a
  category error: 0.4435 is an **SD**, those are **80% interval widths**. For a normal,
  q10–q90 = 2.563·SD = **1.1367**, against which **14 of 25** live windows are narrower.
- **The interval semantics are inconsistent between label kinds.** `inactive` gets medians of the
  *credible-interval bounds* (measured coverage **98.0%**); `weak`/`active` get **q10–q90
  predictive** quantiles (nominal 80%). The floor therefore ships **2.19× wider than consistency
  demands**, and its `lo = 1.0660` sits **below the population minimum 1.9056**, so the lower bound
  can never charge anything. `calibrate()` computes `lo_med`/`hi_med` for every bin and
  `screen_labels` **never reads them** — dead code that is exactly the fix.
- **The floor is more sensitive to the hand-picked `INACTIVE_R` than to any relabelling scheme.**
  0.850 → 2.5483, 0.875 → 2.3927, 0.900 → 2.2111, 0.925 → 2.1532, 0.950 → 2.1312. **A ±0.025
  change in an unjustified constant exceeds B7's entire +0.0966.**
- **A latent monotonicity inversion.** CYP2C9's table has no bin 0, so the pooled fallback assigns
  5.1844 there while CYP2C9's own bin 1 gets 5.5128 — the label **rises as inhibition falls**.
  **0 rows affected on this release**, so latent rather than live, but
  `test_calibration_is_monotone_in_the_readout` asserts monotonicity on the **pooled** table of a
  **synthetic fixture**, not on the per-isoform label map, so it cannot see it.
- **A physical cross-check on what the floor even is.** Inverting the readout under Hill-1
  (r = IC50/(IC50+50 µM)) implies pIC50 **3.238** for r ∈ [0.90, 0.95) against a measured DRC
  median of **2.483**. The fitted value sits **0.53–0.76 log units below what the compound's own
  readout implies**, so **the floor is a property of the organisers' curve-fitting convention for
  an unidentified fit, not of potency.** That argues for matching the scorer's own central tendency
  rather than modelling it — and it cuts against B6/B7, not for them.

### 6. And one thing that survived a serious attempt to break it

The **CYP2D6 hole is real but benign**, verified two ways after a bin-extrapolation route failed
its own validation (over-predicting the measured floor by +1.371/+0.885/+0.810 where it can be
checked). A **label-free marker** — DRC credible-interval width ≥ 2.0, i.e. the fit never bracketed
an IC50 — calibrated on the three isoforms that *do* have a measured floor and transferred to
CYP2D6 gives an independent estimate of **2.157–2.209** against the shipped **2.2111**. The
bootstrap 95% CI on the shipped median is [2.1651, 2.2510] and the pooled value sits inside
**every** per-isoform CI. **The pooling is defensible**, and the mechanism for the hole is now
exact: **the maximum r among CYP2D6's dose-response compounds is 0.6636 < 0.9**, so the two sets
are literally disjoint.

### The cheapest defensible change

**Split the floor at r = 1.0** — an edge already in `R_EDGES` — into **2.2732** (n=189) and
**2.1320** (n=114). It is the only rule besides isotonic to win on **both** OOF MSE (−4.38%) and
OOF MAE (−1.54%); it needs no slope, no extrapolation and no per-row weight; and it stays inside
the calibration range. **It moves CYP2D6 by −0.0723 — the opposite direction to B7.**
