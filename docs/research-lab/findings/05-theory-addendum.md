# Finding 05 — theory addendum: a hazard in the experiment I was running as it arrived

The decoupled strand's censored/weak-label search landed after its main hand-back. It corrects two
of its own claims and supplies one result that applies **directly to the Borda experiment running
at the time**.

---

## 1. The immediately actionable one: never weight by raw per-item variances

`[DERIVED, verified numerically]` If `σ̂ᵢ² = σᵢ²·χ²_ν/ν` then `E[1/σ̂ᵢ²] = ν/((ν−2)σᵢ²)` — **the
mean does not exist for ν ≤ 2 and the variance does not exist for ν ≤ 4.** Weights built from
few-replicate variance estimates are heavy-tailed, and a handful of spuriously small `σ̂ᵢ`
dominate.

Monte Carlo at *modest* heterogeneity (sd ratio 1:1.2), estimated-IVW risk relative to plain
averaging — note the damage **grows with the number of items**:

| items | ν=1 | ν=2 | ν=5 | ν=10 |
|---|---|---|---|---|
| 2 | 1.47× | — | — | 1.06× |
| **10** | **5.45×** | **3.23×** | **1.57×** | **1.21×** |

**The resolution is clean** (Carroll, Wu & Ruppert, JASA 1988): estimating a *variance function*
`σ²(x;θ)` from all n points has **no first-order effect** — asymptotically equivalent to oracle WLS
— whereas estimating each `σᵢ²` from ν replicates is a first-order, potentially catastrophic error.
**Fit a variance model and weight by its fitted values; never weight by raw per-item CI widths.**
Pair it with heteroskedasticity-consistent standard errors (Romano & Wolf 2017) so a wrong variance
model costs efficiency rather than validity.

**Why this matters here, immediately.** `cyp_borda_decode.py` takes `σ̂ᵢ` as the **raw spread of an
8-member ensemble** — ν ≈ 7 replicates per item, squarely in the hazard zone, with ~750 items per
cell where the table says the damage is worst. Borda does not invert σ̂, but it is just as
exposed in the other direction: a spuriously *small* σ̂ᵢ sharpens that item's separation from every
other and over-extremises its rank. **The arm therefore needs a second version whose σ̂ comes from a
fitted variance function**, and the difference between the two is itself the measurement of this
hazard on our data.

This is also a retroactive flag on the strand's own earlier measurement that `w = 1/(0.3+CI)` buys
+0.0070 — that is weighting by **raw per-item CI widths**, exactly the thing ruled against. The
+1.0% may be an underestimate of what a fitted variance function would give.

## 2. Two corrections the strand made to itself

- **"Squared loss is invariant to zero-mean label noise" has an exact converse** (Banerjee, Guo &
  Wang, IEEE TIT 2005): the conditional mean is the unique minimiser for *every* Bregman
  divergence, and conversely any loss whose minimiser is always the conditional mean **must** be
  Bregman. So **Huber and Tukey are not Bregman and are not noise-invariant** — breakdown-point
  robustness is not noise-invariance, and conflating them is a common error.
- **"L1 is robust to label noise" is folklore for regression.** Exact quadrature: squared loss
  shifts by **0.0000**; pinball τ=0.5 by **+0.183**; τ=0.9 by **+0.489**; τ=0.1 by **−0.766**.
  `ℓ_p` for `p ≠ 2` is invariant only if the noise is symmetric **and** the clean conditional law
  is essentially deterministic; pinball/expectile with τ ≠ ½ is **always** biased.

## 3. Structural facts that bear on our screen labels

- **A single fixed threshold identifies the response only up to scale** (Manski 1988). With σ
  unknown and one fixed threshold T, threshold-only labels identify `(x'β − T)/σ` and nothing more.
  **Consequence for us: the precise DRC labels are not merely more efficient — they are what
  identifies the scale. If every weak label uses the same threshold, no volume of weak data
  recovers it. Threshold *variation* is identifying information, not nuisance.**
- **The exchange rate.** The one-bit minimax floor is **π/2 ≈ 1.57 threshold calls per precise
  label**, reachable only with *adaptive* thresholds near the conditional median. At a fixed
  threshold the cost is `1/e(z)`: **1.57 at the median, 2.28 at the 16th percentile, 4.47 at the
  5th, 7.6 at the 2.3rd, 68.6 at the 0.1st.** Our screen is a single-concentration call, so where
  it sits relative to the label distribution decides whether 6,900 weak labels are worth ~4,400
  precise ones or ~100.
- **Threshold-only labels cost a constant on structured targets and a *rate* on unstructured ones**
  (Huang, Ann. Statist. 1996): under case-1 interval censoring the baseline converges at `n^{1/3}`
  yet the **regression parameter is √n-consistent and semiparametrically efficient**. Design rule:
  put the quantity you care about in the parametric part, where threshold labels cost only a
  constant.
- **If the weak-label model is even slightly misspecified, precise labels get MORE valuable as weak
  data accumulates** (Chen et al., AISTATS 2021): the data-value ratio is constant when
  well-specified and grows **linearly in n_U** when misspecified. This reverses the naive intuition
  and argues for keeping a precise-label budget alive indefinitely.
- **Tobit is the wrong tool and CLAD is the unreliable right one.** Tobit is inconsistent under
  heteroscedasticity (Arabmazar & Schmidt 1981) or non-normality; Powell's CLAD is consistent under
  arbitrary both, but its objective is non-convex and at 50% censoring with one regressor and
  n=100 converges **5–37%** of the time. The fix is **Chernozhukov & Hong's three-step censored
  quantile regression** (JASA 2002) — three *convex* steps recovering exactly Powell's asymptotic
  variance, with a first-step classifier that **may be misspecified**. This materially changes the
  "censored/Tobit treatment of the screen labels" candidate from finding 02: build it as CH 3-step,
  not as Tobit.

## 4. The right machinery for our actual design

**Prediction-powered inference** (Angelopoulos, Bates, Fannjiang, Jordan & Zrnic, *Science* 2023):
provably valid confidence intervals from a small gold-standard sample plus a large
machine-labelled one, with **no assumptions whatsoever on the prediction algorithm**. The
censoring-specific analogue that fuses precise and threshold-only data at the semiparametric
efficiency bound is Li & Li (2025) — the closest published analogue of our 1,200-precise +
6,900-weak design.

**And a data-acquisition rule with a matching minimax bound** (van Rooyen & Williamson, JMLR 2017):
`n` labels corrupted by `T` are worth at most **`Clarity(T)·n`** clean ones *for any algorithm*;
buy mixed-quality data greedily in increasing order of **`costᵢ · ‖ℓ_Rᵢ‖²_∞`**. This is the
weak-label analogue of our own Kish-ESS result.

## 5. A pointed observation about our evaluation convention

Rindt et al. (AISTATS 2022) **prove** that time-dependent concordance, survival-CRPS, the
integrated Brier score and the integrated binomial log-likelihood are **not proper scoring rules**.
The strand's generalisation: *a refit-baseline skill score and a censoring-aware metric are both
improper, and in both cases the fix is to anchor the correction, or map the predictive distribution
through the censoring mechanism, before scoring.*

ST-RAE is normalised by a constant predictor at ȳ — i.e. it **is** a skill score, scored against
censored interval labels. We do not control the challenge's metric, but this says plainly that
optimising it is not the same as optimising a proper scoring rule, and that the two can diverge.
That is the formal version of a risk this programme has already seen empirically: placement gains
bought at the cost of the ranking the same board reports.
