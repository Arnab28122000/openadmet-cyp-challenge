# 11 — Distributional models and metric-aware decoding

Research brief, 2026-10-05. Scope: diffusion / flow matching for *conditional regression*, and
decision-theoretic decoding under the challenge's Soft-Threshold Relative Absolute Error.

Every claim is marked **[verified]** (read in the paper, the repo, or measured here) or
**[assumed]** (inference, extrapolation, or not checked). Sources at the end.

---

## 0. Verdict first

1. **The decision theory has an exact, closed solution.** Under the metric's interval loss, the
   Bayes-optimal point estimate is the **weighted median of the pooled set of interval endpoints**
   — pool every Monte-Carlo sample of `lo` with every sample of `hi`, weight each by `1/scale`, take
   the weighted median. `O(S log S)` per molecule, no grid, no tuning. Derived in §2.4,
   **verified numerically to grid resolution** on six adversarial test cases (§2.11). [verified]
2. **For Gaussian bound predictions there is a four-parameter closed form:**
   `a* = (σ_U·μ_L + σ_L·μ_U) / (σ_L + σ_U)` — each bound weighted by the *other* bound's standard
   deviation. Verified against a 4M-sample Monte-Carlo argmin (0.1609 vs 0.1610). [verified]
3. **The user's prior that the loss is "mode-seeking" is wrong as stated, and the correction matters.**
   Our loss is *unbounded* (it grows linearly outside the interval), so it is strictly
   **median/quantile-seeking**, not mode-seeking. Mode-seeking appears only for the *bounded*
   variant (`1{outside}` — a "fraction inside the interval" metric), where the optimum is the
   window-smoothed mode. We score the unbounded one. §2.8. [verified by derivation]
4. **Measured on the real CYP labels, the entire decoding gain collapses onto a one-dimensional
   map.** With a mean-calibrated simulated predictor at residual sd 0.5, macro ST-RAE over 5 seeds:

   | decoder | macro ST-RAE | Δ vs plain |
   |---|---|---|
   | plain point prediction | 0.2803 | — |
   | **+ per-endpoint scalar offset** (what the programme already verified) | 0.2755 | −0.0048 |
   | + full Bayes decode, Gaussian predictive | 0.2601 | −0.0202 |
   | + full Bayes decode, empirical conditional predictive | 0.2546 | −0.0257 |
   | **+ 8-bin shift δ(ŷ) fitted on validation to minimise ST-RAE** | **0.2533** | **−0.0271** |

   The **8-parameter table beats the full Bayesian decode**, and adding a residual shift on top of
   the Bayes decode buys 0.0008. With genuine heteroscedasticity simulated, conditioning the
   decoder on the per-molecule predictive spread made things *worse* (−0.0137 vs −0.0174 for the
   1-D map) because of the extra estimation variance. [verified by simulation; §2.10]
5. **Therefore: do not build a diffusion model for this.** CARD costs T=1000 steps × 1000 samples
   per test point as published [verified from its config], to estimate a predictive distribution
   whose only use here is to be collapsed through a decoder that an 8-row lookup table already
   approximates to within 0.001 ST-RAE.
6. **Rank: (a) 1-D ST-RAE-fitted decode map δ(ŷ) — half a day, expected +0.01–0.03;
   (b) quantile heads on `lo` and `hi` + pooled-median decode — two days, expected +0.00–0.02 over
   (a), and worth it mainly as the *principled* version that generalises; (c) flow matching —
   expected +0.000 ± noise; (d) CARD/diffusion — expected +0.000 and a week gone.** §7, §8.

---

## 1. What the metric actually is (measured locally, not assumed)

From `talindrew-drugrx/drugrx/cyp/metric.py` [verified, read]:

```python
def soft_threshold_error(y_pred, lo, hi):
    return np.clip(lo - y_pred, 0.0, None) + np.clip(y_pred - hi, 0.0, None)

def st_rae(y_pred, y_true, lo, hi):
    err   = soft_threshold_error(y_pred, lo, hi).sum()
    denom = np.abs(y_true - y_true.mean()).sum()
    return err / denom
```

So per endpoint `e`:

```
ST-RAE_e(a) = Σ_i dist(a_i, [lo_i, hi_i]) / D_e ,    D_e = Σ_i |y_i − ȳ_e|
MA-ST-RAE   = (1/4) Σ_e ST-RAE_e
```

Three structural facts that determine everything below:

* **`D_e` does not depend on the predictions.** It is a per-endpoint constant. So the weight on
  molecule `i` is `1/(4 D_e)` — *uniform within an endpoint*. There is **no per-molecule
  reweighting** in the metric as implemented. [verified]
  *Caveat:* the file's own docstring says "the organisers' exact normalisation is not public", so
  this is the programme's reconstruction. If the organisers divide per-molecule (by `|y_i|`, or by
  the interval width), the weights become non-uniform — the general weighted form in §2.4 covers
  that case, so the derivation is robust to the ambiguity. [assumed]
* **The metric decomposes additively across molecules and across endpoints.** Therefore the optimal
  decision for molecule `i`, endpoint `e` depends only on `p(lo_ie, hi_ie | x_i)`. **There is no
  decision-theoretic value in modelling the 4-dimensional joint `p(y_1..y_4 | x)`.** A joint model
  can only help *statistically* (shared structure, as the multi-task heads already do), never
  *decision-theoretically*. This kills the main reason one would reach for a 4-dim diffusion model.
  [verified by derivation]
* **The training loss is already metric-shaped, but Huberised.** `scripts/cyp_train.py`'s
  `build_loss` uses the interval hinge with `huber=1.0` on the primary heads [verified, read].
  Measured on the real labels at residual sd 0.45: **99.1% of molecules have hinge distance < 1.0,
  and 96% of the total loss mass sits in the quadratic region** [verified]. So the net is in
  practice minimising a *squared* dead-zone loss, whose Bayes act is a mean-type balance
  (`E[(a−U)₊] = E[(L−a)₊]`), not the metric's median-type one. I expected this to matter; **it does
  not** — decoding with the squared-optimal rather than the linear-optimal point changed macro
  ST-RAE by 0.0003 (0.2653 vs 0.2650) [verified]. Recorded so nobody spends a day on it.

### Real interval geometry (measured on `cyp-challenge-TRAIN_inhibition.csv`, 4,905 rows)

| endpoint | n | median width | median `d_lo` | median `d_hi` | mean asym `(d_hi−d_lo)/2` | sd asym | corr(width, y) | frac y<4 | MAD(y) |
|---|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 1412 | 0.328 | 0.170 | 0.154 | −0.014 | 0.093 | **−0.868** | 0.164 | 0.751 |
| CYP2C9 | 1285 | 0.526 | 0.246 | 0.279 | −0.015 | 0.100 | **−0.846** | 0.202 | 0.593 |
| CYP2D6 | 1493 | 0.272 | 0.130 | 0.137 | −0.019 | 0.134 | **−0.733** | 0.086 | 0.631 |
| CYP3A4 | 2335 | 0.379 | 0.169 | 0.200 | −0.035 | 0.132 | **−0.891** | 0.404 | 0.896 |

`d_lo = y − lo`, `d_hi = hi − y`. **`y` is inside `[lo, hi]` for 100% of rows in all four
endpoints** [verified] — so `L ≤ y ≤ U` always, which licenses modelling the bounds as `y` minus /
plus a non-negative offset.

Two readings that reshape the whole plan:

* **The intervals are nearly symmetric about `y`.** Median asymmetry is +0.002 everywhere; the mean
  is −0.014 to −0.035. The "shift toward the wider side" exploit the programme hypothesised is
  worth about **0.02 pIC50 units on average** — which is exactly why the fitted per-endpoint offsets
  came out at +0.02 to +0.08 and bought only +0.003. The hypothesis was right in form and an order
  of magnitude too optimistic in size. [verified]
* **The *width* is what carries the signal, and it is strongly `y`-dependent** (r = −0.73 to −0.89).
  Low-activity compounds get wide intervals. That is a *nonlinearity in the decode map*, not an
  offset: the optimal prediction is pushed around because `L(y) = y − d_lo(y)` stretches at the low
  end while `U(y) = y + d_hi(y)` compresses. A single scalar offset cannot represent it; a 1-D
  function of `ŷ` can, and that is the measured win. [verified]
* **The dead zone is narrow relative to the error.** Median width 0.27–0.53 against MAD(y) 0.59–0.90
  and a plausible residual sd of ~0.45–0.55. So the metric is *approximately MAE with a small
  per-molecule credit* — which bounds how much any decoding cleverness can be worth, and explains
  why every model arm lands inside 0.03 of every other. [verified]

---

## 2. Decision theory under the interval loss

**This is the section that matters. Everything is derived from scratch and then checked numerically.**

### 2.1 Setup

For one molecule and one endpoint we choose an action `a ∈ ℝ`. The outcome is a triple
`(L, U, S)` — lower bound, upper bound, scale — with `L ≤ U` and `S > 0`. At decision time
`(L, U, S)` is unknown; we have a predictive distribution over it, induced by whatever model we
have. Write `W = 1/S` for the weight.

The loss is the metric's per-molecule term:

```
ℓ(a; L, U, S) = (1/S) · max(0, L − a, a − U)                                            (2.1)
```

### 2.2 The loss is a distance to an interval, and splits into two hinges

Because `L ≤ U`, at most one of `L − a` and `a − U` can be positive, so the max is their sum:

```
max(0, L − a, a − U) = (L − a)₊ + (a − U)₊ = dist(a, [L, U]) = |a − Π_[L,U](a)|           (2.2)
```

where `Π` is projection onto the interval and `(z)₊ = max(z, 0)`. This identity is exact, not an
approximation, and it is what makes the problem tractable: the loss is a sum of two one-sided
hinges, each convex and piecewise linear in `a`. [verified by derivation; matches `metric.py`,
which literally computes `clip(lo−p) + clip(p−hi)`]

This is the **ε-insensitive loss of support-vector regression** when `L = y − ε`, `U = y + ε`, and
the **dead-zone / interval-censored absolute loss** in the statistics literature. [verified]

### 2.3 Risk, convexity, subgradient

```
R(a) = E[ W · ((L − a)₊ + (a − U)₊) ]                                                    (2.3)
```

`R` is convex (a non-negative mixture of convex functions of `a`) and piecewise linear with kinks
at the support points of `L` and `U`. Its derivative, where it exists:

```
R′(a) = E[ W · ( 1{U < a} − 1{L > a} ) ]
      = E[W·1{U < a}] − E[W·1{L > a}]                                                    (2.4)
```

`R′` is non-decreasing (the first term is non-decreasing in `a`, the second non-increasing), from
`−E[W]` at `a = −∞` to `+E[W]` at `a = +∞`. So a minimiser exists and the first-order condition is:

```
E[W·1{U < a*}] = E[W·1{L > a*}]                                                          (2.5)
```

> **The Bayes act balances the weighted probability that the whole interval lies below the
> prediction against the weighted probability that it lies entirely above.**

### 2.4 Main result: the pooled weighted median

Define the `W`-weighted CDFs of the two bounds,
`F_L(a) = E[W·1{L ≤ a}] / E[W]` and `F_U(a) = E[W·1{U ≤ a}] / E[W]`.
Writing `E[W·1{L > a}] = E[W] − E[W·1{L ≤ a}]`, condition (2.5) becomes

```
F_U(a*) + F_L(a*) = 1                                                                    (2.6)
```

Now observe that the left-hand side is twice the CDF of the **pooled** distribution that puts half
its weight on `L` and half on `U`:

```
G(a) := ½ F_L(a) + ½ F_U(a)        ⟹        G(a*) = ½                                    (2.7)
```

> ### Theorem (interval-loss Bayes act)
> The ST-RAE-optimal point estimate is the **median of the equal mixture `½ p_L + ½ p_U`,
> weighted by `W = 1/S`**:
> ```
> a*  =  median of  ( ½ · law(L)  +  ½ · law(U) )   under weight W
> ```
> In Monte-Carlo form, given samples `{(l_j, u_j, s_j)}_{j=1..S}` from the predictive:
> ```
> a*  =  weighted_median( [l_1..l_S, u_1..u_S],  weights [w_1..w_S, w_1..w_S] ),  w_j = 1/s_j
> ```
> If `R′` is zero on an interval, the whole interval is optimal and any point in it attains the
> minimum; the weighted median is the left endpoint. [verified: derivation above, plus numerical
> check in §2.11]

**Two things about this are worth pausing on.**

* **It depends only on the *marginals* of `L` and `U`, never on their joint.** Their correlation is
  irrelevant. This is a real saving: you do not need a joint model of the interval, only two
  one-dimensional predictive distributions. [verified by derivation]
* **It is not an interval statistic at all — it is a median.** Which means every tool built for
  quantile estimation applies directly, and nothing about this problem requires a generative model.

### 2.5 Corollaries worth having memorised

**(a) Degenerate interval → median.** If `L = U = y`, then `F_L = F_U = F_y` and (2.6) gives
`2F_y(a*) = 1`: the conditional median. Correct — a zero-width interval makes the loss `|y−a|`.
[verified]

**(b) Constant widths → median of the midpoint.** If `L = y − d_lo`, `U = y + d_hi` with `d_lo`,
`d_hi` constants, then (2.6) reads `F_y(a + d_lo) + F_y(a − d_hi) = 1`, and for `F_y` symmetric
about `m` the solution is

```
a* = m + (d_hi − d_lo)/2 = median( (L + U)/2 )                                           (2.8)
```

**So the optimum shifts by exactly half the interval asymmetry, toward the wider side — and
symmetric widening does not move it at all.** Only the asymmetry matters for the *location*; the
width only flattens the risk (it forgives, it does not redirect). On the real CYP intervals the mean
asymmetry is −0.014 to −0.035, so `(d_hi − d_lo)/2 ≈ −0.02` — this corollary *predicts* the tiny
measured offset gain. [verified by derivation; the prediction matches PROGRESS.md's +0.003]

**(c) Gaussian bounds → a closed form.** If `L ~ N(μ_L, σ_L²)` and `U ~ N(μ_U, σ_U²)` marginally
(any correlation), then `Φ((a−μ_L)/σ_L) + Φ((a−μ_U)/σ_U) = 1`, and using `Φ(z) = 1 − Φ(−z)`:

```
(a − μ_L)/σ_L = (μ_U − a)/σ_U        ⟹        a* = (σ_U·μ_L + σ_L·μ_U) / (σ_L + σ_U)     (2.9)
```

> **Each bound is weighted by the *other* bound's standard deviation.** The optimum sits closer to
> whichever bound is predicted more *confidently* — you can hug a bound you know precisely.
> Equal variances give exactly the predicted interval midpoint, which is why "interval midpoint" is
> a natural baseline (and why PROGRESS.md's interval-midpoint floor, 0.6107, edges the median floor,
> 0.6116).

Numerical check: `μ_L=−0.7, σ_L=0.9, μ_U=0.4, σ_U=0.25` → formula 0.1609, 4M-sample MC argmin
0.1610. [verified]

**(d) Non-uniform scale.** If the organisers' normalisation *is* per-molecule, `w_j = 1/s_j` enters
exactly as written. If `s` is the interval width, `1/s` **down-weights the wide low-activity
compounds**, which shrinks the already-small asymmetry exploit further. [verified by derivation]

### 2.6 Comparison: mean, median, quantile

| estimate | minimises | relation to `a*` |
|---|---|---|
| conditional **mean** `E[y\|x]` | squared error | **Not optimal.** Consistent for squared error only (a Bregman score). Equals `a*` only if the predictive is symmetric and the interval is symmetric. [verified — Gneiting 2011] |
| conditional **median** | absolute error | **The `d_lo = d_hi = 0` special case of `a*`.** It is `a*` whenever the interval is symmetric about `y` with constant width — which is *nearly* the CYP case, hence the small gain. [verified] |
| a fixed **quantile** `τ` | pinball loss `ρ_τ` | **Not the right functional, but the right family.** `a*` is the median of a *derived* distribution (the pooled bounds), not a non-central quantile of `p(y\|x)`. [verified] |
| **mode** | bounded/0-1 loss | **Not optimal here.** See §2.8. |

**The pinball connection, exactly.** With `ρ_τ(z) = τ·z₊ + (1−τ)·(−z)₊`:

```
(L − a)₊ = ρ_1(L − a)        and        (a − U)₊ = ρ_0(U − a)
ℓ(a) = (1/S)·[ ρ_1(L − a) + ρ_0(U − a) ]                                                 (2.10)
```

So the loss is the **sum of two degenerate pinball losses at opposite extremes** — it asks for the
100th percentile of `L` and the 0th percentile of `U` simultaneously, and the Bayes act is the
compromise. This is why the functional is elicitable and median-like: by Gneiting's
characterisation, generalised-piecewise-linear scores are exactly the ones consistent for
quantiles, and (2.10) is a sum of two such. [verified — Gneiting 2011, arXiv:0912.0902]

### 2.7 Why this *is* "quantile regression", operationally

(2.6) says: fit the predictive CDFs of `L` and of `U`, and solve a one-dimensional root-find. With
`K` quantile levels per bound you get a step approximation of `F_L` and `F_U` and a bisection
solves (2.6) in `log` time. **No sampling, no generative model, 2K numbers per molecule.** That is
the entire content of the "distributional modelling" direction for this metric.

### 2.8 Mode-seeking: the honest correction

The brief asserted that "an interval-censored loss is minimised by a mode-seeking rather than
mean-seeking estimate". **Half right, and the wrong half matters.**

* Our loss (2.1) is **unbounded** — it keeps growing linearly as you move away. `R` is therefore
  convex and the Bayes act is a *median-type* statistic (§2.4). A multimodal predictive does **not**
  push `a*` into a mode; it pushes `a*` to the pooled-median, which can sit in a valley between two
  modes. **Mode-seeking here would be wrong and would cost score.** [verified by derivation; the
  bimodal test cases in §2.11 confirm the pooled median, not a mode, is the argmin]
* Mode-seeking appears when the loss is **bounded**. For the coverage metric
  `ℓ(a) = 1{a ∉ [L, U]}`, the risk is `1 − C(a)` with

  ```
  C(a) = P(L ≤ a ≤ U) = E[ 1{L ≤ a ≤ U} ]                                               (2.11)
  ```

  maximised at the mode of the **interval-smoothed** coverage function — for `L = y−ε, U = y+ε`
  that is `P(|y − a| ≤ ε)`, i.e. the mode of the density convolved with a box of width `2ε`. That
  *is* genuinely mode-seeking, and `ε` sets the smoothing bandwidth. [verified by derivation]
* **Which do we score?** The unbounded one — `metric.py` sums raw distances with no cap. So:
  median-type. If the organisers ever cap the per-molecule error (a "relative error clipped at 1"
  style metric), the estimator flips to the smoothed mode and this section's §2.11 algorithm must be
  swapped for a kernel-mode search. Worth re-reading the rules once for a cap. [assumed —
  not verifiable, the exact normalisation is not public]
* Intuition for why width does *not* create mode-seeking: a wide interval **flattens** the risk
  (a plateau where many predictions tie) rather than **tilting** it. Flat risk means the decision
  matters *less*, not that it moves to a mode. This is the single most important corrective in this
  document: wide intervals are a reason to stop optimising the decoder, not a reason to build a
  generative model.

### 2.9 The reduction that makes diffusion unnecessary

Suppose — as is true for any model we would actually ship — that the predictive distribution and the
interval model depend on `x` **only through the point prediction `ŷ(x)`** (and possibly a spread
`ŝ(x)`). Then `a*` is a *deterministic function of `ŷ`*:

```
a* = A(ŷ)        (or  A(ŷ, ŝ)  with a spread)                                            (2.12)
```

You never have to estimate `p(y|x)`. You can **learn `A` directly by empirical risk minimisation on
a held-out split**, because the metric itself is the thing you want `A` to minimise:

```
Â = argmin_{A monotone}  Σ_{i ∈ val}  dist( A(ŷ_i), [lo_i, hi_i] )                       (2.13)
```

This is a one-dimensional isotonic/piecewise-constant fit on a few thousand points. It is the
**cheapest correct thing in this entire brief**, it subsumes the per-endpoint scalar offset the
programme already verified (that is the rank-0, `A(ŷ) = ŷ + δ` case), and §2.10 measures it at the
top of the table.

### 2.10 Measured: how much any of this is worth, on the real labels

Setup, deliberately favourable to the fancy methods and honest about it:

* Real `(y, lo, hi)` from `cyp-challenge-TRAIN_inhibition.csv`, all four endpoints.
* A **mean-calibrated** simulated point predictor. Writing `v = Var(y)` and a target residual sd
  `σ`, set `λ = 1 − σ²/v` and `ŷ = ȳ + λ(y − ȳ) + N(0, λσ²)`. Under joint normality this gives
  `E[y|ŷ] = ŷ` exactly and `Var(y|ŷ) = σ²`, so the predictive `N(ŷ, σ)` is *correctly calibrated*.
  (The obvious `ŷ = y + N(0,σ)` is **wrong** and I nearly shipped a conclusion from it — see
  Pitfall P1.)
* Interval model `d_lo(y), d_hi(y)` as 12-bin means, fitted on the fit half only.
* Everything — the interval model, the offsets, the decode map — fitted on half the data and
  reported on the other half. 5 seeds.

**σ = 0.5, macro ST-RAE, mean over 5 seeds:**

| decoder | ST-RAE | Δ |
|---|---|---|
| plain `ŷ` | 0.2803 | — |
| global per-endpoint offset (1 param) | 0.2755 | −0.0048 |
| Bayes decode, Gaussian predictive `N(ŷ,σ)` | 0.2601 | −0.0202 |
| Bayes decode, empirical conditional residuals (8 bins of `ŷ`) | 0.2546 | −0.0257 |
| **8-bin shift `δ(ŷ)` fitted to ST-RAE** | **0.2533** | **−0.0271** |
| Bayes decode + residual 8-bin shift | 0.2538 | −0.0265 |

**Heteroscedastic variant** (per-molecule `σ_i ~ lognormal(log 0.5, 0.5)`, clipped to [0.15, 1.5],
and *known exactly to the decoder* — a best case):

| decoder | ST-RAE | Δ |
|---|---|---|
| plain | 0.3045 | — |
| **8-bin `δ(ŷ)`** | **0.2870** | **−0.0174** |
| 8×3-bin `δ(ŷ, σ̂)` | 0.2908 | −0.0137 |
| Bayes decode with true per-molecule `σ_i` | 0.2906 | −0.0139 |
| Bayes decode with a single pooled `σ` | 0.2902 | −0.0143 |

**Conclusions, and they are not the ones I expected:**

1. The decode gain is real and roughly **5× the verified offset gain**: −0.027 vs −0.005.
   For scale, PROGRESS.md's whole blend win was −0.028 with CI [−0.016, −0.043].
2. **A 1-D lookup table matches or beats the exact Bayes decoder.** It wins because it absorbs the
   predictive's *miscalibration* and the interval geometry in one fit, while the Bayes decoder is
   only as good as the predictive you feed it (Gaussian: −0.0202; empirical: −0.0257).
3. **Knowing the per-molecule predictive spread buys nothing** — it actually hurts, through
   estimation variance over 24 cells on ~700 fit points. The dead zone is too narrow relative to
   the spread for second moments to matter.
4. Point 3 is the empirical death certificate for diffusion/flow on this task: those methods' entire
   selling point is a *rich* per-`x` predictive distribution, and the measurement says the second
   moment is already worthless here.

Caveats, explicitly: the simulated predictor is Gaussian-residual and homoscedastic (or lognormal-
heteroscedastic) around a mean-calibrated centre; a real model's `p(y|ŷ)` is skewed at the
low-activity pile in ways this does not capture. The *direction and rough size* should transfer; the
exact numbers will not. **Re-run this with the real cached validation predictions before believing
the magnitude** — the xgb arm is MAE-trained and the net arm is hinge-trained, so they will have
different decode maps and the blend's map is a mixture of the two. [verified simulation; transfer
assumed]

### 2.11 The algorithm, exactly

```python
def interval_bayes_act(lo_samples, hi_samples, scale_samples=None):
    """Exact ST-RAE-optimal point estimate from Monte-Carlo samples of the interval.

    lo_samples, hi_samples : (S,) draws from the predictive over (L, U). Pairing is
                             irrelevant (Theorem 2.4 depends only on the marginals).
    scale_samples          : (S,) draws of S; None => uniform weight (our metric).
    Returns the minimiser of  E[(1/S)*dist(a, [L, U])].   O(S log S).
    """
    import numpy as np
    v = np.concatenate([lo_samples, hi_samples])
    if scale_samples is None:
        w = np.ones_like(v)
    else:
        inv = 1.0 / np.asarray(scale_samples)
        w = np.concatenate([inv, inv])
    o = np.argsort(v, kind="stable")
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])
```

**Why this is the minimiser, operationally.** `R(a) = Σ_j w_j·dist(a, [l_j, u_j])` is convex and
piecewise linear with breakpoints exactly at the `2S` endpoints. Its right-derivative at `a` is
`Σ_j w_j 1{u_j < a} − Σ_j w_j 1{l_j > a}`, which crosses zero at the pooled weighted median
(§2.4). No grid search, no learning rate, no tolerance.

**Numerical verification** (bimodal `y`, strongly asymmetric widths `d_lo ~ |N(0.8,0.4)|`,
`d_hi ~ |N(0.2,0.1)|`, `n = 4001`, both uniform and `U(0.2,3)` random weights, 200k-point grid
argmin as ground truth):

```
trial 0: grid=0.3146  pooled-wmedian=0.3146   risk 0.949697 / 0.949697
trial 1: grid=0.3315  pooled-wmedian=0.3315   risk 0.905765 / 0.905765
trial 2: grid=0.3054  pooled-wmedian=0.3044   risk 0.958169 / 0.958169
trial 3: grid=0.3196  pooled-wmedian=0.3196   risk 0.939818 / 0.939818
trial 4: grid=0.2887  pooled-wmedian=0.2885   risk 0.961711 / 0.961711
trial 5: grid=0.3394  pooled-wmedian=0.3393   risk 0.970226 / 0.970226
```

Risks agree to six decimals in every case (the small position differences are inside the plateau).
[verified]

**Sample count.** The median of `2S` draws has sampling sd `≈ 1/(2 f(a*) √(2S))`. With a predictive
sd of 0.5 (so `f(a*) ≈ 0.8`), `S = 256` gives ≈ 0.028 pIC50 of jitter and `S = 1024` gives ≈ 0.014.
Since the risk is *flat* near `a*`, the ST-RAE cost of that jitter is second-order, so **S = 256 is
plenty and S = 1024 is comfortable** — against CARD's published 1000 and Treeffuser's 100.
[assumed — order-of-magnitude, from the median's asymptotic variance]

### 2.12 How to get `p(L|x)` and `p(U|x)` when you only have `p(y|x)`

`lo` and `hi` **are available at training time**, which is the whole opportunity. Three routes,
cheapest first:

1. **Predict the bounds directly.** Add two heads per endpoint, `q_lo(x)` and `q_hi(x)`, trained on
   `lo` and `hi` as ordinary regression targets. With quantile heads (§6.1) you get `F_L` and `F_U`
   and can solve (2.6) by bisection. This needs no model of `y` at all. **Recommended.**
2. **Model the offsets as a function of `y`.** Fit `d̂_lo(y), d̂_hi(y)` (12-bin means or isotonic —
   width is monotone in `y`, r ≈ −0.85), then sample `y_j ~ p(y|x)` and set
   `l_j = y_j − d̂_lo(y_j)`, `u_j = y_j + d̂_hi(y_j)`. This is what §2.10 measured.
3. **Joint head.** Predict `(y, d_lo, d_hi)` together. Unnecessary — Theorem 2.4 only needs the
   marginals. [verified by derivation]

Enforce `d̂_lo, d̂_hi ≥ 0` (true in 100% of real rows) so `l_j ≤ u_j` and (2.2) stays exact.

---

## 3. Diffusion for regression targets: CARD and successors

### 3.1 CARD (Han, Zheng, Zhou — NeurIPS 2022, arXiv:2206.07275)

**Forward process.** CARD's one genuinely clever idea: the diffusion does not end at `N(0, I)`, it
ends at `N(f_φ(x), I)` where `f_φ` is a pre-trained conditional-mean estimator. [verified]

```
q(y_t | y_{t-1}, f_φ(x)) = N( y_t ; √(1−β_t)·y_{t-1} + (1−√(1−β_t))·f_φ(x),  β_t I )
q(y_t | y_0,     f_φ(x)) = N( y_t ; √ᾱ_t·y_0 + (1−√ᾱ_t)·f_φ(x),  (1−ᾱ_t) I )
   α_t = 1 − β_t,   ᾱ_t = Π_{s≤t} α_s
```

The mean interpolates from the observed `y_0` to the point prediction `f_φ(x)`, so the prior is
already a good guess and the diffusion only has to learn the *residual* shape of `p(y|x)`.
[verified]

**Network.** `ε_θ(x, y_t, f_φ(x), t)` → a noise estimate of the same dimension as `y`. Inputs: the
features `x`, the noisy target `y_t`, the point prediction `f_φ(x)` (fed as an explicit extra
conditioning channel), and the timestep. [verified]

**Training loss** (after pre-training `f_φ` with MSE):

```
L_ε = E_{y_0, t~U{1..T}, ε~N(0,I)}
      ‖ ε − ε_θ( x, √ᾱ_t·y_0 + √(1−ᾱ_t)·ε + (1−√ᾱ_t)·f_φ(x),  f_φ(x),  t ) ‖²
```
[verified]

**Sampling loop** (Algorithm 2):

```
y_T ~ N( f_φ(x), I )
for t = T, T-1, ..., 1:
    ŷ_0  = ( y_t − (1−√ᾱ_t)·f_φ(x) − √(1−ᾱ_t)·ε_θ(x, y_t, f_φ(x), t) ) / √ᾱ_t
    z    = N(0, I) if t > 1 else 0
    y_{t-1} = γ_0·ŷ_0 + γ_1·y_t + γ_2·f_φ(x) + √β̃_t · z

  γ_0 = β_t·√ᾱ_{t-1} / (1−ᾱ_t)
  γ_1 = (1−ᾱ_{t-1})·√α_t / (1−ᾱ_t)
  γ_2 = 1 + ( (√ᾱ_t − 1)(√α_t + √ᾱ_{t-1}) ) / (1−ᾱ_t)
  β̃_t = β_t · (1−ᾱ_{t-1}) / (1−ᾱ_t)
```
[verified — equations and coefficients read from the paper]

**Published cost for low-dimensional `y`** — read from `regression/configs/uci_boston.yml` in the
official repo [verified]:

| setting | value |
|---|---|
| `diffusion.timesteps` | **1000** |
| `beta_schedule` | linear, 1e-4 → 0.02 |
| `testing.n_z_samples` | **1000** |
| `model.feature_dim` | 128 |
| `f_φ` architecture | MLP `[100, 50]`, 300 pre-train epochs, early stopping |
| `training.n_epochs` | 5000 |
| point estimate reported | sample mean (regression); majority vote (classification) |
| UQ metrics | QICE, PICP, NLL |

**This is 10⁶ network evaluations per test point per endpoint as published.** For 750 test molecules
× 4 endpoints with a 2,500-dim input that is not a toy cost, and nothing in the paper suggests
fewer steps work. [verified for the published config; the extrapolation to our cost is assumed]

**Does `y ∈ ℝ⁴` need 1000 steps?** No — nothing about DDPM's step count is driven by `dim(y)`; it is
driven by the discretisation error of the reverse SDE and the schedule. Treeffuser uses **50**
Euler-Maruyama steps for the same job [verified], and rectified flow gets usable samples in **1**
after reflow [verified]. **So 20–50 steps is the right order for a 1–4 dimensional `y`** and 1000 is
a legacy image-generation default. [assumed, but well supported by the two verified data points]

**The honest objection to CARD here.** Its headline claim is accuracy "when the conditional
distribution of `y` given `x` is multi-modal" [verified, from the abstract]. We have one dose-response
fit per molecule per enzyme — there is no mechanism producing a bimodal `p(pIC50 | x)` beyond the
low-activity censoring pile, and §2.10 shows the *second moment* is already worthless to the
decoder, let alone the modes. [assumed, but §2.10's heteroscedastic result is direct evidence]

### 3.2 Successors

An arXiv full-text search for conditional-diffusion probabilistic regression returns essentially
**one** relevant successor [verified — arXiv API query, 1 hit]:

**Treeffuser** (Beltran-Velez et al., NeurIPS 2024, arXiv:2406.07658) — conditional diffusion whose
score function is **gradient-boosted trees**, not a network. [verified] Mechanism:

```
objective:   argmin_S  E_{π(x)} E_t E_{π(y|x)} E_{p_0t}
                ‖ ∇_y log p_0t(y_x(t) | y_x(0)) − S(y_x(t), t, x) ‖²
```

which **decomposes into `d_y` independent scalar squared-loss problems**, one per output dimension —
so each is an ordinary LightGBM fit. [verified] Concretely:

* For each training row, draw `R = 30` Monte-Carlo pairs `(t ~ U[0,1], ζ ~ N(0, I))`.
* Build augmented features `h(ζ, t, y) = m(t; y) + ζ·σ(t)`; `t` is an input feature, not a separate
  model per step.
* Fit one LightGBM per output dim with target `−ζ_k`, reparameterising the score as `U/σ(t)` —
  stated to be "key to stabilizing the learning process". [verified]
* **VE-SDE**: `f(y,t) = 0`, `σ(t) = α_min(α_max/α_min)^t`, `α_min = 0.01`, `α_max = 20`.
* Sampling: Euler-Maruyama reverse SDE, **50 steps** by default; **100 samples** per test point for
  CRPS, 50 for the conditional mean. [verified]
* Multivariate `y` handled by independent per-dimension objectives; "average marginal CRPS over each
  dimension" is what gets reported — so **it does not model the joint either**. [verified]
* Admitted limitation: "sampling requires solving an SDE at inference time, which can be
  computationally expensive when generating many samples." [verified]

**Treeffuser is the right diffusion baseline for us if we build one**, not CARD: CPU-trainable,
LightGBM-shaped (we already run XGBoost), 50 steps rather than 1000, 100 samples rather than 1000,
and it works on 2,500 tabular features natively. That is roughly **a 200× cheaper inference** than
CARD's published config. [verified components; the 200× is arithmetic, assumed]

Everything else returned by the search is flow-based *posterior sampling* for inverse problems
(arXiv:2606.22346, 2606.10370) or Neural Bridge Processes (arXiv:2508.07220) — function-space
Bayesian inference, not tabular conditional regression. [verified — titles/abstracts only]

---

## 4. Flow matching / rectified flow for low-dimensional conditional densities

### 4.1 The objective, exactly

**Flow Matching** (Lipman et al., ICLR 2023, arXiv:2210.02747). You want a time-dependent velocity
field `v_t^θ` whose ODE flow transports a simple `p_0` to the target `p_1`. The intractable objective
and its tractable surrogate [verified, both read from the paper]:

```
L_FM(θ)  = E_{t, p_t(x)}                ‖ v_t^θ(x) − u_t(x) ‖²
L_CFM(θ) = E_{t, q(x_1), p_t(x|x_1)}    ‖ v_t^θ(x) − u_t(x|x_1) ‖²
```

**Theorem 2** of the paper: "up to a constant independent of θ, `L_CFM` and `L_FM` are equal. Hence
`∇_θ L_FM(θ) = ∇_θ L_CFM(θ)`." [verified, quoted] That is the whole trick — you regress against a
*per-sample* target you can write down, and get the gradient of the thing you cannot.

**Gaussian conditional path** with boundary conditions `μ_0 = 0, σ_0 = 1, μ_1 = x_1, σ_1 = σ_min`:
`p_t(x|x_1) = N(x | μ_t(x_1), σ_t(x_1)² I)`. [verified]

**The conditional-OT (straight-line) path** — the one to use:

```
μ_t(x_1) = t·x_1
σ_t(x_1) = 1 − (1 − σ_min)·t
u_t(x|x_1) = [ x_1 − (1 − σ_min)·x ] / [ 1 − (1 − σ_min)·t ]
L_CFM(θ) = E_{t, q(x_1), p_0(x_0)} ‖ v_t^θ(ψ_t(x_0)) − (x_1 − (1 − σ_min)·x_0) ‖²
```
[verified, all four read from the paper]

**The practitioner's form**, with independent coupling and `σ_min → 0` (this is the version to
implement; from *Flow Matching Guide and Code*, arXiv:2412.06264) [verified]:

```
sample  X_0 ~ N(0, I),  X_1 ~ q(y | x),  t ~ U[0,1]
set     X_t = (1 − t)·X_0 + t·X_1
loss    L = E ‖ v_t^θ(X_t, cond=x) − (X_1 − X_0) ‖²
sample  x_{t+h} = x_t + h · v_t^θ(x_t, cond=x)        # plain Euler
```

Three lines of training code. The regression target is literally `X_1 − X_0`, a straight-line
velocity — *"a remarkably direct regression target"*. [verified]

**Rectified flow** (Liu, Gong, Liu — ICLR 2023, arXiv:2209.03003) is the same object arrived at
independently [verified]:

```
X_t = t·X_1 + (1 − t)·X_0
min_v  ∫_0^1 E[ ‖ (X_1 − X_0) − v(X_t, t) ‖² ] dt
```

plus **reflow**: re-train on the `(Z_0, Z_1)` pairs the *previous* flow actually produces, which
straightens the trajectories. The paper reports FID 4.85 on CIFAR-10 as a **one-step** model and
states results are good "even with a single Euler discretization step" after reflow. [verified]

### 4.2 Why it needs fewer steps than DDPM

Mechanistically, and this is the part worth understanding rather than quoting:

* Sampling is **integrating an ODE**, not simulating an SDE. Euler's global error on an ODE is
  `O(h · sup‖ẍ‖)` — it is governed by the **curvature of the trajectory**. If the trajectory is a
  straight line, `ẍ = 0` and *one* Euler step is exact. [verified by standard numerical analysis]
* The conditional-OT / linear path makes each *conditional* trajectory exactly straight. The
  *marginal* field is still curved (it has to be, to move mass between distributions), which is why
  one step is not exact out of the box — and that is precisely the gap reflow closes, by replacing
  the coupling with the one the flow induces. [verified — this is the stated motivation of reflow]
* DDPM's reverse process injects noise at every step, so its discretisation error has an
  irreducible stochastic component and the schedule's variance must be resolved finely.
* **Verified numbers, not vibes:** Lipman et al. report FM-with-OT needs "roughly only 60% of the
  NFEs" of diffusion for comparable numerical error [verified, quoted]. Rectified flow reports
  usable one-step samples after reflow [verified]. Treeffuser uses 50 steps for a diffusion SDE on
  tabular data [verified].

### 4.3 For us specifically

A conditional flow-matching head for `y ∈ ℝ` per endpoint is genuinely small: condition on the frozen
2,500-dim features (or on a cheap embedding of them plus `ŷ`), a 3-layer MLP velocity field,
`t` embedded as a Fourier feature, 20 Euler steps, 256 samples. That is **~5,000 network evaluations
per test molecule per endpoint**, which is ~200× cheaper than CARD's published setting and entirely
affordable on an L4. [assumed — arithmetic, not run]

**But §2.10 says the output would be collapsed through a decoder that an 8-row table already
matches.** So this is the right *implementation* of the wrong *idea*. Build it only if the 1-D decode
map is already in and you have measured that a richer predictive helps — which the heteroscedastic
experiment says it will not.

---

## 5. "Thinking" / iterative-refinement diffusion models — is there anything to borrow?

### 5.1 What they actually are

**LLaDA** (arXiv:2502.09992): a masked-diffusion language model at 8B scale. Forward process
progressively masks tokens; a Transformer predicts the masked ones; the objective is a likelihood
lower bound — cross-entropy on masked positions with `1/t` time weighting; generation iteratively
demasks. Competitive with LLaMA3-8B on in-context learning, and beats GPT-4o at reversal poem
completion. [verified, abstract]

**Mercury** (Inception Labs, arXiv:2506.17298): commercial diffusion LLM, Transformer-parameterised,
"predict multiple tokens in parallel". 1109 tok/s (Mini) and 737 tok/s (Small) on H100, claimed
"up to 10x" faster than speed-optimised frontier models at comparable quality. **The report does not
quantify the quality-vs-denoising-steps tradeoff and does not state the step count** — which is the
one number you would need to evaluate the claim. [verified, including the absence]

**Diffusion of Thought** (arXiv:2402.07754): chain-of-thought *inside* a diffusion LM — reasoning
steps diffuse over time rather than being emitted left-to-right, giving a tunable
compute/performance tradeoff. Reports a small diffusion model beating a much larger autoregressive
one on multi-digit multiplication, boolean logic and GSM8K, with "self-correction abilities" and
compatibility with self-consistency decoding. [verified, abstract] The paper does not enumerate
limitations in the abstract; the results are confined to mathematical reasoning. [verified]

### 5.2 Is there a non-language analogue worth borrowing for a small regression head?

**Almost certainly not, and I looked.** An arXiv full-text search for iterative-refinement /
test-time-compute applied to non-language regression heads returned **six papers, none of which
match** — the closest were context optimisation for tabular foundation models (arXiv:2605.12904) and
iterative latent inference for commodity futures (arXiv:2505.06795). [verified — arXiv API query]

The reason is structural, and it is worth stating because it rules out a whole family of ideas:

* **What diffusion LLMs buy is joint, order-free decoding of a high-dimensional discrete object.**
  Autoregressive decoding commits to token 1 before seeing token 500; iterative demasking lets the
  model revise. The gain is *combinatorial* — it comes from `y` having `|V|^n` structure.
* **Our `y` is one real number** (four, marginally, and §1 showed the metric does not couple them).
  There is nothing to revise. "Iterative refinement" of a scalar is a fixed-point iteration, and we
  already have the fixed point: it is `argmin` of a convex one-dimensional risk, solved exactly in
  §2.11 by sorting.
* **The thing that *looks* like borrowed iterative refinement and is actually just correct** is the
  Monte-Carlo decode: draw samples, then *decide*. That is a one-shot decision-theoretic step, not a
  refinement loop, and it is already in this brief.
* The only honest transfer is a framing one: **separate the model from the decision.** Diffusion LLMs
  make that separation explicit (sampler ≠ model), and our stack currently fuses them by training the
  net on a Huberised metric surrogate and reading its output directly as the decision. Splitting
  those two is the whole value of §2 — and it costs nothing.

**Verdict: no mechanism to borrow. Mentioning "diffusion of thought" in a CYP regression programme
would be cargo cult.** Recorded so the idea does not come back.

---

## 6. Cheaper routes to the same predictive distribution

Ordered by cost. For each: does it give a usable `p(y|x)` (or better, `p(L|x)` and `p(U|x)`) to decode
from, and is it cheaper than diffusion?

### 6.1 Quantile heads / Simultaneous Quantile Regression — **yes, the clear winner**

Fit `K` quantile levels with the pinball loss:

```
ρ_τ(y, ŷ) = τ·(y − ŷ)₊ + (1 − τ)·(ŷ − y)₊
```

Two designs:

* **Fixed grid:** `K` output units per endpoint at `τ ∈ {0.05, 0.15, …, 0.95}`, each with its own
  pinball term. Simple, and it is exactly `K` extra output neurons on the existing MLP.
* **SQR** (Tagasovska & Lopez-Paz, NeurIPS 2019, arXiv:1811.00908): one head that takes `τ` as an
  *input*, trained with `τ ~ U[0,1]` resampled per point per minibatch [verified]:

  ```
  f̂ ∈ argmin_f  (1/n) Σ_i  E_{τ ~ U[0,1]} [ ℓ_τ( f(x_i, τ), y_i ) ]
  ```

  The authors state joint estimation "greatly alleviates the problem of crossing quantiles" and that
  it can model "non-Gaussian, skewed, asymmetric, multimodal, and heteroskedastic aleatoric noise".
  [verified, quoted] Note *alleviates*, not *prevents* — see Pitfall P3.

**Cost:** `K` extra output units, one forward pass at inference. Versus CARD's 10⁶ evaluations.
**Gives us what we need:** yes, and in the most directly useful form — **run two sets of quantile
heads, on `lo` and on `hi`**, and (2.6) is solved by bisecting `F̂_L(a) + F̂_U(a) − 1` over the
`2K` knots. No sampling at all. **This is the method to build.**

### 6.2 Gaussian head + deep ensemble — **yes, and we are 80% of the way there already**

Lakshminarayanan et al. (NIPS 2017, arXiv:1612.01474). Per-member heteroscedastic Gaussian NLL
[verified, quoted]:

```
−log p_θ(y|x) = ½·log σ²_θ(x) + (y − μ_θ(x))² / (2σ²_θ(x)) + const
```

Ensemble as a uniform mixture `p(y|x) = M⁻¹ Σ_m p_{θ_m}(y|x)`, moment-matched to a single Gaussian
[verified, quoted]:

```
μ*(x)  = M⁻¹ Σ_m μ_{θ_m}(x)
σ*²(x) = M⁻¹ Σ_m ( σ²_{θ_m}(x) + μ²_{θ_m}(x) ) − μ*²(x)
```

`M = 5` is their recommended default, with gains up to `M = 10`. [verified] **We already run 16–32
members** — which is more than the literature's default and means the epistemic half is free.

**For us:** add `log σ²` outputs to `lo` and `hi` heads and the §2.5(c) closed form
`a* = (σ_U μ_L + σ_L μ_U)/(σ_L + σ_U)` applies with **zero sampling**. That is the single cheapest
*principled* decoder available. Caveat: an ensemble's *spread* is epistemic and systematically
under-estimates aleatoric noise unless each member has its own variance head — using member spread
alone will give over-confident `σ` and a decoder that under-shifts. [verified mechanism; magnitude
assumed]

### 6.3 NGBoost — **yes in principle, no in practice for us**

Duan et al., ICML 2020, arXiv:1910.03225. Gradient boosting where the *distribution parameters* are
the boosting targets, with three swappable modules (base learner / parametric distribution /
scoring rule) and the **natural gradient** as the essential ingredient that "corrects the training
dynamics". Claimed to "match or exceed existing methods for probabilistic prediction". [verified,
abstract] Open-source, Stanford ML Group.

**Why not here:** it is a parametric (typically Normal) conditional distribution fitted by boosting
— so it gives you exactly what §6.2 gives you, with a different fitting algorithm, on a feature
matrix where we already know XGBoost's accuracy (0.4432). It is a *sideways* move. Use it only if the
boosting arm needs a variance estimate and you do not want to wrap it in an ensemble. [assumed]

### 6.4 Conformal prediction — **gives calibrated intervals, not a decodable density**

Split conformal / **CQR** (Romano, Patterson, Candès, NeurIPS 2019, arXiv:1905.03222) [verified, all
equations read]:

```
E_i := max{ q̂_{α_lo}(X_i) − Y_i ,  Y_i − q̂_{α_hi}(X_i) }                              (eq. 14)
Q_{1−α}(E, I_2) := (1−α)(1 + 1/|I_2|)-th empirical quantile of { E_i : i ∈ I_2 }        (eq. 16)
C(X_{n+1}) = [ q̂_{α_lo}(X_{n+1}) − Q_{1−α}(E, I_2) ,
               q̂_{α_hi}(X_{n+1}) + Q_{1−α}(E, I_2) ]                                   (eq. 15)
Theorem 1:  if (X_i, Y_i) are exchangeable, then P{ Y_{n+1} ∈ C(X_{n+1}) } ≥ 1 − α      (eq. 17)
```

Note that CQR's conformity score (eq. 14) **is our metric's error, before the `clip` at zero** — the
same max-of-two-one-sided-distances object. That is a pleasing coincidence and not more than that.

**Does it give us what we need?** Not directly. Conformal gives a *set* with a coverage guarantee at
one level `α`; our decoder needs a *CDF*. You can sweep `α` to get a conformal-calibrated quantile
function (the standard trick), and that is the right use: **conformal as a recalibrator for §6.1's
quantile heads**, not as a competing method. The guarantee is **marginal**, not conditional — it
holds on average over `x`, so a per-molecule decode built on it is not guaranteed per-molecule.
[verified mechanism; the sweep is standard practice, assumed]

The crucial fit with this programme: conformal recalibration is exactly the fix for Pitfall P2
(decoding from a miscalibrated distribution), and it is the one place §2.10's measurement says the
Bayes route could beat the lookup table — because the lookup table's advantage *was* that it
absorbed miscalibration, and conformal removes that advantage by fixing the calibration instead.

### 6.5 Summary

| method | gives a decodable `p(·|x)`? | cost vs diffusion | verdict for us |
|---|---|---|---|
| **quantile heads on `lo`/`hi`** | **yes, directly** | ~0 (K output units) | **build this** |
| Gaussian head + existing ensemble | yes, closed-form decode (2.9) | ~0 | **build this first, it is 1 day** |
| SQR (`τ` as input) | yes | ~0 | nicer if you want all quantiles |
| NGBoost | yes, parametric | ~1/100 | sideways; skip |
| conformal / CQR | intervals, not a density | ~0 | **use as the recalibrator**, not the model |
| Treeffuser | yes, non-parametric | ~1/200 of CARD | the only diffusion worth trying |
| flow matching head | yes, non-parametric | ~1/200 of CARD | research luxury |
| CARD as published | yes, non-parametric | 1× (10⁶ evals/point) | no |

---

## 7. What to actually implement here — ranked, cheapest-that-could-work first

### Rank 1 — the 1-D ST-RAE decode map `δ(ŷ)`. Half a day. Expected **+0.01 to +0.03**.

This is a strict generalisation of the per-endpoint scalar offset already verified (+0.003), and
§2.10 measured it at **−0.0271**, the best of every decoder tried.

```python
def fit_decode_map(y_hat_val, lo_val, hi_val, n_bins=8):
    """Per-endpoint piecewise-constant shift minimising ST-RAE. Fit on VALIDATION only."""
    edges = np.quantile(y_hat_val, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-6; edges[-1] += 1e-6
    b = np.clip(np.digitize(y_hat_val, edges) - 1, 0, n_bins - 1)
    grid = np.arange(-1.2, 1.201, 0.02)
    delta = np.zeros(n_bins)
    for k in range(n_bins):
        s = b == k
        if s.sum() < 20:                      # too thin: inherit the global offset
            continue
        cost = [np.sum(np.clip(lo_val[s] - (y_hat_val[s] + d), 0, None)
                     + np.clip((y_hat_val[s] + d) - hi_val[s], 0, None)) for d in grid]
        delta[k] = grid[int(np.argmin(cost))]
    # MANDATORY: make y + delta(y) non-decreasing, or you have invented a crossing bug
    delta = enforce_monotone_output(edges, delta)
    return edges, delta

def apply_decode_map(y_hat, edges, delta):
    n_bins = len(delta)
    return y_hat + delta[np.clip(np.digitize(y_hat, edges) - 1, 0, n_bins - 1)]
```

`enforce_monotone_output` must guarantee `mid_k + δ_k` is non-decreasing in `k` (isotonic regression
on `mid_k + δ_k`, then subtract `mid_k` back). Without it the map can invert the ranking of two
molecules, which is both wrong and invisible in the aggregate score. See Pitfall P4.

**Order of work:** fit it on the existing cached validation predictions for the xgb arm, the net arm
and the blend **separately**. The net is hinge-trained and the xgb is MAE-trained, so their maps will
differ and the blend's map is neither. Report nested-CV, not the fit split.

### Rank 2 — Gaussian heads on `lo`/`hi` + the closed form. One day. Expected **+0.00 to +0.01 over Rank 1**.

```python
# four extra outputs per endpoint: mu_lo, log_var_lo, mu_hi, log_var_hi
# trained with Gaussian NLL on the (available!) lo and hi columns, masked like the primary heads
def gaussian_nll(mu, logvar, target, mask):
    return (mask * (0.5 * logvar + 0.5 * (target - mu) ** 2 * torch.exp(-logvar))).sum() \
           / mask.sum().clamp(min=1.0)

# decode, exactly, no sampling  (Corollary 2.5(c))
s_lo, s_hi = np.exp(0.5 * logvar_lo), np.exp(0.5 * logvar_hi)
a_star = (s_hi * mu_lo + s_lo * mu_hi) / (s_lo + s_hi)
a_star = np.maximum(a_star, np.minimum(mu_lo, mu_hi))   # guard a degenerate sigma
```

Then **recalibrate before decoding**: isotonic or conformal on validation so that the nominal
quantiles of `N(μ_L, σ_L)` match the empirical ones. §2.10's Gaussian row (−0.0202 vs −0.0257 for
the empirical predictive) is the measured cost of skipping this.

### Rank 3 — quantile heads + exact pooled-median decode. Two days. Expected **+0.00 to +0.02 over Rank 1**, and it is the version that generalises.

```python
TAUS = np.array([.05,.10,.20,.30,.40,.50,.60,.70,.80,.90,.95])

# training: 2*K outputs per endpoint — K quantiles of lo, K of hi
def pinball(pred, target, taus, mask):          # pred: (B, K)
    e = target[:, None] - pred
    return (mask[:, None] * torch.maximum(taus * e, (taus - 1) * e)).sum() \
           / mask.sum().clamp(min=1.0) / len(taus)

# decode: solve  F_L(a) + F_U(a) = 1   (eq. 2.6)  by bisection on the 2K knots
def decode(q_lo, q_hi, taus):                   # q_lo, q_hi: (K,) sorted
    q_lo, q_hi = np.maximum.accumulate(q_lo), np.maximum.accumulate(q_hi)   # kill crossing
    F = lambda q, a: np.interp(a, q, taus, left=0.0, right=1.0)
    g = lambda a: F(q_lo, a) + F(q_hi, a) - 1.0
    a_min, a_max = q_lo[0], q_hi[-1]
    for _ in range(60):
        m = 0.5 * (a_min + a_max)
        if g(m) < 0: a_min = m
        else:        a_max = m
    return 0.5 * (a_min + a_max)
```

Equivalent and more flexible: sample from the quantile function and call
`interval_bayes_act` (§2.11). Use the sampling form if the scale `s` ever becomes per-molecule.

### Rank 4 — Treeffuser as the diffusion baseline, if one is wanted at all. Three days. Expected **+0.000 ± noise**.

If the programme wants a diffusion arm for completeness, use Treeffuser's recipe, not CARD's: VE-SDE,
`σ(t) = 0.01·(2000)^t`, `t` as a LightGBM feature, `R = 30` MC augmentations per row, 50 Euler-Maruyama
steps, 256 samples per test point, then `interval_bayes_act`. Fit it on `lo` and on `hi` directly —
not on `y` — so no interval model is needed. [mechanism verified from the paper; the application is
my proposal]

### Rank 5 — conditional flow matching head. Two days, after Rank 3. Expected **+0.000**.

```python
# per endpoint, per bound: velocity field v(x_emb, z, t) -> R
for batch in loader:
    x1 = batch.bound                                   # lo or hi, scalar
    x0 = torch.randn_like(x1)
    t  = torch.rand_like(x1)
    xt = (1 - t) * x0 + t * x1
    loss = ((v(batch.x, xt, t) - (x1 - x0)) ** 2).mean()

# sampling: 20 Euler steps
z = torch.randn(S)
for i in range(20):
    t = i / 20
    z = z + (1 / 20) * v(x, z, t)
```

Correct, cheap, and almost certainly worth nothing here. Build it only to close out the hypothesis
on the record.

### Rank 6 — CARD as published. **Do not.**

---

## 8. Honest expected value

| method | work | expected ΔMA-ST-RAE | confidence | note |
|---|---|---|---|---|
| **1-D decode map `δ(ŷ)`** | 0.5 d | **−0.010 to −0.030** | medium-high | measured −0.027 in simulation on real intervals; bootstrap sd of the metric is ~0.03, so **needs nested CV to claim it** |
| Gaussian `lo`/`hi` heads + eq. 2.9 | 1 d | −0.005 to −0.020 | medium | subsumed by the map unless recalibrated |
| Quantile heads + pooled-median decode | 2 d | −0.015 to −0.027 | medium | the principled version; ties the map, does not clearly beat it |
| + conformal recalibration of those quantiles | +0.5 d | −0.002 to −0.008 extra | low-medium | this is the only route by which the Bayes decoder should *beat* the table |
| Conditioning the decoder on predictive spread | 1 d | **+0.004 (worse)** | medium | measured; extra estimation variance dominates |
| Dropping `huber_delta` from 1.0 to ~0.05 | 1 h | −0.000 to −0.001 | medium-high | measured: functional mismatch costs 0.0003. Theoretically right, practically noise. Do it for correctness, not for score |
| Treeffuser arm | 3 d | −0.000 ± 0.010 | medium | a defensible negative result if you want one on the record |
| Flow-matching head | 2 d | −0.000 ± 0.010 | medium | same |
| CARD as published | 5 d | −0.000 ± 0.010 | medium-high | plus 10⁶ evals/point |
| 4-dim joint diffusion over `y` | 1 w | **exactly 0 decision-theoretic value** | high | §1: the metric decomposes per endpoint |

**The headline:** metric-aware *decoding* is worth roughly what the blend was worth (−0.027 vs
−0.028), and almost none of that value is in the generative model. It is in a monotone
one-dimensional function fitted to the metric on validation data.

**And the counter-consideration nobody has checked:** the neural arm is *already* trained on the
interval hinge, so it may already be emitting something close to `a*(x)`, in which case the decode
map will find `δ ≈ 0` for it and the gain will live entirely in the MAE-trained xgb arm and get
diluted by the blend. **Measure the map on each arm separately before committing.** This is the most
likely way the predicted gain evaporates.

---

## 9. Pitfalls that make naive implementations silently wrong

**P1 — "Calibrated" simulation of a predictor is not `ŷ = y + noise`.** I hit this in §2.10 and it
inverted a conclusion. `ŷ = y + N(0, σ)` is *over-dispersed*: `E[y|ŷ]` is a shrunk version of `ŷ`, so
any fitted offset table learns a huge shrinkage pattern (+0.7 at low `ŷ`, −0.5 at high) and reports a
spurious −0.09 gain. The fix: `λ = 1 − σ²/Var(y)`, `ŷ = ȳ + λ(y−ȳ) + N(0, λσ²)`, which satisfies
`E[y|ŷ] = ŷ` and `Var(y|ŷ) = σ²`. **Any offline study of a decoder on simulated predictions is
worthless without this.** The tell is that the fitted offsets look like shrinkage. [verified — I made
this error and caught it with a binned-offset control]

**P2 — Decoding from a miscalibrated distribution is worse than not decoding.** The Bayes act is
optimal *for the predictive you feed it*. §2.10: the Gaussian predictive got −0.0202 where the
empirical one got −0.0257 — a quarter of the gain lost to a wrong distributional assumption. A deep
ensemble's spread is epistemic and **systematically too narrow** as an estimate of aleatoric noise;
feeding it to (2.9) under-shifts. Always recalibrate (isotonic on PIT values, or a conformal sweep)
on a split that is neither the training nor the reporting split. [verified]

**P3 — Quantile crossing.** Independently-fitted quantile heads can produce `q̂_0.6 < q̂_0.4`, which
makes `F̂` non-monotone and the bisection in (2.6) return garbage without raising anything. SQR
"greatly alleviates" it but does not prevent it [verified, the authors' own word]. **Always apply
`np.maximum.accumulate` to the quantile vector before decoding**, and separately assert
`q̂_lo(τ) ≤ q̂_hi(τ)` for every `τ` — the bounds have their own ordering constraint which no loss
enforces.

**P4 — A non-monotone decode map silently re-ranks molecules.** `δ(ŷ)` fitted per bin can make
`ŷ + δ(ŷ)` decreasing across a bin boundary, so molecule A predicted below molecule B comes out
above it. ST-RAE will not notice (it is a sum of per-molecule terms), but anything downstream that
ranks — and the playground ranks — will be wrong. **Isotonic-project `mid_k + δ_k` and assert
monotonicity in a test.**

**P5 — Fitting the decoder on the split you report.** `fit_offset` in `scripts/cyp_optimise.py`
already does this correctly (fit on validation, apply blind to test) [verified, read] — keep it that
way as the parameter count grows from 1 to 8 to 2K. With `n ≈ 650` per endpoint per half and a
bootstrap sd of ~0.03 on the metric, an 8-parameter table fitted and reported on the same split will
show a gain that is not there. **Nested CV, and quote the bootstrap CI, exactly as PROGRESS.md did
for the blend.**

**P6 — The interval model must be fitted on `ŷ`, not on `y`, if it is applied at test time.** `d_lo`
and `d_hi` are strongly `y`-dependent (r ≈ −0.85 for width). At test time you do not have `y`. A
model `d̂(y)` fitted and *evaluated* on true `y` is a leak that will look like a large gain. §2.10's
route-2 fits `d̂` as a function of `y` but *evaluates it at sampled `y_j ~ p(y|x)`*, which is
legitimate; fitting `d̂(ŷ)` directly is safer. [verified — this is a close cousin of the auxiliary-
column leak PROGRESS.md already caught at 7×]

**P7 — Monte-Carlo jitter in the decode is not free, but it is cheap.** The pooled median of `2S`
samples has sd `≈ 1/(2 f(a*)√(2S))`. At `S = 64` that is ~0.055 pIC50, comparable to the entire
asymmetry signal (0.02). Use `S ≥ 256`, and **fix the seed** so the decode is reproducible — a
decoder that returns a different answer each run will produce score differences inside the
measurement noise and send someone chasing a phantom.

**P8 — Enforce `l_j ≤ u_j`.** The identity (2.2) that the whole derivation rests on requires it. If
your bound predictions cross (a quantile head for `lo` above the one for `hi`), the loss is no longer
a distance to an interval and the pooled median is no longer the argmin. Clamp:
`l, u = minimum(l, u), maximum(l, u)`. It never happens in the real labels (100% of rows have
`lo ≤ y ≤ hi`) so a crossing is always a model bug. [verified]

**P9 — Do not read a cap into the metric that is not there, or miss one that is.** Everything in §2
turns on the loss being **unbounded**. If the organisers clip the per-molecule relative error (at 1,
say), the Bayes act switches from the pooled median to a **window-smoothed mode** and the algorithm
in §2.11 becomes wrong — not slightly wrong, a different estimator. `metric.py`'s own docstring says
the exact normalisation is not public. **Re-read the rules once, specifically for a cap.**

**P10 — `huber_delta = 1.0` makes the training loss inconsistent for the metric's functional.**
Measured: 99% of molecules sit in the quadratic region, so the net minimises a squared dead-zone
loss whose Bayes act is a mean-type balance, not the metric's median-type one. Measured cost: 0.0003.
**Correct it for hygiene, do not expect score.** It is recorded here so it is not re-discovered as a
promising lead. [verified]

---

## 10. Sources

**Diffusion / flow for regression**
- CARD: Classification and Regression Diffusion Models — Han, Zheng, Zhou, NeurIPS 2022.
  https://arxiv.org/abs/2206.07275 · full text https://ar5iv.labs.arxiv.org/html/2206.07275
- CARD reference implementation (configs read for T, n_z_samples, architecture).
  https://github.com/XzwHan/CARD ·
  https://raw.githubusercontent.com/XzwHan/CARD/main/regression/configs/uci_boston.yml
- Treeffuser: Probabilistic Predictions via Conditional Diffusions with Gradient-Boosted Trees —
  Beltran-Velez et al., NeurIPS 2024. https://arxiv.org/abs/2406.07658 ·
  https://ar5iv.labs.arxiv.org/html/2406.07658

**Flow matching**
- Flow Matching for Generative Modeling — Lipman et al., ICLR 2023.
  https://arxiv.org/abs/2210.02747 · https://ar5iv.labs.arxiv.org/html/2210.02747
- Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow — Liu, Gong,
  Liu, ICLR 2023. https://arxiv.org/abs/2209.03003 · https://ar5iv.labs.arxiv.org/html/2209.03003
- Flow Matching Guide and Code — Lipman et al., 2024. https://arxiv.org/abs/2412.06264 ·
  https://ar5iv.labs.arxiv.org/html/2412.06264

**Diffusion language models**
- LLaDA: Large Language Diffusion Models. https://arxiv.org/abs/2502.09992
- Mercury: Ultra-Fast Language Models Based on Diffusion (Inception Labs).
  https://arxiv.org/abs/2506.17298
- Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models.
  https://arxiv.org/abs/2402.07754
- Negative search result (no non-language regression analogue):
  `http://export.arxiv.org/api/query?search_query=abs:"iterative+refinement"+AND+abs:"regression"+AND+abs:"test-time"`

**Decision theory and scoring rules**
- Making and Evaluating Point Forecasts — Gneiting, JASA 2011. https://arxiv.org/abs/0912.0902
  (consistent scoring functions; Bregman ⇒ mean, generalised piecewise linear ⇒ quantiles;
  elicitability)
- Bayes estimator — loss functions and the corresponding Bayes acts (mean / median / mode /
  asymmetric linear ⇒ quantile). https://en.wikipedia.org/wiki/Bayes_estimator
- The pooled-weighted-median theorem (§2.4), the Gaussian closed form (§2.5c), the bounded-loss
  mode result (§2.8) and all numerical verifications in this document are **derived and checked
  here**, not quoted. The verification scripts were run in the session scratchpad and are not
  committed; §2.11 reproduces the results.

**Cheaper distributional methods**
- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles —
  Lakshminarayanan, Pritzel, Blundell, NIPS 2017. https://arxiv.org/abs/1612.01474 ·
  https://ar5iv.labs.arxiv.org/html/1612.01474
- Single-Model Uncertainties for Deep Learning (SQR + Orthonormal Certificates) —
  Tagasovska & Lopez-Paz, NeurIPS 2019. https://arxiv.org/abs/1811.00908 ·
  https://ar5iv.labs.arxiv.org/html/1811.00908
- NGBoost: Natural Gradient Boosting for Probabilistic Prediction — Duan et al., ICML 2020.
  https://arxiv.org/abs/1910.03225
- Conformalized Quantile Regression — Romano, Patterson, Candès, NeurIPS 2019.
  https://arxiv.org/abs/1905.03222 · https://ar5iv.labs.arxiv.org/html/1905.03222
- A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification —
  Angelopoulos & Bates. https://arxiv.org/abs/2107.07511 (abstract only; the technical details
  cited here come from CQR)

**Local sources (read, not assumed)**
- `talindrew-drugrx/drugrx/cyp/metric.py` — the metric and `interval_loss`
- `talindrew-drugrx/scripts/cyp_train.py` — `build_loss`, the Huberised interval hinge, `huber=1.0`
- `talindrew-drugrx/scripts/cyp_optimise.py` — `fit_offset`, the blend, validation-only fitting
- `docs/drugrx-training/PROGRESS.md` — the bake-off table, the +0.003 offset result, the 0.61 floors
- `hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv` — the
  interval geometry table in §1
