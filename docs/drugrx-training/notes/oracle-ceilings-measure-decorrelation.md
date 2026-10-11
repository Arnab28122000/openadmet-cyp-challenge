# A reported oracle ceiling measures error decorrelation, not complementarity

**Claim.** If you take two models, compute per-example which one was closer to the
truth, and report the score of that hindsight selection as "headroom" for routing,
mixture-of-experts or model selection, the number you get is largely determined by
how decorrelated the two models' errors are — and is therefore **positive even when
the two models are equivalent**. The useful question is not whether the ceiling
exceeds zero, but whether it exceeds what two interchangeable models produce from
seed noise alone.

We measured this on 36 trained models and the ceiling was **entirely** accounted
for by decorrelation, with the residual attributable to the models actually
differing coming out **signed against** complementarity.

## Why a positive gap is arithmetic

Oracle selection evaluates `min(e₁, e₂)` per example. For two draws from the same
distribution the minimum has a strictly smaller expectation than either draw. For
bivariate-normal errors with correlation ρ,

```
E[min(X, Y)] = μ − σ · sqrt((1 − ρ) / π)
```

so the apparent gain scales as **√(1 − ρ)** and vanishes only at ρ = 1. Nothing in
that expression refers to the models being different in kind. Two checkpoints of one
training run, differing only in seed, produce a gap.

## What we measured

36 models on one shared validation split (4 primary regression endpoints, 922 rows),
spanning two training objectives and four auxiliary-head configurations. Every one of
the **C(36,2) = 630** pairs gives a (ρ, gap) point, where ρ is the mean per-endpoint
correlation of the two models' per-example errors and the gap is the oracle score
minus the **better** of the two members — an oracle that cannot beat its own best
member buys nothing.

| ρ band | n | mean ρ | mean gap | same-config | cross-config |
|---|---|---|---|---|---|
| 0.743–0.851 | 79 | 0.825 | −0.1790 | −0.1758 | −0.1797 |
| 0.851–0.881 | 79 | 0.869 | −0.1516 | −0.1571 | −0.1513 |
| 0.881–0.895 | 78 | 0.888 | −0.1367 | −0.1512 | −0.1357 |
| 0.895–0.909 | 79 | 0.902 | −0.1292 | −0.1327 | −0.1287 |
| 0.909–0.921 | 79 | 0.916 | −0.1189 | −0.1216 | −0.1183 |
| 0.921–0.936 | 78 | 0.929 | −0.1080 | −0.1123 | −0.1069 |
| 0.936–0.953 | 79 | 0.945 | −0.0963 | −0.0989 | −0.0955 |
| 0.953–0.988 | 79 | 0.967 | −0.0732 | −0.0736 | −0.0729 |

Fitting both an empirical and the theoretical form:

| form | intercept | slope | R² |
|---|---|---|---|
| `gap ~ ρ` | −0.7824 | +0.7274 | 0.9316 |
| **`gap ~ √(1 − ρ)`** | **+0.0095** | **−0.4456** | **0.9431** |

**The theoretical form fits better, and its intercept is small where the derivation
requires zero**: at ρ = 1 the two models make identical errors and the gap must vanish.
The linear fit's intercept of −0.78 at ρ = 0 has no such interpretation. One number — ρ
— explains 94% of the variance in the "headroom".

**The intercept is small but not zero**, and we state that rather than rounding it: a
run-level bootstrap gives **+0.0095, 95% CI [+0.0042, +0.0149]**, which excludes zero.
It is 5–13% of the gaps being fitted. Mild misspecification is expected — the Gaussian
derivation assumes symmetric unbounded errors and this metric's errors are non-negative
with a point mass at exactly zero — so we read the form as the right mechanism with an
approximation error, not as exact.

### Standard errors: 630 pairs are not 630 observations

Each of the 36 runs appears in 35 pairs, so the pairs are heavily dependent and the
effective sample size is nearer 36. Naive OLS errors are optimistic. A **cluster
bootstrap resampling the runs** (`cyp_oracle_bootstrap.py`, B=1000, self-pairs
excluded) gives:

| quantity | point | bootstrap sd | 95% CI |
|---|---|---|---|
| slope | −0.4456 | 0.0098 | [−0.4637, −0.4255] |
| intercept | +0.0095 | 0.0027 | [+0.0042, +0.0149] |
| R² | 0.9431 | 0.0119 | [+0.9158, +0.9612] |
| residual difference | −0.00395 | **0.00148** | [−0.00680, −0.00111] |

The residual's standard error is **1.8× the naive 0.00081**, taking its t from −4.89 to
**−2.66**. R² and the sign are far outside any plausible correction; the error bar on
the residual is not, so the clustered figure is the one to quote.

## The part that would have been evidence

After conditioning on ρ, do pairs that differ in configuration beat the fitted curve by
more than pairs differing only in seed? That excess is the only part that could
indicate region-specific competence.

```
same-config  n=114   mean residual −0.00323
cross-config n=516   mean residual +0.00071
difference          −0.00395   (clustered se 0.00148, t = −2.66)
```

**The sign convention, stated explicitly, because a reader applying this will apply it
in the direction the sign implies.** Gaps are negative (an oracle improves on its better
member), so a *more negative* residual means the pair beat the curve — a **larger
improvement in magnitude**. Same-config pairs sit at −0.00323, i.e. they achieve
**larger** oracle improvements than the curve predicts; cross-config pairs sit at
+0.00071, i.e. **smaller** ones.

**What complementarity would have required:** if differing in configuration conferred
region-specific competence, cross-config pairs would beat the curve by *more* than
same-config pairs — a **negative** cross-config residual, larger in magnitude than the
same-config one. They showed the **opposite**: cross-config pairs underperform the curve
while same-config pairs exceed it. So the only candidate evidence for complementarity is
present, measurable, and points away from it.

Magnitude 0.0039 against gaps of 0.07–0.18 — about 3% — and it survives clustering at
t = −2.66 with 99.8% of bootstrap draws keeping the sign.

Two independent measurements on the same models agree. A realisable router (switch
models on the predicted value) captured **2%** of the ceiling and sat inside noise.
And averaging across configurations added only 0.002–0.005 over averaging two models
of the same configuration — our ensembling gain tracked **single-model variance**, not
configuration identity: the noisiest configuration gained most (−0.0241) and the
tightest least (−0.0076).

## The ceiling and the ensembling gain are one fact seen twice

Both come from errors being partly independent. Averaging is the **achievable**
exploitation of that independence and needs no label. Selection appears to exploit far
more but needs the label to know which error was smaller. On our models the achievable
part was **0.0094** and the hindsight part **0.1056**; the difference is, almost
exactly, what is visible only after the fact.

## What to do instead

1. **Compute the same-config null.** Two seeds of one configuration, same statistic,
   no extra training. Report it beside the ceiling.
2. **Or read ρ off the curve.** With errors correlating at ρ, expect roughly
   `0.45·√(1 − ρ)` in these units for free. Only the excess needs explaining.
3. **Condition on ρ rather than matching on it.** ρ is often *downstream* of the
   treatment — our noisier configurations had systematically lower ρ — so a single
   matched comparison conditions on a variable the treatment moves. Fitting across the
   range separates "the models are equivalent" from "the models differ in internal
   variance".
4. **If the excess is real, test the realisable selector before building it.** The
   gating signal has to exist at inference. Ours did not.

## Scope and limits

- **One dataset, one task family, one metric** (a soft-threshold relative absolute
  error: a hinge that charges zero inside a credible interval). The √(1−ρ) *form* is
  general; the slope of 0.4456 is specific to these units.
- The Gaussian derivation should not strictly apply — this metric's errors are
  non-negative with a mass at exactly zero — yet the form still fits at R² = 0.94.
  We report that as measured, not as explained.
- **All 36 models share one architecture family.** We did not test genuinely
  heterogeneous models, where real complementarity is more plausible. The claim is
  about the *null* being omitted, not that complementarity never exists.
- Gaps are on validation, paired, single split. The split is shared across all 36
  models, which is what makes the pairing valid.

## Reproducing it

`talindrew-drugrx/scripts/cyp_oracle_rho.py` over the stored per-run prediction files
(`data/cyp_cache/cyp_loss/`, `data/cyp_cache/cyp_aux/`: 36 `.json` scores and 36
`.npz` prediction arrays). No GPU. The same-config two-point version is
`cyp_oracle_null.py`; the routing and averaging checks are `cyp_route_ceiling.py`,
`cyp_ensemble_control.py`, `cyp_seed_curve.py`, and `cyp_oracle_bootstrap.py` for
the clustered standard errors.

Every figure above is produced by those scripts from those stored artifacts.

## How we got it wrong first

Worth recording, because the error is the one this note is about. We computed the
cross-config ceiling, got −0.1077 at t = −17.60, and wrote it up as "the largest
unexploited signal this programme has measured". The t-statistic was real and the
null it rejected was one nobody held. Then, asked for a control, we matched on ρ with
a *single* same-config pair whose errors correlated at 0.971 against the cross-config
0.935 — understating the null and manufacturing 35% of apparent headroom — before
noticing that ρ had to be conditioned on across its range rather than matched at a
point.
