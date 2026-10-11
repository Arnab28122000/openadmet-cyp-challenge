# 10 — SIGReg audit: the random sketch is 280× blind on our data, and three claims I had to correct

**Status: transfer objects 1 and 4 closed on quantitative grounds.** Two Opus research shells
audited LeJEPA's mathematics; every load-bearing number below was **re-measured by me on our own
(6145, 3091) matrix** rather than taken from a shell's report, and three of the shells' claims did
not survive that. Script: `scratchpad/headroom.py`; code `drugrx/distest.py`.

## First, a correction to `findings/08` — I measured the wrong quadrature

`findings/08` reported that "LeJEPA's 17-knot quadrature has an **absolute** accuracy floor,
~1e-5" and concluded it "is a training signal, not a measurement instrument." **I had measured my
own Gauss–Hermite variant, not LeJEPA's.** Their reference implementation
(`MINIMAL.md`, github.com/galilai-group/lejepa) does this:

```python
t = torch.linspace(0, 3, knots); dt = 3 / (knots - 1)
weights = torch.full((knots,), 2 * dt); weights[[0, -1]] = dt   # trapezoid, doubled
window = torch.exp(-t.square() / 2.0)                            # w(t) = e^{-t^2/2}
statistic = (err @ (weights * window)) * N
```

Trapezoid over the **half-line [0, 3]** with interior weights doubled for evenness, an
unnormalised Gaussian window, times N. Not Gauss–Hermite, and not [−5, 5]. **Two research shells
gave contradictory accounts of this** — one had the weight right and the grid wrong, the other had
both wrong — which is why it was read off the source.

Measured against the exact ℝ-integral at the same weight (`epps_pulley_lejepa` vs
`epps_pulley(beta=1)`):

| n | exact | LeJEPA 17-knot | bias | relative |
|---|---|---|---|---|
| 256 | 6.00e-4 | 5.86e-4 | −1.4e-5 | −2.4% |
| 2048 | 6.74e-5 | 6.52e-5 | −2.2e-6 | −3.3% |
| 6145 | 2.64e-5 | 2.60e-5 | −4.2e-7 | −1.6% |

**The error is RELATIVE (~2–3%, from the |t| > 3 truncation), not absolute**, and the absolute
error therefore *shrinks* with n instead of forming a floor. My finding-08 characterisation
described a real property of Gauss–Hermite-17 on an oscillatory integrand and wrongly attributed
it to LeJEPA. **LeJEPA's own quadrature is better than I said it was.** The practical advice is
unchanged — quote the exact form, which costs under a second at our n — but the criticism of their
configuration is withdrawn.

## The real floor is statistical, not numerical — and it is 17× larger

An Opus shell derived the plug-in/V-statistic bias `E[L̂_N] = L + (1/N)∫w(t)(1−|φ_P(t)|²)dt`,
which for the N(0,1) target and LeJEPA's weight is exactly **1.0594/N**. Verified:

| n | predicted 0.4226/n | measured, true-moment standardisation | measured, **sample** standardisation |
|---|---|---|---|
| 512 | 8.25e-4 | 8.81e-4 (**1.07×**) | 2.61e-4 (0.32×) |
| 2048 | 2.06e-4 | 2.23e-4 (**1.08×**) | 6.86e-5 (0.33×) |
| 6145 | 6.88e-5 | 7.43e-5 (**1.08×**) | 2.10e-5 (0.31×) |

**The formula is confirmed** (the 7–8% excess is the higher-order term), and a fact the shell did
not have: **standardising by the sample's own mean and sd shrinks the floor by ~3×**, because it
removes two degrees of freedom from the deviation being measured. This vindicates the module's
design choice of Monte-Carlo calibration (`ep_null`) over published critical values, and it is why
finding 08's measured null at n = 6145 (2.10e-5) sits a factor 3 below the textbook prediction.

So the statistic's resolution floor at our n is ~2e-5, **of which the quadrature contributes
~4e-7 — the statistical term dominates by 50×.** Finding 08 blamed the wrong term.

And the shell's sharpest observation survives: **the bias term `(1/N)∫w(1−|φ|²)` is minimised by
|φ| ≡ 1, i.e. by a point mass.** The plug-in bias therefore pulls in the *collapse* direction —
the exact failure SIGReg exists to prevent — at O(1/N). It does not move the global optimum
(that needs N < 2.59) but the direction is the wrong one.

## The decisive result: SIGReg's random sketch cannot see our data

LeJEPA tests Gaussianity along **random** unit directions. By Diaconis–Freedman's
projection-pursuit asymptotics and Klartag's CLT for convex sets, most 1-D projections of a
high-dimensional distribution are close to Gaussian *whether or not the distribution is* — so a
random sketch loses power as d grows. Measured on our matrix, 256 random directions against 256
coordinate axes:

| matrix | random-direction mean | random max | axis-aligned mean | ratio |
|---|---|---|---|---|
| our features, z-scored | 0.00046 | 0.00291 | **0.12912** | **280×** |
| true N(0,I), same shape | 0.00002 | 0.00010 | 0.00002 | **1×** |

**The non-Gaussianity of our features is 280× larger along coordinate axes than along random
directions, and the Gaussian control returns exactly 1× — so the effect is real and not an
artefact of the estimator.** Count fingerprints have an atom at zero in every coordinate (13.8%
non-zero overall); a random dense combination of 3091 such coordinates is nearly Gaussian by the
CLT, and the atom — the entire structure — is invisible to it.

This is the clean kill on transfer object 1 for multivariate use, and it **retrospectively
explains finding 08's negative result**: the one place the statistic did work there was on 1-D
label distributions, where there is no sketch.

## The claim I could not confirm, and the shell overreached

A shell's headline was that our features are *"more isotropic, by LeJEPA's own figure of merit,
than a genuine isotropic Gaussian sample of the same shape"* — i.e. no headroom at all. LeJEPA's
Theorem 3.3 inflates the kNN/kernel integrated squared bias over the isotropic optimum by the
AM/HM ratio `(Σλ)(Σ1/λ)/K²`. Re-measured:

| matrix | tol 1e-8 | tol 1e-4 | tol 1e-2 |
|---|---|---|---|
| raw features | 7840× | 40.5× | 9.35× |
| z-scored | 674× | 10.67× | **1.69×** |
| **true N(0,I), same shape** | **2.01×** | **2.01×** | **2.01×** |

| top-k PCs, z-scored | 16 | 32 | 64 | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|
| **ours** | 1.250 | 1.363 | 1.436 | 1.481 | 1.485 | 1.633 |
| N(0,I) control | 1.000 | 1.000 | 1.001 | 1.002 | 1.006 | 1.054 |

**The claim holds at exactly one truncation tolerance (1e-2) and fails at every other, including
every PCA truncation.** The reason is visible in the table: the Gaussian control's inflation is
**2.01× at every tolerance** because it has no near-null directions, while ours swings 1.69× →
674× **entirely according to how many near-null directions are retained**. The comparison is only
well-posed at matched K — and at matched K **ours is uniformly worse**, 1.25–1.63 against
1.00–1.05.

So: **there is headroom, it is small (≈1.5× in integrated squared bias, ≈1.2× in RMS bias), and
the "already better than Gaussian" framing was an artefact of a tolerance choice.** Recorded
because a shell's conclusion that happens to agree with mine is still a claim to check — this one
was 5× off at tol 1e-4 and 335× off at 1e-8.

## Why object 4 closes anyway, on the two arguments that do survive

The headroom result does not matter, because the theorem does not reach our regime:

1. **The asymptotic regime is unreachable.** Theorem 3.3 needs `N·r₀^K → ∞` with `r₀ → 0`. At
   n = 6145 and r₀ = 0.5 that caps **K ≤ 12.6**, and ≤ 5.9 for 100 neighbours. At LeWAM's own
   K = 192 you would need N ≈ 10^57.8. The bias term isotropy minimises is simply not the leading
   error term at any embedding width we would use.
2. **Isotropy of the embedding is *derived from assumed isotropy of the task ensemble*.**
   `E[∇η∇ηᵗ] = τ_g²·I` is the task-agnosticism axiom. We have **four known, fixed, correlated
   endpoints** — the axiom is false by construction, and under a known task the optimum is
   anisotropic and task-aligned. A third party has now confirmed this empirically in the
   world-model setting: *Anisotropic Representations Improve Planning in JEPA World Models*
   (arXiv [2609.37441](https://arxiv.org/abs/2609.37441)) replaces the fixed isotropic target with
   a learnable diagonal covariance under LeJEPA's own fixed-trace constraint and "improves planning
   success over LeWorldModel in all four" environments.

And the mechanism is already measured in our own history: fine-tuned CheMeleon scored 0.5179
macro ST-RAE, but **embeddings read back out of that same fine-tuned encoder scored 0.5862/0.5886
≈ frozen ECFP**. The gain was in the loss, not the representation.

## What the audit says about LeWAM's own evidence for SIGReg

Worth recording, because it weakens the source for the object being transferred:

- **SIGReg's planning gain is one task of four.** Across LeWAM's Table 7: Push-T +14.7 vs +2.7,
  Scene +5.3 vs +2.0, **Puzzle +4.7 vs +8.0 — SIGReg loses** — Cube +0.7 vs 0.0. Finding 08 quoted
  the Push-T figure from the project page without noticing it is the best of four.
- **LeWAM without SIGReg does not collapse.** Effective rank falls 42.90 → 10.94 — a 4× drop —
  while object R² moves only 0.90 → 0.88 and proprio 0.86 → 0.85. **A 4× change in effective rank
  bought 0.02 R².** The total-collapse row (R² = 0.00) is the *predecessor* LeWM with SIGReg
  removed, where it is the only non-degeneracy term. That is independent evidence for finding 08's
  conclusion that effective rank is a poor proxy for representation quality.
- **"Single hyperparameter" does not hold as stated.** At V = 8, λ = 0.001 → 76.67 and
  λ = 0.05 → 84.32, a 7.7 pp swing; the optimum moves with view count and with batch size (the
  statistic carries a factor N, so λ is batch-coupled — independently corroborated by arXiv
  [2610.10299](https://arxiv.org/abs/2610.10299), which derives a critical SIGReg weight that
  *decreases* with batch size). And `V_g = 1` vs `2` at V = 4 is **53.06 vs 72.26** — a 19.2 pp
  swing, the largest knob in the paper. LeJEPA removes the *anti-collapse* heuristics
  (stop-gradient, EMA, teacher, whitening, negatives); it leaves the *invariance-prior* heuristics
  — DINO's full augmentation recipe — carrying most of the signal. The defensible claim is "one
  anti-collapse trade-off hyperparameter with a wide basin, replacing DINOv2's
  momentum/temperature/centering/sharpening schedules", which is still a real simplification.
- **LeJEPA has no peer review.** Both shells searched OpenReview independently: no ICLR 2026,
  ICML 2026 or ICLR 2027 page. Its reception is arXiv follow-ups and a GitHub tracker, and at
  least four ICLR 2027 submissions attack the isotropic-Gaussian target itself (feature
  suppression, λ-JEPA, SPHERE-JEPA, UR-JEPA), two of them reporting they beat it on ImageNet-1k.
  **No pretrained weights were ever released** (issue #3, open since 2025-11-13), and an ImageNet
  reproduction diverged (issue #25).
- **A name collision to avoid:** there is a second, unrelated *LeWAM* — arXiv
  [2610.12407](https://arxiv.org/abs/2610.12407), Hegde et al., "A JEPA World Action Model with
  Diffusion-Steering-Based MPC". Ours is the Fu/Siebert/Halicki/Balestriero one, whose arXiv
  posting is still pending.

## What survives the whole audit

**One prescription, and it is not the one the paper markets.** The theory's own asymptotics bind
only for `K ≲ 12` at our n — so if a learned molecular embedding were ever to be probed by
kNN or a kernel here, LeJEPA's actual advice is an **8–16 dimensional isotropic embedding**, not a
2048-dimensional one. That is testable and nobody has tested it. It is also consistent with the
molecular-LeJEPA paper's own observation (`findings/09`) that **"embeddings saturate at ~16–32
effective dimensions, whereas Morgan fingerprints improve to 1024 bits."** Noted, not opened.

**And the general principle, which costs nothing and is the real yield:** distribution matching as
a differentiable loss is cheap and well-conditioned **in one dimension**, where there is no sketch
and no curse of dimensionality. The 1-D law in this pipeline that is both changeable and *measured*
to carry value is **the law of the predicted score, not the law of an embedding** — affine
placement is worth 0.089–0.094 macro ST-RAE on 8/8 seeds, while every representation lever
measured here is 0.000.

---

## ADDENDUM 2026-10-12 — adversarial verification. **The headline word "blind" is withdrawn**, and two defects in my own code are fixed

An adversarial verifier re-derived every claim independently (own algebra, own quadrature, own
Monte Carlo, LeJEPA's source pulled from GitHub). **The mathematics survives: 1a, 1b, 1d, 2a, 2c,
3a's numbers, 3b's mechanism, 4a, 4b, 4c's direction, 4d and my retraction of the "more isotropic"
claim all confirmed**, several now with closed forms I had only measured. Three things do not.

### 1. "Blind" is the wrong word, and this is the correction that matters

I wrote that SIGReg's random sketch "cannot see the structure in our data." **That is false.** Of
1024 random directions on the real matrix, against a 4000-draw null:

| | |
|---|---|
| exceed null q95 | **98.93%** |
| exceed null q99 | 97.85% |
| exceed the **maximum of 4000 null draws** | **89.94%** |
| mean random distance vs null mean | **20.0×** |
| the quantity SIGReg actually minimises (the mean over directions) | **z ≈ 173σ** |

A random sketch **rejects normality in essentially every direction, with overwhelming
significance.** What collapses by 280× is the **effect size** — and therefore the gradient a
regulariser receives — not the detectability.

**The defensible claim, which replaces the headline:** *SIGReg rejects normality in ~99% of random
directions but reports an effect size 280× below the axis-aligned truth, so its gradient is ~280×
weaker than a coordinate-aware penalty's and it cannot rank or localise which directions are bad.*
That is still a strong, useful statement about why it is the wrong instrument here. "Blind" was an
overstatement that a referee would have broken immediately.

**And a related caveat I should have written:** the supremum over the sphere **is** ≥ 0.169,
because coordinate axes *are* unit vectors. **LeJEPA's sup-consistency theorem is therefore
untouched.** What fails is *Monte-Carlo estimation of the sup from M random draws* — 16,384 draws
reached a max of 0.0028 against an axis median of 0.159, and a Pareto tail fit extrapolates to
~1e14 draws. Phrase it as an estimation failure, not a theoretical one.

### 2. The 280× is normalisation-dependent by 6.5×, which I flagged as a question and never answered

| matrix | random mean | axis mean | **ratio** |
|---|---|---|---|
| raw | 2.982e-3 | 0.13075 | **43.8×** |
| centred | 2.982e-3 | 0.13075 | 43.8× |
| row-L2 | 2.554e-3 | 0.13034 | 51.0× |
| **z-scored** (what I quoted) | 4.605e-4 | 0.13075 | **283.9×** |
| z-scored, **LeJEPA's own unstandardised statistic** | 7.30e-4 | 0.13075 | **179.1×** |

The axis mean is essentially invariant; **all** the movement is in the random-direction number,
because on the raw matrix a few high-variance descriptor columns dominate every projection and it
inherits one column's non-Gaussianity. The conclusion survives everywhere — 44× is still enormous
— but **the figure must be quoted as "on the z-scored matrix, with each projection standardised."**

**The dimension mechanism is confirmed**, which was the attack I most wanted run: ratio ∝ **d^0.917**,
monotone 2.0× → 280.7× across d = 8 → 3065, no saturation. My stated Diaconis–Freedman reading was
right. And two attacks *backfired* in my favour: the axis distribution is **left**-skewed, so the
**median** ratio is **399×**, and a column-shuffled control (identical marginals and sparsity, all
inter-column dependence destroyed) gives **2113×** — so sparsity is not what keeps random
projections non-Gaussian; **inter-column dependence is**.

### 3. Two defects in `drugrx/distest.py`, both now fixed

**(a) `sigreg` defaulted to `standardise=True`, and LeJEPA does not standardise.** Standardising
each projection removes its location and scale, so the statistic sees only **shape** — and the
thing SIGReg regularises is largely a **scale** defect. Measured on one collapsed cloud:

| scale | `standardise=True` | `standardise=False` |
|---|---|---|
| 0.1 | 0.000058 | **0.156892** |
| 1.0 | 0.000058 | 0.000204 |
| 10.0 | 0.000058 | **0.277573** |

The shape-only statistic is **identical across three orders of magnitude of collapse**. Default is
now `standardise=False` with `n_proj=256` (LeJEPA's own `MINIMAL.md` uses 256, not the 1024 I
attributed to "their config").

**And the gate that claimed to prove otherwise asserted something mathematically false.**
`randn(n,4) @ randn(4,dim)` has rows `A'g` with `g ~ N(0,I₄)` — every row is **exactly Gaussian**
with singular covariance, so **every 1-D projection is exactly Gaussian** and a shape statistic
*must* be blind to it. My assertion `b["mean"] > a["mean"]` asked for a difference that cannot
exist; it held on **8 of 20** data seeds. Seed 13 was luck. Rewritten to assert what is true: the
unstandardised statistic separates them by >10×, the standardised one cannot.

**(b) `sigreg` still contained the exact `sd <= 0` bug this finding is partly about**, 60 lines
below the comment documenting it. Consequence: `sigreg(np.full((512,16), 0.0))` **raised** while
`sigreg(np.full((512,16), 4.2))` returned a number from 23 of 32 projections — same collapsed
input, two answers, decided by float representability. Both now return the same value and report
`n_degenerate`.

**(c) And the guard had a false-positive direction with zero test coverage.** The test was
`sd <= 1e-12·max(|mean|,1)`, so `1e12 + 0.1·randn(4000)` — 4000 distinct float64s from a clean
Gaussian — was reported as `degenerate=True, distance=0.163137`. Now the guard is on **ptp against
the local floating-point spacing**, which is scale-aware: `np.full(n,v)` has ptp exactly 0 for
every `v`, while a real Gaussian at any magnitude is many ULPs wide. Verified in both directions.

### 4. Two closed forms I measured instead of deriving

- **The sample-standardised null level is exactly `(1 − √3/2)/N = 0.1339746/N`**, with the
  true-moment level `(1 − 1/√3)/N = 0.4226497/N`, so standardising costs an **exact 3.154701×**
  shrink. I had reported "~3×, measured."
- **My Gaussian control was measuring that floor, not isotropy.** 0.1339746/6145 = **2.18e-5**
  against my measured 2.12e-5. The control is the right control, but it must be *labelled* as the
  plug-in floor so nobody reads "a ratio of 1×" as evidence about isotropy.
- **The Gaussian AM/HM value 2.01 is exact**: `1/(1 − d/n) = 2.0121` at d/n = 0.50301. And
  `erank/d = 0.9255` follows from Marchenko–Pastur, so 2860.78 is a **law of the shape**, not a
  sample artefact — a stronger statement than I made.
- **The source of the retracted "more isotropic" claim is identified**: it is the tol = 1e-2 row,
  where ours reads 1.69 against the Gaussian's 2.01 — but ours keeps only **K = 1142 of 3091**
  eigenvalues there while the Gaussian keeps all 3091. At matched K = 1142 the Gaussian is ≈1.07,
  so ours is **1.58× worse**. A mismatched-K artefact, and my matched-K fix was exactly right.

### 5. And a reasoning error of mine, not a number error

I reported the plug-in bias ratio as "**1.07–1.08×** of prediction — the formula is confirmed." The
formula is **exact** under true-moment standardisation, so the only admissible answer is 1.000, and
**a 7% deviation from an exact formula is a failed check, not a passed one.** Re-measured with
4000/1500/800 reps instead of 200: **1.010 / 0.997 / 0.936**, and 200-rep sub-blocks span
**0.836–1.100**. My three readings were noise. The lesson is not about the number — it is that I
accepted a 7% miss as confirmation because it was close, when the hypothesis admitted no slack.

### 6. The test suite: 6 of 30 gates did not test what they claimed

Beyond the two rewritten above: `test_null_is_calibrated` compared `ep_null` to a loop computing
**the same statistic**, so replacing the statistic with `7.3·distance + 0.9` still passed —
now anchored to the external closed form. The heavy-tail quadrature gate passes **52%** of fresh
seeds and its `err257 < 1e-9` assertion reaches 1.6e-5 elsewhere. The point-mass regression test
had **3 of 5 parameters decorative** (the old guard *did* fire for 0.0, −17.5, 1e6). Several
tolerances were 15–200× loose. And the monotonicity gate — "GATE 3, the load-bearing one" —
**fails outright when the point mass sits at the bulk's mean**, which is the case closest to "a
concentrated imputation you cannot see." Suite now 39 tests, 223 passing repo-wide.
