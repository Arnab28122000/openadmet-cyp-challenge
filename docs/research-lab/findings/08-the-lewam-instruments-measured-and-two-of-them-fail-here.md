# 08 — The LeWAM instruments, implemented and pointed at our own data: two of them do not transfer

**Status: Phase 1 closed.** `drugrx/distest.py`, `tests/test_distest.py` (24 gates, all passing),
`scripts/cyp_distshape.py`. Raw numbers in `data/cyp_cache/distshape.json`. Pre-registered in
`PLAN.md` before the run.

Source: *LeWAM: End-to-End World and Action Modeling with JEPAs* (Fu, Siebert, Halicki,
Balestriero — <https://le-wam.github.io/>), which regularises its 192-d latents with

    L_SIGReg = (1/M) Σ_m T( Z u^(m) )

for M random unit directions and T the univariate **Epps–Pulley** statistic against N(0,1)
(their config: 1024 projections, 17 knots), and reports **effective rank** (Roy & Vetterli 2007)
as its anti-collapse readout. The question was whether either object tells us something about our
own data that we did not already have.

## What the implementation itself turned up

The Epps–Pulley statistic was **derived from the BHEP integral rather than copied**, which gave
two independent routes to the same number — the O(n²) closed form and the Gauss–Hermite
quadrature LeJEPA actually uses — and therefore a real cross-check. Two defects fell out of it:

1. **A literal point mass does not have sd 0 in floating point.** `np.full(1000, 4.2).std(ddof=1)`
   is **1.78e-15**, because 4.2 is not representable and the deviations are ULP noise. A
   `sd <= 0` degeneracy guard therefore never fires, the standardisation divides by 1.78e-15, and
   amplified rounding noise is reported as a distance of **0.476 instead of the true ceiling
   0.163** — a 3× error in precisely the case the instrument exists to measure. Guard is now
   relative (`sd <= 1e-12·max(|mean|,1)`), with 2.211 itself in the regression test.
2. **A 17-node Gauss–Hermite quadrature of this integral has an *absolute* accuracy floor,
   ~1e-5 in distance units, not a relative one.** The integrand carries `cos(t·y_j)`, so its
   oscillation frequency is set by `max|y|`; 17 nodes are exact only to polynomial degree 33.
   Measured error is 1e-7–8e-6 and roughly flat in n, while `distance` under the null falls as
   1/n — so relative accuracy *degrades* with batch size, reaching 1.3e-2. On a heavy-tailed
   sample the error is not even monotone in the node count (33 nodes came out worse than 17; 257
   reached machine precision).

   > **CORRECTED 2026-10-11 — see `findings/10`.** This paragraph originally attributed that
   > floor to **LeJEPA's** 17-knot configuration and concluded theirs "is a training signal, not
   > a measurement instrument". **I had measured my own Gauss–Hermite variant, not theirs.**
   > Their reference implementation uses a *trapezoid over [0, 3] with doubled interior weights*
   > and an unnormalised `e^{-t²/2}` window — whose error is **relative, ~2–3%, from truncation**,
   > and therefore shrinks with n rather than forming a floor. The criticism of their
   > configuration is withdrawn. The practical advice is unchanged (quote the exact O(n²) form,
   > which costs under a second at our n), and the real resolution floor at n = 6145 is the
   > **statistical** plug-in bias, ~2e-5, which dominates the quadrature term by 50×.

## Claim 1 — FAILED

> Pre-registered: "the Epps–Pulley `distance` on the combined training target must order the four
> isoforms the same way as the known point-mass share (61 / 29 / 15 / 2 %). PASS = exact rank
> agreement. **If it fails, the instrument does not measure what I claim and Phase 2 does not
> start.**"

| isoform | mass % | `distance`, combined | `distance`, real labels only | Δ |
|---|---|---|---|---|
| CYP1A2 | 29.5 | 0.00641 | 0.01568 | **−0.00927** |
| CYP2C9 | 14.9 | 0.02388 | 0.00246 | +0.02142 |
| CYP2D6 | **60.5** | **0.03940** | 0.01652 | +0.02288 |
| CYP3A4 | 1.9 | 0.01426 | 0.00703 | +0.00723 |

Observed order **CYP2D6 > CYP2C9 > CYP3A4 > CYP1A2** against a share order of
**CYP2D6 > CYP1A2 > CYP2C9 > CYP3A4**; Spearman **+0.40**, not 1.0. **Claim 2 failed too** (the
pseudo-labels were to raise `distance` on all four; they raise it on 3/4 and *lower* it on
CYP1A2).

**The reason is a specification error of mine, and it is identifiable.** `distance` measures
*total* non-Gaussianity. A point mass is one contributor among several, and the four isoforms'
real-label distributions differ **6.7×** in baseline non-Gaussianity before any pseudo-label is
added (0.0025 to 0.0165, the third column). That range swamps the 30× mass-share range. On
CYP1A2 the 29% mass actually made the combined distribution look *more* Gaussian, because it
partially cancelled an existing departure. A cross-isoform *level* comparison cannot isolate the
mass, and I pre-registered one anyway.

**This is the third consecutive specification error of the same family** — finding 06's mechanism
was inverted while its suspect was right, finding 07's falsification condition did not vary the
quantity its claim was about, and now this. The common root is stating a claim about quantity A
and testing it with a statistic that measures A + B, where B is uncontrolled. It is now a
standing trap in `AGENTS.md`.

## A third defect, caught by the result being incoherent

The concentration axis first came out **monotone decreasing on CYP2D6 and monotone *increasing*
on CYP2C9** — opposite signs for the same manipulation. That was my bug, not a finding:
imputing `2.211 − |N(0,σ)|` is the right *shape* for left-censoring, but a half-normal has mean
`−σ√(2/π) ≈ −0.8σ`, so the manipulation moved **location and spread together** — the exact
confound the axis was built to isolate. Re-centring the draw on 2.211 (mean drift now < 1e-12)
makes it monotone on all four:

| isoform | mass % | sd(imputed)/s_y = 0 | → 1.0 | drop | relative |
|---|---|---|---|---|---|
| CYP2D6 | 60.5 | 0.03940 | 0.01454 | **0.02487** | 63.1% |
| CYP1A2 | 29.5 | 0.00641 | 0.00148 | 0.00493 | 77.0% |
| CYP2C9 | 14.9 | 0.02389 | 0.02187 | 0.00202 | 8.4% |
| CYP3A4 | 1.9 | 0.01427 | 0.01368 | 0.00059 | 4.1% |

The **within-isoform paired drop** orders the four exactly by mass share, because the paired
contrast cancels the baseline that defeated claim 1. **Labelled post-hoc and not a rescue:** the
statistic was chosen after seeing the data, four points admit 24 orderings, and one exact match
after trying two statistics is worth about p ≈ 1/12. It is a hypothesis for a future test. Claim
1 failed.

## Effective rank — UNINFORMATIVE, exactly as pre-registered

> Pre-registered: "reported under four normalisations and **not** called a finding if the answer
> flips qualitatively between them."

Feature matrix (6145, 3091):

| normalisation | effective rank | stable rank | erank/d |
|---|---|---|---|
| raw | 61.94 | 1.02 | 0.020 |
| centred | 287.15 | 1.23 | 0.093 |
| standardised | **2240.73** | 45.24 | 0.725 |
| row-L2 | 68.32 | 1.02 | 0.022 |

**36.2× spread.** The caveat fired, so this is recorded as uninformative rather than as the
headline it would otherwise have made ("our features have effective rank 62 of 3091"). That
number is an artefact of a handful of unbounded RDKit descriptor columns — molecular weight,
TPSA — dominating the spectrum of an unstandardised matrix that also contains binary fingerprint
bits. **Roy & Vetterli's Property 2/3 is the formal reason:** effective rank is invariant to
*global* scaling and to unitary transforms, and "the only elementary operation that preserves the
effective rank corresponds to the interchange of two rows or two columns." **Per-column rescaling
is not an erank-preserving operation**, so standardising is not neutral preprocessing — it changes
the quantity being measured.

**The one defensible way to read the number is against a shape-matched control**, which was
measured afterwards: z-scored ours is **2240.73 of min(n,d) = 3091 (0.725)**, against **2860.78
(0.926)** for a true N(0,I) sample of identical shape. Marchenko–Pastur spreads a Gaussian
spectrum at n/d = 1.99, so the ceiling is 0.926, not 1.0 — **comparing an empirical erank to d is
a category error.**

> **CORRECTED 2026-10-11.** The sentence above originally read that a low effective rank here is
> "evidence of REDUNDANCY, not of the representational collapse the SSL literature uses it to
> detect." **The first half is also wrong.** On a matrix of this species, adding 500 exact
> duplicate columns moves RankMe by **−1.1%**, while adding 500 columns of pure iid noise raises
> it by **+23%**; stable rank and NESum go *up* under duplication. These measures are not
> redundancy detectors in either direction — they count directions carrying non-negligible
> variance, and **white noise maximises them**. Effective rank here is evidence of neither
> redundancy nor collapse.

**And the reason it cannot be fixed by picking a normalisation is the interesting part.** In
LeWAM effective rank is meaningful because a learned embedding has **no arbitrary per-coordinate
scale**: every coordinate leaves the same network under the same loss. Our 3091 columns mix
binary bits with unbounded physical quantities in different units, so there is no canonical
normalisation and therefore **no canonical effective rank**. Effective rank is a property of a
learned representation, not of a feature matrix. The tests also show it reads **heavy-tailedness
as collapse** — a full-rank Cauchy cloud scores 17.2 against 63.5 for a Gaussian cloud of
identical shape — and sparse count fingerprints have heavy tails in abundance.

## The decisive practical result: the apparatus is not needed for the job it was borrowed for

Phase 2's manipulation check needs one thing — *did the arm change the imputation's concentration
relative to s_y?* That is `sd(imputed)/s_y`, one line of numpy:

| arm | sd(imputed)/s_y | `distance`, CYP2D6 | `distance`, CYP3A4 |
|---|---|---|---|
| point mass (status quo) | 0.000 | 0.0394 | 0.0143 |
| Tobit-mean (finding 07's fail) | **0.030** | 0.0393 | 0.0143 |
| draw | **0.60** | 0.0145 | 0.0137 |

The scalar is monotone by construction, identical across isoforms, and separates the three arms
by a factor of 20. `distance` separates them by 2.7× on CYP2D6 and **1.04× on CYP3A4**, and its
level is not comparable across isoforms at all. **For this job the one-line scalar is the better
instrument**, and it would have caught finding 07's defect — 0.030 against a target near 1.0 —
before any GPU was spent.

## Verdict on the transfer

| object | verdict |
|---|---|
| Epps–Pulley / SIGReg as a univariate label diagnostic | **does not earn its keep.** A scalar sd ratio does the job better. Still the right tool where the defective direction is *unknown* and the data are *multivariate* — i.e. where LeWAM uses it |
| effective rank on a fixed feature matrix | **fails.** 36× normalisation-dependent (formally: per-column rescaling is not erank-preserving, Roy & Vetterli Prop. 2/3); and it is not a redundancy detector either way — noise maximises it. Only readable against a shape-matched control: ours 0.725 of min(n,d) vs 0.926 for N(0,I) |
| "MSE targets the conditional mean" → mean imputation re-concentrates | **stands, and needs none of this apparatus.** Phase 2 proceeds on the scalar |
| SIGReg as a training regulariser | untouched by Phase 1. Needs an encoder to exist |

Two of the five catalogued objects are now measured and both fail for our use. That is the
honest return on a day's reading, and the instrument build still paid for itself three times
over — two implementation defects and one experimental-design confound, all three caught by
gates and incoherence rather than by inspection.

## The methodological lesson

**Pre-registering the caveat is what made the effective-rank number safe to compute.** The
pre-registration said "not a finding if it flips across normalisations"; it flipped 36×; nothing
was claimed. Had the caveat not been written first, the raw 61.94 was sitting there ready to be
reported as a dramatic redundancy result.

And: **a result that is incoherent across units you expect to agree is a bug signal, not a
finding.** Opposite signs on CYP2D6 and CYP2C9 for the same manipulation was not biology.
