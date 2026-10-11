# 11 — 2.211 is the **median of a measured distribution**, not a censoring bound. Half the rows it is applied to lie above it.

**Status: this refutes the premise of `findings/07`, of its explicitly-left-open follow-up, and of
Phase 2 of the LeWAM transfer plan.** Surfaced by an Opus research shell; **every load-bearing
number below was re-measured independently** from our own release
(`scratchpad/verify_latent.py`) before anything was written down.

## The measurement

`drugrx/cyp/screen.py:allr_floor` computes the floor pseudo-label as the **median** over every
compound carrying *both* a dose-response curve *and* a 50 µM readout showing no inhibition
(`r ≥ INACTIVE_R = 0.9`). That population is measured, and it is right there in the release:

```
allr_floor() -> {'y': 2.21113, 'lo': 1.06603, 'hi': 3.52222, 'n': 303}
```

| | value |
|---|---|
| n (dual-labelled) | **303** |
| per isoform | CYP3A4 **250**, CYP1A2 **42**, CYP2C9 **11**, **CYP2D6 0** |
| mean | **2.3822** |
| **median** | **2.2111** ← this *is* the constant |
| sd | **0.4435** |
| 1.4826·MAD | 0.3144 |
| min / max | **1.906** / 4.175 |
| q01 / q10 / q25 | 1.918 / 1.977 / 2.053 |
| q75 / q90 / q95 / q99 | 2.595 / 3.100 / 3.309 / 3.692 |
| **P(y\* > 2.211)** | **0.4983** |
| P(y\* > 3.0) | 0.1287 |

**49.8% of screen-inactive compounds have a true pIC50 *above* 2.211.** The distribution is
**right-skewed with a hard floor near 1.9** — there is essentially nothing below it (q01 = 1.918,
min = 1.906). That is the *opposite shape* from a left tail.

> So the screen label is not a censoring bound. It is a **point estimate of a location with a
> measured dispersion of 0.4435**, and the pipeline asserts that dispersion is zero.

`screen.py` computes this distribution and keeps only its median. The dispersion was available all
along.

## What this invalidates

**`findings/07`'s whole frame.** It opens: *"a screen says the compound is inactive, which means
`y ≤ 2.211`, not `y = 2.211`. That is a textbook left-censored regression, so use the textbook
estimator."* **The first clause is false for half the rows.** This is not censoring; it is
**coarsening** — a labelling rule *assigns* c while the latent value sits on either side of it.
Powell's censored LAD and the whole `y_obs = max(c, y*)` family do not apply at all.

**And fixing the σ bug would have made the arm actively harmful.** `findings/07`'s addendum
(written earlier today) established that `cyp_censored_floor.py:136` took σ from an in-sample GBT
residual, 5–7× too small. Correcting it:

| σ | E[y\*\|y\*≤c] at μ = c | vs the measured population |
|---|---|---|
| 0.084 — the bug | **2.1441** | inside support. A near no-op |
| 0.4435 — measured σ_pop | **1.8573** | **below the measured minimum (1.906)** |
| 0.9161 — held-out σ_out | **1.4802** | far below the entire support |

**A correctly specified left-censored Tobit imputes values the data says are impossible.** The
in-sample σ bug is the only reason that arm was merely uninformative rather than damaging.

> **This corrects my own addendum from earlier today**, which closed with "any future censored arm
> takes σ from OOB." That advice is right about σ estimation in general and **wrong here**: no σ
> rescues the censoring family, because the censoring model is the thing that is false. Do **not**
> fix the σ and re-run. The bug was protecting us.

**CYP2D6 has zero dual-labelled compounds.** The promotion rule (`DRC iff log2FC < −0.6`) makes
"has a curve" and "is screen-inactive" **disjoint by construction** on CYP2D6. So
P(y\*\|x, inactive) is **unidentified** from CYP2D6's own data — positivity fails — and every σ
plugged into a CYP2D6 imputation is an extrapolation across a selection boundary, transferred from
CYP3A4 (250 of the 303). **That is the isoform carrying 60.5% of the mass.** No imputation scheme
can fix an unidentified conditional.

## The de-concentration question is answered: real mechanism, dead intervention

The shell ran the decisive test — 4 seeds, 288 production fits, ~40 CPU-min, location held
*exactly* at 2.211 per member (not in expectation), bootstrap index sets pre-generated and reused
across arms so every contrast is paired on the resample as well as the seed. CYP2D6:

| arm | imputed sd | r(σ̂,p) | **Δ r(σ̂,p)** | **Δρ** | wins |
|---|---|---|---|---|---|
| `full` (baseline) | 0.0000 | +0.324 | — | — | — |
| `draw10` σ=0.10 | 0.100 | +0.304 | −0.019 | −0.0019 | 2/4 |
| `tobit44` **the literal proposal** | 0.256 | +0.264 | −0.060 | **+0.0003** | 2/4 |
| `shared44` one *shared* draw | 0.436 | +0.239 | −0.085 | +0.0044 | 3/4 |
| `emp44` nonparametric per-member | 0.441 | +0.243 | −0.081 | −0.0018 | 1/4 |
| `draw44` per-member σ=0.4435 | 0.446 | +0.230 | −0.094 | +0.0029 | 3/4 |
| `draw80` per-member σ=0.80 | 0.795 | +0.023 | **−0.301** | +0.0038 | 3/4 |

CYP3A4 negative control (1.9% mass): Δr(σ̂,p) −0.016, Δρ +0.0006. The control behaves.

**The mechanism is confirmed beyond reasonable doubt** — Δr(σ̂,p) is *perfectly monotone* in
imputed sd with no inversions, 0.000 → −0.019 → −0.060 → −0.085/−0.081/−0.094 → −0.301. Finding 06
was right about the cause.

**Three results that kill the intervention anyway:**

1. **The repair is unreachable at the honest σ.** At the measured 0.4435 the artefact only falls
   +0.324 → +0.230, still positive on 4/4 seeds — 29% of the repair. Reaching zero needs imputed
   sd ≈ 0.80, which is **1.8× the measured latent sd**. *Repairing the diagnostic requires
   injecting more dispersion than the data contains.* The arm that works is not an imputation, it
   is variance injection — σ̂ is a dial, and the setting that fixes it is one the measurement
   refutes.
2. **The literal proposal is the *weakest* de-concentration arm**, for a derivable reason.
   Truncating at c is **itself a concentrating operation**: it compresses the draw by √(1−δ(α)),
   so σ = 0.4435 arrives as an imputed sd of **0.2555** (predicted 0.267 from σ√(1−δ(0)) — the
   derivation called the arm's own dispersion to within 4%). **The proposed fix partly re-commits
   the sin it was designed to cure.**
3. **Most of the effect is the histogram, not proper multiple imputation.** `shared44` — *one*
   draw shared by all eight members — delivers −0.085 of `draw44`'s −0.094. So ~90% of the repair
   is the label histogram changing tree structure in the mass region; the Rubin-rules semantics,
   the entire point of MI, are worth the residual **−0.009**.

And Δρ is second-order *by construction*. For a tree of fixed structure a leaf value is a weighted
mean of its rows' labels, so `E_draw[leaf]` equals the mean-imputation leaf exactly — MI-by-draw
and MI-by-mean have **identical expected point estimates for any estimator linear in y**. Boosting
picks structure greedily from y, so Taylor-expanding gives
`p̄ = f̂(x; E[Y]) + ½ tr(∇²_Y f̂ · Σ) + O(Σ²)`: the first-order term *is* the mean-imputation fit and
the correction is second order in σ, of either sign. **Δr(σ̂,p) is first-order; Δρ is
second-order.** Exactly the measured pattern.

Scaled by the measured mass fractions (CYP2D6 0.605 against the 0.267 mean ⇒ **2.27× macro
dilution**), a +0.005 macro bar demands **CYP2D6 Δρ ≥ +0.0113**. The best arm gives +0.0044 and
the literal proposal +0.0003 — **short by 2.6× to 38×**. And that bar asks for **22% of the
mass's entire information content** (dropping it costs −0.0516) from a procedure that by
construction adds **no information at all**: the draws come from our own model's fit, so MI-by-draw
and MI-by-mean condition on the *identical* information set and differ only in propagation.

## Two of the five alternatives are already falsified in our own repo

| option | status |
|---|---|
| censored / Tobit **loss** (grad λ(α)/s, hess δ(α)/s²) | **already measured: `cyp_hinge_rho.py`, macro Δρ −0.0094, 0/5 seeds** |
| pairwise / ranking loss | **already measured: `cyp_rank_objective.py`, lambdarank −0.1296 (t=−73), rank_xendcg −1.2013** |
| Chernozhukov–Hong 3-step CQR | dead on *identification*, not algebra — CYP2D6 has 0 dual-labelled rows |
| PU / one-sided label noise | η = P(y\*>c \| inactive) = **0.498**, far too large for PU corrections — but it is the **correct diagnosis** |
| XGBoost `survival:aft` | same estimator as the censored loss. And **feeding pIC50 would be a silent bug**: AFT fits `ln{Y} = T(x) + σZ`, and pIC50 is *already* a log quantity. The correct parameterisation is Y = IC50 with **right**-censoring at 50 µM |

**And there is now a mechanism for why the censored loss lost**, which the original run could not
supply. At s = 0.4435 a censored row's gradient λ(α)/s is **5–24× larger** than an L2 row's at the
same prediction. With 61% of CYP2D6's rows censored the objective becomes dominated by downward
pressure — and **49.8% of those rows truly lie above c**, so half of that enormous pressure is
simply wrong. It is not a gentle hinge here; it is an aggressive, half-mistaken shove.

## Where the data actually points: inverse-variance weighting

If the screen label is a point estimate with known noise, the fix is not an estimator — it is a
**weight**. Under squared loss with known per-row label-noise variance, the efficient (GLS) weight
is `w_i ∝ 1/σ_i²`, and **both quantities are measured and already in the release**:

| quantity | measured |
|---|---|
| floor pseudo-label σ | **0.4435** (n = 303) |
| DR label σ from the release's own `*_conf_low/_conf_high`, width/3.92 | median **0.0869**, rms **0.2632** (n = 6525) |

| implied `w_floor / w_drc` | |
|---|---|
| at median σ_d | **0.038** |
| at rms σ_d | **0.352** |
| with σ_assay ≈ 0.3 added in quadrature | **0.496** |

So **`PSEUDO_W["inactive"] = 1.0` is between 2× and 26× too high**, and the bracket is *derived*
rather than swept — which matters, because re-looking at a sweep would trip
`feedback-slices-share-noise-resamples-dont`. Note the `down` arm at w = 0.2 already saw the
direction (ρ up on all four isoforms, t = 1.92, 10/16) from a point that is *below* the derived
bracket, i.e. probably overshooting.

**Two free inconsistencies the same logic exposes:**

1. **`PSEUDO_W` is ordered inversely to the measured precisions.** `inactive` gets **1.0** and
   `weak`/`active` get **0.5** — but `weak`/`active` labels are q10–q90 *windows* from the
   calibration table, demonstrably wider than the floor label's 0.4435. The more precise label is
   down-weighted.
2. **`allr_floor` returns the median, but the training loss is squared error.** The risk-minimising
   constant under L2 is the **mean**, 2.3822; the median is L1-optimal. That is a **+0.171**
   location correction with a stated reason, zero compute, and currently wrong.

## What is recorded, and what is not claimed

**Priced honestly, and it does not change.** `PLAN.md` priced this whole phase at **0.35% of
absolute accuracy against a 13.6% requirement**. Nothing here raises that. The weighting line is
recorded as a **correctness** matter with a derivation, not as a board lever, and it gets
pre-registered like anything else before it is run.

**Phase 2 of the LeWAM transfer is therefore closed — on a false premise rather than a failed
bar**, which is a different and better reason than the one I expected to write. The surviving
transfer object was *"MSE targets the conditional mean, so mean imputation re-concentrates"*. It is
**true, measured, monotone, and worth nothing here**, because the quantity it repairs (σ̂) has a
measured deployment value of **−0.00034 Borda, 4/16 seeds** — the 2-D decode family closed in
findings 04–05. **All five catalogued objects are now closed.**

## The methodological lesson, and it is the cheapest one yet

**Measure the latent distribution before modelling the censoring.** One `groupby` on data already
in the release — 20 seconds — gives σ_pop = 0.4435, η = 0.498 and the right-skewed shape, and it
refutes the premise of a 6-seed GPU arm that had already been run, written up and closed. It would
also have pre-empted finding 06's inverted mechanism and finding 07's mis-specified falsification
condition.

This is the fourth consecutive error of one family (`AGENTS.md` trap 8): **a claim about quantity
A tested with a statistic measuring A + B.** Here the claim was about a *bound* and the data
describe a *location*. The fix is the same and it keeps getting cheaper: before modelling a
quantity, go and look at whether anything in the release already measures it.
