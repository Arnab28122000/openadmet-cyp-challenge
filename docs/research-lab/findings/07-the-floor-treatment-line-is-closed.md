# 07 — Censored-floor treatment: both pre-registered claims failed, and the line is closed for board value

**Status: closed, by its own pre-registered bar.** 6 seeds × 4 isoforms × 3 arms,
`scripts/cyp_censored_floor.py`, pre-registered in `829eb5d` before the run. Raw numbers in
`data/cyp_cache/censored_floor.json`.

## What was tested

Finding 06 established the screen pseudo-label point mass at 2.211 is causal for CYP2D6's σ̂
artefact, and that the information in it is real (dropping it costs **−0.0516 ρ**). The proposed
principled fix was to treat "screen-inactive" as left-censored — `y ≤ 2.211`, not `y = 2.211` —
via the Gaussian EM step `E[y* | y* ≤ c] = μ − σ·φ(α)/Φ(α)`.

## Claim 1 — FAILED

> Pre-registered: "r(σ̂,p) on CYP2D6 … under `tobit` it must fall to ≤ 0."

| isoform | r(σ̂,p) full | r(σ̂,p) tobit | positive seeds, full | **positive seeds, tobit** |
|---|---|---|---|---|
| CYP1A2 | −0.040 | −0.102 | 2/6 | 0/6 |
| CYP2C9 | −0.473 | −0.462 | 0/6 | 0/6 |
| **CYP2D6** | **+0.336** | **+0.264** | **6/6** | **6/6** |
| CYP3A4 | −0.336 | −0.354 | 0/6 | 0/6 |

It fell by 0.072 and stayed positive on **every seed**. That is a fail against the stated threshold.

## Claim 2 — FAILED, both arms

> Pre-registered bar: macro Δρ ≥ **+0.005** AND positive on ≥ **5 of 6** seeds.

| arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | **macro** | wins | verdict |
|---|---|---|---|---|---|---|---|
| `down` (w=0.2) | +0.0044 | +0.0050 | +0.0049 | +0.0006 | **+0.0037** | 5/6 | below bar |
| `tobit` | +0.0027 | +0.0017 | +0.0012 | −0.0007 | **+0.0012** | 3/6 | below bar |

`down` clears the *wins* half and misses the *magnitude* half; `tobit` misses both. A near miss is
a miss (`feedback-a-magnitude-bar-is-not-a-significance-bar`). **Neither goes into a submission.**

Worth recording plainly: **the dumb weight hack beat the principled estimator on all four
isoforms.** Theory earned nothing here.

## The pre-registered falsification condition was mis-specified — by me, and it is not a rescue

I wrote: *"if it does not happen, finding 06 is wrong and says so."* Reporting that honestly
requires reporting that **the inference was wrong, not that finding 06 was vindicated**, and the
reason is a number I could have predicted before spending the compute:

| arm | imputed target for the 2648 inactive rows |
|---|---|
| `full` | 2.211, **sd exactly 0.000** — a point mass |
| `tobit` | ~2.10, **sd 0.030–0.084** — **3–5% of s_y** |

The EM step imputes a conditional **mean**. Compounds with similar μ receive near-identical
imputations, so the arm moved the point mass's **location** (2.211 → ~2.10) and left its
**concentration** essentially intact. Finding 06's mechanism is about concentration causing
ensemble members to agree. **This arm never varied that.**

So the run is **uninformative about finding 06**, not evidence against it — which is precisely
`feedback-negatives-need-range`: *a null is only as strong as the range of the variable it was
tested over; don't test a mechanism where it cannot appear.* I wrote that rule into this lab and
then violated it in the next experiment.

Finding 06 continues to stand on its own evidence, which **did** have range: the weight
dose-response 1.0 → 0.2 → dropped, where r(σ̂,p) positivity goes **4/4 → 2/4 → 0/4**.

## Why the line closes anyway

The obvious follow-up is **multiple imputation by draw** — sample each censored row from the
truncated normal rather than taking its mean, with a different draw per ensemble member. That is
the textbook correction for exactly this pathology (mean imputation understates variance; proper
multiple imputation preserves it), and it would be a real test of de-concentration.

**It is not being built, and the reason is pricing, not fatigue:**

1. **The family's ceiling is now measured and it is below the bar.** Three arms spanning the
   natural range of floor treatments — reweight (+0.0037), delete (−0.0516 on CYP2D6), re-locate
   (+0.0012) — and the best is **+0.0037 against a +0.005 bar**. A fourth variant plausibly lands
   in the same band (`feedback-compute-the-ceiling-before-building`).
2. **A σ̂ repair has no downstream consumer.** The only use for better per-item uncertainty was the
   2-D decode, and that family is **closed**: Borda −0.0003 macro, and the cross-fitted ceiling
   over *all* functions of (p, σ̂) negative at 3/4/6 bins (findings 04–05). Repairing σ̂ would fix a
   diagnostic nothing reads.
3. **The mass is largest where the headroom is smallest.** CYP2D6 is 61% floor and rank **28**
   (needs 5.4%); CYP3A4 is 2% floor and rank **134** (needs 24.7%). This whole line helps the
   isoform we least need helped — stated before the run, and borne out.

**Explicitly left open, not claimed either way:** whether de-concentration repairs the σ̂ artefact.
The question is unresolved; what changed is that we no longer have a use for the answer.

## The process lesson

Two in one day. Finding 06: the stated *mechanism* was inverted while the *suspect* was right.
Finding 07: the stated *falsification condition* was mis-specified, because the arm did not
implement the intervention it was claimed to test. Both were caught only by measuring something
cheap afterwards — the label distribution in 06, the imputed sd here. **Check that an arm varies
the quantity its claim is about, before running it, not after.**

---

## ADDENDUM 2026-10-11 — there was a **second**, independent defect, and it was the σ

Found while chasing the open check `PLAN.md` carried into the LeWAM transfer's Phase 2. **This
corrects the diagnosis above, not the verdict.** The arm still failed; it failed for *two*
reasons, and only one of them was identified at the time.

`cyp_censored_floor.py:136` sets the censored model's σ from

```python
sig = float(np.std(y[dd] - m.predict(X[dd]), ddof=1))
```

where `dd` are the **dose-response rows that `m` was just fitted on**. That is an **in-sample
residual sd from an overfit gradient-boosted ensemble**. Measured three ways, 2 seeds:

| isoform | s_y | **(a) in-sample** (what the code used) | **(b) OOB** | **(c) held-out** | c/a | c/s_y |
|---|---|---|---|---|---|---|
| CYP1A2 | 1.029 | **0.130** | 0.887 | 0.799 | **6.2×** | 0.78 |
| CYP2C9 | 0.786 | **0.086** | 0.586 | 0.591 | **6.8×** | 0.75 |
| CYP2D6 | 0.901 | **0.190** | 0.945 | 0.961 | **5.1×** | 1.07 |
| CYP3A4 | 1.092 | **0.115** | 0.686 | 0.718 | **6.3×** | 0.66 |

Seed 1 reproduces every cell to within a few percent. **σ was 5–7× too small**, so the Tobit step
believed it knew each latent y\* to within ±0.13 when the honest predictive uncertainty is
±0.59–0.96, i.e. **66–107% of s_y**.

**Two consequences.**

1. **The write-up above is incomplete, not wrong.** It attributed the surviving concentration
   entirely to the conditional *mean* ("compounds with similar μ receive near-identical
   imputations"). That is true — sd(μ) across the inactive rows is only 0.11–0.18 — but a σ that
   is 5–7× too small compresses the Mills-ratio map on top of it. The imputations' sd (0.030–0.084)
   came out *below* even sd(μ), so both effects were acting.
2. **It means switching from mean to draw would NOT have rescued it either.** A truncated-normal
   draw at σ = 0.13 gives an imputed sd of order 0.1 — still the far-left of `findings/08`'s
   concentration axis, still no manipulation of the quantity the claim is about. **The two defects
   had to be fixed together, and nothing in the original run would have revealed that.**

**The free fix for σ estimation generally.** (b) and (c) agree in every cell (6.8/6.2, 6.8/6.8,
5.0/5.1, 6.0/6.3), so **out-of-bag residuals are a valid substitute for a held-out estimate** —
and they cost nothing, because the ensemble members are already bootstrap fits with out-of-bag
rows sitting unused.

> **CORRECTED the same day — see `findings/11`. This paragraph originally ended "any future
> censored arm takes σ from OOB." Do NOT do that.** 2.211 turns out to be the **median of a
> measured 303-compound distribution**, not a bound: **49.8% of screen-inactive compounds have a
> true pIC50 above it**, and the population has a hard floor at 1.906. A correctly specified
> Tobit at the honest σ therefore imputes **1.857** (σ_pop = 0.4435) or **1.480** (σ_out = 0.9161)
> — **both below the measured minimum.** The in-sample σ bug is the only reason this arm was
> merely uninformative instead of actively harmful: **the bug was protecting us.** No σ rescues
> the censoring family, because the censoring model itself is false. The OOB lesson stands as a
> general point about estimating a noise scale; it is not a route back into this family.

**Phase 2 was briefly thought to be fully specified on this basis** — a manipulation-check
instrument, a corrected σ, and a predicted landing point. **`findings/11`, later the same day,
closed it instead, on a false premise rather than a failed bar.** The de-concentration mechanism
is real and perfectly monotone when measured properly (Δr(σ̂,p) 0.000 → −0.019 → −0.060 → −0.094 →
−0.301 in imputed sd), and the intervention is still worth nothing: Δρ is **second-order by
construction**, the measured arms span −0.0019 to +0.0044 against a required +0.0113, and ~90% of
even that comes from the label histogram rather than from multiple-imputation semantics. The
verdict above stands for a better reason than the one originally given.

**And the general lesson, which is not specific to censoring:** an in-sample residual sd is not an
estimate of predictive uncertainty, and any estimator that consumes a σ — Tobit, censored
likelihoods, calibration, acquisition, anything Gaussian — inherits that error silently. The
number does not look wrong. It looks like a σ.
