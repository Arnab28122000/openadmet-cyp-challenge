# 06 — CYP2D6's σ̂ anomaly is an artefact of a training point mass, and both clauses of our explanation were backwards

**Status: resolved.** 4 seeds × 4 isoforms × 3 arms, `scripts/cyp_sigma_anomaly.py`,
pre-registered in commit `dd1652a` before the run. Raw numbers in
`data/cyp_cache/sigma_anomaly.json`.

## What was anomalous

`cyp_sigma_quality.py` found σ̂ tracking |residual| positively on three isoforms (+0.086 /
+0.129 / +0.107) and **negatively on CYP2D6** (−0.164 ensemble, −0.157 fitted), on every seed and
with both estimators. Compounds the model was *more* uncertain about had *smaller* errors.

## The explanation we gave, and why it was wrong

> "CYP2D6 carries the largest inactive mass … compounds predicted near the floor plausibly draw
> both high ensemble disagreement *and* labels compressed against the floor, so |error| is
> mechanically small there."

**The first clause is refuted by the labels alone, before any model is fitted.** The DRC pIC50
distributions have no censoring spike: 0.1–0.2% of labels sit on the lowest observed value on
every isoform, the modal value holds 0.2–0.5%, and CYP2D6's is not distinctive. The scored
compounds are all `kind == "drc"`. There is nothing compressed against a floor in the labels.

**Both clauses are refuted by the measurement, with the signs exactly inverted.** Near the floor
the ensemble **agrees** rather than disagrees, and the error is **largest** rather than smallest.

## What is actually happening

The floor is in the **training pseudo-labels**, not the scored ones. `screen_labels` pins every
screen-inactive compound at the constant **2.211**, `PSEUDO_W["inactive"]` is **1.0** — the same
weight as a real measurement — and the mass varies 30× across isoforms:

| isoform | floor share of labelled rows | inactive / active |
|---|---|---|
| CYP2D6 | **61%** | 2648 / 53 |
| CYP1A2 | 29% | 1290 / 771 |
| CYP2C9 | 15% | 654 / 1687 |
| CYP3A4 | **2%** | 91 / 2200 |

With 61% of the training signal on one repeated value, every ensemble member predicts that value
in the regions the point mass dominates. **The members agree there, so σ̂ is small at low p.**
Meanwhile the DRC-labelled compounds that fall in those regions are where the model is most wrong.
The two effects have opposite signs in p, and their product is the anomaly:

| isoform | r(σ̂,\|err\|) | **partial \| p** | r(σ̂,p) | r(\|err\|,p) |
|---|---|---|---|---|
| CYP1A2 | +0.086 | +0.073 | −0.053 | −0.347 |
| CYP2C9 | +0.129 | +0.063 | −0.491 | −0.144 |
| **CYP2D6** | **−0.164** | **+0.038** | **+0.324** | **−0.592** |
| CYP3A4 | +0.107 | +0.048 | −0.343 | −0.180 |

**Controlling for the prediction level, CYP2D6's σ̂ behaves exactly like the other three** — +0.038
against +0.073 / +0.063 / +0.048, and positive on **4 of 4 seeds** where the raw figure is negative
on 4 of 4. The anomaly was never a statement about uncertainty.

CYP2D6-full is the **only cell in the entire 4 × 3 grid** where r(σ̂,p) is reliably positive.

## The causal test

Three arms differing only in the floor's weight — 1.0, 0.2, dropped:

| isoform | floor % | full | down (0.2) | none | full→none |
|---|---|---|---|---|---|
| CYP1A2 | 29% | +0.086 | +0.145 | −0.015 | −0.101 |
| CYP2C9 | 15% | +0.129 | +0.114 | +0.113 | −0.017 |
| **CYP2D6** | **61%** | **−0.164** | **+0.014** | **+0.110** | **+0.273** |
| CYP3A4 | 2% | +0.107 | +0.124 | +0.144 | +0.037 |

CYP2D6 moves **+0.273**, monotonically, and is the only isoform that starts negative and ends
positive. The cleanest readout is the sign of r(σ̂,p) as the weight falls: **positive 4/4 → 2/4 →
0/4**. The point mass is causal.

## What it costs — so deletion is not the fix

| isoform | ρ full | ρ down | ρ none | full→none |
|---|---|---|---|---|
| CYP1A2 | 0.5536 | 0.5566 | 0.5333 | −0.0203 |
| CYP2C9 | 0.6594 | 0.6674 | 0.6455 | −0.0138 |
| **CYP2D6** | 0.4824 | 0.4854 | **0.4308** | **−0.0516** |
| CYP3A4 | 0.7721 | 0.7723 | 0.7717 | −0.0004 |

Dropping the floor costs CYP2D6 **−0.0516 ρ**, which on this board is tens of places. The
information in "this compound was inactive in the screen" is real; what is wrong is asserting it
equals 2.211 exactly.

**Explicitly NOT claimed:** the `down` arm gains ρ on all four isoforms (pooled **+0.0035**), but
at **t = 1.92, 10/16 wins**, with every isoform losing on seed 0. That does not clear a bar
(`feedback-a-magnitude-bar-is-not-a-significance-bar`). It is a candidate for more seeds, not a
result, and it is not being deployed on that evidence.

## What this explains elsewhere

**Why Borda hurt CYP2D6 most** (t = −3.10, 0/6 seeds, finding 04). Borda shrinks high-σ̂ items
toward the middle of the order. On CYP2D6 the high-σ̂ items are the high-p items, which are the
*well-predicted* ones — so Borda corrupts the top of the ranking precisely where it is right. That
reading survives from the earlier notebook entry; what was wrong was the floor-compression route
offered for it.

## The principled next step

Not deletion and not re-weighting, but **honest censoring**: a screen-inactive compound tells us
`y ≤ 2.211`, not `y = 2.211`. That is a left-censored regression problem, which is exactly what
Chernozhukov–Hong 3-step censored quantile regression addresses, and it is already a briefed but
unbuilt strand. It keeps the information the `none` arm throws away (−0.0516 ρ) while removing the
false precision that produces the artefact. **Not yet tested.**

## The methodological lesson

This is the third stated mechanism this programme has tested and the second to be refuted
outright — and the first where the data inverted *both* clauses while confirming the *suspect*.
Pointing at the right culprit is not the same as knowing what it did
(`feedback-test-the-explanation-you-just-gave`).
