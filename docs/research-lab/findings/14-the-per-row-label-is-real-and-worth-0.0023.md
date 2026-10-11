# 14 — The per-row label is real, measurable, and worth +0.0023 macro ρ. The bar was +0.005.

**Status: closed by its own pre-registered bar.** `scripts/cyp_rowwise_shard.py`, **144 cells** =
6 arms × 4 isoforms × 6 seeds, sharded across two L4s with lightgbm pinned to 4.7.0 and
`N_MEMBERS = 8` asserted against production by import (`tests/test_shard_parity.py`). Arnab's
hypothesis, built as specified and graded as pre-registered.

## The hypothesis and what the gate did to it

Arnab, 2026-10-11: *train a weight **matrix** rather than one scalar, loop over a bank of
weightings, combine at inference.* `findings/12`'s gate redirected it: the pseudo-label error
**does** depend on the continuous readout `r` (Spearman −0.377 signed, holding within isoform) but
is **not** predictable from the 3091 features out of fold (oracle +0.9949, cross-fitted +0.1274).
So the per-row object is a **label** keyed on `r`, not a weight keyed on structure.

## The result: every arm misses, and the family ceiling held

| arm | CYP1A2 | CYP2C9 | CYP2D6 | CYP3A4 | **macro** | seeds | verdict |
|---|---|---|---|---|---|---|---|
| `meanfloor` (uniform +0.174) | +0.0025 | +0.0025 | +0.0026 | +0.0005 | **+0.0020** | 4/6 | below bar |
| **`rowlabel`** (log-linear g(r)) | +0.0041 | +0.0017 | +0.0020 | +0.0015 | **+0.0023** | **5/6** | below bar |
| `rowiso` (isotonic g(r)) | −0.0010 | +0.0022 | +0.0014 | −0.0002 | +0.0006 | 4/6 | below bar |
| `rowboth` (+ per-row weight) | +0.0054 | +0.0043 | +0.0005 | +0.0003 | **+0.0026** | 4/6 | below bar |
| **`bank`** (per-member calibration) | +0.0042 | +0.0001 | **−0.0020** | +0.0008 | +0.0008 | 4/6 | below bar |

**Bar: macro Δρ ≥ +0.005 on ≥5/6 seeds. Nothing reaches it.** The manipulation check is clean —
`base` 0.0000 ± 0.0000, `meanfloor` +0.1737 ± **0.0000** (a constant, as designed), `rowlabel`
+0.1411 ± **0.0793** (genuinely per-row).

**The pre-registered ordering test technically passes and means nothing.** `rowlabel` (+0.0023)
does beat `meanfloor` (+0.0020), so the r-dependence is worth more than the uniform shift — **by
+0.0003**, which is noise. The honest reading is that the r-dependent part adds nothing detectable
over simply changing the constant.

**And the whole family lands at +0.0006 to +0.0026 — inside the +0.0037 ceiling `findings/07`
measured for the *scalar* weight family.** That is the result: extending from one scalar to 24,580
per-row labels **did not escape the family ceiling**. The ceiling is a property of how much the
pseudo-labels can contribute, not of how finely they are parameterised.

## Two things the arms say that the headline does not

**`bank` — Arnab's bank, done properly — is the second-worst arm, and it is *negative* on the
isoform that matters.** Each ensemble member got its own calibration from a bootstrap of the
in-split dual sample, so the ensemble averages over calibration uncertainty. On CYP2D6, which
carries **60.5%** of the relabelled mass, it scores **−0.0020**. Averaging over calibration
uncertainty does not pay here, and the clean variance argument for it does not survive contact.

**`rowiso` is the worst calibration despite being the best fit.** Isotonic had in-sample R² 0.1773
against log-linear's 0.0526 — **3.4× better** — and finishes last at +0.0006 with two isoforms
negative. Textbook overfitting of a non-parametric fit on n = 303, and the honest CV that the
verifier ran afterwards predicted it (isotonic shrinks 1.9×, log-linear 3.0×, but isotonic's
flexibility is what costs it on transfer).

## The arm was built on a calibration whose sign is wrong

`findings/12`'s addendum records that **B7's `+0.0966` shift direction is an artefact of fitting
OLS — a conditional-*mean* estimator — to improve a *median***. Four of five defensible rules move
the label **down**; under L1 the shift is **−0.0665**, and on CYP2D6 it goes from +0.0299 to
**−0.1765**.

**So `rowlabel` and `rowboth` were run with the up-shifting OLS calibration (+0.141 realised), and
still returned +0.0023/+0.0026.** The L1 and local-median versions are **untested**. That is the
one live thread left in this family, and `findings/12` names the cheapest form: **split the floor
at r = 1.0** into 2.2732/2.1320 — the only rule besides isotonic to win on *both* OOF MSE (−4.38%)
and OOF MAE (−1.54%), with no slope, no extrapolation and no per-row weight. It moves CYP2D6 by
**−0.0723**, the opposite direction to what was tested here.

Given that the up-shifted version bought +0.0023 against a +0.005 bar, a down-shifted version
would have to be **2× better** to clear it. **Priced that way, it does not justify a run** — but it
is a one-line change to a constant, so it is worth folding into the next submission's build rather
than measured as its own arm.

## What this cost and what it bought

Two L4s, ~3 hours, 144 production-learner fits per shard. It bought a **clean negative on a
well-specified hypothesis**, a **confirmed instrument** (the readout genuinely carries label
information — Spearman −0.377, within-isoform), and the finding that **the ceiling is in the data,
not the parameterisation**.

It also bought the discovery that was worth more than the arm: tracing why the per-row label did so
little led to the **production early-stopping defect** (`findings/12` addendum §4) — `cyp_final.py`
selects XGBoost's early stopping under MAE on a set that is **61.8–65.1% the single constant
2.2111 on CYP2D6**, and MAE against a constant is minimised by predicting it. That ordering is
exactly `findings/06`'s anomaly ordering, and it is a live defect in the submission path.
