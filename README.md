# OpenADMET CYP Blind Challenge — method report

**Entry:** `talindrew` (HuggingFace `arnab2812`) · Regression (direct inhibition) track
**Data:** challenge release only. No proprietary data.

---

## 1. Summary

Three things carry this entry, in descending order of measured effect:

1. **The dose-response labels are selected labels, and the unselected compounds are in
   the release.** Treating the non-promoted compounds as *interval* labels is worth
   **−0.243 macro ST-RAE** on a validation surface that reproduces the board.
2. **A validation surface that contains non-inhibitors.** Our original split held only
   dose-response rows — every one of them a compound selected *for activity* — and so
   could not see the defect above. Rebuilding it reproduced the board per isoform.
3. **Per-isoform affine placement** onto the blinded population's location and scale,
   chosen by a minimax rule over the uncertainty in that population.

Nothing here is a new architecture. The models are a LightGBM and a small multitask MLP
ensemble, both unremarkable. That is the finding: on this task, what the model is
matters far less than what it was trained on and where its predictions are placed.

## 2. The defect: every dose-response label is a selected label

The release ships a single-concentration screen (50 µM, log2 fold-change) for 4,376
compounds on all four isoforms, and a dose-response curve exists for a compound/isoform
**only if it was promoted out of that screen**. Promotion rules differ per isoform —
measured on the release, not assumed:

| isoform | DRC'd / screened | what decides promotion |
|---|---|---|
| CYP1A2 | 1,412 / 4,376 | rises with inhibition; 100 % below log2FC −1.5 |
| CYP2C9 | 1,285 / 4,376 | ~one third, **flat** in log2FC — effectively random |
| CYP2D6 | 1,493 / 4,376 | **hard threshold**: DRC iff log2FC < ≈ −0.6 |
| CYP3A4 | 2,335 / 4,376 | if anything enriched for *weak* readouts |

A model trained on dose-response labels alone **has never seen a CYP2D6 non-inhibitor**,
and over-predicts every unselected compound. The two isoforms whose promotion depends on
activity (1A2, 2D6) were the two we scored worst on (board 0.88 and 1.46, against
0.52/0.57 for the other two).

### What one 50 µM readout says about pIC50

Calibrated on the compounds carrying both. With `r = 2**log2fc` the fraction of signal
remaining, dose-response pIC50 is monotone in `r` on every isoform, and a compound
showing no inhibition at 50 µM (`r ≥ 0.9`) has pIC50 ≈ 2.2 with a credible interval of
about **[1.07, 3.52]** (n = 303 compounds carrying both).

Those compounds become **interval labels, not point labels** — the loss is zero anywhere
inside the window, which is the honest statement of what one readout knows. Three
higher-ranked entries report the same rows *costing* score as point pseudo-labels and
*helping* as censored bounds. For intermediate readouts the label is the [10th, 90th]
percentile of dose-response pIC50 among compounds with the same readout.

Implementation: `drugrx/cyp/screen.py`.

## 3. V1 vs V2: the validation surface is part of the method

- **V1** — the original scaffold split, dose-response rows only. Every row is a compound
  selected for activity.
- **V2** — V1 plus held-out screened compounds scored against the floor window.

**V2 reproduces the board per isoform (0.77 / 0.53 / 1.35 / 0.48 against board
0.88 / 0.52 / 1.46 / 0.57). V1 never did.** A selection fix *must* look worse on V1:
promotion selects on `y`, the test set's analog expansion selects on `x`, and the test
wants `P(y|x)` on the **unselected** population. Both are reported for every arm.

Measured effect of the fix: macro V2 **−0.243** (LightGBM, 3/3 seeds), **−0.313** (the
net, 2/2 seeds). CYP2D6 1.51 → 0.60. Ranking improves too (2D6 ρ 0.378 → 0.476), which
matters because placement is monotone and cannot change it.

## 4. Placement

A model trained on a selected label set predicts on that set's scale. The last step is a
per-isoform affine map onto the blinded population.

ST-RAE is **not** squared error, and its optimum is not squared error's. Squared error
wants centre = population mean and spread = ρ·sd (Murphy 1988: `R² = 2ρk − k² − b²`, so
`k = ρ`). ST-RAE wants **less spread and a centre above the mean**, because an inactive
compound's credible interval is a free window reaching to ≈ 3.5. Our board probe measured
the spread discount at 0.896.

The targets actually submitted were chosen by a **minimax** scan over the uncertainty in
(inactive fraction 0.55–0.81, correlation 0.33–0.60), not by a point estimate:

| isoform | centre | sd |
|---|---|---|
| CYP1A2 | 4.66 | 0.95 |
| CYP2C9 | 4.98 | 0.55 |
| CYP2D6 | 3.43 | 0.30 |
| CYP3A4 | 5.03 | 0.75 |

Because the map is monotone, **ρ and τ must return bit-identical from the board** — a
free end-to-end check, used three times.

Implementation: `drugrx/cyp/placement.py`, `drugrx/cyp/decide.py`,
`scripts/cyp_submit.py`.

## 5. The decision rule

ST-RAE's risk is convex piecewise linear, so its minimiser is the **balance point** where
`P(lo > p) == P(hi < p)` — a weighted median over pooled bound draws, verified exact
against a 4,001-point grid on 600 random cases. Two consequences that cost real score:

- **Ties go to the right-hand end of a flat segment.** When every draw says inactive the
  risk is flat across the free window and no risk-based test can distinguish the ends;
  the error that actually happens is an "inactive" compound being somewhat active, and
  the right end is 2.45 log units closer.
- A probability-weighted **mean** is the wrong operation. A compound 80 % likely inactive
  should be predicted at **3.52**, not the mean's 2.87.

## 6. Negative results worth recording

- **External inactives do not substitute for the challenge's own screen.** 10.5k Veith
  bounds made things *worse*: macro V2 **+0.038**, 3/3 seeds.
- **More training does not help.** 300 → 3,000 epochs moved the score by **−0.0000**,
  identical per endpoint to four decimal places. Without early stopping the 3,000-epoch
  runs bottom out at epoch 190–365 then rise monotonically, ending 10–17 % worse than
  their own best. Capacity is similar: doubling width bought 0.013.
- **Concatenating feature blocks hurt.** CheMeleon ⊕ ECFP scored worse than either alone
  under the same in-context model.

## 7. Reproducibility notes and known bugs we fixed

Recorded because they were silent failures that looked healthy:

- `chemeleon_fp.py` returned **atom-level** vectors. Chemprop's `BondMessagePassing`
  gives (n_atoms, 2048), not (n_molecules, 2048); indexing it by molecule produced one
  arbitrary atom's embedding from the wrong molecule. Shape, finiteness and
  dead-dimension checks all pass on that garbage. What exposed it: a molecule's nearest
  neighbour in the embedding had **Tanimoto 0.132 against a random-pair baseline of
  0.132**. Fixed with mean aggregation over `batch.batch`.
- `allr_floor` returned NaN on an empty pool; trainers mask NaN, so the selection fix
  would have become a silent no-op on a healthy-looking run. It now raises.
- The screen labeller decided "inactive" by calibration-bin membership rather than by the
  readout, so on an isoform whose promotion excludes weak bins (exactly CYP2D6) a plainly
  inactive compound could be labelled **active**.

## 8. Code

The method is fully specified above. The implementation lives in a private
monorepo; a standalone extract of the four modules named in this report can be
published on request.

---

*Report drafted 2026-10-08. Metric: macro-averaged soft-threshold relative absolute error
against the published credible intervals; 1.0 = as good as predicting the mean.*
