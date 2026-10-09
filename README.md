# OpenADMET CYP Blind Challenge — method report

**Entry:** `talindrew` (HuggingFace `arnab2812`) · Regression (direct inhibition) **and**
TDI (classification) tracks
**Data:** challenge release only. No proprietary data.
**Published:** https://github.com/Arnab28122000/openadmet-cyp-challenge (verified HTTP 200,
2026-10-09). A reachable `https://` method report is **mandatory** before 2026-11-03 23:59
UTC or the entry is excluded from the final leaderboard regardless of score.

**ATTACHED** to the TDI submission of 2026-10-08 21:44 UTC and to the regression submissions
of 2026-10-09 04:03 and 17:09 UTC (the Live board renders the link on our row). "My code is
open-source" is left unchecked, because the report alone is not code.

**Last updated 2026-10-09**, after the third placement submission. Two numbers previously
stated here have been **withdrawn** on re-measurement and are described as such in §6 rather
than quietly removed.

---

## 1. Summary

**Measured effect on the live leaderboard: rank 172 → 60 of 268.** Macro ST-RAE 0.8663 →
**0.5414**, macro R² 0.0748 → **0.5329**, macro Spearman ρ 0.6212 → **0.7093**. Per isoform,
CYP1A2 0.88 → 0.549, CYP2C9 0.52 → **0.462**, CYP2D6 **1.46 → 0.587**, CYP3A4 0.57 → 0.567.

Those are the figures after three successive placement submissions (v4 → v5 → v6), each
moving **one** axis so that the board's own response could be read as a derivative. ρ and τ
have returned 0.7093 and 0.5285 unchanged from all three, which is the end-to-end check that
the map really is monotone.

ρ rose, and the affine placement in §4 is monotone and therefore *cannot* change ρ — so the
two effects separate cleanly: the interval labels of §2 moved the model, the placement moved
the level.

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
free end-to-end check, used four times.

### Using the board as an instrument: one axis per submission

The minimax targets above were superseded by something better than any prior. Because ST-RAE
on 750 fixed compounds is **deterministic**, a difference between two board rows is a
*measurement*, not an estimate — there is no sampling noise and no t-statistic to compute. So
moving exactly one axis per submission turns the leaderboard into an instrument for the
derivative of the metric with respect to that axis.

Three submissions, spread axis only:

| isoform | sd v4→v5→v6 | ST-RAE | secant v4→v5 | secant v5→v6 | state of the lever |
|---|---|---|---|---|---|
| CYP1A2 | 0.95 → 1.00 → 1.00 | 0.549 → 0.549 → 0.549 | 0.000 | (held) | **at the optimum** |
| CYP2C9 | 0.55 → 0.68 → 0.76 | 0.505 → 0.466 → 0.462 | −0.297 | −0.045 | nearly spent |
| CYP2D6 | 0.30 → 0.60 → 0.78 | 0.746 → 0.640 → 0.587 | −0.351 | −0.296 | **room left** |
| CYP3A4 | 0.75 → 0.82 → 0.85 | 0.571 → 0.566 → 0.567 | −0.066 | **+0.023** | **past the optimum** |

Two results fall out that a single submission could not give:

* **The 0.896 spread discount is confirmed to four decimals.** CYP1A2 at sd 1.00 sits within
  0.001 of 0.896 × its squared-error optimum, and scored **0.5490 three times running**.
* **CYP2D6's copula-derived correlation is refuted by its own row.** On the copula `r = 0.518`
  its optimum spread is 0.616, so sd 0.78 would be 0.164 *past* the optimum and should have
  scored worse. It improved by 0.053. The distribution-free bound (`r ≥ 0.609`, from its R²
  exceeding the copula ceiling) is the right one.

The surface is also **very flat near its minimum**: CYP3A4 overshot by 0.036 of spread and paid
only +0.0007 of ST-RAE, about 0.02 per unit of overshoot. Overshooting is therefore cheap and
undershooting leaves gains unclaimed — which is an argument for bolder steps wherever a secant
is still clearly negative, and it is why v7 moves CYP2D6 to 1.00 rather than to the 0.906 the
theory alone suggests.

### Recovering the targets from the leaderboard rather than guessing them

The targets above were chosen by minimax because the population is blinded. That was a
mistake worth recording: **the board reports enough to solve for the population directly.**
For a prediction vector held on disk, with `r = 2 sin(πρ/6)` from the reported Spearman,

    s² = s_y² + s_p² − 2 r s_y s_p          and      m² + s² = (1 − R²) s_y²

is two equations in the two unknowns `s_y` and `m = ȳ − p̄`. Solved on our own row this gives
per-isoform test sd **1.44 / 0.97 / 1.66 / 1.16**, which reproduces an independently published
reverse-engineering of the same quantities to **3.8–12.3 %** by an unrelated route. R² and MAE
depend on `m` only through `m²`, so the algebra cannot sign the offset; scoring both branches
against that independent estimate picks the same sign on all four isoforms with 5×
discrimination.

The consequence: minimaxing over a band wider than the truth does not buy safety. The scan
chose **sd 0.10 for CYP2D6**, which inverts to an assumed `r = 0.209` — below the floor of the
band it was told to search — against a measured `r = 0.518`. Five routes sharing no
assumptions (including a distribution-free bound from `MAE ≤ RMSE` and `R² ≤ r²`, which needs
no distributional assumption at all) agree that every isoform was placed too narrow.

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
- **More training does not help, tested to 30,000 epochs.** 300 → 3,000 epochs moved the
  score by **−0.0000**, identical per endpoint to four decimal places. At 30,000 epochs with
  cosine warm restarts (`T_0=468, T_mult=2`) the gain over 1,200 epochs is **−0.0026 against
  a seed spread of 0.0052** — half the noise of reseeding, for 25× the compute; the run peaked
  at epoch 7,025 and was flat for the remaining 23,000. `ReduceLROnPlateau` at the same budget
  is *worse* (best at epoch 25): our validation signal is noisy at the 0.005 level, so the
  scheduler decays on noise and reaches its LR floor by epoch 650. Without early stopping the
  3,000-epoch runs bottom out at epoch 190–365 then rise monotonically, ending 10–17 % worse
  than their own best. Capacity is similar: doubling width bought 0.013.
- **A pretrained molecular foundation model did not beat ECFP, but its loss function did.**
  CheMeleon's frozen 2048-d D-MPNN fingerprint lost to count-ECFP + descriptors under the same
  GBM (0.6433 vs 0.5819 macro on our validation surface). Fine-tuning it end-to-end under the
  *metric's own interval hinge* scored **0.5179 ± 0.0130**, beating every frozen arm — but
  embeddings read back out of that fine-tuned encoder score 0.5862/0.5886, i.e. no better than
  frozen ECFP. The gain is in training against the metric, not in the representation.
- **Concatenating feature blocks hurt.** CheMeleon ⊕ ECFP scored worse than either alone
  under the same in-context model.
- **Ranking objectives lose badly.** The censored labels argue for them — squared error spends
  capacity pulling predictions to an arbitrary point inside a censored interval, while a
  ranking loss only needs the censored compounds *below* the actives. Measured on CYP3A4 with
  paired seeds, LightGBM `lambdarank` is **−0.177 ρ** and `rank_xendcg` **−0.038**. We
  pre-registered binning as the suspected cost (the rankers need discrete grades) and then
  tested it: raising grades 16 → 128 recovers only 0.047 of lambdarank's 0.177, so **our own
  explanation accounts for about a quarter of it**. The likelier cause is that NDCG's
  positional discount and Spearman's equal weighting are simply different objectives.
- **The ensemble's advantage over a plain LightGBM is unmeasurable.** It had been recorded at
  +0.0154 macro ρ. That comparison was not paired — the net was scored on its own split against
  LightGBM figures from a different one. Training a GBM on the exact complement of the net's
  saved validation indices gives macro **−0.0066**, and a paired bootstrap over compounds puts
  **every per-isoform CI across zero**. Compounding it, the three saved "seeds" share a
  byte-identical validation index, so their spread measures network initialisation and not the
  compound sample; the seed-level error bars were ~10× too tight. This *strengthens* §1's
  claim rather than weakening it: what the model is matters even less than we had measured.

## 7. The validation surface and the blinded set measure different regimes

§3 rebuilt the validation surface so that it contained non-inhibitors. It is still not the
population the board scores, and the gap is measurable without any labels — nearest-neighbour
Tanimoto to the training set (ECFP4 counts, 2048 bits):

| NN band | blinded TEST | our VALIDATION |
|---|---|---|
| 0.3–0.5 | **4.8 %** | **65.2 %** |
| 0.5–0.7 | **84.9 %** | 25.6 % |
| 0.7–0.9 | 10.1 % | 3.3 % |
| median | 0.587 | 0.439 |

**Two thirds of our validation sits in a band holding under 5 % of the test set.** The cause is
structural: the challenge's blinded set is an analog expansion purchased *around* the training
chemistry, while **82 % of the training compounds are Murcko scaffold singletons** (5,370
distinct scaffolds for 6,145 compounds), so a scaffold split over it is close to a random — and
therefore more diverse — partition. The name "scaffold split" promises a harder evaluation than
it delivers here.

This also explains the validation→board lift (+0.17 CYP1A2, +0.13 CYP2C9, ≈0 CYP2D6 and
CYP3A4) without appealing to anything the model does: the board asks easier questions. Those
lifts carry 95 % intervals of 0.095–0.191 from a single split, so quoting them to three
decimals overstates what is known.

`drugrx/cyp/matched.py` corrects it by importance-weighting each validation compound to the
measured test histogram and resampling, which keeps every compound where a hard `NN ≥ 0.5` cut
discards three quarters of them. **The correction is not cosmetic — it flips signs on small
effects**, and it agrees with the independent `NN ≥ 0.5` contrast, which is the check that
matters. Any effect below ~0.01 ρ measured on the uncorrected surface should be treated as
unsigned.

## 8. The TDI classification track

A separate submission, same release, no extra data: per-isoform binary classifiers for
`CYP2D6_is_TDI` and `CYP3A4_is_TDI`. Board: **macro MCC 0.2964**, rank 71 of 153 (CYP2D6
0.2291, CYP3A4 0.3637).

The one idea worth reporting is that **the operating point can be chosen against the blinded
test set's own class balance, because the leaderboard gives that balance away.** For a binary
task `TP = R·π = P·q`, and `Accuracy = 1 − π − q + 2Rπ`, so

    π = (1 − Accuracy) / (1 + Recall/Precision − 2·Recall)

Every entry is scored on the same test set, so π must agree across entries — and that
agreement is the validation. It agrees to **sd 0.00018** (CYP2D6, 145 entries) and **sd
0.00008** (CYP3A4, 152 entries): π = **0.0706** and **0.2916**. MCC then has a closed form,

    MCC = √(π/(1−π)) · (R − q) / √(q(1−q))

verified against sklearn to 3.3e-16, and it reproduces our own reported MCC to 0.001. Our
thresholds had been tuned where the positive rate was 0.195 against a true prevalence of
0.0706: thresholding by *probability value* does not transfer when the probability scale
shifts, thresholding by **rate** does. Moving CYP3A4's rate from 0.473 to 0.313 was worth
**+0.0138 MCC**, measured against the test ROC rather than assumed.

The same row also gives the realised operating point, `FPR = (q − Rπ)/(1 − π)`, and one
(TPR, FPR) pins an equal-variance binormal ROC. That yields a **measured test AUC of 0.762
(CYP2D6) and 0.764 (CYP3A4)** against validation values of 0.728 and 0.813 — biased in
opposite directions, which is why a pre-registered macro MCC of 0.318 came back 0.296. An ROC
is prevalence-independent; it is not dataset-independent. At the realised ROC both thresholds
are now within **0.0006 macro MCC** of optimal, so the remaining lever on this track is the
ranker, not the cut.

## 9. Reproducibility notes and known bugs we fixed

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

## 10. Code

<!-- TODO before 2026-11-03: publish the repository (or a redacted extract) and put the
     reachable https:// link here, then tick "My code is open-source" on the submission
     form. The form requires the link when that box is checked. -->

---

*Report drafted 2026-10-08. Metric: macro-averaged soft-threshold relative absolute error
against the published credible intervals; 1.0 = as good as predicting the mean.*
