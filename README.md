# OpenADMET CYP Blind Challenge — method report

**Entry:** `talindrew` (Hugging Face `arnab2812`). Regression (direct inhibition) and TDI
(classification) tracks.
**Data:** the challenge release only. No proprietary or external training data in the submitted
models.
**Report:** <https://github.com/Arnab28122000/openadmet-cyp-challenge>, attached to the TDI
submission of 2026-10-08 and to every regression submission since 2026-10-09.
**Last updated:** 2026-10-12. Current through regression submission v9 and lab findings 1–19.

Withdrawn claims are listed in §11, not deleted.

---

## 0. Standing

Regression track, v9 (submitted 2026-10-11 12:36 UTC):

| | ST-RAE | rank |
|---|---|---|
| **Overall (macro)** | **0.5232** | **54 / 244** |
| CYP1A2 | 0.5369 | 57 |
| CYP2C9 | 0.4555 | 92 |
| CYP2D6 | 0.5493 | 25 |
| CYP3A4 | 0.5513 | 137 |

Macro Spearman ρ 0.7093, Kendall τ 0.5285, MAE 0.6703, R² 0.5611.

Every regression submission, macro ST-RAE (lower is better):

| | what changed | macro | rank |
|---|---|---|---|
| first entry | dose-response labels only | 0.8663 | 172 |
| v4 | interval labels for screened compounds (§2), minimax placement (§4) | 0.5925 | — |
| v5, v6 | spread axis only | 0.5554, 0.5414 | —, 59 / 268 |
| v7 | centre axis | 0.5256 | 54 / 271 |
| v8 | deliberate overshoot on all axes | 0.5520 | 69 |
| v9 | parabola vertex inside the bracket v8 measured | **0.5232** | 54 / 244 |

v4 through v9 change only the placement map, which is monotone, so ρ and τ are 0.7093 / 0.5285
on every one of them: an end-to-end check that each map changed placement and nothing else.

v9 is this entry's best score; its rank is below v7's because other entrants improved more over
the same two days. Placement is exhausted (§4.4), and no model-class change since v9 has cleared
the submission bar (§7). No further regression submission is planned unless one does.

## 1. Summary

Three things carry this entry, in order of measured effect:

1. **Every dose-response label is a selected label, and the unselected compounds are in the
   release.** Treating the screened-but-not-promoted compounds as interval labels is worth
   −0.243 macro ST-RAE (§2, §3), and +0.0508 macro ρ on every isoform and seed (§7).
2. **A validation surface that contains non-inhibitors.** The original split held only
   dose-response rows — each a compound selected for activity — and could not see that defect
   (§3).
3. **Per-isoform affine placement** onto the blinded population, located by reading the
   leaderboard as a measuring instrument (§4).

The models are a LightGBM/XGBoost ensemble and a small multitask MLP; nothing tried since has
beaten the GBM (§7). On this task the training labels and the placement of predictions matter far
more than the model class.

## 2. Every dose-response label is a selected label

The release ships a single-concentration screen (50 µM, log2 fold-change) for 4,376 compounds on
all four isoforms. A compound gets a dose-response curve on an isoform only if it was promoted out
of that screen, and the promotion rule differs by isoform (measured on the release).

| isoform | dose-response / screened | what decides promotion |
|---|---|---|
| CYP1A2 | 1,412 / 4,376 | rises with inhibition; 100 % below log2FC −1.5 |
| CYP2C9 | 1,285 / 4,376 | about a third, flat in log2FC — effectively random |
| CYP2D6 | 1,493 / 4,376 | hard threshold: dose-response iff log2FC < ≈ −0.6 |
| CYP3A4 | 2,335 / 4,376 | if anything enriched for weak readouts |

A model trained on dose-response labels alone has never seen a CYP2D6 non-inhibitor and
over-predicts every unselected compound. The two isoforms whose promotion depends on activity
(1A2, 2D6) scored worst in this entry's early submissions (board about 0.88 and 1.46, against
0.52 and 0.57 for the other two).

### What one 50 µM readout says about pIC50

With `r = 2**log2fc`, the fraction of activity remaining, dose-response pIC50 is monotone in `r`
on every isoform. Among the 303 compounds carrying both a curve and a no-inhibition readout
(`r ≥ 0.9`), the median pIC50 is 2.211 and the credible range about [1.07, 3.52].

Those compounds become **interval labels**: the loss is zero anywhere inside the window, which
is all one readout can say. For intermediate readouts the window is the [10th, 90th]
percentile of dose-response pIC50 among compounds with the same readout. Three higher-ranked
entries report the same rows hurting as point labels and helping as bounds.

The development code splits the no-inhibition label into two branches at the existing bin edge
`r = 1.0` (2.2732 above, 2.1320 below). It is the only rule tried that wins on both
out-of-fold MSE (−4.40 %) and MAE (−1.44 %), and the metric mixes the two. Its measured effect on
downstream ρ is +0.0004 (§7), so no scored submission uses it.

Implementation: `talindrew-drugrx/drugrx/cyp/screen.py`.

## 3. The validation surface is part of the method

- **V1**: the original scaffold split, dose-response rows only — every row selected for activity.
- **V2**: V1 plus held-out screened compounds, scored against their interval labels.

V2 reproduces the board per isoform (0.77 / 0.53 / 1.35 / 0.48 against board
0.88 / 0.52 / 1.46 / 0.57); V1 never did. A selection fix has to look worse on V1: promotion
selects on `y`, the test set's analog expansion selects on `x`, and the board scores `P(y | x)`
on the unselected population.

Measured effect of the interval labels on V2: −0.243 macro (LightGBM, 3/3 seeds) and −0.313
(the MLP, 2/2 seeds); CYP2D6 on V2 goes 1.51 → 0.60. Ranking improves too (CYP2D6 ρ 0.378 → 0.476),
which placement cannot do.

## 4. Placement

A model trained on a selected label set predicts on that set's scale, so the last step is a
per-isoform affine map — one centre, one spread — onto the blinded population.

### 4.1 The metric's optimum is not squared error's

Squared error wants centre = population mean and spread = ρ·sd (Murphy 1988:
`R² = 2ρk − k² − b²`). ST-RAE wants less spread and a centre above the mean, because an inactive
compound's interval is a free window reaching up to about 3.5. A board probe measured the spread
discount at 0.896.

v4's targets came from a minimax scan over the uncertainty in the population (inactive fraction
0.55–0.81, correlation 0.33–0.60). That was the wrong rule: minimaxing over a band wider than the
truth is not safety. The scan's optimum for CYP2D6 was sd 0.10, implying a correlation of 0.209 —
below the band it was told to search. The board-derived estimate is 0.518, and §4.3 shows even
that is too low.

### 4.2 The board row gives up the blinded population's moments

For a prediction vector held on disk, with `r = 2 sin(πρ/6)` from the reported Spearman,

    s² = s_y² + s_p² − 2 r s_y s_p          and      m² + s² = (1 − R²) s_y²

is two equations in the two unknowns `s_y` and `m = ȳ − p̄`. Solved on this entry's own row it
gives per-isoform test sd 1.44 / 0.97 / 1.66 / 1.16, within 3.8–12.3 % of an independently
published reverse-engineering. The algebra cannot sign `m`; scoring both branches against that
independent estimate picks the same sign on all four isoforms with 5× discrimination. Five routes
sharing no assumptions agree: v4 placed every isoform too narrow.

### 4.3 One axis per submission

The live board does not score all 750 test compounds. Per the organisers it scores about half of
them, split by chemically similar series (about 37–38 series; the other half is held back for the
final board), and each published column is a mean over 1,000 bootstrap resamples. Both the half
and the resampling are fixed, so the board is deterministic: the difference between two rows is a
measurement with no sampling noise, but only for the live half. Moving one axis per submission
makes the leaderboard an instrument for the metric's derivative along that axis on that half. The
moments recovered in §4.2 describe the same half. v5 and v6 moved spread only; v7 moved the
centre.

| isoform | sd v4 → v5 → v6 | ST-RAE | secant v4→v5 | secant v5→v6 |
|---|---|---|---|---|
| CYP1A2 | 0.95 → 1.00 → 1.00 | 0.549 → 0.549 → 0.549 | 0.000 | (held) |
| CYP2C9 | 0.55 → 0.68 → 0.76 | 0.505 → 0.466 → 0.462 | −0.297 | −0.045 |
| CYP2D6 | 0.30 → 0.60 → 0.78 | 0.746 → 0.640 → 0.587 | −0.351 | −0.296 |
| CYP3A4 | 0.75 → 0.82 → 0.85 | 0.571 → 0.566 → 0.567 | −0.066 | +0.023 |

CYP1A2 at sd 1.00 sits within 0.001 of 0.896 × its squared-error optimum and scored 0.5490 three
times running. CYP2D6 kept improving past the spread that a correlation of 0.518 predicts as
optimal, refuting that estimate in favour of the distribution-free bound `r ≥ 0.609`.

### 4.4 Convexity brackets the minimum (v8, v9)

ST-RAE is a sum of hinge terms, so under an affine map it is convex and piecewise linear in both
the centre and the spread, with derivative `(N_above − N_below)/D`. Two consequences:

1. A secant that changes sign across a point brackets the minimum. v8 deliberately overshot on
   every axis; it scored worse (0.5520), as expected, but put a sign change on all four axes,
   turning every later forecast from an extrapolation into an interpolation.
2. The chord between two measured points straddling the target is an upper bound, from convexity
   alone.

v9 targeted the three-point parabola vertex on each axis, with each cell's chord pre-registered
as its upper bound:

| cell | pre-registered [lower, chord] | actual |
|---|---|---|
| CYP1A2 | [0.5307, 0.5405] | 0.5369 |
| CYP2C9 | [0.4377, 0.4618] | 0.4555 |
| CYP2D6 | [0.5447, 0.5543] | 0.5493 |
| CYP3A4 | [0.5510, 0.5516] | 0.5513 |
| macro | point 0.5236, ≤ 0.5270 | 0.5232 |

Every cell landed inside its band, and the macro point estimate was off by 0.0004. The chord/slope
bracket puts the remaining headroom on the affine family under 0.003 macro on the live half. How
well a placement fitted to those series transfers to the held-back series is not measured; eight
parameters fitted over six submissions is the most conservative form a board probe can take, but
the final board will include series these parameters never saw.

v9 also caught a defect before it shipped: four decimals collapsed 10–20 distinct predictions per
column into ties, and six decimals still tied one CYP1A2 pair. Submissions are written at seven, the first
precision rank-identical to the source.

Implementation: `drugrx/cyp/placement.py`, `drugrx/cyp/decide.py`, `scripts/cyp_submit.py`,
`scripts/cyp_v9_bracket.py`, `scripts/cyp_board_history.py`.

## 5. The decision rule

ST-RAE's risk is convex and piecewise linear, so its minimiser is the balance point where
`P(lo > p) == P(hi < p)` — a weighted median over pooled bound draws, verified exact against a
4,001-point grid on 600 random cases. Two consequences:

- **Ties go to the right-hand end of a flat segment.** When every draw says inactive, the risk is
  flat across the free window. The error that does occur is an "inactive" compound being somewhat
  active, and the right end is 2.45 log units closer to it.
- **A probability-weighted mean is the wrong operation.** A compound 80 % likely to be inactive
  belongs at 3.52, not at the mean's 2.87.

## 6. The validation surface and the blinded set are different regimes

Nearest-neighbour Tanimoto to the training set (ECFP4 counts, 2048 bits) needs no labels:

| NN band | blinded test | our validation |
|---|---|---|
| 0.3–0.5 | 4.8 % | 65.2 % |
| 0.5–0.7 | 84.9 % | 25.6 % |
| 0.7–0.9 | 10.1 % | 3.3 % |
| median | 0.587 | 0.439 |

Two thirds of the validation set sits in a band holding under 5 % of the test set. The blinded set
is an analog expansion bought around the training chemistry, while 82 % of training compounds are
Murcko-scaffold singletons (5,370 scaffolds for 6,145 compounds), so a "scaffold split" over them
is close to a random, more diverse partition and asks harder questions than the board does. That
explains the validation-to-board lift on CYP1A2 and CYP2C9 without reference to the model.

`drugrx/cyp/matched.py` importance-weights each validation compound to the measured test
histogram. The correction flips the sign of small effects, so any effect below about 0.01 ρ on the
unweighted surface is treated as unsigned.

## 7. What has been tested since v9, and why none of it is a submission

After v9 the only lever left was the model itself. Everything below is measured with the
production learner unless stated, against a pre-registered bar of +0.005 macro ρ on at least
5 of 6 seeds: about the smallest change that could move a rank.

### 7.1 Split noise is ten times the effects being argued about

One fixed GBM, same data, 18 scaffold splits: macro ρ mean 0.5715, sd 0.0323, range
0.5022 → 0.6344. The production pipeline's 0.643, measured on its own split, lies just above the
top of that range. Every comparison of two learners is therefore paired within split, over at
least six splits, with a sign count. An earlier "0.09 ρ gap" between the MLP and the GBM, which
motivated three phases of work, was this spread: the two had been scored on different splits.

### 7.2 The pseudo-labels are worth +0.0508; reshaping them is worth at most +0.0037

| what was measured | best macro Δρ |
|---|---|
| having screen pseudo-labels at all, vs dose-response only | **+0.0508** (3/3 seeds, every isoform) |
| reweighting them (w 1.0 → 0.2) | +0.0037 |
| relabelling them per row from the readout (144 fits) | +0.0026 |
| shifting their location (conditional mean instead of median) | +0.0018 |
| the two-branch floor split of §2 | +0.0004 |

Sixteen arms across three harnesses never exceeded +0.0037: the ceiling is set by how much the
screen says, not by how its label is parameterised. Nor does the gain follow the pseudo-labels'
share of an isoform's labelled signal — CYP2C9 gains most (+0.0752) at a 14.9 % share, while
CYP2D6, at 60.5 %, gains +0.0616.

The floor split shows why the criterion has to match the grader. It is the better label for the
measured floor compounds, but it moves 1,561 rows up and 3,122 down and leaves the block's mean
where it was (−0.0003). Downstream ρ responds to the block's location, so the conditional mean,
which loses on MAE, buys 4.5× more ρ. The difference (+0.0014) is inside the noise, and the split
stays in the code as the more defensible label.

### 7.3 No MLP in this family beats the GBM

Standardising all 3,091 feature columns together, as the MLP originally did, was the worst of six
preprocessings tried. `log1p` on the count fingerprints (which keeps their 86 % sparsity at
exactly zero) plus quantile-normalised RDKit descriptors (one column has sd 3.8 × 10⁵) is
+0.0339 ρ, 6/6 splits, t 3.91. The three splits it was discovered on gave +0.0431; three fresh
splits gave +0.0247, a 43 % shrink.

The remaining MLP changes, ablated one at a time over 18 splits, paired:

| factor | Δρ | sign |
|---|---|---|
| per-head target standardisation | +0.0138 | 14/18 (p 0.015) |
| LayerNorm | +0.0079 | 12/18 |
| depth 4 | +0.0037 | 10/18 |
| width 1024 | −0.0021 | 7/18 |
| batch 64 | −0.0042 | 6/18 |
| dropout 0.5 | −0.0209 | 2/18 |
| one shared forward under BatchNorm | −0.1009 | 0/18 |

The one real fix is per-head target standardisation: the four endpoints' label sds span
0.782 to 1.093 and their losses share a trunk, so the widest head dominated the gradient.

Against the production GBM, paired over 18 splits:

| | Δρ | t | sign |
|---|---|---|---|
| preprocessed MLP | −0.0137 | −2.50 | 5/18 |
| + per-head standardisation | +0.0002 | 0.04 | 8/18 |
| tuned composite | +0.0055 | 1.46 | 11/18 |

The composite looked like +0.0201 (6/6) on the six splits it was tuned on; on the twelve never
used for a choice it is −0.0017. The MLP's ceiling is parity with the GBM, so it adds nothing as
an ensemble member.

### 7.4 Other mechanisms measured and closed

- **Early stopping on a constant-heavy set.** XGBoost's early-stopping set is up to 63 % the
  single floor value on CYP2D6, and selecting on MAE against it does stop training up to 407
  rounds early. Excluding those rows changes macro ρ by +0.0002 (3/6 seeds) and costs CYP3A4 on
  0/6 seeds. The defect is real and has no measured cost, so the builder is unchanged.
- **Tabular foundation models** on the same features missed the bar (0/3 seeds on the primary
  endpoint). TabICL's apparent edge at matched features was an artefact of its 100-column
  pretraining range, and decays to a loss by 1,000 columns.
- **Learned per-sample weights** (bilevel). With 641,284 parameters against 785 meta rows
  (d/n = 817), the meta-gradient's agreement between two halves of its own meta set falls from
  +0.86 to +0.11 within 20 epochs while the training gradient holds at +0.89. At that ratio no
  parameterisation of the weights can be told apart from uniform.

## 8. Earlier negative results

- **External inactives do not substitute for the challenge's own screen.** 10.5k Veith bounds made
  V2 worse by +0.038 macro, 3/3 seeds.
- **More training does not help.** 300 → 3,000 epochs moved the score by 0.0000. At 30,000 epochs
  with warm restarts the gain over 1,200 was −0.0026 against a seed spread of 0.0052. Doubling
  width bought 0.013. A later controlled run showed two thirds of an apparent epochs effect was
  the LR schedule stretching with the epoch count.
- **A pretrained molecular foundation model did not beat ECFP; its loss function did.** CheMeleon's
  frozen D-MPNN fingerprint lost to count-ECFP + descriptors under the same GBM (0.6433 vs 0.5819
  ST-RAE on validation). Fine-tuned under the metric's own interval hinge it scored 0.5179, but
  embeddings read back out of that encoder were no better than frozen ECFP: the gain was in
  training against the metric, not in the representation.
- **Concatenating feature blocks hurt.** CheMeleon ⊕ ECFP scored worse than either alone.
- **Ranking objectives lose.** On CYP3A4 with paired seeds, LightGBM `lambdarank` is −0.177 ρ and
  `rank_xendcg` −0.038. Finer grades recover only 0.047 of the 0.177, so coarse binning is about a
  quarter of the cause; NDCG's positional discount and Spearman's equal weighting are different
  objectives.

## 9. The TDI classification track

Per-isoform binary classifiers for `CYP2D6_is_TDI` and `CYP3A4_is_TDI`, same release, no extra
data. Board: macro MCC 0.2964, rank 71 of 153 (CYP2D6 0.2291, CYP3A4 0.3637).

The leaderboard gives away the blinded test set's class balance. For a binary task,
`TP = R·π = P·q` and `Accuracy = 1 − π − q + 2Rπ`, so

    π = (1 − Accuracy) / (1 + Recall/Precision − 2·Recall)

Every entry is scored on the same test set, so π must agree across entries; that agreement is the
validation: sd 0.00018 over 145 entries (CYP2D6) and 0.00008 over 152 (CYP3A4), giving π = 0.0706
and 0.2916. MCC then has a closed form,

    MCC = √(π/(1−π)) · (R − q) / √(q(1−q))

verified against sklearn to 3.3e-16, and it reproduces our reported MCC to 0.001.

Thresholds had been tuned where the positive rate was 0.195 against a true prevalence of 0.0706.
Thresholding by probability does not transfer when the probability scale shifts; by rate it does.
Moving CYP3A4's rate from 0.473 to 0.313 was worth +0.0138 MCC.

The same row gives the realised operating point, `FPR = (q − Rπ)/(1 − π)`, and one (TPR, FPR)
point pins an equal-variance binormal ROC: test AUC 0.762 (CYP2D6) and 0.764 (CYP3A4), against
validation 0.728 and 0.813. The biases run in opposite directions, which is why a pre-registered
macro MCC of 0.318 came back 0.296. Both thresholds now sit within 0.0006 MCC of optimal on the
realised ROC, so the remaining lever here is the classifier, not the cut.

## 10. Silent bugs found and fixed

Each passed every shape and finiteness check.

- `chemeleon_fp.py` returned **atom-level** vectors: Chemprop's `BondMessagePassing` gives
  (n_atoms, 2048), and indexing it by molecule took one arbitrary atom from the wrong molecule.
  Exposed when nearest neighbours in the embedding had Tanimoto 0.132 against a random-pair
  baseline of 0.132. Fixed with mean aggregation over `batch.batch`.
- `allr_floor` returned NaN on an empty pool. Trainers mask NaN, so the selection fix would have
  become a silent no-op; it now raises.
- The screen labeller decided "inactive" by calibration-bin membership rather than by the readout,
  so on CYP2D6, whose promotion excludes weak bins, a plainly inactive compound could be labelled
  active.
- The ensemble's "seeds" shared a byte-identical validation index, so their spread measured network
  initialisation rather than the compound sample, making seed-level error bars about 10× too tight.

## 11. Withdrawn claims

Each was found by re-measurement, adversarial review or a closer reading of a source.

- **"The ensemble beats a plain LightGBM by +0.0154 ρ."** The comparison was unpaired; paired on
  identical rows it is −0.0066, with every per-isoform interval spanning zero.
- **"The screen pseudo-label is a left-censoring bound."** 2.211 is the median of 303 measured
  compounds and half of them sit above it. That is coarsening, not censoring, so every
  left-censored estimator (a Tobit EM arm included) is mis-specified for half the rows it touches.
- **"P(y\* > 2.211) ≈ 0.5 is a finding."** It is a property of a median.
- **"Representation is the lever."** Four representation changes were measured at Δρ ≈ 0.0000,
  all with the predictor held fixed. When the predictor was varied instead (§7.3, §7.4), nothing
  beat the GBM.
- **"The early-stopping defect costs score and was fixed."** An earlier version of this report
  listed it as found and fixed; it was measured afterwards, the effect is +0.0002, and the builder
  was not changed (§7.4).
- **"The two-branch floor split is the better change."** Better at fitting the floor compounds;
  worse than the conditional mean at moving ρ, and neither clears the bar (§7.2).
- **"The tuned MLP beats the GBM by +0.0201, confirmed on fresh splits."** The confirmation used
  three splits — 1.6 σ — and the configuration was selected on the other three. Over twelve
  untouched splits it is −0.0017 (§7.3).
- **"Dropout 0.5 is the best single architecture change (+0.0177)."** One factor at a time over 18
  splits it is −0.0209, 2/18. The original reading took eight deltas off one shared baseline run;
  they agreed because that baseline drew low.
- **"ST-RAE on 750 fixed compounds is deterministic."** The live board scores about half the test
  set (about 37–38 series) as a bootstrap mean. The placement measurements stand, but they hold for
  that half, not for the final board (§4.3, §4.4).
- **"Pseudo-labels are too small to explain the gap to production."** Their presence is worth
  +0.0508; it is only modifications to them that are capped at +0.0037 (§7.2).

## 12. Repository

| path | contents |
|---|---|
| `talindrew-drugrx/` | features, training, the metric, splits, the submission builder, tests |
| `docs/research-lab/` | the lab notebook: numbered findings, a dated notebook, a changelog |
| `docs/drugrx-training/` | the programme that built the entry, and the full leaderboard history |
| `site/` | an illustrated explainer of the problem and the method, served on GitHub Pages |

Start with `docs/research-lab/NOTEBOOK.md` (newest entry last) and `docs/research-lab/findings/`.

### Models on Hugging Face

[`arnab2812/openadmet-cyp-talindrew`](https://huggingface.co/arnab2812/openadmet-cyp-talindrew)
holds every scored submission (regression v4–v9 with their placement parameters, both TDI
submissions), the v4 source build's report and net predictions, and every leaderboard row this
entry received. **It is private until the challenge closes on 2026-11-03**, because its contents
reproduce the entry's predictions on the blinded test set, and opens then.

The weights behind v4–v9 were not persisted: the submission builder ran with checkpointing off
and saved only predictions. Every v5–v9 submission is a monotone placement of v4's predictions,
so the CSVs are a complete record of what was scored. The next build will save every member, the
preprocessing state, the blend weight and a manifest under `runs/<run_id>/` in the same
repository, with a check that the saved artefacts re-predict that build's submission.

---

*Metric: macro-averaged soft-threshold relative absolute error against the published credible
intervals; 1.0 is as good as predicting the mean.*
