# 27 — Competitor intel on the OpenADMET CYP Blind Challenge

Read 2026-10-08 (second pass, afternoon). Supersedes nothing; **extends** `22-challenge-winners-and-intel.md`
(programme context, past challenges) and `24-leaderboard-model-reports.md` (13 model reports read in the
morning). Everything here is new or re-measured.

**Attribution correction.** The challenge is run by **OpenADMET (an OMSF project)** with **Octant** doing
the assays and the **UCSF Fraser lab** the cryo-EM structures. **ASAP Discovery is not a CYP-challenge
organiser** — it co-ran the earlier 2025 ASAP × Polaris × OpenADMET antiviral challenge, which is a
different event. Funders named: ARPA-H, Radial (Astera Institute), Schrödinger, the Gates Foundation;
partners Enamine, HuggingFace, OpenEye, CDD Vault, Discovery Life Sciences, NSLS-II.

What this pass added that the earlier two files did not have:

1. **The complete live and interim leaderboards, all 265 / 219 rows, extracted from the Space's own
   Gradio payload** — not a hand scrape. Full score distribution, per isoform, plus every Model Report URL.
2. **The actual scoring code**, read line by line, from two public repos. The metric's denominator is
   prediction-independent; the loss is exactly an interval hinge; the bootstrap seed is `0`.
3. **The credible-band geometry, computed from the training file we already hold.** This is the single
   biggest reframe in this file: the tolerance band is 6–12× wider for inactives than for potent
   compounds, so the metric is decided almost entirely on the potent tail.
4. **How the 750 test compounds were chosen, in the organisers' own words** — and the fact that
   **CYP2D6 was never used to select them**, plus the fact that 2D6 is measured on a *different assay
   platform*. Together these explain the 2D6 anomaly that files 22 and 24 could only describe.
5. **Scigantic — rank 19 live / rank 11 interim — has a full public write-up that is not linked on the
   leaderboard.** It is the highest-ranked published method in the challenge, better than briford.
6. **briford has published the solved blind-half mean and sd for all four isoforms** (§9.2). The blind
   CYP2D6 mean is **3.107 against train's 4.784** — a 1.68-log shift. Four entrants independently report
   **0.15–0.29 macro gained from placement alone**, the largest single effect in the challenge.

> ### ⚠ Two things that are time-critical, ahead of any modelling
>
> 1. **A comprehensive model report is mandatory**, or we are excluded from the final leaderboard
>    regardless of score — the organisers' words, §9.1. Our row has no report link. Deadline
>    **2026-11-03 23:59 UTC**.
> 2. **We are not on the interim board at all** (§1.6), having missed the 24 Sep deadline, so we have no
>    full-750 datapoint and no tier. The final board scores the full 750.

---

## 1. The board, 2026-10-08

Extracted from `https://openadmet-cyp-challenge.hf.space/config`, which embeds every leaderboard as a
`gradio_leaderboard` component value. Component ids: 58 = live master, 61/64/67/70 = live
1A2/2C9/2D6/3A4, 92 = interim master (with tiers), 95/98/101/104 = interim per isoform. Saved as
`lb_<id>.csv` in the session scratchpad.

Live = **half** the test set, split by chemisimilar series. Interim (one-off, 25 Sep) = the **full** 750.

### 1.1 Live master, MA-ST-RAE (lower is better), n = 265

| Rank | Alias | MA-ST-RAE | MA-MAE | MA-R² | MA-ρ |
|---|---|---|---|---|---|
| 1 | preheat-to-450 | **0.3706** | 0.5919 | 0.6648 | 0.7844 |
| 2 | wbot | 0.3776 | 0.6063 | 0.6419 | 0.7775 |
| 3 | beast | 0.3961 | 0.6407 | 0.6060 | 0.7804 |
| 4 | cypadeedoodah | 0.4132 | 0.6222 | 0.6249 | 0.7770 |
| 5 | EleanorRigby | 0.4139 | 0.6510 | 0.6007 | 0.7721 |
| 6 | SVM | 0.4195 | 0.5938 | 0.6238 | 0.7745 |
| 7 | hfger | 0.4222 | 0.6222 | 0.6307 | 0.7605 |
| 8 | Queimo | 0.4273 | 0.6299 | 0.6192 | 0.7613 |
| 9 | furtivepeony | 0.4301 | 0.6273 | 0.6269 | 0.7626 |
| **10** | leebrubeck | **0.4315** | 0.6180 | 0.6331 | 0.7604 |
| 11 | curiousone | 0.4354 | | | |
| 12 | first-chair | 0.4367 | | | |
| 13 | briford | 0.4378 | | | |
| 14 | hf-far | 0.4419 | | | |
| 15 | C_CYPher-cypext-B1cal-restore | 0.4428 | | | |
| 16 | z | 0.4436 | | | |
| 17 | rasayan-labs | 0.4443 | | | |
| 18 | ADMETForge | 0.4452 | | | |
| 19 | **Scigantic** | 0.4456 | | | |
| 20 | random-forest | 0.4470 | 0.6409 | 0.6109 | 0.7507 |
| 25 | Tutu | 0.4617 | | | |
| 30 | tguttenb1 | 0.4737 | | | |
| 40 | Moyu | 0.4897 | | | |
| **50** | jeremy | **0.5185** | 0.6285 | 0.6024 | 0.7343 |
| 60 | harinu123 | 0.5427 | | | |
| 75 | tmp1234 | 0.6145 | 0.8196 | 0.3316 | 0.7622 |
| 100 | Creme | 0.6915 | 0.8785 | 0.2707 | 0.6598 |
| 125 | alchemy | 0.7611 | | | |
| 133 (median) | ADMET Bo$$ | **0.7726** | 0.9316 | 0.1672 | 0.6829 |
| 150 | coolbeans | 0.8292 | | | |
| **173** | **talindrew** | **0.8663** | 0.9829 | **0.0748** | |
| 175 | CYP-GNN-19 | 0.8701 | 1.0086 | 0.0608 | 0.6437 |
| 200 | ADI | 0.9158 | | | |
| 225 | FreeT4-CYP | 0.9564 | | | |
| 250 | ripper | 1.0793 | 1.1316 | −0.1726 | 0.2357 |
| 265 | apple1 | 7.0296 | 4.4485 | −42.4598 | 0.6152 |

**Answer to "what score is rank N":** rank 1 = **0.3706**, rank 10 = **0.4315**, rank 50 = **0.5185**,
median (rank 133) = **0.7726**.

Percentiles: p5 = 0.4419, p10 = 0.4661, p25 = 0.5625, p50 = 0.7726, p75 = 0.9144, p90 = 1.0015.
Counts: **20 entries below 0.45, 43 below 0.50, 72 below 0.60, 102 below 0.70, 238 below 1.00.**

**Our row has recovered.** PROGRESS's last note was 0.9628 at rank 225/260. The live board now shows
**talindrew 0.8663, rank 173 / 265** — back to roughly the v3deep level (0.8598). Still 0.496 behind the
leader and 0.094 worse than the median.

### 1.2 Per isoform, live

| Isoform | leader | leader alias | rank 10 | rank 50 | median | p25 | % worse than a constant | **ours** |
|---|---|---|---|---|---|---|---|---|
| CYP1A2 | 0.3812 | beast | 0.4250 | 0.5368 | 0.7435 | 0.5701 | 3% | 0.9070 |
| CYP2C9 | 0.3056 | preheat-to-450 | 0.3540 | 0.4075 | 0.5277 | 0.4317 | 3% | 0.5810 |
| **CYP2D6** | **0.4135** | preheat-to-450 | 0.4889 | 0.6203 | **1.3501** | 0.6939 | **64%** | **1.3827** |
| CYP3A4 | 0.3515 | preheat-to-450 | 0.3896 | 0.4447 | 0.5547 | 0.4702 | 3% | 0.5946 |

Leader secondary metrics: 1A2 MAE 0.6441 / R² 0.6705 / ρ 0.8461 · 2C9 MAE 0.3722 / R² 0.7540 / ρ 0.8808 ·
2D6 MAE 0.8930 / R² 0.4773 / ρ 0.5428 · 3A4 MAE 0.4338 / R² 0.7727 / ρ 0.8741.

**2D6 is bimodal, not uniformly hard.** Median 1.3501 but p25 0.6939 and leader 0.4135. 64% of the field
is worse than predicting a constant; the quarter that is not is *far* better. That is the signature of a
single discrete trick (location/scale placement) that either has or has not been applied — not of a
gradually harder regression problem. Our 1.3827 sits just above the median, i.e. squarely in the
"has not been applied" group.

Our 2C9 (0.5810) and 3A4 (0.5946) are now *worse* than the per-isoform medians (0.5277, 0.5547) —
the morning read had them at or better than median. 1A2 0.9070 vs median 0.7435 and 2D6 1.3827 vs
1.3501 are the big holes, but all four cells are now below median.

### 1.3 The organisers' own baselines are on the board, with ranks

Every `openadmet-models` baseline is submitted under its own alias, which gives an exact public ladder:

| Rank | Alias | MA-ST-RAE |
|---|---|---|
| 95 | **TabICL-baseline** | **0.6755** |
| 142 | baseline_icl_chemprop | 0.8082 |
| 143 | tabicl_moljepa_cyp_no_ft | 0.8096 |
| 145 | LightChemeleon | 0.8140 |
| 153 | **CheMeleon-baseline** | **0.8335** |
| 165 | chemfm-baseline | 0.8529 |
| **173** | **talindrew** | **0.8663** |
| 186 | LGBM-baseline | 0.8929 |
| 190 | XGB-baseline | 0.8975 |

The blog's "0.68" is the live board's 0.6755 at **rank 95**, and "0.83" is 0.8335 at rank 153. **We are
currently below CheMeleon-baseline and chemfm-baseline, and above only the LGBM/XGB baselines.** Beating
the TabICL-baseline would, on its own, move us from 173 to ~95.

### 1.4 Per isoform for the entries that matter

| Alias | rank | 1A2 | 2C9 | **2D6** | 3A4 |
|---|---|---|---|---|---|
| preheat-to-450 | 1 | 0.4120 | **0.3056** | **0.4135** | **0.3515** |
| wbot | 2 | 0.3846 | 0.3354 | 0.4278 | 0.3626 |
| beast | 3 | **0.3812** | 0.3785 | 0.4377 | 0.3868 |
| Scigantic | 19 | 0.4739 | 0.3833 | 0.5442 | 0.3809 |
| briford | 13 | 0.4483 | 0.3632 | 0.5502 | 0.3895 |
| **talindrew** | **173** | 0.9070 | 0.5810 | **1.3827** | 0.5946 |

**The decisive observation in this file.** The leader is not best on 1A2 (beast is) and wins on 2C9 and
2D6. The top three all reach **2D6 ≈ 0.41–0.44**, while the two best *published* methods sit at
**0.54–0.55** — a 0.11–0.14 gap on the one isoform that is 25% of the macro. On 1A2/2C9/3A4 the published
methods are within 0.02–0.09 of the top three. **So whatever separates the undisclosed top entries from
the best published ones is specific to CYP2D6**, which is also the isoform that was never used to select
the test compounds and the only one measured on a different instrument (§5).

### 1.5 The blind set's label spread, backed out of the leaders' own metrics

For each top entry, `sigma_resid = MAE / 0.7979` (folded normal) and `sd_y = sigma_resid / sqrt(1 − R²)`.
Across the top 8 per isoform the estimates are tight, which is what makes them usable as placement targets:

| Isoform | consensus **blind sd_y** | spread across top 8 | train sd (§4.1) |
|---|---|---|---|
| CYP1A2 | **1.407** | 1.366 – 1.455 | 1.030 |
| CYP2C9 | **0.942** | 0.925 – 0.957 | 0.782 |
| CYP2D6 | **1.565** | 1.518 – 1.581 | 0.916 |
| CYP3A4 | **1.129** | 1.119 – 1.147 | 1.093 |

**The blind set is more spread out than the training set on every isoform, and on CYP2D6 it is 71% wider
(1.565 vs 0.916).** A model fitted to train and applied unchanged is therefore under-dispersed on the blind
set by construction — most of all on 2D6 — and §4.3 prices that shrinkage at 0.05–0.25 ST-RAE. This is the
quantitative core of why placement is worth more than architecture here, and it is independently derivable
from public numbers without any held-out label.

Caveat on the method: `sigma = MAE/0.7979` assumes Gaussian residuals. Real residuals are heavier-tailed,
which inflates MAE relative to sigma less than it inflates MSE — so these are mild **under**-estimates of
`sd_y`, most of all on 2D6 where R² is lowest (0.48) and the tails are fattest. Treat 1.565 as a floor.
The agreement to ±0.03 across eight independent submissions is evidence the *relative* ordering is sound.

### 1.6 Interim (full 750), top 12 with significance tiers

| Rank | Alias | Tier | MA-ST-RAE |
|---|---|---|---|
| 1 | preheat-to-450 | **Tier 1 (alone)** | 0.3721 |
| 2 | wbot | Tier 2 | 0.3912 |
| 3 | cypadeedoodah | Tier 2 | 0.3921 |
| 4 | Rubyy | Tier 2 | 0.4014 |
| 5 | EleanorRigby | Tier 3 | 0.4173 |
| 6 | beast | Tier 3 | 0.4263 |
| 7 | leebrubeck | Tier 4 | 0.4336 |
| 8 | hfger | Tier 4 | 0.4357 |
| 9 | first-chair | Tier 4 | 0.4362 |
| 10 | furtivepeony | Tier 4 | 0.4370 |
| 11 | **Scigantic** | Tier 4 | 0.4382 |
| 12 | psk | Tier 4 | 0.4387 |

Interim master distribution (n = 219): rank 1 = 0.3721, rank 10 = **0.4370**, rank 50 = **0.5188**,
median = **0.7064**. **41 tiers**, sizes from the top: 1, 3, 2, 12, 6, 9, 10, 4, 4, 7 — so Tier 1 is a
single entry and the first 18 entries occupy four tiers.

Interim per isoform (full 750 — this is the best available predictor of the **final** board):

| Isoform | leader | rank 10 | rank 50 | median | p25 | % worse than constant |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.3752 | 0.4081 | 0.4944 | 0.6673 | 0.5091 | 3% |
| CYP2C9 | 0.3130 | 0.3615 | 0.4229 | 0.5125 | 0.4354 | 2% |
| **CYP2D6** | **0.4568** | 0.5308 | 0.6974 | **1.1794** | 0.7169 | **60%** |
| CYP3A4 | 0.3331 | 0.3629 | 0.4250 | 0.5144 | 0.4346 | 4% |

Live-vs-interim agreement for the leader is 0.3706 vs 0.3721 (0.4%), so neither half is systematically
easier — but it is a *different set of series*, which is a different risk. `Rubyy` is top-4 interim but
absent from the live top 20 — an entry that has not resubmitted. **talindrew does not appear on the interim
board at all**, having missed the 24 Sep interim deadline, so we have no full-750 datapoint of our own and
no tier.

---

## 2. Who actually publishes a method

**24 of 265 live entries have a Model Report link, and the best of them is rank 13.** Ranks 1–12
(preheat-to-450, wbot, beast, cypadeedoodah, EleanorRigby, SVM, hfger, Queimo, furtivepeony, leebrubeck,
curiousone, first-chair) publish **nothing at all**. There is no way to read the winning method; it does
not exist in public. Any "what the top entries do" claim has to be inferred from ranks 13+ and from the
secondary metrics of the top rows.

| Rank | Alias | MA-ST-RAE | Open code | Report URL |
|---|---|---|---|---|
| 13 | briford | 0.4378 | Yes | https://supercowpowers.github.io/workbench/blogs/cyp_challenge/ |
| 17 | rasayan-labs | 0.4443 | No | https://github.com/whis9/rasayan-cyp (repo empty) |
| **19** | **Scigantic** | **0.4456** | — | **not linked on the board** → https://github.com/Scigantic/open-admet-challenge-cyp |
| 35 | stir_bar | 0.4769 | No | https://github.com/lachrymator/openadmet-cyp-challenge-public |
| 44 | alexxi11 | 0.5069 | Yes | https://github.com/datagrok-ai/admetica |
| 49 | JacksonBurns | 0.5181 | Yes | https://github.com/JacksonBurns/cyp-challenge |
| 50 | jeremy | 0.5185 | Yes | https://github.com/jeremycheminf/openadmet_scripts/tree/main/CYP_Challenge |
| 55 | Lizard Wizard Gizzard | 0.5357 | Yes | https://github.com/doctawho42/CYPChallenge/blob/main/docs/method_report.md |
| 73 | 450nm | 0.6004 | No | https://openadmet-cyp-dashboard.vercel.app/operating_dashboard |
| 83 | CompChem Magicano | 0.6404 | Yes | https://github.com/mokhalaf22/openadmet-cyp-public |
| 90 | adlvdl | 0.6578 | Yes | https://github.com/adlvdl/cyp_competition |
| 123 | leeherman | 0.7585 | No | Google Doc |
| 128 | popcorn2 | 0.7671 | No | https://github.com/etiu/cyp_challenge |
| 133 | ADMET Bo$$ | 0.7726 | No | https://github.com/Rebell-Leader/OpenADMET_CYP/blob/main/paper/PAPER_DRAFT.md |
| **146** | **Woong** | **0.8141** | No | https://gist.github.com/ByungwoongYoo/4904441a9b30db077fbb784f9c22e423 |
| **156** | **AIv_2709** | **0.8373** | Yes | https://github.com/avnikonenko/openadmet_challenge_cyp_2026/blob/main/docs/method_report.md |
| 166 | ademmanlincoln07 | 0.8551 | Yes | https://github.com/AdemolaTolulope/cyp-inhibition-profiler |
| 215 | Waz-Dogz9000 | 0.9437 | Yes | https://github.com/repowazdogz-droid/omega-openadmet-cyp-challenge |
| 224 | nkwork | 0.9554 | Yes | https://github.com/nkwork9999/mollibra-challenges/blob/main/reports/openadmet_method.md |
| 225 | FreeT4-CYP | 0.9564 | No | https://huggingface.co/spaces/momo00798/freet4-cyp-method |
| 229 | Jackson Yamvwa | 0.9603 | Yes | https://github.com/Jackson-yamvwa/DrugDiscovery-Py |
| 238 | SoftMachine | 0.9946 | Yes | https://github.com/acnash/Strange-Matter-Engine |
| 251 | curiouswavefn | 1.0850 | Yes | https://github.com/ajogalekar/openadmet-cyp-challenge/blob/main/METHODS.md |
| 255 | HBI | 1.1173 | No | https://huggingface.co/buildingpeps/hbi-cyp-challenge-v2 |
| 258 | dargason | 1.1609 | Yes | https://admet-cyp.aetherark.com/regression-submissions.html |

Every entry declares **Proprietary Data = No**. Nobody is winning with private data.

Two further public repos exist that are *not* linked from any board row and whose alias we could not
match: `https://github.com/codie-freeman/OpenADMET-CYP-Blind-Challenge` (a very large agent-driven repo —
295 KB `CLAUDE.md`, a 65 KB `docs/leaderboard_submissions.md`, Butina and analog-holdout fold files,
Chemprop/CheMeleon screen logs, a Mordred-PCA pipeline) and
`https://github.com/mileshuangg-alt/tripwire-project`.

### 2.1 The anti-cheat list, which is itself intel

`utils.py` in the Space carries a hard-coded `BANNED_USERNAMES` dict of **~70 accounts**, almost all
"Suspected of using multiple HF accounts", grouped by cluster with comments. Notable entries:

- Reinstated after explanation: `teamprescience`, `safi029`, `novice11` ("Suspicious entries. Explained,
  reinstated"), and the pair `al0084279` / `j10b84` — "Reproduced j10b84's public repo, reinstated".
- `mzkchem` banned with the comment "**psk (submitted rasayan-labs' unsubmitted predictions)**" — note
  that `psk` is still rank 12 on the interim board.
- A cluster comment reads "**Daenerys / linda single-compound probe**" and another "**samsung / PCA
  alias-rotation cluster**" — i.e. the organisers detected and banned people probing the metric with
  single-compound or constant-column submissions.

`MIN_PREDICTION_STD = 0.01` rejects constant columns outright, "with or without slight noise added",
because "constant columns can be used to game the leaderboard metric". Rate limit: one submission per
**12 h**; only the latest valid submission counts.

---

## 3. The scoring harness — exact, open source, and more exploitable than it looks

Two public copies, near-identical (14,174 vs 14,395 bytes):

- Participant copy: `OpenADMET/CYP-Challenge-Tutorial` → `evaluation/custom_scoring_functions.py`,
  `evaluation/config.py`, `evaluation/evaluate_predictions.py`.
  https://github.com/OpenADMET/CYP-Challenge-Tutorial
- **The real backend**: `OpenADMET/blind-challenge-template` → `backend/custom_scoring_functions.py`,
  `backend/evaluate_predictions.py`, `backend/leaderboard.py`, `backend/utils.py`,
  `docs/scoring-and-leaderboards.md`, plus tests. https://github.com/OpenADMET/blind-challenge-template
  The Space's `config.py` repeatedly says "Must mirror `backend.config.X`", confirming this is the same code.

### 3.1 The metric, verbatim

```python
def rae_soft_threshold_absolute_error(y_true, y_pred, y_true_upper=None,
                                     y_true_lower=None, confidence_interval=None):
    y_true_lower, y_true_upper = _resolve_bounds(...)
    above_upper   = np.clip(y_pred - y_true_upper, a_min=0, a_max=None)
    below_lower   = np.clip(y_true_lower - y_pred, a_min=0, a_max=None)
    soft_abs_error = above_upper + below_lower

    mean_true = np.mean(y_true)
    baseline_above_upper = np.clip(mean_true - y_true_upper, a_min=0, a_max=None)
    baseline_below_lower = np.clip(y_true_lower - mean_true, a_min=0, a_max=None)
    soft_baseline_error  = baseline_above_upper + baseline_below_lower

    return np.sum(soft_abs_error) / np.sum(soft_baseline_error)
```

Bounds come from `{endpoint}_conf_high` / `_conf_low` (`REGRESSION_CREDIBLE_INTERVALS_*_SUFFIX`), which
ship in the training CSV and are held back for test.

### 3.2 Eight consequences that matter for optimisation

1. **The denominator does not depend on the prediction.** It is a function of `y_true` and the bands only.
   So **minimising ST-RAE is *exactly* minimising `Σ soft_abs_error`** — an unweighted sum, in pIC50
   units, of a two-sided hinge. There is no per-compound normalisation and no variance term. Train on
   that hinge and you are training on the metric.
2. **The loss is piecewise linear with gradient ±1 outside the band and 0 inside.** Its minimiser is a
   *median-like* quantity, not a mean. An L2-trained model is solving a different problem; this is the
   algebraic root of the "placement is a separate lever" finding already in the programme's memory.
3. **No clipping of predictions anywhere.** `ENDPOINTS_TO_LOG_TRANSFORM` is empty for CYP, so
   `clip_and_log_transform` never runs. Nothing stops an absurd prediction — see rank 265 at 7.03.
4. **Missing ground truth is dropped per endpoint**, not per row: `has_ground_truth = ~pd.isna(y_true)`
   filters `y_pred`, `y_true` and both bound arrays. So the four isoforms are scored on **different
   subsets of different sizes**, and a compound untested for 2D6 costs nothing there.
5. **A missing *prediction* for a compound that has ground truth raises**, and a submission containing any
   NaN is rejected at validation (`nullable=False`). You must predict all 750 × 4.
6. **Merge is `how="right"` on `Molecule_Name` alone.** Not SMILES. Row order is irrelevant; the name is
   the join key.
7. **The baseline is the *resample* mean, not the full-set mean.** The metric is invoked as
   `metric_func(y_true[idx], y_pred[idx], y_true_upper=y_true_upper[idx], y_true_lower=y_true_lower[idx])`,
   so `mean(y_true)` inside it is the mean of that bootstrap resample and the **denominator varies from
   sample to sample**. The reported figure is therefore a **mean of 1000 ratios**, not a ratio of totals —
   which makes the effective per-compound weighting mildly non-uniform.
8. **The bootstrap is fully deterministic**: `BOOTSTRAP_SEED = 0`, `BOOTSTRAP_SAMPLES = 1000`,
   `np.random.default_rng(0).choice(n, size=(1000, n), replace=True)`, and the docstring says "The random
   seed is fixed so all submissions are evaluated on the same bootstrap samples." Because the reported
   number is the **mean over 1000 ratios-of-sums** rather than one ratio of totals, the effective
   per-compound weight is slightly non-uniform *and exactly reproducible given n*. Anyone who knows the
   per-isoform `n` can regenerate the identical index matrix.

Macro: `groupby("Sample")[metric].mean()` — a **plain arithmetic mean** across the four endpoints computed
*per bootstrap sample*, then averaged. (Spearman was previously Fisher-z averaged; they removed it
deliberately, documented at length.) Because the mean of a mean is linear, **the reported MA-ST-RAE equals
the plain average of the four per-endpoint bootstrap means.** All four isoforms carry exactly 25%.
`METRIC_NAN_FALLBACK` gives Spearman/Kendall 0.0 on a zero-variance sample; **ST-RAE has no fallback and a
non-finite value raises**, failing the whole submission.

### 3.3 Tiers (interim and final only)

`backend/leaderboard.py`. Pairwise, on the **same 1000 fixed bootstrap samples**, merged on `Sample`:

```python
p = 2 * min((diff > 0).mean(), (diff < 0).mean())     # two-sided sign test on paired deltas
if diff.abs().sum() == 0: p = 1.0
```

`alpha_threshold = 0.05`, Benjamini–Hochberg (`alpha * i / m`, i = 1-based rank of the sorted p-value,
m = total comparisons), stepping down to the largest i with `p_i <= threshold`. Tiers are then
"**Localized Sequential Chaining**" (`generate_tiers`): walk the ranked board, the current tier's first
entry is the active baseline, keep adding until an entry differs significantly from *that baseline*, which
then starts a new tier and becomes the new baseline. Per-endpoint boards get **no** significance testing.

Consequence: with m = C(219,2) = 23,871 comparisons, the smallest usable p from a 1000-sample sign test is
0.002, and BH makes p = 0.002 significant only at rank i ≥ ~955. Tiering is therefore permissive — Tier 1
holding exactly one entry at interim means preheat-to-450 beat wbot in essentially every one of the 1000
paired resamples.

---

## 4. The credible-band geometry — computed, and it reframes the problem

Computed here from `cyp-challenge-TRAIN_inhibition.csv` (4,905 rows, 18 columns), the file we already
hold. **Not previously written down in this programme.**

### 4.1 Band width is a function of potency

| Isoform | n labelled | y mean | y sd | **median band width** | mean | p10 | p90 |
|---|---|---|---|---|---|---|---|
| CYP1A2 | 1412 | 4.955 | 1.030 | **0.328** | 0.583 | 0.179 | 1.441 |
| CYP2C9 | 1285 | 4.581 | 0.782 | **0.526** | 0.708 | 0.227 | 1.241 |
| CYP2D6 | 1493 | 4.784 | 0.916 | **0.272** | 0.492 | 0.170 | 0.949 |
| CYP3A4 | 2335 | 4.096 | 1.093 | **0.379** | 0.878 | 0.131 | 2.523 |

Median band width **by potency bin** — this is the important table:

| Isoform | y ≤ 4 | 4–4.5 | 4.5–5 | 5–5.5 | 5.5–6 | y > 6 |
|---|---|---|---|---|---|---|
| CYP1A2 | **2.09** (n=232) | 0.73 | 0.39 | 0.29 | 0.20 | **0.19** (n=138) |
| CYP2C9 | **1.21** (n=259) | 0.77 | 0.43 | 0.29 | 0.22 | **0.23** (n=34) |
| CYP2D6 | **2.59** (n=129) | 0.30 | 0.29 | 0.22 | 0.21 | **0.22** (n=110) |
| CYP3A4 | **2.15** (n=944) | 0.35 | 0.22 | 0.14 | 0.13 | **0.14** (n=43) |

The band is **symmetric** around the point estimate: median `(y − lo)/(hi − lo)` = 0.508 / 0.493 / 0.491 /
0.490. It is never degenerate: 0% of rows have width < 0.05.

So the metric grants **± 0.6 to ± 1.3 pIC50 of free error on inactives** and **± 0.07 to ± 0.11 on potent
compounds**. Inactives are nearly free *provided you predict them low*; the potent tail has effectively no
tolerance. Scigantic independently reports the same thing from the other side: "At CYP1A2 the margin is
about 0.17 either side, and 23% of our predictions land inside it and cost nothing."

Label floor: min pIC50 ≈ 1.91–2.10; fraction below 4 is **1A2 16.4% · 2C9 20.2% · 2D6 8.6% · 3A4 40.4%**.
Below 4.5: 23.7 / 42.6 / 32.1 / 58.4%.

### 4.2 Where the score actually lives

Share of the **denominator** (baseline soft error of the constant `mean(y_true)`) by potency bin:

| Isoform | y ≤ 4 | 4–4.5 | 4.5–5 | 5–5.5 | 5.5–6 | y > 6 | rows with zero baseline error |
|---|---|---|---|---|---|---|---|
| CYP1A2 | 33.9% | 3.8% | 0.8% | 9.2% | 23.5% | **28.9%** (n=138) | 22.7% |
| CYP2C9 | 29.5% | 1.1% | 4.1% | 26.3% | 25.1% | 14.0% (n=34) | 36.6% |
| CYP2D6 | 19.9% | 15.3% | 2.1% | 14.9% | 19.2% | **28.6%** (n=110) | 18.7% |
| CYP3A4 | 24.6% | 2.6% | 22.6% | 26.0% | 16.7% | 7.5% (n=43) | 15.3% |

**On CYP2D6, 110 compounds — 7.4% of the labelled rows — carry 28.6% of the score. On CYP1A2, 138 rows
(9.8%) carry 28.9%.** The mid-range bins are nearly free: 1A2's 250 compounds in (4.5, 5] contribute
**0.8%** of the total, because the constant baseline already sits inside their bands. 15–37% of rows score
zero even for the trivial constant predictor.

This is why 2D6 and 1A2 are our two bad cells and 2C9/3A4 are not: both are dominated by a small potent
tail with narrow bands, and a model that regresses to the mean simply never reaches it.

### 4.3 Placement sensitivity, measured

ST-RAE of a **perfect ranker** (predicting `y` exactly) after a uniform additive shift:

| shift | +0.0 | +0.1 | +0.2 | +0.3 | +0.5 | +1.0 |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.000 | 0.005 | 0.100 | 0.244 | 0.567 | 1.497 |
| CYP2C9 | 0.000 | 0.003 | 0.074 | 0.217 | 0.641 | 2.125 |
| CYP2D6 | 0.000 | 0.007 | 0.141 | 0.350 | 0.788 | 1.900 |
| CYP3A4 | 0.000 | 0.013 | 0.082 | 0.192 | 0.456 | 1.203 |

ST-RAE of a perfect ranker after **shrinkage** `mu + k·(y − mu)`:

| k | 0.6 | 0.8 | 0.9 | **1.0** | 1.1 | 1.2 |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.188 | 0.060 | 0.012 | **0.000** | 0.012 | 0.060 |
| CYP2C9 | 0.186 | 0.053 | 0.009 | **0.000** | 0.008 | 0.056 |
| CYP2D6 | 0.213 | 0.065 | 0.013 | **0.000** | 0.012 | 0.064 |
| CYP3A4 | 0.252 | 0.093 | 0.026 | **0.000** | 0.025 | 0.086 |

Three numbers to keep:

- **A global offset below ~±0.1 is free** (cost 0.003–0.013). Beyond that the cost is roughly quadratic
  then linear: ±0.3 costs 0.19–0.35 macro, ±0.5 costs 0.46–0.79.
- **Shrinking a perfect ranker by 20% costs 0.053–0.093 ST-RAE**; by 40%, 0.19–0.25. Expansion costs the
  same. The optimum is exactly k = 1.0, and the penalty is symmetric — so **variance matching, not
  variance shrinking, is optimal**, which is the opposite of what an L2 objective delivers under shift.
- **The best constant barely beats the mean**: 1A2 5.15 → 0.9541, 2C9 4.72 → 0.9692, 2D6 4.75 → 0.9988,
  3A4 4.31 → 0.9688. So ST-RAE ≈ 1.0 really is the constant-predictor line, and 64% of the field being
  above 1.0 on 2D6 means 64% are genuinely worse than a constant there.

---

## 5. Data provenance — the organisers' own words, and the 2D6 explanation

All quoted from the Space's own About/FAQ text (`config.py`, same repo as the app):

> "**Primary Screen:** Single-concentration screening across Enamine **DDS10** (diversity set) and **FDAA**
> (FDA-approved compounds) libraries."

> "**Analog Expansion Test Set:** Hit expansion was performed on the **top 25 hits from CYP1A2, CYP2C9,
> and CYP3A4** by purchasing the **top 10 chemisimilars (ECFP4 Tanimoto similarity) per hit compound**
> from the Enamine US catalog."

> "**Dense Test Matrix:** The resulting **750-compound test set** was fully assayed in 12-point DRC mode
> against **all four CYPs (including CYP2D6)**, yielding a dense test matrix with detailed SAR and
> **activity cliffs**."

**25 parents per CYP × 3 CYPs = 75 parents; × 10 analogs = 750.** The announcement is more precise and
adds the reason 2D6 was left out:

> "After evaluating the DRC dataset, we carried out hit expansion on the most potent compounds to create
> our test set, selecting the **top 25 hits per CYP for 3 CYPs (75 compounds total)** and purchasing the
> top 10 chemisimilars (ranked by Tanimoto similarity) per hit compound from the **Enamine US in-stock
> catalog**. This hit expansion was conducted over CYP1A2, CYP2C9, and CYP3A4 hits only, **as the CYP2D6
> assay was still in development on the Echo-MS system.** … The resulting DRC training set is a **sparse**
> matrix, while the test set is **dense** across the CYPs."

Parents are **DRC-confirmed and potency-ranked**, not raw single-point hits ("Rank confirmed hits by
potency", announcement Fig. 2). In-stock catalog, not REAL/make-on-demand. No similarity cutoff is
published for CYP; PXR, by contrast, used the "on-demand catalog" with an explicit **ECFP4 Tanimoto > 0.4**
and 63 parents → 513 compounds after QC.

### 5.1 The organisers themselves say CYP2D6 is distribution-shifted

The halfway post (25 Sep) states it outright:

> "the test set compounds were selected from a hit expansion on the most potent compounds on the training
> set … **Because CYP2D6 hits were not explicitly prioritized in this expansion, the test set for CYP2D6
> contains inherently less potent compounds than the other three isoforms. This distribution shift between
> training and test sets makes CYP2D6 a much harder target.**"

So the 2D6 anomaly is not a modelling mystery and not a label defect alone — it is an **acknowledged
covariate and label shift, by construction**. Combined with §1.5 (blind sd_y 1.565 vs train 0.916) and §1.4
(the top three beat the best published methods by 0.11–0.14 on exactly this cell), 2D6 is the one place
where knowing the construction is worth more than a better model.

### 5.2 Measured structure of the split

Measured over the released files (ECFP4 = Morgan r2, 2048 bits):

- **The 75 × 10 design is visible in the data.** 39 train compounds are the sole nearest neighbour of ≥10
  test compounds (37 with exactly 10); a capacity-10 optimal assignment saturates 39 parents at 10 and
  covers 649/750 in the 75 largest groups. A null of 750 random train compounds produces **zero** groups
  of ≥8.
- **Parents are potent**: median max pIC50 over (1A2, 2C9, 3A4) = **6.34** vs 4.67 for train overall. 2D6
  parents show little enrichment — exactly as the official text says.
- **Test-to-nearest-train max Tanimoto**: mean 0.598, median 0.587, p5 0.500, p25 0.543, p75 0.639,
  p95 0.741, **max 0.952, min 0.318**; 10.1% ≥ 0.7. Train-to-nearest-other-train median is 0.453, so the
  test compounds are *closer* to train than train is to itself. Test-to-nearest-other-test median 0.656.
- **It is not a scaffold holdout.** 24.4% of test compounds share an **exact Bemis–Murcko scaffold** with
  train (62.7% generic scaffold); 88.5% of the 546 unique test scaffolds are unseen. So scaffold-split CV
  is *harder* than the real test set, and Butina/series CV at ~0.59 similarity is the right analogue.
- **The test set is a separate, later purchase**: 735/750 test ids sit in one contiguous block
  `OCNT-2534775..2535825`, while every train id is ≤ 2,395,799. 15 test compounds carry older ids.
- **There is an undocumented analog-expansion batch already inside train.** 527 train compounds at
  `OCNT-2395272..2395799` are **all CYP3A4-only DRCs with no single-concentration rows**, internally much
  tighter than the library (NN-within-block median 0.633 vs 0.442), their nearest screened neighbours are
  87% 3A4 hits, and **24 parents each have exactly 10 of them**. 101 of the 750 test compounds have this
  block as their nearest train neighbour. This is the closest thing available to a *free dress rehearsal*
  of the real test construction, and it is sitting unused in the training file.
- Live/full: `head_to_head.py` references a non-public `ground_truth/{track}-identifiers.parquet` with a
  `phase` column (1 = "Used for live leaderboard", 2 = "Remaining test set"). All compounds from one parent
  go to the same half, so live ≈ 37–38 series. Series ids are not released.

### 5.3 Three data facts that change how the auxiliary columns should be used

1. **The single-concentration screen file is itself selected.** 17,504 rows = 4,376 compounds × 4 enzymes,
   and **4,375 of those 4,376 compounds are also in the DRC train set** — so it is the primary-screen
   readout *of DRC-promoted compounds only*, not of the whole ~11,000-compound library, despite the Space
   and blog text implying otherwise. Every row is `concentration_M = 4.95e-5` (~49.5 µM) and the screen was
   run **in the TDI/preincubation arm only** ("to catch as many positives as possible"). Columns are
   `log2fc_estimate, log2fc_std_error, p_value, log2fc_fdr, log2fc_median, cohens_d` — **no `Emax`,
   `max_response` or `_std`**. Hit rule in the tutorial: `log2fc_estimate < −1 and log2fc_fdr < 0.05`.
   This is a second, independent selection layer on top of the DRC-promotion one the programme already
   found, and it means the screen cannot be treated as an unbiased low-fidelity sample of chemical space.
2. **`Emax` is a separate config** (`TRAIN_Emax.csv`, 6,145 rows × 30 cols:
   `{CYP}_EmaxVsPosCtrl_{TDI_condition,direct_inhibition}` each with `_conf_high`/`_conf_low`, no `_std`,
   plus `{CYP}_is_TDI`).
3. **1,240 TDI-file compounds have a TDI-arm pIC50 but no direct pIC50, and all carry `is_TDI = False` as a
   default rather than a measurement** (1,238 of them CYP3A4). Pat Walters' own post confirms it: "All
   1,249 … carry a TDI label of False." Training a TDI classifier on those rows as negatives learns the
   default, not the biology.

### 5.4 The interval columns, verified

`_conf_low`/`_conf_high` are **95% credible bounds** (2.5/97.5 percentiles of 4,000 brms draws) and `_std`
is the posterior SD: `(high − low) / (3.92 · std)` has median **1.00** across all 6,525 released values, and
the point estimate always lies inside the interval. The prior has a **hard floor at pEC50 = 1**, so
`conf_low` bottoms out at 1.03 — which is why the inactive band is so wide and roughly spans **[1.05, 3.7]**.
The organisers' own description matches what §4.1 measures: "Empirically, these intervals **expand
significantly at lower activity levels**, aligning with assay precision limits."

Top assay concentration is ~50 µM (the TDI rule uses pIC50 > 4.3 ≈ "IC50 < ~50 µM"; the Feb 2026 pilot
protocol says "~0.1 nM to ~50 µM, ~3-fold serial dilutions"). **So the real measurement floor is pIC50
≈ 4.3, and the oft-quoted "pIC50 < 4" is already an extrapolation below the lowest tested dose.**

### 5.5 Operational warning: the ground truth was revised after launch

The dataset changelog shows an initial 2026-08-03 release (4,901 train rows) and a 2026-08-05/06
"Finalised ground truth: refreshed direct-inhibition and TDI training/test data". For the same compounds,
pIC50 moved by mean |Δ| of **0.33 (1A2), 0.61 (2C9), 0.07 (2D6), 0.12 (3A4)**, with max |Δ| up to **4.0**.
A cisplatin row was removed 2026-08-27. HEAD is `146282d2`, 2026-09-23. Anything derived from the 3 Aug
files — cached features, fold assignments, screen joins — is wrong on 2C9 and 1A2 in particular and must be
re-pulled. Our local snapshot is `146282d2`, so we are current.

### 5.6 Inconsistencies in the organisers' own text

Worth knowing so as not to chase them: the announcement's Fig. 2 says "8-point" DRC while every other
source says 12-point (the figure appears reused from PXR); the announcement names EOMCC as the probe for
both 1A2 and 2C9 while the data post says BOMF for 2C9; the announcement describes the 2D6 readout as
parent depletion while the data post says product formation; and the announcement describes a
**downweight-below-pIC50-4** metric that exists in `custom_scoring_functions.py`
(`rae_weight_below_threshold`, threshold 4, weighting 0.25) but **is not the function wired into
`config.py`** — the credible-interval soft threshold is. Do not optimise the documented metric; optimise
the wired one.

**The assays — and 2D6 is a different instrument:**

> "**Fluorescence Panel (CYP3A4, CYP2C9, CYP1A2):** Adapted from ThermoFisher Vivid fluorogenic kits.
> Recombinant enzymes metabolize masked probes to produce fluorescent readouts."

> "**Echo-MS Mass Spectrometry Panel (CYP2D6):** Because fluorogenic 2D6 probes exhibited poor
> signal-to-noise ratios, we developed a label-free mass spectrometry assay."

Run at **Octant** on the **Navigator** platform in **1536-well** format, 12-point DRC. So CYP2D6 differs
from the other three in *selection*, *assay technology*, *band width* (narrowest, 0.272) and *inactive
fraction* (smallest, 8.6%). Four independent reasons it behaves unlike the rest; none of them is "the
model needs to be bigger".

TDI: "Direct and time-dependent inhibition were measured **simultaneously** using an IC50-shift design",
every compound pre-incubated ± cofactor; positive = IC50 shift > 2-fold.

Structure track: 20 novel **cryo-EM** CYP3A4 complexes from UCSF — 15 from train, **2 from the test set**,
3 additional.

Rules worth noting: external data and pretrained models **fully permitted** (disclose proprietary data via
a checkbox, shown publicly — every entry says No); one HF account per team; submissions rate-limited to
one per 12 h, latest valid counts; deadline **3 Nov 2026 23:59 UTC**; final board from **4 Nov**. There is
a separate discretionary **"Most Innovative Machine Learning Approach"** award judged on "architectural
novelty, creative data usage, novel uncertainty quantification, and clever engineering, **independent of
your final leaderboard** [position]".

---

## 6. The organisers' own recipe, exactly

Source: `OpenADMET/pxr-challenge-tabicl` → `blogpost.md` (= the Ghost post "Are tabular foundation models
all you need?", Sean Colby, 23 Sep 2026, DOI `10.5281/zenodo.23198306`).
https://github.com/OpenADMET/pxr-challenge-tabicl ·
https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/

**Correcting the brief's premise in three places:**

- The canonical CheMeleon PCA width is **256**, not 128. "PCA reductions to 256, 384, and 512 were
  statistically indistinguishable, while 128 and below were clearly separated and performed worse, so we
  chose **256** dimensions." They could not test above 512 (memory).
- **128 is the canonical width for the RDKit *descriptor* block**, a different thing: "Native-dimension
  RDKit led the comparison at 0.53 MAE, but was statistically indistinguishable from PCA-reducing it to
  **128**." Mordred and RDKit+Mordred trailed everywhere.
- The shipped config is **TabICL**, chosen over TabPFN v3 on licence grounds, not accuracy: seed means
  **0.427 (TabICL) vs 0.436 (TabPFN v3)**, statistically tied; TabICL is BSD-3 while TabPFN carries the
  Prior Labs Licence. TabFM, LightGBM, XGBoost and TabPFN v2.6 all separated and worse.

The chain, measured on PXR (260-compound Phase-2 set, MAE, 5 seeds, Tukey HSD):

| Step | Result |
|---|---|
| CheMeleon baselines in `openadmet-models` | ~0.50 MAE; leaders were ~0.41 |
| Buterez-style GNN transfer configs (E1/E3/E4) | **none separated**; HSD critical distance 0.10 MAE, closest p = 0.62. "None of these experiments reached the accuracy N283T reported." |
| What the Buterez paper actually contributed | "a single, albeit important, component: an **auxiliary encoder trained on primary-screen data to predict a log2FC readout**" — not the architecture |
| Feed embedding + log2FC readout to a TFM | **0.50 → 0.43** |
| Best feature combo (36 tested) | log2FC-finetuned CheMeleon embedding + Chemprop log2FC readout + RDKit-128; **tied with 15 of 36 rows**. Three ties omit descriptors entirely → "the descriptor block adds no detectable value beyond an embedding plus log2FC readout" |
| What they shipped | **off-the-shelf CheMeleon embedding + Chemprop log2FC readout**, "which ties the leader while keeping the implementation much simpler, **though the case for retaining the readout was weak**" (the log2FC-finetuned embedding alone was also tied) |
| TFM uncertainty | intervals well calibrated (miscalibration area 0.04) but **Spearman ρ = 0.26** between predicted sd and absolute residual — "not a reliable ranking of which compounds are uncertain" |
| Compute | TabPFN v3 GPU 14.6 s/run; TabICL fell off a 24 GB GPU to CPU at 184 s; TabICL@128-d on GPU 13 s. 4,392 in-context rows fit TabPFN/TabICL; TabFM capped at `max_num_rows=500` |

**On the CYP board specifically**, verbatim:

> "Off-the-shelf CheMeleon embedding + log2FC readout is the **TabICL-baseline**, which held the top spot
> on the regression live leaderboard … for... a few days (**0.68 MA-ST-RAE**). As of this writing, the top
> performers have reduced that error by ~36% (**0.43 MA-ST-RAE**). Still, the new method improves upon
> previous `openadmet-models` baselines by ~20% in this challenge (see the leaderboard's
> **CheMeleon-baseline at 0.83 MA-ST-RAE**)."

So the brief's "~0.68 recipe" is real, is the organisers' own, and sits on the live board right now as
`TabICL-baseline` at **0.6755, rank 95** (§1.3). It is a *baseline*, not a contender — but it is 78 ranks
above us. `CheMeleon-baseline` is 0.8335 at rank 153, also above our 0.8663. Confirmed: the headline gain the organisers found is a **predicted** log2FC readout as feature
columns, which is legitimate at inference (the blinded file carries only `Molecule_Name` and `SMILES`).

Implementation hooks they added to `openadmet-models`: `CheMeleonFeaturizer` (2,048-d off-the-shelf), a PCA
transformer applicable per featurizer group (`None` = passthrough), `PretrainedModelFeaturizer` ("train a
Chemprop model on primary screen log2FC data, whose output then gets read into the main tabular foundation
model as predicted log2FC features"), and `TabICLRegressor`/`TabICLClassifier`.

The PXR repo also ships the raw evidence: `experiments/manifest.yaml` (44 KB, "the single source of what
runs"), `results/gates/{best_featureset,best_regressor,canonical_descriptors,embedding_reduction}.json`,
and per-seed `predictions.csv` for `chemeleon_frozen` and `chemeleon_log2fc_finetune`.

Tutorial repo (the CYP one) contents worth knowing:
`notebooks/activity_prediction.ipynb`, `TDI_prediction.ipynb`, `structure_prediction.ipynb`;
`inputs/boltz-2_cyp3a4_predictions/` (20 Boltz-2 CIFs); `evaluation/posebusters_utils.py`;
`validation/activity_validation.py`.

---

## 7. The two new reports that matter, and the pattern they complete

### 7.1 Scigantic — rank 19 live / **rank 11 interim**, the best published method

https://github.com/Scigantic/open-admet-challenge-cyp (Aaron Kanzer, a software engineer specialising in
information retrieval — "I am not a cheminformatician"). Not linked from the leaderboard; found only by
search. The write-up is half platform marketing, but the method half is the most informative in the field.

**The table they trained on — ~32K compounds, 180K+ labels, 26 endpoints, 6 sources:**

| Source | endpoints | labels | compounds |
|---|---|---|---|
| Organisers, scored direct-inhibition pIC50 | 4 | 6,525 | 4,905 |
| Organisers, single-concentration log2FC | 4 | 17,500 | 4,375 |
| Organisers, pIC50 under the second assay condition (TDI arm) | 4 | 6,538 | 4,904 |
| Retrieved, ChEMBL pIC50 | **5** | 56,705 | 24,921 |
| Retrieved, Veith qHTS AC50 | 4 | 15,070 | 7,362 |
| **Retrieved, qHTS maximum response** | 5 | **80,665** | 16,133 |

> "**The scored labels are 3.6% of the table.** The rest is never scored and never predicted at test time.
> Training against it is what stops a 4,900-compound fine-tune from memorizing."

Three deliberate choices, in their words:

- **qHTS maximum response is the single largest block** and the one they call out: "it is not a potency
  value but the raw screening response, which carries **whether a compound does anything at all**, and
  that turns out to be worth more rows than any curated endpoint."
- **CYP2C19 is included although it is never scored**: "a correlated fifth task improves the four that
  are scored."
- **Leakage was caught by structure, not by name**: "One held-out compound did turn up in an early table,
  under a screening accession that looked nothing like its challenge identifier, carrying auxiliary
  screening values rather than a scored measurement. **Name-based checks had all passed.**" They now key
  every external join on canonical SMILES + InChIKey skeleton and keep an exclusion file.

**The stack**: 364 SageMaker jobs, 12 members from 2 backbones. Eight **CheMeleon** multi-task regressors
over all 26 endpoints at once, differing in which auxiliary heads they carry, the embedding width, and
**whether the credible interval is fed in as an input**. Four **Chemprop** D-MPNNs, "including one trained
to predict the interval rather than the point value". Mordred + GBT alongside as a non-neural check.
MolFormer, ChemBERTa and Uni-Mol were fine-tuned and measured but did not earn a place.

**The ensembling, which is the part we should copy:**

> "The metric forgives near misses. … At CYP1A2 the margin is about 0.17 either side, and 23% of our
> predictions land inside it and cost nothing. Squared error penalizes for those misses although the
> metric does not. So the twelve member weights are **fitted to the competition metric itself, on held-out
> folds, rather than to squared error**. After fitting, **the column is rescaled to its pre-fit mean and
> standard deviation**, which stops the search from shifting or squashing the whole column to lower
> training loss without improving any single prediction."

That last clause is a non-obvious and important guard: fitting weights on a hinge loss will otherwise
exploit the hinge by collapsing the spread, which helps training loss and hurts the blind set. They pin
the first two moments and let the weights change only the *shape*.

### 7.2 Woong (rank 146, 0.8141) and AIv_2709 (rank 156, 0.8373) — the control group

Both are technically careful, both use interval-aware losses, **neither places onto the blind population**,
and both land at ~0.82–0.84 from a cross-validated ~0.65.

**Woong** (https://gist.github.com/ByungwoongYoo/4904441a9b30db077fbb784f9c22e423): six branches per
endpoint — interval-aware LightGBM on Morgan bits + descriptors; interval-aware LightGBM on count
fingerprints; a **mixed-Tanimoto interval SVR** (C = 1); a masked multi-assay MLP; an "SC-prior" kernel
ridge (α = 0.3) on *cross-fitted predicted* single-concentration values mapped through a spline to
direct-inhibition intervals; and a `dose8` model (768 Nyström landmarks, rank-8 shared component, monotone
link). Blended by non-negative weights summing to 1 plus an intercept, fitted on a **smoothed
interval-distance loss (smoothing 0.035) with L2**. Dev cohort 5,378 molecules, three repeated 4-fold
cluster splits plus a 3-fold scaffold stress test; production refit on 6,146. **Reported dev MA-ST-RAE
0.6490** → board **0.8141**.

**AIv_2709** (https://github.com/avnikonenko/openadmet_challenge_cyp_2026): three families × 5 folds × 3
seeds = 45 prediction sets. LightGBM (2048-bit Morgan r3 + 8 RDKit descriptors); a **transfer D-MPNN**
first trained on single-concentration inhibition then re-headed and fully fine-tuned to pIC50 (width 300,
depth 3, dropout 0.10); a **joint multi-assay D-MPNN** trained on single-concentration and pIC50
simultaneously (width 600, depth 4, 32-d CYP and 16-d assay embeddings). Interval-aware loss for the
neural runs. **Unweighted mean of the three families.** Calibration is only training-fold target
standardisation, inverse-transformed. No external data, no pretrained weights. **CV macro ST-RAE 0.6500
(ECFP6 cluster) / 0.6482 (scaffold)** → board **0.8373**. Their note: "CYP2D6 remained the hardest
endpoint."

**The pattern, across every report now read:** cross-validated ST-RAE clusters around **0.65** for
competent work regardless of architecture. What separates 0.43 from 0.84 is **not** the model — it is
whether a per-isoform affine placement onto the blind population was applied, and whether the blend was
fitted on the metric.

| Entry | Score | Interval-aware loss | Aux screen signal | **Blind placement** |
|---|---|---|---|---|
| briford (13) | 0.4378 | no (MSE) | yes | **yes — affine, k = ρ·sd** |
| Scigantic (19/**11**) | 0.4456 | yes (interval head) | yes, huge | **yes — moments pinned after metric-fitted blend** |
| alexxi11 (44) | 0.5069 | `BoundedMSELoss` | yes | **yes — soft-RAE-optimal shift** |
| JacksonBurns (49) | 0.5181 | no | yes | **yes — `BLIND_MOMENTS` z-placement** |
| jeremy (50) | 0.5185 | no | yes | **yes — blind-moments rescale** (2D6 R² −0.749 → +0.377) |
| Lizard Wizard (55) | 0.5357 | yes (dead zone) | yes (Heckman) | **yes — per-fold affine** |
| CompChem Magicano (83) | 0.6404 | **yes (hinge)** | yes (predicted log2FC) | partial (−0.5 2D6 shift) |
| **Woong (146)** | 0.8141 | **yes (smoothed interval)** | yes (SC-prior) | **no** |
| **AIv_2709 (156)** | 0.8373 | **yes** | yes (transfer + joint) | **no** |
| popcorn2 (128) | 0.7671 | no | yes (briford's heads) | **no — "none implemented"** |
| **talindrew (173)** | **0.8663** | yes | yes | — |

popcorn2 is the cleanest natural experiment in the whole challenge: they reproduced **briford's exact four
Chemprop members** and scored **0.7671 against briford's 0.4378**, the only stated difference being that
briford placed and they did not.

---

## 8. What the top entries do that we do not

Ordered by the evidence behind them, not by appeal.

1. **Fit the ensemble weights on ST-RAE itself, then pin the column's mean and sd back to their pre-fit
   values.** Scigantic's guard. Nobody else reports the second half, and without it a hinge-fitted blend
   degenerates by collapsing spread.
2. **Match variance, do not shrink it.** §4.3 measures the cost: k = 0.8 on a perfect ranker costs
   0.053–0.093; the penalty is symmetric about k = 1.0. Every entry at 0.43–0.52 does a per-isoform
   affine placement; every entry at 0.77–0.87 does not.
3. **Spend the error budget on the potent tail.** §4.2: 110 compounds carry 28.6% of 2D6's score, 138
   carry 28.9% of 1A2's, and 1A2's entire (4.5, 5] bin carries 0.8%. Uniform weighting of compounds is
   mis-spending; the mid-range is already inside the bands.
4. **Use the raw screening *response* as the largest auxiliary block, not the curated potencies.**
   Scigantic's 80,665 qHTS max-response labels, "worth more rows than any curated endpoint", because they
   carry whether a compound does anything at all. We have `max_response` available and under-use it.
5. **Add CYP2C19 as an unscored fifth correlated task.** Cheap, and Scigantic says it improves the four.
6. **Feed the credible interval in as an input, and train a member to predict the interval itself.**
   Scigantic does both (3 of 12 members). Nobody else reports it. It is the natural way to let the model
   know where its tolerance is.
7. **Key every external join on structure, never on name.** Scigantic found a test compound that passed
   all name-based leakage checks and was caught only by canonical SMILES + InChIKey skeleton.
8. **Treat the test set as ~75 ECFP4 analog series around 1A2/2C9/3A4 hits** (§5) — so cluster CV should
   be series-level, and 2D6's blind moments are the least inferable of the four.
9. **A predicted log2FC readout as feature columns** (the organisers' own 0.83 → 0.68 step) — available at
   inference, leaks nothing. Our arm uses the screen as heads but, per the programme's own note, has not
   tried dense-embedding + predicted-readout → TFM.

10. **Whatever the top three do on CYP2D6 specifically.** §1.4: the top three reach 2D6 0.41–0.44 while the
    best published methods manage only 0.54–0.55, and on the other three isoforms the published methods are
    close. This is the only cell where the undisclosed leaders have a large, unexplained edge — and 2D6 is
    the isoform with the narrowest bands (0.272), the fewest inactives (8.6%), a different assay (Echo-MS),
    and no role in test-set selection. The hypothesis worth testing is that they model 2D6 *as the
    different problem it is* rather than as a fourth head.

11. **Place onto briford's published blind moments, and sanity-check them against §1.5.** §9.2 gives both
    mean *and* sd per isoform from a solved board inversion (1A2 4.412/1.553 · 2C9 4.830/1.101 ·
    **2D6 3.107/1.599** · 3A4 4.880/1.272); §1.5 independently reproduces the sds from the leaders' MAE/R²
    to within 10–15% with identical ordering. Set `k*` between `0.8ρ` and `ρ` per isoform (not 1.0 — that is
    only the perfect-ranker case in §4.3), and target a 2D6 centre of **~3.6**, where the board probes in
    §9.2 bottom out. This is the single highest-expected-value change available, and it needs no new model.
12. **Validate on the analog-expansion batch that is already inside train.** §5.2: 527 CYP3A4-only
    compounds at `OCNT-2395272..2395799`, in 24 parent-groups of exactly 10, built the same way the test
    set was. Holding those out series-wise is a far closer rehearsal of the real task than scaffold CV —
    and §5.2 shows scaffold CV is *harder* than the real test set, which is why everyone's CV (~0.65)
    mispredicts their board score in the pessimistic direction on rank but the optimistic direction on
    placement.
13. **Stop treating the screen as unbiased.** §5.3: the single-concentration file covers only
    DRC-promoted compounds, at one concentration, in the TDI arm only. It is a second selection layer, not
    a low-fidelity sample of chemical space.
14. **Probe the board with a plateau fit, not a V-apex fit** (§10.1–10.2). In NeurIPS Open Polymer 2025 —
    the closest structural analogue there is, an L1 metric on a hidden shifted set — the winner found his
    offset by submitting a few scaled offsets and fitting a V to the leaderboard points. Our metric has a
    deadband, so the bottom is a *plateau*: aim for its centre. We have 12 h between submissions and ~26
    days, i.e. roughly 50 probes available, and §9.2 already gives four 2D6 points for free. The
    counterweight is mandatory: MoA went 11th → 118th and Novozymes 1st → 967th on probe-driven
    post-processing, and Roelofs et al. show the failures cluster exactly where public and private are drawn
    differently — which ours are (different series). So probe **only** a per-isoform location and scale, and
    hold to mokhalaf22's rule: act only on board moves above ~0.03.
15. **Fit the offset on the hinge, not on the mean residual** (§10.3). With one-sided/skewed residuals the
    L1-optimal intercept differs from the OLS one by ~0.25 at identical slope. The programme's own memory
    already records this; Gneiting 2011 is the formal statement.
16. **Use CLAD/Tobit for the inactive tail, where the literature says the gain is largest.** Svensson et al.
    report the biggest censored-regression gains above 35% censoring; CYP3A4 is 40.4% below pIC50 4 (§10.6).
    Censored LAD is the natural pairing with a hinge metric because medians commute with monotone maps.
17. **Prefer a MapLight-style feature union at our sample size.** On TDC, at n ≈ 1–2k per task — our regime —
    CatBoost over ECFP + Avalon + ErG + 200 RDKit descriptors wins and pretrained GNNs lose; the GNN
    embedding only pays above n ≈ 12k (§10.5). MapLight is also one of only three TDC families that survive
    a reproduction audit.
18. **Median-aggregate the ensemble, not mean-aggregate.** For L1-family metrics the median blend is the
    documented winner (CHAMPS: −3.08 → −3.22 from medianing alone; §10.2).

What is *not* a differentiator, on this evidence: architecture (the organisers' own HSD test could not
separate ten GNN configs; MolFormer/ChemBERTa/Uni-Mol all failed to make Scigantic's stack), descriptor
blocks (no detectable value beyond embedding + readout), proprietary data (nobody has any), and bigger
pretraining.

---

## 9. Community discussion, and the numbers participants have published

No public forum exists: the Space's Community tab has **0 open and 0 closed discussions** (probed #1–60,
all empty), and the dataset has only a `parquet-converter` bot thread. All Q&A happens on a **Discord**
("OpenADMET Challenges", guild `1412827471488745545`, ~850 members), channels `#cyp-challenge` and
`#cyp-challenge-submissions` (bot receipts via `backend/discord_bot.py`). **The invite in the Space FAQ
(`discord.gg/HJGYrHnsa`) is expired**; working invites: `discord.gg/FeSZCTf5sa` (welcome),
`MY5cEFHH3D`, `4mERqNsQh7`, `ndgtXfYhJe`. Nothing found on Reddit or X (OpenADMET has no X account; its
site links LinkedIn only). Bluesky dormant since 2025-08-20. The one newsletter mention is
*This Week in Cheminformatics* #35, https://thisweekincheminformatics.substack.com/p/thirty-five.
Pat Walters' LinkedIn launch post (159 reactions, 3 comments) is the only social footprint of note.

### 9.1 ⚠ A model report is **mandatory** for the final leaderboard

From the halfway post, verbatim: *"we require a comprehensive model report along with your final
submission. **If this report is not included by the time the submission closes, you will not be considered
for the final leaderboard.**"* Also: *"Our system automatically overwrites multiple entries with the same
username, keeping only the latest one"*, and alt-account use is grounds for removal. Deadline
**2026-11-03 23:59 UTC**. This is an operational requirement, not advice — we currently have no report
linked on our row.

### 9.2 briford published the solved blind-half moments — the most valuable number in public

`SuperCowPowers/workbench` → `ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py` carries the
**live-half population mean and sd, solved from the board's own published metrics**:

| Isoform | briford's solved blind mean / sd | train mean / sd (§4.1) | **mean shift** | my sd estimate (§1.5) |
|---|---|---|---|---|
| CYP1A2 | 4.412 / 1.553 | 4.955 / 1.030 | **−0.54** | 1.407 |
| CYP2C9 | 4.830 / 1.101 | 4.581 / 0.782 | +0.25 | 0.942 |
| **CYP2D6** | **3.107 / 1.599** | 4.784 / 0.916 | **−1.68** | 1.565 |
| CYP3A4 | 4.880 / 1.272 | 4.096 / 1.093 | +0.78 | 1.129 |

Two independent confirmations of each other: my §1.5 sd estimates, derived from the *leaders'* MAE and R²,
come in 10–15% below briford's — exactly the under-estimate my stated Gaussian caveat predicts, and the
ordering is identical. **The blind sd is larger than train on all four isoforms, and the blind CYP2D6 mean
is ~1.7 log units BELOW train.** That single shift, applied to an unplaced model, costs ~1.9 ST-RAE on 2D6
by §4.3's shift table — which is precisely the 1.38 we score there. briford separately notes the 2D6 blind
centre is "about 3.7 by qHTS inactivity, where trained models predict 4.69".

**briford's algebra**, worth stating exactly: with `k` = (prediction sd / truth sd) and `b` = mean offset,
`R² = 2ρk − k² − b²`. So R² is capped at `ρ²`, and the **R²-optimal `k` is ρ, not 1**. This reconciles with
§4.3: my shrinkage table is the `ρ = 1` case, where `k* = 1`; for a real model `k*` lies between `0.8ρ`
and `ρ`. mokhalaf22 measured the ST-RAE-optimal spread at **≈ 0.8ρ**, "but flat at its minimum (0.0002
ST-RAE difference)", and warns `k*/ρ` is **not transportable between populations**. briford's own warning:
"R²-optimal and ST-RAE-optimal placements point opposite ways."

**Board probes of the 2D6 centre** (Workbench script comments) — submissions whose 2D6 mean was set to a
constant and scored:

| 2D6 mean | 3.11 | 3.32 | 3.57 | 4.35 |
|---|---|---|---|---|
| ST-RAE | 0.6944 | 0.6271 | **0.5502** | 0.8558 |

So the ST-RAE-optimal 2D6 centre sits **between 3.57 and 4.35, nearer 3.57**, and the penalty above it is
much steeper than below. Codie Freeman's complementary measurement: **under-predicting costs 1.65–2.79×
more than over-predicting** under this metric.

### 9.3 Placement gains participants report, all board-measured

| Entrant | before → after | note |
|---|---|---|
| jeremy | 0.7247 → **0.5214** | 2D6 R² −0.749 → +0.377 |
| mokhalaf22 | 0.9356 → **0.6411** | rank 200 → 77; −0.5 2D6 shift + two spread expansions |
| codie-freeman | 0.7138 → 0.6364 → **0.5982** | recentre 2D6, then refine |
| joewa6 | 0.790 → **0.557** | |

Four independent entrants, all gaining **0.15–0.29 macro from placement alone**. This is the largest
single effect anyone reports, and it is larger than every modelling lever combined.

### 9.4 What is reported to work, and what is reported null

**Works:** interval-hinge loss zero inside the CI (mokh: OOF 0.445 → 0.434, closed the 3A4 gap); a
*predicted* primary-screen log2FC feature (mokh: D-MPNN OOF 0.434 → 0.415 — the same lever the organisers
found); TabICL over LightGBM (avaliev: −0.039; TabICL × multitask blend −0.031); separate heads per source
from a **population-matched** source like AID 1851 rather than ChEMBL (briford, jeremy); non-negative
stacking (lachrymator).

**Null or negative, reported by more than one entrant:** ChEMBL/PubChem augmentation (avaliev +0.057
worse; jeremy found ChEMBL is 1–1.65 log units more potent than the diversity-library screen);
frozen CheMeleon (avaliev +0.010 worse — though Jackson Burns found full fine-tune beat frozen on his
folds, contradicting jeremy); pseudo-labelling missing pIC50 from the screen ("hurt badly" — jeremy;
mokh's real-label proxies cost +0.0442); pairwise-ranking / SQRL / DeepDelta losses; 3D shape descriptors;
near-neighbour retrieval corpora (mokh: 111,361 compounds, null; only 13.3% of blind compounds have a
public neighbour at Tanimoto ≥ 0.7); HPO (Workbench: stock Chemprop defaults beat HPO on PXR's analog set);
loss reweighting in either direction (mokh: "ST-RAE's per-row gradient is uniform, so reweighting loses" —
exactly §3.2 point 1).

**On CV→board transfer, the field disagrees and that disagreement is itself the finding:**
lachrymator reports **100% of regression submission-endpoint pairs were worse on blind than in CV**
(median MAE +0.208) and that random splits "flattered every model"; avaliev puts local scaffold CV at
**+0.113 to +0.127 optimistic**; codie-freeman found an analog-holdout split meant to mimic the blind
construction *widened* the CV→blind gap by **2.3–9.3×**, directly contradicting lachrymator's claim that
Butina folds track the board. Jackson Burns fits the field as **ST-RAE ≈ 0.695 − 0.210 × R²**.

### 9.5 Two hard facts about CYP2D6 from participants

- **Rafal Wiewiora**: a *label-free* classifier separating released 2D6 training structures from blind ones
  reaches **ROC AUC 0.972** (https://github.com/rafwiewiora/openadmet-retrospective-reproducibility,
  `CYP_CAMPAIGN.md`). The blind 2D6 compounds are trivially distinguishable from train — the covariate
  shift is not subtle.
- **Workbench**: the `<4.0` and `4.0–4.5` label bands are mostly noise (reliability ceilings ρ 0.25 and
  ≈ 0). The AUC separating `<4.0` from `≥4.5` is **0.750 for 2D6** against 0.939 / 0.912 / 0.838 for
  3A4 / 2C9 / 1A2. Repeat seeds disagree more than candidate models do, and the smallest resolvable
  out-of-fold Spearman difference is 0.018–0.056 per isoform — i.e. most model comparisons in this
  challenge are under-powered.
- **joewa6**: the blind compounds have **no single-concentration screen data at all**; supplying *measured*
  screening in matched comparisons improves 0.446 → 0.196. So the screen is worth an enormous amount and is
  unavailable at test time — which is exactly why a *predicted* screen readout is the only usable form.

### 9.6 Organiser conduct, and a possible data overlap

Space commit history doubles as the Q&A record: 2026-09-04 "Add guard for near-constant predictions";
09-10→09-15 "Block suspicious users"; **09-24 "Ban 29 accounts linked to multi-account submission
clusters"; 09-25 "Ban 22 accounts in six community-reported multi-account clusters"** plus two
reinstatements; two more on 10-05. HF usernames are **not authenticated** — a commit reads "Validate HF
username exists instead of requiring OAuth login". There is no public complaint about ST-RAE itself.

mokhalaf22 reports that **5 of the 750 blind ids appear in
`openadmet/Octant_CYP_inhibition_reactivity_blog_release`** (plus one near-duplicate) and says he told the
organisers. An independent id-only check found the 5 overlaps but concluded **at most 1 carries a CYP3A4
pIC50** (in `inhibition.tsv`; the others appear only in `reactivity.tsv` / `will_it_fly_in_mass_spec.tsv`),
so mokh's stronger claim does not hold. Immaterial at 1/750, but worth knowing it exists.

### 9.7 Other agent-run entries, and one identity to verify

Several entrants are visibly running agent pipelines: `JacksonBurns/cyp-challenge` has
`docs/NEXT_AGENT_PROMPT.md` and `HANDOFF.md` ("DO NOT SUBMIT without Jackson's explicit go-ahead", notes
written by a "Hermes agent"); `codie-freeman` has a 295 KB `CLAUDE.md`; `zchboswell/cypshift` has an
`AGENTS.md`; `buaiml/openadmet-cyp`'s `RULES.md` says only "Denali" submits; Workbench ships an agent
guide at `src/workbench/agent/guides/challenges/cyp_challenge.md`. One live 2D6 top-5 entry is openly
labelled **"Autopilot (fully autonomous agent entry)"** (rank 32 overall on 3 Oct, 0.4767). **None of these
files was opened or acted on** — they are recorded as observations only.

**One item needs the owner's eye, not mine.** The community sweep asserted that the rank-55 entry
*Lizard Wizard Gizzard* (`github.com/doctawho42/CYPChallenge`, 0.5357) is this programme's own work,
on the grounds that its README themes match this repo's commits. **That is unverified and sits awkwardly
with the fact that our board alias is `talindrew` at 0.8663**, and with `24-leaderboard-model-reports.md`
having read it as a third party's report. Either it is a different entrant whose methodology has converged
remarkably closely on ours, or there is a second alias — and the latter would collide with the one-account
rule in §5. Worth five minutes to settle before the final submission, not something to assume either way.
Separately, Jackson Burns' notes record trying that entry's dead-zone loss and finding "Lizard Wizard's
+0.02 dead-zone gain does NOT transfer to our pool".

### 9.8 The TDI track, briefly

Interim Tier 1 holds **26 statistically indistinguishable entries** (down to MCC 0.3406) versus exactly one
on the regression board — so TDI is far less separable and far cheaper to place well in. Live leaders
(3 Oct): TeamPozeSCAF 0.5346 (precision 0.851, recall 0.454), nova 0.5132, NIPERKOLKATA_PI_LAB 0.4752;
median 0.285; the official LGBM baseline is 0.2788 at rank 81. Jackson Burns' board analysis: **MCC
correlates 0.94 with precision at fixed recall**, and every entry above 0.4 has precision ≥ 0.43 — i.e.
the metric rewards a conservative, high-precision threshold. Interim per-isoform leaders: 2D6 MCC 0.345,
3A4 MCC 0.521. mokh found the derived-from-potency TDI label fails (2D6 MCC ≈ 0.03 vs 0.125 for a direct
classifier); Jackson Burns found oracle potency arms give 0.98 MCC but predicted-shift correlation is only
0.16 (2D6) / 0.35 (3A4). TeamPozeSCAF's published method: 2D6 = RF/ExtraTrees/LightGBM over **13
mechanism-based structural-alert features**; 3A4 = LightGBM 0.40 + TabICL 0.60 on PCA-200 Chemprop
embeddings; thresholds by OOF MCC on Butina folds.

### 9.9 Organiser blog posts with usable chemistry

- **"Do Our CYP Structural Alerts Actually Work?"** (08-11): over 7,779 compound-isoform records, the
  heme-ligating aromatic-N alert has enrichment ≈ **1.03** and **12 of 30 alert/isoform combinations are
  anti-predictive**. The best simple predictor is **cLogP > 3** (enrichment 1.98 for 3A4, 1.60 for 2C9);
  the 2D6 basic-amine alert reaches 1.43; TDI alerts such as methylenedioxyphenyl reach 2.16–2.63 at only
  48% recall. Conclusion: most alerts proxy lipophilicity.
  https://openadmet.ghost.io/do-our-cyp-structural-alerts-actually-work/
- **"Escape from Flatland, CYP Edition"** (10-06): Fsp3 vs inhibition on this challenge's training data —
  **2D6 shows no clear correlation** (it favours protonated amines), 3A4 only weak.
  https://openadmet.ghost.io/escape-from-flatland-cyp-edition/
- **"Practicing What We Preach: Why Open Science Needs Clear Rules"** (09-29) — the conduct policy written
  after the banning waves. https://openadmet.ghost.io/practicing-what-we-preach-why-open-science-needs-clear-rules/

---

## 10. Related competitions, and the calibration literature

### 10.1 The closest precedent is not a chemistry challenge — it is NeurIPS Open Polymer 2025

**An L1 metric on a hidden, distribution-shifted test set, where the podium was decided by one post-hoc
correction to one property.** Post-competition report: https://arxiv.org/abs/2512.08896 ·
https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025

The shift, in the organisers' own data (new structures plus a changed Tg-extraction method — explicitly
*not* a unit error, https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/608103):

| Tg split | n | mean / sd |
|---|---|---|
| train | 511 | 96 (median 74) |
| public | 95 | 102.9 / 103.7 |
| **private** | 166 | **179.8 / 134.9** |

The report's fitted public→private map is `a = 1.3, b = 46`. What that was worth:

- An **independent replicator** ran an unchanged pipeline to 0.09612 private; adding the single line
  `Tg*9/5+45` gave **0.07262**, while public barely moved (0.07265 → 0.07297). The shift was worth
  **0.0235 against 0.0059 for six rounds of modelling — about 4×.**
  https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/734755
- **1st place** (James Day, https://github.com/jday96314/NeurIPS-polymer-prediction): raw ensemble
  0.058 public / 0.089 private; with `Tg += std(pred) * 0.5644`, **0.054 / 0.075**.
- **2nd place** tuned an offset by probing (+273.15 → +300 → +30 → final +40) for 0.077 private, and
  afterwards found the *same* predictions with `(9/5)x + 45` would have scored **0.066**. He had discarded
  the scale because it hurt public. **The functional form differed between the probe subset and the scored
  subset** — a constant won on public, a 1.8 scale won on private.
- Forum: Tg ×1.3 reached gold, ×1.5 top 5, ×2.3 scored 0.073. "Not the first time a scaling factor decides
  MAE/RMSE competitions."
- 4th place baked the bias into the objective instead: **LightGBM quantile objective, alpha = 0.85** for the
  two shifted properties, 0.5 for the rest. 3rd place used fold-wise linear calibration (+5.5%).
  8th place harmonised the external data *before* training rather than correcting after.

**The V-shape probe, which is the transferable technique.** Because MAE is piecewise linear, score versus a
constant offset is a **V** with equal and opposite slopes. 1st place probed every property at ±0.1σ, found
only Tg wrong, submitted a handful of offset coefficients and **fitted a V to the leaderboard points**; the
apex at 0.5644 beat everything he had tried by hand. Three to five submissions, one or two parameters.

**The critical adaptation for ST-RAE: our metric has a deadband, so the bottom of the V is a plateau, not a
point.** §4.3's shift table shows the first ±0.1 costs only 0.003–0.013. So probe for the **centre of the
flat region**, not an apex — and §9.2's 2D6 probes (3.11 → 0.6944, 3.32 → 0.6271, 3.57 → 0.5502,
4.35 → 0.8558) are exactly a half-V already measured by someone else.

### 10.2 The same trick, and the same failure mode, elsewhere

**Where post-hoc correction won:** Allstate Claims Severity (MAE) — 1st place stacked 81 models then applied
**separate piecewise power maps to the low and high tails**, weights fitted by Nelder–Mead on out-of-fold
data. The forum thread quantifies the sub-tricks: a prediction-rank-based multiplier (0.95–1.05, chosen OOF)
gained ~3 MAE for default-objective GBMs, ~0.8 for a custom-MAE XGBoost, and **nothing** for Keras — i.e.
the correction is worth exactly as much as the model's own bias, which came from log-transform plus L2
retransformation rather than from the metric.
https://www.kaggle.com/competitions/allstate-claims-severity/discussion/25682

**Median blending for L1 metrics:** CHAMPS — a plain **median** across ensemble members moved the best single
model from −3.08 to ~−3.22, and the winning −3.245 came from medians of medians. For an L1-family metric,
median-aggregate rather than mean-aggregate.

**Where leaderboard-driven post-processing destroyed a team — read these before probing:**

- **Mechanisms of Action**: post-processing derived from public-test analysis improved public by 0.00011 and
  worsened private by 0.00008 — **11th → 118th**.
  https://www.kaggle.com/competitions/lish-moa/writeups/phi-3rd-place-public-we-should-have-trusted-cv-118
- **Novozymes**: Chris Deotte recovered public labels by probing and trained on them → **1st public, 967th
  private**. The split was by *residue position*, so public and private covered different regions. A depth-4
  random forest on the same features would have won.
  https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/writeups/chris-deotte-1st-place-public-shakedown-to-967th-p

**The rule that reconciles them:** Roelofs et al., NeurIPS 2019 — across 100+ Kaggle competitions there is
little adaptive overfitting, **and the failures cluster precisely where the public and private sets are
drawn differently** (Open Polymer, Novozymes).
https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html
Our live half and blind half are **different chemisimilar series** (§5), which is the risk case, not the safe
one. So: probe a *low-dimensional, mechanistically motivated* correction (a per-isoform location and scale),
never a per-compound or high-capacity one, and hold to mokhalaf22's published discipline — **act only on
board moves larger than ~0.03**. Blum & Hardt's "Ladder" is the formal treatment:
https://arxiv.org/abs/1502.04585

### 10.3 Why an L1/hinge metric wants a median, and where variance-matching stops being right

- **Gneiting 2011** (JASA 106, https://arxiv.org/abs/0912.0902): squared error is consistent for the **mean**,
  absolute error for the **median**, and a relative-error score for a density-reweighted median. His Table 5
  puts the median at 0.455× the mean for a chi-square-like predictive distribution. This is the formal
  statement behind §3.2 point 2.
- **A consequence we should not miss: "mean alignment" is the wrong intercept for an L1 metric.** In a
  simulation with right-skewed residuals (mean 0, median < 0), the LAD intercept was **−0.254** against OLS's
  **0.001** at identical slope, and MAE 0.713 vs 0.742. Our residuals *are* skewed — the inactive tail is
  one-sided. So the offset should be fitted on the hinge, not set from the mean residual.
- **Shrinkage theory, which cuts against naive variance matching:** Copas 1983 (JRSS B 45,
  DOI 10.1111/j.2517-6161.1983.tb01258.x) ties shrinkage to the slope of actual-on-predicted in new data;
  van Houwelingen & le Cessie 1990 give the uniform factor. A calibration slope < 1 means predictions are too
  extreme; ensembles and L1-heavy models can have slope > 1, which calls for *stretching*. But **von Storch
  1999** (J. Climate 12) shows variance inflation is inappropriate when the predictors explain only part of
  the variance — it raises RMSE. Reconciliation: **stretch when the test labels are genuinely wider than
  train (which §9.2 establishes they are, on all four isoforms), not merely because an error metric is in
  use.** The quantity to estimate is the test mean and sd — which is what briford did.
- Quantile/CDF recalibration (Kuleshov, Fenner & Ermon, ICML 2018, https://arxiv.org/abs/1807.00263)
  recalibrates *predictive distributions*, not point location and scale, so it does not address our problem
  directly. One practical warning: **isotonic maps go flat and create ties**, which would damage the Spearman
  and Kendall columns — use a strictly monotone map.

### 10.4 Target/label-shift estimation, the one principled route

- **Zhang, Schölkopf, Muandet & Wang, ICML 2013** (https://proceedings.mlr.press/v28/zhang13d.html) handle
  target and conditional shift by kernel mean matching, regression included.
- **Guo et al., ICML 2020** (http://proceedings.mlr.press/v119/guo20d/guo20d.pdf), label-transformation for
  continuous targets, beats kernel mean matching.
- For classification the mature line is BBSE (https://arxiv.org/abs/1802.03916) and the unified BBSE/MLLS
  view (https://arxiv.org/abs/2003.07554), where **calibration is the critical ingredient**.
- **A synthetic check worth repeating on our data:** under label shift with `p(f|y)` invariant, regressing
  `f` on `y` in train and then estimating the test mean and sd from *unlabelled* predictions as
  `(mean f − b)/a` and `sqrt(var f − σ²)/a` recovered the truth to 1.502/1.197 against 1.5/1.2, and a
  Gaussian posterior-mean update reached oracle test MAE (0.556) where the uncorrected forward map gave
  0.780 with bias −0.64. **This is the same inversion briford performed on the leaderboard, and it can be
  run on our own blind predictions with no board access at all.** Caveat: analog expansion also moves
  `p(f|y)`, and 2D6 is a different assay platform, so treat it as an estimator, not a guarantee.

### 10.5 TDC ADMET — what tops the CYP tasks, and why the ranks cannot be trusted

Scaffold split, 5 seeds. https://tdcommons.ai/benchmark/admet_group/overview/

CYP **inhibition** (Veith, AUPRC, n ≈ 12–13k): **MapLight+GNN leads all three** — 2C9 **0.859**,
2D6 **0.790**, 3A4 **0.916** — followed by ContextPred (0.839 / 0.739 / 0.904) and AttrMasking. Adding the
pretrained-GIN embedding to MapLight is worth **+0.076 / +0.067 / +0.035** AUPRC. CYP **substrate** tasks
(n ≈ 665) are topped instead by MiniMol / ContextPred / CFA, with Random Forest tied for third on 2C9.
Regression analogues: Caco2 (n = 906) is won by CaliciBoost 0.256 / XGBoost 0.274 / MapLight 0.276, while the
pretrained GINs collapse (ContextPred 0.502, AttrMasking 0.546); at n ≈ 10k (solubility) MiniMol leads.

**The pattern that matters for us: at n ≈ 1–2k per isoform — which is exactly our labelled count (§4.1) —
boosted trees over rich feature unions win and pretrained GNNs lose; the GNN features only start paying at
n ≈ 12k.** MapLight is CatBoost over ECFP + Avalon + ErG + 200 RDKit descriptors
(https://arxiv.org/abs/2310.00174).

**And the leaderboard is partly an artefact.** A reproduction audit
(https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full ·
https://doi.org/10.1021/acs.jcim.6c00819) found **only 3 of 10 families reproduce cleanly** (CaliciBoost,
MapLight, MapLight+GNN): MiniMol has pretraining leakage (stereo-stripped near-duplicates such as danazol);
XGBoost and GradientBoost leaked test molecules through a fixed-seed re-split during HPO (GradientBoost fell
1st → 3rd on ppbr_az, XGBoost 2nd → 8th on caco2_wang); CFA's code is unavailable, ZairaChem fails at
runtime, ADMETrix and SimGCN are irreproducible. The authors deliberately overfitted an in-house LightGBM
and took **10 of 22 tasks into the top 3** against 2 of 22 honestly — `cyp2d6_substrate` went from rank 12
to 1. **MapLight is top-3 on 17 of 22 tasks and is the one trustworthy backbone.** Their recommendation is
hidden test sets — i.e. exactly this challenge.

### 10.6 Censored labels and analog-series generalisation

- **Censored regression**: Svensson et al. 2024/25 (https://arxiv.org/abs/2409.04313) — Tobit/survival
  likelihood plus a one-sided `CensoredMSE`, over 15 AstraZeneca assays with 0–63% censoring. **Gains are
  largest at ≥35% censoring** — and our CYP3A4 is **40.4% below pIC50 4** (§4.1). Chemprop ships
  `BoundedMSE` with `lt_mask`/`gt_mask`, which is what alexxi11 (rank 44) used. Foundations: Tobin 1958,
  Powell 1984 (censored LAD — note medians commute with monotone maps, so CLAD is the natural fit for a
  hinge metric), Heckman 1979, Zadrozny 2004.
- **Activity cliffs**: MoleculeACE (https://doi.org/10.1021/acs.jcim.2c01073) — 30 targets, 24 methods,
  **SVM+ECFP best on cliffs** with no significant gap to GBM/RF on ECFP, and an MLP on ECFP beating every
  graph or SMILES network; descriptor choice mattered more than algorithm, and training-set size did not
  predict cliff error. ACNet (https://arxiv.org/abs/2302.07541) agrees across 400K+ matched pairs. SQRL
  (https://arxiv.org/abs/2501.09103) lifts deep nets a lot (MLP 0.32 → 0.73) but **lowers trees**
  (XGBoost 0.79 → 0.76; on cliffs 0.72 → 0.64–0.68) — consistent with mokhalaf22 finding SQRL/DeepDelta null
  here. CheMeleon is the counterexample, at 97% win rate on MoleculeACE
  (https://arxiv.org/abs/2506.15792).
- **Pairwise / delta learning** cuts error up to 60% across 274 datasets
  (https://www.frontiersin.org/journals/drug-discovery/articles/10.3389/fddsv.2026.1859068/full), **but it
  needs near-neighbour anchors** — and only **2.9–6.1%** of the blinded compounds have a ≥0.7 Tanimoto
  training anchor (mokh's independent figure: 13.3%). That is why it is null in this challenge specifically,
  not because the method is weak.

### 10.7 Polaris Hub is a dead end for this purpose

https://polarishub.io/competitions lists **only** the Jan–Mar 2025 antiviral competition (already covered in
`22-challenge-winners-and-intel.md`). Everything else on Polaris is a static benchmark with a random split
— e.g. Biogen ADME (2,812/704, random), topped by ChemLlama variants at MAE ≈ 0.324–0.332. No other Polaris
competition write-ups exist.

---

## 11. Open questions this pass could not close

- **Ranks 1–12 publish nothing.** preheat-to-450's method is unknown and will stay unknown until the
  4 Nov wrap-up blog posts and webinars. The only signal is its secondary metrics: ρ 0.7844 with R²
  0.6648, and per-isoform ρ 0.5428 on 2D6 — a *modest* 2D6 ranking turned into the field's best 2D6
  ST-RAE (0.4135), which is again placement, not ranking.
- **The web-search budget for this session was exhausted (200/200)** partway through, so §3–§8 rest on
  source code and on data we already hold rather than on commentary. §9's community sweep worked around it
  with site-native APIs (GitHub, HuggingFace, Discord invite metadata, Bluesky). X/Twitter could not be
  read at all (HTTP 402) and reddit.com was unreachable, so those two venues are genuinely unchecked rather
  than checked-and-empty.
- **No paper gives a recipe for calibrating point predictions to an unknown blinded distribution.** §10.4
  has the nearest theory (target/label-shift estimation) and a synthetic check that it works when `p(f|y)`
  is invariant; analog expansion and the 2D6 assay change both violate that, so it is an estimator and not
  a guarantee. `22-challenge-winners-and-intel.md` holds the PXR, ExpansionRx, Polaris/ASAP, CACHE, D3R and
  SAMPL material; nothing in this pass contradicts it.
- Not read, behind 403s: NovoExpert-2 (ChemRxiv), the AMS climate papers on variance inflation, and the
  JCIM version of the TDC audit (the bioRxiv preprint was used instead).
- **No per-compound ground truth is reachable.** The Space's `head_to_head.py` reads
  `ground_truth/activity-dataset.parquet` and `ground_truth/{track}-identifiers.parquet` (the latter
  carrying the `phase` column that defines the live half) from a private S3 bucket using server-side
  credentials. The scatter plot it renders is restricted to `phase == ACTIVITY_TARGET_PHASE` compounds. None
  of it is fetchable without the bucket, and no attempt was made.
- `codie-freeman/OpenADMET-CYP-Blind-Challenge` and `mileshuangg-alt/tripwire-project` are public,
  substantial, and unmatched to any board alias. The former's `docs/leaderboard_submissions.md` (65 KB) is
  probably a submission history worth reading.

---

## 12. Links

**Challenge and organisers**
- Space: https://huggingface.co/spaces/openadmet/cyp-challenge
- Space source (all app code, About/FAQ text, ban list): https://huggingface.co/spaces/openadmet/cyp-challenge/tree/main
- Live Gradio payload with every leaderboard: https://openadmet-cyp-challenge.hf.space/config
- Dataset: https://huggingface.co/datasets/openadmet/cyp-challenge-train-test
- Announcement: https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge
- Underway post: https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/
- TFM post: https://openadmet.ghost.io/are-tabular-foundation-models-all-you-need/ (DOI 10.5281/zenodo.23198306)
- Uncertainty post: https://openadmet.ghost.io/concerning-uncertainty/

**Code that scores us**
- Tutorial (participant copy of the metric): https://github.com/OpenADMET/CYP-Challenge-Tutorial
- **Backend (the real harness, tiering, bootstrap)**: https://github.com/OpenADMET/blind-challenge-template
- PXR TabICL recipe + ablation artefacts: https://github.com/OpenADMET/pxr-challenge-tabicl
- `openadmet-models` (CheMeleonFeaturizer, PretrainedModelFeaturizer, TabICLRegressor): https://github.com/OpenADMET/openadmet-models
- Leakage auditor: https://github.com/OpenADMET/challenge-data-checker — v0.3.3, Apache-2.0. Checks only
  *exact/near-identical* overlap (raw + canonical SMILES, InChIKey, identifier columns, plus a Morgan-r2
  bulk-Tanimoto fallback whose default `max_train_test_similarity` is **1.0**, i.e. fingerprint-identical
  only), within-pool duplicates, and quality flags. It does **not** test scaffold or series holdout or
  similarity distributions; open issues #5/#10/#11 admit salt-vs-free-base, stereoisomer and
  "similar ≠ leaked" gaps. No CYP-specific audit was ever published. An independent emulation found **zero**
  train/test overlap by canonical SMILES, full InChIKey or connectivity block.
- Platform open-sourcing post: https://openadmet.ghost.io/open-sourcing-our-blind-challenge-platform/
- Data-generation post: https://openadmet.ghost.io/lowering-inhibitions-one-cyp-at-a-time-building-the-openadmet-cyp-inhibition-and-tdi-dataset/
- Halfway post (the 2D6 distribution-shift admission): https://openadmet.ghost.io/halfway-through-the-openadmet-cyp-challenge/
- Pat Walters on the structural alerts and the 1,249 default-False TDI rows: https://openadmet.ghost.io/do-our-cyp-structural-alerts-actually-work/
- Octant assay protocols: https://github.com/OpenADMET/Octant_CYP_blog_post/blob/main/protocols/cyp_inhibition_assay.md · https://openadmet.github.io/octant-cyp-inhib-blog-post/
- CheMeleon: https://github.com/JacksonBurns/chemeleon
- Buterez et al. 2024, multi-fidelity transfer: https://www.nature.com/articles/s41467-024-45566-8
- N283T's PXR model report (the 0.41 entry the organisers reverse-engineered): https://n283t.github.io/openadmet-pxr-model-report/

**Community sweep (§9)**
- Discord "OpenADMET Challenges" (FAQ invite is expired): working invites https://discord.gg/FeSZCTf5sa ·
  `#cyp-challenge` https://discord.com/channels/1412827471488745545/1480419832787763412
- **briford's recalibration script with the solved blind moments**: `SuperCowPowers/workbench` →
  `ml_pipelines/OpenADMET/cyp/scripts/cyp_recalibrate.py` · blog https://supercowpowers.github.io/workbench/blogs/cyp_challenge/
- Rafal Wiewiora, 2D6 train-vs-blind separability AUC 0.972: https://github.com/rafwiewiora/openadmet-retrospective-reproducibility
- Codie Freeman's 9-post student series: https://codiefreeman.com/blog/
- Jackson Burns' notes/handoff: https://github.com/JacksonBurns/cyp-challenge
- lachrymator (stir_bar): https://github.com/lachrymator/openadmet-cyp-challenge-public
- avaliev / ADMET Bo$$: https://github.com/Rebell-Leader/OpenADMET_CYP
- TeamPozeSCAF (TDI leader): https://docs.google.com/document/d/1ohRu6s3tod3sjRF2gV3mYdoe9cHzgn-dYE5ruSyBrWg/edit
- Structure-track negative results: https://github.com/xX-its-amit-Xx/openadmet-cyp-structure
- Newsletter mention: https://thisweekincheminformatics.substack.com/p/thirty-five
- Structural-alert audit: https://openadmet.ghost.io/do-our-cyp-structural-alerts-actually-work/
- Fsp3 / flatland: https://openadmet.ghost.io/escape-from-flatland-cyp-edition/
- Conduct policy: https://openadmet.ghost.io/practicing-what-we-preach-why-open-science-needs-clear-rules/

**Related competitions and calibration literature (§10)**
- NeurIPS Open Polymer 2025 report: https://arxiv.org/abs/2512.08896 · competition
  https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025 · 1st-place code
  https://github.com/jday96314/NeurIPS-polymer-prediction · the one-line replication
  https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/734755
- Allstate post-processing thread: https://www.kaggle.com/competitions/allstate-claims-severity/discussion/25682
- The two cautionary shake-ups: https://www.kaggle.com/competitions/lish-moa/writeups/phi-3rd-place-public-we-should-have-trusted-cv-118 ·
  https://www.kaggle.com/competitions/novozymes-enzyme-stability-prediction/writeups/chris-deotte-1st-place-public-shakedown-to-967th-p
- Adaptive overfitting across 100+ competitions: https://proceedings.neurips.cc/paper/2019/hash/ee39e503b6bedf0c98c388b7e8589aca-Abstract.html
  · the Ladder https://arxiv.org/abs/1502.04585
- Gneiting 2011, consistent scoring rules (mean vs median): https://arxiv.org/abs/0912.0902
- Target/label shift: https://proceedings.mlr.press/v28/zhang13d.html ·
  http://proceedings.mlr.press/v119/guo20d/guo20d.pdf · BBSE https://arxiv.org/abs/1802.03916 ·
  unified view https://arxiv.org/abs/2003.07554
- Shrinkage/inflation: Copas 1983 DOI 10.1111/j.2517-6161.1983.tb01258.x · van Houwelingen & le Cessie 1990
  DOI 10.1002/sim.4780091109 · von Storch 1999 DOI 10.1175/1520-0442(1999)012<3505:OTUOII>2.0.CO;2
- Distribution recalibration: https://arxiv.org/abs/1807.00263
- Censored labels: https://arxiv.org/abs/2409.04313 · Chemprop `BoundedMSE`
  https://chemprop.readthedocs.io/en/latest/tutorial/python/loss_functions.html · Powell 1984 censored LAD
  DOI 10.1016/0304-4076(84)90004-6
- Activity cliffs / pairwise: MoleculeACE https://doi.org/10.1021/acs.jcim.2c01073 · ACNet
  https://arxiv.org/abs/2302.07541 · SQRL https://arxiv.org/abs/2501.09103 · pairwise review
  https://www.frontiersin.org/journals/drug-discovery/articles/10.3389/fddsv.2026.1859068/full
- TDC ADMET group: https://tdcommons.ai/benchmark/admet_group/overview/ · MapLight
  https://arxiv.org/abs/2310.00174 · MiniMol https://arxiv.org/abs/2404.14986 · **reproduction audit**
  https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1.full
- Polaris competitions (only one exists): https://polarishub.io/competitions

**Participant write-ups newly read in this pass**
- Scigantic (19 / **11 interim**): https://github.com/Scigantic/open-admet-challenge-cyp
- Woong (146): https://gist.github.com/ByungwoongYoo/4904441a9b30db077fbb784f9c22e423
- AIv_2709 (156): https://github.com/avnikonenko/openadmet_challenge_cyp_2026/blob/main/docs/method_report.md
- Unmatched to a board alias: https://github.com/codie-freeman/OpenADMET-CYP-Blind-Challenge · https://github.com/mileshuangg-alt/tripwire-project

The other 13 reports are tabulated in §2 and analysed in `24-leaderboard-model-reports.md`.

---

## 13. Provenance of this file

Leaderboards: extracted from the Space's own Gradio config payload, 2026-10-08, components 58/61/64/67/70
(live, n = 265) and 92/95/98/101/104 (interim, n = 219). Scoring behaviour: read from
`OpenADMET/blind-challenge-template` and `OpenADMET/CYP-Challenge-Tutorial` at HEAD. Band geometry,
denominator mass, placement and shrinkage sensitivity in §4: **computed in this session** from
`~/.cache/huggingface/hub/datasets--openadmet--cyp-challenge-train-test/.../cyp-challenge-TRAIN_inhibition.csv`
(4,905 rows), using the organisers' own metric implementation. Provenance quotes in §5 are verbatim from
the Space's `config.py`. Nothing was submitted, no repository was cloned or run, no leaderboard was
touched, and no account was created.

All web content was treated as data. One item to flag under that heading: the
`codie-freeman/OpenADMET-CYP-Blind-Challenge` repo contains a 295 KB `CLAUDE.md`, i.e. a file of
instructions addressed to an AI agent. It was **not** opened or acted upon; it is recorded here only as an
observation that at least one other entrant is running an agent-driven pipeline of the same shape as ours.
