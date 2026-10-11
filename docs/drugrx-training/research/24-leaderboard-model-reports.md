# 24 — What the public CYP-challenge model reports actually say

Read 2026-10-08. Thirteen entries whose Model Report field points at a public page, read from the
public pages only (GitHub raw/API, two static sites, one Google Doc export). Nothing was cloned,
nothing was run, no leaderboard was touched.

Our position for reference: macro ST-RAE **0.866** (rank ~170); per isoform **1A2 0.88 / 2C9 0.52 /
2D6 1.46 / 3A4 0.57**. Leaders ~0.37.

**Field context, which reframes our gap** (450nm's own board scrape, n=138, dashboard card A64):
field *median* per-isoform ST-RAE is **1A2 0.752 · 2C9 0.539 · 2D6 1.303 · 3A4 0.568 · macro 0.773**.
So our 2C9 (0.52) and 3A4 (0.57) are already at or slightly better than the field median. Our whole
deficit is **1A2 (0.88 vs 0.75) and 2D6 (1.46 vs 1.30)**, and 59.4% of that field is *worse than a
constant* on 2D6. The 2D6 leader is 0.489. 2D6 carries the largest absolute headroom of any cell on
the board and the largest for us.

---

## Summary table

| # | alias | macro | (a) single-conc / unlabelled | (b) 2D6 + inactives | (c) external | (d) features / models | (e) loss + post-proc | licence |
|---|---|---|---|---|---|---|---|---|
| 12 | briford | 0.4378 | separate aux heads, never pooled rows; censored variants published | affine recalibration onto a qHTS-derived blind prior; 2 of 4 members are 2D6 specialists | ChEMBL 37, AID 1851 (keeps `max_response`), Tox21 (built, unused) | 4 Chemprop D-MPNNs, SMILES only, stock HPs | MSE; **affine placement, k = ρ·sd** | Workbench repo |
| 16 | rasayan-labs | 0.4443 | — | — | — | — | — | repo **empty** (0 KB) |
| 33 | stir_bar | 0.4769 | masked multi-task aux pretraining | — (no explicit inactive handling) | public bioactivity for pretraining; 88,683 retrieved blind-set neighbours, 4,999 admitted properties-only | transformer + MPNN + 3D + Mitra/TabICL on frozen emb + FP + fragment | **non-negative stacking**; Butina folds | NOASSERTION, write-up only |
| 40 | alexxi11 (datagrok) | 0.5069 | per-source heads; AID 1851 inactives as **left-censored bounds** | per-isoform **soft-RAE-optimal additive shift** | AID 1851 (+inactives), Veith, NCGC, ChEMBL 37, Tox21, PharmaBench, Octant | Chemprop multitask, CheMeleon warm start | **`BoundedMSELoss` + `lt_mask`**; bias shift | MIT |
| 46 | JacksonBurns | 0.5181 | aux heads (log2fc ×4, w 0.3) in a 23-head model | **`BLIND_MOMENTS` z-placement**, 2D6 on ST-RAE-specific moments, floor 1.0 | ChEMBL 37 (36k), AID 1851 (14.5k) | LGBM + CheMeLeon/chemprop FT + **TabPFN on frozen emb** | MSE; moment placement + `F_SPREAD` | none |
| 47 | jeremy | 0.5185 | log2fc as aux heads; **pseudo-labelling hurt badly** | blind-moments rescale; 2D6 R² −0.749 → **+0.377** | AID 1851 (population-matched), ChEMBL 37 | LGBM + TabPFN on frozen chemprop; 17-head multitask | MSE; **blind-population placement** | none |
| 53 | Lizard Wizard Gizzard | 0.5357 | **Heckman-style joint likelihood**, screen as a second head | reads blind band geometry off the board; 2D6 cell 0.771 → 0.681 | none (organisers' files only) | GBM + GP + trunk, 5 members, per-enzyme subset search | **dead zone: `clip(p, lo, hi)`**; per-fold affine pair | MIT |
| 69 | 450nm | 0.6004 | **Emax+std_error-calibrated row imputation** of sparse cells | variance recalibration 0.955 → 0.900; 2D6 1.794 → 1.668 | AID 1851/884/891, NCGC panel | LGBM + mechanistic blocks, per-isoform configs | band weighting; **band-edge-asymmetry offset** | dashboard + structure repo |
| 79 | CompChem Magicano | 0.6404 | **predicted** log2FC as 4 features (best single gain) | validated **−0.5 2D6 location shift**; two dispersion expansions | AID 1851 (hurt), ChEMBL/PubChem neighbours (null), Octant (**leakage**) | from-scratch D-MPNN multitask, 3 seeds | **interval hinge**, uniform weights | MIT |
| 87 | adlvdl | 0.6578 | "main source of auxiliary signal" (baseline only published) | linear calibration 1.063 → 0.947 on 2D6 | — | LGBM/XGB on ECFP4, nested 5×5 scaffold CV | linear calibration; Tukey-HSD gating | MIT |
| 122 | leeherman | 0.7585 | — | "affine transformation to 2D6 predictions (as per Brian Wylie)" | — | Chemprop v2 + CheMeleon fingerprints, AutoGluon | quantile + margin-ranking loss (failed) | Google Doc |
| 127 | popcorn2 | 0.7671 | briford's aux heads, task weights 0.3 | **none — no calibration implemented** | ChEMBL 37, AID 1851 `max_response` | briford's exact 4 Chemprop members | none | none |
| 132 | ADMET Bo$$ | 0.7726 | pseudo-labels help **only after clipping at the assay floor** | 2D6 = orthogonal SAR, basic-nitrogen driver | public augmentation **hurt** (+0.057) | ECFP4+RDKit → SVD-256; TabICL + multitask MLP blend | blend by nested inner CV; +0.12 CV→board constant | NOASSERTION |

---

## 12 · briford — 0.4378 · https://supercowpowers.github.io/workbench/blogs/cyp_challenge/

**(a)** Auxiliary readouts enter as **separate encoder heads, never as extra rows in the scored
column**. Stated reason: "every extra source measures something adjacent to the scored target, not
identical to it" — the screen reports a fold-change, ChEMBL reads ~0.5 log more potent on shared
compounds, qHTS reports efficacy at the top concentration. Pooling needs a cross-assay correction;
separate heads need none, "since the encoder learns from all of them and each head keeps its own
scale." Every public source is also published in a **censored variant** where `IC50 > x` or an
inactive call carries the bound with an `_lt` flag rather than a null — "a bound is not a
measurement" — though they report it "has not moved the scored metric" for them.

**(b)** The whole 2D6 strategy is a prior-driven affine move. Two filters create the training
population: a compound was screened because someone thought it might be active, and labelled only
if its curve fit — "Compounds that did nothing produce no curve and no label." The blind set is hit
expansion on 1A2/2C9/3A4 only, so **2D6 "got no enrichment at all."** The gap is estimated *before*
submitting: the qHTS panel puts **2D6 inactivity near 65%**, a set that inactive **centres near
pIC50 3.7, against the 4.69 a curve-fit-trained model predicts**, and the blind population should
also be *wider* because squared-error models shrink and a fit-only label set is already narrower
than its parent population. Raw 2D6 predictions "sit in a narrow band well above the centre the
qHTS prior implies, and the model never calls a compound a non-inhibitor." No flooring, no
clipping, no classifier stage.

**(c)** Challenge (4,905) + **ChEMBL 37** (24,918 skeletons, 185 overlap the deck, zero overlap the
blind set; nothing below pIC50 4.0 because potency exists only where a curve fit) + **PubChem AID
1851 / Veith** (16,546 compounds, 15-point; they keep **`max_response`, not pIC50**, because it is
at 100% coverage so "the 42,355 rows the screen called inactive still carry signal") + **Tox21 CYP**
(7,196, P450-Glo triplicate, "the only public source weak-inhibitor-rich enough to reach below
pIC50 5.0" — built but not in the entry). Merged on **standardised InChIKey connectivity blocks**
so salts and stereo variants match. Final feature set 31,670 compounds / 161,400 labels.

**(d)** Four Chemprop D-MPNNs, SMILES-only, stock hyperparameters, differing only in which targets
share an encoder: `union-p30` (all four isoforms, challenge+ChEMBL+qHTS, 18 heads, public weight
0.30), `mt-aux-100` (challenge arms only, 8 heads), `2d6-isoform` (every 2D6 readout, nothing
else, 6 heads), `2d6-single` (2D6 pIC50 alone). 1A2/2C9/3A4 average two members, **2D6 averages all
four** — "a different slice of the data, so different compounds missed." Out-of-fold on 2D6 the
average beats every member it contains, 0.503 Spearman against 0.445 for the best single.

**(e)** The reusable core. Murphy's (1988) decomposition:

```
R² = 2ρk − k² − b²        k = sd(pred)/sd(true),  b = mean offset in sd(true) units
```

Only ρ depends on ordering, so `R² ≤ ρ²` is a hard ceiling and **"a model far below its own ρ² is
mis-calibrated, not weak."** The optimum is **k = ρ, not k = 1**: "a model with correlation ρ should
be ρ times as wide as reality." Recipe: estimate the target population's centre and spread, then
calibrate each isoform onto it with spread ρ·sd. Spearman and Kendall return bit-identical, which
doubles as the integrity check. **The documented catch:** "The calibration that maximises R² does
not minimise the scored metric" — ST-RAE is zero inside a band, low-activity compounds have wide
bands, so predicting high is nearly free and predicting low is punished by the actives. The ST-RAE
optimum sits **above** the population centre, and "on CYP2D6, the lowest-centred isoform, the two
objectives point in opposite directions." They say the direction is known, the distance is not, and
that **"a credible-interval width model would give it directly."**

**(f)** Each member trained on 100% of the data, no held-out splits. Honest caveat: "Repeated
training runs of one configuration at different seeds disagree by more than candidate models do,
so out-of-fold cross-validation cannot resolve the differences." Out-of-fold only used for ensemble
design on 2D6.

**(g)** Macro 0.4378 on the live half; per-isoform board cells not disclosed.
**(h)** `ml_pipelines/OpenADMET/cyp` in github.com/SuperCowPowers/workbench; eight named scripts.

---

## 16 · rasayan-labs — 0.4443 · https://github.com/whis9/rasayan-cyp

**The repository is empty** (GitHub API reports `size: 0`, no tree, no README, last push
2026-09-19). Nothing to extract. JacksonBurns' notes record the same finding independently
("EMPTY repo, nothing to mine"). A rank-16 method with 0.4443 therefore has no public description.

---

## 33 · stir_bar — 0.4769 · https://github.com/lachrymator/openadmet-cyp-challenge-public

Write-up only; the training code is not published. Claims rank 28/219 regression and **rank 6/114
TDI (Tier 1)** on the organisers' full-set interim board.

**(a)** "Masked multi-task pretraining on public bioactivity data, warm-starting each backbone
before the high-fidelity fine-tune." No imputation/censoring/pseudo-labelling described.

**(b)** No inactive-specific handling described. Their reported 2D6 is the worst cell: OOF MAE
0.712 / R² 0.585, blind ST-RAE 0.653, MAE 1.141, R² 0.290, ρ 0.449.

**(c)** The distinctive lever is **retrieval into the blind neighbourhood**: 88,683 near-neighbours
of the held-out structures pulled from a large public catalogue, filtered to the blind set's own
physicochemical envelope, screened with `rd_filters` (~1,150 alerts; selection is *subtractive* —
"any alert that fires on even one held-out compound is dropped", ~75 of 1,249), then the **4,999
closest admitted with only calculated physical properties and no assay endpoints**. Those
calculated-property heads are **down-weighted per task**, because a property computable from
structure labels every row and otherwise supplies most of the pretraining gradient.

**(d)** Representation-diverse base learners: SMILES transformers, MPNNs, 3D-conformer models,
tabular foundation models over frozen embeddings, a fingerprint baseline, an additive-fragment
model. The tabular legs use **Mitra** (AutoGluon, Apache 2.0); Mitra beat TabPFN on macro held-out
R² in 6 of 7 matched embedding sources and their residuals correlated 0.94–0.98, "they were making
very nearly the same mistakes."

**(e)** Four controls on the stacker, each added after measuring a failure:
- **Non-negative stacking.** Residual correlations between members run 0.55–0.88, so unconstrained
  ridge/elastic-net returned cancelling coefficients (+1.9 against −0.6; one endpoint's best
  standalone model went *negative*, used as a correction term). "One revision improved
  cross-validated error on all four endpoints while real held-out error got worse on three."
- **PCA feature budget** on every embedding source before a tabular model sees it.
- **Per-source legs** rather than one blended tabular column.
- **Pruning by correlation, not count** — dropping correlated tabular legs improved all four
  endpoints; dropping *representation-diverse* members cost most of it back.

**(f)** **Butina cluster-disjoint folds**, clusters assigned whole. "Random splits flattered every
model. Butina folds made the held-out numbers track the leaderboard." All reported numbers OOF.
Their CV-vs-blind audit: **100% of regression submission-endpoint pairs came out worse on the blind
set, median MAE +0.208.**

**(g)** Blind interim per isoform (ST-RAE / MAE / R² / ρ): 1A2 0.441 / 0.656 / 0.660 / 0.826 ·
2C9 0.380 / 0.480 / 0.686 / 0.851 · 2D6 0.653 / 1.141 / 0.290 / 0.449 · 3A4 0.386 / 0.475 / 0.743 /
0.863 · macro 0.465 / 0.688 / 0.595 / 0.747.

**(h)** NOASSERTION licence; write-up only. Stack: RDKit, PyTorch, Chemprop, scikit-learn, HF
Transformers, AutoGluon.

---

## 40 · alexxi11 / datagrok — 0.5069 · https://github.com/datagrok-ai/admetica

Code under `examples/cyp_challenge/`. The clearest *implemented* treatment of censoring on the
board.

**(a)/(b)** `build_multisource_dataset.py` builds **one column per source × isoform**, with the
stated reason: "Pooling calibrated sources dragged the scored target down to the qHTS spread (~0.6)
when the challenge assay's own spread is ~1.0; keeping sources in their own columns lets each become
its own prediction head, so the scored head trains only on challenge-scale values." AID 1851's
inactives (`aid1851_inactives.csv`, "every value in this file is an upper bound (4.244), never a
measurement") are admitted as **left-censored labels** with a parallel `<col>__lt` boolean; a fitted
measurement from any export always wins over a censored bound for the same compound. Training uses
Chemprop's **`nn.BoundedMSELoss()`** with `lt_mask` per datapoint, i.e. zero loss when the
prediction is already below the bound. There is a direct **`multihead_censored` vs
`multihead_nocensored` ablation** in `results/`, and a `--no-censored` flag.

The post-processing is a **per-isoform soft-RAE-optimal additive constant** fitted on champion OOF
predictions over a grid −0.30…+0.60 step 0.02 (`results/bias_shift/check.txt`), described as
correcting "the systematic under-prediction the pooled qHTS training data introduces":

| endpoint | per-fold optima | shipped | OOF RAE before → after |
|---|---|---|---|
| cyp1a2 | 0.30 0.28 0.16 0.18 0.28 | **+0.24** | 0.8829 → 0.8139 |
| cyp2c9 | 0.16 0.14 0.20 0.10 0.20 | **+0.16** | 0.8308 → 0.7763 |
| cyp2d6 | 0.02 0.00 0.10 0.06 0.04 | **+0.04** | 0.8969 → 0.8941 |
| cyp3a4 | 0.16 0.14 0.20 0.18 0.14 | **+0.16** | 0.6017 → 0.5698 |
| macro | | | **0.803 → 0.764** |

All folds agree in sign per endpoint. Note the shift is **positive** (upward) here — the opposite
direction from the 2D6 downward shift everyone else applies, because their pooled qHTS training
data had already biased them low.

**(c)** AID 1851 (fitted + inactives), Veith, **NCGC panel**, ChEMBL 37 (OpenADMET's curated panel,
`pchembl_value_mean`), Tox21 luciferase (**with a `luciferase_artifact` filter — "luciferase
inhibitors read as CYP inhibitors in this luminogenic assay"**, kept/removed files are separate),
PharmaBench, and **Octant's own CYP3A4 release restricted to `drc_qc_status == PASS`** ("the
challenge assay, run by the challenge lab" — see the leakage warning under Magicano below).

**(d)** Chemprop multitask D-MPNN, one head per source × isoform, `--pretrained` warm start from
CheMeleon weights.

**(f)** 5-fold; early stopping on `val_loss`. A logged negative result: monitoring *only* the scored
heads (`--monitor-scored`) **made things worse** — macro 0.8172 → 0.8820 on fold 1 — because "the
aggregate signal was acting as a stabiliser" for heads carrying only 1,100–1,900 labels each.

**(h)** MIT.

---

## 46 · JacksonBurns — 0.5181 · https://github.com/JacksonBurns/cyp-challenge

A 1,674-line competition journal (`NOTES.md`) with four scored board points and a full dead-end
ledger. The single most useful public artefact for calibration numbers.

**(a)** `single-concentration-TRAIN.csv` enters as **four `log2fc` auxiliary heads at weight 0.3**
inside a 23-head NaN-masked multi-task model (direct pIC50 ×4 primary, TDI-condition pIC50 ×4 at
0.5, `is_TDI` ×2 BCE at 0.5, log2fc ×4 at 0.3, ChEMBL ×4 + qHTS ×4 at 0.3). Verified fact recorded:
log2fc covers 4,376 molecules × 4 enzymes and correlates with direct pIC50 at **Spearman −0.83 to
−0.94**. A stacked multitask with TDI-arm aux was *worse* than the plain blend (0.770 vs 0.743 OOF).
External rows are flagged **"NOT DRC-calibrated — use only for RANKING (pretraining / multitask
auxiliary / similar-compound features), never placement moments."**

**(b)** The `BLIND_MOMENTS` placement, which is the public reverse-engineering of the live scored
half's per-isoform mean/sd (attributed to SuperCowPowers, via R² under affine transforms):

```
CYP1A2 4.412 / 1.553      CYP2C9 4.830 / 1.101
CYP2D6 3.107 / 1.599      CYP3A4 4.880 / 1.272
```

against train label means ~4.96 / 4.58 / 4.78 / 4.10. Test predictions are z-scored per isoform onto
those moments with `target_sd = clip(ρ_oof × inflation, 0.05, 0.95) × sd_blind`, where ρ_oof is OOF
Pearson (0.527 / 0.604 / 0.399 / 0.771) and `OOF_TO_BLIND = 1.32 / 1.23 / 1.66 / 1.07`. **CYP2D6 is
deliberately placed on *ST-RAE-specific* moments, mean 3.57 sd 0.90, not the R²-optimal
3.107/1.599**, "because 2D6 test is near-random." All predictions **floored at 1.0**. A later
`F_SPREAD = {1A2 0.6, 2C9 0.6, 2D6 1.0, 3A4 0.7}` shrink was adopted after simulation.

The crucial correction after their first scored point: blind MA-R² (0.387) ≈ mean squared OOF
Pearson (0.349), so blind correlation ≈ OOF correlation (ratio ~1.05, **not** the 1.32 assumed) —
"our placement is already near R²-optimal. The remaining gap is 100% RANKING quality." Their own
placement probe reproduced briford's conflict: **R² is maximised at spread factor f ≈ 1.2–1.35,
ST-RAE at f ≈ 0.9–0.95**, and near f=1 the two are roughly zero-sum.

**(c)** ChEMBL 37 CYP IC50 via paginated API: 36,015 activities, 14,453 unique SMILES (1A2 5,924 /
2C9 7,520 / 2D6 8,684 / 3A4 13,887). PubChem AID 1851 via batched PUG-REST POST with recursive
splitting: 14,496 CIDs, `Fit_LogAC50`-derived pIC50 + binary Active/Inactive. Board observation that
drove this: **"top ~20 ALL have external data ... or non-fp deep features."**

**(d)** RDKit descriptors + Morgan counts r2/2048 + Morgan r3/1024 + MACCS (3,455 cols) → 4 LightGBM
configs × 3 seeds, plus k-NN **target** features from Morgan-Tanimoto nearest labelled rows
(self-exclusion mandatory in CV or the model leaks). Then CheMeLeon frozen embeddings (ridge probe
alone nearly matches the whole FP+desc GBM on 2C9/3A4), CheMeLeon/chemprop fine-tunes, a
**briford-style 2D6-slice specialist** (`dmpnn_2d6.py`, external rows restricted to 2D6 readouts,
off-slice aux weight 0 — "best 2D6 single in the pool", blend weight 0.30–0.35 on 2D6), and finally
**TabPFN-v2 on PCA-128 frozen chemprop-medium embeddings**, which was their biggest honest gain:
nested macro R² 0.4584 → **0.4682 (+0.0098), winning on all four isoforms**, and a paranoid
family-block audit showed the gain is new non-linear signal from the *same* embeddings the linear
ridge missed.

**(e)** MSE; all metric awareness is in placement. A **dead-zone refit** against `clip(OOF, lo, hi)`
was tried (replicating Lizard Wizard's lever) and came out **neutral** in their pool: cp17 0.4583 vs
cp16 0.4584 on R², 0.7633 vs 0.7631 on train ST-RAE; the standalone dead-zone member was 0.9177 —
"a weaker ranker fitted to the clipped target adds no in-band accuracy." They also note the
**oracle** version (clipping the truth into its own band) is degenerate and scores 0.0000, so it is
not a real target. Per-compound band-width placement was found **infeasible** because
`TEST-BLINDED.csv` carries only `Molecule_Name` and `SMILES` — no DRC bands.

**(f)** Scaffold-grouped 5-fold (Murcko), later fold-nested blends with **family-block LOFO audits**
to check a gain is not family-internal. Nested→blind conversion measured at k ≈ 1.04–1.06 on R²
across three scored points.

**(g)** Scored board points: interim 0.6766 (rank 93/229, MAE 0.808, R² 0.387, ρ 0.616) → Sep 26
**0.5354** (57/219, R² 0.519, ρ 0.725) → Oct 3 0.5366 (58/255, R² 0.496, ρ 0.709). Same file scored
0.5192 and 0.5366 on two boards, so **board-side noise is ~0.02 per reveal**. Scaffold-OOF of the
pure-GBM blend was 0.743 (1A2 0.821 / 2C9 0.692 / 2D6 0.947 / 3A4 0.512).

**(h)** No licence file; public repo.

**Dead-end ledger worth not repeating:** CheMeLeon-2048 ridge probe (zero blend weight);
near-test retrieval on external aux rows (+0.0002, i.e. stir_bar's retrieval lever did **not**
transfer); Uni-Mol 3D encoder; frozen ChemBERTa SMILES transformer (standalone NO-GO on 0/4
isoforms *and* pool-redundant — zero greedy weight); qHTS-only domain adaptation (clears the
standalone gate, best 2D6 single at ridge-probe level 0.378, but **pool-redundant**: the 2D6 signal
was already owned by existing members); SimCLR-style SSL pretraining.

---

## 47 · jeremy — 0.5185 · https://github.com/jeremycheminf/openadmet_scripts/tree/main/CYP_Challenge

The origin of the blind-moments move, and the cleanest before/after on it.

**(a)** The log2fc screen enters as **4 auxiliary heads inside a 17-head multitask chemprop**
(4 challenge pIC50 + 4 log2fc + 4 ChEMBL pIC50 + 5 PubChem qHTS). Explicit negative result:
**"Pseudo-labeling missing pIC50 rows from the single-concentration screen — hurt badly (assay
saturation at high concentration); using the same data as a genuine auxiliary multitask target ... is
fine, treating it as ground truth for missing rows is not."**

**(b)** Pure recalibration, no classifier or clipping. The numbers:

| submission | macro ST-RAE | CYP2D6 R² |
|---|---|---|
| own-population placement | 0.7247 | **−0.749** |
| blind-moments-calibrated | **0.5214** | **+0.377** |

Same models, same ordering. Their own population has 2D6 mean ≈ 4.8; the true blind population
mean ≈ 3.1. They fit SuperCowPowers' affine placement (`k = ρ`) but **against the blind population's
moments rather than their own**, shrunk by their own OOF Pearson ρ.

**(c)** **PubChem AID 1851** is chosen deliberately over ChEMBL on *population shape*:
`adme_pretrain`'s ChEMBL-sourced CYP data "runs ~1–1.65 log units more potent on average than this
challenge's own diversity-library screening population, which is exactly what makes *fine-tuning*
on it hurt (the encoder gets pulled toward a population it won't see at test time)." AID 1851 is the
NCATS qHTS panel over ~16.5k substances from the "Biodiverse" collection — much closer in shape.
Fetched via the bulk per-SID CSV export, because **the "concise" export only has binary outcomes,
not the continuous `Fit_LogAC50`**.

**(d)** Five regression models, Caruana-combined: (1) LightGBM on ECFP4+RDKit2D; (2) **TabPFN on
frozen `chemprop_medium` embeddings — best single model**; (3) TabPFN on frozen
`chemprop_chemeleon`, "a genuinely decorrelated second cluster"; (4) the 17-head multitask chemprop
fine-tune, "the single largest ensemble contributor." Two doctrines: **"freeze, don't fine-tune"**
(fine-tuning consistently hurt — a population-mismatch effect) and **Caruana bagged ensemble
selection over a plain mean or NNLS** ("structurally robust to correlated candidates and doesn't
destructively reallocate weight the way a continuous optimizer like NNLS can on true blind data").

**(e)** `calibration.py` is nine lines: `np.polyfit(y_pred, y_true, 1)` on OOF, apply the same two
scalars to test. The documented refinement is to retarget the *destination* moments, not the fit.

**(f)** Butina folds (inherited from their PXR-challenge lessons). Kit OOF macro ST-RAE 0.614 vs
full research ensemble 0.604 — and they state plainly that the true-blind score with blind-moments
calibration is **better than the OOF number suggests**, "precisely because the fix targets the
population actually being scored."

**(h)** No licence file. Checkpoints `chemprop_medium.pt` / `chemprop_chemeleon.pt` and
`cyp_chembl.csv` committed in-repo. Also tried and dropped: Monroe GRIT transformer (real
decorrelated contributor), AutoGluon, 3D shape/polarity, vanilla CheMeleon (redundant with classical
descriptors), **qHTS-only domain adaptation = "best-ever single-model CYP2D6 score"** (left out of
the kit for script count), SimCLR SSL ("the worst embedding source we tried").

---

## 53 · Lizard Wizard Gizzard — 0.5357 · https://github.com/doctawho42/CYPChallenge/blob/main/docs/method_report.md

Organisers' files only, no external data. The deepest treatment of the metric itself, and the only
entry that writes down the selection problem as a statistical correction.

**(a) The Heckman move — the finding most directly on our discovery.** The screen file is 17,504
rows, 4,376 molecules × 4 enzymes at 4.95e-5 M; **11,509 of those readings have no dose-response
curve, 1.8× the 6,525 labelled pairs.** They state the reason the readings cannot be labels: "curves
were run on what the screen flagged: on CYP2D6 every curve compound is screen-active against **5.7
per cent** of the rest." Then:

> "That selection is what makes a joint likelihood legitimate rather than a hazard. The missingness
> is not at random, but **the selection variable is observed**: written through a latent activity,
> the factor `p(R | y_screen)` carries no parameters and leaves the gradient, **provided the screen
> enters as an observation and not as an input**. It is a Heckman correction whose first stage is
> measured rather than modelled."

Sized by simulation with known truth (40 repeats, 6,000 compounds, curves on the top 28% by screen):
**fitting the selected labels alone recovers coefficients at calibration slope 0.731; the joint fit
at 1.012.** Control: select on the *latent* value instead and hide the screen for the unselected, and
the joint scheme fails as badly as the naive one (0.681). Implementation: a shared trunk with two
heads — pIC50 from the curves, log2fc from the screen — and **no screening column at inference**,
since the 750 test molecules have none. It multiplies the trunk's supervision by 2.76. Against its
own control (identical network, screening head receiving no gradient) the channel is worth
**−0.0264 ST-RAE after the affine pair (p = 0.002) and +0.0350 of rank**; the architecture alone
gives −0.0008. In their headline table it is the largest effect in the project at **+0.0308 macro
rank, 4/4 seeds, t = +18.61, p = 0.0003**.

Also: the TDI table's **1,238 rows outside the modelled set** have placeholder all-negative
`is_TDI` while 1,048 of them (84.6%) clear the rule's second branch — their `pi_TDI` is a real
measurement and 53% more rows for the 3A4 arm regressor.

**(b)** No clipping or constant. The 2D6 fix came from reading the blind band geometry off the
board (below), and it is the single largest board move they report: **the 2D6 cell went 0.7707 →
0.6805 and the entry ten board places.** Their diagnosis of why their own optimiser had missed it is
the sentence we should pin up:

> "Band width is close to a step function of the label: below pIC50 ≈ 4.13 the mean width on our own
> data is 2.379, above it 0.21–0.44 ... Our placement is fitted out of fold against our own labels —
> and the direct-inhibition assay was run on compounds a screen had already flagged, so **9.8 per
> cent of our CYP2D6 training mask lies under the step against 66–73 per cent of the blind set. The
> optimiser had the right bands and the wrong population.**"

**(c)** None used. Closed with measurements: pretrained representations (chemprop `rdkit2d`
−0.0075 rank; MoLFormer-XL 0.5637 vs 0.5651); **ChEMBL 8,068 compounds — a raw gain of 0.0128
becomes +0.0068, i.e. harmful, after the affine pair**; NCGC panel 54,177 rows (+0.0048, below the
0.007 floor); quantum descriptors null.

**(d)** Five members over a shared matrix (2,048 Morgan counts r=2, 217 RDKit descriptors, **30
mechanistic columns**), member set chosen **per enzyme** by search over all 31 subsets: per-enzyme
GBM, a contrast-pooled model, a Gaussian process, a neural trunk, a re-projection of the trunk.
Shipped: 1A2 = per-enzyme+GP+trunk; 2C9 = GP+trunk; **2D6 = all five**; 3A4 = GP+trunk. The
mechanistic block (nitrogen/acid counts, protonation state under a pKa rule, distances from a basic
nitrogen to an aromatic ring) is worth +0.0163 rank, +0.0313 with rows reweighted toward the test
regime — and permuting its ten **salt-bridge** columns costs **+0.1545 of rank on CYP2D6** above a
size-matched random block, while charge-derived descriptors fall below their own null. "The block
delivers geometry of the basic centre, not the protonation call and not the charge." A trained pKa
predictor is bounded at nothing (a full unit of pKa noise moves macro rank −0.0020).

**(e) Two reusable constructs.**

*The dead zone.* ST-RAE's per-compound loss is `L(p) = |p − clip(p, lo, hi)|`; absolute error
against `clip(p, lo, hi)` majorises it and is tight at the optimum, so the bands can be **optimised
against directly**. Every member is refit by majorise–minimise against the clipped target: **+0.0197
of rank, sign holding in all 16 per-enzyme cells.**

*P(hit).* The metric-native uncertainty object is not an interval around the prediction but
`P(hit_i) = P(lo_i ≤ ŷ_i ≤ hi_i)`, the per-compound probability of scoring exactly zero. OOF,
**31.3% of their predictions already score exactly zero** (per enzyme 0.247 / 0.439 / 0.198 /
0.369). It beats a base-rate constant on one enzyme of four (3A4, AUC 0.750).

*The affine pair.* A per-fold shrink and shift fitted to the metric, applied before writing
predictions. The warning attached to it is severe: **"Five separately measured improvements in our
own log collapse into a band of 0.013 once the pair runs — none clears the 0.007 noise floor and
three of the five change sign to harmful."** A concatenated Chemprop embedding improved raw macro
ST-RAE by 0.0084 and **lost 0.0113 after the pair**. "A raw improvement in this challenge is not
evidence of anything until it has been through the post-processing that ships." Mechanism: the pair
is a shrinkage fitted to the metric, so anything that merely reduces prediction variance or removes
an offset is something the pair already does — **it substitutes for such gains rather than adding to
them.** Corollary: they optimised Spearman/Kendall on purpose, being the columns an affine transform
cannot touch. Also measured: **the ST-RAE-optimal spread is 0.69–0.88 of the R²-optimal, on every
enzyme and every feature set.**

**§9 — the two results we can use immediately with no model at all.**

1. **The credible band is a deterministic function of the label.** Isotonic regression from the
   label alone onto band width: 1A2 ρ=−0.885 R²=0.963 · 2C9 −0.899 / 0.928 · 2D6 −0.558 / 0.955 ·
   3A4 −0.928 / 0.970; and the label sits at a near-fixed relative position in its own band
   (sd of `(y−lo)/w` under 0.09 everywhere). So ST-RAE is a **potency-weighted absolute error**, with
   the forgiveness threshold ≈ **1.25 pIC50 below a label of 3.5 and ≈ 0.13 above 4.6** — a fiftyfold
   difference. *And the obvious inference from that is wrong:* the penalty is **U-shaped**, not
   monotone. Share of the ST-RAE numerator by potency quartile (weakest → potent): mean **27.6 /
   12.5 / 14.4 / 45.5 %**. The weak end is not free because the median absolute residual there runs
   1.27–1.67 pIC50, which overruns even a half-log threshold. "Effort belongs at **both** ends and
   the middle can be left alone."
2. **Blind band geometry is readable off the public board.** Board MAE is plain
   `mean_absolute_error`, and ST-RAE's denominator is a hinge on the constant `mean(y_true)` and so
   **contains no `y_pred`** — one constant `C` shared by the whole field. Therefore `MAE / ST-RAE` is
   a pure measure of how much of an entrant's raw error the bands absorb, comparable across every
   submission. With `hinge_i ≤ |e_i|` and `|e_i| − hinge_i ≤ W_i`:

   ```
   C/n  ≤  min over entrants of  MAE/ST-RAE
   W̄    ≥  max over entrants of  (MAE − (C/n)·ST-RAE)
   ```

   Nothing is fitted. Their own absorption: 1A2 1.3425 / C/n≤0.7822 / W̄≥0.4132 / ≥41.7% ·
   2C9 1.0151 / 0.5081 / 0.3380 / ≥49.9% · **2D6 1.5622 / 0.5641 / 0.9300 / ≥63.9%** ·
   3A4 1.1112 / 0.7491 / 0.2100 / ≥32.6%. The tell that caught their error: **"Sixty-nine entrants
   were posting a strictly worse CYP2D6 MAE than ours and a strictly better ST-RAE"** — arithmetically
   impossible without band absorption.

**(f)** Butina-clustered 5-fold, four seeds, with explicit noise floors per enzyme (e.g. 3A4 MCC
floor 0.0281, macro rank floor 0.0036) and 301 numbered items in `verify/README.md`.

**(g)** OOF macro ST-RAE **0.6459**, macro Spearman 0.6342; from a conventional start (same
features, per-enzyme boosting, no ensemble) at 0.7150/0.5651, i.e. +0.0579 rank / −0.0583 ST-RAE.
TDI OOF MCC 3A4 0.3510 / 2D6 0.1235 / macro 0.2373.

**(h)** MIT.

---

## 69 · 450nm — 0.6004 · https://openadmet-cyp-dashboard.vercel.app/operating_dashboard

JS-rendered; the payload is a 3.7 MB embedded JSON with **270 findings and 376 numbered
experiments** (verdicts: deployed / rejected / refuted-my-claim / inconclusive / blocked). Code for
the structure track at github.com/xX-its-amit-Xx/openadmet-cyp-structure. The dashboard snapshot
predates their current 0.6004 and records a board at 0.9549 → 0.8998; the experiments below are the
reason for the move.

**(a) The most systematic treatment of the sparse-cell problem on the board.**
- **E004 (rejected in part) — screen as a learned auxiliary feature.** Deploys on 3A4 (−0.0129, LCO
  −0.0365) and 2C9 (−0.0220); rejects on 2D6 (LCO +0.0193, actively hurts) and 1A2 (+0.0091).
  **"Payoff is monotonic in how well the aux predicts its OWN target: ρ −0.761/−0.645/−0.505/−0.376
  maps to LCO −0.0365/−0.0277/+0.0091/+0.0193, no inversions."** The pre-flight test they distil:
  fit aux-target ~ features and only proceed above ρ ≈ 0.6. Mechanism: "an auxiliary trained on the
  SAME features as the base can only redistribute information, never add it."
- **E014 (deployed, 3A4 only) — sparse-matrix row imputation from the screen.** Sign **flips** by
  isoform: 3A4 0.5307 → **0.4773** (−0.0534, the largest single gain of their campaign); 2C9 −0.0119
  but LCO regresses; **2D6 +0.110; 1A2 +0.5316, catastrophic.** "Harm tracks systematic **BIAS**
  (imputed minus real mean): 2C9 −0.30, 3A4 +0.41, 2D6 −0.74, **1A2 −1.56** — not the noise ratio."
  Their own stated reason: "donors are plausibly not exchangeable because the organisers chose which
  compounds got full dose-response curves, so **imputation transfers a population shift**."
- **E015** — imputation and the auxiliary **substitute** rather than stack (both encode the same
  screen, one as predicted features and one as derived labels); stacking cost 0.013 for complexity.
- **E026/E027/E038 — improve the log2fc → pIC50 calibrator and the imputation flips from reject to
  win.** Adding **Emax** as a calibration input lifts calibration R² on every isoform (2D6 0.454 →
  0.604, 2C9 0.646 → 0.763, 1A2 0.784 → 0.837, 3A4 0.837 → 0.865; Emax alone calibrates poorly, so
  it is complementary) and Emax spans 100% of the pIC50 train set against the screen's 89%. 2D6
  0.9304 → **0.8930**; **2C9 0.7275 → 0.6033 (−0.1242, their largest single-isoform gain)**; 1A2
  still rejects; 3A4 rejects (it did not need it, and its donors are *more* active than its measured
  compounds). Adding **`log2fc_std_error`** (and `cohens_d`) to the calibrator improves it on all
  four with the gain **perfectly inversely ordered against baseline R²** — 2D6 +0.0342 > 2C9 +0.0106
  > 1A2 +0.0066 > 3A4 +0.0043 — "the signature of a real signal being added rather than variance
  being fitted."
- **E034 (inconclusive, but the clearest statement of the principle) — the public panel's inactives
  were the information they were dropping.** Censoring improves **both** metrics on **all four**
  isoforms, no inversions (transfer-ρ 3A4 +0.356 → +0.592, 1A2 +0.344 → +0.414, 2C9 +0.174 → +0.411,
  2D6 +0.230 → +0.274; external rows roughly double, 3A4 8,870 → 15,220), and the mean shift
  external-minus-official **shrinks** on every isoform (3A4 +0.93 → +0.63). "One `dropna` was
  throwing away **38,877 measurements of non-binding**. A row with no fitted curve is not missing; it
  is the assay saying 'nothing happened', and it carries the boundary the model most needs ... an
  actives-only model is biased potent, and censoring removes most of that bias rather than merely
  adding rows."
- **E161** — the screen as an *auxiliary task* for 2D6 rescues a bad MLP (1.1057 → 0.9724, 25.6× the
  seed SD) but the absolute level still trails the deployed LightGBM stack (0.8738), so it does not
  ship. "An effect is a property of a configuration."

**(b) Their board-verified calibration arc, which contains a sign flip we should not repeat.**
- **E023/E024/E025 (rejected on OOF)** — variance matching looked expensive: 2D6 k=1.00 → 0.9072,
  k=1.25 → 0.9362, k=1.50 → 0.9845, **k=2.57 full match → 1.3465 (+48.4%)**, with band-hit rate
  *falling* 18.5% → 14.2% as spread grew. Replicated on 3A4 (k=1.32, cost +13.3%). They concluded
  "compression is not a pathology but the correct expression of genuine epistemic uncertainty" and
  "never variance-match."
- **Then the board.** Their own status line: *"Per-isoform ST-RAE is rank-ordered exactly by how
  compressed each prediction column was — **CYP2D6 sd 0.234 against a training-label 0.916 scored
  1.794, worse than a constant**. Three deployed shrinkage layers that won under scaffold-CV are
  fatal on a test set of actives with tight bands."*
- **E192 (deployed) — variance recalibration works.** Macro **0.9549 → 0.8998** (rank 118 → 99).
  1A2 0.9268 → 0.8231 (+28 ranks; it was compressed 0.45×); **2D6 1.7940 → 1.6684** after a **3.92×
  expansion**; 2C9 unchanged by design; **3A4 0.5900 → 0.5989, four ranks worse** — it was only
  mildly compressed (0.75×) and full expansion overshot. The lesson they draw: "Scale was the cheap
  half of the problem; ranking is the expensive half."
- **E232** — the compression instrument's rank ordering of the four columns (1.149 / 0.749 / 0.452 /
  0.255) is **exactly** the ordering of their board scores (0.5087 / 0.5900 / 0.9268 / 1.7940); and
  on 2D6 "the plain model is 0.1082 **worse** on scaffold cross-validation and 0.3393 **better** on
  the board" — a sign flip, not an underestimate. Instrument now exhausted: no column remains
  compressed.
- **E264/E268 — the ST-RAE-optimal constant offset is set by band-edge asymmetry, not the mean
  residual, and can point the other way.** On 3A4 (n=2,335, 6 seeds) 35.8% of predictions fall below
  `conf_low` against 27.3% above `conf_high` (asymmetry +0.084) while `mean(label − prediction)` is
  −0.1014, i.e. the model over-predicts on average — yet the ST-RAE-optimal constant shift is
  **+0.065, positive, opposing the mean residual.** Bands are curve-fit likelihood intervals
  asymmetric about the label on 64–78% of compounds. Across five rows the band-edge rule called the
  sign correctly 5/5 and the mean residual 2/5.
- **E272 — a constant fitted out-of-fold is fitted on the wrong object.** Submitted sd over OOF sd
  is 1.12 on 3A4 and 2.13 on 2C9; at the shipping spread the 2C9 constant **reverses** from +0.0535
  to −0.0453 and 3A4's benefit collapses. Across three spreads of the same 3A4 predictions the
  violation asymmetry runs +0.0864, +0.0369, −0.1016 and the optimal shift follows from +0.060 to
  −0.100 while the mean residual stays negative. **Refit any constant at the spread you will ship.**
- **E261/E263 — monotone recalibration beyond affine is harmful, and the objective is the reason.**
  Isotonic and bin-spline both lose to their own affine control on 1A2 and 3A4, 0/6 seeds. But
  holding the form at two parameters and changing **only the fitting objective** from L2 to ST-RAE
  moves +0.03885/+0.02362 to +0.00187/−0.00397: **ST-RAE wants slope ≈ 1.10/1.00 where L2 wants
  1.484/1.183.** Fit the transform on ST-RAE, not on squared error.

**(c)** AID 1851 and the NCGC panel. Two traps measured: **E169** — "the panel's 28,230 CYP3A4
measurements are 14,155 compounds counted twice, and **one of the two assays is for ACTIVATORS**"
(AIDs 884/891 screen the identical 14,155 CIDs with the same `Potency` field, identical to 4 d.p. on
90.8% of paired rows); the visible symptom was that pIC50 "does not separate the outcome classes at
all (Active 5.30 / Inactive 5.18 / Inconclusive 5.03)", which reads as noise when it is two assays
stacked. **E293** — their own PubChem puller "silently dropped every CYP1A2 potency value."
**E171** — what predicts a donor set's value is **label-distribution match (ρ +0.91)**, not chemotype
coverage (+0.03).

**(d)** LightGBM on Morgan+RDKit plus a mechanistic block and screen-derived features; per-isoform
configurations chosen independently ("four isoforms, four DIFFERENT winners"), and interactions
differ in **sign** between isoforms. **E009 — band weighting** (measurement precision in the loss)
deploys on 3 of 4: 2D6 1.0005 → 0.9626, 1A2 0.8695 → 0.8345, 3A4 0.5541 → 0.5307; 2C9 rejects; a
control gives only a third of the gain.

**(f)** Scaffold CV with leave-cluster-out (LCO) as a second gate, seed-stability requirements and
pre-registered magnitude bars. **E196 refutes their own doctrine**: scaffold CV and random CV are
the same instrument on this data (ARI +0.0002 against a random-vs-random null of +0.0004), and the
**blinded test sits +0.12 to +0.13 Tanimoto *closer* to train than any fold** (0.5979 vs 0.5701 ±
0.0069 at matched pool size). **E110** — an independent probe confirms the blind set is enriched for
actives: the single-concentration model predicts test at **−2.248 log2fc against train −1.710**
(lower = more active), and band width correlates with potency at −0.891 / −0.846 / −0.733 / −0.868
on 3A4 / 2C9 / 2D6 / 1A2. **E216** is the methodological warning: "207 of our 209 measured claims
have never met the blind board," and both claims that did "came back with bad news."

**(g)** Per-isoform board cells at the snapshot: 2C9 0.5087, 3A4 0.5900 (→0.5989), 1A2 0.9268
(→0.8231), 2D6 1.7940 (→1.6684).

---

## 79 · CompChem Magicano — 0.6404 · https://github.com/mokhalaf22/openadmet-cyp-public

`REPORT.md` plus a 2,718-line `FINDINGS.md` (52 numbered sections). Board trajectory **0.9356 →
0.7114 → 0.6683 → 0.6411 → 0.6404** (rank 200 → 95 → 83 → 77), and they are explicit that
**"ranking is at the representational ceiling; the gains are calibration."**

**(a)** Three measured uses of the screen, with opposite signs.
- **As four OOF-*predicted* features: the only auxiliary lever that worked.** Raw log2FC is
  test-unavailable, but a model that predicts it from structure is test-available. On LightGBM macro
  OOF ST-RAE 0.451 → 0.436; **on the D-MPNN 0.434 → 0.415, "the largest single model gain in the
  entry."** Their general rule: a surrogate helps only when it is *strongly learnable from structure*
  (OOF Pearson 0.59–0.74 here) **and** carries *different-assay* information.
- **As point pseudo-labels filling sparse cells: it cost ST-RAE +0.0442** while Spearman moved only
  −0.0141, even though the `log2FC + structure → pIC50` mappings were of "genuinely good quality"
  (held-out Pearson **0.901 / 0.860 / 0.756 / 0.933**) and supplied **11,505 proxy cells against
  6,525 real ones.** The diagnosis is the reusable part: proxy cells are necessarily *point* targets,
  so even at weight 0.3 they "pull the model back toward point regression and destroy the hinge's
  zero-error zone inside the reported interval, which *is* the metric." Rule: **"more supervision at
  the wrong fidelity is worse than less supervision at the right one."**
- **Sampling the target from inside `[lo, hi]`** (one draw per epoch) instead of using the hinge:
  ST-RAE +0.0065. Same failure, milder.

A factual correction worth carrying: **"there are no screen-only molecules."** All 4,376 unique
single-concentration compounds are already in the dose-response table by InChIKey. The screen is "a
second readout on the same compounds, not an additional pool of compounds" — which is why it pays as
a *feature* and not as extra *rows*. (This is compound-level; the missing labels are per-isoform
**cells**, which is what Lizard Wizard's 11,509 and ADMET Bo$$'s ~2,900-per-isoform figures count,
and what our own promotion analysis is about. The two statements are consistent.)

**(b)** Two separate post-hoc moves, both board-validated.
- **A −0.5 location correction on CYP2D6.** "The organizers confirmed the CYP2D6 test compounds are
  less potent than training (the test excluded CYP2D6 hit-expansion)." Validated first on a
  shifted-eval simulation, then on the board.
- **Dispersion expansion, which OOF structurally could not judge.** Expanding toward the training
  spread at 0.85× cut blind macro ST-RAE **0.7114 → 0.6683** (rank 95 → 83); a second expansion to
  full training spread on the three isoforms with no known shift (2D6 held gentler) cut it to
  **0.6411** (rank 83 → 77; R² 0.2495 → 0.3477; **Spearman unchanged at 0.6965**). "Compression was
  costing us, not encoding honest uncertainty."
- **The two populations, stated as a number.** OOF in-distribution `sd(true)` is 0.78–1.09 per
  isoform; **the blind population's, backed out of their own score, is 1.297 — 36% wider than their
  label sd of 0.956, "because a label set built only from successful curve fits silently excludes
  non-inhibitors."** This is why OOF said expanding spread hurt every isoform while the board said it
  gained 0.043 — **"and the board was right."**
- They also inverted their board score with `R² = 2ρk − k² − b²` and got k = 0.7108 against
  ρ = 0.6965 (on the R² optimum), attributing **100% of the remaining R² headroom to location**
  (b = +0.48 log units). Acting on it moved blind R² 0.3477 → 0.4272 and ST-RAE 0.6411 → 0.6404:
  "the algebra was right and the scored metric did not care." Swept on OOF with location refit at
  every k, the **ST-RAE-optimal spread sits below the R²-optimal k = ρ — about 0.8ρ** (1A2 0.42,
  2C9 0.56, 2D6 0.33, 3A4 0.63 against ρ of 0.53/0.67/0.44/0.78) — but the curve is **flat at its
  minimum** (their k = 0.497 against an optimum of 0.48 is 0.0002 of ST-RAE) and k*/ρ is **not a
  transportable constant** (0.50–0.65 within strata vs 0.80 on the full population), because k* is a
  property of the evaluation population's composition.

**(c)** AID 1851 as auxiliary heads **degraded** the model and was dropped. ChEMBL/PubChem used only
to retrieve structural near-neighbours for an unsupervised physicochemical-only warm-start
(admission floor set so admitted compounds are closer to the blind set than their own training
self-similarity, ≈0.50) — yielded **324 neighbours at 13.3% anchor density** and the warm-start was
null. Their explanation for the whole class of failures: the blinded set is **Enamine
make-on-demand chemistry that public bioactivity databases do not cover.**

**The leakage finding, which is a standing rule.** A pre-ingest structure-level check of
`openadmet/Octant_CYP_inhibition_reactivity_blog_release` (CC-BY-4.0, same lab as the challenge)
read **only** SMILES and identifiers and found **5 of the 750 blinded test compounds present by
exact structure and OCNT identifier**, plus one near-duplicate at Tanimoto 0.952, and that the
release is **99.3% the challenge's own CYP3A4 campaign** (1,076/1,084) rather than independent data.
All six quarantined, disclosed to the organisers, nothing built from it. Octant's `CYP3A4_pIC50` is
also a *different assay condition* (active-enzyme pre-incubation = reversible + time-dependent
combined). **Rule: when a challenge is run by a lab that also publishes datasets, check its public
releases for the blinded test compounds by canonical structure/InChIKey before using anything.**

**(d)** ECFP4 counts + RDKit 2D with degenerate columns dropped, plus a from-scratch chemprop
D-MPNN. The signal is **essentially linear** on the feature set (ridge ≈ LightGBM, macro OOF ST-RAE
0.489 vs 0.451; a tuned MLP no better), so "the lever is never a bigger model — it is the
representation and the loss." A CheMeleon encoder, frozen (→ TabICL) or fine-tuned end-to-end, did
**not** beat the from-scratch D-MPNN. Final model: D-MPNN + predicted-primary, multi-task over the
four isoforms, interval targets, 3-seed ensemble.

**(e) The interval hinge, and the proof that uniform weighting is correct.** Loss = L1 distance to
the nearest credible-interval bound, zero inside — "matching the competition's soft-threshold metric
exactly, so training and scoring agree." Point → interval targets improved macro OOF 0.445 → 0.434
and **recovered the CYP3A4 gap (0.323 → 0.305)**, the isoform with 23.6% of rows wider than 2 log
units, whose gap had survived every capacity lever. Then they swept reweighting by `1/(1+width)^p`
in both directions:

| p | −1 | −0.5 | **0 (uniform)** | +1 | +2 |
|---|---|---|---|---|---|
| macro OOF ST-RAE | 0.4311 | 0.4177 | **0.4131** | 0.4197 | 0.4391 |

A V with its minimum exactly at uniform, and the dial demonstrably engaged (Q4's inside-interval
rate moved 20.7% → 47.3% across the range). The reason is structural: ST-RAE's numerator is a plain
sum over a prediction-independent denominator, so **`∂ST-RAE/∂pᵢ = ±1/den` for every excursing row
regardless of interval width.** "Where the error concentrates is not a weighting signal" — 85% of
their ST-RAE landing in the three narrowest quartiles describes where error *lands*, not how the
metric *weights*.

**(f)** Scaffold-split OOF; blind = the live half-set board. **OOF underestimated blind error ~2×**
(OOF 0.451 vs first blind 0.9356) with ranking preserved (ρ 0.6336) and magnitude collapsed
(R² ≈ 0). Their **live-board discipline**: OOF is a weak prior; act only on board moves above the
split standard error (~0.03); require OOF-and-board agreement for the final pick; treat the unscored
half as a permanent holdout.

**(g)** Board: MA-ST-RAE 0.6404, MAE 0.8394, R² 0.3477→0.4272, Spearman 0.6965. TDI MA-MCC 0.273 →
0.3097 after a threshold fix (2D6 0.30, 3A4 0.45); tightening 3A4 to 0.65 then *lost* MCC.

**(h)** MIT. Data not committed; pinned revision `3ac9c5d…`.

**Their "do not bother" list, which overlaps our backlog:** distant external assays as auxiliary
heads; foundation-model fine-tuning over a from-scratch D-MPNN at this label count; same-assay
surrogate features (predicted Emax — weakly learnable, no new information); cross-model blending once
the primary feature has converged the models; explicit pairwise-ranking losses; difference learning
without a dense neighbour corpus; and **ensembling in the regime where only one of your models is
good** — "averaging pays only when members are *both* mutually diverse *and* comparably strong," with
the gain on 2D6 correlating **+0.892 with member strength and −0.736 with member decorrelation**, and
the only member that helped being the *most* correlated one (ρ = 0.920) because it was the strongest.
Conversely, **sharing across isoforms earns its place and matters most where data is thinnest**:
splitting into single-task models costs −0.0189 macro Spearman, by isoform 2D6 −0.038 (n=1,493),
2C9 −0.019, 1A2 −0.014, 3A4 −0.004 (noise, n=2,335).

---

## 87 · adlvdl — 0.6578 · https://github.com/adlvdl/cyp_competition

Only the baseline notebook's numbers are published; the later runs are code without a write-up.
Useful for its infrastructure and its explicit statement of the mechanism.

**(a)** "**A compound only has a dose-response curve where the primary screen flagged it**, so every
target column is mostly NaN," and "the single-concentration screen covers far more compounds than any
dose-response table and is the main source of auxiliary signal for compounds that never got a curve."
The modelling that follows is not written up.

**(b)** `calibration.py` — "leakage-free cross-fit calibration." Baseline effect, mean ST-RAE over
25 folds: 3A4 0.720 → **0.707**; 2C9 0.961 → **0.883**; 1A2 0.986 → **0.903**; 2D6 1.063 → **0.947**.
"Only CYP3A4 clears the mean predictor by a wide margin before calibration."

**(d)** LightGBM/XGB on ECFP4 (scikit-fingerprints dispatch), with `models.py`, `ensemble.py`,
`evaluation.py` (paired bootstrap, Holm–Bonferroni), `mcs.py` (repeated-measures Tukey HSD).

**(f)** **Nested 5×5 scaffold CV (25 folds)**, plus a decision discipline worth copying: the gating
statistic is the **macro-average** across the four endpoints, not per-endpoint panels. On their run
`lgbm` and `xgb` are **not significantly different** on macro ST-RAE (p=0.239) despite both clearly
beating mean/ridge/knn — "this is the number that should actually decide a submission." LightGBM
beats the mean baseline at p<0.01 on 1A2/2C9/3A4 but only p<0.05 on 2D6, "one more sign CYP2D6 needs
a different approach rather than more tuning of the same models." They also keep **dated data
snapshots** because "OpenADMET has amended the dataset mid-challenge," with a `make data-diff`.

**(h)** MIT; code ported from their PXR-challenge repo.

---

## 122 · leeherman — 0.7585 · Google Doc

A tried/not-tried ledger rather than a report. Extractable: regression is a "Chemprop v2 hybrid
architecture ... multi-modal feature fusion" with CheMeleon fingerprints augmenting the graph; CV by
Murcko–Bemis scaffolds worked, molecular-weight-based CV did not; multi-target models beat
single-target; rows weighted by occurrence count per CYP; and — the relevant line — **"Apply affine
transformation to 2D6 predictions (as per Brian Wylie)"**, i.e. briford's placement correction, was
implemented. A "hybrid quantile and direct margin ranking loss with affine transformation" was tried
and marked unsuccessful. ORGANIC atom featurisation unsuccessful. No external data, loss detail,
per-isoform scores or licence given.

---

## 127 · popcorn2 — 0.7671 · https://github.com/etiu/cyp_challenge

A single notebook, `activity_prediction_v3_popcorn.ipynb`, titled **"CYP Position-8 Pipeline"** — an
explicit, faithful reimplementation of briford's four-member recipe: the same members
(`mt-aux-100` 8 heads, `union-p30` 18 heads incl. CYP2C19, `2d6-isoform`, `2d6-single`), the same
`task_weights [1,1,1,1, 0.3,0.3,0.3,0.3]`, ChEMBL 37 + AID 1851 `max_response` (AIDs 410/884/883/891),
InChIKey-connectivity standardisation, Chemprop `--from-foundation CHEMELEON`,
`--split SCAFFOLD_BALANCED`, plain prediction averaging.

**What it does not have is the calibration.** The notebook prints "Prediction stats (raw, before
placement)" and sets up an integrity check ("Spearman should be 1.0 before vs after placement"), but
there is no calibration code in it: no `rho`, no blind moments, no affine fit. So the pair

> **briford 0.4378 (model + affine placement) vs popcorn2 0.7671 (approximately the same model, no
> placement)**

is close to a natural experiment on the value of the placement step, worth ≈ **0.33 macro ST-RAE**.
It is not a controlled comparison — different training runs, and popcorn2 adds a CheMeleon foundation
init and a fifth ChEMBL isoform — but it is consistent with jeremy's within-pipeline 0.7247 → 0.5214
and with 450nm's 0.9549 → 0.8998.

No licence file.

---

## 132 · ADMET Bo$$ — 0.7726 · https://github.com/Rebell-Leader/OpenADMET_CYP/blob/main/paper/PAPER_DRAFT.md

**(a) The clearest public statement of the censoring point, plus a cautionary leak.**
> "Roughly **2,900 compounds per isoform carry a screen reading but no fitted dose–response curve**;
> these are precisely the compounds the target labels do not cover. Adding them as pseudo-labelled
> training rows helps (**−0.008**), but **only after clipping at the assay floor**: compounds without
> a fitted curve are **censored at pIC50 4, not point-valued near 2.9**, and linear extrapolation
> below the floor was catastrophic (**ST-RAE 1.42 and 2.11 on two isoforms**)."

The median pseudo-label shift predicts the damage at **rank correlation −1.00 across isoforms**,
giving an a-priori isoform-selection rule that needs no CV scores and reproduces the CV-optimal
subset. Auxiliary log2FC heads add **0.015** on top of multitask sharing.

**The 0.557 leak.** Fitting the log2FC donor models on *all* rows — so each donor had seen the
held-out compound's own measured screen value — produced ST-RAE **0.2771** with R² **0.834** on
CYP1A2. The identical experiment with donors refitted **inside each outer fold** scored **0.8342**,
slightly worse than not doing it at all. "Any pipeline that builds surrogate features on a compound
set shared with the target labels has this failure mode, and its magnitude here — 0.557, enough to
fabricate a decisive leaderboard win — argues for nesting as a default rather than a robustness
check." Also: pseudo-labels (−0.008) and cross-isoform stacking (−0.008) together give only −0.003,
because both inject estimates of the same underlying quantity. **"Gains from the same information
compete; gains from different function classes compose."**

**(b)/(§3.4) CYP2D6 is a different problem, and they separated two hypotheses.** Assay noise is
**refuted** — 2D6 has the **narrowest** credible intervals of the four. Orthogonal SAR is supported
on three lines: near-zero mean correlation with the other three isoforms on co-measured compounds;
inverted physicochemical drivers, where **basic-nitrogen count is the only simple descriptor
positively correlated with 2D6 while being negative for all others** and lipophilicity is inert
where it dominates elsewhere; and a monotone potency response to basic-nitrogen count. Two
consequences: **the blinded test set is *depleted* in basic nitrogen relative to training (11.6% vs
18.9%)**, and hand-built pharmacophore descriptors score worse than the mean, so the signal is
"real, weak, and already captured by the full feature set."

**(c)** **Public data augmentation hurt: +0.057 macro, three of four isoforms** (1A2 the exception).
"The best public source is measured in the pre-incubation arm rather than the direct arm, and a
model trained only on public data reaches R² 0.117 against real assay values — near no-skill."
Screen distillation hurt (+0.036, all four).

**(d)** ECFP4 (2,048) + 217 RDKit 2D; for the in-context and neural models the fingerprint block is
**compressed to 256 dims by truncated SVD** and standardised (473 dims total) — load-bearing, not
cosmetic: full width scores 0.7611 against 0.7239 compressed. Models: TabICL in-context + a
multitask MLP, blended with weights from nested inner CV. They also quantified their transductive
SVD fit (basis fitted on train+test structures stacked): refitting on training rows only **improves**
macro from 0.7818 to 0.7740, so the transductive fit conferred no advantage.

**(e)** Soft-threshold-aware framing only; the gains are sharing (0.016), aux log2FC heads (0.015)
and blending (0.031). Blend beats both parents on every isoform despite error correlations of
**0.888–0.938** — "the gain is therefore not decorrelation but variance reduction from averaging two
function classes with a tuned weight." Ensembling saturates at 8 estimators.

**(f)** 5-fold grouped by Bemis–Murcko. **Random splitting is optimistic by 0.039–0.051 RAE** vs
scaffold grouping, consistently across all four isoforms; and **local scaffold-grouped CV is +0.113
to +0.127 optimistic relative to the leaderboard** (measured against two scored submissions) — they
use **+0.12 as a conversion constant throughout.** An analog-series-aware protocol mimicking the test
construction could not be populated: the training set has too few analog series.

**(g)** Blend OOF ST-RAE: 1A2 0.7939, 2C9 0.5904, **2D6 0.9012**, 3A4 0.4617, macro 0.6868.

**(h)** NOASSERTION; `00_docs/METHODS.md`, `metric_spec_reverse_engineered.md`, an experiment ledger
and a submission log are all public.

---

# Synthesis

## What the best public methods share

1. **Every entry above ~0.55 applies a per-isoform affine placement, and every entry that measured
   it found it was the largest single effect.** jeremy 0.7247 → 0.5214 same models; Magicano 0.9356 →
   0.7114 → 0.6683 → 0.6411 with rank 200 → 77 and Spearman *unchanged*; 450nm 0.9549 → 0.8998;
   datagrok macro OOF 0.803 → 0.764; adlvdl 2D6 1.063 → 0.947; briford and leeherman both ship it.
   popcorn2, which reimplements briford's model and omits the placement, scores 0.7671 against
   briford's 0.4378. **The algebra is agreed** (`R² = 2ρk − k² − b²`, optimum `k = ρ`, Murphy 1988),
   **and so is the caveat**: the ST-RAE optimum is **not** the R² optimum. Three independent
   measurements of the gap: Lizard Wizard "ST-RAE-optimal spread is 0.69–0.88 of the R²-optimal, on
   every enzyme and every feature set"; Magicano "about 0.8ρ in-distribution"; JacksonBurns "R² is
   maximised at f ≈ 1.2–1.35, ST-RAE at f ≈ 0.9–0.95." And on 2D6 specifically, briford: "the two
   objectives point in opposite directions."
2. **The selection mechanism is understood, named and quantified by at least five teams, and all
   of them treat the unlabelled cells as *bounds* or as *a second observation*, never as points.**
   briford: "Compounds that did nothing produce no curve and no label"; "the qHTS panel is the only
   source that says anything at all about the compounds that did nothing." Lizard Wizard: a Heckman
   correction with a measured first stage; selected-only fit recovers coefficients at slope 0.731,
   joint fit at 1.012. ADMET Bo$$: "censored at pIC50 4, not point-valued near 2.9," with linear
   extrapolation below the floor costing ST-RAE 1.42 and 2.11. datagrok: `BoundedMSELoss` + `lt_mask`
   with AID 1851 inactives at a bound of 4.244. 450nm: "one `dropna` was throwing away 38,877
   measurements of non-binding ... an actives-only model is biased potent."
3. **Auxiliary data enters as separate heads per source × isoform, never pooled into the scored
   column.** Unanimous across briford, datagrok, jeremy, Magicano and JacksonBurns. datagrok gives
   the sharpest reason: pooling "dragged the scored target down to the qHTS spread (~0.6) when the
   challenge assay's own spread is ~1.0."
4. **The single-concentration screen pays as a *predicted feature* or an *auxiliary head*, and loses
   as a *point pseudo-label* for the scored column.** jeremy: pseudo-labelling "hurt badly," log2fc
   heads fine. Magicano: predicted-log2FC features were the largest model gain (0.434 → 0.415) while
   11,505 accurate proxy *cells* cost +0.0442. 450nm: the auxiliary's payoff is monotone in how
   learnable the aux target is, and imputation's harm tracks the imputed-minus-real **mean bias**,
   not the noise. The only team that made raw imputation pay (450nm on 3A4, and 2C9 after an
   Emax-improved calibrator) did it by **de-biasing the imputed labels first**.
5. **Grouped CV, nested donors, and an explicit OOF→board conversion.** Butina or Murcko folds
   everywhere; ADMET Bo$$ quantifies the whole stack (random is +0.039–0.051 optimistic vs scaffold;
   scaffold is +0.113–0.127 optimistic vs the board, conversion constant +0.12); Magicano measured
   OOF underestimating blind error ~2×; stir_bar found 100% of regression submission-endpoint pairs
   got worse on the blind set, median MAE +0.208. And ADMET Bo$$'s 0.557 leak is the reason donor
   models must be refit inside the outer fold.
6. **A 2D6 specialist.** briford dedicates two of four members to 2D6 and averages all four there
   (OOF Spearman 0.503 vs 0.445 best single); JacksonBurns' replication of that specialist was the
   best 2D6 single in a 19-member pool at blend weight 0.30–0.35; jeremy's qHTS-only domain adaptation
   was his "best-ever single-model CYP2D6 score." Magicano shows why sharing helps most there:
   splitting costs −0.038 ρ on 2D6 against −0.004 on 3A4.
7. **Representation diversity with a constrained combiner.** stir_bar's non-negative stacking,
   jeremy's Caruana selection over NNLS, JacksonBurns' family-block LOFO audits — all three
   independently found that unconstrained continuous optimisers over correlated members fit folds and
   transfer badly.

## What separates rank ~12–50 from rank ~170

Three things, and only the first is cheap.

- **Placement (worth ~0.15–0.33 of macro ST-RAE).** This is the whole difference between popcorn2 at
  0.7671 and briford at 0.4378, and between jeremy's two submissions at 0.7247 and 0.5214. Our
  2D6 = 1.46 — *worse than predicting the mean* — is the signature of exactly this defect, and 59.4%
  of the field shares it. Nothing about our ranking needs to change to fix it.
- **External breadth, as separate heads (worth ~0.05–0.15).** JacksonBurns' board scrape: "top ~20
  ALL have external data ... or non-fp deep features." But it is not free: Magicano and ADMET Bo$$
  both measured public augmentation *hurting*, and Lizard Wizard measured ChEMBL's raw gain inverting
  sign after the affine pair. The teams for whom it worked chose sources on **population match**
  (jeremy on AID 1851 over ChEMBL; 450nm's E171: label-distribution match predicts donor value at
  ρ +0.91 against +0.03 for chemotype coverage) and admitted the **inactives**.
- **Ranking (the expensive half, worth ~0.1 and not reachable by post-processing).** JacksonBurns'
  single clearest diagnostic: his blind MA-R² (0.387) equalled his mean squared OOF Pearson (0.349),
  so "our placement is already near R²-optimal. The remaining gap is 100% RANKING quality." Top field
  Spearman 0.75–0.79 / R² 0.62–0.68 against his 0.616/0.387. 450nm reached the same conclusion from
  the other side: after a 3.92× expansion 2D6 was still 1.668, "no affine map can repair ordering."
  What actually moved ranking in public: multi-task D-MPNNs with many auxiliary heads, **frozen**
  pretrained encoders read by an in-context learner (TabPFN/TabICL/Mitra), and representation
  diversity under a non-negative combiner. What did not: fine-tuning on ChEMBL-shaped data, SSL
  pretraining, 3D/quantum descriptors, frozen ChemBERTa, hand-built pharmacophores, bigger models.

---

# Ranked recommendations

Ordered by expected ST-RAE per unit of effort, with the evidence for each.

### 1 — Per-isoform affine placement onto the *blind* population, fitted on ST-RAE, not L2
**Expected: 0.866 → ~0.60–0.65 macro, most of it from 2D6.**
Z-score each isoform's test predictions onto an estimate of the blind population's mean and spread,
with `target_sd = ρ_oof × sd_blind` as the starting point and then a sweep down toward
`0.7–0.9 × ρ_oof × sd_blind` because ST-RAE wants less spread than R² does. Published destination
moments for the live scored half (SuperCowPowers' reverse-engineering, used by JacksonBurns):
**1A2 4.412/1.553 · 2C9 4.830/1.101 · 2D6 3.107/1.599 · 3A4 4.880/1.272**, with the note that 2D6's
*ST-RAE*-optimal destination is **higher and tighter than its R²-optimal one** (JacksonBurns shipped
mean 3.57 / sd 0.90). Floor predictions at 1.0.
*Evidence:* jeremy 0.7247 → 0.5214 and 2D6 R² −0.749 → +0.377 on identical models; Magicano
0.7114 → 0.6683 → 0.6411 with Spearman fixed at 0.6965; 450nm 0.9549 → 0.8998; briford/leeherman ship
it; popcorn2 vs briford ≈ 0.33.
*Guards:* (i) verify Spearman and Kendall are bit-identical before and after — briford's integrity
check; (ii) **fit the transform's objective on ST-RAE, not squared error** — 450nm's E263 found L2
wants slope 1.48 where ST-RAE wants 1.10, and every monotone map fitted on L2 was harmful; (iii)
**refit any constant at the spread you will actually ship**, because OOF predictions are shrunk
relative to a full-fit submission and the optimal shift is sign-sensitive to spread (450nm E272,
2C9's constant reversed from +0.0535 to −0.0453); (iv) expect over-expansion to *hurt* a column that
was only mildly compressed (450nm's 3A4 went four ranks worse).

### 2 — Train on the non-promoted compounds as left-censored labels (`y ≤ 4`), not as points
**Expected: a real ranking gain on 1A2/3A4 and the removal of a systematic potency bias everywhere.**
Mechanically: for every (compound, isoform) cell that was screened but produced no curve, add a row
with the bound as the label and an `lt` flag, and use a bounded loss that charges nothing when the
prediction is already below the bound. Chemprop ships this as `nn.BoundedMSELoss()` with
`lt_mask`/`gt_mask` per datapoint; datagrok's `train_cyp_multitask.py` and
`build_multisource_dataset.py` are a working reference, including the `__lt` column convention and
"a fitted measurement always wins over a censored bound for the same compound."
*Evidence:* ADMET Bo$$ — pseudo-labelled rows help (−0.008) **only after clipping at the assay
floor**; point-valuing them near 2.9 or extrapolating linearly below the floor was catastrophic
(ST-RAE 1.42 and 2.11). 450nm E034 — censoring the public panel's inactives improved transfer-ρ and
self-ρ on **all four isoforms with no inversions**, roughly doubled usable rows, **and shrank the
external-minus-official mean shift on every isoform**. briford publishes a censored variant of every
source on the stated principle that "a bound is not a measurement" (though it did not move their own
score). datagrok has an explicit `multihead_censored` vs `multihead_nocensored` ablation.
*Note:* this is the lever that most directly addresses the promotion pattern we found, and our
per-isoform promotion curves tell us where it will bite hardest: a **hard threshold on 2D6** means
the non-promoted 2D6 cells are confidently `≤ 4` and are the most informative censored rows on the
board; **~random promotion on 2C9** means its censored rows carry least.

### 3 — Replace the regression loss with the interval hinge, uniformly weighted
**Expected: ~0.01–0.02 macro OOF, concentrated on the wide-band isoform (3A4), plus agreement
between training and scoring.**
Loss = L1 distance to the nearest bound of the reported `[conf_low, conf_high]`, zero inside; i.e.
L1 against `clip(p, lo, hi)`.
*Evidence:* Magicano — point → interval improved macro OOF 0.445 → 0.434 and **recovered the CYP3A4
gap (0.323 → 0.305)** that had survived every capacity lever. Lizard Wizard — a majorise–minimise
refit against `clip(p, lo, hi)` in all five members, **+0.0197 of rank, sign holding in 16/16
per-enzyme cells**, their largest single effect.
*Do not* reweight by band width: Magicano swept `1/(1+width)^p` over five values in both directions
and got a V with its minimum exactly at uniform, because `∂ST-RAE/∂pᵢ = ±1/den` regardless of width.
*Do not* sample the target from inside the interval (+0.0065).
*Caveat:* JacksonBurns found a dead-zone **refit of an already-strong 19-member blend** neutral
(+0.0002) — the lever is for the base learners' objective, not a post-hoc member.

### 4 — Use the screen as a *predicted* feature and an auxiliary head; never as a point label for a scored cell
**Expected: ~0.015–0.02 macro, and it is the cheapest ranking gain available.**
Two forms, both test-available: (a) train OOF models `structure → log2fc` per isoform and feed the
four predictions as features; (b) add four `log2fc` heads at weight ~0.3 to a multi-task model and
drop them at inference.
*Evidence:* Magicano — predicted-primary features were "the largest single model gain in the entry,"
0.451 → 0.436 on LightGBM and **0.434 → 0.415 on the D-MPNN**. jeremy's 17-head multitask with four
log2fc heads was "the single largest ensemble contributor." ADMET Bo$$ — aux log2FC heads +0.015.
Lizard Wizard's joint trunk — −0.0264 ST-RAE / +0.0350 rank against its own zero-gradient control.
*Pre-flight test (450nm E004):* fit `aux-target ~ features` first and only proceed where it is
learnable — payoff was monotone in that ρ with no inversions (ρ −0.761 → LCO −0.0365; ρ −0.376 → LCO
+0.0193). For us that predicts the feature pays on 3A4/2C9 and may hurt on 2D6/1A2, so **gate it per
isoform.**
*Mandatory guard:* refit the donor models **inside every outer fold**. ADMET Bo$$ measured a **0.557
ST-RAE leak** (0.2771 vs 0.8342) from donors that had seen the held-out compound's own screen value —
"enough to fabricate a decisive leaderboard win."
*If you do impute cells anyway* (450nm's route): de-bias first. Their harm tracked the
imputed-minus-real **mean** (1A2 −1.56 → +0.53 catastrophic; 3A4 +0.41 → −0.0534 win), and improving
the `log2fc → pIC50` calibrator with **Emax** and **`log2fc_std_error`/`cohens_d`** flipped 2C9 from
reject to their largest single-isoform gain (0.7275 → 0.6033). Emax covers 100% of the pIC50 train
set against the screen's 89%. ADMET Bo$$'s a-priori rule needs no CV at all: the **median
pseudo-label shift ranks the damage at −1.00** across isoforms.

### 5 — Set the constant offset from band-edge violation asymmetry, not the mean residual
**Expected: ~0.005 per isoform, free, and it prevents a sign error.**
Compute the OOF fractions below `conf_low` and above `conf_high`; the ST-RAE-optimal constant shift
follows the asymmetry and **can oppose the sign of the mean residual**, because the bands are
likelihood intervals asymmetric about the label on 64–78% of compounds.
*Evidence:* 450nm E264/E268 — on 3A4, 35.8% below vs 27.3% above while `mean(label − pred)` is
−0.1014, yet the optimal shift is **+0.065**; the band-edge rule called the sign correctly on 5/5
rows and the mean residual on 2/5. Their standing check, "compare predicted mean against label mean,"
was measuring the wrong quantity.

### 6 — Read the blind band geometry off the public leaderboard before trusting any placement
**Expected: it tells you whether step 1's spread choice is right, with no model and no submission.**
ST-RAE's denominator contains no `y_pred`, so it is one constant `C` shared by the field, and
`MAE / ST-RAE` measures band absorption comparably across every entry. Then
`C/n ≤ min(MAE/ST-RAE)` and `W̄ ≥ max(MAE − (C/n)·ST-RAE)` over entrants — nothing fitted.
*Evidence:* Lizard Wizard §9. It caught their own two-point estimate concluding the blind 2D6 bands
were negligible when the field spanned 0.5641–2.2625 on that column, and acting on the corrected
reading moved their 2D6 cell **0.7707 → 0.6805 and the entry ten board places.** The diagnostic tell
was that **69 entrants had a strictly worse 2D6 MAE and a strictly better 2D6 ST-RAE** than theirs.
Complement it with the fact that **band width is an isotonic function of the label** (R² 0.928–0.970;
ρ(y,width) −0.56 to −0.93), so you can model a band for any predicted label and compute expected
hinge loss and `P(lo ≤ ŷ ≤ hi)` directly. Forgiveness is ≈1.25 pIC50 below a label of 3.5 and ≈0.13
above 4.6 — but the penalty is **U-shaped**, not monotone: by potency quartile the numerator shares
are **27.6 / 12.5 / 14.4 / 45.5 %**, so the weak end is not free and the middle can be ignored.

### 7 — Add external data as separate heads, chosen on population match, inactives included
**Expected: ~0.05–0.15, but the most likely to backfire.**
Priority by population match to the challenge's diversity-library screen: **PubChem AID 1851 /
Veith** first (NCATS "Biodiverse" collection, ~16.5k substances; fetch the per-SID CSV export for
continuous `Fit_LogAC50`, not the "concise" export which is binary-only; keep `max_response`, which
is at 100% coverage, alongside the fitted pIC50); then **Tox21 CYP P450-Glo** ("the only public
source weak-inhibitor-rich enough to reach below pIC50 5.0"); then **ChEMBL 37**, knowing it reads
~0.5–1.65 log units more potent than this population and that nothing in it sits below pIC50 4.0.
*Traps measured publicly:* AIDs 884/891 double-count the same 14,155 CIDs and **one of the pair is an
activator assay** (450nm E169 — the symptom is pIC50 failing to separate Active/Inactive at all);
Tox21's luminogenic readout needs a **luciferase-artifact filter** (datagrok); the **Octant release is
99.3% the challenge's own compounds and contains 5 of the 750 blinded test compounds** (Magicano §9)
— run an InChIKey-level leakage check against the blinded set before ingesting anything from the
challenge lab, and disclose any hit.
*Counter-evidence to respect:* Magicano found AID 1851 as auxiliary heads **degraded** the model;
ADMET Bo$$ found public augmentation **+0.057 worse** on three of four isoforms; Lizard Wizard found
ChEMBL's raw +0.0128 became **harmful (+0.0068) after the affine pair**. So (i) gate per isoform,
(ii) **evaluate every external-data gain only after the placement step is applied** — Lizard Wizard's
central warning is that five of their own findings collapsed into a 0.013 band once the affine pair
ran, and three changed sign.

### 8 — A dedicated CYP2D6 arm
**Expected: 2D6 is ~0.26 of the macro and has ~1.0 of absolute headroom; this is where the remaining
rank lives after placement.**
Four public levers, in order of measured support: (a) **a 2D6-only specialist member** — briford runs
two and averages four members on 2D6 (OOF ρ 0.503 vs 0.445 best single); JacksonBurns' replication
was the best 2D6 single in a 19-member pool at blend weight 0.30–0.35. (b) **qHTS-only domain
adaptation** of a frozen encoder (no challenge labels, to avoid leaking) — jeremy's "best-ever
single-model CYP2D6 score," and JacksonBurns reproduced a standalone ridge-probe R² of 0.378 on 2D6
with it (but found it **pool-redundant** once he already had four 2D6-capable members, so add it
*instead of*, not *on top of*, an existing 2D6 specialist). (c) **Basic-nitrogen geometry** —
basic-nitrogen count is the only simple descriptor positively correlated with 2D6 and negative for
every other isoform (ADMET Bo$$ §3.4), and Lizard Wizard's ten salt-bridge columns are worth
**+0.1545 of rank on 2D6** above a size-matched random block, while charge-derived descriptors fall
below their own null — "geometry of the basic centre, not the protonation call." Note the blinded set
is **depleted** in basic nitrogen (11.6% vs 18.9%). (d) **Keep 2D6 in a shared multi-task encoder** —
splitting it off costs −0.038 macro Spearman, the largest of the four (Magicano).
*And two facts that constrain the 2D6 placement:* its credible intervals are the **narrowest** of the
four (ADMET Bo$$), with a half-threshold of 0.245 against 3A4's 1.233 (Lizard Wizard), so its weak
compounds are **not** cheap to miss — 29.2% of its ST-RAE numerator sits in the weakest quartile. And
the organisers confirmed the 2D6 test compounds are **less potent than training** because the test
excluded 2D6 hit expansion (Magicano §6); briford's independent prior from the qHTS panel is
**~65% inactive, centring near pIC50 3.7 against the 4.69 a curve-fit-trained model predicts.**

### 9 — Fix the validation and gating protocol before trusting any of the above
- **Grouped folds, and donors nested inside them.** Murcko or Butina; random splits are +0.039–0.051
  optimistic (ADMET Bo$$). But note 450nm measured scaffold CV ≈ random CV on *this* data (ARI
  +0.0002) and the blinded test sitting **+0.12–0.13 Tanimoto closer to train than any fold** — so
  grouped CV here is a guard against analogue leakage, not a simulation of the test's difficulty.
- **Carry an explicit OOF→board conversion and do not re-litigate it.** ADMET Bo$$ uses +0.12;
  Magicano saw OOF understate blind error ~2×; JacksonBurns measured nested→blind k ≈ 1.04–1.06 on
  R² across three scored points; stir_bar found 100% of regression pairs worse on the blind set.
- **Dispersion is structurally OOF-unjudgeable.** Both Magicano and 450nm first concluded from OOF
  that expanding spread hurt, shipped the compression, and were refuted by the board — 450nm's 2D6
  was "0.1082 worse on scaffold CV and 0.3393 better on the board," a **sign flip**. Treat OOF as a
  weak prior on spread and act on board moves above the half-set standard error (~0.03).
- **Gate on the macro, with a noise floor.** adlvdl's Tukey-HSD finding that LGBM and XGB are
  indistinguishable on macro ST-RAE (p=0.239) while both beat the baselines per endpoint is the
  cautionary case; briford's note that seed-to-seed disagreement exceeds candidate-model disagreement
  is the reason to set a floor before measuring; and JacksonBurns' family-block LOFO audits are the
  cheap way to check a gain is not internal to one model family.
- **Evaluate every candidate *after* the post-processing that ships.** Lizard Wizard §3 is the single
  most transferable methodological warning on the board.

### 10 — Deprioritise
Fine-tuning pretrained encoders on challenge labels (jeremy: "freeze, don't fine-tune" — fine-tuning
consistently hurt, a population-mismatch effect; Magicano: CheMeleon frozen or fine-tuned did not
beat a from-scratch D-MPNN); SSL/contrastive pretraining (jeremy's worst source; JacksonBurns'
ft_pre weak); 3D-conformer and quantum descriptors (null for Lizard Wizard, net-negative for jeremy,
dead for JacksonBurns' Uni-Mol); frozen SMILES transformers (JacksonBurns: standalone NO-GO on 0/4
*and* pool-redundant at zero greedy weight); hand-built pharmacophores (worse than the mean, ADMET
Bo$$); pairwise-ranking losses (Magicano, leeherman); per-compound band-width placement (infeasible —
`TEST-BLINDED.csv` carries only `Molecule_Name` and `SMILES`); and **ensembling before you have two
independently strong models** — Magicano's gain on 2D6 correlated **+0.892 with member strength** and
**−0.736 with member decorrelation**, and the only member that helped was the most correlated one
(ρ 0.920) because it was the strongest. "The practical pre-test is not 'do my members differ?' but
'do I have two or more independently built models of comparable accuracy?'"

---

## Links

| rank | alias | macro | link |
|---|---|---|---|
| 12 | briford | 0.4378 | https://supercowpowers.github.io/workbench/blogs/cyp_challenge/ · https://github.com/SuperCowPowers/workbench |
| 16 | rasayan-labs | 0.4443 | https://github.com/whis9/rasayan-cyp *(empty)* |
| 33 | stir_bar | 0.4769 | https://github.com/lachrymator/openadmet-cyp-challenge-public |
| 40 | alexxi11 | 0.5069 | https://github.com/datagrok-ai/admetica/tree/main/examples/cyp_challenge |
| 46 | JacksonBurns | 0.5181 | https://github.com/JacksonBurns/cyp-challenge · `NOTES.md`, `HANDOFF.md`, `docs/` |
| 47 | jeremy | 0.5185 | https://github.com/jeremycheminf/openadmet_scripts/tree/main/CYP_Challenge |
| 53 | Lizard Wizard Gizzard | 0.5357 | https://github.com/doctawho42/CYPChallenge/blob/main/docs/method_report.md · `METHOD.md` |
| 69 | 450nm | 0.6004 | https://openadmet-cyp-dashboard.vercel.app/operating_dashboard · https://github.com/xX-its-amit-Xx/openadmet-cyp-structure |
| 79 | CompChem Magicano | 0.6404 | https://github.com/mokhalaf22/openadmet-cyp-public · `REPORT.md`, `FINDINGS.md` |
| 87 | adlvdl | 0.6578 | https://github.com/adlvdl/cyp_competition |
| 122 | leeherman | 0.7585 | https://docs.google.com/document/d/1Rhm161zVYxms7FYOXR6zVja5e62jtVTMXY1MXdyTqMM/edit |
| 127 | popcorn2 | 0.7671 | https://github.com/etiu/cyp_challenge |
| 132 | ADMET Bo$$ | 0.7726 | https://github.com/Rebell-Leader/OpenADMET_CYP/blob/main/paper/PAPER_DRAFT.md |

Supporting references named in those reports: Murphy, *Skill Scores Based on the MSE…*, Mon. Weather
Rev. 1988 (the `R² = 2ρk − k² − b²` decomposition); Caruana et al., *Ensemble Selection from
Libraries of Models*, ICML 2004; Zhang et al., *Mitra*, NeurIPS 2025 (arXiv:2510.21204); CheMeleon
(arXiv:2506.15792, Zenodo 15460715); Monroe (arXiv:2608.18982); Capricho
(github.com/David-Araripe/Capricho) for ChEMBL curation; `rd_filters`
(github.com/PatWalters/rd_filters); PubChem AID 1851.
