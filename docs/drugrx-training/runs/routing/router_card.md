---
license: cc-by-4.0
tags: [chemistry, admet, drug-discovery, molecular-property-prediction, applicability-domain]
library_name: lightgbm
---

# DrugRx ADMET router

**Not one model — a routing table.** 21 TDC ADMET endpoints, each served by whichever model won a
measured comparison on that endpoint, with an applicability domain attached to every value.

## What is in here

| path | what |
|---|---|
| `routing.json` | which model serves which endpoint, with the effect size and interval behind it |
| `gbm/<endpoint>.txt` | 21 LightGBM boosters, native text format (not pickles) |
| `gbm/meta.json` | per endpoint: feature arm, task, metric, test score, domain threshold |
| `domain/<endpoint>.npz` | packed training fingerprints + the calibrated in-domain threshold |
| `ensemble.pt` | the 24-member neural ensemble |
| `reproduce.json` | every booster reloaded from its own file and re-scored |

## What it is for

**In-silico triage: ranked hypotheses for wet-lab testing.** Not a therapeutic claim. The modal
outcome of a blind prospective screen is **zero hits** — on CACHE #1 only 7 of 23 teams found any
binder (3.7% overall, at the weak threshold K_D < 150 uM), and on CACHE #2 only 5 of 23 (0.7%).

**37 of CACHE #1's 73 hits had poor solubility despite computational prediction.** Aqueous solubility
is one of the 21 endpoints here, and on it we cannot distinguish our model from a tuned fingerprint
baseline. That is why these values annotate candidates and never filter them.

## The routing table

**The ensemble serves 1 endpoint: pgp. Gradient boosting serves the other 20.**

An endpoint routes to the ensemble only when its 95% CI excludes zero in the ensemble's favour.
**19 of 21 are statistically undecided**, and an undecided endpoint goes to gradient boosting —
it is cheaper, needs no GPU, and "we could not tell" must never resolve to the more expensive
model by omission.

| endpoint | serves | feature arm | test score | metric |
|---|---|---|---|---|
| `AMES` | gradient boosting | maplight | 0.8699 | roc_auc |
| `BBB_Martins` | gradient boosting | count-rdkit2d | 0.9006 | roc_auc |
| `Bioavailability_Ma` | gradient boosting | count-rdkit2d | 0.7393 | roc_auc |
| `Caco2_Wang` | gradient boosting | maplight | 0.2829 | mae |
| `Clearance_Microsome_AZ` | gradient boosting | count-rdkit2d | 0.5630 | spearman |
| `CYP2C9_Veith` | gradient boosting | maplight | 0.7858 | pr_auc |
| `CYP2C9_Substrate_CarbonMangels` | gradient boosting | count-rdkit2d | 0.3230 | pr_auc |
| `CYP2D6_Veith` | gradient boosting | maplight | 0.7163 | pr_auc |
| `CYP2D6_Substrate_CarbonMangels` | gradient boosting | maplight | 0.6974 | pr_auc |
| `CYP3A4_Veith` | gradient boosting | maplight | 0.8831 | pr_auc |
| `CYP3A4_Substrate_CarbonMangels` | gradient boosting | count-rdkit2d | 0.6647 | roc_auc |
| `DILI` | gradient boosting | maplight | 0.9161 | roc_auc |
| `Half_Life_Obach` | gradient boosting | count-rdkit2d | 0.3286 | spearman |
| `hERG` | gradient boosting | count-rdkit2d | 0.8134 | roc_auc |
| `HIA_Hou` | gradient boosting | maplight | 0.9733 | roc_auc |
| `LD50_Zhu` | gradient boosting | maplight | 0.6187 | mae |
| `Lipophilicity_AstraZeneca` | gradient boosting | maplight | 0.5024 | mae |
| `Pgp_Broccatelli` | ensemble | maplight | 0.9110 | roc_auc |
| `PPBR_AZ` | gradient boosting | maplight | 7.9111 | mae |
| `Solubility_AqSolDB` | gradient boosting | count-rdkit2d | 0.7856 | mae |
| `VDss_Lombardo` | gradient boosting | maplight | 0.5173 | spearman |

## The applicability domain, which took two attempts

Every prediction carries `in_domain`, the nearest-neighbour Tanimoto that decided it, and
**`measured_percentile`** — where the query sits among the molecules the reported scores were computed
on. The percentile is the signal; the binary is only a floor.

**The first design was calibrated, defensible and useless.** A 95th-percentile threshold against a
random drug-like background lands near Tanimoto 0.53, while the mean test molecule sits 0.25-0.42 from
its own training set — so it would have flagged nine of every ten molecules our published scores
describe, and a warning on nine rows in ten is the same as no warning.

The threshold is now the 5th percentile of the **validation** fold's own nearest-neighbour
distribution, which leaves the test fold's rate an honest measurement:

- **median in-domain rate on the models' own test folds: 94.0%** (range 75.2-98.5%)
- spread across endpoints: **23.3 points**, against **50.4 points** for a fixed Tanimoto 0.3 cut

That spread is the point: a threshold meaning the same thing everywhere should admit a similar share
of each endpoint's own measured molecules. At a fixed 0.3, `dili` admits 44.8% and `cyp3a4_inhib`
95.2% — a gap about fingerprint density, not coverage.

## Caveats that travel with the numbers

- **Test molecules sit closer to their training set than ordinary chemistry on 20 of 21 endpoints**
  (background-calibrated shift +0.019 to +0.180). Scores overstate performance on new chemistry.
- Eight endpoints have test folds of 95-175 molecules; differences around 0.01 are inside split noise.
- The ensemble side of every interval is **one specific checkpoint**, so re-shipping it obliges a
  rebuild of the routing table.
- **Six ExpansionRx auxiliary heads** are served under an `ExpansionRx_` prefix. They are a different
  laboratory's measurements on their own scale; the two panels share **zero** molecules, so no offset
  to any TDC endpoint is estimable and no conversion exists.

## Provenance

| source | licence | obligation |
|---|---|---|
| TDC ADMET Benchmark Group | per upstream dataset | attribution to upstream curators |
| ExpansionRx via OpenADMET | CC BY 4.0 | attribution |
| **ChEMBL** | CC BY-SA 3.0 | **not used — contributes nothing to these weights** |

No ShareAlike obligation attaches to this artifact.
