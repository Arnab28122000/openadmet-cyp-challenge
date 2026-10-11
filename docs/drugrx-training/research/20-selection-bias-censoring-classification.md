# Exploiting the single-concentration screen, correcting DRC selection bias, and where classification enters

Research note, read 2026-10-08. Scope: the OpenADMET CYP inhibition challenge (Octant Bio data), 750
blinded Enamine-analog test compounds, macro ST-RAE over CYP1A2/2C9/2D6/3A4.

**Screening done for this note:** 580 unique records title/abstract-screened across 84 distinct queries
— 324 from the arXiv API, 150 from Europe PMC REST, 106 from Semantic Scholar — then 21 papers, repos
and documentation pages deep-read. Raw hits and the per-topic rankings are in the session scratchpad
(`lit/ranked.txt`); they are scratch and will not survive.

---

## Executive summary — what to implement first, and the expected effect

1. **The metric, not the model, is the biggest lever.** ST-RAE decomposes per compound, so each
   prediction can be chosen independently to minimise its own expected loss. The optimum is not the
   predictive mean.
2. **Derived here:** the ST-RAE-optimal point `p*` satisfies `P(hi < p*) = P(lo > p*)` — an
   interval-censored median, not a mean. Proof and numbers in §1.
3. **For CYP2D6 this alone is worth more than any model change.** A *single constant* at ≈3.5 — the
   upper part of the inactive window — scores ST-RAE ≈0.68 in simulation, against 1.46 on the
   leaderboard now. Predicting potency-like values (≈5.3) scores 6.7.
4. **Never predict the middle of the inactive window.** 2.4 scores 1.12 (worse than the mean
   predictor); 3.5 scores 0.68. The plateau of zero loss is wide but the optimum sits at its **top**,
   because that is free for inactives and reduces the shortfall on the few actives.
5. **Do it first: train on the non-promoted screen compounds as left-censored labels.** Your own
   `data/cyp_cache/screen_lgbm.json` already proves this: `drc` → `drc+screen` moves 2D6 1.354 → 0.721
   and macro 0.784 → 0.541 on v2. That is the literature's headline effect, already measured in-house.
6. **The right loss for your label format already exists.** Labels are `[lo, hi]` credible intervals —
   that is *interval-censored*, exactly XGBoost's `survival:aft` input, and exactly Arany's CensoredMSE
   / Svensson's TobitLoss in a neural net. Use the interval, not the midpoint.
7. **Make it a hurdle (two-part) model, not a Tobit.** Promotion to DRC is a genuinely separate process
   from potency among actives, which is precisely when the hurdle beats the Tobit. Simulated gain over
   the best constant: 0.68 → 0.34 at 2D6-like class balance.
8. **Calibration of `P(inactive)`, not its AUC, is what decides whether the hurdle helps.** Simulated:
   an AUC-0.84 classifier with miscalibrated probabilities scores **1.50** (worse than a constant);
   the same classifier Platt-calibrated scores **0.64**. Fit the calibrator on a held-out fold.
9. **Multi-fidelity: you are in the inductive setting** (no screen readout for the 750 test compounds),
   so Buterez's embedding augmentation must run off a *predicted* low-fidelity value. Reported inductive
   gains are 20–40% MAE and up to 100% R².
10. **Projected macro:** applying the in-house per-isoform ratios to the leaderboard gives
    0.866 → ≈0.59. Leader is 0.377, so this closes roughly half the gap and is not by itself a win.

---

## 1. The decision-theoretic core (do this before anything else)

This section is a derivation, not a citation. It is the single highest-value item in the note and it
costs no training.

### 1.1 The optimum is an interval-censored median

ST-RAE's numerator is `Σ_i [relu(p_i − hi_i) + relu(lo_i − p_i)]`. It is **separable**: compound `i`'s
loss depends only on `p_i`. So there is no joint optimisation — pick each `p_i` to minimise its own
expected loss under your predictive distribution over `(lo_i, hi_i)`.

```
E[L(p)] = E[relu(p − hi)] + E[relu(lo − p)]
d/dp E[L(p)] = P(hi < p) − P(lo > p)
```

Setting the derivative to zero:

```
p*  such that   P(hi < p*) = P(lo > p*)
```

Read it in words: **the optimal prediction is the point at which the probability that the true interval
lies entirely below it equals the probability that it lies entirely above it.** When `lo = hi = y` this
collapses to `P(y < p) = P(y > p)`, the ordinary median — the familiar minimiser of L1 loss. Widening
the interval creates a *plateau* of equally optimal points rather than a unique one.

Two consequences that matter:

- **Fit the median, not the mean.** An MSE-trained ensemble averaged over members returns a conditional
  mean. Under a right-skewed or bimodal predictive distribution the mean sits above the median, which is
  exactly the direction ST-RAE punishes on a mostly-inactive isoform. Average *quantiles* or take the
  member-wise median instead.
- **Break plateau ties at the top.** When `P(inactive) → 1` every point in the inactive window is
  equally free, so an argmin over a grid picks arbitrarily. Picking the bottom (1.1) is catastrophic the
  moment the compound is actually active; picking the top (≈3.4–3.6) is nearly free either way. Clip
  `P(inactive)` below 1 and break ties upward.

This loss family is the piecewise-linear "check"/pinball family whose minimisers are quantiles
(Koenker & Bassett 1978); the interval version simply replaces the point by a band. Chemprop 2.x already
ships `quantile`/`pinball` regression losses, so the machinery is available off the shelf
([loss functions docs](https://chemprop.readthedocs.io/en/main/tutorial/python/loss_functions.html)).
Mervin et al.'s argument that model evaluation must be done under the *downstream* loss rather than a
generic one is the same point made from the evaluation side
([arXiv:1807.08926](https://arxiv.org/abs/1807.08926)).

### 1.1a Independently derived in-tree — and one tie-break worth changing

While this note was being written, a concurrent session derived the same condition and implemented it as
`talindrew-drugrx/drugrx/cyp/decide.py::balance_point`, with an exact O(n log n) knot walk instead of the
grid search used here. **Two independent derivations reaching `P(lo > p) == P(hi < p)` is the strongest
validation this result is going to get**, and the exact walk is the better implementation — adopt it.

One substantive disagreement. `balance_point`'s docstring says that on a flat segment "the left end is
returned". Across most of the range that is a mildly *conservative* choice, and §1.2 shows why it is
survivable: under-prediction is much cheaper than over-prediction. But it has a tail failure. When every
draw for a compound is inactive — `p_inactive` rounding to 1.0, or a small draw set all landing in the
window — the flat segment becomes the whole inactive window and the left end is ≈1.07. If that compound
is in fact active, the loss is the full ~2.4 log units.

The fix is free. In §1.2 the top-tie and left-tie rules score **identically** once `P(inactive)` is
calibrated (0.474 / 0.474, 0.643 / 0.643, 0.675 / 0.675) — because calibrated probabilities never
saturate, so the flat segment is bounded and both rules pick the same point. So changing the tie-break to
the **right** end of the flat segment costs nothing measurable and removes the tail. Do both: clip
`p_inactive` strictly below 1, and return the right end.

### 1.2 Numbers (simulation, 750 compounds, inactive window `[1.1, U(3.4,4.0)]`, actives ~N(5.3,0.8) with CI half-width U(0.1,0.45))

Best **constant** prediction, by fraction inactive:

| π(inactive) | mean(y) | optimal constant `p*` | ST-RAE at `p*` | at 2.4 (mid-window) | at 3.7 | at 5.3 (potency-like) |
|---|---|---|---|---|---|---|
| 0.90 | 2.68 | **3.47** | **0.678** | 1.122 | 0.862 | 6.651 |
| 0.80 | 2.98 | **3.55** | **0.761** | 1.292 | 0.799 | 3.344 |
| 0.60 | 3.56 | **3.77** | **0.960** | 1.722 | 0.965 | 1.885 |
| 0.30 | 4.41 | **4.77** | **0.957** | 2.455 | 1.287 | 1.064 |

The ridge is narrow and asymmetric: at π=0.9, moving from 3.47 to 4.0 costs 0.678 → 1.644. **Over-prediction
is punished far harder than under-prediction**, because the inactive window extends ~2.4 log units below
the active mode and 0 above it.

Bayes point as a function of calibrated `P(inactive)` (ties broken at the top):

| P(inactive) | 0.0 | 0.3 | 0.5 | 0.7 | 0.8 | 0.9 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|---|
| `p*` | 5.30 | 4.83 | 3.94 | 3.65 | 3.54 | 3.46 | 3.43 | 3.41 |

Note how fast it collapses: by `P(inactive) = 0.5` the optimum has already fallen to 3.94, within half a
log unit of its asymptote. **A compound only has to be a coin-flip for inactivity to be worth predicting
near the window.** This is the mechanism behind 2D6 = 1.46.

What a per-compound classifier buys over the best constant (π=0.9):

| strategy | ST-RAE |
|---|---|
| mean predictor (definition) | 1.000 |
| best constant (3.47) | 0.678 |
| **calibrated** `P(inactive)`, AUC 0.97 → Bayes point | 0.474 |
| **calibrated** `P(inactive)`, AUC 0.84 → Bayes point | 0.643 |
| *un*calibrated, AUC 0.84 → Bayes point | **1.501** ← worse than a constant |
| hurdle: calibrated `q` + potency model (RMSE 0.7) | **0.343** |
| oracle class + potency RMSE 0.7 | 0.216 |
| oracle class + oracle potency | 0.072 |

Three readings. (a) The ceiling is very high — getting the active/inactive call right matters far more
than potency precision. (b) A **calibrated** classifier beats the best constant at every AUC tested,
down to 0.63. (c) An **uncalibrated** one is worse than doing nothing. The 1.50-vs-0.64 pair is the same
classifier; only the probability mapping differs.

---

## 2. Ranked methods

### Rank 1 — Train on the non-promoted screen compounds as left-censored labels

**What it is.** The 2,882 compounds that failed to promote out of the 50 µM screen are not missing data;
they are *information*. "No inhibition at 50 µM" means pIC50 is below the lowest tested dose — a
left-censored observation, `pIC50 ∈ (−∞, ~4.3]`, or more honestly the same fixed window the challenge
itself assigns to inactives, `[1.1, 3.4–4.0]`. Train with a loss that is zero anywhere inside the
admissible region and penalises only excursions outside it. That is Arany et al.'s **CensoredMSE** (a
one-sided squared error) and Svensson et al.'s **TobitLoss** (the Gaussian-NLL extension), or equivalently
Chemprop's `bounded-mse` / `bounded-mae`.

**Evidence.**
- **Your own data already shows it.** `data/cyp_cache/screen_lgbm.json`, v2 split, mean of 3 seeds:

  | arm | 1A2 | 2C9 | 2D6 | 3A4 | macro |
  |---|---|---|---|---|---|
  | `drc` | 0.773 | 0.525 | 1.354 | 0.482 | 0.784 |
  | `drc+inactive` | 0.565 | 0.510 | 0.729 | 0.489 | 0.573 |
  | `drc+screen` | **0.539** | **0.429** | **0.721** | **0.474** | **0.541** |

  2D6 falls 47%; macro falls 31%. The literature below explains *why* and how to extract the rest.
- Svensson et al. (2025), 15 internal AstraZeneca assays, 0–63% censored (CYP2C9 itself at **63%**,
  12,876 compounds): training with censored labels was "overwhelmingly favourable" on NLL, significant
  at p<0.05 (Mann-Whitney) across most model/assay pairs; epistemic-uncertainty models improved on all
  but 2 assays. Aleatoric gains concentrated where censoring exceeded 35%.
  [arXiv:2409.04313](https://arxiv.org/abs/2409.04313) · code
  [MolecularAI/uq4dd](https://github.com/MolecularAI/uq4dd) (Apache-2.0)
- MELLODDY: 20–80% of partner data is censored; dropping it is not an option at industrial scale.
  [J Chem Inf Model 64:2331](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00799)
- Lanevskij & Didziapetris rebuilt a hERG potency model with XGBoost's **AFT** objective, which "accepts
  both quantitative and censored data", and reached pIC50 R²≈0.4, MAE<0.5 while keeping classification
  accuracy at what they call the performance ceiling — explicitly recommending the approach "for
  quantitative modeling of various ADME and drug safety endpoints with a high prevalence of censored
  data". [doi:10.1007/s10822-022-00483-0](https://doi.org/10.1007/s10822-022-00483-0)
- Older but exactly on point: Bruce et al. showed that keeping qualified `<`/`>` values in the training
  set matters because otherwise "less active compounds are often predicted to be more active than they
  really are" — our failure mode verbatim. PMID 27464349.

**Implementation sketch.**
The reference implementation is 15 lines (from `uq4dd/utils/loss_functions.py`, Arany et al. 2022):

```python
# mask: -1 left-censored, 0 observed, +1 right-censored
diff = target - pred
diff[mask ==  1] = clamp(diff[mask ==  1], min=0.0)   # only penalise pred below a '>' bound
diff[mask == -1] = clamp(diff[mask == -1], max=0.0)   # only penalise pred above a '<' bound
loss = (diff ** 2).mean()
```

For us the label is a two-sided interval, so generalise to the interval hinge you already use —
`relu(pred − hi) + relu(lo − pred)` — and simply extend it to the screen compounds with
`lo = 1.1, hi = 3.4..4.0`. Three concrete arms, per isoform:

- `drc+screen-censored`: screen non-promotees in at the inactive window, full weight.
- `drc+screen-censored, weighted`: down-weight them by `w` and sweep `w ∈ {0.1, 0.25, 0.5, 1}` —
  3,000 censored labels against 1,300–2,300 observed ones will otherwise dominate the gradient.
- **Per isoform, not globally.** 2D6's promotion rule is a hard threshold, so its non-promotees are
  genuinely, cleanly inactive. 2C9's promotion is ~random one-third, so its non-promotees are a random
  sample and the censored labels are *unbiased extra data*. 3A4's is enriched for weak inhibitors, so its
  non-promotees are the least informative. Expect the gain to be largest on 2D6 and 1A2 — which is what
  the table above shows.

**Risks.** The hard one: the inactive window's upper edge (3.4–4.0) is a property of the *challenge's*
labelling convention, not of the 50 µM screen. 50 µM is pIC50 4.3. If you censor at 4.3 you are
consistent with the assay; if you censor at 3.4–4.0 you are consistent with the metric. These differ by
up to 0.9 log units and the §1 analysis says that is a large distance. **Sweep the censoring ceiling
`{3.4, 3.7, 4.0, 4.3}` as a hyperparameter and let v2 choose.** Second risk: `drc+inactive` already gets
most of the way (0.573 vs 0.541), so the marginal value of the full continuous screen signal over a
binary inactive flag is real but modest — do not over-engineer the log2FC regression head before the
censored-label arm is in.

---

### Rank 2 — Reparameterise the output as a hurdle (two-part) model, and emit the Bayes point

**What it is.** Two heads per isoform: a **classifier** `q(x) = P(inactive at 50 µM)` trained on all
4,376 screened compounds (a label you have for everything), and a **potency regressor** `f(x)` trained
on the promoted DRC compounds. At inference, do *not* multiply or average them. Combine them into a
predictive distribution and return the §1 Bayes point:

```
E[L(p)] = q(x) · E_window[relu(p−hi) + relu(lo−p)]  +  (1−q(x)) · E_active[relu(p−hi) + relu(lo−p)]
p*(x)   = argmin_p E[L(p)]        # 1-D grid search over p ∈ [1, 9], ties broken upward
```

**Why hurdle and not Tobit.** The Tobit assumes *one* latent process censored at a threshold; the hurdle
assumes the zero/non-zero decision and the magnitude are **separate processes**. The 2026 comparison is
blunt: use the Tobit "only when these assumptions are scientifically plausible and empirically
supported; otherwise the hurdle model is preferable", because real semi-continuous data frequently
violates Tobit's restrictions ([arXiv:2608.09725](https://arxiv.org/abs/2608.09725)). Here the processes
*are* different — whether a compound touches the enzyme at 50 µM is not the same question as how potent
it is once it does, and we know the promotion rule differs per isoform. That is the hurdle case.

**Evidence.** §1.2: hurdle with a calibrated classifier and an RMSE-0.7 potency model scores **0.343**
against 0.678 for the best constant, at π=0.9 — a 2× improvement, and 0.533 at AUC 0.85 with π=0.8.
Consistent in direction and roughly in magnitude with your measured `drc+screen` 2D6 = 0.721, which is a
*loss-level* correction without an explicit Bayes point; the extra factor is what the explicit decision
rule should add.

**Implementation sketch.** You already have the pieces (`cyp_dmpnn.py`, `screen_nets/`). Add:
1. an `inactive` head on the existing multitask trunk, trained on the 4,376 × 4 screen matrix with BCE
   (label = `log2FC > −0.6` or the per-isoform promotion threshold);
2. **a Platt or isotonic calibrator per isoform**, fitted on a held-out fold, never on training data;
3. a vectorised Bayes-point lookup table `p*(q)` precomputed on a grid of `q ∈ [0,1]` — the active
   component only needs `f(x)` and its predictive spread, so the table is 1-D per compound and the whole
   thing is a `np.searchsorted`.

**Risks.** The dominant one is calibration (§1.2: 1.50 vs 0.64 for the same classifier). Second: `q(x)`
is trained on 4,376 Octant screen compounds and applied to 750 Enamine analogs of *hits* — a shifted
population, which is Rank 4's problem. Third: for 1A2/2C9/3A4 the test set is deliberately
hit-enriched, so `π(inactive)` there is **low** and pushing predictions toward the window will *hurt*.
Gate the correction on the per-isoform prior; do not apply one global rule.

---

### Rank 3 — Use the interval directly via XGBoost `survival:aft` as a strong, cheap second model

**What it is.** XGBoost's AFT objective takes `label_lower_bound` / `label_upper_bound` per row and
natively handles uncensored `[a,a]`, right-censored `[a,∞)`, left-censored `[0,b]` and
**interval-censored `[a,b]`** in one model, fitting `ln(Y) = T(x) + σZ` with `Z` normal, logistic or
extreme. ([docs](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html);
Barnwal, Cho & Hocking, *JCGS* 31:1292, [doi:10.1080/10618600.2022.2067548](https://doi.org/10.1080/10618600.2022.2067548))

This is the only off-the-shelf objective that matches our label format exactly — we *have* per-compound
`[lo, hi]`, which almost no public benchmark does.

**Evidence.** Lanevskij & Didziapetris's hERG model is the direct precedent: same objective, same
"high prevalence of censored data" setting, pIC50 R²≈0.4 / MAE<0.5 with no loss of classification
accuracy and the benefit of being untied from any cut-off.

**Implementation sketch.** The one trap: AFT works on `log Y`, and our labels are already a log scale.
Feed `exp(pIC50)` as the bound so that `T(x)` lands directly on the pIC50 scale:

```python
dtrain.set_float_info('label_lower_bound', np.exp(lo))   # lo, hi are pIC50
dtrain.set_float_info('label_upper_bound', np.exp(hi))   # inactives: exp(1.1), exp(3.7)
params = dict(objective='survival:aft', eval_metric='aft-nloglik',
              aft_loss_distribution='normal', aft_loss_distribution_scale=0.5,
              tree_method='hist')
# prediction is on the Y scale -> pIC50_hat = np.log(bst.predict(d))
```

`exp(9) ≈ 8100`, numerically harmless. Sweep `aft_loss_distribution_scale` — it is the assumed label
noise and directly controls how far predictions are pulled toward the interval centre. Run it on the
same RDKit2D + count-ECFP4 features for a cheap, decorrelated ensemble member.

**Risks.** AFT gives a point estimate plus a global σ, not a per-compound predictive distribution, so
the §1 Bayes point has to borrow its spread from elsewhere. Gradient-boosted trees on 1,300–2,300 rows
will underperform the MLP ensemble on raw accuracy; its value is decorrelation and the fact that it
consumes the interval honestly. Also: the AFT likelihood on a `[1.1, 3.7]` interval is very flat, so
inactives contribute little gradient — down-weighting may be unnecessary here, unlike Rank 1.

---

### Rank 4 — Multi-fidelity: predicted low-fidelity value as an input feature (inductive setting)

**What it is.** Buterez et al.'s result is the closest published analogue of our exact data shape:
low-fidelity = single-dose primary screen, high-fidelity = dose-response pXC50, with HF 2–3 orders of
magnitude smaller. Their strategies, in their own ranking:

| strategy | mechanism | wins (transductive) |
|---|---|---|
| **Embedding augmentation** | train an LF model, concatenate its *latent embedding* to the HF model's input | **40/51** |
| Label augmentation | concatenate the actual LF label | 10/51 |
| Predicted label | LF model predicts a label for HF inputs | moderate |
| Tune readout | pre-train on LF, fine-tune only the readout | strong |
| Tune network | pre-train on LF, fine-tune everything | weaker |

**Evidence (numbers).**
- Nat Commun 2024: 28M+ protein-ligand measurements, 37 targets. Transductive: MAE down 10–40%;
  adaptive embeddings beat raw labels by up to 10%. **Inductive** (what we are in): "often between
  20%–40% in MAE and up to 100% in R²". Sparse-HF regime: R² 0.049 → 0.331 (LF labels) → **0.455**
  (adaptive embeddings) at 1K HF samples — the "up to eight times" headline, with an order of magnitude
  less HF data. [PMC11258334](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/) ·
  [doi:10.1038/s41467-024-45566-8](https://doi.org/10.1038/s41467-024-45566-8) ·
  [code](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics)
- MF-PCBA (J Chem Inf Model 2023), 60 datasets: average R² with SD labels — RF 0.219→0.285,
  SVM 0.228→0.279, DL 0.244→0.312, i.e. **+30–40% relative**; best case 0.048→0.435. Primary screens
  average ~310k compounds, confirmatory ~1,400 — and **35.6% of DR compounds have no explicit XC50**,
  which they handle by assigning the dataset's lowest activity value, i.e. treating them as censored
  lower bounds. [PMC10170507](https://pmc.ncbi.nlm.nih.gov/articles/PMC10170507/) ·
  [code](https://github.com/davidbuterez/mf-pcba)
- MF-PCBA's caveat is important for us: gains are largest at **low landscape roughness** (ROGI < 0.3)
  and "extremely small or even negative" at ROGI > 0.5 — and separately, "some of the highest gains
  were for datasets with extremely low correlation" between SD and DR, so SD/DR correlation is "not
  always indicative of the performance uplift". Do not pre-screen the idea on the log2FC-vs-pIC50
  correlation; just run the arm.

**Implementation sketch.** The decisive constraint: **the 750 test compounds have no screen readout**,
so label augmentation and embedding augmentation are unavailable in their transductive form. Buterez is
explicit that transductive methods "are generally inapplicable" in real drug discovery. So:

1. Train an LF model `g(x) → log2FC` (4,376 × 4 — you have `screen_nets/`, `screen_lgbm`).
2. Take its **penultimate-layer embedding** `z(x)`, not just the scalar, and concatenate to the potency
   model's features. This is embedding augmentation run on predicted LF, the inductive variant.
3. Crucially, train the HF model on `z(x)` computed *out-of-fold* for the DRC compounds, so the HF model
   never sees an LF embedding that had the compound's own label in it — otherwise the train/test
   distribution of `z` differs and the gain evaporates. This is Buterez's "hybrid label" failure mode
   (train on real LF, evaluate on predicted) made concrete.
4. Also try `Tune readout`: pre-train the trunk on the 4,376 × 4 log2FC matrix, freeze it, fine-tune only
   the heads on DRC. Cheap and it was one of their stronger strategies.

**Risks.** Your auxiliary PubChem AID 1851 heads already do something adjacent, so the marginal gain may
be small — the LF signal may already be in the trunk. The out-of-fold discipline in step 3 is easy to get
wrong and silently inflates validation. And the whole family is a *variance* play on 1,300–2,300 HF
labels; it will not fix a mis-specified output distribution, which is Ranks 1–2.

---

### Rank 5 — Inverse-propensity weighting of the DRC labels, used carefully and per isoform

**What it is.** Promotion to DRC is non-random, so the DRC set is a Missing-Not-At-Random sample of the
screened library. Fit a propensity `e(x) = P(promoted | x)` and weight each DRC label by `1/e(x)` to
recover an unbiased risk estimate over the *screened* population (Schnabel et al.'s IPS-ERM,
[ICML 2016](https://arxiv.org/abs/1602.05352); Heckman 1979 for the selection-model lineage).

**We are in an unusually strong position to do this.** IPW normally fails because the propensity is
unknown. Here you have the log2FC for *all* 4,376 compounds and you know which were promoted, so `e(·)`
is directly estimable — and you have already characterised it: 2D6 a hard threshold at log2FC ≈ −0.6,
1A2 monotone in inhibition, 2C9 ~uniform one-third, 3A4 mildly enriched for weak inhibitors.

**The catch that demotes this to Rank 5.** IPW re-weights training toward the **screened library**. That
is *not* our test distribution. The 750 compounds are the top-10 ECFP4 analogs of the top-25 hits for
1A2/2C9/3A4 — a deliberately hit-enriched, selection-conditional sample. So:

- For **2D6**, whose hits were *not* used to build the test set, the test compounds are close to a
  generic Enamine sample, and correcting toward the unselected population is the right move. This is the
  isoform where IPW points the same way as Rank 1.
- For **1A2/2C9/3A4**, the DRC-promoted set may be *closer* to the hit-analog test set than the full
  screen is. IPW would correct in the wrong direction. Their current scores (0.52–0.88) are consistent
  with that reading.

The formally correct framing for a test set defined by "top-K by a score" is **selection-conditional**
coverage, which conditions the calibration on the selection event rather than treating it as marginal
(Jin & Candès, [arXiv:2403.03868](https://arxiv.org/abs/2403.03868)). Worth knowing; probably more than
this challenge needs.

**Implementation sketch.** Fit `e(x)` per isoform by logistic regression on log2FC + its standard error
+ a few descriptors; clip weights at the 95th percentile (unclipped IPW has notorious variance);
run as an *ablation on 2D6 only* first. Keep the pre-registration discipline: this has a real chance of
being a null or a regression on three of four isoforms.

**Risks.** Weight-clipping choices dominate the result; IPW is high-variance at n≈1,300; and a
mis-specified `e(x)` can be worse than none. Note also Schnabel's own condition — IPS is unbiased only
when estimated propensity equals true propensity.

---

### Rank 6 — Per-isoform priors from CYP promiscuity base rates

**What it is.** Set each isoform's prior `π(inactive)` and the strength of the Rank 2 correction from
published base rates, rather than from the training set's promoted subsample (which is biased by
construction).

**Evidence (numbers).**
- Veith et al., 17,143 compounds × 5 isozymes (1A2, 2C9, 2C19, 2D6, 3A4), qHTS: cross-library isozyme
  inhibition **30–78%**; only **7% of a typical screening library is inactive against all five**, versus
  **33% of FDA-approved drugs**. [Nat Biotechnol 27:1050](https://doi.org/10.1038/nbt.1581)
- Sun, Veith, Xia, Austin & Huang, modelling the same data — the decisive table for us, active fraction
  per isoform: **1A2 39.9%, 2C9 38.0%, 2C19 44.7%, 2D6 13.6%, 3A4 34.3%**. 2D6 is the outlier by a
  factor of ~2.5–3. Model AUCs: 1A2 0.93, 2C9 0.89, 2C19 0.89, 2D6 0.85, 3A4 0.87 — 2D6 also the
  hardest, attributed to class imbalance plus an "adaptive ligand binding site, capable of accommodation
  of structurally diverse molecules". [PMC3200453](https://pmc.ncbi.nlm.nih.gov/articles/PMC3200453/) ·
  [doi:10.1021/ci200311w](https://doi.org/10.1021/ci200311w)
- **Cross-isoform transfer is weaker than multitask papers imply.** The same paper reports substructure
  overlap only between 2C9 and 2C19 (16 of the top 20 atom types shared, 91% sequence identity) and
  provides *no* cross-prediction benefit between the others. We do not have 2C19 in this challenge, so
  the one genuinely correlated pair is absent. Treat the four isoforms as four tasks that share a trunk,
  not as one task.
- Promiscuity is itself predictable: Hit Dexter 2.0 reaches MCC up to 0.64 / AUC up to 0.96 for
  (highly) promiscuous vs non-promiscuous, over both primary *and* confirmatory assays.
  [doi:10.1021/acs.jcim.8b00677](https://doi.org/10.1021/acs.jcim.8b00677)

**Implementation sketch.** Two cheap uses. (a) Initialise each isoform's classifier bias to the Veith
base rate so the untrained prior is right. (b) Feed a predicted-promiscuity feature (number of the 4
isoforms the screen says the compound touches, out-of-fold) into each head — a compound inactive on the
other three is much more likely inactive here. Given the Veith numbers, the 2D6 head should carry a
visibly stronger inactive prior than the other three.

**Risks.** Veith used Promega bioluminescent assays on recombinant enzymes; Octant's platform and
concentration range differ, so the absolute rates do not transfer — the *ordering* and the ~3×
2D6 deficit are what to rely on. And the test set is Enamine analogs of hits, not a library sample, so
the prior must be the *conditional* one for the analog series, not the library base rate.

---

### Rank 7 — Do *not* convert log2FC to pIC50 by Hill-equation inversion

Listed as a ranked method because it is the obvious idea and the literature says it is a trap.

**What it is.** Invert the four-parameter Hill equation at the single 50 µM point to impute a pIC50 from
% inhibition, then train on the imputed values as if they were measurements.

**Evidence against.** Gubler, Schopfer & Jacoby examined exactly this on HDAC4 and ENPP2 HTS data
against numerical simulations, varying screening concentration, Hill parameters and assay quality.
Their finding: a "good" correlation between single-point % inhibition and IC50 "is not always observed,
and in the literature there seems confusion about what correlation even to expect", and **"the typical
variations of the actual compound concentrations in existing screening libraries generate the largest
contributions to imperfect correlations"** — i.e. the dominant error is not the Hill-slope assumption
but the fact that nominal library concentrations are wrong. Excess scatter beyond their model was
explained by compound aggregation.
[J Biomol Screen 18:1 (2013), doi:10.1177/1087057112455219](https://doi.org/10.1177/1087057112455219)

The practical reading: a single-point readout carries solid information about *whether* a compound
inhibits and very little about *how much*. So use it as (a) a censoring bound (Rank 1), (b) a classifier
label (Rank 2), (c) an auxiliary regression target or embedding (Rank 4) — and **not** as an imputed
pIC50 with the same loss weight as a DRC measurement. Note that even MF-PCBA, which is built on exactly
this data shape, does not invert the Hill equation; it assigns unmeasured DR compounds "the lowest
activity value recorded in each dataset", a censored bound.

---

## What NOT to do

1. **Do not impute pIC50 from log2FC via the Hill equation and train on it as an observation.** Rank 7.
   Nominal-concentration error dominates; you would be fitting noise with high confidence.
2. **Do not plug an uncalibrated classifier probability into the Bayes point.** §1.2: the same AUC-0.84
   classifier scores 0.64 calibrated and **1.50** uncalibrated — worse than a constant. Fit Platt or
   isotonic on a held-out fold, and check the reliability curve, not the AUC.
3. **Do not predict the middle of the inactive window.** 2.4 scores 1.12 at π=0.9, worse than the mean
   predictor; 3.47 scores 0.68. The free plateau is real but its optimum is at the **top**.
4. **Do not return the left end of the flat segment.** `drugrx/cyp/decide.py::balance_point` currently
   does. It is harmless once `P(inactive)` is calibrated (§1.2: identical scores either way) but it has a
   tail failure when every draw lands inactive, which sends the prediction to ≈1.07 and costs ~2.4 log
   units if the compound is active. Clip `p_inactive` strictly below 1 and return the right end. §1.1a.
5. **Do not apply one global de-biasing rule across the four isoforms.** Promotion differs per isoform
   *and* the test set was built from 1A2/2C9/3A4 hits but not 2D6's. The correction that fixes 2D6 will
   damage the other three if applied at the same strength.
6. **Do not average ensemble members' means and call it a point prediction.** ST-RAE wants a median-like
   quantity (§1.1). Take the member-wise median, or average quantiles.
7. **Do not trust v1/v2 validation to measure any of this.** Your `screen_lgbm.json` records
   `test_frac_lt_3p6` of 0.027–0.156 on the validation splits, so the validation sets are *not*
   mostly-inactive and cannot see the 2D6 failure mode. The whole point of `DISCONNECTED-BENCH.md`.
   Build a held-out split that is deliberately inactive-enriched before believing any number here.
8. **Do not pre-screen the multi-fidelity idea on the log2FC-vs-pIC50 correlation.** MF-PCBA found some
   of the largest gains on datasets with "extremely low correlation" between fidelities.
9. **Do not expect cross-isoform transfer to carry much.** The one strongly-coupled pair in the
   literature is 2C9/2C19, and 2C19 is not in this challenge.
10. **Do not drop `circular_assembly`-style scope creep into this** — none of Ranks 1–3 requires a new
    architecture. The output distribution and the loss are mis-specified; fix those before adding a
    pretrained encoder.

---

## Bibliography

Censored / interval-censored regression
- Svensson E, Friesacher HR, Winiwarter S, Mervin L, Arany A, Engkvist O. *Enhancing uncertainty
  quantification in drug discovery with censored regression labels.* Artificial Intelligence in the Life
  Sciences (2025). https://arxiv.org/abs/2409.04313 · code https://github.com/MolecularAI/uq4dd
- Arany A, Simm J, Oldenhof M, Moreau Y. *SparseChem: Fast and accurate machine learning model for small
  molecules* (2022) — origin of CensoredMSE. https://arxiv.org/abs/2203.04676 ·
  code https://github.com/melloddy/SparseChem
- Heyndrickx W et al. *MELLODDY: Cross-pharma Federated Learning at Unprecedented Scale…* J Chem Inf
  Model 64:2331 (2024). https://pubs.acs.org/doi/10.1021/acs.jcim.3c00799
- Heid E et al. *Chemprop: A Machine Learning Package for Chemical Property Prediction.* J Chem Inf
  Model (2024). https://pubs.acs.org/doi/10.1021/acs.jcim.3c01250 ·
  losses https://chemprop.readthedocs.io/en/main/tutorial/python/loss_functions.html
- Lanevskij K, Didziapetris R. *Physicochemical QSAR analysis of hERG inhibition revisited: towards a
  quantitative potency prediction.* J Comput Aided Mol Des 36 (2022).
  https://doi.org/10.1007/s10822-022-00483-0
- Barnwal A, Cho H, Hocking T. *Survival Regression with Accelerated Failure Time Model in XGBoost.*
  JCGS 31:1292 (2022). https://doi.org/10.1080/10618600.2022.2067548 ·
  docs https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html
- Bruce CL et al. *QSAR analysis involving assay results which are only known to be greater than, or
  less than some cut-off limit* (2010). PMID 27464349.
- Mervin LH, Trapotsi M-A, Afzal AM, Barrett IP, Bender A, Engkvist O. *Probabilistic Random Forest
  improves bioactivity predictions close to the classification threshold by taking into account
  experimental uncertainty.* J Cheminform (2021). https://doi.org/10.1186/s13321-021-00539-7

Selection bias, MNAR, hurdle vs Tobit
- Heckman JJ. *Sample Selection Bias as a Specification Error.* Econometrica 47:153 (1979).
- Schnabel T, Swaminathan A, Singh A, Chandak N, Joachims T. *Recommendations as Treatments: Debiasing
  Learning and Evaluation.* ICML (2016). https://arxiv.org/abs/1602.05352
- *Comparing Tobit and Two-Part Hurdle Models for Semi-Continuous Longitudinal Data* (2026).
  https://arxiv.org/abs/2608.09725
- *On Prediction Feature Assignment in the Heckman Selection Model* (2023).
  https://arxiv.org/abs/2309.08043
- Jin Y, Candès EJ. *Confidence on the Focal: Conformal Prediction with Selection-Conditional Coverage*
  (2024). https://arxiv.org/abs/2403.03868
- *Contextual Representation Anchor Network to Alleviate Selection Bias in Few-Shot Drug Discovery*
  (2024) — modest (+2.6% AUC MoleculeNet, +3.28% ΔAUC-PR FS-Mol). https://arxiv.org/abs/2410.20711

Multi-fidelity
- Buterez D, Janet JP, Kiddle SJ, Oglic D, Liò P. *Transfer learning with graph neural networks for
  improved molecular property prediction in the multi-fidelity setting.* Nat Commun 15 (2024).
  https://doi.org/10.1038/s41467-024-45566-8 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/ ·
  code https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics
- Buterez D, Janet JP, Kiddle SJ, Liò P. *MF-PCBA: Multifidelity High-Throughput Screening Benchmarks
  for Drug Discovery and Machine Learning.* J Chem Inf Model 63 (2023).
  https://doi.org/10.1021/acs.jcim.2c01569 · https://pmc.ncbi.nlm.nih.gov/articles/PMC10170507/ ·
  code https://github.com/davidbuterez/mf-pcba

Single-point to potency
- Gubler H, Schopfer U, Jacoby E. *Theoretical and experimental relationships between percent inhibition
  and IC50 data observed in high-throughput screening.* J Biomol Screen 18 (2013).
  https://doi.org/10.1177/1087057112455219

Decision theory / conformal / ordinal
- Koenker R, Bassett G. *Regression Quantiles.* Econometrica 46:33 (1978) — the check-loss/quantile
  correspondence underlying §1.
- *A decision theoretic approach to model evaluation in computational drug discovery* (2018).
  https://arxiv.org/abs/1807.08926
- Svensson F, Afzal AM, Norinder U, Bender A. *Maximizing gain in high-throughput screening using
  conformal prediction.* J Cheminform 10 (2018). https://doi.org/10.1186/s13321-018-0260-4
  (Fredrik Svensson — distinct from Emma Svensson's censored-regression work above.)

CYP-specific
- Veith H, Southall N, Huang R et al. *Comprehensive characterization of cytochrome P450 isozyme
  selectivity across chemical libraries.* Nat Biotechnol 27:1050 (2009). https://doi.org/10.1038/nbt.1581
- Sun H, Veith H, Xia M, Austin CP, Huang R. *Predictive models for cytochrome P450 isozymes based on
  quantitative high throughput screening data.* J Chem Inf Model 51 (2011).
  https://doi.org/10.1021/ci200311w · https://pmc.ncbi.nlm.nih.gov/articles/PMC3200453/
- Stork C, Chen Y, Šícho M, Kirchmair J. *Hit Dexter 2.0: Machine-Learning Models for the Prediction of
  Frequent Hitters.* J Chem Inf Model 59 (2019). https://doi.org/10.1021/acs.jcim.8b00677

---

## Reproducing the simulation in §1

The numbers in §1.2 come from three short scripts written to the session scratchpad
(`dec.py`, `dec3.py`, `dec4.py`) and not committed. They are ~40 lines each: build a synthetic test set
of 750 compounds with a given inactive fraction, inactive window `[1.1, U(3.4,4.0)]`, actives
`N(5.3,0.8)` with CI half-width `U(0.1,0.45)`; grid-search the ST-RAE-optimal constant; precompute the
Bayes-point lookup `p*(q)`; and compare oracle, calibrated and uncalibrated hurdle variants. The
assumptions about the active distribution are guesses — **re-run against the real v2 interval
distribution before trusting the absolute values.** The *orderings* and the mechanism (§1.1) are
analytic and do not depend on those guesses.
