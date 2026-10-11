# 13 — Tabular foundation models miss the bar, and 1000 of our 3091 features carry the problem

**Status: closed by its own pre-registered bar.** `scripts/cyp_tabfm.py`, 3 seeds × 4 isoforms ×
{lgbm_all, lgbm_topk, tabicl_topk} × k ∈ {100, 250, 500, 1000}, on L4. Pre-registered in the
script docstring before the run.

## Why this was the one untried family

Model class was the only lever left — placement is measured-exhausted (`v9` took the vertex on all
four axes, produced the best row this programme has had, and the rank still fell 54 → 59). And the
literature pointed at exactly one thing: **Mol-JEPA reports that on its own FDR-corrected paired
Wilcoxon test, TabICLv2 + AlvaDesc beat it 47% to 38%** across 23 ADME/potency datasets — an
off-the-shelf tabular foundation model on descriptors beating a purpose-built molecular JEPA, at
roughly our data shape (n = 6145, d = 3091, regression).

**TabPFN was excluded by licence and stayed excluded.** Its weights require a one-time licence
acceptance, which is not a decision this programme can make on Arnab's behalf, and its terms
prohibit production deployment and commercial benchmarking — so a win could not have been
submitted or shipped. TabICL is permissively licensed, so it is the one that could have been acted
on.

## The result: TabICL's edge is a 100-column artefact

Matched-feature contrast, `tabicl_topk − lgbm_topk` at the same budget:

| isoform | k=100 | k=250 | k=500 | k=1000 |
|---|---|---|---|---|
| CYP1A2 | **+0.0264** | +0.0088 | −0.0027 | −0.0166 |
| CYP2C9 | **+0.0174** | −0.0014 | −0.0019 | −0.0115 |
| CYP2D6 | −0.0109 | −0.0092 | −0.0118 | −0.0332 |
| CYP3A4 | +0.0090 | −0.0036 | −0.0085 | −0.0205 |

**Monotone decay in k on all four isoforms**, which is precisely what TabICL's documentation
predicts — pre-trained on 2–100 columns, "can degrade when going much beyond 100 features". The
edge is real at k=100 and gone by k=500.

Against the production baseline on the **primary endpoint** (CYP3A4 — board rank 137, the most
real labels at 2335, and almost no pseudo-label contamination):

| arm | CYP3A4 Δρ vs `lgbm_all` | seeds won |
|---|---|---|
| `lgbm_topk` (k=100) | −0.0241 | 0/3 |
| `tabicl_topk` (k=100) | **−0.0151** | **0/3** |

**The bar was +0.015 on ≥4 of 5 seeds. TabICL never beats the full-feature production GBM on
CYP3A4 at any budget.** Recorded and not pursued (`feedback-a-magnitude-bar-is-not-a-significance-bar`).

This matches the expectation pre-registered before the run — *"lgbm_all ≥ tabpfn_all > tabicl_topk
≈ lgbm_topk, and the most valuable outcome is the control"* — and it matches the sceptical
literature: Guan et al. (CSIRO) find that for small-molecule work with ECFP/RDKit descriptors "no
single predictor-representation pairing dominates", concluding tabular foundation models are
strong "**only when coupled with expressive representations**". A sparse fingerprint block is not
one.

## The control was the valuable part, and the single-seed reading had it backwards

| budget | `lgbm_topk − lgbm_all`, macro |
|---|---|
| k=100 | **−0.024** |
| k=1000 | **−0.0003** |

**1000 of 3091 features carry the problem essentially perfectly** (CYP1A2 +0.0040, CYP2C9 −0.0022,
CYP2D6 −0.0013, CYP3A4 −0.0019), while 100 features cost a real 0.024. So the matrix *is* largely
redundant — just not 30× redundant.

**And this is the honest correction to what I reported first.** From the k=100 seed-0 cell alone I
told Arnab "TabICL on 100 features beat the full 3091-feature LightGBM ensemble". That was true of
CYP1A2 and false of the other three, and the full sweep shows the whole effect was the feature
budget. The pre-registration's item 1 — *"if `lgbm_topk` loses more than 0.02 macro, the k-limited
arms are handicapped rather than tested"* — fired correctly at k=100 and then resolved itself at
k=1000, which is why sweeping the budget rather than assuming it was the right call.

## What this does and does not close

**Closes:** tabular foundation models *on raw descriptor tables* for this task. The literature and
our own measurement now agree.

**Does NOT close, and is the live successor:** the same models on an **expressive representation**.
This programme has varied the representation four times with LightGBM nailed down as the head, and
**never varied the predictor on a learned embedding** — which is the configuration an independent
benchmark (Walters, also an OpenADMET programme author) reports beating Morgan+LightGBM on 5 of 9
ExpansionRx endpoints and 6 of 6 Biogen endpoints at n ≈ 5.3k. `scripts/cyp_predictor_vs_representation.py`
is built for exactly that grid, with a replication control that can void it.

**And the cheaper lever this exposed, which is on the production learner:** a like-for-light
control on identical splits scores the production GBM at **0.549 macro** against the **0.643**
constant quoted in our own code — and the deficit is **CYP3A4 0.758 vs 0.765 (reproduces) while
CYP1A2, CYP2C9 and CYP2D6 are 0.07–0.14 lower**. So the production pipeline's extra supervision
(pseudo-labels, 20 auxiliary heads, the larger 6145-compound set) buys 0.07–0.14 ρ on exactly
three isoforms and nothing on the fourth. **Which ingredient does it is untraced and cheap to
trace, and that is a real lever.**
