# Finding 01 — the model class cannot close this gap, and the gap is not where we thought

Strand: mathematical model classes (Fable). Returned 2026-10-10. **Verified and corrected against
this programme's own prior measurements before acceptance** — two of its recommendations were
already refuted here, and the correction matters as much as the finding.

---

## 1. What the literature establishes, and I accept

**The entire model-class ladder, from a tuned GBM to the 2026 state of the art, is worth 4–8% of
absolute error on tasks of this exact shape.** Two independent protocols agree, which is what
makes it credible — not any single paper.

| MoleculeACE, 30 ChEMBL potency assays at our n | RMSE | vs GBM |
|---|---|---|
| MolFormer | 0.748 | +10.0% |
| **GBM** | **0.680** | — |
| CheMeleon + TabPFN | 0.641 | −5.7% |
| Monroe (81M-mol PM6 pretrain + TabPFNv3) | 0.629 | **−7.5%** |

Under a shift protocol **matched to ours** (Bemis–Murcko scaffold holdout *plus* exclusion of test
molecules with max Morgan Tanimoto > 0.60, 58 tasks × 5 seeds) the same ladder compresses to
**4.2%**, and several celebrated methods are *worse than tuned XGBoost*:

| | RMSE | vs XGBoost |
|---|---|---|
| best (CheMeleon+TabPFN+pair context) | 0.8849 | −4.2% |
| **XGBoost (tuned)** | **0.924** | — |
| Chemprop (tuned 2D GNN) | 1.046 | **+13.2%** |
| DeepDelta | 1.0588 | +14.6% |
| **iMoLD** (built for molecular distribution shift) | 1.0796 | **+16.8%** |

**This independently confirms three levers we had already closed** — domain-generalisation
objectives (iMoLD last of 11, losing 47/47 regression tasks; MoleOOD 11/11), graph neural networks,
and pairwise/delta learning. Our closures were right, and now they have outside corroboration.

## 2. Where I corrected the report

**Its headline recommendation was already measured and declined here.** It proposed external CYP
data, leading with "PubChem AID 1851 = 17,143 compounds × 5 isoforms", arguing that external data
expands *covariate support* in a way a learning-curve exponent cannot measure. The argument is
methodologically sound in general. It does not apply here:

* **AID 1851 is the Veith dataset**, which this programme already rebuilt as continuous potency
  (`cyp_veith_continuous.py`) and then **declined at 0% incremental coverage** against a
  pre-registered ≥25% bar. Coverage at Tanimoto 0.3: train 100.0%, Veith 87.6%, **incremental
  0.0%**.
* **The support-expansion mechanism it proposed is measured absent.** test→train similarity is
  median 0.581 / p90 0.702; test→Veith is median 0.368. Veith covers nothing the training set does
  not already cover.
* **The test set is already in-domain** — 100% coverage at Tanimoto 0.3 — so low overlap is not
  the binding constraint, which also retires the shift-based half of the argument.
* **Its leakage hypothesis does not fit our data**: exact InChIKey overlap with Veith is **0/750
  both ways**. (That does not rule out leakage on *other* entrants' side, which remains unfalsified
  and unfalsifiable from here.)

A ChEMBL-specific acquisition audit was also already done (2026-10-04) and found two distinct
leakage mechanisms with two different rules; the decision is recorded in `LICENSING.md`.

**What survives from its data argument:** ChEMBL quantitative CYP IC50s are *largely complementary
chemistry to Veith* by this programme's own note, and were never ablated to completion. That is a
narrower and more honest version of its claim.

## 3. The reframing this produced — the gap is concentrated, not uniform

Prompted by the ceiling result, I asked where our deficit actually lives. In σ\* units, against the
best entrant per isoform (`scripts/cyp_where_is_the_gap.py`):

| isoform | our ρ | best ρ | our σ\* | best σ\* | **cut needed** | our rank |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.7597 | 0.8561 | 0.910 | 0.718 | **21.1%** | 54 |
| CYP2C9 | 0.8084 | 0.8808 | 0.551 | 0.440 | **20.0%** | 92 |
| **CYP2D6** | 0.5004 | 0.5700 | 1.420 | 1.343 | **5.4%** | **28** |
| **CYP3A4** | 0.7685 | 0.8741 | 0.721 | 0.543 | **24.7%** | **134** |
| mean | | | 0.900 | 0.761 | 17.8% | |

**This inverts the target.** CYP2D6 has our *worst* ρ by far (0.50) and is the cell I would have
attacked first. It is in fact our *best* cell relative to the field: rank 28, needing only a 5.4%
cut, because **the whole field is bad at CYP2D6** — the best entrant manages only 0.5700. The
4–8% model-class ladder covers **73–100%** of that gap on its own.

**CYP3A4 is the real problem**: the widest gap (24.7%), our worst rank (134 of 271), and a
**100-place spread** against our own CYP2D6 cell. The macro average was hiding this completely.

Matching the ceiling against each gap:

| isoform | cut needed | fraction the best model class could close |
|---|---|---|
| CYP1A2 | 21.1% | 19–38% |
| CYP2C9 | 20.0% | 20–40% |
| CYP2D6 | 5.4% | **73–100%** |
| CYP3A4 | 24.7% | **16–32%** |

## 4. What this means, stated plainly

**The best available model class closes roughly a fifth to two-fifths of our deficit on three
isoforms, and most of it on the fourth.** It is worth doing — 4–8% is four to eight times any
lever this programme has previously found — but **it does not reach the leaders**, and no amount of
model-class work will. Something other than the model class produces the leaders' CYP3A4 0.8741.

The honest consequence for the programme: **CYP3A4 deserves a diagnosis of its own, not a better
model class.** It is the cell where we are mid-field, where the known mechanism is strongest
(heme-iron coordination, a flexible multi-site pocket, and IC50 that genuinely shifts 0.3–0.5 log
with probe substrate), and where our one physics attempt failed for a *diagnosed* reason rather
than a mysterious one.

## 5. Live candidates, re-ranked after the above

1. **Uncertainty-aware decode** — *my own ceiling result does not close this, and the report was
   right to say so.* My oracle ranged over maps of the **prediction alone**, `{f: p ↦ ŷ}`: +0.028
   in-sample, negative cross-fitted. A hedging rule is two-dimensional, `{f: (p, σ̂) ↦ ŷ}`, and no
   1-D map can express it. I scoped this correctly in the memory ("a ceiling over functions of x
   does not bound a method using information outside x") but then wrote "placement/decode:
   saturated" in the lever table, which overclaims. **Corrected.** Cheap, offline, 2–4 h.
   Caveat from the literature: scaffold splits break conformal exchangeability badly (0.34 coverage
   against an 0.80 target in one benchmark), so use **Mondrian conformal grouped by NN-Tanimoto
   band**. And note our validation (NN 0.439) is *harder* than the test set (0.587), so any
   similarity-blind calibration gives intervals that are systematically too wide.
2. **Censored/Tobit treatment of the 6,900 screen labels** — genuinely new, and we already own the
   data. A single-concentration screen reports a *threshold crossing*, so its implied pIC50 is
   censored. Fitting those as point labels teaches the model the screen's threshold rather than
   potency — which is precisely this programme's own "the selection is the defect" finding,
   arriving from a different direction. 4–10 h.
3. **CheMeleon (or equivalent) embedding → TabPFN** — the measured best of the model-class ladder,
   3–7%. Caveat worth pricing in: CheMeleon is pretrained to predict **Mordred descriptors**, and
   our GBM already consumes RDKit 2D descriptors, so part of the published gain is "the baseline
   was not given Mordred". 4–12 h.
4. **Multi-task across the 4 isoforms** — the only direct CYP multitask study found multitask
   *alone* gave no significant gain; the gain came from **imputing** the sparse compound×isoform
   matrix. Our 6,900 screen compounds are exactly that sparse 4-column matrix. Expect ~2%, and
   expect CYP2D6 (basic amines) to be the cell that degrades against CYP2C9 (acidic).

## 6. Provenance

The magnitude claim rests on two independent protocols, not one paper. But several sources are
2026 arXiv preprints and not peer-reviewed: MolPAIR (2609.38744), Monroe (2608.18982),
TabPFN-molecular (2604.16123), the structural-frontier study (2607.10729). Published: the Polaris
challenge analyses (JCIM), the CYP multitask study (J Cheminform 2025), MoleculeACE (JCIM 2022),
GAUCHE (NeurIPS 2023). **Unverified by us:** every number in §1 is as reported by the strand; I
have not reproduced any of it. The §3 table is our own measurement off the live board.
