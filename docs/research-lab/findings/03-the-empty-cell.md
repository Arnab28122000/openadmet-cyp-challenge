# Finding 03 — the empty cell of a 2×2, and the board is noisier than its ordering suggests

Strand: ADMET/CYP state of the art (Opus). It independently re-derived the target correction
(reaching ~12.5% where I got 13.6% — same conclusion, slightly different averaging), **ran its own
measurements on the release**, and caught a contradiction in our durable docs.

## 1. The board's top-20 is a lottery

The live board **scores only half the test set** (n = 375/isoform). Macro SE ≈ **0.0131**
(≈0.0092 on the full 750). The entire top-20 macro ρ spans **0.0296** — about **2.3 SE**.

So the ordering above us is substantially noise, and a de-noised frontier is ≈0.775, i.e. a
**~11% cut** rather than 13.6%. This does not change what to build; it changes how much to read
into rank movements, including our own.

Also measured: `corr(rank, ρ) = −0.797` against `corr(rank, ST-RAE) = +0.561`. **Placement is
near-exhausted** — our ST-RAE 0.5256 is already 2nd-best among the 39 entries within ±0.012 ρ of
us, and the best in that band would move us 54 → 41. **~13 places left in placement, total.**
That is an independent confirmation of our own saturation result, and it prices what v8 can buy.

## 2. The empty cell

The programme has filled three quarters of a 2×2 and never tested the fourth:

| | descriptors / fingerprints | frozen FM embedding |
|---|---|---|
| **GBM / our net** | baseline | tested → **0.0000** (CheMeleon, closed) |
| **in-context tabular** | tested → **null** (−0.0032 ± 0.0051) | **NEVER TESTED** |

Every source points at that one cell, including two entries **ranked above us in this challenge**:

* **Rank 53** (`jeremy`, ρ 0.7343) ensembles LightGBM-on-ECFP4+RDKit2D with **TabPFN on frozen
  chemprop_chemeleon**, and states plainly that what did *not* work was "TabPFN/TabICL on classical
  descriptors — too correlated with LightGBM". **That is our null reproduced, and the fix named.**
* **Rank 37** (`stir_bar`) likewise uses Mitra/TabICL over frozen embeddings as stack members.
* Monroe's own table, against our exact baseline: tuned LightGBM on count-ECFP4+RDKit2D **0.680**;
  CheMeleon alone 0.662 (−2.6%, ≈ our measured null); **CheMeleon+TabPFN 0.641 (−5.7%)**;
  Monroe+TabPFN **0.629 (−7.5%)**. MolFormer 0.748 — *worse than our GBM*.
* Our own `icl_top` arm already **beat xgb on 3 of 4 endpoints** (+0.0062/+0.0101/+0.0116, losing
  only 2D6) at half the across-split variance — on classical features.

**The honest counter-evidence, which the strand raised against its own recommendation:** Özçelik &
Grisoni 2025 find that in regression with *structurally similar* test molecules neither deep models
nor XGBoost is superior, and deep models win only on dissimilar sets (Tanimoto < 0.4). **This blind
set's median NN Tanimoto is 0.587 with 96% ≥ 0.5** — precisely the regime where the literature says
foundation models have no edge. Weigh that against the in-challenge evidence, which is from this
exact test set and points the other way.

## 3. Measurements it ran (not literature — actual numbers on our release)

- **Read-across is dead, measured.** Design-matched holdout (61 series, 429 held-out compounds):
  GBM ρ **0.7632**, pure read-across ρ **0.3438**, optimal blend weight on the anchor **exactly
  0.00**. Its own random-effects ceiling predicted ρ ≈ 0.62, so the model-implied ICC was optimistic
  by ~40%. This closes the ENDGAME Phase 4 anchored/delta line.
- **A calibrated cheap ruler for CYP3A4.** That same GBM scores ρ 0.7599–0.7632 against our board
  CYP3A4 of **0.7685** — within **0.009**. CYP3A4 is our weakest cell (rank 134/271, +0.106 to
  best), and we now have a design-matched offline surface for it.
- **Label-quality weighting priced and rejected:** `w = 1/(0.3+CI)` is **+0.0070 ± 0.0029, 6/6
  seeds** = 1.0% σ\*. **Dropping** wide-CI labels is decisively harmful: **−0.0480** (CI>2) and
  **−0.0727** (CI>1), 0/6 seeds. CYP3A4 carries 29.4% of labels with CI>1 and they are load-bearing
  — corroborating our own "censored compounds rank better".
- **The test set's anchor structure, from the data:** all 750 blinded compounds have their ECFP4
  nearest neighbour among just **132 distinct training compounds**, consistent with a
  "25 hits × 3 isoforms × 10 analogs" design. Anchor label coverage is only ~50%, so any
  anchor-based estimator is capped at half the set before it starts.

## 4. The contradiction it caught in our own docs — verified and fixed

`PROGRESS.md` records **`depth @ 3000 ep = −0.0234 ± 0.0048, t = −4.92, 5/5 seeds, DECIDED`**,
while both `AGENTS.md` lever tables read "capacity spent, +0.0019 ρ, p = 0.723".

Verified: the arm values there (0.7055 → 0.6764) are **ST-RAE**, not ρ. Both statements are true
and the table was hiding the interesting half — **capacity bought placement, not ranking**, and
only as an *interaction*: depth is −0.0085 (noise) at 400 epochs and the long schedule −0.0057
(noise) on L3; neither factor alone carries it. As written, the table would have stopped someone
from using depth + a long schedule. **Both tables corrected.**

## 5. Where this leaves external data

The strand argues our "data closed" verdict is right about in-distribution labels and over-broad
about **external multi-task heads on a shared encoder**: Inductive Bio's Polaris post-mortem
reports **8 of the top 10 used additional ADMET data, and the best model without it had 23% higher
error**. Both informative public in-challenge reports (ranks 15 and 53) attribute their position to
external CYP labels as **separate heads, frozen, never pooled rows, never cross-assay calibration**.

Our counter-measurements (Veith bounds as rows: worse 3/3; the 20 release heads: +0.0010;
Veith declined at 0% incremental coverage) were all made on our **production model class**. The
specific untested experiment is a **from-scratch Chemprop multitask D-MPNN on challenge + AID 1851
DRCs + ChEMBL CYP potencies as 13–18 separate heads**. I am not calling that closed.
