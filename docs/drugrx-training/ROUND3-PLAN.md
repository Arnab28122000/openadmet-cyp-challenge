# DrugRx CYP — round 3: the selection-corrected round (opened 2026-10-08)

**Read `PROGRESS.md` (newest last) and `AGENTS.md` first.** This file supersedes the phase list in
`ENDGAME-PLAN.md` for the remaining window (challenge closes **3 Nov 2026 23:59 UTC**, one
submission per 12 h per track; the board shows the *latest* submission, not the best).

## Why a new round

On 2026-10-08 the single-concentration file was read against the DRC labels, and the programme's
standing belief — *"every label we hold is promoted"* (PROGRESS, 2026-10-06) — turned out to be
false. The 50 µM screen covers **4,376 compounds on all four isoforms**, and a dose-response curve
exists only for compounds promoted out of it:

| isoform | DRC'd | non-promoted screen compounds | promotion rule (measured) |
|---|---|---|---|
| CYP1A2 | 1,412 | 2,963 | rises with inhibition (100% below log2FC −1.5) |
| CYP2C9 | 1,285 | 3,090 | ~1/3, flat in log2FC — random |
| CYP2D6 | 1,493 | 2,882 | **hard threshold**: DRC iff log2FC < ~−0.6; others at median +0.38 |
| CYP3A4 | 2,335 | 2,570 | enriched for *weak* readouts; non-promoted are mostly strong inhibitors |

The two isoforms with activity-dependent promotion are exactly the two we score worst on (board:
1A2 0.88, 2D6 1.46 vs 2C9 0.52, 3A4 0.57). The per-isoform leaders show the same signature from the
other side: on CYP2D6 they reach ST-RAE ~0.41 with Spearman only ~0.54 but R² ~0.48 — their
*ranking* is barely better than ours (ρ 0.33), their *level* is right.

## Standing rules (inherited, still binding)

Effect size over noise band before any GPU; a negative is only as strong as the range tested;
a flagged unknown blocks downstream decisions; never pool across a metric change (`metric_id`);
pool only comparable members; compare validation to the evaluation set before tuning on it.
**New this round:** every validation number is reported on **V1** (DRC-only, the old surface)
*and* **V2** (V1 + held-out screen inactives with the floor window) — a selection fix is
expected to be invisible on V1 by construction.

## Phases (expanded one at a time; each lands a PROGRESS section and a CHANGELOG row)

### Phase A — research sweep *(running in parallel, background agents)*
`research/20`–`24`: selection bias / censoring / classification; foundation models + LLMs;
past-challenge winners + CYP challenge intel; analog-series local models; the live board's
published model reports (briford rank 12, rasayan-labs 16, stir_bar 33, datagrok 40, …).

### Phase B — screen-derived labels (the selection fix) *(in progress)*
`drugrx/cyp/screen.py`: every screened compound without a DRC on an isoform gets an **interval
label** calibrated on compounds that carry both — floor window `[1.07, 3.52]` for no inhibition at
50 µM, the bin's [q10, q90] of DRC pIC50 otherwise. B1: LightGBM, 3 arms × 3 scaffold seeds, V1/V2
(`scripts/cyp_screen_lgbm.py`). B2: the incumbent net (`cyp_net_one.py`) with the pseudo-rows in
its interval loss, on Colab. **Expected effect:** CYP2D6 V2 large (the model has never seen a
2D6 inactive); macro on the board ≥ −0.11 if 2D6 merely reaches the mean predictor.

### Phase C — classification / hurdle
P(active at 50 µM) from a classifier on all 4,376 screened compounds (+ PubChem AID 1851 actives/
inactives), potency among actives from the regressor, combined by the metric-optimal point
(weighted median of the interval-loss risk), per isoform. Also: the **TDI classification track**
exists on the board and we have never entered it.

### Phase D — representations: foundation models, LLMs, tabular FMs
Ranked by Phase A's evidence (CheMeleon, CLAMP, MiniMol, TxGemma embeddings, TabPFN v2 …).
Scored on V1+V2; kept only if it beats the fingerprint arm at the same data.

### Phase E — analog-series local correction
The test set is 75 series × 10 analogs of training hits. kNN/GP residual correction anchored on
parent hits, validated on a hit-expansion re-split (bounded: our library cannot reach the test's
0.665 within-set similarity — state the range).

### Phase F — build, sanity-check, submit, read the board
One submission per 12 h. Pre-registered expectation written down *before* each submission; the
board read per isoform via the browser (alias `talindrew`). Model report link + open-code flag set
with the final submission.

### Phase G — record
PROGRESS / CHANGELOG / AGENTS.md / HYPOTHESES / memory updated at the end of every phase, not at
the end of the round.
