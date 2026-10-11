# Provenance of every number this programme can quote

The write-up is itself an interpretation artifact, and seven defects have now lived in
that layer — including a source package that was never in version control. So before
anything is published, this is the inventory of what resolves to a stored run and what
does not. **`PROGRESS.md` is the log; this is the audit of it.** A figure that appears
only in a log entry whose artifacts are gone is labelled as such, here and in the
write-up.

Three categories, and the middle one is the trap.

## A. Quotable directly: stored, stamped, scored under the current metric

| artifact | what | provenance |
|---|---|---|
| `talindrew-drugrx/data/cyp_cache/cyp_loss/*.{json,npz}` | 20 runs: interval vs point, 10 seeds | `metric_id: official-strae-v1`, `batch_size_final: 4096`, full config, val+test predictions |
| `talindrew-drugrx/data/cyp_cache/cyp_aux/*.{json,npz}` | 16 runs: none/veith/chembl/both, 4 seeds | same |
| `talindrew-drugrx/data/cyp_cache/*.parquet` | the rebuilt Veith + ChEMBL + merged tables | reproduce the lost run exactly: 9,045 / 25,195 / 93 |
| `tests/test_official_metric.py` | the metric itself | pins our implementation to a verbatim transcription of the organisers' function, 40 random cases at 1e-12 |
| `tests/test_oracle_note.py` | every figure in the oracle note | re-derives from the `.npz` files, not from a log |
| `tests/test_benchmark_group.py` | `ADMET_TASKS` is 21 endpoints | pins the denominator of every "/21" macro |

All 36 prediction files share **one** validation split (verified by hash), which is
what makes every paired statistic in the write-up valid.

## B. Stored but scored under the WRONG metric — must be re-derived, never quoted

`runs/cyp/*.json` predate the denominator fix of 2026-10-05 and carry **no
`metric_id`**. They are kept because the log entries that cite them must stay
reproducible, not because their numbers are usable.

The clearest single illustration in the programme sits here: `runs/cyp/report.json`
records the submitted model's validation at **0.4346**, and the leaderboard scored
that same model **0.8752**. Per endpoint it recorded 0.498 / 0.340 / 0.631 / 0.271.

**None of those four numbers may appear in the write-up as scores.** The correction is
**not uniform per endpoint** (ratios 0.54–0.67), so they cannot be rescaled by a
constant either — they have to be recomputed or dropped. Specifically affected:
`arms.json`, `bakeoff.json`, `basicity.json`, `ceiling_nested.json`,
`external_result.json`, `optimise_report.json`, `replicate_combined.json`,
`report.json`, and both `submission*.csv`.

## C. Stored, unstamped, but on a different metric family — quotable with care

`runs/{baseline,h3,h3b,h4,h5,h6,h7a,acquisition,eval_mode,routing}/` are the
21-endpoint TDC ADMET work, scored per endpoint in TDC's own metrics (PR-AUC,
Spearman, MAE), not in ST-RAE. The denominator bug does not touch them. They carry no
`metric_id` because none existed when they were written; the stamp was added on
2026-10-05 (`drugrx/cyp/metric.py`).

Quotable, with the caveat that each endpoint's metric must be named alongside its
value — a bare number across these endpoints is meaningless because the metrics are
not interchangeable.

## D. Gone: numbers with no artifact at all

| number | why it is gone |
|---|---|
| the original 4-seed loss ablation (`point vs interval −0.0121 ± 0.0052, t=−2.32, 0/4`) | four Colab sessions died together; the driver had no per-seed push. **Retired** — it was an overestimate, and the n=10 rerun gave +0.0078 ± 0.0054 |
| the first 10-seed loss ablation and the first 4-arm aux ablation | same crash, plus a second session death where `HF_TOKEN` was absent so the new push could not fire |
| whatever differed between the original run and the reruns | reruns reproduce each other to ~1e-4 but sit ~0.013 from the original. Mirror drift, the `cyp_net_one.py` change and the external tables are all ruled out; the original's config is unrecoverable |

**These may be cited as history, never as evidence.** The write-up says "an earlier run
reported X; its artifacts are lost and the rerun reports Y" rather than averaging or
preferring either.

## The rule for the write-up

Every quantitative claim carries one of: a path under category A, a re-derivation
performed now from category A, or an explicit "artifacts lost" label. Nothing is
quoted from a conversation, a terminal transcript, or a `PROGRESS.md` entry alone.

Two consequences worth stating up front, because they shape the document:

- **The headline leaderboard figures (rank 163, MA-ST-RAE 0.8752, leader 0.3772) are
  external observations**, not our measurements, and are labelled as such. They are
  the only numbers in the programme that were not produced by our own code.
- **The honest validation figure is 0.737**, from the re-measurement under the
  corrected metric — not the 0.4346 in `report.json` and not the 0.44 quoted
  throughout the earlier log.
