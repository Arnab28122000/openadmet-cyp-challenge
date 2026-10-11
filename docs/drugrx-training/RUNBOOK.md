# Runbook — what is in flight and exactly how to finish it

Written 2026-10-09 19:55 UTC, updated 21:50 UTC. This file exists so that a context loss costs nothing: it states
the pending actions, their deadlines, and the exact commands, with the traps that have already
cost time on this programme. Delete a section once its action is done.

**All times are UTC.** The shell reports local time (IST, UTC+5:30), so the local date rolls
over at 18:30 UTC and has already caused one date confusion. Check with `date -u`.

---

## 1. Docking — DONE, FAILED its bar, sessions released (2026-10-09 21:39 UTC)

6,895/6,895 merged, 6,852 scored (99.4%), 43 errors (41 × 300 s timeout on very flexible
ligands, 2 embed failures). Matched Δρ **+0.0068 ± 0.0035** against a +0.01 bar → **FAIL**.
No stage 2, features not deployed, all three Colab sessions stopped. Full reasoning in
`PROGRESS.md`. Artifacts kept in `data/cyp_cache/dock/` (7.1 MB) — `dock_cyp3a4.jsonl`,
`desc3d.npz`, `ligands.tgz`, `smiles.json` + `manifest.json` (sha `18cc8cbc5e74abf3`) — so a
future metal-aware attempt reuses the conformers rather than regenerating them.

## 2. v9 — SUBMITTED 2026-10-11 12:40 UTC, awaiting its score

Accepted: `POST /gradio_api/queue/join` **200**, then the panel's *"Submission received from
'talindrew' for the Regression Prediction track."* **One click, one submission.**

**v8 scored and REGRESSED: macro 0.5256 → 0.5520, rank 54 → 69, all four cells worse.** Its row is
recorded in `cyp_board_history.py` as `V8_FULL`. Graded against its pre-registration, only the
ρ/τ invariant passed; the headline claim — *no centre cell may rise above its v7 value* — was
**falsified by all three**, and the CYP2D6 point forecast missed by +0.0294, outside its band.

**What that refuted is narrower than it looks.** ST-RAE is a sum of hinge terms, hence convex and
piecewise *linear* in both centre and spread. Convexity was never the failed assumption — the
claim's own proviso was *"provided we do not cross the balance point"*, and we crossed it on every
axis at once. **v8's real yield is the bracket:** a secant sign change on all four axes, which
puts each minimum inside a measured interval for the first time.

v9 is therefore the first placement submission that **interpolates** rather than extrapolates.
Pre-registered in `scripts/cyp_v9_bracket.py`, committed before the result existed:

| claim | content |
|---|---|
| ρ / τ | **0.7093 / 0.5285 unchanged** — 7th consecutive |
| CYP1A2 | centre 4.4679 → 4.5128, band [0.5307, **0.5405**] |
| CYP2C9 | centre 4.8436 → 4.9091, band [0.4377, **0.4618**] |
| CYP2D6 | spread 1.0007 → 1.0446, band [0.5447, **0.5543**] |
| CYP3A4 | centre 4.8916 → 4.8886, band [0.5510, **0.5516**] |
| macro | point estimate **0.5236**, rigorous band [0.5160, **0.5270**] |
| **the falsifiable one** | **every cell at or below its bolded CHORD bound, macro ≤ 0.5270** |

The chord bound follows from convexity plus three measured points — nothing assumed. If a cell
exceeds it, either the scorer is not deterministic or ST-RAE is not convex in that parameter, and
**both would be far bigger news than the placement**. Honest expected value, stated first: the
point estimate beats v7 by only +0.0020 and the rigorous worst case is +0.0015 against it. This
mainly recovers v8's loss; it is worth the slot because it also measures the vertex on all four
axes.

A poller is armed (`--wait-after "2026-10-10 17:21 UTC"`, 10 h) writing `data/cyp_cache/board_v9.json`.
**Record the row in `cyp_board_history.py` under `"v9"` before the next submission overwrites it.**

### The submission path, corrected twice now — read this before clicking anything

- **`POST /gradio_api/queue/join` is the receipt.** It can show `statusCode: pending` for well over
  a minute before settling to 200 — that pending line is itself positive proof the click fired.
  The Submission Status panel takes ~90 s and `GET /gradio_api/queue/data` **503s routinely** on
  every submission, which is why the panel can stay blank long past its usual delay.
- **`read_network_requests` LAGS. An absent POST is NOT evidence the click failed.** On v8 three
  independent signals said the first click had not fired — no `queue/join`, no status heading, an
  idle button — and all three were wrong. Only the rate limiter made the retry harmless. On v9 the
  same absence appeared at 50 s and the POST was simply in flight. **Wait and re-read; never
  re-click on a negative read.**
- **The 12 h window is enforced to the SECOND from the board's timestamp, not your click.** v7 was
  clicked 05:15 and recorded 05:20:39, so a trigger armed for 17:15 was rejected 51 s early. The
  `:SS` appears only in the rejection message, so arm the next attempt at `board HH:MM + 12 h + 2 min`.
  A rejection is free: no slot consumed, previous row intact.
- **The in-page nav is position-dependent and has a duplicate.** `find` returned a `✉️ Submit`
  button whose click did nothing; the one that works is the in-page nav strip at **(283, 486)**
  *with the page scrolled to the top*. `cmd+Up` does not scroll this page — it is a scroll
  container, so scroll up with wheel ticks. Then scroll **down ~26 ticks** to the form; its inputs
  do not exist in the accessibility tree until they are in view.
- Identity: HF username `arnab2812`, alias `talindrew`, track **Regression Prediction**, report
  link `https://github.com/Arnab28122000/openadmet-cyp-challenge`. Leave *Submit anonymously*
  **checked**; *proprietary data*, *open-source* and *future publication* **unchecked**.
- The file input is a **child** of the upload button — `file_upload` on the button's own ref fails.
  Use `find` for "file input element inside the upload area".
- **Write seven decimals.** Four collapses 10–20 distinct predictions per column into ties and six
  still tied one CYP1A2 pair (1−τ = 1.8e-6). Seven is the first precision that is exactly
  rank-identical to the v4 source, which is what the ρ/τ invariant actually claims.

**Do not drive the UI to read the score.** `cyp_board_fetch.py` reads the whole board over HTTP and
`--wait-after` prints the board-wide newest timestamp every poll — the number that separates a
**stalled scorer** from a lost submission. The queue ran ~7 h behind on v7 and froze for over two
hours at 15:25 UTC during v8; v8 took **6 h 10 m** to score. Never resubmit on a slow board.

**Not a fault, so do not chase it:** we have no row on the ⏱️ Interim Leaderboard. That is the
September 25th snapshot; our first submission was October 8th.

## 3. Standing facts that are easy to get wrong

- **`colab exec` is serial and has a client-side reply timeout.** Long jobs must be launched
  detached via `dock_launch.py`, which now refuses a `CMD` containing a shell chain (`nohup cd`
  cannot work — `cd` is a builtin) and verifies the pid survives 5 seconds.
- **A keepalive's quiet log is not evidence it stopped.** It prints every 20th beat and a
  saturated VM slows each beat well past its nominal interval. Judge liveness by rows arriving.
- **macOS has no `pgrep -c`** and no `timeout`.
- Commits use `Arnab Bhattacharya <53032587+Arnab28122000@users.noreply.github.com>` and carry
  **no Claude attribution**.
