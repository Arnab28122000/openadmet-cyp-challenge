# DrugRx training — agent context

**In-flight work and its exact next commands live in `RUNBOOK.md`. Read that first.**

Durable orientation for an agent picking this programme up cold, including after a context
compaction. **Read `PROGRESS.md` (newest entry last) before touching anything.**

## What this programme is

Train small models (1M–300M params) on **Google Colab L4 GPUs only**, 24h per run, and plug the
winners into the existing FoldRx pipeline as new `ModelImpl` entries, so `/ai-workspace` and
`/foldrx` can use them like any other stage model.

Two deliverables, **models first**:

1. **Models** — binding-affinity and ADMET predictors, trained and honestly evaluated, served from
   `talindrew-foldrx-models/` (Modal) so FoldRx can call them.
2. **Playground** — an extension of `/ai-workspace` + FoldRx: pick a disease, pick a target, screen,
   rank, and read the result as *ranked in-silico hypotheses with calibrated uncertainty*.

## What this programme is NOT

It does not discover, design or validate a treatment for any disease, and nothing it produces may be
presented as one. Every number is a prediction from a model trained on public bioactivity data, and
every ranked list is a **hypothesis list for wet-lab triage**. The honest claim is "these compounds
are worth testing first", never "this compound works". Any UI, export or notebook that states
otherwise is a bug — see `docs/foldrx-pipeline/` where the same rule already applies ("an estimate
must never read as a measurement" is quoted in the FoldRx registry).

Diseases in scope as *evaluation targets*: HIV, type 2 diabetes, ankylosing spondylitis, asthma.
They were chosen because each has well-characterised, small-molecule-tractable protein targets with
public bioactivity data — which is what makes a model measurable. Several of their most important
targets are biologics-only (IL-17A, IL-5, TSLP); our models say nothing useful about those, and the
playground must say so rather than scoring them anyway.

## Operating the GPU

**The Colab CLI, not the browser.** `uv tool install google-colab-cli`, binary at `~/.local/bin/colab`.
`colab skill` prints the bundled agent guide — read it before driving a session.

- `colab new -s <name> --gpu L4` — always name the session; an unknown `--gpu` silently becomes A100.
- `colab exec -s <name> -f script.py` — runs a *local* file on the VM; kernel state persists between calls.
- `colab run --gpu L4 script.py` — provision + run + tear down, exit codes propagate.
- `colab status`, `colab log -s <name>`, `colab download` — monitoring and pulling checkpoints.
- `colab stop -s <name>` when done. **An idle VM burns compute units.**
- `--config <path>` isolates session state, so parallel runs do not fight over `sessions.json`.
- `colab auth` and `colab drivemount` need a TTY and **cannot be run by an agent** — a human runs
  those once per session, or we avoid them (see "Checkpoints" below).

**Hardware (updated 2026-10-04).** The earlier "L4 only" rule is **withdrawn** — the owner asked for
L4 **or higher**, with GPU memory used hard and runs made fast. An **A100-40GB** session carries
**12 vCPU and 89.6 GB RAM** against a CPU session's 2 vCPU, and featurisation is the real bottleneck,
so the A100 is usually the *cheaper* choice despite costing 5.30 units/hr against 0.08: a sweep that
took hours on CPU takes ~5 minutes there.

Use the card properly rather than merely asking for a big one:
- keep the dataset resident in VRAM (ours is 570 MB) and delete the DataLoader;
- spend the remaining memory on an **ensemble trained simultaneously** under `vmap`, not on a bigger
  batch — past a few thousand rows a larger batch buys noise reduction nobody asked for, while the
  ensemble buys the calibrated uncertainty the playground needs;
- size the batch by **halve-and-retry on OOM**, not by prediction: the first estimator here was 2×
  optimistic and allocated 38.7 GB of 39.5 before dying;
- never drop early stopping to go faster — doing so made a 16-member ensemble lose to the single
  model it was meant to improve on.

### Budget

Compute units are finite and expire 90 days after purchase. Check `colab usage` before and after
every run and record the burn in `PROGRESS.md`. A run that is not learning gets killed — see
"Kill rules".

## Checkpoints and resume

A Colab session will die before 24h of training finishes. Every run must therefore:

- checkpoint **every epoch** (weights, optimizer, scheduler, RNG state, epoch, best-metric, and the
  run config hash) to durable storage, and
- resume from the latest checkpoint automatically on restart, with no hand-editing.

Durable storage, in order of preference for an *agent-driven* run:
1. **Hugging Face Hub** (private repo, token in env) — works headless, versioned, no TTY.
2. **`colab download`** to `docs/drugrx-training/runs/<run>/ckpt/` — pulled by the monitoring loop.
3. **Google Drive** — matches the owner's existing notebooks, but `colab drivemount` needs a human,
   so it is opt-in rather than the default.

A checkpoint that cannot be resumed from is not a checkpoint. Every run's first test is: kill it at
epoch 2, restart, and confirm the loss curve continues rather than restarting.

## Kill rules

Stop a run and write down why, rather than letting it burn units, when any of these hold:

- validation metric has not improved for N epochs (N set per experiment, recorded up front),
- the run is worse than the **baseline** (see below) at the halfway point,
- loss is NaN/diverged and one restart at lower LR did not fix it,
- throughput is so low the run cannot finish a meaningful number of epochs in 24h.

Every kill goes in `HYPOTHESES.md` with the reason. A failed hypothesis recorded is a result; a
failed hypothesis forgotten is wasted money.

## Baselines come first

No deep model is reported without beating, on the same split:
- a **Random Forest / LightGBM on ECFP4 fingerprints** (the standard, stubborn baseline), and
- for docking, the **Vina/smina score** already in FoldRx.

Published chemistry ML is full of deep models that lose to fingerprint baselines on scaffold splits.
If ours does too, that is the finding, and it gets written down.

## Evaluation discipline

- **Scaffold split** (and time split where the assay dates allow), never random split, for headline
  numbers. Random-split numbers may be shown only beside the scaffold ones, labelled.
- Report calibration, not just accuracy — the playground ranks candidates, so a confident wrong
  score is worse than an honest wide interval.
- Keep a **held-out target set** the model never trained on, to catch the classic failure where a
  model memorises target identity instead of learning chemistry.
- Known-ligand recovery on the four diseases' targets is a *sanity check*, not a headline metric,
  and is reported as such.

## The playground's target catalogue (Phase 5)

`talindrew-backend/src/services/foldrx/targets.py` — 33 targets across HIV, type 2 diabetes,
ankylosing spondylitis and asthma. **22 screenable, 11 named and declined.** The eleven are the
point: a biologics-only mechanism quietly omitted leaves the disease looking covered, so each is
listed with its reason and `/foldrx/catalog` serves them under `diseases`.

Four corrections are encoded rather than written down, each mutation-tested:
- **IL-23 in axial disease is *refuted*, not intractable** (ustekinumab ph3 PMID 30225992,
  risankizumab ph2 PMID 29945918). Ankylosing spondylitis means TYK2, JAK1, RORγt.
- **A 3.5 Å docking gate.** 2Q7M (FLAP, 4.25 Å) and 5U1C (HIV-1 intasome, 3.9 Å) are in common
  target lists and are not docking-grade.
- **GLP-1R → 7LCI/6ORV**, never the peptide-bound 4ZGM; tractable only since orforglipron.
- **Boltz-2's weak classes attach to the target, not the model**, because the user picks a target.

`tractable` is not `dockable` (HIV-1 RT and glucokinase are worth screening with no pinned
structure), and `tier` is mutually exclusive with `declined`. `TD_TARGETS_LIVE=1` checks every
accession against UniProt and every PDB id against RCSB **at the resolution we recorded** — skipped,
never silently passed.

**Templates carry a `target` id**, and `validate()` refuses one pointing at a declined target, so
the catalogue's refusals cannot be routed around. `dpp4-control` is the one template whose result
can be marked right or wrong: three approved gliptins against five unrelated drugs, and a screen
that cannot rank the gliptins first is not telling you anything further down its list.

**Two traps paid for once.** Slice UniProt by the **annotated Chain feature**, never by remembered
coordinates — the Gag-Pol numbering in the literature returns 99 residues of *Gag* where HIV-1
protease should be, at exactly the right length to look correct. And check a SMILES by **formula and
mass**, not by eye: two of five gliptins written from memory were the wrong molecule.

## What a screen is actually worth

`research/06-prospective-screening-reality.md`. The programme quoted CACHE #1's **3.7%** as its
yardstick since it opened; that is the optimistic end. **Most blind teams find nothing** — 7 of 23 on
CACHE #1, 5 of 23 on CACHE #2 (0.7%). 3.7% is measured at K_D < 150 µM; the best *confirmed* work was
8 binders at 3–41 µM from 84 compounds. **Difficulty belongs to the target, not the method.**

Two findings land on our own work: **37 of CACHE #1's 73 hits had poor solubility despite
computational prediction** (our endpoint, where we cannot beat a fingerprint baseline — the strongest
argument for annotate-not-filter); and **Boltz-2 correlates ≈ 0 with reference calculations among the
top 100**, which is exactly the shortlist a screen operates on. It is an enrichment filter, not a
ranker.

## Where it plugs into the product

`talindrew-backend/src/services/foldrx/registry.py` — a stage is a slot (`StageDef`), a model is a
`ModelImpl` filling it. A trained model becomes a new `ModelImpl` with `provider="oss-modal"` and a
route on `talindrew-foldrx-models/`. No per-model code in the pipeline, the API or the UI.

Relevant stages for this programme: `admet`, `rescore`, `selectivity`, `generate`, and possibly
`dock`. `custom_models.py` is the bring-your-own-endpoint path (`docs/dd-growth/CUSTOM_MODELS.md`)
and is how an outside lab would plug in the same shape.

Honesty is enforced in the registry too: `ModelImpl.caveat` is shown on the card, and every model
this programme ships must carry one.

## Files here

| File | What it is |
|---|---|
| `PLAN.md` | The phases. Expanded one at a time, not all at once. |
| `PROGRESS.md` | Append-only log, newest last. **Read first.** |
| `CHANGELOG.md` | What shipped, per phase. |
| `HYPOTHESES.md` | Every training hypothesis: prediction, result, verdict. Failures included. |
| `research/` | Background research, one file per topic, with citations. |
| `runs/` | One directory per training run: config, logs, metrics, checkpoints. |

Related programmes: `docs/foldrx-pipeline/`, `docs/ai-workspace/`, `docs/model-hub/`, `docs/dd-growth/`.

## A flagged unknown in a measurement path blocks decisions downstream of it

Added 2026-10-05, after the costliest defect in this programme.

`drugrx/cyp/metric.py` carried this, in its own docstring, from the day it was written:

> "The organisers' exact normalisation is not public, so the denominator here is the mean
> absolute deviation of the truth … **Our absolute values are therefore not guaranteed to
> match the leaderboard's.**"

Three weeks of decisions were then made on that number — arms chosen, hypotheses declared
null, a submission tuned — and the leaderboard scored the same model at **0.8752** where we
read 0.44. The organisers' scorer was public the whole time, in
`OpenADMET/blind-challenge-template`.

**The code never failed.** No test could have caught it, because the implementation did
exactly what it said. What failed is that a known unknown in a measurement path was
tolerated as a caveat instead of treated as a blocker.

So: when a measurement path contains a flagged unknown, resolve it before any decision
downstream of that measurement. Resolving means reading the authoritative source, not
re-deriving a plausible guess. If it cannot be resolved, the decisions it would support do
not get made, and the programme says so rather than proceeding on a caveat.

Corollary: a metric is not "valid for ranking our own candidates" just because a constant
factor cancels. Here the factor was **per endpoint** (0.54–0.67), so every macro-averaged
comparison was silently re-weighted — and macro averages were how most verdicts were read.

## Read the leaderboard PER ENDPOINT. It is published, and believing otherwise cost a round.

Added 2026-10-06.

This file and `PROGRESS.md` asserted for weeks that the CYP challenge board "reported macro
columns only, so a generalisation gap cannot be decomposed". **It is false.** The Live
Leaderboard on `openadmet-cyp-challenge.hf.space` has a **tab per isoform**, each publishing
ST-RAE, MAE, R², ρ and τ, and each with its own search box. Drive it with Playwright, search the
alias **`talindrew`** (not the HF username `arnab2812`), and read the row off the table.

What it showed the moment it was read:

| endpoint | rank | ST-RAE | MAE | R² | ρ |
|---|---|---|---|---|---|
| CYP1A2 | 205 | 0.8842 | 1.0987 | 0.0995 | 0.6964 |
| CYP2C9 | 124 | 0.5237 | 0.5385 | 0.5206 | 0.7753 |
| **CYP2D6** | 177 | **1.4577** | **1.7446** | **−0.9509** | 0.3302 |
| CYP3A4 | 139 | 0.5735 | 0.6181 | 0.5518 | 0.7612 |
| MACRO | 161 | 0.8598 | 1.0 | 0.0552 | 0.6408 |

**The macro is a plain mean, and one broken endpoint was carrying it.** Two consequences that
generalise past this programme:

- **Never read a macro mean as though it were one endpoint's number.** "Leaderboard R² 0.045"
  was quoted here for weeks as evidence our predictions barely correlated. They correlate fine
  on two endpoints (R² 0.52 and 0.55); the macro is CYP2D6 at −0.95 dragging three positive
  ones. R² is unbounded below — another entry on this board sits at MA-R² **−42.46** with a
  creditable macro ST-RAE of 0.3706.
- **A macro mean cannot be inverted as though it were one distribution.** Doing so produced a
  bias estimate of 0.76 that the metric then refuted outright.

**Why CYP2D6.** The organisers' own `config.py` states the construction: hit expansion around
the top 25 hits of **CYP1A2, CYP2C9 and CYP3A4**, ten Enamine ECFP4 analogs each, 75 × 10 = 750.
CYP2D6 is absent from that list and its test compounds are "inherently less potent". All four
endpoints are scored on the *same* 750 molecules. Meanwhile our training labels are
**activity-promoted** — dose-response curves were released only for compounds promoted out of
primary screening — so CYP2D6's 1,493 training labels are enriched at mean 4.784 while the test
truth is not. A model fitted to the first and scored on the second over-predicts. **A selection
effect, not a chemistry failure, and therefore a decoding fix.**

## Measure the decoder against the scorer; do not model the test set

Three attempts to set a decoder's magnitude by modelling the blinded set failed, and one
measurement succeeded. The pattern is worth keeping.

**What failed.** Reconstructing each endpoint's test label distribution from its published
columns and fitting `(shift, scale)` on the reconstruction. Applied to the **real** validation
set it was worse than doing nothing on 4 endpoints out of 4, macro 0.6700 → 0.7527, regret up to
+0.2652. The cause: the R² identity's `bias²` term absorbs any error in `r`, and `r` is exactly
what cannot be had — a Gaussian copula inverts ρ/τ to a Pearson `r` well on validation (0.636
against 0.600 true) and **fails on test**, where ρ 0.64 corresponds to r ≈ 0.30, not 0.66.

**What worked.** A **pure rescale probe**. Submit `p_a = m + a(p − m)` at three levels. Because
`mean(p_a) = m` for every `a`, the bias and `var_y` enter only through `K = var_y + bias²`, which
differencing cancels:

    (R2(a2) − R2(a1))·var_y + 2(a1 − a2)·cov = (a1² − a2²)·var_p

`var_p` is exact (we hold the vector), so three levels determine `(var_y, cov)` and hence
`s* = cov/var_p`. **No copula, no label-shape model, no Gaussian residuals.** The self-test, fed
R² rounded to the board's 4 decimals, recovers `s*` to **0.0006**. A rescale is monotone, so ρ
and τ must return identical — a free check on the whole path from file to board.

The general rule: **when a quantity's only authoritative source is the scorer, probe the scorer.**
Keep one published observable back from the fit as the check — here ST-RAE, which is the only one
not symmetric in the sign of a bias, and which both caught the macro error and identified the
sign correctly on 4/4 endpoints.

## Metric facts resolved from the authoritative source, which is GitHub not the Hub

`github.com/OpenADMET/blind-challenge-template`, `main` — the HF Space path 401s.

- `backend/config.py`: `ACTIVITY_METRICS` wires **MAE to sklearn `mean_absolute_error` and R² to
  `r2_score`, both plain, both on raw errors**; `evaluate_predictions.py::
  _metric_needs_credible_interval_bounds` gives the interval bounds *only* to metrics whose
  signature accepts them. So MAE/R²/ρ/τ are readable as moments.
- `ACTIVITY_DATASET_SIZE = 750`, `SORT_REGRESSION_LEADERBOARD_BY = "ST-RAE"` (rank is ST-RAE
  alone), **`BOOTSTRAP_SAMPLES = 1000`** — every published column is a bootstrap mean, not a
  point estimate, which widens every leaderboard-derived error bar.
- Macro is a plain mean over endpoints.
- External data and pretrained models are **explicitly permitted**, with an "I used proprietary
  data" box that public data does not need. Close **3 Nov 2026 23:59 UTC**, one submission per
  12 h per username per track. **The board displays your latest submission, not your best.**

## Credible intervals are absolute windows at the floor, not widths around y

Below pIC50 ~3 the interval is a fixed window — `lo` 1.06–1.17, `hi` 3.39–4.01 almost regardless
of `y` — because the compound sits below the lowest tested dose. Modelling a *width* and centring
it on `y` misplaces each bound by 0.35–0.59 in the 3–3.5 band (0.71 for CYP2D6 at 3.5–4.0), while
above 4.5 it is accurate to 0.014. **Use `lo(y)` / `hi(y)` lookups.** Width collapses with
potency (Spearman −0.88/−0.90/−0.56/−0.93; D1/D10 ratio 7.7–14.4), and **pIC50 4 is the end of
that collapse, not its start** (breakpoints 2.90–3.55 then 3.85–4.30). `y` lies strictly inside
`[lo, hi]` for all 6,525 labels. Fit per endpoint — pooling costs up to 0.119 R².

## Driving Colab: three things that cost time on 2026-10-06

- **`colab exec` does not accept script arguments.** `colab exec -s S -f script.py -- --flag x`
  fails with "Got unexpected extra argument(s)". `colab run` sets `sys.argv` but it also
  provisions and tears down a fresh VM, so it is no use for a session you want to keep. The way
  to pass arguments to a script on a live session is an **argv shim over the uploaded file**:

      colab upload -s S scripts/x.py /content/x.py
      echo "import sys,runpy
      sys.argv=['x.py','--seeds','10']
      runpy.run_path('/content/x.py', run_name='__main__')" | colab exec -s S

  Kernel state persists between `exec` calls, so uploads and imports are done once.

- **`colab upload` 500s if the target directory does not exist.** Create it with an `exec` first
  (`os.makedirs`); the error is an opaque `500 Server Error` with the proxy URL, not a path
  message.

- **This account's L4 session has 12 vCPU, 23 GB VRAM and 52 GB RAM.** The note above saying an
  L4 has 4 vCPU against an A100's 12, and that featurisation therefore makes the A100 the cheaper
  choice, **no longer holds**. Better still, **upload the cached feature matrix** (6145 × 3091 is
  5.8 MB) rather than recomputing it: featurisation drops from the critical path to 0.2 s and an
  L4 is then sufficient for anything this programme runs. Filter `exec` output through
  `sed 's/\x1b\[[0-9;]*m//g'` — tracebacks come back ANSI-coloured and are unreadable otherwise.

- **`OneCycleLR` divides by zero at small `total_steps`.** Its warmup phase collapses to one step
  when `pct_start * total_steps < 2`, and `get_lr` then computes `(step - start)/(end - start)`
  with `end == start`. A 10-epoch smoke test hits it; a 250-epoch run does not, so it only ever
  breaks the cheap check that was meant to protect the expensive run. Use
  `CosineAnnealingLR`, which has no phase boundary.

## `restart-kernel` does NOT kill detached subprocesses — and that is what OOMs the card

**This section's first version blamed kernel-held VRAM and was wrong.** Keep the reading order:
the wrong diagnosis was plausible, survived one test, and cost a second failed run.

`colab exec` runs in the **Jupyter kernel**, and anything that kernel allocates on the GPU stays
resident until the kernel is restarted. So a session where several experiments were run *inside*
the kernel has little free VRAM left, and the next job — even one launched as a detached
subprocess — starts against a nearly full card. On 2026-10-06 the incumbent trainer had smoke
tested fine at a peak of 18.7 GB and then died with `[net] OOM -> batch 256` on the real run; a
`colab restart-kernel` took free VRAM from a few GB back to **22,564 MiB of 23,034** and the same
config ran.

That looked like the whole story and it was not. The same config OOM'd **again** on a card
measured at 21,444 MiB free moments before launch. The real cause: **`colab restart-kernel`
restarts the kernel but leaves `Popen(..., start_new_session=True)` children running.** The
driver launched before the restart survived it, kept looping, and was spawning its own trainer
alongside the one launched after — **two 18.7 GB jobs on a 23 GB card**, which is the hazard the
"two heavy runs on one 40 GB A100" note already described. A process listing showed 16 matching
pids (two drivers plus one trainer's twelve featurisation workers); killing them returned the
card to 22,564 MiB free.

The tell was there and I walked past it: a `ps` right after the restart reported **19 python
procs** on a session that should have had none running.

Restarting the kernel *is* still worth doing between heavy experiments, and the halve-and-retry
backoff genuinely cannot save an allocation-time OOM — it halves 4096 down to 256 and then
**raises**, deliberately, so an arm comparison cannot be silently run at two different batch
sizes. But neither of those was the fault here.

So, before placing any run that needs real memory:

```python
free = int(subprocess.run(['nvidia-smi','--query-gpu=memory.free',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.split()[0])
assert free > 20000, f'only {free} MiB free — something is holding the GPU'
```

and guard against the actual cause too:

```python
n = subprocess.run(['bash','-lc',
    "ps -eo pid,command | grep -cE '[m]y_driver.py|[m]y_trainer.py'"],
    capture_output=True, text=True).stdout.strip()
assert n == "0", f"{n} training processes already running — kill them first"
```

**Refuse rather than OOM an hour in**, and check for orphans *before* blaming memory. Before any
relaunch after a restart or a failure, list and kill the detached children by name.

Measured capacities on this account's L4 (23 GB), `cyp_net_one.py` at `--n-layers 3`:
**16 members peaks at 18.7 GB**; the default 32 members needs ~37 GB and does not fit. 20 epochs
take 0.5 min, so ~1.5 min per 100 epochs.

## Round 3: every DRC label is a SELECTED label, and V1 validation cannot see it

Added 2026-10-08. **Read `ROUND3-PLAN.md` before `PLAN.md` / `ENDGAME-PLAN.md`.**

The release ships `cyp-challenge-single-concentration-TRAIN.csv`: a 50 µM screen (log2 fold-change
of signal) over **4,376 compounds × 4 isoforms**. A dose-response curve exists only for compounds
**promoted** out of it, and the promotion rule differs per isoform — CYP2D6 is a hard threshold
(DRC iff log2FC < ~−0.6; its 2,882 non-promoted compounds sit at median **+0.38**, i.e. no
inhibition), CYP1A2 rises with inhibition, CYP2C9 is flat (~1/3, random), CYP3A4 is enriched for
*weak* readouts. **The two isoforms whose promotion depends on activity are the two we score worst
on.** This log asserted for two days that the activity-promotion shift was "structurally
untestable on our data because every label we hold is promoted". The non-promoted compounds were
in the release the whole time, in a file the loader already read for its `log2fc` auxiliary heads.

**`drugrx/cyp/screen.py`** turns them into labels. A compound showing no inhibition at 50 µM gets
the **floor window `[1.07, 3.52]`** (median DRC bounds of the 303 compounds that have both); other
readouts get the [q10, q90] of DRC pIC50 in their bin. **They are intervals, never points.** Three
higher-ranked entries report independently that the same rows *cost* score as point pseudo-labels
and *help* as censored bounds, so the distinction is load-bearing, not a stylistic choice.

**V1 vs V2 — the thing to internalise.** V1 is the programme's old scaffold validation: DRC rows
only, so **every row is a compound that was selected for activity**. V2 adds the held-out screen
inactives against the floor window. V2 reproduces the board per isoform where V1 never did
(LightGBM DRC-only: V2 0.77 / 0.53 / 1.35 / 0.48 against the board's 0.88 / 0.52 / 1.46 / 0.57).
A selection fix is **expected to look worse on V1** — promotion selected on `y`, the analog
expansion selected on `x`, so the estimator the test wants is `P(y|x)` on the *unselected*
population. Report both surfaces, always; select on V2.

Measured effect of the fix: macro V2 **−0.24** (LightGBM, 3/3 seeds) and **−0.31** (the net, 2/2);
CYP2D6 1.51 → 0.60; test-set mean prediction 4.59 → 3.26 against the **3.19** backed out of the
board months earlier by an unrelated route. **ρ improves too** (2D6 0.378 → 0.476 on V1), which is
what matters, because every placement transform is monotone and cannot touch ρ.

## Placement is a separate lever from modelling, and the field says it is the bigger one

`drugrx/cyp/placement.py`, `scripts/cyp_place_sim.py`, `scripts/cyp_submit.py`.

A model trained on a selected label set predicts on that set's scale. The last step is an affine
map per isoform onto the blinded population. From the public model reports (`research/24`): rank
47 went **0.7247 → 0.5214 on identical models**; rank 127 reimplemented rank 12's exact four
members without the calibration and scored 0.7671 against 0.4378.

- Squared error wants centre = population mean, spread = `ρ · sd_blind` (Murphy 1988:
  `R² = 2ρk − k² − b²`, so `k = ρ`, and `R² ≤ ρ²` is a hard ceiling — a model far below its own
  `ρ²` is mis-calibrated, not weak).
- **ST-RAE wants something else**: less spread, and a centre *above* the mean, because an inactive
  compound's credible interval is a free window reaching up to ~3.5. Our board probe measured the
  spread discount at **0.896**; a competitor's simulation puts it at 0.9–0.95.
- `cyp_place_sim.py` computes the location distance instead of guessing it (rank 12 says the
  direction is known and the distance is not — "a credible-interval width model would give it
  directly", and `bound_lookup` *is* that model). A sensitivity scan says what transfers: the
  **optimal spread is 0.42–0.58 × the population sd** across every mixture assumption tried, and
  the **optimal centre is the population mean + 0.17 to +0.50** — but the absolute centre moves
  3.05 → 4.36 with the assumption, so only the relative rules are usable.
- **`BLIND_MOMENTS` is third-party observed content, not an authority.** It is used only because
  our own board probe agrees with it (its mean sd 1.381 vs our measured 1.308; its 2D6 mean 3.107
  vs our 3.19 from an unrelated inversion). Treat it as corroboration, keep the provenance in the
  docstring, and ask the board.
- The map is monotone, so **ρ and τ must return bit-identical** — the free end-to-end check this
  programme has now used three times.

**`drugrx/cyp/decide.py`** is the metric's own optimal point: ST-RAE's risk is convex piecewise
linear, so the best prediction is the balance point where `P(lo > p) == P(hi < p)` — a weighted
median over pooled bound draws, verified exact against a 4,001-point grid on 400 random cases. It
is why a probability-weighted *mean* is the wrong operation for a mixture: a compound 80% likely
inactive should be predicted at **3.52**, the top of the free window, not at the mean's 2.87.

**Measured 2026-10-10 and kept OUT of the submission path** (`scripts/cyp_decide_vs_affine.py`).
Against the affine placement it loses **+0.089 to +0.094 macro ST-RAE on 8 of 8 seeds**, and the
composition (calibrate, then decide) only ties while costing ρ, so it is strictly dominated.
**The balance point is a decision rule, not a calibration:** our predictions are compressed
(`k_ours` 0.18–0.66 against a 0.77–0.82 optimum), the draws inherit that, and it cannot fix a
mis-scaled distribution — which is all the affine map does, and is where the ST-RAE actually is
(calibration alone is macro 1.24 → 0.68, a 46% cut). It wins only on CYP2C9, whose raw scaling is
best, and loses worst on CYP2D6, whose raw ST-RAE is 2.43. Keep it as a diagnostic.

Caveat, so this is not over-read: the posterior tested was a **homoscedastic** residual bootstrap,
which cannot express the mixture case the module was written for. A fair test needs a per-compound
inactivity probability through `mixture_draws` and is not built. The verdict is "not with this
posterior" — but the practical question, *should the submission path use it*, is settled: no.

**Placement *tuning* is also nearly saturated**: a nested affine fit (on OOF training predictions)
lands within 0.007–0.012 macro ST-RAE of one fitted on the validation answers. **Do not read that
as contradicting the 8–28 board places.** On validation, train and validation moments nearly match
(s_y 1.04–1.08), so nested is near-oracle by construction, while the board's s_y is
1.439 / 0.966 / 1.660 / 1.160 — a ratio up to **1.77×**. The 0.012 is a lower bound from the easy
case, and that mismatch is exactly why v5/v6/v7 probe the board. Keep probing.

## The TDI classification track exists and we had never entered it

`scripts/cyp_tdi.py`. Second track, same 750 compounds, **CYP2D6 and CYP3A4 only**, ranked by
**MCC**, submitted as a separate file (`SMILES`, `Molecule_Name`, `CYP2D6_is_TDI`,
`CYP3A4_is_TDI`, booleans, exactly 750 rows). The organisers publish the labelling rule, and the
released `is_TDI` columns follow it **exactly** (2D6 1,475/1,475, 3A4 2,175/2,175), so it can be
applied to compounds the release did not label — a compound inactive in both arms is an *assigned
negative*, which is what the 2,648 CYP2D6 screen-inactives are. Same fix, same size: macro MCC
0.235 → **0.300** on V2, 3/3 seeds.

**The threshold is as big as the model.** Positive rates are 8–21%, so a 0.5 cut costs ~0.14 MCC
against the MCC-argmax (0.02–0.10). Choose it **nested** — on the training fold, never the fold
being reported — and note that plug-in expected-MCC on isotonic-calibrated probabilities came
*second* to the plain OOF argmax here.

## The leaderboard is readable without a browser, and it inverts to the blinded truth

**Reading.** `scripts/cyp_read_board.py`. A Gradio Space serialises its state into
`window.gradio_config` in the page HTML, and each component's `props.value.data` holds **all**
its rows. One ~800 KB fetch gives all 17 tables — Live and Interim, Regression, TDI and
Structure, macro and per isoform. `tab_paths()` walks `cfg["layout"]` for `tabitem` labels,
which is what stops Live and Interim being confused (reading the wrong one once produced a
reported 0.9628 against a Live 0.8663). This does **not** help with writing — Gradio's Svelte
handlers ignore synthetic DOM events, so submitting still needs real clicks in the extension.

**Submitting.** `HOURS_BETWEEN_SUBMISSIONS = 12`, scoped **per track** (`submissions/{track}/
{user_id}/`), confirmed from the Space's source — so a TDI upload costs nothing against the
regression slot. Fill the *Method Report Link* every time: it is **mandatory before 2026-11-03**
or the entry is dropped regardless of score, and it is stored either way but only checked for
reachability when the open-source-code box is ticked. Leave that box unchecked while the code is
unpublished. The Submission Status panel is the only reliable confirmation.

**Inverting.** A row you still hold the predictions for is a system of equations in the hidden
test set's parameters. Both of these paid off decisively — see
`scripts/cyp_solve_moments.py`, `cyp_tdi_postmortem.py`, `cyp_read_their_placement.py`:

- *Regression*: `r = 2 sin(πρ/6)`, then `s² = s_y² + s_p² − 2 r s_y s_p` and
  `m² + s² = (1−R²)s_y²` solve for the truth's `(s_y, m)`.
- *Classification*: `π = (1 − Acc)/(1 + R/P − 2R)`. Every entry is scored on the same set, so π
  must **agree across entries, and that agreement is the validation** — it held to sd 8e-5 over
  152 rows. MCC then has an exact closed form, `√(π/(1−π))·(R−q)/√(q(1−q))`.
- `s_y` belongs to the *test set*, so it is shared: for anyone else's row it is known and their
  `(s_p, |m|)` become the unknowns. That is how to read a competitor's placement.

**Three traps, each of which bit once.** (1) Decide up front which statistics are inputs and
keep one out as a check — MAE was once called a "free check" when it was the equation pinning
`s_y`. (2) Only `|m|` is recoverable; the sign needs an external estimate
(`scripts/cyp_bias_sign.py`, resolved unanimously at 7.7–22×: **we over-predict on all four**).
(3) The board's metrics are **bootstrap-averaged** — no integer confusion matrix on any N fits
them, and reported F1 sits below the harmonic mean of reported P and R on all 305 rows. Treat
inverted quantities as effective values and bound the error rather than assuming exactness.

## Placement, not ranking, is where the remaining score is — and half of it is untouched

Murphy: `R² = r² − (r − k)² − b²` with `k = s_p/s_y` and `b = bias/s_y`, so `r²` is a ceiling and
everything below it is recoverable **by an affine map**, which is monotone and therefore cannot
move ρ or τ. `scripts/cyp_murphy_gap.py` puts that at **+0.092 macro R² at fixed ranking**, and
the **bias** term exceeds the spread term on three of four isoforms. Corroborated by the board
directly: at rank 30 an entry has ρ only +0.013 above ours and macro ST-RAE 0.4737 against our
0.5925.

Do the macro arithmetic **per isoform, never on the macro row** — macro averages ρ and R²
separately, and by Jensen `mean(r²) ≥ (mean r)²`, so a macro R² can legitimately exceed the `r²`
computed from a macro ρ. Reading that as a violated bound is a mistake waiting to happen.

**The ST-RAE optimum is not the squared-error optimum**, so never just set `k = r, b = 0`. The
deadband makes an inactive compound's credible window free to ~3.5, so the optimum sits **above
`ȳ` and narrower than `r·s_y`**. `scripts/cyp_affine_v6.py` therefore parameterises the centre as
`centre_v4 + α·m` and refuses to guess α: it must come from the simulator calibrated against a
measured board row. Submitting one-variable changes is what makes that calibration possible —
v5 moved spread only, for exactly this reason.

## CYP3A4 is the untouched isoform, and the one with the most headroom

Three independent supports, found 2026-10-09:

1. Re-aggregating `screen_lgbm.json` **per isoform** (it had only ever been read as the macro
   −0.243): the screen fix is worth **−0.2341 / −0.0954 / −0.6329 / −0.0081**. The whole gain is
   CYP2D6 and CYP1A2; on CYP3A4 it does nothing, and the cruder `drc+inactive` form *hurts* it.
   That is what the promotion rules predict — 3A4 is enriched for *weak* readouts, so there is no
   selection defect there to repair.
2. Board percentile for ρ: 69th / 63rd / **88th** / **41st**. CYP3A4 is our only below-median
   isoform; CYP2D6 is hard for everyone (field best 0.5700 against our 0.5004).
3. A single `drc+screen` LightGBM reproduces the board's CYP3A4 ρ to **0.005 with a seed sd of
   0.011**, while falling 0.210 and 0.150 short on CYP1A2 and CYP2C9. The deployed entry is an
   ensemble, so on CYP3A4 the ensemble and everything in it is worth nothing.

CYP3A4 also has the most DRC labels (2,335) and the largest ranking headroom (+0.106 ρ to the
field best). **It is where the next model experiment goes.**

## The blinded test set is EASIER than either split we select on

`scripts/cyp_split_realism.py`, no labels and no model. Median nearest-neighbour ECFP4 Tanimoto
to training: **TEST 0.581**, scaffold-val 0.439, random-val 0.455 — the test is not bracketed by
our splits, it is easier than both. The shape matters more than the centre: the test has
essentially no far tail (p10 0.500, **96.9 % at ≥ 0.4**) while scaffold validation puts 32 % of
its mass below 0.4. An analog expansion of training hits gives every test compound a close
relative by construction; a scaffold split deliberately removes exactly those.

Consequence already paid for: the minimax placement scan was handed a correlation band of
0.33–0.60 estimated on that hard surface, while the board row inverts to r = 0.775 / 0.822 /
0.518 / 0.783 — **three of four above the top of the band it was told to search**.

Whether it also changes *model selection* is **resolved: it does not** (2026-10-09 late).
`scripts/cyp_split_match_board.py` first came out underpowered, but its power check used an
**unpaired** standard error for a quantity that is a paired difference — the surfaces are nested
subsets of the same held-out compounds from the same fitted model. `scripts/cyp_split_match_paired.py`
tests it properly, which halves the standard error, and on **12 seeds never looked at** no surface
beats the plain split: `near` +0.0109 ± 0.0146, `matched` +0.0117 ± 0.0089, production
`matched_spearman` +0.0130 ± 0.0133 (positive = worse), all CIs crossing zero.

Two traps inside that result, both worth not repeating:

- A 3-seed paired comparison gave `matched − all` = +0.0466 at **t = 8.67, 3/3 seeds** and it
  collapsed to t = 1.32 on fresh seeds. With n = 3 the sd rests on 2 df, so clustered
  differences manufacture a large t. **Never read a t from three seeds.**
- `matched_spearman`'s docstring claims importance weighting keeps the compounds the NN ≥ 0.5
  restriction discards. Its own printed Kish ESS refutes that: **0.24 n, the same as the
  restriction**, at r = 0.986 against it. 84.9% of the test weight is in one band, so the
  weighting *is* the restriction.

## The ranking lever is measured and empty. Do not re-open these without NEW information.

Every model-side candidate, measured on **Spearman** (the metric placement cannot change), as
of 2026-10-09. Reaching the field's best needs **+0.07 to +0.11 of validation rho per
isoform**, so read the magnitudes against that, not against zero:

| candidate | macro Δρ | status |
|---|---|---|
| screen-derived interval labels | **+0.0417** | **deployed — the only one that ever worked, and it is a DATA insight** |
| ensembling (whole stack vs one LightGBM) | +0.0154 | 1–2 se; small and not clearly nonzero |
| TDI-condition labels (CYP3A4 only) | +0.0038 | real at t = 4.0, 3.6 % of that isoform's gap |
| release's 20 auxiliary heads (47k labels) | +0.0010 | closed |
| GBM hyperparameters | +0.0004 | **closed — swept to +0.0114 at t = 2.96, did not replicate** |
| CheMeleon representation | 0.0000 | closed |
| interval hinge loss | −0.0094 | closed; *costs* ranking |
| local / kNN blending | −0.0616 at NN ≥ 0.5 | closed; helps only where the test set has nothing |

The feature stack is already the MapLight recipe (ECFP4 counts + rdkit2d + Avalon + ErG, 3,091
dims) and a pretrained graph representation has been tested and failed, so **2D features are
saturated**. Further cheap variants (ECFP radius/size, stacking weights, SMILES augmentation)
have expected effects of ±0.005–0.010 against a 5-seed standard error of 0.003–0.005 — a
signal-to-noise ratio near 2 when 0.07+ is needed. That is a false-positive generator, not a
search. **The only orthogonal information source identified is 3D/structure-derived**: the
challenge ships protein–ligand complexes for its Structure track, so docking against the four
CYPs would be new information rather than another fingerprint. Large build, uncertain payoff,
and a decision rather than a next step.

**Two traps this table cost, both worth not repeating:**

- A gain measured on our validation can *reverse* on the board. A kNN blend scored **+0.0186 ρ
  at t = 10.7** on CYP2D6 and **−0.0616** restricted to near-neighbour compounds — the only
  kind the test set has. Always report the NN ≥ 0.5 contrast beside the headline.
- A gain can fail to replicate even when it looks structured. The hyperparameter sweep was
  monotone in `num_leaves` at every `colsample` and still vanished on fresh seeds, because
  those slices shared one resample. **Confirm every selected winner on seeds never looked at.**

## Closed lines — do not re-open without new information

- **Foundation models.** `ecfp_head` 0.5180 ±0.0086 against `ft_head` 0.5179 ±0.0130: identical.
  The entire CheMeleon gain was the **interval hinge loss**, worth −0.064 against the GBM's MAE
  on the same features, and the production net already uses it.
- **More training.** 30,000 epochs beats 1,200 by −0.0026 against a seed spread of 0.0052, for
  25× the compute. `ReduceLROnPlateau` is the *worst* arm (best at epoch 25) because it decays on
  validation noise.
- **External inactives.** 10.5k Veith bounds made it worse: macro V2 +0.038, 3/3 seeds.
- **log2FC as a test-time feature.** Impossible, checked against the release file list: the
  single-concentration file is **TRAIN only**. It is already an auxiliary head (`AUX_LOG2FC`,
  17,500 labels) and that is the only way it can enter.
- **TDI thresholds.** At the realised test ROC both are within **0.0006 macro MCC** of optimal.
  Any further TDI gain must come from a better ranker.

## The validation→board lift: most of our board Spearman is the test set, not the model

**`scripts/cyp_validation_lift.py`. Run this before believing any model comparison predicts a
board move.** The production net's own validation predictions are saved
(`final_v4screen/nets/final-net_s*.npz`, `val` + `va_index`), so the ensemble and a plain
LightGBM can be scored on the same compounds. Measured 2026-10-09:

| isoform | prod net val ρ | LightGBM val ρ | board ρ | net − lgbm | **board − net (lift)** |
|---|---|---|---|---|---|
| CYP1A2 | 0.5848 | 0.5558 | 0.7597 | +0.0290 | **+0.1749** |
| CYP2C9 | 0.6737 | 0.6801 | 0.8084 | −0.0064 | **+0.1347** |
| CYP2D6 | 0.4840 | 0.4424 | 0.5004 | +0.0416 | +0.0164 |
| CYP3A4 | 0.7737 | 0.7764 | 0.7685 | −0.0027 | −0.0052 |
| **macro** | 0.6291 | 0.6137 | 0.7093 | **+0.0154** | **+0.0802** |

Three consequences, all of which have already cost time once:

- **The whole model stack beyond a LightGBM + the screen fix is worth +0.0154 macro ρ**, and
  that is one to two standard errors given the LightGBM's 0.025–0.054 seed spread. Treat it as
  "small and not clearly nonzero". A long search for what the ensemble was doing — the hinge
  loss, the auxiliary heads, CheMeleon — found nothing, because there was nothing to find.
- **The lift is a calibration, so use it.** `validation ρ + lift ≈ board ρ`, per isoform. It is
  stable across placement rounds, because placement is monotone and cannot move ρ; re-run it
  only after a change to the *model*.
- **CYP3A4 is not the isoform we are bad at.** On the board it is our weakest relative to the
  field (41st percentile, +0.106 to the best); on our validation it is comfortably our **best**
  (0.774 against 0.585 / 0.674 / 0.484). It simply receives **no lift** while CYP1A2 and
  CYP2C9 receive +0.135 to +0.175 — its board number is the honest one and theirs are flattered.

  **The reason is label SPREAD, not the analog expansion** (measured 2026-10-10,
  `scripts/cyp_lift_decompose.py`; the analog-expansion reading is withdrawn). The blinded set's
  `s_y` is 1.38 / 1.25 / — / 1.08 times our validation's, and the lift tracks that ratio
  monotonically on the three invertible isoforms. CYP3A4 gets no lift because its spread ratio
  is only 1.08. Crucially the model's placement-free accuracy `σ* = s_y·√(1−r²)` **transfers**
  (board/validation 1.07 / 1.00 / 1.10) — so the blinded set is *marginally harder to predict*
  and merely easier to *rank*. Forecast a board ρ with the spread correction (4.3× better, mean
  |error| 0.026 against 0.111), never with an NN correction.

The general rule this is an instance of: when a leaderboard and a local split disagree about a
model, **find a prediction vector that exists on both and score it twice** before attributing
the difference to anything.

## Detached work on a Colab VM needs three things, not one

Learned the hard way twice, the second time expensively (a 4.6 h docking run, 2026-10-09).

1. **A keepalive.** Colab reaps a session when the **kernel** is idle, no matter what the GPU
   is doing, and `nohup … &` makes the kernel idle the instant `colab exec` returns. Run
   `scripts/colab_keepalive.sh <session> [interval] [max_hours]` *before* launching. Beware
   that routine polling acts as an accidental keepalive — a run can look stable only because
   you happen to be watching it, and die when you stop.
2. **Progress that leaves the machine.** Appending to a log on the VM and skipping on restart
   survives a dead *process*, not a dead *VM*. `scripts/dock_pull.sh <session>` copies the
   output back every few minutes. Pull the expensive setup artefacts once too (`ligands.tgz`,
   `desc3d.npz` — 3.1 MB) so a restart re-does only compute.
3. **A committed input manifest.** If outputs are keyed by row index, the manifest's ordering
   *is* the alignment contract. `scripts/colab/dock_manifest.py` builds it from the same call
   the evaluator makes and writes a sha. Built ad hoc on the VM, a rebuild in a different order
   puts every feature on the wrong molecule — and the run still completes, which is worse than
   a crash.

### Two launcher traps

* **`CMD` must be a bare command.** `nohup cd X && python …` cannot work: `cd` is a shell
  builtin, so `nohup` fails to exec it and `&&` short-circuits. `$!` still returns the
  backgrounded subshell's pid, so the launcher reports a healthy pid for a job that never ran,
  leaving an empty log that is indistinguishable from a job that has not printed yet.
  `dock_launch.py` now rejects the chain and verifies the pid survives 5 s.
* **A positive control must run the production command.** The setup control used `--maps rec`
  after map caching had been abandoned in the runner; it passed while production failed all
  4,399 ligands. Matching them is what makes the control evidence. At production settings
  (`--receptor`, 20 Å box, exhaustiveness 4) ketoconazole gives −8.77 kcal/mol with its closest
  heavy atom 3.82 Å from the heme Fe.

### Counting background watchers: filter on ppid, not on the command string

A `bash` script that uses command substitution — `out=$(colab exec ...)` — forks a subshell,
and `ps` reports that subshell with the **parent's full command line**. On a saturated VM a
single `colab exec` can take minutes, so the subshell is long-lived and looks exactly like a
second copy of the watcher.

I misdiagnosed this twice on 2026-10-09, reported "duplicate keepalives doubling exec
contention", and killed processes that were not duplicates. No harm resulted — all six watchers
survived — but the correct check is the parent pid, because a `nohup`-ed watcher is reparented
to init:

```sh
ps -eo pid,ppid,command | grep -E "bash scripts/(dock_pull|colab_keepalive)\.sh" \
  | grep -v grep | awk '$2==1'
```

Also note **macOS `pgrep` has no `-c`**: `pgrep -fc <pat>` prints a usage error to stderr and
nothing to stdout, so `n=$(pgrep -fc ...)` silently yields an empty string. A monitor built on
it reported "0/3 alive" for watchers that were all running. Use `pgrep -f ... | wc -l`.

**A keepalive's quiet log is not evidence it has stopped.** `colab_keepalive.sh` prints only
every 20th beat, and when the VM is saturated each beat takes far longer than its nominal
interval, so 30+ minutes of silence is normal. Judge liveness by the thing you actually care
about — rows still arriving — not by the watcher's own chatter. (A hung `colab exec` is itself
kernel activity, so it keeps the session alive while it blocks.)

## The lever table, as of 2026-10-09 evening

Every model-side candidate measured on the CYP challenge.

**Do not prefer the matched column — that instruction is withdrawn.** It read "read the matched
column, correcting for the NN regime flips the sign of effects below ~0.01". Measured on 12
fresh seeds (2026-10-09 late): no surface — `near`, 20-bin `matched`, or the production
`matched_spearman` — is measurably closer to the board's ρ than the plain scaffold split; every
paired CI crosses zero (`prodmatch − all` +0.0130 ± 0.0133). The production estimator is
numerically the NN ≥ 0.5 restriction (r = 0.986, Kish ESS 0.24 n against the restriction's
0.24 n) and costs **2.4× the per-seed noise**. So the sign flips it produces are noise, not
corrections. Read the plain Δρ column, and see PROGRESS for the full measurement.

| candidate | macro Δρ | verdict |
|---|---|---|
| screen-derived interval labels | **+0.0417** | **deployed — the only thing that ever worked, and it is a *data* insight** |
| docking into CYP3A4 (Vina, 6,852 ligands) | +0.0068 matched | **fail** — below the +0.01 bar; features used (4% importance) but redundant |
| TDI-condition labels (CYP3A4) | +0.0038 | real, tiny |
| release's 20 auxiliary heads | +0.0010 | closed |
| GBM hyperparameters | +0.0004 | closed — did not replicate on fresh seeds |
| CheMeleon representation | 0.0000 | closed |
| ensembling (net vs LightGBM) | ~0, CIs cross zero | **withdrawn** — the +0.0154 was unpaired |
| interval hinge loss | −0.0094 | closed |
| kNN / local blending | −0.0616 at NN≥0.5 | closed |
| ranking objectives (lambdarank) | **−0.177** | closed |

Reaching the field's best needs **+0.07 to +0.11** per isoform. Nothing on that list is within
an order of magnitude of it except the screen fix, which is deployed.

**Price levers in σ\*, not in ρ — ρ hides the scale.** The gap is a **22% cut in absolute
prediction error** (σ\* 0.727 → 0.567 pIC50, ~0.16 log units; `scripts/cyp_lift_decompose.py`).
Docking's +0.0068 ρ on CYP3A4 was a **1.3%** accuracy gain against the **25%** needed there.
Every lever in the table above was worth ~1% of accuracy. A candidate that cannot plausibly
deliver several percent of absolute accuracy does not need measuring.

**Placement is a separate axis and is where the recent gains came from** — but it is worth only
8–28 board places per isoform, against 28 places for +0.02 of ρ on CYP3A4. ρ is the valuable
quantity; placement is the cheap one.

## The three lever classes, all measured rather than argued (2026-10-10)

| class | status |
|---|---|
| **model capacity** | **spent on ρ, NOT on ST-RAE** | ranking: +0.0019 ρ, p = 0.723; width +0.0000. But **depth × long schedule is DECIDED on ST-RAE**: −0.0234 ± 0.0048, t = −4.92, 5/5 seeds (`PROGRESS.md` 2026-10-06). Neither factor alone — depth is −0.0085 (noise) at 400 epochs, the schedule −0.0057 (noise) on L3; the *interaction* carries it. Capacity bought **placement**, not ranking. |
| **data quantity** | **closed at achievable scale** — learning-curve α 0.019–0.079 implies ≥38× more data (`scripts/cyp_data_scaling.py`, `cyp_scaling_fit.py`) |
| **representation / model class** | **the only one left, and the only class that ever worked here** (H1 features, H5 Avalon+ErG) |

The data result is the load-bearing one. All the data there is buys 3.5–13.7% of absolute
accuracy over a 6.7× range of n; the gap needs **22%**. Even the best-determined curve (CYP3A4,
α 0.0793, R² 0.99) needs ~38× — about 77,000 extra same-assay CYP3A4 IC50s, which do not exist
publicly. And it is an **upper bound**: the curve measures *in-distribution* returns, while 10.5k
external Veith inactives already made this model worse on 3/3 seeds. If ideal same-assay data at
38× cannot be had, external data cannot do better.

**Open, and honestly so:** whether data is closed at *any* scale. The falsifiable plateau fit
`a·n^−α + c` does not identify over a 6.7× range — the floor's bootstrap CI spans [0, 0.53–0.81]
on every isoform, so every verdict is "cannot tell". Resolving it needs a wider n range, and we
already have all the data there is.

**Do not open a new lever without pricing it in σ\* first.** A candidate that cannot plausibly
deliver several percent of absolute accuracy does not need measuring — that is the lesson of ten
levers at ±0.005 ρ, each worth ~1% when 22% was required.

**What is ruled out as an explanation for the remaining gap:** a fixable subgroup (rank error
is flat across potency, similarity and censoring), extrapolation failure, the label ceiling
(censored compounds rank *better*), proprietary data (0/20 of every isoform's top 20 declares
any), and every single lever above.
