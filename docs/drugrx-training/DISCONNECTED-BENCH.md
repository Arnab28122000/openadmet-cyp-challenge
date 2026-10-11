# A disconnected benchmark, and the test it has to pass

Opened 2026-10-06. **Pre-registration — written before any candidate data was scored.**

## The problem this is for

Our validation set is a **scaffold split of a diversity library**. The challenge's evaluation set
is an **analog expansion**: the organisers' own `config.py` says hit expansion around the top 25
hits of CYP1A2, CYP2C9 and CYP3A4, ten nearest ECFP4 analogs each from the Enamine US catalog,
75 × 10 = 750. Measured geometry: test within-set nearest-neighbour Tanimoto **0.665** against
validation's **0.354**.

The programme established that **no re-split of our own data can emulate that** — an
analog-expansion re-split of our library reaches within-set 0.446, not 0.665, because the
organisers *purchased* analogs we do not hold at any density. That is a hard bound on re-splitting.

It is **not** a bound on acquiring analogs from somewhere else. That is what this is.

## What changed, and why the bar is now higher

The earlier framing — "we measured novel-chemistry performance all programme while being scored
on analog expansion" — turned out to be an inference from an aggregate that was not homogeneous.
Decomposed per endpoint:

| endpoint | validation | leaderboard | gap | share | ρ val → test | R² val → test |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.7582 | 0.8842 | +0.126 | 17% | 0.569 → **0.696** | 0.309 → 0.100 |
| CYP2C9 | 0.5523 | 0.5237 | **−0.029** | −4% | 0.674 → **0.775** | 0.442 → **0.521** |
| CYP2D6 | 0.8938 | 1.4577 | **+0.564** | **74%** | 0.391 → 0.330 | 0.144 → **−0.951** |
| CYP3A4 | 0.4758 | 0.5735 | +0.098 | 13% | 0.766 → 0.761 | 0.469 → **0.552** |

So the analog expansion is **easier to rank** on two endpoints and the gap is 74% one endpoint.
A benchmark built to "be more like the test set" therefore has a much sharper job than before:
it must reproduce a **strongly non-uniform profile across endpoints**, not merely be harder.

That makes it falsifiable, which is the point.

## Construction — the rules, fixed now

1. **Disconnection is by standardised InChIKey, never SMILES.** A SMILES-level identity check
   once hid 425 train/test collisions in this programme. The exclusion set is
   (challenge training ∪ every external source already in our pipeline ∪ the 750 blinded test
   compounds). Zero intersection, or the candidate is rejected outright.
2. **Continuous potency only** for the headline benchmark. Binary active/inactive sets cannot
   measure a calibration failure, which is the thing we most need to measure.
3. **Analog series, not a diversity library.** The whole purpose is the geometry. A candidate
   whose within-set nearest-neighbour similarity is near validation's 0.354 adds nothing we
   already have.
4. **Source provenance recorded per compound**, and no source silently merged with another.
   Pooling across labs is what broke the ExpansionRx `cl_micro` merge.
5. **No interval model in the acceptance test.** See below — the test uses MAE, R² and ρ, which
   are computable on any labelled set. ST-RAE needs `lo`/`hi`, which external data will not have;
   it may be reported with modelled bounds, clearly labelled, but it may not be the criterion.

## The cross-lab offset, and why the test is a contrast

External labels carry a systematic offset against the challenge assay — different probe
substrate, different enzyme source, fluorogenic versus LC-MS readout. CYP3A4 IC50 in particular
moves substantially with probe substrate. An offset inflates MAE and depresses R², which is
**exactly the signature we are trying to detect on CYP2D6**. A naive external benchmark would
therefore show "bad calibration everywhere" and look like it had reproduced the finding when it
had only reproduced its own offset.

The defence is that a lab offset is **roughly uniform across endpoints**, while the leaderboard's
profile is violently non-uniform (CYP2D6 R² −0.95 against CYP3A4 +0.55). So the criterion is the
**across-endpoint contrast**, not the absolute level, and every number is reported both raw and
per-source-centred.

## ACCEPTANCE TEST — pre-registered, pass/fail

Score the *same* model artifacts we hold (`POOL/everything`, 81 seeds, and the two arms whose
leaderboard scores we know) on each candidate benchmark, and compute per-endpoint **MAE, R²,
Spearman ρ**.

**The incumbent to beat is validation itself.** Validation's per-endpoint profile differs from the
board's by a mean absolute ST-RAE error of **0.204**; the equivalent profile distances on
(MAE, R², ρ) are computed in the harness and recorded before any candidate is scored.

A candidate benchmark **passes** only if all four hold:

- **A1 — disconnection.** Zero standardised-InChIKey overlap with the exclusion set.
- **A2 — geometry.** Within-set median nearest-neighbour Tanimoto ≥ **0.55**, materially above
  validation's 0.354 and the 0.446 ceiling of a re-split of our own data. (Not required to hit
  0.665; required to beat what we can already build.)
- **A3 — profile.** Reproduces the board's across-endpoint profile **better than validation
  does**, on all three of MAE, R² and ρ, measured as mean absolute error against the board's
  per-endpoint vector after per-source centring.
- **A4 — the CYP2D6 contrast.** CYP2D6 is the worst endpoint on the candidate, and its R² deficit
  relative to CYP3A4 is at least **half** the board's (board: −0.951 vs +0.552, a contrast of
  1.503 — so the candidate must show ≥ 0.75).

**A4 is the one that matters.** A1–A3 can be satisfied by a benchmark that is merely hard. Only
A4 says the benchmark has reproduced *the actual failure*, and A4 is also the criterion most at
risk from the lab offset, which is why it is stated as a contrast against CYP3A4 measured on the
same compounds in the same assay.

**If no candidate passes, that is the result** and it gets written down. An external benchmark
that cannot reproduce the leaderboard's profile is not a safer place to tune — it is a second
unvalidated surface, and this programme has already paid for tuning on one of those.

## What a pass would buy

A surface on which to choose a decoder, a representation or an architecture that is **not** the
922-row validation set whose gains transfer at 17%. That is worth more than any single
score, because every remaining lever has to be chosen somewhere.

## Noted while writing this, and worth its own look

The test set's construction is **fully known**: 75 anchors (top 25 hits of three isoforms) × 10
nearest ECFP4 Enamine analogs. The 750 test structures are public — only the labels are blind. So
**the 750 can be grouped into 75 analog series, each anchored on a training compound whose label
we hold.** That is new information the programme has never used: not a similarity estimate, but
the actual generative grouping. It is also the one setting where the pre-registered pairwise/delta
model has its anchors handed to it rather than guessed. Pursue separately.
