# Research Lab — durable orientation

**Read this first, then `NOTEBOOK.md` (newest entry last), then `PLAN.md`.**

A standing research programme on **model class and representation** for molecular property
prediction, run as a lab with multiple concurrent agents. It was opened 2026-10-10 because the
DrugRx/CYP programme had measured every other lever to exhaustion and representation was the only
one left. It is deliberately broader than the challenge: a decoupled mathematics/CS strand runs
alongside, and its findings feed back.

Sister programme: `docs/drugrx-training/` (the CYP challenge itself). **Do not duplicate it.**
That programme owns submissions, placement and the leaderboard; this one owns model class.

---

## The one rule that governs everything here

**Price a lever in σ\* before building it.**

    σ* = s_y · sqrt(1 − r²)      the residual sd after an optimal affine map
    r   = 2 · sin(π·ρ / 6)        Pearson from Spearman, bivariate normal, no ties

**The target is a 13.6% cut in absolute error.** Measured off the live board, not inherited:
our macro σ\* is **0.9004** (rank 92/271); the best macro σ\* any single entrant achieves is
**0.7780** (`preheat-to-450`, macro ρ 0.7844). Re-derive it with `scripts/cyp_real_frontier.py`.

**The 22% figure this programme used until 2026-10-10 was wrong**, and it was wrong in the
direction that makes everything look hopeless. It came from a leader σ\* of 0.567 that **no
entrant achieves** — even the impossible composite of the best ρ on each isoform, stacked from
four different competitors, is only a 15.5% cut. Pricing levers against 22% priced them against
a competitor who does not exist, and it is why the measured 4–8% model-class ceiling first read
as "hopeless" when it actually covers **29–59%** of the real gap.

Two traps that produced it, both worth remembering:
- **Per-isoform bests do not compose.** Different entrants hold each isoform's best ρ.
- **A macro of ρ is not a macro of accuracy.** σ\* must be computed per isoform (the s_y differ
  by 1.7×) and then averaged; converting an averaged ρ once is a different, wrong number.

Still true, and still the reason ten levers failed: each was worth about **1% of absolute
accuracy**, against a 13.6% requirement.

So: a candidate that cannot plausibly deliver **several percent** of absolute accuracy does not
need measuring, and does not get built. State the price before the build, in writing, here.

### The corollary that keeps being re-learned

**Compute the ceiling over the whole method family before building any member of it.** For a
separable objective the ceiling is usually closed-form and takes minutes. Report the **oracle**
and the **cross-fitted** number together — the oracle is real, always positive, and unattainable;
the gap between them is the entire story. This killed a mixture-posterior decoder in six minutes
that would otherwise have cost an hour and produced a convincing in-sample lie.

**And the ceiling does not have to be computed here.** The first step of expanding any phase is a
literature sweep for someone who has already run it. `findings/09` closed a whole phase this way
in an afternoon with no GPU: three groups had spent 10²–10⁴× our compute on the method family, and
one had run our exact experiment (LeJEPA + SIGReg on molecular graphs, Chemprop D-MPNN included) —
frozen-probe +0.123, **finetuning null**, the one significant partition gone across five. Verify
the load-bearing claims against the primary source by hand; a shell's summary is a lead, not a
citation.

---

## What is already closed. Do not re-propose these.

| lever | verdict | evidence |
|---|---|---|
| **model capacity** | **spent on ρ, NOT on ST-RAE** | ranking: +0.0019 ρ, p = 0.723; width +0.0000. But **depth × long schedule is DECIDED on ST-RAE**: −0.0234 ± 0.0048, t = −4.92, 5/5 seeds (`PROGRESS.md` 2026-10-06). Neither factor alone — depth is −0.0085 (noise) at 400 epochs, the schedule −0.0057 (noise) on L3; the *interaction* carries it. Capacity bought **placement**, not ranking. |
| **data quantity** | closed at achievable scale | learning-curve α **0.019–0.079** ⇒ ≥ **38×** more data. Plateau term does not identify over a 6.7× range of n |
| **placement / decode** | saturated | oracle ceiling over *all* maps of the prediction = +0.028 in-sample, **negative** cross-fitted at every bin resolution (3…32) |
| **docking (AutoDock Vina)** | failed its bar | matched Δρ +0.0068 vs a pre-registered +0.01 |
| **matched/importance-weighted metric** | rejected as an instrument | `matched_spearman` **is** the NN≥0.5 restriction it replaced (r = 0.986, Kish ESS 0.24n both) at **2.4× the noise** |
| **σ̂-based 2-D decode (Borda / any f(p,σ̂))** | closed | Borda −0.0003 macro, 7/24; cross-fitted ceiling over *all* functions of (p,σ̂) negative at 3/4/6 bins. A fitted variance function is ~80% better than ensemble spread and changes nothing (−0.0004) |
| **CYP2D6's "backwards" σ̂** | explained, not a lever | artefact of the screen-inactive point mass; partial on p is **+0.038 (4/4 seeds)** vs raw −0.164. See `findings/06` |

**Why docking failed, which is the live lead:** Vina's scoring function has **no metal-coordination
term**, and CYP inhibition is substantially **type-II** — direct coordination of a ligand nitrogen
to the **heme iron**. Ketoconazole's imidazole docked **13.4 Å** from the Fe; the pose was simply
wrong. Conformers and artifacts are kept in `talindrew-drugrx/data/cyp_cache/dock/`
(manifest sha `18cc8cbc5e74abf3`) so a metal-aware retry reuses them.

**A structural fact that constrains half the obvious ideas:** there are only **four protein
targets**, all with known structures. Protein-side features are *constant within an isoform*, so
for per-isoform models they contribute exactly **zero**. Protein language-model embeddings can
only matter through cross-isoform joint modelling. Check this before proposing anything
protein-side.

---

## Measurement discipline — the traps this programme has actually fallen into

**The screen pseudo-labels are a point mass, and it is large.** `screen_labels` assigns every
screen-inactive compound the constant **2.211**, and `PSEUDO_W["inactive"]` is **1.0** — the same
weight as a real measurement. **And 2.211 is the MEDIAN of a measured 303-compound distribution,
not a bound** (mean 2.3822, sd 0.4435, min 1.906, **P(y\* > 2.211) = 0.498**) — so this is
*coarsening*, not censoring, and every left-censored estimator is mis-specified for half the rows
it touches (`findings/11`). CYP2D6 has **zero** dual-labelled compounds, so its floor law is
**unidentified**. `allr_floor` computes that whole distribution and keeps only its median. That is **61% of CYP2D6's labelled training signal** and **2% of
CYP3A4's**, a 30× range. It makes ensemble members *agree* where it dominates, so any per-item
uncertainty estimate is distorted there in a way that looks like a model pathology and is not
(`findings/06`). Anything that reads σ̂, residual structure or calibration per compound must check
this first. Dropping the mass costs CYP2D6 **−0.0516 ρ**, so it is informative — the defect is
asserting `y = 2.211` instead of `y ≤ 2.211`.

Each of these cost real time here. They are not generic advice.

1. **Three seeds cannot carry a t-statistic.** A paired effect at t = 8.67 on 3 seeds became
   t = 1.32 on 12 fresh ones. n = 3 gives 2 df and manufactures significance.
2. **Slices share noise; resamples don't.** Agreement across a hyperparameter axis is not
   evidence. Confirm every winner on seeds never looked at.
3. **A magnitude bar is not a significance bar.** More seeds shrink the error bar, not the effect.
   Do not re-look at a pre-committed fail.
4. **Test the explanation you just gave.** A stated mechanism is an untested hypothesis. Ours was
   ~25% right once and flatly refuted another time (the label-spread mechanism reversed sign,
   −0.205 → +0.044, when taken from 3 seeds to 12).
5. **Weighting that concentrates is a restriction.** Always print an importance-weighted
   estimator's **Kish ESS** `(Σw)²/Σw²` against the subset it claims to replace.
6. **A correct structural argument is not an effect size.** The ST-RAE optimum genuinely is
   non-monotone; it was worth at most +0.028 in-sample and nothing attainable.
7. **Pre-register the numeric prediction AND the decision rule**, and commit them before the
   result exists. Grade honestly afterwards: a band hit with a missed point estimate is a miss on
   the point, and gets recorded as one.
8. **Do not state a claim about quantity A and test it with a statistic that measures A + B.**
   Three consecutive specification errors here share this root: `findings/06` (right suspect,
   both clauses of the mechanism inverted), `findings/07` (the arm changed a point mass's
   location, the claim was about its concentration), `findings/08` (the claim was about
   point-mass share, the statistic measures total non-Gaussianity — and the isoforms' baseline
   non-Gaussianity varies 6.7×, swamping a 30× share range). The fix is cheap and mechanical:
   **before running, name B and say how it is held fixed.** A *within*-unit paired contrast
   usually cancels it where a cross-unit level comparison cannot.
9. **A result that is incoherent across units you expected to agree is a bug signal, not a
   finding.** Opposite signs on CYP2D6 and CYP2C9 for the same manipulation turned out to be a
   half-normal's `−σ√(2/π)` mean shift confounding location with spread. Chase the incoherence
   before interpreting either arm.
10. **Measure the latent distribution before modelling the censoring** — or more generally,
   before modelling any quantity, check whether something in the release already measures it. One
   `groupby`, 20 seconds, refuted the premise of a 6-seed GPU arm that had already been run,
   written up and closed, and would have pre-empted two earlier findings' errors as well
   (`findings/11`).
11. **Pre-register the caveat, not just the prediction.** "Not a finding if it flips across
   normalisations" is what made effective rank safe to compute: it flipped **36×**, and nothing
   was claimed. Without that sentence written first, the raw number was sitting there ready to
   be reported.

---

## The lab

| role | model | does |
|---|---|---|
| **Principal investigator** | Opus 5 (the main session) | sets the questions, prices every lever in σ\*, decides, owns `NOTEBOOK.md`, runs submissions |
| **Senior researchers** | Opus | deep single-topic strands, adversarial verification of findings |
| **Research scientists** | Sonnet | focused technical sweeps and implementation |
| **Research assistants** | Fable | literature and practitioner sweeps, surveys, breadth |
| **Technicians** | Haiku / Sonnet | mechanical reruns, data prep, verification against spec |

**Agents return conclusions, never file dumps.** Every research brief carries the shared context,
the closed-lever list and the σ\* pricing rule, and must answer in priced form: expected σ\* gain,
evidence strength, implementation cost, and *the single most likely reason it fails here*.

**The decoupled strand is genuinely decoupled.** It is briefed abstractly, with no mention of
chemistry, so it cannot pattern-match to the application and tell us what we want to hear. Its
deliverable is transferable principle.

---

## Where things live

| path | what |
|---|---|
| `NOTEBOOK.md` | **the ELN.** Dated entries: question → method → result → conclusion. Newest last. Negative results get the same space as positive ones. |
| `PLAN.md` | the phased plan. One phase expanded at a time, not all at once. |
| `CHANGELOG.md` | what changed, when, and in which commit. |
| `findings/` | one file per completed research strand, as returned and then verified. |
| `experiments/` | scripts and results for hypotheses actually tested. |
| `talindrew-drugrx/` | the code. Model, features, metric, splits, submission path. |

**Screenshots are scratch** — never committed, deleted when the test is done. Record what a
screenshot proved in words here instead. See the repo's `CLAUDE.md`.

Commits use `Arnab Bhattacharya <53032587+Arnab28122000@users.noreply.github.com>` and carry
**no Claude attribution**.
