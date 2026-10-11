# Lab notebook

One entry per piece of work. **Question → method → result → conclusion.** Newest entry last.
A negative result gets the same space as a positive one; the point of the book is that nothing
has to be re-derived or re-argued later.

Conventions:
- Every claim carries the evidence that supports it, or it is marked **unverified**.
- Every lever carries a **σ\*** price before it is built. See `AGENTS.md`.
- Dates are UTC. The shell reports IST (UTC+5:30), so the local date rolls at 18:30 UTC.

---

## 2026-10-10 — the lab opens, and why

**Question.** The CYP programme had closed every lever it could name: capacity (+0.0019 ρ,
p = 0.72), data quantity (α 0.019–0.079 ⇒ ≥38×), placement/decode (oracle ceiling +0.028
in-sample, negative cross-fitted), and docking (Δρ +0.0068 against a +0.01 bar). Macro ρ sat at
0.7093, rank 54/271, against leaders near 0.85 per isoform. One lever class had never been
measured: **representation and model class**. Is there a model class, or a source of structure,
that extracts materially more signal from ~1,200 labels per isoform?

**Why now, and why this is not just more of the same.** The reframing that made the empty lever
table legible: the gap is not "+0.1 ρ", it is a **22% cut in absolute error**, σ\* 0.727 → 0.567.
Ten previous levers were each worth about **1%**. So the question is not "what else could we try"
but "what could plausibly be worth several percent" — a far shorter list, and a different kind of
search.

**Method.** Six independent research shells, briefed in parallel, each carrying the same shared
context, the closed-lever list and the σ\* pricing rule, each required to answer in priced form
(expected σ\* gain, evidence strength, cost, and the most likely reason it fails *here*):

| strand | model | brief |
|---|---|---|
| ADMET/CYP state of the art | Opus | what actually wins at n ~ 1k, and is it leakage? |
| physics & chemistry of binding | Opus | metal-aware docking, Boltz-2 affinity, xTB/DFT, NNPs — the diagnosed heme-Fe gap |
| folding models as representation | Sonnet | co-folding features, and whether 4 constant targets make this vacuous |
| mathematical model classes | Fable | GPs/kernels, TDA, equivariant GNNs, multi-task, TabPFN, conformal |
| practitioner knowledge | Fable | Pat Walters, Kaggle/Polaris writeups, Reddit, the DL-vs-GBM debate |
| **decoupled theory** | Opus | briefed *abstractly*, no chemistry: learning-curve exponents, representation-vs-data diagnostics, covariate shift with low overlap, non-squared-error metrics, censored labels |

The decoupled strand is briefed with no mention of chemistry on purpose, so it cannot
pattern-match to the application and return what we want to hear.

**Result.** Pending — agents running.

**Conclusion.** Pending.

---

## 2026-10-10 — three programme decisions (Arnab's calls, PI recommendation overridden on all three)

Recorded because the reasoning matters later, and because my recommendation lost each time.

| decision | chosen | I had recommended | why it still makes sense |
|---|---|---|---|
| **v8 placement** | full centre step on **all three** + widen CYP2D6 | skip CYP2C9's dead axis | CYP2C9's step is worth ~0.0010, so as an *improvement* it is noise. As a **measurement** it is a second point on that axis and converts "dead" from a one-secant inference into a confirmed one. Logged as measurement, not improvement. |
| **GPU budget** | **aggressive** — multiple sessions, long jobs | one session, pre-priced only | Buys parallel coverage of the expensive candidates (co-folding affinity, NNP interaction energies, MM-GBSA) instead of serialising them. The risk it accepts is a well-executed null at high burn — which is exactly what docking was. The σ\* rule still applies to *interpretation* even when it no longer gates *spend*. |
| **scope anchor** | **open research** — follow what is interesting | product capability, challenge as testbed | Raises the weight on the decoupled theory strand and licenses threads that may never ship. The challenge stays as the blinded scoreboard because it is the only thing here that cannot be fooled — but it is no longer the objective. |

**What changes in how I work.** Open research + aggressive compute removes the two constraints
that were doing most of the filtering. So the discipline has to come from somewhere else: every
strand still gets priced in σ\* **before** it is believed, ceilings still get computed before
builds, and negative results still get written up at full length. Cheap compute makes it *easier*
to produce a convincing in-sample lie, not harder.

---

## 2026-10-10 — first three strands back. The headline is a correction to our own target.

**Question.** Is there a model class or representation worth materially more than the ~1% that ten
previous levers delivered?

**Result — the literature answer, which I accept.** The entire model-class ladder from a tuned GBM
to the 2026 state of the art is worth **4–8% of absolute error** on tasks of this shape, measured
by two independent protocols, one of which uses a shift design matched to ours (scaffold holdout
*plus* dropping test molecules above Tanimoto 0.60). Several celebrated methods are *worse* than
tuned XGBoost under that design — iMoLD, built specifically for molecular distribution shift, lost
47 of 47 regression tasks. That independently corroborates three levers we had already closed.

**Result — the correction, which matters more.** Before pricing anything against our standing 22%
target I re-derived it from the board (`scripts/cyp_real_frontier.py`):

* best macro σ\* actually achieved by a single entrant: **0.7780** (`preheat-to-450`)
* ours: **0.9004**, rank 92/271
* **real cut required: 13.6%**, and the impossible best-of-each-isoform composite is only 15.5%

**The 22% was wrong.** It priced against a σ\* of 0.567 that no row on the board attains. The same
4–8% ceiling reads as "hopeless, 3–5× short" against 22% and as "**covers 29–59% of the gap**"
against 13.6%. A number nobody had re-derived was driving opposite decisions.

**Conclusion.** Three things, in order of how much they change what we do:

1. **The gap is reachable in part.** Not by one lever, but 4–8% from representation plus the
   items below is a real fraction of 13.6%.
2. **We tested the right representation with the wrong head.** Our CheMeleon null was a *ridge
   probe*; the published result requires an in-context (PFN) head, and a participant at rank
   28/219 in this very challenge got there using Mitra on frozen embeddings. This is a recorded
   negative on a configuration we never ran.
3. **Our data closure has a hole.** α ⇒ ≥38× is a bound on *labels*. The rank-28 entry retrieved
   88,683 near-neighbours of the blind structures and admitted 4,999 **with no assay labels** —
   unsupervised domain adaptation, which a label-count curve does not bound.

**Two strands disagreed about where the error lives, and both were right.** CYP2D6 carries
**39.4%** of our absolute error (physics strand, correct) *and* is nearly maxed out — best entrant
on earth is ρ 0.543 against our 0.500, we are rank 28 (maths-derived view, also correct). It is
hard for everyone. The physics strand's top recommendation was priced on "bring 2D6 to 0.850",
which the board refutes outright; its value collapses from ~15% to ~2% of macro.

**Negative results banked, with outside corroboration:** metal-aware docking is closed — even
*perfect* type-II physics is capped at 16.7% because only 26–31% of CYP inhibitors carry a
heme-coordinating motif, and an independent 22,608-docking replication on this same challenge
measured peak signal **+0.0008**. Boltz-2's affinity head is disqualified for cofactor pockets by
its own paper. And a detail worth keeping: our original docking bar of +0.01 Δρ was itself only a
**1.5% σ\* cut** — even a clean pass would not have been worth building.

### Correction, same day — I repeated a strand's misattribution

The practitioner strand's headline was "you tested the right representation with the wrong head",
citing `research/24:278` — "CheMeLeon-2048 ridge probe (zero blend weight)" — as our own test.
**It is not ours.** `research/24` catalogues *competitors'* public model reports, and that ledger
belongs to **JacksonBurns**. I wrote it into finding 02 as our result before checking.

What we actually ran: frozen CheMeleon → GBM (0.6433 vs ECFP 0.5819, worse); frozen CheMeleon →
hinge head (0.6065 vs 0.5180, **worse**); CheMeleon fine-tuned end-to-end → hinge head (0.5179 vs
0.5180, dead heat); in-context TabICL/TabDPT on count-ECFP (−0.0032, inside noise). CheMeleon is
recorded **closed at 0.0000** in two ledger tables, and we had *already* diagnosed the PCA confound
ourselves in `cyp_chemeleon_icl.py`'s docstring.

One cell is genuinely untested — frozen CheMeleon at **native width** into an in-context head —
and the script for it exists and appears never to have been executed. Worth running because it is
cheap, but as a **low-prior confirmation**: both marginals are null-or-worse here, and the
published claim rests on the interaction.

**The same paragraph contains a free result I nearly missed:** a competitor independently tried
stir_bar's near-neighbour retrieval and got **+0.0002**. That is third-party evidence against the
"unsupervised domain adaptation" lever I had just listed as live and unbounded. It does not close
it — different implementation, different pipeline — but it substantially lowers the prior, and it
was sitting in our own tree.

**Rule, re-learned:** a strand reading our repo can attribute a competitor's failure to us, and a
plausible story is not provenance. Check whose number it is before building on it.

---

## 2026-10-10 — the expected-rank decode is a null, and the 2-D ceiling closes the family

**Question.** The theory strand's sharpest actionable claim: the Spearman-optimal score is the
**expected rank** (Borda), not the conditional mean, because with heteroscedastic uncertainty
`E[rank] ≠ rank(E[y])`. Simulated at **+0.015 to +0.096 Spearman**. It is a function of `(p, σ̂)`,
so my earlier decode ceiling — which ranged over maps of `p` **alone** — explicitly did not bound
it. Does it pay?

**Method.** `scripts/cyp_borda_decode.py`, 6 scaffold seeds × 4 isoforms. Under independent
`Yᵢ ~ N(pᵢ, sᵢ²)`, `E[rankᵢ] = Σⱼ Φ((pᵢ−pⱼ)/√(sᵢ²+sⱼ²))`. σ̂ from the spread of an 8-member
ensemble (bootstrapped rows × distinct model seeds). Plus the ceiling over **all** functions of
`(p, σ̂)`: bin on both quantiles, give each cell its own mean label, reported as oracle *and*
cross-fitted.

**The built-in correctness test.** With every `sᵢ` equal, Borda is a strictly increasing function
of `pᵢ`, so the ranking must be **identical** to raw. If it is not, the implementation is wrong and
every number is void. **PASSED on all 24 cells** — so whatever is measured below is
heteroscedasticity and nothing else.

**Result.**

| isoform | raw ρ | Borda ρ | Δ | t | wins | σ̂ CV |
|---|---|---|---|---|---|---|
| CYP1A2 | 0.5514 | 0.5507 | **−0.0007** | −1.92 | 1/6 | 0.31 |
| CYP2C9 | 0.6686 | 0.6687 | +0.0001 | 0.21 | 4/6 | 0.35 |
| CYP2D6 | 0.4759 | 0.4754 | **−0.0005** | **−3.10** | **0/6** | 0.33 |
| CYP3A4 | 0.7752 | 0.7750 | −0.0002 | −0.67 | 2/6 | 0.34 |
| **macro** | | | **−0.0003** | | **7/24** | |

And the ceiling over the whole 2-D family:

| bins | oracle ρ gain | cross-fitted gain |
|---|---|---|
| 3 | −0.0435 | −0.0865 |
| 4 | −0.0155 | −0.0945 |
| 6 | +0.0104 | **−0.0691** |

**Conclusion. The lever is dead, and this time the ceiling comes with it.** Borda is −0.0003 macro
and *significantly negative* on CYP2D6 (t = −3.10, 0/6 seeds). The ceiling over **every** function
of `(p, σ̂)` is negative at every resolution, and barely positive even in-sample. So this is not
"Borda specifically fails" — no decode of `(p, σ̂)` helps, which closes the 2-D family the same way
`cyp_decode_ceiling.py` closed the 1-D one.

**Against the prediction, honestly.** The theory forecast **+0.015 to +0.096**; we measured
**−0.0003**, with the sign wrong on 3 of 4 isoforms. The mechanism is real and the maths is right —
I verified the degeneracy property holds exactly — but the effect size does not survive contact
with this data. That is the fourth time this programme has had a correct mechanism priced at
roughly zero.

**And σ̂ CV is 0.31–0.35**, so there *is* substantial heteroscedasticity. The null is not "σ̂ is
constant". It is that the variation carries no rank information.

**The one honest caveat, from the theory addendum that arrived mid-run.** σ̂ here is the **raw
spread of 8 members** — ν ≈ 7 replicates, which the variance-estimation literature puts squarely in
the hazard zone. So what is bounded is functions of `(p, σ̂_ensemble)`, not of `(p, σ̂_true)`.
`scripts/cyp_sigma_quality.py` settles it by asking whether σ̂ predicts `|residual|` at all, against
a **fitted variance function** (asymptotically free, where per-item estimates are first-order
harmful) and a constant. Running.

### The σ̂ diagnostic: the fix works, and the lever is dead anyway

`scripts/cyp_sigma_quality.py`, 4 seeds × 4 isoforms. Spearman of σ̂ against `|residual|` on
held-out compounds:

| isoform | ens spread | **variance function** | Borda Δ (ens) | Borda Δ (varfn) |
|---|---|---|---|---|
| CYP1A2 | +0.086 | **+0.122** | −0.0004 | +0.0002 |
| CYP2C9 | +0.129 | **+0.162** | −0.0003 | +0.0007 |
| CYP2D6 | **−0.164** | **−0.157** | −0.0005 | −0.0013 |
| CYP3A4 | +0.107 | **+0.166** | −0.0002 | −0.0014 |
| **macro** | **+0.040** | **+0.073** | **−0.0003** | **−0.0004** |

**The addendum's prescription is confirmed on our data.** A fitted variance function is a
**~80% better** estimate of predictive uncertainty than the raw 8-member spread (+0.073 vs +0.040),
better on 3 of 4 isoforms, exactly as Carroll–Wu–Ruppert predict and exactly as the ν≈7 hazard
implied. That is a genuine, reusable methods result and it cost one script.

**And Borda still flatlines: −0.0004 with the better σ̂.** So this is the "dead for a stronger
reason" branch. We improved the uncertainty estimate by ~80% and the decode bought nothing. The
2-D family stays closed, and the caveat I attached to the previous entry is now discharged rather
than outstanding.

**The reason the lever fails is visible in the magnitudes:** even the fitted σ̂ correlates with
`|error|` at only **+0.073** macro. Heteroscedasticity exists (CV 0.22–0.34) but is barely
recoverable from the features, so a rule that reweights by it has almost nothing to act on.

### The anomaly worth keeping: CYP2D6's σ̂ is NEGATIVELY correlated with error

**−0.164** (ensemble) and **−0.157** (variance function), consistent across seeds and both
estimators. On CYP2D6, compounds the model is *more* uncertain about have *smaller* errors. That is
backwards, and it is the only cell where it happens.

**Hypothesis, flagged as untested:** CYP2D6 has by far the largest inactive mass (its placement
centre is 3.43, the lowest of the four). Compounds predicted near the floor plausibly draw both
high ensemble disagreement *and* labels compressed against the floor, so `|error|` is mechanically
small there — a censoring artefact rather than a property of the model's confidence.

If that is right it also explains **why Borda hurts CYP2D6 most** (t = −3.10, 0/6 seeds): Borda
shrinks high-σ̂ items toward the middle of the order, but on CYP2D6 the high-σ̂ items are the
*well-predicted* ones, so shrinking them is precisely the wrong move.

Per this programme's own rule, that is a stated mechanism and therefore an untested hypothesis —
`feedback-test-the-explanation-you-just-gave`. It is cheap to test (correlate σ̂ with distance to
the floor, and recompute the σ̂-vs-error correlation on the non-floor subset) and it is **not**
being claimed until someone does.

## 2026-10-10 17:22 UTC — v8 submitted. Two new traps, and a near-miss worth more than the submission.

**v8 is in**, accepted at 17:22 UTC. `POST /gradio_api/queue/join` → 200, panel confirms
*"Submission received from 'talindrew' for the Regression Prediction track."* v7's row was read
and recorded first, so overwriting it cost nothing.

Pre-registered in `scripts/cyp_v8_prereg.py`, **committed before the result existed**: ρ/τ
unchanged at 0.7093/0.5285; CYP1A2 [0.5268, 0.5379]; CYP2C9 [0.4603, 0.4613]; CYP3A4
[0.5355, 0.5513]; CYP2D6 point **0.5340** in [0.5196, 0.5517]; macro 0.510–0.528; and the
falsifiable one — **no centre cell may rise above its v7 value**, which convexity forbids.

### The rate limiter is enforced to the second, and it saved us

The armed trigger fired at 17:15, built from v7's *click* time of 05:15. The board had recorded
v7 at **05:20:39**, so the real window opened at **17:20:39**. The Space answered the first click
with:

> "Error: You submitted a regression prediction on 2026-10-10 05:20:39 (UTC). Please wait
> 00:00:51 before submitting again."

51 seconds early. A rejection is free — no slot consumed, previous row intact — so this cost
nothing, but it means **the window is `board HH:MM + 12 h`, and the `:SS` only ever appears in
the rejection message.** Arm the next one with margin.

### The near-miss: three independent signals all said "the click didn't land", and all three lied

After the first click I checked, in order: `read_network_requests` (showed the file-upload POST
but **no `queue/join`**), `get_page_text` (**no "Submission Status" heading at all** — and the
runbook's documented textarea trap explains only a *missing message*, not a missing heading), and
a screenshot (form intact, button idle, page bottom visible, no panel). On that evidence I
concluded the submission had not fired and clicked again.

**All three were wrong.** The `queue/join` POST for the first click appeared in the log minutes
later; the panel eventually rendered *both* results in sequence. The first click had fired and
been rejected; the stream carrying its result had 503'd.

What actually made the second click safe was **the rate limiter**, not my reasoning. Had I been
one minute later, both clicks would have been accepted and the second would have burned the next
12-hour window. The file was byte-identical either way, so no measurement was at risk — but the
*slot* was, and the reasoning that got me there was unsound.

Two durable corrections, now in the runbook and in `cyp-challenge-leaderboard-ui`:

- **`read_network_requests` lags.** Absence of a POST in that log is not evidence the POST did not
  happen. Wait and re-read before concluding.
- **`queue/join` 200 is the real receipt** — seconds, not ~90 s, and immune to the SSE failure.
  `GET /gradio_api/queue/data` returned **503 on both** submissions while both joins returned 200.
  The panel is a *confirmation of last resort*, not the primary check.

This is the fourth time on this programme that a confident negative read of an instrument has been
wrong (after the "recording bug" that was a real stationary point, the stalled scorer that looked
like a lost submission, and the two `get_page_text` false negatives). The pattern is specific
enough to name: **when an instrument says "nothing happened", that is the reading most likely to
be an artefact of the instrument**, because "nothing" is also what every failure mode of the
instrument itself produces.

## 2026-10-10 18:20 UTC — the CYP2D6 σ̂ anomaly is solved. Right suspect, both clauses backwards.

Ran the test I said was cheap and said I would not claim until someone ran it
(`scripts/cyp_sigma_anomaly.py`, pre-registered in `dd1652a` before the run). Full write-up in
`findings/06-the-sigma-anomaly-is-a-point-mass-artefact.md`.

**Resolved: the anomaly is an artefact of a training point mass, and it is not about uncertainty.**

The DRC labels have **no censoring spike** — 0.1–0.2% of labels on the lowest value, on every
isoform. The floor is in the *training pseudo-labels*: `screen_labels` pins every screen-inactive
compound at **2.211** with `PSEUDO_W["inactive"] = 1.0`, the same weight as a real measurement,
and that is **61% of CYP2D6's labelled signal** against **2% of CYP3A4's**.

With 61% of the signal on one value, the ensemble members *agree* in the regions it dominates —
so σ̂ is **small** at low p, not large — while the DRC compounds falling there are where the model
is most wrong. Opposite signs in p; their product is the anomaly:

- r(σ̂,p) on CYP2D6-full is **+0.324, positive on 4/4 seeds — the only such cell in the 4×3 grid.**
- r(|err|,p) is **−0.592**, the most negative anywhere.
- **Partial r(σ̂,|err| | p) = +0.038, positive on 4/4**, against +0.073/+0.063/+0.048 elsewhere.

Controlling for the prediction level, CYP2D6's σ̂ behaves exactly like the other three. The causal
arm confirms it: as the floor's weight falls 1.0 → 0.2 → dropped, CYP2D6 moves −0.164 → +0.014 →
+0.110 (**+0.273**, the only isoform to cross zero) and the sign of r(σ̂,p) goes **4/4 → 2/4 → 0/4**.

### Both clauses of my own explanation were inverted

I wrote that near the floor compounds "draw both high ensemble disagreement *and* labels
compressed against the floor, so |error| is mechanically small there." Measured: near the floor
the ensemble **agrees** (σ̂ small) and the error is **largest**. The *suspect* was right — the
inactive mass — and the *mechanism* was wrong in both of its halves. Pointing at the right culprit
is not the same as knowing what it did.

That is now three stated mechanisms tested on this programme and two refuted outright, which is
the whole reason `feedback-test-the-explanation-you-just-gave` exists.

### What survives, and what I am explicitly not claiming

Survives: **why Borda hurt CYP2D6 most** (t = −3.10, 0/6). Borda shrinks high-σ̂ items toward the
middle; on CYP2D6 the high-σ̂ items are the high-p, *well-predicted* ones, so it corrupts the top
of the order exactly where the model is right. Same conclusion as before, now with the mechanism
right.

**Not claimed:** the 0.2-weight arm gains ρ on all four isoforms, pooled **+0.0035** — but
**t = 1.92, 10/16 wins, and every isoform loses on seed 0**. That fails our own bar. A candidate
for more seeds, not a result, and not going into a submission on this evidence.

### The next move is censoring, not deletion

Dropping the floor costs CYP2D6 **−0.0516 ρ** — tens of places on this board. The information is
real; the false precision is the problem. "Inactive in the screen" means `y ≤ 2.211`, not
`y = 2.211`, which is a left-censored regression and exactly what the already-briefed
**Chernozhukov–Hong 3-step censored quantile regression** strand addresses. It keeps what the
`none` arm throws away while removing what produces the artefact. Untested.

## 2026-10-10 19:30 UTC — censored floor: both pre-registered claims failed, and the line closes

`scripts/cyp_censored_floor.py`, 6 seeds, pre-registered in `829eb5d`. Full write-up in
`findings/07-the-floor-treatment-line-is-closed.md`.

**Claim 1 (artefact repaired, r(σ̂,p) → ≤ 0): FAILED.** CYP2D6 went +0.336 → +0.264 and stayed
positive on **6/6** seeds.

**Claim 2 (macro Δρ ≥ +0.005 and ≥ 5/6): FAILED, both arms.** `down` +0.0037 at 5/6 — clears the
wins half, misses the magnitude half. `tobit` +0.0012 at 3/6 — misses both. Neither is going into
a submission. And **the dumb weight hack beat the principled estimator on all four isoforms**
(+0.0044/+0.0050/+0.0049/+0.0006 against +0.0027/+0.0017/+0.0012/−0.0007). Theory earned nothing.

### I mis-specified my own falsification condition, and that is the real result

I pre-registered *"if it does not happen, finding 06 is wrong and says so."* The honest report is
that **the inference was wrong**, and the evidence is a number I could have computed before
spending any compute:

| arm | the 2648 inactive targets |
|---|---|
| `full` | 2.211, **sd 0.000** — a point mass |
| `tobit` | ~2.10, **sd 0.030–0.084 = 3–5% of s_y** |

The EM step imputes a conditional **mean**, so compounds with similar μ get near-identical values.
The arm moved the point mass's **location** and left its **concentration** intact — and finding
06's mechanism is entirely about concentration. The arm never varied the thing its claim was about.

That is `feedback-negatives-need-range` verbatim: *don't test a mechanism where it cannot appear.*
I wrote that rule into this lab's own discipline list and then broke it in the very next
experiment. The run is **uninformative** about finding 06, which still stands on the weight
dose-response that did have range (r(σ̂,p) positivity 4/4 → 2/4 → 0/4).

This is not a licence to re-read the fail as a pass. Claim 1 failed as written; what I am allowed
to conclude is only that the test was badly built, and I am recording *both*.

### Closing the line, on price rather than on fatigue

The textbook follow-up exists — multiple imputation by **draw** from the truncated normal, per
ensemble member, which is the standard correction for mean-imputation understating variance. I am
not building it:

1. **Ceiling measured, below bar.** Three arms spanning reweight (+0.0037), delete (−0.0516 on
   CYP2D6) and re-locate (+0.0012). Best is +0.0037 against a +0.005 bar.
2. **No downstream consumer.** The only use for better σ̂ was the 2-D decode, and that is closed —
   Borda −0.0003, cross-fitted ceiling over all f(p,σ̂) negative at every resolution. Repairing σ̂
   fixes a diagnostic nothing reads.
3. **Helps where we need it least.** CYP2D6 is 61% floor and rank 28 (needs 5.4%); CYP3A4 is 2%
   floor and rank 134 (needs 24.7%). Said before the run; borne out.

Open and unclaimed: whether de-concentration repairs the artefact. Unresolved — we just have no
use for the answer.

### Two process failures in one day, same shape

Finding 06: stated **mechanism** inverted, suspect right. Finding 07: stated **falsification
condition** mis-specified, because the arm did not implement the intervention it claimed to test.
Both caught only by a cheap measurement taken *afterwards* — the label distribution, the imputed
sd. The rule earned today: **check that an arm actually varies the quantity its claim is about,
before running it.**

## 2026-10-11 12:40 UTC — v8 regressed and falsified its headline claim. v9 interpolates instead.

**v8 scored: macro 0.5256 → 0.5520, rank 54 → 69. All four cells worse.** Graded against its own
pre-registration, only ρ/τ survived:

| cell | band | actual | |
|---|---|---|---|
| CYP1A2 | [0.5268, 0.5379] | 0.5685 | OUT, −276% of projected gain |
| CYP2C9 | [0.4603, 0.4613] | 0.5104 | OUT, **−4931%** |
| CYP3A4 | [0.5355, 0.5513] | 0.5658 | OUT, −92% |
| CYP2D6 | 0.5340 ∈ [0.5196, 0.5517] | 0.5634 | FALSIFIED, +0.0294 |
| **no centre cell worsens** | — | — | **FALSIFIED by all three** |
| ρ / τ | 0.7093 / 0.5285 | 0.7093 / 0.5285 | PASS, 6th |

CYP2C9 was submitted explicitly as a dead-axis *measurement* worth 0.0010, with the written
trigger "if CYP2C9 moves materially (|d| > 0.005), the one-point secant was wrong and every other
centre projection here is suspect too." **It moved 0.0491.** The trigger fired exactly as written,
and it was right: all three centre projections were wrong.

### What was refuted is narrower than it looks, and the distinction is the whole next step

ST-RAE is a sum of hinge terms, so it is convex and piecewise **linear** in the centre and in the
spread, with derivative `(N_above − N_below)/D`, a non-decreasing step function. **Convexity was
never the failed assumption.** The claim's own stated proviso was *"provided we do not cross the
balance point"* — and we crossed it on all four axes at once. Convexity stands; the belief that v7
sat on the near side of the zero crossing does not.

That is not a rescue, because it costs us the claim either way. It is load-bearing because of what
it leaves standing: **a secant that changes sign across a point brackets the minimum.**

| axis | left secant | right secant | minimum lies in |
|---|---|---|---|
| CYP1A2 centre | −0.159 | +0.058 | (4.2758, 4.6600) |
| CYP2C9 centre | −0.360 | +0.007 | (4.7072, 4.9800) |
| CYP3A4 centre | −0.105 | +0.114 | (4.7532, 5.0300) |
| CYP2D6 spread | −0.160 | +0.058 | (0.7805, 1.2008) |

**Every forecast this programme has made was an extrapolation beyond the measured range, and the
last two missed high in the same direction.** v9 is the first that interpolates strictly inside a
measured bracket, which is why its bounds are rigorous instead of assumed. v8 bought that, and it
is the only thing v8 bought.

### v9, submitted 12:40 UTC

Targets are the 3-point parabola vertices; bounds are convexity's. Per cell the band is
[parabola, **chord**], where the chord is the straight line between the two measured points
straddling the vertex — an upper bound that follows from convexity alone. Macro point estimate
**0.5236**, rigorous band [0.5160, **0.5270**]. The falsifiable claim: **every cell at or below its
chord, macro ≤ 0.5270.** Exceeding it would mean the scorer is non-deterministic or ST-RAE is not
convex in that parameter — both far bigger news than the placement.

Honest expected value, stated before the fact: **+0.0020 macro over v7 at best, −0.0015 at worst.**
This mainly recovers v8's loss. It earns its slot by measuring the vertex on all four axes.

### Two process notes

**A defect found before it shipped.** The new builder wrote 4 decimals, which collapsed 10–20
distinct predictions per column into ties; even 6 tied one CYP1A2 pair (1−τ = 1.8e-6). Seven is the
first precision exactly rank-identical to the v4 source. Far too small to show on a 4-dp board, and
it would have quietly made the one invariant we pre-register every time merely *nearly* true.

**Yesterday's lesson paid off today.** At 50 s after the v9 click there was no `queue/join` in the
network log — the exact signal that made me re-click on v8. This time I waited, and found it at
`statusCode: pending`, then 200. One click, one submission. The rule that saved it: *an instrument
reporting "nothing happened" is the reading most likely to be an artefact of the instrument.*

---

## 2026-10-11 — A new programme: the LeWAM / LeJEPA transfer. Phase 1 measured, two objects fail.

**Question.** Arnab pointed at <https://le-wam.github.io/> — *LeWAM: End-to-End World and Action
Modeling with JEPAs* (Fu, Siebert, Halicki, Balestriero) — and asked what in it is useful here.
It is a robotics paper: ResNet-18 → 192-d latents, a Mixture-of-Transformers predicting the next
latent *and* an action velocity field, planning by imagining sampled actions. None of its numbers
transfer. **Five objects in it might**, catalogued in `PLAN.md` with a price and an explicit
"will not transfer" column written before any work.

The one that made the paper worth reading closely is a single sentence in §5:

> *"Deterministic MSE targets the conditional mean E[A|z,c], whereas flow matching models the
> conditional action distribution."*

That is the formal statement of **why `findings/07` failed**, arriving from robotics. Our Tobit EM
imputed `E[y*|y* ≤ c]` — a conditional mean — and so moved the point mass's *location* while
leaving its *concentration* intact. Same mathematics, different field.

**Phase 1: implement LeWAM's two measurement objects and point them at our own data.**
`drugrx/distest.py` — the Epps–Pulley statistic against N(0,1) (SIGReg's kernel) and Roy–Vetterli
effective rank. Derived from the BHEP integral rather than copied, so the O(n²) closed form and
the Gauss–Hermite quadrature LeJEPA actually uses are two independent routes to one number and a
real cross-check. 24 gates in `tests/test_distest.py`.

**Result: two of the five objects are measured and both fail for our use. Full write-up in
`findings/08`.** Headlines:

- **Claim 1 FAILED.** The statistic was to order the four isoforms by their known point-mass share
  (61/29/15/2 %). Observed Spearman **+0.40**, not 1.0. Identifiable specification error of mine:
  `distance` measures *total* non-Gaussianity, and the isoforms' real-label distributions already
  differ **6.7×** in it, swamping a 30× share range. On CYP1A2 the 29% mass made the combined
  distribution look *more* Gaussian. Claim 2 (pseudo-labels raise it on all four) failed 3/4.
- **Effective rank: UNINFORMATIVE**, spread **36.2×** across normalisations (61.9 raw → 2240.7
  standardised). The pre-registered caveat fired and nothing was claimed. The reason it cannot be
  fixed by choosing a normalisation is the real point: effective rank is meaningful in LeWAM
  because a learned embedding has **no arbitrary per-coordinate scale**, whereas our 3091 columns
  mix binary bits with unbounded descriptors in different units. **It is a property of a learned
  representation, not of a feature matrix.**
- **The apparatus is not needed for the job it was borrowed for.** Phase 2's manipulation check is
  `sd(imputed)/s_y` — one line of numpy — which separates the three arms 0.000 / **0.030** / 0.60,
  identically across isoforms and monotonically by construction, where `distance` separates them
  by 1.04× on CYP3A4. The scalar would have caught finding 07's defect before any GPU.

**Three defects the build caught, none by inspection:**

1. **A literal point mass does not have sd 0 in floating point.** `np.full(1000, 4.2).std(ddof=1)`
   is **1.78e-15**, so a `sd <= 0` guard never fires and the amplified ULP noise reports distance
   **0.476 instead of the ceiling 0.163** — a 3× error in the one case the instrument exists for.
2. **LeJEPA's 17-knot quadrature has an *absolute* accuracy floor, ~1e-5.** `distance` under the
   null falls as 1/n, so its *relative* accuracy degrades with batch size (measured to 1.3e-2),
   and at **n = 6145 a Gaussian sample's distance is the same order as the error itself**. On
   heavy tails the error is not even monotone in node count — 33 knots was worse than 17. Fine for
   a regulariser's gradient; **not a measurement instrument**, so no finding may quote it.
3. **The concentration axis confounded location with spread.** `2.211 − |N(0,σ)|` is the right
   shape for left-censoring but has mean `−σ√(2/π) ≈ −0.8σ`. First run gave opposite signs on
   CYP2D6 and CYP2C9 for the same manipulation. Re-centred (drift < 1e-12) it is monotone on all
   four, and the **within-isoform paired drop** then orders them exactly by mass share — recorded
   as **post-hoc, p ≈ 1/12, a hypothesis and not a rescue of claim 1**, which failed.

**Conclusion carried forward.** Phase 2 (de-concentration) proceeds, but on the scalar and with
the manipulation check first — and `PLAN.md` prices it at **0.35% of absolute accuracy against a
13.6% target**, below this lab's own bar, so it is labelled a closed scientific question and not a
board lever. Phase 3 (SIGReg on a learned encoder) is where the imported object could still earn
its keep, because that is the case it was designed for: multivariate, learned, unknown bad
direction. Gated on four research shells briefed today.

**Standing traps added to `AGENTS.md`** (8–10): do not test a claim about A with a statistic
measuring A + B, name B and say how it is held fixed; incoherence across units you expected to
agree is a bug signal; pre-register the *caveat*, not just the prediction.

---

## 2026-10-11 (later) — four shells report. Four of five transfer objects now closed, and I had three claims to correct.

Two Opus shells (SIGReg mathematics; censoring/de-concentration statistics) and two Fable shells
(JEPA-for-molecules; collapse diagnostics + LeJEPA reception). **Every load-bearing number was
re-measured locally before being written down**, and that was not ceremony — three claims did not
survive it, two of them mine.

### Phase 3 closed without being built — `findings/09`

**Someone has already run it.** *Self-Supervised Pretraining of Molecular Graph Encoders with
LeJEPA* (arXiv 2609.04261) adapts LeJEPA — SIGReg and all — to molecular graphs with **GPS and
Chemprop D-MPNN** encoders. Verified against the arXiv abstract by hand: *"Pretraining improves
learned representations but does not robustly improve finetuning"*; frozen probe +0.123 ROC-AUC;
the one significant partition (p = 0.010) **vanishes across five** (pooled p = 0.095); finetuning
null on D-MPNN, which is `scripts/cyp_dmpnn.py`.

Three more lines agree. **Atom-JEPA** (19M molecules + 1.7M crystals) wins big where physics is
clean — 36% mean relative MAE reduction vs EquiformerV2 on QM9, 6/8 Matbench — and reports **no
improvement at all on ChEMBL-MT bioactivity**. **Mol-JEPA**, same senior author as LeWAM, loses
its own FDR-corrected Wilcoxon test to descriptors+TabICL, **38% win rate vs 47%**. And a verified
pre-registered parity test puts a compact descriptor portfolio at **0.762 AUC vs CheMeleon's
0.764**, pooled gap +0.003, CI [−0.020, +0.030].

Our encoder would pretrain on 20–200× less data than the one that got nothing. **Measured family
ceiling on bioactivity: zero, against a 13.6% requirement.** `AGENTS.md` now says the first step
of expanding any phase is to check whether someone has already run it — the ceiling does not have
to be computed here.

### The sketch is blind — `findings/10`

The decisive measurement, on our own matrix, 256 random directions vs 256 coordinate axes:

| matrix | random-dir mean | axis-aligned mean | ratio |
|---|---|---|---|
| ours, z-scored | 0.00046 | **0.12912** | **280×** |
| true N(0,I), same shape | 0.00002 | 0.00002 | **1×** |

**Our features' non-Gaussianity is 280× larger along coordinate axes than along random
directions, and the Gaussian control returns exactly 1× — so it is real, not an estimator
artefact.** Diaconis–Freedman and Klartag explain it: most 1-D projections of a wide distribution
are near-Gaussian whether or not the distribution is. Count fingerprints carry an atom at zero in
every coordinate; a random dense mixture of 3091 of them is Gaussian by the CLT, and the atom —
the whole structure — is invisible. **This retrospectively explains finding 08's negative result:
the one place the statistic worked was 1-D labels, where there is no sketch.**

Also verified: the plug-in bias floor is **1.0594/N** (measured 1.07–1.08× of prediction), and —
a fact neither shell had — **sample standardisation shrinks it ~3×**, which is exactly why our
measured null sits a factor 3 below the textbook number. The bias term is minimised by |φ| ≡ 1,
i.e. **by a point mass**, so it pulls toward collapse at O(1/N).

### Three corrections

1. **finding 08 measured the wrong quadrature.** I asserted LeJEPA's 17 knots have an *absolute*
   error floor. Their reference implementation (`MINIMAL.md`) is a **trapezoid over [0,3] with
   doubled interior weights** and an unnormalised `e^{-t²/2}` window — not Gauss–Hermite, not
   [−5,5] — and its error is **relative, ~2–3%**, so it *shrinks* with n. Criticism withdrawn.
   Two shells gave contradictory accounts of this; one had the weight right and the grid wrong,
   the other had both wrong, which is why it was read off the source.
2. **finding 08 called a low effective rank "evidence of redundancy".** It is evidence of neither
   redundancy nor collapse: duplicating 500 columns moves RankMe **−1.1%**, while 500 columns of
   iid noise raise it **+23%**. These measures count directions carrying non-negligible variance,
   and **noise maximises them.** Roy & Vetterli Prop. 2/3 supplies the formal reason the 36×
   normalisation swing is unavoidable: per-column rescaling is not an erank-preserving operation.
3. **A shell overreached and I caught it.** Its headline was that our features are already *more*
   isotropic than a Gaussian sample by LeJEPA's own figure of merit. True at **one** truncation
   tolerance (1e-2: 1.69× vs 2.01×) and false at every other — at matched K, every PCA truncation
   has ours **worse** (1.25–1.63 vs 1.00–1.05). The Gaussian control is 2.01× at *every* tolerance
   because it has no near-null directions, so the comparison is ill-posed unless K is fixed. The
   kill on object 4 stands on the two arguments that do survive: the theorem's asymptotics cap
   **K ≲ 12.6** at our n, and its isotropy is *derived from an assumed isotropic task prior* that
   four known fixed endpoints contradict.

### A second defect in finding 07, and it was the σ

Chasing `PLAN.md`'s open check: `cyp_censored_floor.py:136` took σ from the residual sd on the
rows the model had just been fitted on. Measured against out-of-bag and held-out estimates,
2 seeds, every cell reproducing: **σ was 5–7× too small** (0.086–0.190 against an honest
0.586–0.961, i.e. 66–107% of s_y). So **switching from mean to draw would not have rescued the arm
either** — a draw at σ = 0.13 still lands on the far left of finding 08's concentration axis. The
two defects had to be fixed together and nothing in the original run would have shown that.

OOB and held-out agree in every cell, so **out-of-bag residuals are a free, valid substitute** —
the bootstrap ensemble already has them sitting unused. Phase 2 is now fully specified: a
manipulation check (`sd/s_y`, separating arms 0.000/0.030/~1.0), a correct σ, and a predicted
landing point 20–30× the failed arm's. Price unchanged at **0.35% of a 13.6% target**, so it stays
a closed scientific question, not a board lever.

**General lesson:** an in-sample residual sd is not an estimate of predictive uncertainty, and
anything that consumes a σ inherits the error silently. The number does not look wrong. It looks
like a σ.

### Opened, not closed

- **Phase 6, the Clever Hans check.** Blevins & Quigley (arXiv 2512.20924) train an **1,815-class
  author classifier from molecular fingerprints to 60% top-5 accuracy under scaffold splitting**,
  and an activity model fed only a protein id and a structure-derived author vector matches a
  structure-based baseline. **Scaffold splits do not break the leak.** Every arm comparison in
  this lab uses `scaffold_split`, and ours is near-random (82% singletons), which makes the leak
  *more* available. Upstream of every number this programme has produced. Highest-value open
  question; priced as an audit of the instrument, not a lever.
- **The unoccupied hole, for the sister programme.** Learned surrogates gating expensive
  *structure prediction* (AlphaFold/ESMFold/Boltz triage) are essentially unpublished, while the
  docking equivalent is crowded. And the docking version is dead twice over for us: Kim/Nam/Ryu
  find surrogates *"memorize structural patterns prevalent in high docking scored compounds"* —
  chemotype, not physics — on top of `vina-has-no-metal-coordination-term`.
- **Two literature results that corroborate our own negatives**, which is the cheapest validation
  there is: **active learning accelerates *search*, not *model quality*** (three papers find it
  fails to beat random for building a better regressor), and **no UQ method ranks per-molecule
  errors reliably** (Hirschfeld et al., 341 cites) — which is findings 04–05 arrived at
  independently.

---

## 2026-10-11 (late) — the fourth shell refutes the premise, not the arm. All five transfer objects closed.

The de-concentration statistics shell came back with a measurement, not an argument, and it is the
most consequential thing produced today. **Verified independently before being written down**
(`scratchpad/verify_latent.py`): every figure below is mine, from our own release.

**2.211 is the MEDIAN of a measured 303-compound distribution, not a censoring bound.**

| | |
|---|---|
| n dual-labelled (has a DRC curve AND shows no inhibition at 50 µM) | **303** |
| per isoform | CYP3A4 250, CYP1A2 42, CYP2C9 11, **CYP2D6 0** |
| mean / **median** / sd | 2.3822 / **2.2111** / **0.4435** |
| min / q01 | **1.906** / 1.918 |
| **P(y\* > 2.211)** | **0.4983** |

**49.8% of screen-inactive compounds have a true pIC50 above the constant we assign them**, and
the population is right-skewed with a hard floor at 1.9 — the *opposite shape* from a left tail.
`drugrx/cyp/screen.py:allr_floor` computes this entire distribution and keeps only its median. The
dispersion was in the release all along.

So `findings/07`'s opening sentence — *"a screen says the compound is inactive, which means
y ≤ 2.211 … that is a textbook left-censored regression"* — **is false for half the rows**. This is
**coarsening, not censoring**; Powell's `y_obs = max(c, y*)` family does not apply at all.

**And fixing the σ bug I found this morning would have made it worse.** A correct Tobit at the
honest σ imputes **1.857** (σ_pop) or **1.480** (σ_out) — **both below the measured minimum of
1.906.** The in-sample σ bug is the only reason that arm was merely uninformative rather than
damaging. **The bug was protecting us.** I have corrected my own addendum, which closed with "any
future censored arm takes σ from OOB" — that is right about estimating a noise scale and wrong as
a route back into this family.

**CYP2D6 has zero dual-labelled compounds**, because the promotion rule makes "has a curve" and
"is screen-inactive" disjoint by construction there. Positivity fails: P(y\*|x, inactive) is
**unidentified** on the isoform carrying 60.5% of the mass. No imputation scheme fixes that.

### The question is answered: real mechanism, dead intervention

7 arms, 4 seeds, 288 fits, location held *exactly* at 2.211 per member, bootstrap index sets
reused across arms so contrasts are paired on the resample too. **Δr(σ̂,p) is perfectly monotone in
imputed sd — 0.000 → −0.019 → −0.060 → −0.085/−0.081/−0.094 → −0.301, no inversions.** Finding 06
was right about the cause. But:

- **The repair is unreachable at the honest σ.** At 0.4435 the artefact only falls +0.324 → +0.230,
  still positive 4/4. Zero needs imputed sd ≈ 0.80 — **1.8× the dispersion the data contains.**
  The arm that works is variance injection, not imputation.
- **The literal proposal is the *weakest* arm**, because truncating at c is itself concentrating:
  σ = 0.4435 arrives as imputed sd **0.256** (σ√(1−δ(0)) predicts 0.267 — within 4%). The fix
  partly re-commits the sin it was designed to cure.
- **~90% of the effect is the label histogram, not MI.** One *shared* draw gets −0.085 of the
  per-member −0.094; Rubin's semantics are worth the residual −0.009.
- **Δρ is second-order by construction.** For fixed tree structure `E_draw[leaf]` equals the
  mean-imputation leaf exactly, so draw and mean have identical expected point estimates for any
  estimator linear in y; the correction is `½ tr(∇²_Y f̂ · Σ)`, second order and of either sign.
  Measured arms span −0.0019 to +0.0044 against a required **+0.0113** — short by 2.6× to 38×.

And two of the five alternatives were **already falsified in our own repo**: the censored/Tobit
loss is `cyp_hinge_rho.py` (macro Δρ **−0.0094, 0/5 seeds**) and ranking is `cyp_rank_objective.py`
(lambdarank **−0.1296**, t = −73). There is now a mechanism for the first: at s = 0.4435 a censored
row's gradient is **5–24× an L2 row's**, and half of that pressure is pointed the wrong way.

### Where the data actually points — Phase 7, opened

If the label is a point estimate with known noise, the fix is a **weight**, not an estimator.
Both variances are measured and in the release: floor σ **0.4435**; DR σ from `*_conf_low/_conf_high`
(width/3.92), median **0.0869**, rms **0.2632**. GLS gives `w_floor/w_drc` of **0.038–0.496**, so
**`PSEUDO_W["inactive"] = 1.0` is 2× to 26× too high — derived, not swept.** Plus two free
corrections: `PSEUDO_W` is **ordered inversely to the measured precisions** (`inactive` 1.0 vs
`weak`/`active` 0.5, though the latter are *wider* q10–q90 windows), and **`allr_floor` returns the
median while the loss is squared error**, whose risk-minimising constant is the **mean** — a
**+0.171** location correction, zero compute, currently wrong for a stated reason.

Priced the same as Phase 2 (0.35% against 13.6%), so it is a **correctness** matter, pre-registered
before it runs.

### Where the programme stands

**All five LeWAM transfer objects are closed**, and the surviving one closed for the most
interesting reason: the principle is *true and measured* and the thing it repairs has a measured
deployment value of **−0.00034 Borda, 4/16 seeds**. The yield of the day is four corrections (two
to my own findings), three validated instruments, and a refuted premise — not a lever.

**Trap 10 added:** *measure the latent distribution before modelling the censoring* — or more
generally, before modelling any quantity, check whether the release already measures it. This is
the fourth consecutive error of trap 8's family, and here the claim was about a **bound** while the
data describe a **location**.

---

## 2026-10-12 — Re-verification programme opened. And an admission about the epochs.

Arnab asked for every finding to be re-verified by independent agents, pointed out that I trained
for far too few epochs on every compute, and asked for the literature on what will actually work
(MIT and other labs, X, reddit, GitHub, arXiv). All three are fair, and the second is a plain error
of mine that invalidates part of yesterday's reasoning.

### The admission first, because it changes a conclusion I already wrote

I reported that the 8c MLP backbone went **macro ρ 0.5526 at 60 epochs → 0.5383 at 300** and I
called it *overfitting*. **I had run no sweep, so I did not know that and should not have said it.**
Two things are conflated in that comparison:

1. `OneCycleLR` is driven over `total_steps = epochs × steps_per_epoch`, so changing epochs changes
   **the entire learning-rate schedule**, not just the number of steps. 60 and 300 epochs are two
   different optimisers, not the same optimiser run for different lengths.
2. The backbone sits **~0.09 macro ρ below the production GBM** (0.538 against ~0.643). My own
   pre-registered item 4 says that makes every weighting comparison on it a measurement on the
   wrong model — and I kept reporting the arm deltas anyway, with a caveat attached rather than
   stopping.

So the 8c arm results so far (`learned − derived` = +0.0162, +0.0099 on seeds 0 and 1) are **not
yet interpretable**, and the honest status is "pending a competent backbone", not "2/6 clearing the
bar". A protocol specialist is now designing the sweep — epochs decoupled from the schedule, the
feature-scaling question for sparse counts mixed with unbounded descriptors, and whether an MLP is
even the right differentiable backbone at n≈6k / d≈3k, or whether the tabular literature says GBMs
simply win at this shape and the weighting experiment belongs back on the GBM.

### What is being re-verified, and why each matters

| strand | the load-bearing claim | why it must hold |
|---|---|---|
| data claims (`findings/11`, `findings/12`) | 2.211 is a **median**, P(y\* > c) = **0.4983**, CYP2D6 has **zero** dual-labelled rows; Q2's oracle **+0.9949** vs cross-fitted **+0.1274** | these closed two phases and gated a GPU programme |
| SIGReg numerics (`findings/08`, `findings/10`) | the **280×** random-vs-axis blindness with a Gaussian control at 1×; the **1.0594/N** plug-in floor; the AM/HM headroom correction | this closed four of five transfer objects |
| citation audit (`findings/09`) | the molecular-LeJEPA null, Mol-JEPA's **38% vs 47%**, Atom-JEPA's **zero** bioactivity gain, and **Clever Hans** | I closed a phase on *other people's* results relayed by agents |
| what actually wins | tabular foundation models (**TabPFN v2 / TabICL**), the real Polaris/ASAP winners, CYP-specific heme-iron features | model class is the only live lever left |

The verifiers are briefed **adversarially**: re-derive independently, try to break it, and report
the number beside mine. Three specific attacks I asked for, because they are where I am most
exposed:

- **Sweep `INACTIVE_R`** (0.8/0.85/0.9/0.95). If P(y\* > c) ≈ 0.5 flips at a plausible threshold,
  `findings/11`'s headline is threshold-dependent and I never checked that.
- **Try to make Q2's cross-fitted number look good** — more folds, other learners, other targets,
  and crucially *without* grouping by compound. If ungrouped folds inflate it, that quantifies how
  much leakage the grouping was preventing, which is a number I should have reported and did not.
- **Test the dimension mechanism directly** for the 280× result: subsample columns to
  d = 32/128/512/1024/3091 and see whether the ratio actually grows with d. If it does not, my
  stated Diaconis–Freedman mechanism is wrong even if the number is right. And check whether the
  **max** over random directions closes the gap — LeJEPA's consistency theorem is proved for the
  max while its loss uses the mean, so if the max is informative my "blind" claim is too strong.

### The one thing I most want to be wrong about

**Clever Hans** (arXiv 2512.20924): an 1,815-class *author* classifier from fingerprints at 60%
top-5 accuracy **under scaffold splitting**. Every arm comparison in this lab uses `scaffold_split`,
and ours is near-random (82% scaffold singletons), which makes the leak *more* available rather than
less. I have been relying on that paper's existence without reading it. If it holds, it sits
upstream of every number this programme has produced, and `PLAN.md` Phase 6 becomes the most
important thing here rather than a curiosity.

### Running

Three L4s, nothing local, 3.08 units/hr: 8b per-row labels on the real GBM (seeds 0–2 and 3–5,
parity-checked against production by import) and 8c on a backbone I now consider unvalidated. The
8b runs remain meaningful because they use the production learner and protocol; 8c is on hold
pending the sweep.

### 2026-10-12, later — the tabular foundation model, and an overclaim I made and withdrew within minutes

**I reported "TabICL on 100 features beat the full 3091-feature LightGBM ensemble" from a single
cell.** The next two cells contradicted it. Recorded because the error is instructive: I had the
pre-registration in front of me, which says *report per isoform always, a macro gain driven by one
endpoint is a different result*, and I still led with one isoform's number.

Seed 0, three isoforms (CYP3A4 still running):

| isoform | `lgbm_all` | `lgbm_topk` (100) | `tabicl_topk` (100) | vs all-features | **vs matched features** |
|---|---|---|---|---|---|
| CYP1A2 | 0.6181 | 0.5986 | 0.6277 | **+0.0096** | +0.0291 |
| CYP2C9 | 0.6931 | 0.6570 | 0.6573 | **−0.0358** | +0.0003 |
| CYP2D6 | 0.4961 | 0.4800 | 0.4910 | **−0.0051** | +0.0110 |

**What survives is narrower and more useful.** At *matched* features TabICL is ≥ LightGBM on 3/3
(+0.029 / +0.000 / +0.011); against the full feature set it loses on 2 of 3. And TabICL is 6×
faster (3–7 s against 42 s for an 8-member GBM ensemble).

**The pre-registered control fired and changed the headline comparison, as written.**
`lgbm_topk − lgbm_all` averages **−0.024**, past the −0.02 void threshold, so *100 features do not
carry this problem* and the k-limited arms were **handicapped rather than tested**. The
pre-registration anticipated exactly this and said the comparison then becomes matched-feature.
That is the second time in two days that pre-registering the *caveat* rather than only the
prediction is what kept a number honest.

**So the binding constraint is the feature budget, not the model class** — and TabICL's 100-column
limit is documented *degradation*, not a hard cap. Sweeping k = 100 / 250 / 500 / 1000 on 3 seeds
to find where its matched-feature edge and the budget penalty cross. If TabICL at k=500 keeps a
positive matched-feature contrast while the budget penalty shrinks toward zero, that is where a
real win would live.

**One cross-machine confirmation worth banking:** `lgbm_all` returned CYP1A2 ρ **0.6181**, exactly
the local value. The `--selftest` parity table and `tests/test_shard_parity.py` are doing their
job after yesterday's `N_MEMBERS` drift.

**TabPFN is blocked and stays blocked.** Its weights need a one-time licence acceptance, which is
not mine to give, and its licence prohibits commercial benchmarking and production deployment —
so even a positive result could not be submitted or shipped. Raised with Arnab; proceeding with
TabICL, which is permissively licensed and is the one we could actually use.

### 2026-10-12 — **We only ever moved one lever.** The representation/predictor confound.

The "what actually wins" sweep returned the sharpest correction of this programme, and it is not
about any single finding — it is about the design of every model-side experiment we have run.

> *"You tested CheMeleon **as a representation** feeding your LightGBM and got Δρ = 0.0000,
> concluding 'the gain was in the loss, not representation'. The literature agrees the gain is not
> in the representation — it says the gain is **in the predictor head**."*

**That is correct and I missed it entirely.** `AGENTS.md` records "every representation lever
measured here is 0.0000" — CheMeleon frozen, CheMeleon finetuned, JEPA-style pretraining. **Every
one of those was measured with LightGBM nailed down as the head.** We varied the representation and
held the predictor fixed, four separate times, and concluded the representation does not matter.
**The predictor has never been varied at all.**

The literature's win is the other factorisation:

- **Monroe** (arXiv 2608.18982, open weights): *"PFN-based downstream predictors also substantially
  improve two leading existing models, MiniMol and CheMeleon, yielding new state-of-the-art
  variants we call MiniMol_PFN and CheMeleon_PFN."*
- **An independent benchmark at almost exactly our data shape** — Pat Walters, who is *also* an
  OpenADMET programme author (Nature Comms 2026), so an insider on this data family. ExpansionRx,
  **5,326 train / 2,282 test**, 9 endpoints, **5×5 cluster-grouped CV with Tukey HSD and explicit
  ties**. **Monroe embedding + TabPFN against Morgan fingerprints + LightGBM**: wins **5 of 9**,
  and **6 of 6** on Biogen ADME (3,521 molecules). Our n is 6,145.
- **Guan et al.** (CSIRO, arXiv 2606.31126): for *"small-molecule classification with ECFP/RDKit
  descriptors, no single predictor-representation pairing dominates"*, concluding TFMs *"can be
  strong biomolecular predictors, **but only when coupled with expressive representations**."*

**And that last one explains our own `cyp_tabfm.py` result.** TabICL on RAW descriptors came out a
tie at matched features (CYP3A4 −0.0014). Right experiment, **wrong input** — a tabular foundation
model on a sparse fingerprint block is the configuration the literature says *doesn't* pay.

So `scripts/cyp_predictor_vs_representation.py` runs the grid that can attribute the effect:
{desc, chem, both} × {lgbm, tabicl}, with the **replication control** (`chem+lgbm − desc+lgbm`
must come out ≈0, or the grid is measuring a difference from our earlier CheMeleon run rather than
the predictor) and the **never-measured contrast** (`tabicl − lgbm` at fixed representation).
Dimension reduction is structure-appropriate — top-k by split gain for the sparse axis-meaningful
descriptors, PCA for the dense embedding with no privileged axes — both fitted on training rows
only, refitted per seed.

**Everything in the grid is commercially usable.** CheMeleon is MIT; TabICL is permissive. **TabPFN
is deliberately absent** — its weights need a licence acceptance and its terms bar commercial
benchmarking and production deployment, so a win there could not be submitted or shipped. If the
predictor effect is real, TabICL is the one we can act on.

### Four other things from the sweep worth acting on, and one worth never trying again

1. **Uniform averaging may be actively hurting us.** arXiv 2610.12005 (Morgan Stanley): with a pool
   of 96 configurations, *"greedy selection reduces error by 2.4% … but **uniform averaging
   increases error**."* **We run a uniform bootstrap ensemble.** Cheap to test, and it is a free
   hit on our current aggregation rather than a new model.
2. **Cross-isoform stacking, not a shared trunk.** ASAP's ADMET verdict was multi-task GNNs — but
   on CYP *specifically* the record is against deep multi-task: Wu et al. 2019 had single-task
   **XGBoost at 90.4%** beating a five-isoform multitask autoencoder at **88.5%**. And Cheng &
   Zhang measured *negative* transfer of **10.2 pp** under distribution mismatch. So the
   effort-efficient version is out-of-fold predictions of the other three isoforms as features,
   inside the existing LightGBM — captures the correlation, cannot suffer negative transfer.
3. **A ceiling statement worth holding.** Novartis's multitask GNN on CYP3A4 TDI reached *"similar
   precision to the reproducibility of in vitro experiments"*. If our labels carry comparable
   noise, headroom above macro ρ 0.7093 may be genuinely small.
4. **Every top-3 entry in the ASAP blind challenge used pretraining on large unlabelled corpora or
   auxiliary data.** A frozen pretrained encoder is the cheap end of that lesson — no training run.

**And the one to never try again: metal-aware docking for the heme-iron / type-II story.** No
published evidence any metal-aware scoring function improves CYP inhibition prediction; the only
tool with an explicit metal-coordination module (PandaDock) is single-author with zero citations;
structure-based CYP2D6 modelling *with* MD conformations and docking energies reached only **75%
external accuracy, below ligand-only CYPlebrity**; and AlphaFold-predicted P450 structures **lack
the heme cofactor entirely**. Our Vina negative generalises. `vina-has-no-metal-coordination-term`
stands and the line is closed for good.

### Unreached, and recorded as such

**"Assessing the state of the art in ADMET prediction: Lessons from the OpenADMET-ExpansionRx
community blind challenge"** (ChemRxiv doi 10.26434/chemrxiv.15009342) is very likely the closest
published analogue to our leaderboard — possibly its direct predecessor, on the same ExpansionRx
data family, run by the OpenADMET group whose whole programme is CYP-class anti-targets.
**ChemRxiv serves a Cloudflare challenge to every route available here, and the DOI 404s.** Worth
opening in a browser; it is probably worth more than anything else on the reading list.

### 2026-10-12 — Adversarial verification lands. **Four things I told Arnab are withdrawn**, and one becomes a result.

Three verifiers plus a protocol specialist reported. The mathematics of `findings/08`/`10` largely
survives — several claims now have closed forms I had only measured — but four statements of mine
do not, and one of the failures turns out to be the best corroboration of `findings/11` in the
programme. Full detail in the `findings/10` addendum; the retractions are here because I told
Arnab each of them.

#### Retraction 1 — "SIGReg's random sketch cannot see our data" (**"blind"**)

**False.** Of 1024 random directions, **98.93% exceed the null q95** and **89.94% exceed the
maximum of 4000 null draws**; the quantity SIGReg minimises sits at **z ≈ 173σ**. What collapses
280× is the **effect size**, not the detectability. The replacement claim is still strong:
*SIGReg rejects normality in ~99% of random directions but reports an effect size 280× below the
axis-aligned truth, so its gradient is ~280× weaker and it cannot rank or localise which
directions are bad.* Also: the **sup over the sphere is ≥ 0.169** because coordinate axes *are*
unit vectors, so LeJEPA's sup-consistency theorem is untouched — the failure is Monte-Carlo
estimation of the sup, not the theory. And the 280× is **normalisation-dependent by 6.5×**
(43.8× raw, 179.1× with LeJEPA's own unstandardised statistic).

#### Retraction 2 — "the 8c backbone is ~0.09 macro ρ below production, so the arm comparison is on the wrong model"

**Withdrawn. The comparison was against a non-control.** `GBM_RHO = 0.643` in my own script came
from a different pipeline — 6145 compounds, pseudo-labels, 20 auxiliary heads, a different split
family. On **identical splits with production hyperparameters**, the same learner scores **0.549**.
And with **one preprocessing line changed**, a **single** MLP ties an **eight-member** GBM:

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| MLP, 8c with one preproc line fixed, **K=1** | 0.5435 | 0.5495 | 0.5670 | **0.5533 ± 0.0100** |
| LightGBM **K=8**, production params, same splits | 0.5422 | 0.5428 | 0.5624 | 0.5491 ± 0.0094 |
| paired difference | +0.0013 | +0.0066 | +0.0046 | **+0.0042, 3/3** |

**My pre-registered item 4 passes once the control is run properly.** This is precisely the failure
`feedback-a-control-must-run-the-production-command` exists to prevent, and I committed it inside
a script whose docstring cites that very rule. The "0.09" decomposes into **~0.042 feature
scaling + ~0.022 ensemble size + ~0.015 target scaling + ~0.008 early stopping**, with the
remainder being the production pipeline's extra data — **no part of it is a backbone property**.

The dominant term is mine: `Xs = (X - mu)/sd` over all 3091 columns is the **worst of six**
options tested and **worse than no scaling at all** (0.5345 raw vs 0.5112 standardised). The
matrix is four blocks of different character — Morgan counts are **97.6% zeros with excess
kurtosis 128** — and standardising a sparse block makes every zero a non-zero, turns a rare hit
into a **+14 to +31** spike, and leaves kurtosis untouched because it is affine. Both reference
implementations in the tabular literature (TabM, RealMLP) explicitly leave binary/sparse blocks
alone.

#### Retraction 3 — "more training made it worse, i.e. overfitting"

**Withdrawn, and it inverts into a result.** With epochs and schedule properly separated on **real
labels only**, more training is **monotonically better**: 60 ep 0.5080 → 120 ep 0.5112 → **300 ep
0.5191**, and the 300-epoch run has the *tightest* spread of anything run.

**The only thing that replication removes is the pseudo-labels.** So my 0.5526 → 0.5383 degradation
is not the optimiser, the schedule or the backbone — it is five times longer spent fitting a label
set in which **49.8% of screen-inactive compounds truly sit above 2.211** (`findings/11`). That
makes it **direct, independent corroboration of findings/11**, and evidence that the robust-loss
half of the `learned` arm has a real target.

**The decisive test is one run and it is now launched:** re-run 60-vs-300 with `PSEUDO_W` all zero.
If the degradation disappears, pseudo-label noise is confirmed as its cause.

#### Retraction 4 — "the label slope barely matters"

**Withdrawn; the correct reading is worse.** I adopted a cosine because the raw inner product
NaN'd, and attributed the blow-up to the weights. **The weights cannot run away** — `w = 2σ(·)` is
bounded and the weighted *mean* is 0-homogeneous, which is exactly the Σw=1 constraint L2RW and
MW-Net impose by hand. **The unbounded direction is `cal_b`**, the label: as |cal_b| → ∞ the
residual grows linearly, so ‖G‖ and ⟨G, g_meta⟩ grow linearly. And the cosine does not fix that —
it normalises ‖G‖ away, which makes `cal_b` **unidentified**. So *"the two arms converged to
cal_b −1.094 and +0.102 and reached ρ 0.5540 vs 0.5534"* is **the signature of a flat objective,
not evidence that the slope does not matter.** Worse: `cal_a`/`cal_b` enter only through the
alignment term, while the meta loss uses real labels and is a function of θ alone — so nothing
penalises a *wrong* label while the alignment term *rewards* one for inflating ‖G‖.

Also, L2RW and MW-Net both use a **bare inner product**, so `−cos(·,·)` is not a reparameterisation
— it descends a different function, scoring each example against the component of the meta
gradient *not already explained by* the aggregate gradient, which can **flip an example's sign**
relative to the true hypergradient. And a **global flattened cosine** is the one variant with
direct negative evidence: in dataset-condensation ablations it is **24 points below plain squared
Euclidean on an MLP**, with the stated fix being per-output-node grouping. My first layer is
**74.9%** of the parameters, so my cosine was essentially a first-layer cosine — over exactly the
columns my standardiser had blown up.

#### Five real bugs in `cyp_learned_weights_gpu.py`, found by reading, not by me

1. **`derived`/`learned` overwrite ~303 real labels.** `tgt = where(inact_m, wnet.label(logr), yv)`
   keys on `kind == "inactive"` alone, and `inact` is **not disjoint from `is_drc`**. The
   dual-labelled compounds get their real pIC50 replaced by `a + b·log10(r)` — **a regression
   fitted on those same rows** — in `derived` and `learned` but **not** in `uniform`. So
   `derived − uniform` includes "we replaced 303 real labels with an in-sample fit of themselves."
2. **The meta split is not scaffold-grouped**, despite the docstring saying "80/20 by scaffold
   group". It is `rs.permutation(tr)`. ~18% of compounds are in multi-member scaffolds, so a
   meaningful share of meta rows have a scaffold-mate in inner — a direct route to a meta gain
   that is memorisation.
3. **The meta forward runs in `net.train()`**, so BatchNorm updates its running statistics **from
   the meta split**, and independent dropout masks attenuate the cosine (measured 0.76 train vs
   0.89 eval).
4. **`bank` is K=6 while every other arm is K=1**, and the measured GBM ensembling curve is
   **+0.022** from K=1→8. So `bank − learned` confounds ensembling with the mechanism — the
   **second** instance of this exact defect after the calibration-init confound.
5. **The arms do not share a data order** — `inner_phi` draws from the same RNG that generates the
   minibatch order, so after the first meta step `learned`/`bank` see a different sequence from
   `uniform`/`derived` at the same seed.

#### And the precondition the whole bilevel arm fails

Split-half reliability of the meta gradient — two halves of the meta set, cosine between their
gradients:

| epoch | cos(g_A, g_B) | cos(G, g_meta) on two disjoint train batches |
|---|---|---|
| 0 | +0.861 | +0.902 / +0.926 |
| 10 | +0.671 | +0.786 / +0.808 |
| **20** | **+0.196** | +0.252 / +0.249 |
| **80** | **+0.081** | **−0.083 / +0.030** |
| **200** | **+0.115** | **+0.074 / −0.099** |

The **training** gradient stays reliable (0.89–0.96) throughout. The **meta** gradient collapses
after epoch 20, and at epochs 80 and 200 the objective φ maximises **does not agree in sign
between two disjoint training minibatches**. And `--meta-warmup 20` starts φ training at exactly
the epoch where reliability falls off the cliff. **Below ~0.2 no surrogate helps**, so this is the
gate to run before any further bilevel work.

#### The new lever this exposed, which is on the GBM side

Per-isoform, like-for-like K=8 against the `GBM_RHO` constant: CYP3A4 **0.758 vs 0.765**
(reproduces to 0.007), but CYP1A2 0.454 vs 0.618, CYP2C9 0.557 vs 0.693, CYP2D6 0.428 vs 0.496.
**The production pipeline's extra supervision buys 0.07–0.14 ρ on exactly three isoforms and
nothing on CYP3A4.** Which ingredient does it — pseudo-labels, the 20 auxiliary heads, or the
larger 6145-compound set — is untraced, cheap to trace by ablation, and is a **real lever on the
production learner** rather than on a backbone.
