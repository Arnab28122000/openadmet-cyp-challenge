# PLAN — the LeWAM / LeJEPA transfer

Opened **2026-10-11** from a reading of *LeWAM: End-to-End World and Action Modeling with JEPAs*
(Fu, Siebert, Halicki, Balestriero — <https://le-wam.github.io/>, PDF
<https://minghaofu.github.io/files/lewam.pdf>, code <https://github.com/MinghaoFu/lewam>; arXiv
"coming soon" as of today). Predecessor LeWM: <https://le-wm.github.io/>.

One phase is expanded at a time. A phase is not started until the one before it has produced a
number, and no phase is started without a price in σ\* and a pre-registered decision rule.

---

## The source, stated accurately

LeWAM is a **robotics** paper: image observations, ten simulated manipulation/navigation tasks,
three physical robot tasks. A shared ResNet-18 encoder maps frames to 192-d latents; a
Mixture-of-Transformers predicts the next latent *and* an action velocity field; planning samples
actions, imagines their latent consequences, and refines. The objective is

    L_LeWAM = L_dyn + L_act + λ_sig · L_SIGReg

with `L_dyn` and `L_act` plain MSE, and

    L_SIGReg = (1/M) Σ_{m=1..M} T( Z u^(m) )

where `u^(m)` are M random unit directions, `Z` is the minibatch of latents, and `T` is the
**univariate Epps–Pulley** statistic (Epps & Pulley 1983) against N(0,1). The paper's config uses
**1024 projections and 17 knots**, and reports **effective rank** (Roy & Vetterli 2007) as its
anti-collapse diagnostic.

**None of its numbers transfer to us.** No molecules, no regression, no small-n tabular data. What
transfers is three objects, and the honest claim for each is below.

| # | object in LeWAM | our analogue | transfers? |
|---|---|---|---|
| 1 | Epps–Pulley-on-random-projections as a **differentiable distributional distance** | an instrument that measures *concentration* of a distribution | **CLOSED, `findings/08` + `findings/10`.** Works in 1-D; the random **sketch is 280× blind** on our wide sparse matrix (axis-aligned deviation 0.129 vs random-direction 0.00046; Gaussian control returns exactly 1×). For the 1-D job a scalar sd ratio beats it |
| 2 | **effective rank** as the collapse readout | redundancy of our 3091-d feature matrix / of CheMeleon embeddings | **CLOSED, `findings/08`.** 36× normalisation-dependent (per-column rescaling is not erank-preserving, Roy & Vetterli Prop. 2/3), and not a redundancy detector either way — **noise maximises it** |
| 3 | *"MSE targets the conditional mean, flow matching models the conditional distribution"* | **finding 07's exact open question**: mean imputation re-concentrates, draws do not | **CLOSED, `findings/11`.** The principle is **true and measured** — Δr(σ̂,p) is perfectly monotone in imputed sd, 0.000 → −0.301 — and worth **nothing here**, because Δρ is second-order by construction and σ̂'s deployment value is already measured at −0.00034 Borda, 4/16 |
| 4 | SIGReg as a **training regulariser** | a representation lever on a learned molecular encoder | **CLOSED, `findings/09` + `findings/10`.** Family ceiling on bioactivity measured at **zero** by three other groups, one of whom ran this exact experiment. And the theorem does not reach us: its asymptotics cap **K ≲ 12.6** at n = 6145, and its task-agnosticism axiom `E[∇η∇ηᵗ] = τ²I` is false with four known fixed endpoints |
| 5 | latent planning over learned dynamics | a surrogate for an expensive simulator, to choose what to run | **yes, but not for CYP.** Phase 4, redirected |

### What will NOT transfer, written before the work

- **Scale.** SIGReg is demonstrated in the image-SSL regime. We have n ≈ 6145 compounds and
  d ≈ 3091 features — d is half of n. Every distributional-shaping argument is asymptotic in the
  batch; a 6k dataset in 3091 dimensions is not where "the embedding distribution" is well
  estimated. Expect the optimality argument to degrade and say so if it does.
- **No augmentation family.** JEPA needs a view-generating process. Molecules have no uncontested
  augmentation (fingerprint dropout and scaffold masking both change the molecule's semantics).
  This is the single most likely reason Phase 3 fails.
- **The scoring-function defect is upstream of any surrogate.** Phase 4's naive form — learn a
  surrogate of AutoDock Vina and plan in its latent space — **inherits Vina's missing
  metal-coordination term** (`vina-has-no-metal-coordination-term`; ketoconazole's imidazole
  docked 13.4 Å from the heme Fe). A learned surrogate of a wrong scoring function is wrong
  faster. Phase 4 is therefore not a CYP lever and is redirected to FoldRx, where the expensive
  step is not Vina.
- **Our production model is gradient-boosted trees on fixed features.** Trees are invariant to
  monotone per-feature transforms, so *nothing* that reshapes the marginal geometry of a fixed
  feature matrix can change a GBM's predictions. Any claim that SIGReg helps the current pipeline
  directly is false by construction, and Phase 3 is honest only as "train an encoder instead".

---

## Pricing, in the lab's own units

σ\* = s_y·√(1−r²), r = 2·sin(πρ/6). **The target is a 13.6% cut in absolute error.**

| phase | what it buys | priced | verdict |
|---|---|---|---|
| **1 — instruments** | measurement capability, not accuracy | cost ≈ 0 (CPU, minutes) | **build.** It improves pricing itself, which is the one thing never priced |
| **2 — de-concentration** | ~~finding 07's family ceiling~~ | **premise refuted** | **CLOSED 2026-10-11 WITHOUT A FULL BUILD, `findings/11`.** 2.211 is the **median of a measured 303-compound distribution**, not a bound — **49.8% of the rows lie above it**, and the population has a hard floor at 1.906. A correct Tobit imputes **below the measured minimum**. CYP2D6 has **zero** dual-labelled compounds, so its floor law is **unidentified** |
| **3 — SIGReg on a learned encoder** | ~~representation is the one lever AGENTS.md still calls live~~ | **measured family ceiling on bioactivity is ZERO** — see `findings/09` | **CLOSED 2026-10-11, before being built.** Someone has run this exact experiment (molecular LeJEPA, arXiv 2609.04261): frozen-probe +0.123, **finetuning null**, effect gone across five partitions, D-MPNN included. Atom-JEPA at 19M molecules gets **no** bioactivity gain. Mol-JEPA, same senior author as LeWAM, loses its own Wilcoxon test to descriptors+TabICL 38% vs 47% |
| **4 — latent planning over a simulator** | compute saved in FoldRx, not ρ on the CYP board | different programme, different currency | **defer.** Not a CYP lever; see the Vina note above |

**Phase 2's price is below the lab's own "several percent" bar.** That is stated here, in advance,
so the result cannot later be dressed up as a board play. It is being run because the scientific
question is open and the instrument to settle it is now free — not because it will move a rank.

---

## Phase 1 — The instruments *(expanded; in progress)*

**Question.** Can LeWAM's two measurement objects be implemented correctly, validated against
cases with known answers, and then told something we do not already know about our own data?

**Deliverable.** `drugrx/distest.py` — Epps–Pulley against N(0,1), both the exact O(n²) closed
form and the O(n·K) knot-quadrature form LeJEPA actually uses, plus Roy–Vetterli effective rank.
Unit-tested in `tests/test_distest.py`.

**Validation gates — the statistic is not used on real data until all of these pass.**

1. Standard normal samples ⇒ statistic near its null value; calibrated rejection rate ≈ α.
2. A point mass ⇒ statistic large. Monotone in the mixing fraction of the mass.
3. The two implementations (exact vs 17-knot) agree to within quadrature error on the same sample.
4. `effective_rank` of a rank-k matrix is k; of an isotropic Gaussian matrix, ≈ d.

**Then, applied — and pre-registered as to what would mean something:**

- **(a) The point mass, in test-statistic units.** Epps–Pulley on each isoform's labelled training
  target distribution. Prediction: CYP2D6 (61% mass) ≫ CYP3A4 (2% mass), and the ordering of all
  four should follow the mass share monotonically. *If it does not, the statistic is not measuring
  what I think it is and Phase 2 does not start.*
- **(b) Effective rank of the feature matrix**, per isoform's training rows, on raw / centred /
  standardised / row-L2-normalised versions. Pre-registered caveat: for a **fixed, sparse,
  hand-engineered** matrix a low effective rank is evidence of redundancy, **not** of the
  representational collapse the SSL literature uses it to detect. The normalisation choice is
  reported for all four because it is expected to matter, and a number that flips qualitatively
  across normalisations is reported as uninformative rather than as a finding.
- **(c) The concentration axis.** The statistic evaluated along a mixing fraction from 0 to 0.61,
  which is the dose-response curve finding 07 needed and did not have.

**Falsifiable claim for Phase 1.** The statistic orders the four isoforms by their known point-mass
share. This is a genuine test: the shares are 61 / 29 / 15 / 2 %, known independently, and the
statistic has never seen them.

---

## Phase 2 — De-concentration *(CLOSED 2026-10-11, `findings/11`)*

Closed on a **false premise** rather than a failed bar, which is a better reason than the one this
plan expected to write. One `groupby` on data already in the release — 20 seconds — measured the
latent distribution the whole censoring frame assumed away: median 2.2111 (**= the constant**),
mean 2.3822, sd 0.4435, min 1.906, **P(y\* > 2.211) = 0.498**. Right-skewed with a hard floor, the
opposite shape from a left tail. This is **coarsening, not censoring**.

The de-concentration *mechanism* is confirmed — Δr(σ̂,p) perfectly monotone in imputed sd across
seven arms — and the *intervention* is dead three times over: the repair needs 1.8× more
dispersion than the data contains; the literal Tobit-draw proposal is the **weakest** arm because
truncation is itself concentrating (σ 0.4435 → imputed sd 0.256); and ~90% of the effect comes
from the label histogram rather than from multiple-imputation semantics.

**All five catalogued transfer objects are now closed.** The programme's yield is in the
corrections and the instruments, not in a lever.

## Phase 7 — Inverse-variance weighting *(NEW, where the data actually points)*

Not an estimator — a weight. `findings/11` §"Where the data actually points". Both variances are
measured and already in the release: floor pseudo-label σ = **0.4435**; DR label σ from the
release's own `*_conf_low/_conf_high` (width/3.92), median **0.0869**, rms **0.2632**. Under
squared loss the efficient weight is `w ∝ 1/σ²`, giving `w_floor/w_drc` between **0.038** and
**0.496** — so **`PSEUDO_W["inactive"] = 1.0` is 2× to 26× too high**, *derived* rather than swept.

Two free corrections fall out of the same argument:
- **`PSEUDO_W` is ordered inversely to the measured precisions**: `inactive` 1.0 vs `weak`/`active`
  0.5, while the latter are q10–q90 windows *wider* than 0.4435.
- **`allr_floor` returns the median while the loss is squared error**, whose risk-minimising
  constant is the **mean** — a **+0.171** location correction, zero compute, currently wrong for a
  stated reason.

Priced the same as Phase 2 (**0.35% against 13.6%**), so it is a **correctness** matter with a
derivation, not a board lever, and it is pre-registered before it runs like anything else. Note
the `down` arm at w = 0.2 already saw the direction (ρ up on 4/4 isoforms, t = 1.92, 10/16) from
*below* the derived bracket.

## Phase 2 (historical) — the design as originally specified

The experiment `findings/07` declined to build, now cheap to do correctly. Arms: `full` (point
mass), `tobit-mean` (the known fail), `tobit-draw` (independent truncated-normal draw per ensemble
member). **The manipulation check comes first and can void the arm**: the draw arm must move the
imputed distribution's Epps–Pulley distance from a point mass by a pre-registered amount, measured
*before* any ρ is looked at. Finding 07's error was running an arm that did not vary the quantity
its claim was about; this phase cannot repeat that error, by construction.

Open check carried in from reading the code: `cyp_censored_floor.py:136` sets σ from the residual
sd on dose-response rows, which should be ≈0.6–0.9 pIC50 units, yet the imputations had sd
0.030–0.084. A near-constant μ across inactive rows and a single global σ would explain it. **Verify
the actual σ before drawing conclusions about why the mean arm concentrated.**

## Phase 3 — **REOPENED 2026-10-12** after a citation audit found my closure was wrong

**The closure below was wrong and is retracted.** See the dated addendum in `findings/09`. I closed
a **low-data** programme on a **high-data** null: Atom-JEPA's bioactivity null is at **114,112
labels**, while the same paper's pretraining *wins* are at **3,521** (−9.6%, p = 0.011, 6/6) and
**7,608** (−8.5%, p < 0.001, 9/9) — and our per-isoform real-label counts are **1285–2335**, the
scale where it won. The authors also explicitly disown that null as hyperparameter-confounded, and
the "36% vs EquiformerV2" figure I used is bundled with an architecture upgrade, not a pretraining
measure at all.

**THE REOPENED FORM, and it is narrow and specific.** Not finetuning, and not SIGReg as a
regulariser. The one variant with good controls is **concatenation**: a truncated pretrained
embedding ⊕ the existing fingerprint. Molecular LeJEPA reports 0.803 → **0.830** (+0.027, CI
[+0.003, +0.054], p = 0.014), with an untrained-encoder control at **−0.003** and the pretrained
embedding beating its untrained twin in **all twelve cells**.

**And the reason it belongs here specifically:** that gain is **zero on validation** (0.847 vs
0.845, *"well within noise"*) and appears only under chemotype shift — *"it is adding signal that
survives the change of chemotype."* Our validation is a **diversity scaffold split** (within-set
similarity 0.354) while the board's test set is a **purchased analog expansion** (0.665), and no
re-split of our data can emulate it. **So our validation instrument is structurally blind to
exactly this kind of gain.** That is the single most important methodological consequence in this
programme: a lever whose benefit only appears under shift will be rejected by every measurement we
currently make.

Which means the reopened phase needs an **evaluation** change before a model change: build a
shift-holdout (analog-expansion-like re-split, or a time/assay-disjoint split) and verify a known
lever reproduces on it, before trusting it to grade anything.

## Phase 3 (superseded text) — SIGReg where it could act *(closed 2026-10-11, never built)*

Closed on price by `findings/09`, on four independent literature lines, two of them verified
against the primary sources by hand. The decisive one is a **direct replication of this phase** —
LeJEPA with SIGReg on molecular graphs, GPS and Chemprop D-MPNN encoders — whose abstract states
*"Pretraining improves learned representations but does not robustly improve finetuning"*, with
the single significant partition (p = 0.010) vanishing across five (pooled p = 0.095).

**The pre-registered failure mode was the right one.** `PLAN.md` named "no uncontested molecular
augmentation family" as the most likely reason Phase 3 would fail. Mol-JEPA replaced augmentations
with modality masking for exactly that reason — and still lost its own significance test to
descriptors plus a tabular foundation model.

**This closes transfer objects 4 of 5.** Three of the five catalogued objects are now measured
and all three fail here; the surviving one (object 3, "MSE targets the conditional mean") needs
none of the imported apparatus.

## Phase 4 — Latent planning over an expensive simulator *(deferred, redirected to FoldRx)*

The sweep sharpened this considerably and it is now **two** different propositions:

- **Docking surrogate — dead twice over for CYP.** The field is crowded and the numbers are real
  (Deep Docking 100×, DDU 45× on top of that, MolPAL 94.8% of top-50k at 2.4% docked, Thompson
  Sampling >half the top-100 at 1%). But Kim/Nam/Ryu (arXiv 2406.12919) find surrogates
  *"memorize structural patterns prevalent in high docking scored compounds obtained during
  acquisition steps"* — they learn the enriched chemotype, not the docking function. Ranking, not
  calibrated scores. Add our own `vina-has-no-metal-coordination-term` and there is nothing here.
- **Structure-prediction triage — an unoccupied hole.** Learned surrogates deciding whether an
  AlphaFold / ESMFold / Boltz-2 run is worth paying for are **essentially unpublished**. That is
  `docs/foldrx-pipeline/` and `talindrew-nim` territory, where the expensive step is not Vina, and
  it is the one place LeWAM's actual contribution — planning over learned latent dynamics to avoid
  paying for the real step — has an unclaimed application. **Belongs to the sister programme, not
  this lab.** Raise it with Arnab rather than starting it here.

And the framing constraint, from three independent papers: **active learning accelerates *search*,
not *model quality*.** Any acquisition system here must be scored on "did it find the top-k
cheaply", never on "did it make the regressor better" — where random is a genuinely hard baseline
that AL frequently loses to.

## Phase 5 — Write-up, memory, changelog *(continuous, not terminal)*

---

## Phase 8 — Per-row pseudo-labels *(Arnab's, 2026-10-11; 8a closed, 8b/8c running)*

Arnab asked for a learned per-prediction weight **matrix**, a bank of such weightings, an ensemble
over them, and dynamic learning rates on L4s/A100s. `findings/12`.

**Not priced away by Phase 7**, because the scalar family's +0.0037 ceiling bounds *scalar*
weightings only — 1 parameter against 24,580 cells — and because `screen_labels` thresholds the
continuous readout `r = 2^log2fc` at 0.9 and **discards the within-bin position**.

**8a, the gate, closed.** On the 303 compounds where the pseudo-label error is observable:
- **Q1 PASS** — Spearman(r, signed err) **−0.377** (p < 1e-4), holding within isoform
  (CYP1A2 −0.614, CYP3A4 −0.234). True mean y\* across terciles of r: **2.59 / 2.30 / 2.25**, a
  0.34 spread = 77% of the floor sd. **It is a LOCATION effect, so the per-row object is the
  LABEL, not the weight.** The constant 2.2111 scores **R² = −0.149** against the floor
  population — worse than its own mean.
- **Q2 FAIL** — oracle in-fold **+0.9949**, cross-fitted **+0.1274**, R² **−0.048**, 1/5 folds at
  bar. A **+0.87** oracle gap. A weight-net on structure would be fitting noise.

**Design consequence:** the weight-net reads `[log10(r), kind, CI width, per-row loss]` and never
the 3091 features. Without the gate, the natural build would have produced a convincing and
worthless result.

**8b** — `base/meanfloor/rowlabel/rowiso/rowboth/bank`, `g(r)` cross-fitted per split (the 303
carry real DR labels and some land in validation). **`bank` is Arnab's idea done properly**: each
member gets its own calibration from a bootstrap of the in-split dual sample, so the ensemble
averages over *calibration uncertainty* — and unlike `findings/11`'s draw arm, it has a downstream
consumer: the point estimate.

**8c** — learned `w_i = 2σ(h_φ(u_i))` and label `a + b·log10(r)`, trained by bilevel lookahead
against a **meta** split carved out of *train* so validation is never tuned on; OneCycle on the
backbone, cosine on φ after a uniform-weight burn-in. Sharded across L4/A100 sessions with
lightgbm pinned to match local.

**Bar: `learned` must beat `derived` by macro Δρ ≥ +0.005 on ≥5/6.** Beating `uniform` is not
sufficient — `derived` is free and already does that (+0.0050 on one seed).

## Phase 6 — The Clever Hans check *(NEW, opened by the sweep 2026-10-11)*

The sweep's most valuable product was not a closure but a **threat to this lab's validation
instrument**. Blevins & Quigley (arXiv 2512.20924) train an **1,815-class author classifier from
molecular fingerprints** to **60% top-5 accuracy under scaffold splitting**, and show an activity
model fed only a protein id and a structure-derived author-probability vector matches a
structure-based baseline. **Scaffold splits do not break the leak.**

Every arm comparison in this lab and the sister programme uses `scaffold_split`, and
`cyp-set-is-82pc-scaffold-singletons` means ours is **near-random** — which makes the leak *more*
available, not less. We hold ChEMBL provenance already (`scripts/cyp_chembl_build.py`,
`scripts/chembl_audit.py`), so the test is cheap: can our 3091 features predict assay / document /
source identity, and does an assay-disjoint split change any conclusion this lab has drawn?

This is now the highest-value open question the programme has, because it is upstream of every
number in it. **Priced as: not a lever at all — an audit of the instrument.** Expand after Phase 2.

## Open questions carried in from the four research shells

Four were briefed 2026-10-11: SIGReg mathematics (Opus), censoring/de-concentration statistics
(Opus), JEPA-for-molecules literature (Fable), collapse diagnostics + LeJEPA reception (Fable).
Their answers gate Phase 3 and refine Phase 2. Recorded in `NOTEBOOK.md` on arrival.
