# Changelog

Newest first. Each entry: what changed, why, and the commit.

## 2026-10-10

- **Lab opened.** `docs/research-lab/` created with `AGENTS.md` (durable orientation, the σ\*
  pricing rule, the closed-lever table, the measurement-discipline list), `NOTEBOOK.md` (the ELN),
  `PLAN.md`, this changelog, and `findings/` + `experiments/`.
  Rationale: the DrugRx programme had measured capacity, data, placement/decode and docking to
  exhaustion; **representation / model class** was the only unmeasured lever, and it needed a
  programme of its own rather than another one-off.
- **Six research strands briefed** in parallel across Opus / Sonnet / Fable, including one
  deliberately decoupled theory strand briefed abstractly with no chemistry.
- **v8 submitted** 17:22 UTC, pre-registered beforehand in `scripts/cyp_v8_prereg.py`; v7's scored
  row (macro 0.5256, rank 54/271) recorded in `cyp_board_history.py` first. Poller armed.
  Runbook §2 and the `cyp-challenge-leaderboard-ui` memory rewritten with three facts learned
  during the submission: the 12 h window is enforced **to the second from the board's timestamp**
  (a trigger built from our own click time fired 51 s early); `POST /gradio_api/queue/join` 200 is
  the **real receipt** and arrives in seconds, where the status panel takes ~90 s and the
  `queue/data` SSE **503s routinely**; and `read_network_requests` **lags**, so an absent POST is
  not evidence a click failed. That last one nearly caused a duplicate submission — only the rate
  limiter made the second click harmless.
- **CYP2D6 σ̂ anomaly resolved** (`scripts/cyp_sigma_anomaly.py`, pre-registered `dd1652a`;
  `findings/06`). It is an artefact of the screen-inactive pseudo-label point mass at 2.211 —
  61% of CYP2D6's labelled signal at full weight — which makes the ensemble spuriously *agree*
  at low predictions. Partial correlation on p is +0.038 (positive 4/4) against a raw −0.164
  (negative 4/4), so it was never about uncertainty; the dose-response moves it +0.273 and flips
  r(σ̂,p) from 4/4 positive to 0/4. Our own stated mechanism was inverted in both clauses.
  Deleting the floor costs −0.0516 ρ on CYP2D6, so the principled fix is censored regression
  (Chernozhukov–Hong), not deletion. The 0.2-weight ρ gain (+0.0035 pooled) is **not claimed**:
  t = 1.92, 10/16.
- **Censored-floor line closed** (`scripts/cyp_censored_floor.py`, pre-registered `829eb5d`;
  `findings/07`). Both claims failed: r(σ̂,p) on CYP2D6 fell only +0.336 → +0.264 and stayed
  positive 6/6, and neither arm cleared the macro Δρ ≥ +0.005 / ≥5-of-6 bar (`down` +0.0037 at
  5/6, `tobit` +0.0012 at 3/6). The weight hack beat the principled estimator on all four
  isoforms. The pre-registered falsification of finding 06 was **mis-specified**: the EM step
  imputes a conditional mean, so it moved the point mass's location (2.211 → ~2.10) while leaving
  its sd at 3–5% of s_y — it never varied concentration, which is what finding 06 is about. Run
  recorded as uninformative about 06, not as evidence against it, and the fail recorded as a fail.
  Line closed on price: measured ceiling +0.0037 < bar, and a σ̂ repair has no consumer now the
  2-D decode is closed.
- **v8 scored and regressed** (macro 0.5256 → 0.5520, rank 54 → 69; all four cells worse). Only
  ρ/τ passed; the headline "no centre cell may rise above its v7 value" was falsified by all
  three, and the CYP2C9 dead-axis trigger (|d| > 0.005) fired at 0.0491 exactly as written.
  Convexity was *not* refuted — the claim's own proviso ("provided we do not cross the balance
  point") failed. **v9 submitted 2026-10-11 12:40 UTC** (`scripts/cyp_v9_bracket.py`,
  pre-registered `82e80bf`): the first placement forecast that interpolates inside a measured
  bracket, since v8's overshoot produced a secant sign change on all four axes. Macro point
  estimate 0.5236, rigorous chord band ≤ 0.5270. Found and fixed a real defect before shipping:
  4-decimal output tied 10–20 predictions per column, 6 tied one pair, 7 is exact.

## 2026-10-11 — the LeWAM / LeJEPA transfer programme

- **`PLAN.md` created** (it had been referenced by `AGENTS.md` since the lab opened and never
  existed). Holds the phased LeWAM transfer: five catalogued objects, each priced in σ\*, with an
  explicit **"will not transfer"** section written before any work — scale (n≈6145, d≈3091), no
  uncontested molecular augmentation family, and the fact that **GBMs are invariant to monotone
  per-feature transforms**, which makes "SIGReg helps the current pipeline" false by construction.
- **`drugrx/distest.py`** — Epps–Pulley against N(0,1) (exact O(n²) closed form *and* the
  Gauss–Hermite knot quadrature LeJEPA uses), `sigreg()` over random projections, Roy–Vetterli
  effective rank, RankMe, and a Monte-Carlo null. Derived from the BHEP integral, not copied, so
  the two forms cross-check.
- **`tests/test_distest.py`** — 24 validation gates, all passing. Two caught real defects: the
  floating-point point-mass guard (`np.full(1000,4.2).std(ddof=1)` = 1.78e-15 ⇒ distance 0.476
  instead of 0.163) and the 17-knot *absolute* accuracy floor (~1e-5, comparable to the null
  signal at n = 6145).
- **`scripts/cyp_distshape.py`** — Phase 1 on real data. `data/cyp_cache/distshape.json`.
- **`findings/08`** — Phase 1 verdict. **Claim 1 FAILED** (Spearman +0.40 vs a required 1.0:
  `distance` measures total non-Gaussianity, which varies 6.7× across isoforms on the real labels
  alone). **Effective rank UNINFORMATIVE** (36.2× across normalisations; it is a property of a
  learned representation, not a feature matrix). **The one-line scalar `sd(imputed)/s_y` beats the
  imported apparatus** for the job it was borrowed for, separating the arms 0.000/0.030/0.60.
- **`AGENTS.md` traps 8–10 added**: don't test a claim about A with a statistic measuring A+B;
  incoherence across units you expected to agree is a bug signal; pre-register the caveat.
- **`findings/10`** — the SIGReg audit, from two Opus shells, with every load-bearing number
  **re-measured locally** rather than trusted. Closes transfer objects 1 and 4 quantitatively:
  * **the random sketch is 280× blind on our data** — axis-aligned Epps–Pulley 0.1291 vs
    random-direction 0.00046, with a true-N(0,I) control returning exactly 1×. Diaconis–Freedman
    / Klartag: most 1-D projections of a wide distribution are near-Gaussian regardless.
  * the plug-in bias floor **1.0594/N** confirmed (measured 1.07–1.08× of prediction), plus a
    fact neither shell had: **sample standardisation shrinks it ~3×**, which is why our measured
    null sits a factor 3 below the textbook figure.
  * LeJEPA's theorem caps **K ≲ 12.6** at n = 6145 (`N·r₀^K → ∞`), and its isotropy follows from
    an assumed **isotropic task prior** we do not have — four known fixed endpoints.
- **Three claims corrected, two of them mine.** (a) `findings/08` attributed a 17-knot *absolute*
  error floor to LeJEPA; I had measured Gauss–Hermite, while their trapezoid-on-[0,3] has a
  **relative** ~2–3% error that shrinks with n — criticism withdrawn, and the real floor at our n
  is the *statistical* term, 50× the quadrature one. (b) `findings/08` called a low effective rank
  "evidence of redundancy"; it is evidence of neither — duplicating 500 columns moves RankMe −1.1%
  while 500 noise columns raise it **+23%**. (c) a shell claimed our features are already *more*
  isotropic than a Gaussian sample by LeJEPA's own figure of merit; that holds at **one**
  truncation tolerance and fails at every PCA truncation, where ours is 1.25–1.63× against
  1.00–1.05× — the Gaussian control is 2.01× at *every* tolerance because it has no near-null
  directions, so the comparison is only well-posed at matched K.
- **`drugrx/distest.py`** gained `epps_pulley_lejepa` (LeJEPA's exact trapezoid-on-[0,3] with the
  doubled interior weights and unnormalised `e^{-t²/2}` window, read off `MINIMAL.md` after two
  shells contradicted each other about it), a `beta`-parameterised exact closed form checked
  against brute-force numerical integration at three betas, and the published RankMe epsilon
  convention reproduced deliberately (ε *outside* the normalisation). 30 gates.
- **`findings/11`** — **2.211 is the median of a measured 303-compound distribution, not a
  censoring bound.** Verified independently from the release: mean 2.3822, median 2.2111 (= the
  constant), sd 0.4435, min 1.906, **P(y\* > 2.211) = 0.4983**, right-skewed with a hard floor.
  `allr_floor` computes it and keeps only the median. So `findings/07`'s frame is false for half
  the rows it applies to — **coarsening, not censoring** — and a correct Tobit at the honest σ
  imputes **1.857 or 1.480, both below the measured minimum**. *The in-sample σ bug found this
  morning was the only thing preventing active harm.* CYP2D6 has **0** dual-labelled compounds, so
  its floor law is unidentified on the isoform carrying 60.5% of the mass.
  - De-concentration: **mechanism confirmed, perfectly monotone** (Δr(σ̂,p) 0.000 → −0.301 across
    7 arms, 288 fits); **intervention dead** — the repair needs 1.8× the dispersion the data has,
    the literal Tobit-draw proposal is the *weakest* arm (truncation is itself concentrating,
    0.4435 → 0.256), ~90% of the effect is the label histogram not MI semantics, and Δρ is
    second-order by construction (−0.0019..+0.0044 against a required +0.0113).
  - Two of five alternatives already falsified in-repo: censored loss = `cyp_hinge_rho.py`
    (−0.0094, 0/5), ranking = `cyp_rank_objective.py` (−0.1296, t = −73) — now with a mechanism
    (a censored row's gradient is 5–24× an L2 row's, and half of it points the wrong way).
- **`findings/07` addendum corrected the same day**: "any future censored arm takes σ from OOB" is
  right about noise-scale estimation and wrong as a route back into the censoring family.
- **Phase 2 CLOSED on a false premise rather than a failed bar; Phase 7 OPENED** — inverse-variance
  (GLS) weighting, where both variances are measured and already in the release. `w_floor/w_drc` =
  **0.038–0.496**, so `PSEUDO_W["inactive"] = 1.0` is **2–26× too high, derived not swept**. Plus:
  `PSEUDO_W` is ordered **inversely** to the measured precisions, and `allr_floor` returns the
  **median** under a **squared-error** loss whose optimal constant is the **mean** (+0.171, free).
- **All five LeWAM transfer objects now closed.** `AGENTS.md` trap 10 added.
- **`findings/12` + Phase 8** — Arnab's per-prediction weight-matrix hypothesis, gated and
  redirected. **Q1 PASS**: the pseudo-label error depends on the continuous readout
  (Spearman(r, signed err) **−0.377**, within-isoform too), and across terciles of r the true mean
  y\* runs **2.59 / 2.30 / 2.25** — a location effect, so **the per-row object is the LABEL, not
  the weight**. The production constant 2.2111 scores **R² = −0.149** against the floor
  population. **Q2 FAIL**: the error is not predictable from the 3091 features out of fold —
  oracle **+0.9949** vs cross-fitted **+0.1274**, R² **−0.048**, a **+0.87** gap. So the weight-net
  reads 1-D meta-features only; a net on structure would have fitted noise with a +0.99 in-sample
  story.
- New scripts: `cyp_weight_predictability.py` (the gate), `cyp_rowwise_labels.py` (8b, with the
  `bank` arm averaging over calibration uncertainty), `cyp_learned_weights_gpu.py` (8c, bilevel
  lookahead against a meta split carved out of train, two-timescale LRs),
  `cyp_rowwise_shard.py` (shardable, `--selftest` asserts the learner against the production
  literal), `cyp_gls_weights.py` (Phase 7 scalar baseline).
- **Four bugs in my own code, caught pre-result**, three of which would have produced a
  publishable-looking number: a meta step that reduced to `mloss.backward()` and gave φ **zero**
  gradient (learned ≡ uniform, silently); an **unbounded** meta objective (NaN, macro ρ −0.40);
  **no minibatching**, so 30 epochs = 30 steps and the MLP looked hopeless against the GBM; and
  `np.clip(NaN, 0.9, None) = NaN` feeding **0 × NaN** into the label-calibration gradient, which
  broke only the arm that differentiates through the label.
- **Environment defect:** the Colab VM shipped lightgbm **4.6.0** against **4.7.0** locally.
  Paired within-seed deltas survive that; pooled absolute ρ does not. Pinned.
