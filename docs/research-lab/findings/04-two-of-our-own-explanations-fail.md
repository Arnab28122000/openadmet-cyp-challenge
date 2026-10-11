# Finding 04 — the final two strands refute one of our explanations and hand us a new decode

Strands: folding-models-as-representation (Sonnet) and decoupled theory (Opus, briefed abstractly
with no chemistry).

---

## 1. "Vina failed because it has no metal-coordination term" is NOT supported

This programme recorded that as the **diagnosed cause** of the docking failure, in `AGENTS.md`, in
the memory `vina-has-no-metal-coordination-term`, and in the retained-artifacts rationale ("so a
metal-aware retry reuses the conformers"). The structure strand measured it on the public training
data and the mechanism does not survive.

Its baseline is a LightGBM on Morgan counts + RDKit descriptors (OOF ρ 0.52 / 0.61 / 0.39 / 0.78).
It built six 2D heme-ligation SMARTS flags, the headline one being an unhindered sp2 N with both
ortho carbons CH:

* **The type-II effect is real but already inside the fingerprints.** On CYP3A4 the flag shifts
  mean pIC50 by **+0.76**, and 8.9% of flagged compounds reach pIC50 ≥ 6 against 1.2% of the rest —
  but the baseline already captures **~80% of that shift**, leaving a mean residual of just +0.14.
* **Correlation of the best flag with the OOF residual is at most 0.07–0.10** per isoform → at most
  **0.4–0.5% σ\***.
* **Adding the flags to the GBM**: Δρ +0.0048 / +0.0027 / −0.0006 / +0.0015, i.e. −0.4% to 0% σ\*.
* **Oracle bound for a *perfect* Fe–N feature**, from the excess residual variance of the flagged
  subset over the unflagged: CYP3A4 **0.2%**, CYP1A2 **0%**, CYP2D6 0–0.3%, CYP2C9 up to 2.4%.
  **Macro ≈ 0.7%.**

So docking did not fail *because* Vina lacks a metal term. **There is almost no residual signal in
the type-II-capable subset for any metal-aware method to find** — the ligand fingerprints already
hold it. The physics strand reached the same ceiling from a completely different direction (only
26–31% of CYP inhibitors carry a coordinating motif ⇒ ≤16.7% even if perfect), and an independent
22,608-docking replication measured +0.0008.

**This is our own rule catching us.** `feedback-test-the-explanation-you-just-gave`: a stated
mechanism is an untested hypothesis. We diagnosed a cause from a single vivid observation —
ketoconazole's imidazole docking 13.4 Å from the Fe — and wrote it into durable docs as settled.
The pose observation was true; **the inference from it to "this is why the features failed" was
never measured**, and it is now contradicted.

The correct statement is narrower: *Vina's poses on CYP are physically wrong, and separately, the
docking features carried no signal. The first does not explain the second, because the signal is
not there to be recovered.* Flagged as heuristic, not proof — the bound is from residual-variance
excess, not a direct test.

**Also settled, and it closes the obvious follow-ups:** protein-side features contribute
**exactly zero** for per-isoform models (they are constants absorbed into the intercept), and for a
joint model four distinct vectors are just an isoform ID that a one-hot already supplies. ESM/AF2
embeddings of four proteins cannot be anything else. Boltz-2's trunk pair representation is
identical across poses (spread ~1e-8) — with a fixed protein it is simply a ligand embedding, and
it scores *below* MiniMol on TDC CYP panels.

---

## 2. The plateau was unidentifiable by a hard limit, and there is a test for it

`[DERIVED]` by the theory strand. For `E(n) = c + A·n^−α`, with `u = ln n`, `θ(u)` the plateau's
share of the error:

> **y′(u) = −α(1−θ)** and **y″(u) = α²·θ(1−θ)**

Two consequences that matter:

* **Aliasing identity.** Over a bounded window, `c + A·n^−α` is indistinguishable from a *pure*
  power law with exponent `α(1−θ̄)`; the mismatch is second-order. **A short window cannot tell a
  plateau from a smaller exponent.**
* **Detectability criterion:** `α²θ(1−θ)L²/8 ≥ 6.71·σ_log/√K`, best case θ=½ ⟹ **(αL)² ≥
  215·σ_log/√K**. Memorable form: **the error must visibly fall by ≈2–5× across your n-range before
  `c` is identifiable at all.**

At our numbers (α=0.05, 6.7× range, K=6, σ_log=0.03) the ratio is **0.0034 — short by a factor of
291**. Simulation agrees, and even a **10⁶×** n-range would not fix it at α=0.05. Our "cannot
tell" was a **fundamental limit of the n-range, not a methodological failure** — which is what we
recorded, and it is now proved rather than assumed.

**The fix is one fewer free parameter, not a better fitter.** Pin `ε₀ = 1` exactly (our metric is
normalised by the constant predictor, so the small-n end is *known by construction*) and estimate
the floor **directly** rather than fitting it — nearest-neighbour residual-variance estimators
converge at √n independently of dimension, which a learning-curve fit never will.

## 3. The single cheapest decisive experiment we have not run

Inside the standard kernel framework, a trace-class kernel with the target *inside* the RKHS gives
excess-risk exponent **≥ ½**. Ours is 0.019–0.079, which requires the source condition `c ≪ 1`:
**the target is far outside the RKHS induced by our representation.** That is the sharpest
available statement that our constraint is representational.

**But it is confounded.** Kpotufe–Martinet show the *source-sample* exponent is attenuated by the
transfer exponent γ, while the *target-sample* exponent is free of γ. We measured α against a
shifted surface, so a small α could mean misspecification **or** shift — and the two imply
**opposite actions** (better features vs. a handful of target-region labels).

> **Measure the learning curve twice — once on an in-distribution held-out split, once on the
> shifted region.** That separates the two, and it is cheap because the curve machinery already
> exists (`cyp_data_scaling.py`, `cyp_scaling_fit.py`).

## 4. The ESS collapse has a name, a theorem, and the literature's prescription

Kish ESS/n consistently estimates `1/E_P[ρ²] = 1/(1+χ²(Q‖P))`, and the minimax rate under
χ²-bounded shift is **exactly the no-shift rate at sample size n/E_P[ρ²] = ESS**. So our 0.24n was
**real and a lower bound over the class**, not an artefact of our estimator. And with a
capacity-saturated (≈well-specified) model, Shimodaira-type theory says importance weighting buys
**nothing** and costs exactly the variance we measured.

The literature's answer to poor overlap is not "weight harder" but **"restrict the estimand and say
so"** (overlap weights, trimming). That is precisely what our own finding
`feedback-weighting-that-concentrates-is-restriction` said, now with the theorem behind it.

## 5. The new lever: the ρ-optimal predictor is the expected RANK, not the conditional mean

This is the one genuinely new, cheap, actionable item in the whole sweep, and it lands exactly in
the gap my decode ceiling left open.

"Oracle post-hoc recalibration is worth nothing" has a name: **MCB** in the MCB–DSC–UNC
decomposition of any consistent scoring function (Gneiting & Resin, EJS 2023). It means our model
is **already monotonically calibrated**. A scalar map can remove miscalibration; **it cannot add
discrimination**. That is why my bin-and-optimise ceiling came back at +0.028 in-sample and
negative cross-fitted — and it is *not* evidence against item-dependent transforms.

Two transformations are **invisible to any scalar map** because they depend on per-item quantities:

1. **The interval-hinge Bayes act** — depends on the conditional skew *and* the per-item interval
   width. Our scored metric is exactly a soft-threshold loss against per-item intervals.
2. **The Spearman-optimal score is the expected rank** (the Borda score), **not the conditional
   mean.** With heteroscedastic uncertainty, `E[rank]` ≠ `rank(E[y])`, and the gap is an
   uncertainty-aware shrinkage toward the centre. Simulated at **+0.015 to +0.096 Spearman**.

Both are functions of `(p, σ̂)`, which is the 2-D family my 1-D ceiling explicitly did not bound.
**This is the next build.**
