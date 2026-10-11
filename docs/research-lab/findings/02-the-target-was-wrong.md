# Finding 02 — the target was 13.6%, not 22%, and we tested the right representation with the wrong head

Two corrections, both of which make the programme's position **better** than it looked, and both
of which came from checking a research strand's claim rather than accepting it.

---

## A. The target, re-derived from the board

`scripts/cyp_real_frontier.py`. 271 entrants, all four isoforms scored.

| | macro σ\* | macro ρ | 1A2 | 2C9 | 2D6 | 3A4 |
|---|---|---|---|---|---|---|
| **preheat-to-450** (best) | **0.7780** | 0.7844 | 0.840 | 0.881 | 0.543 | 0.874 |
| Rubyy | 0.7790 | 0.7819 | 0.855 | 0.870 | 0.532 | 0.870 |
| wbot | 0.7848 | 0.7775 | 0.856 | 0.856 | 0.525 | 0.873 |
| **talindrew** (rank 92) | **0.9004** | 0.7093 | 0.760 | 0.808 | 0.500 | 0.768 |

**Real cut required: 13.6%.** The impossible composite — best ρ on each isoform, held by four
different entrants — is 15.5%. **No one is at a 22% cut, and no one can be.**

The 22% came from a leader σ\* of 0.567. That number does not correspond to any row on the board.
Two ways it could have been produced, both of which this programme should guard against:
per-isoform bests were composed into a single imaginary competitor, and/or an averaged ρ was
converted to σ\* once rather than converting per isoform and averaging (the s_y differ by 1.7×,
and the map is non-linear, so these give different answers).

**Why it mattered.** Every lever for weeks was priced against 22%. The measured model-class
ceiling of 4–8% reads as "hopeless, 3–5× short" against 22%; against the true 13.6% it covers
**29–59% of the gap**. The same evidence supports opposite decisions depending on a number nobody
had re-derived.

### Where our deficit sits, and the two questions that have different answers

| isoform | our σ\* | share of our error | best ρ | cut to best |
|---|---|---|---|---|
| CYP1A2 | 0.910 | 25.3% | 0.8561 | 21.1% |
| CYP2C9 | 0.551 | 15.3% | 0.8808 | 20.0% |
| **CYP2D6** | **1.420** | **39.4%** | 0.5700 | **5.4%** |
| CYP3A4 | 0.721 | 20.0% | 0.8741 | **24.7%** |

The two research strands disagreed about where the problem is, and **both were right about
different questions**:

* The physics strand said CYP2D6 carries ~35% of macro error. **True** — 39.4% by our own
  numbers, the largest share by far.
* The maths-derived view said CYP3A4 is the gap. **Also true** — 24.7% cut needed, our rank 134.

They are not in conflict. **CYP2D6 dominates our absolute error and is simultaneously almost
maxed out**: the best entrant on earth manages ρ 0.543 against our 0.500, and we are rank 28
there. It is hard for everyone and is not where the field's edge comes from.

**But the physics strand priced its top recommendation on "bring 2D6 from 0.449 to 0.850" —
a 15% macro cut from one isoform. The board refutes the premise: nobody exceeds 0.5700.** That
recommendation's value collapses from ~15% to at most ~5.4% × 39.4% ≈ 2% of macro.

Against #1 specifically, our ρ deficits are **1A2 +0.080, 2C9 +0.073, 2D6 +0.043, 3A4 +0.106** —
broadly uniform across three isoforms, which argues for a general representation fix rather than
four bespoke mechanistic ones.

---

## B. Our CheMeleon null was measured with the wrong head

`MODEL-REPORT.md:202` records CheMeleon's frozen 2048-d fingerprint losing to count-ECFP +
descriptors, and `research/24:278` records a **"CheMeLeon-2048 ridge probe → zero blend weight"**.

The 2026 evidence says **that is a head result, not a representation result**, and the head is
exactly the part we got wrong:

* Monroe's paper states the PFN head is transferable and that attaching it produces new
  state-of-the-art variants of **MiniMol and CheMeleon specifically** — i.e. CheMeleon + ridge is
  not the thing that was benchmarked.
* Pat Walters' `model-validation-central` re-runs published methods on identical folds and
  statistics (1,425 fold models, 3.19M predictions) and finds **the representation contributes
  roughly ten times more predictive value than the downstream head** — while also finding frozen
  Monroe at the top of 21 of 27 ExpansionRx combinations and all 18 Biogen ones.
* **A ranked participant in this very challenge did it**: `stir_bar` (interim rank 28/219) used
  **Mitra on frozen embeddings**, reporting Mitra beating TabPFN on 6 of 7 embedding sources.
  Per-isoform ρ 1A2 0.826 / 2C9 0.851 / 3A4 0.863.
* MolPAIR, under a shift protocol matched to ours, finds the **frozen foundation-model
  representation already beats tuned RF/ExtraTrees/XGBoost/LightGBM with no parameter updates**.

Cost to retest is near zero: Monroe is MIT, a 720-d fingerprint, ~300 MB of weights, one-command
inference. Mitra (AutoGluon, Apache 2.0) also sidesteps the TabPFN licence gate noted at
`PROGRESS.md:2665`.

### CORRECTION, after checking our own tree — the claim is mostly wrong

The strand cited `research/24:278` ("CheMeLeon-2048 ridge probe → zero blend weight") as *our*
test. It is not ours. `research/24` is our catalogue of **competitors' public model reports**, and
that line is **JacksonBurns's** dead-end ledger (§46), not this programme's. The same paragraph's
"near-test retrieval on external aux rows (+0.0002)" is likewise a competitor's result — which,
separately, is third-party evidence that **stir_bar's retrieval lever did not transfer**.

What we actually ran, from `PROGRESS.md` and `MODEL-REPORT.md`:

| arm | score | verdict |
|---|---|---|
| frozen CheMeleon 2048-d → GBM | 0.6433 vs count-ECFP 0.5819 | worse |
| frozen CheMeleon → hinge head | 0.6065 vs frozen ECFP 0.5180 | **worse** |
| CheMeleon fine-tuned end-to-end → hinge head | 0.5179 vs ecfp_head 0.5180 | **dead heat** |
| in-context (TabICL/TabDPT) on count-ECFP, top-500 | 0.4597, −0.0032 ± 0.0051 | inside noise |

CheMeleon is recorded **closed at 0.0000** in two separate ledger tables. We also already
diagnosed the PCA confound ourselves — `cyp_chemeleon_icl.py`'s own docstring says the earlier
in-context null "was not a null about in-context learning" because the featuriser was held at
count-ECFP and PCA'd, both documented losing choices.

**What is genuinely untested is one cell: frozen CheMeleon at native width → an in-context head.**
Both of its marginals are null-or-worse here, so my prior is low; the published claim is
specifically about the *interaction*. `scripts/cyp_chemeleon_icl.py` exists for exactly this and
has apparently never been executed. It is cheap (MIT weights, <1 L4-hour), so it is worth running
— but it should be run as a **low-prior confirmation**, not as "the single clearest actionable
item", which is what I called it before checking.

**The lesson, which is the programme's own rule arriving again:** a research strand reading our
repo attributed a competitor's recorded failure to us, and I repeated it. Verify whose number it
is before building on it.

Counter-evidence, kept honest: EMD Serono found SMILES foundation models underperforming graph
models and pre-training past ~10M molecules giving diminishing returns; and the ExpansionRx
winners' board was topped by multitask Chemprop GNNs, with TabPFN *underperforming* at the apex.
The plausible reconciliation is that those winners had 700k–10M-compound proprietary pretraining
corpora — they bought the representation rather than downloading it.

---

## C. The data-quantity bound does not cover what the rank-28 entry did

Our α 0.019–0.079 ⇒ ≥38× bound is about **labels**. `stir_bar` retrieved **88,683 near-neighbours
of the held-out structures** from public catalogues, filtered with rd_filters, and admitted the
**4,999 closest with calculated physicochemical properties only — no assay labels at all**, so
"every backbone encountered the chemical neighbourhood it would be scored within."

That is **unsupervised domain adaptation**, and a label-count learning curve says nothing about
it. This is a genuine gap in our closure — distinct from the Veith/AID-1851 question, which was
about labels and which we correctly declined at 0% incremental coverage.

---

## D. Fingerprint hygiene — a possible artifact in our own MAP4 null

`PROGRESS.md:3884` records `map4 | 0.7023 | +0.0028 | WORSE`. A systematic 2025 benchmark finds
**folded MAP4 has 0.00 Spearman correlation with the graph-based similarity reference, while
unfolded MAP4 has 0.59** — hash-folding collisions distort similarity severely for exactly MAP4
and RDKit path fingerprints. If ours was hash-folded (the RDKit default), that arm measured a
collision artifact rather than MAP4.

Same source: **count beats binary and log-count beats count**, consistently. Our prior citation
that fingerprint size does not matter (Deng et al.) swept radius {2,3} × bits {1024, 2048} on
**folded binary** fingerprints — it does not cover log-count, unfolded/sparse, or frequency-folding.

This matters more than usual here because the blind set is an **analog expansion**: the failure
mode is close analogs mapping to near-identical vectors, which is precisely the specificity axis
on which folded binary ECFP is worst.

*(The inference that our MAP4 null was a folding artifact is mine, not the paper's — it needs the
20-minute check of which variant we computed.)*

---

## E. Verdicts

**Closed, with outside corroboration:**
- Metal-aware docking. Even perfect type-II physics is capped at **16.7%** because only 26–31% of
  CYP inhibitors carry any heme-coordinating motif — and realistically it is 3–5%. An independent
  replication on this same challenge (doctawho42, 22,608 dockings, all four isoforms) measured
  peak signal **+0.0008**, below every isoform's significance floor. Our +0.0068 was the same null
  at smaller scope. Separately: our docking bar (+0.01 Δρ) was itself only a **1.5% σ\* cut** —
  *even a clean pass would not have been worth building*.
- Boltz-2's affinity head for this target, by its own paper: it "processes only a cropped set of
  protein and ligand heavy atoms" and "in cases where essential ions or cofactors are present in
  the binding pocket, affinity predictions will be unreliable." CYP is the canonical case. The
  organisers' own co-folding evaluation found Boltz-2 at 57.8% BiSyRMSD <2 Å on CYP3A4 **despite
  78.9% of those structures being in its training set**.
- Domain-generalisation objectives, GNNs trained from scratch, pairwise/delta learning.

**Live, ranked:**
1. **Frozen CheMeleon at native width → in-context head.** The one untested cell; both marginals
   are null here, so treat as a low-prior confirmation. `cyp_chemeleon_icl.py` already exists.
2. **Unsupervised domain adaptation** — retrieve near-neighbours of the blind structures from
   public catalogues, no labels needed. Not covered by our data-quantity closure.
3. **Fingerprint hygiene** — log-count, unfolded or frequency-folded; recheck the MAP4 variant.
4. **Uncertainty-aware decode** — my decode ceiling ranged over maps of the prediction alone and
   does not bound a rule using σ̂.
5. **Censored/Tobit treatment of the 6,900 screen labels.**

**Open question worth one hour:** order the training release by compound/plate index and roll the
label moments. Two independent ExpansionRx teams found a real assay-protocol change that way and
both gained on the blind board; these organisers have already revised this ground truth once.
