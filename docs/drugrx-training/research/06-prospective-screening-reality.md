# What a virtual screen actually delivers, and what the playground may therefore claim

Research for **Phase 5**, read 2026-10-05. Everything the playground says about its own output has to
survive these numbers, so they are gathered in one place with citations rather than paraphrased into
a caveat.

The programme already carries one number from this literature — CACHE #1's **3.7% hit rate** — and
uses it as the yardstick in `AGENTS.md`, `TARGETS.md` and the model card. **That number is true and
it is the optimistic end of the range.** Three things sit beside it that we were not carrying.

---

## 1. The median blind screening team finds nothing

| | CACHE #1 (LRRK2 WDR) | CACHE #2 (SARS-CoV-2 Nsp13 helicase) |
|---|---|---|
| teams | 23 | 23 |
| teams finding **any** confirmed binder | **7** | **5** |
| overall confirmed hit rate | **3.7%** (73 / 1,955 at K_D < 150 µM) | **0.7%** |

Sources: [CACHE #1 preprint](https://www.biorxiv.org/content/10.1101/2024.07.18.603797.full.pdf),
[Virtual Screening: Hope, Hype, and the Fine Line In Between](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844436/).

**The headline is not "3.7%", it is "most teams got zero".** 16 of 23 groups on CACHE #1 and 18 of 23
on CACHE #2 — professional computational chemistry groups, with the target handed to them — returned
no confirmed binder at all. A playground that presents a ranked list without that framing implies a
base rate it has no right to.

**And 3.7% is measured at K_D < 150 µM**, which is a very weak binder: a starting point for chemistry,
not a candidate. The winning method on CACHE #1 (FRASE-bot) confirmed **8 binders at K_D 3–41 µM from
84 compounds over two rounds**, with SPR *and* ¹⁹F-NMR as an orthogonal assay
([Chem. Sci.](https://pubs.rsc.org/sc/article/16/8/3430/870344/Functionally-active-modulators-targeting-the-LRRK2)).
That is what the best result in a blind challenge looks like, and it is still micromolar.

**Target difficulty dominates method choice.** The same 23 teams, mostly the same methods, went from
3.7% to 0.7% by changing target. So a per-disease playground must not imply a uniform expectation
across its targets — and `TARGETS.md`'s tiering is the right shape for exactly this reason.

## 2. The compounds that fail, fail on properties we claim to predict

Of CACHE #1's initial hits, **37 of 73 showed poor solubility** despite computational prediction
([Hope, Hype](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844436/)). Solubility is one of our 21
endpoints — `Solubility_AqSolDB`, routed to gradient boosting, verdict *undecided* against the
ensemble.

This is the most directly load-bearing finding for Phase 4's output being consumed in Phase 5: the
documented failure mode of a real screening campaign is **precisely the property our ADMET model
predicts, on a benchmark where we cannot distinguish our model from a fingerprint baseline**. It is
the strongest available argument for the annotate-not-filter decision (P4.0 D1): an ADMET predictor
that cannot be shown to beat a baseline must not be the thing removing candidates, and the one time
the literature checks, the predictions were wrong on half the hits.

## 3. Our rescore stage's ranking is uncorrelated where it is used

FoldRx's `rescore` stage offers `boltz2-affinity`, and `_rank_stage` feeds its probability into the
composite at weight 0.20. Two 2026 assessments:

- **[On the Reliability of AI Methods in Drug Discovery](https://arxiv.org/html/2603.05532v1)** —
  Boltz-2 against physics-based ESMACS on 16,780 compounds (3CLPro) and 21,702 (TNKS2). Global
  agreement is weak to moderate (**Pearson r = 0.24–0.45**); **among the top 100 compounds there is
  no correlation at all** (≈ 0). Run-to-run reproducibility is high (r = 0.913–0.962), so it is
  *consistent* rather than accurate. The authors conclude it "cannot serve as a reliable substitute
  for high precision, physics-based methods".
- **[The Last Mile Problem](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00942)** (JCIM) — alchemical
  ABFE, MM/PBSA, MM/GBSA against GNINA, K<sub>DEEP</sub>, OnionNet-2, TopologyNet, Yuel, RF-score-VS
  and **Boltz-2** on 632 PDBbind complexes and DUD-E. End-state methods "scored well in recognizing
  actives from decoys in the DUD-E data set but showed little correlation with experimental values in
  PDBbind".

**"No correlation among the top 100" is the number that matters**, because the shortlist is exactly
where a screening pipeline operates. A tool that separates binders from decoys but cannot order the
survivors is an enrichment filter, not a ranker — and the registry already says something close to
this in `boltz2-affinity`'s caveat ("a ranking signal only: the values compress into a narrow band").
That caveat is now backed by a citation rather than by intuition.

**Separately: DUD-E enrichment is not evidence of affinity prediction.** Scoring well on
actives-vs-decoys and showing little correlation with measured affinity is the same split H6 found
between ranking and resolution, in a different domain.

## 4. One claim deliberately *not* carried

A search summary attributed to the 2026 Boltz-2 literature the claim that **"Boltz-2 retained high
enrichment even when binding sites were modified through mutations, demonstrating profound and
systematic memorization artifacts."** If true it would be decisive — a model whose enrichment
survives mutating the pocket is not reading the structure.

**We could not verify it.** The paper it appeared to point at (arXiv 2603.05532) contains no
mutation experiment; its authors state so. This programme has been burned by exactly this once
already — the "only 3 of 10 top TDC architectures reproduce" claim was carried for a day on a
research brief and then had to be marked uncorroborated.

So it is recorded here as **externally sourced and unverified, and is not used to justify anything.**

**Instead we measure it ourselves.** The claim suggests a control that is cheap, decisive and that
this programme's own methodology already demands (T3/T4: a detector certified only against the null
cannot tell "no effect" from "blind"). **Run the same compound set against a decoy target** — the
same protein with its binding pocket scrambled, or an unrelated protein — and compare the rankings.
If the ranking barely moves, the pipeline is scoring the ligands and not the target, and the user is
reading a compound-quality prior dressed as a prediction about their protein. That is Phase 5's
negative control, and it is ours rather than a citation.

---

## What this fixes in the programme's own wording

`AGENTS.md`, `TARGETS.md` and the shipped model card all quote CACHE #1's 3.7% as "what a good result
looks like". That stays true, and is now qualified in three ways that change how a reader should take
it:

1. **most teams get zero** — 7 of 23, and 5 of 23 on the harder target;
2. **3.7% is at 150 µM**, not at a drug-like potency; the best confirmed work was 3–41 µM;
3. **0.7% on a different target with the same teams** — difficulty is the target's, not the method's.

## Sources

- [CACHE Challenge #1: targeting the WDR domain of LRRK2](https://www.biorxiv.org/content/10.1101/2024.07.18.603797.full.pdf)
- [Functionally active modulators targeting the LRRK2 WD40 repeat domain identified by FRASE-bot in CACHE Challenge #1](https://pubs.rsc.org/sc/article/16/8/3430/870344/Functionally-active-modulators-targeting-the-LRRK2)
- [Virtual Screening: Hope, Hype, and the Fine Line In Between](https://pmc.ncbi.nlm.nih.gov/articles/PMC11844436/)
- [On the Reliability of AI Methods in Drug Discovery: Evaluation of Boltz-2](https://arxiv.org/html/2603.05532v1)
- [The Last Mile Problem: A Critical Assessment of Physics-Based and AI Tools](https://pubs.acs.org/doi/10.1021/acs.jcim.6c00942)
