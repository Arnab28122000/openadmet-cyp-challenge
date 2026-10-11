# Targets — what this programme may honestly say anything about

The canonical list the playground's disease templates are built from. Derived from
`research/02-targets-and-data.md` §A, where every UniProt accession and PDB ID was verified live
against `rest.uniprot.org` and `data.rcsb.org` on 2026-10-04.

**The tractability column decides everything.** Our models predict small-molecule properties and
small-molecule binding. For a target whose only drugs are antibodies, they have nothing to say, and
a playground that scores molecules against it anyway is generating noise with a confidence interval
on it. Those targets are listed here precisely so the UI can *name them and refuse*, rather than
quietly omitting them and looking like the disease is fully covered.

---

## Tier 1 — train and benchmark here

≥5,000 pChEMBL points each, multiple approved drugs with co-crystal structures.

| Target | ChEMBL | Disease |
|---|---|---|
| JAK1 | `CHEMBL2835` | Ankylosing spondylitis |
| RORγt | `CHEMBL1741186` | Ankylosing spondylitis |
| TYK2 | `CHEMBL3553` | Ankylosing spondylitis |
| PPARγ | `CHEMBL235` | Type 2 diabetes |
| DPP-4 | `CHEMBL284` | Type 2 diabetes |
| ADRB2 | `CHEMBL210` | Asthma |
| HIV-1 reverse transcriptase | `CHEMBL247` | HIV |
| HIV-1 integrase | `CHEMBL3471` | HIV |
| HIV-1 protease | `CHEMBL243` | HIV |

**RORγt is the best honest hypothesis-generation story in this programme**: 16,623 pChEMBL points,
162 structures, and **no approved drug**. Data-rich enough to train on, and genuinely unsolved.

## Tier 2 — usable, smaller (1,300–4,500 points)

PDE4B/PDE4D, DP2, SGLT2, 5-LOX, FLAP, GCK, FFAR1, and **hERG** (`CHEMBL240`, `Q12809`, 13,766
points). hERG is not a therapeutic target — it is the highest-value *safety* anti-target available,
and it is the bridge between the target half of this programme and the ADMET half.

## Tier 3 — low-data; few-shot and transfer tests only, with wide error bars

ERAP1 (293 points), HIV-1 capsid (184), IL-17A PPI (57), IL-23R (595), CysLT1 (276).

## Out of scope — biologic targets, 0–50 small-molecule points

IL-4Rα, IL-5, IL-5Rα, IL-17RA, the IL-23 complex, TSLP, IL-33/ST2, IgE, insulin receptor, CD4.

The same ChEMBL query that returns **9,257** activities for HIV-1 protease returns **0** for IL-4Rα,
IL-5Rα, IL-17RA and the IL-23 heterodimer. These are antibody targets. The playground names them and
says so.

---

## Corrections that cost someone a wrong answer if missed

- **IL-23 is a *refuted* target in axial disease**, not merely an intractable one: ustekinumab failed
  phase III (PMID 30225992) and risankizumab failed phase II (PMID 29945918). A model proposing
  IL-23 binders "for ankylosing spondylitis" would be contradicting the clinical record, not
  generating a hypothesis. Ankylosing spondylitis in this programme means **TYK2, JAK1 and RORγt**.
- **GLP-1R became small-molecule tractable in April 2026** — orforglipron (`CHEMBL4446782`), the
  first oral non-peptide GLP-1R agonist, was approved. Dock against **7LCI / 6ORV**, *not* the
  peptide-bound 4ZGM.
- **PDB 1HSG is HIV-1 protease** despite a deposited title reading "HIV II protease" (SIFTS maps it
  to P03367, 100% identity, both chains).
- **2Q7M (FLAP, 4.25 Å) and 5U1C (HIV-1 intasome, 3.9 Å) are not docking-grade** despite appearing
  in common target lists. A newer FLAP structure is needed before ALOX5AP can be docked at all.
- **Boltz-2's documented weak classes are GPCRs, transporters and ion channels** — which covers
  ADRB2, DP2, CysLT1, SGLT2 and hERG, five of the shortlist. Its binary confidence is also
  uncalibrated (everything scores >0.80). Where Boltz-2 is used on those five, the caveat travels
  with the number.
- **Type 1 diabetes has no approved small molecule.** Any T1D content is mechanism hypothesis only,
  and is labelled as such or left out.

## The number every claim here is measured against

Blind prospective ML virtual screening (CACHE #1, LRRK2 WDR domain): 23 teams, 1,955 predicted
compounds → **73 binders at K_D < 150 µM, a 3.7% hit rate**. That has been this programme's yardstick
since it opened, and it is the **optimistic end** of the range (`research/06-prospective-screening-reality.md`,
2026-10-05). Three things belong beside it:

- **Most teams found nothing.** 7 of 23 on CACHE #1, and **5 of 23** on CACHE #2 (Nsp13 helicase),
  where the overall rate was **0.7%**. The honest headline is not 3.7%, it is *most blind screening
  teams get zero*.
- **3.7% is measured at K_D < 150 µM**, which is a starting point for chemistry rather than a
  candidate. The best confirmed work on CACHE #1 was **8 binders at K_D 3–41 µM from 84 compounds
  across two rounds**, with SPR and ¹⁹F-NMR as an orthogonal assay.
- **Difficulty belongs to the target, not the method.** The same 23 teams and largely the same methods
  went from 3.7% to 0.7% by changing target — which is why the tiering above has to reach the UI
  rather than stay in this file.

**And the compounds that failed, failed on a property we claim to predict:** 37 of CACHE #1's 73 hits
had poor solubility despite computational prediction. `Solubility_AqSolDB` is one of our 21 endpoints,
routed to gradient boosting, verdict *undecided* against the ensemble.

The pipeline's own sanity check is that it re-finds **sitagliptin at DPP-4**.
