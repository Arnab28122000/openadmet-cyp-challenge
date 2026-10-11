# 02 — Targets and training data for in-silico triage

**Scope.** Research notes for a small-model *in-silico triage* capability: rank candidate
molecules by **predicted binding affinity** and **predicted ADMET**, to generate hypotheses
for later wet-lab validation. Nothing here is a therapeutic claim, a dosing recommendation
or a clinical assertion. Approved and clinical ligands are listed only as **benchmark
anchors** — molecules whose answer we already know, so we can tell whether a model is
learning chemistry or learning the dataset.

**Compute envelope assumed throughout:** one Google Colab **L4, 24 GB VRAM**, ~**24 h per
run**, ~**100–200 GB** of Google Drive for cached data. Every recommendation below is
filtered through that.

Compiled 2026-10-04. All counts were pulled live from the source APIs on that date unless
marked otherwise; queries are given so they can be re-run.

---

## A. Targets

### A.0 How to read the "tractable?" column — and why it decides everything

Our models are a **ligand-based affinity/ADMET ranker** plus, optionally, **structure-based
docking/rescoring**. Both assume the thing we are ranking is a **drug-like small molecule
binding a defined pocket**. That assumption holds for enzymes, GPCRs, nuclear receptors and
transporters. It does **not** hold for a soluble cytokine neutralised by an antibody: there
is no pocket, the interface is a large flat protein–protein surface, and the "ligand" is a
150 kDa IgG that no SMILES-based model can represent.

This is not a theoretical worry — it shows up directly in the data we would train on. The
same ChEMBL query that returns **9,257 activities for HIV-1 protease** returns **0** for
IL-4Rα, **0** for IL-5Rα, **0** for IL-17RA and **0** for the IL-23 heterodimer. Counts
pulled 2026-10-04 from the ChEMBL 37 web services:

| Target | ChEMBL target | Total activities | With pChEMBL (IC50/Ki/Kd/EC50) |
|---|---|---|---|
| HIV-1 reverse transcriptase | `CHEMBL247` | 19,411 | 10,785 |
| HIV-1 integrase | `CHEMBL3471` | 10,179 | 5,777 |
| HIV-1 protease | `CHEMBL243` | 9,257 | 6,399 |
| JAK1 | `CHEMBL2835` | 47,757 | 15,005 |
| RORγt | `CHEMBL1741186` | 47,808 | 16,623 |
| TYK2 | `CHEMBL3553` | 44,880 | 10,610 |
| PPARγ | `CHEMBL235` | 16,079 | 6,758 |
| β2-adrenergic receptor | `CHEMBL210` | 13,208 | 3,180 |
| DPP-4 | `CHEMBL284` | 8,397 | 5,391 |
| FLAP (ALOX5AP) | `CHEMBL4550` | 7,390 | 3,994 |
| 5-LOX | `CHEMBL215` | 7,092 | 3,563 |
| PDE4B | `CHEMBL275` | 7,617 | 2,974 |
| DP2 / CRTH2 | `CHEMBL5071` | 6,926 | 4,431 |
| SGLT2 | `CHEMBL3884` | 4,114 | 2,326 |
| Glucokinase | `CHEMBL3820` | 3,222 | 1,389 |
| CysLT1 | `CHEMBL1798` | 2,143 | 276 |
| IL-23 receptor | `CHEMBL4296013` | 1,119 | 595 |
| ERAP1 | `CHEMBL5939` | 749 | 293 |
| HIV-1 capsid | `CHEMBL2366504` | 390 | 184 |
| IL-17A | `CHEMBL3390822` | 305 | 57 |
| TSLP | `CHEMBL3712931` | 114 | 50 |
| **IL-5** | `CHEMBL1169600` | **4** | **3** |
| **IL-5Rα** | `CHEMBL3580483` | **0** | **0** |
| **IL-17RA** | `CHEMBL3580485` | **0** | **0** |
| **IL-4Rα** | `CHEMBL3580490` | **0** | **0** |
| **IL-23 (p19/p40 complex)** | `CHEMBL2364154` | **0** | **0** |

Reproduce any row:

```bash
curl -s -H 'Accept: application/json' \
  'https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL243&standard_type__in=IC50,Ki,Kd,EC50&pchembl_value__isnull=false&limit=1' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["page_meta"]["total_count"])'
```

**Practical rule for this project:** train and benchmark on the rows with ≥~1,000 pChEMBL
points. Treat the biologic targets as *out of scope for the affinity model* and say so in
writing — do not quietly produce a number for IL-5 and present it alongside a number for
DPP-4.

A caveat on the big numbers: **GLP-1R shows 113,886 total activities but only 2,312 with a
pChEMBL value**, because most of those rows are peptide-analogue functional assays. Always
filter on `pchembl_value__isnull=false` before believing a target is data-rich.

---

### A.1 HIV

HIV is the best case in this whole brief: four well-characterised small-molecule pockets,
decades of SAR in ChEMBL, and crystal structures of nearly every approved drug bound to its
target. If our pipeline cannot produce credible numbers on HIV protease, it is broken.

A naming trap first. HIV-1 protease, reverse transcriptase and integrase are **not separate
UniProt entries** — they are domains of the **Gag-Pol polyprotein**. The reference entry is
`P04585` (`POL_HV1H2`, HXB2 isolate, 1,435 aa, 282 PDB entries). ChEMBL, by contrast, *does*
split them into three separate targets mapped to strain-specific accessions. **Use the
ChEMBL target IDs for activity data and `P04585` for structure**; do not expect the
accessions to match.

| Target | UniProt | ChEMBL target | Tractable? |
|---|---|---|---|
| Protease (PR) | `P04585` (Gag-Pol, HXB2) | `CHEMBL243` (acc `Q72874`) | **Small molecule** — textbook case |
| Reverse transcriptase (RT) | `P04585` | `CHEMBL247` (acc `Q72547`) | **Small molecule** (NNRTI allosteric pocket + NRTI active site) |
| Integrase (IN) | `P04585` | `CHEMBL3471` (acc `Q7ZJM1`) | **Small molecule** (INSTI, metal-chelating) |
| Capsid (CA, p24) | `P04591` (`GAG_HV1H2`, 500 aa, 29 PDB entries) | `CHEMBL2366504` (acc `Q76129`) | **Small molecule** — newest validated pocket |
| Env gp120/gp41 | `P04578` (`ENV_HV1H2`, 856 aa, 132 PDB entries) | — | Mostly **biologic** (bNAbs); fostemsavir is a small-molecule exception |
| CCR5 (host) | `P51681` | — | **Small molecule** (maraviroc) |
| CD4 (host) | `P01730` | — | **Biologic** (ibalizumab) |

#### Verified drug-bound structures

Every PDB ID below was confirmed on 2026-10-04 by intersecting the UniProt accession with
the drug's PDB chemical-component ID in the RCSB search API. The component ID is given
because it is what you filter on when extracting the ligand pose.

**Protease** (chem comps: darunavir `017`, ritonavir `RIT`, lopinavir `AB1`, saquinavir `ROC`, amprenavir `478`)

| Ligand | ChEMBL | Representative PDB | Notes |
|---|---|---|---|
| Darunavir | `CHEMBL1323` (phase 4) | **1T3R** (1.2 Å, WT), **4HLA** / **4LL3** (1.95 Å, WT) | 79 PDB entries contain `017` |
| Ritonavir | `CHEMBL163` (phase 4) | **1HXW** (the classic), 1N49, 2B60 | 19 entries contain `RIT` |
| Lopinavir | `CHEMBL729` (phase 4) | **6DJ1**, 6DJ2 | |
| Saquinavir | `CHEMBL114` (phase 4) | **3OXC** (1.16 Å, WT), 1HXB, 3N3I | |
| Atazanavir | `CHEMBL1163` (phase 4) | — | |

**Reverse transcriptase** (efavirenz `EFZ`, rilpivirine `T27`, nevirapine `NVP`, doravirine `2KW`, etravirine `65B`)

| Ligand | ChEMBL | Representative PDB |
|---|---|---|
| Efavirenz | `CHEMBL223228` (4) | **1FK9**, 1JKH, 6BSG, 6BSH (8 entries with `EFZ`) |
| Nevirapine | `CHEMBL22230`* | **1VRT**, 3QIP, 3LP1 (16 entries with `NVP`) |
| Rilpivirine | `CHEMBL175691` (4) | **3MEE**, 3MEG |
| Doravirine | `CHEMBL2364608` (4) | **4NCG** |
| Etravirine | `CHEMBL308954` (4) | (comp `65B`) |
| Zidovudine (NRTI) | `CHEMBL129` (4) | acts on the polymerase site as the triphosphate |
| Tenofovir / lamivudine / emtricitabine / abacavir | `CHEMBL1538` / `CHEMBL141` / `CHEMBL885` / `CHEMBL1380` (all 4) | NRTIs — **prodrug nucleotides**, so a poor affinity benchmark; exclude from the regression set |

\* nevirapine's ChEMBL ID was not re-verified in this pass; look it up with
`?pref_name__iexact=nevirapine` before quoting it.

**Integrase** — the important structural caveat: HIV-1 integrase crystallises badly, so the
canonical INSTI co-structures are **prototype foamy virus (PFV) intasome** surrogates, not
HIV-1:

| Ligand | ChEMBL | PDB | What it actually is |
|---|---|---|---|
| Raltegravir | `CHEMBL254316` (4) | **3OYA** (2.65 Å) | **PFV intasome**, not HIV-1 |
| Dolutegravir | `CHEMBL1229211` (4) | **3S3M** (2.49 Å), 3S3N/3S3O, 6RWN | **PFV intasome**; 20 entries contain `DLU` |
| Bictegravir | `CHEMBL3989866` (4) | **6PUW** (2.9 Å cryo-EM) | **HIV-1 cleaved synaptic complex** — use this one for HIV-1 |
| Cabotegravir | `CHEMBL2403238` (4) | — | |

If you dock against a PFV intasome and report it as an HIV-1 result, that is a silent
species swap. Use **6PUW** for HIV-1 integrase and label 3OYA/3S3M as PFV.

**Capsid**

| Ligand | ChEMBL | PDB | Notes |
|---|---|---|---|
| Lenacapavir (GS-6207) | `CHEMBL4594438` (4) | **6V2F** (2.0 Å, CA hexamer + GS-6207), 6VKV, 7RHN, **9D6C** (2.1 Å cryo-EM immature lattice, with bevirimat) | 12 entries contain `QNG` |

Lenacapavir is worth flagging as the single most interesting HIV benchmark: a
**first-in-class small molecule against a pocket that was considered undruggable**, now
approved for both treatment and prevention (see e.g. the 2026 Cochrane review,
[PMID 41919720](https://europepmc.org/article/MED/41919720), DOI
[10.1002/14651858.cd016347](https://doi.org/10.1002/14651858.cd016347)). With only 184
pChEMBL points for `CHEMBL2366504`, it is a **low-data** target — a good test of whether a
pretrained model transfers, and a bad target to claim good numbers on without wide error
bars.

---

### A.2 Type 2 diabetes

The most mixed modality of the four, and the one where the 2026 landscape genuinely changed.

| Target | UniProt | ChEMBL | PDB entries | Tractable? |
|---|---|---|---|---|
| **DPP-4** | `P27487` (766 aa) | `CHEMBL284` | 120 | **Small molecule** — gold standard |
| **SGLT2** | `P31639` (672 aa) | `CHEMBL3884` | 12 | **Small molecule**; all co-structures are cryo-EM of the SGLT2–MAP17 complex |
| **PPARγ** | `P37231` (505 aa) | `CHEMBL235` | **380** | **Small molecule** — the richest structural set here |
| **GLP-1R** | `P43220` (463 aa) | `CHEMBL1784` | 80 | **Both** — see below |
| Glucokinase (GCK) | `P35557` (465 aa) | `CHEMBL3820` | 35 | **Small molecule** (allosteric activators) |
| GPR40 / FFAR1 | `O14842` (300 aa) | `CHEMBL4422` | — | **Small molecule** |
| Insulin receptor | `P06213` | — | — | **Biologic** (insulin analogues) |

#### GLP-1R is no longer a biologic-only target

This is the single most consequential A-section update. **Orforglipron (Foundayo™,
`CHEMBL4446782`) — an orally administered non-peptide GLP-1R agonist — received its first
FDA approval in April 2026** for long-term weight management, per *Drugs* "Orforglipron:
First Approval" ([PMID 42479349](https://europepmc.org/article/MED/42479349), DOI
[10.1007/s40265-026-02363-5](https://doi.org/10.1007/s40265-026-02363-5)); EU/Japan/Canada
filings including T2D are in review. So GLP-1R should be modelled as
**small-molecule-tractable**, with the peptides as a separate, excluded class.

The two modalities need separate structures:

- **Small-molecule / non-peptide agonists:** **7LCI** (2.9 Å cryo-EM, PF-06882961
  [danuglipron] bound to GLP-1R:Gs), **6ORV** (3.0 Å, TT-OAD2). These are the right
  templates for docking.
- **Peptide agonists:** **4ZGM** (1.8 Å, semaglutide backbone + GLP-1R ECD), 5OTU/5OTT/5OTX.
  Semaglutide `CHEMBL2108724` and liraglutide `CHEMBL4084119` are typed `Protein` in ChEMBL;
  tirzepatide `CHEMBL4297839` is a dual GIP/GLP-1 peptide. **Exclude all three from the
  affinity regression set** — a SMILES model cannot represent them meaningfully.

#### Verified small-molecule complexes

| Target | Ligand | ChEMBL | Comp | PDB |
|---|---|---|---|---|
| DPP-4 | Sitagliptin | `CHEMBL1422` (4) | `715` | **1X70** (2.1 Å), 4FFW, 7Y4G |
| DPP-4 | Linagliptin | `CHEMBL237500` (4) | `356` | **2RGU** (2.6 Å), 6Y0F, 8XTD |
| DPP-4 | Alogliptin | `CHEMBL376359` (4) | `T22` | **3G0B** (2.25 Å, "TAK-322") |
| DPP-4 | Vildagliptin | `CHEMBL142703` (4) | `LF7` | **6B1E**, 3W2T |
| DPP-4 | Saxagliptin | `CHEMBL2103745` | — | |
| SGLT2 | Empagliflozin | `CHEMBL2107830` (4) | `7R3` | **7VSI** (2.95 Å cryo-EM, SGLT2–MAP17), 7Y0B |
| SGLT2 | Dapagliflozin | `CHEMBL429910` (4) | `LE6` | **8HEZ** (2.8 Å) |
| SGLT2 | Canagliflozin | `CHEMBL4594217` (4) | `L3R` | **8HDH** (3.1 Å) |
| SGLT2 | Ertugliflozin | `CHEMBL1770248` (4) | — | |
| PPARγ | Rosiglitazone | `CHEMBL121` (4) | `BRL` | **2PRG** (2.3 Å, the classic), **4EMA** (2.55 Å), **7AWC** (1.74 Å), 5YCP (2.0 Å) |
| PPARγ | Pioglitazone (M-IV metabolite) | `CHEMBL595` (4) | `GFV` | **6DHA** (1.88 Å) |
| PPARγ | Lobeglitazone | `CHEMBL3585580` (3) | — | |
| GCK | Dorzagliatin | `CHEMBL4297508` (3) | — | allosteric activator; approved in China |

Avoid **1ZGY** for PPARγ: the RCSB search returns it for the `BRL` component, but its
deposited title is an LRH-1/SHP structure. Use 2PRG / 4EMA / 7AWC instead. (This is exactly
the kind of error that only surfaces if you check titles rather than trusting a search hit —
worth building into the extraction script.)

**Why T2D is a good second target family:** DPP-4, SGLT2 and PPARγ each have thousands of
pChEMBL points *and* multiple approved drugs with co-crystals, which makes them usable for
both the regression benchmark and a docking-pose sanity check. PPARγ's 380 structures also
make it one of the few targets where a structure-based model has enough holo conformations to
be tested for induced-fit robustness.

---

### A.3 Ankylosing spondylitis (axial spondyloarthritis)

**This is the hard case, and the honest answer is that AS is a biologics disease.** Five of
the validated targets are cytokines or cytokine receptors hit by antibodies. Our affinity
model applies to exactly one validated mechanism — **JAK/TYK2** — plus two interesting
unapproved ones.

| Target | UniProt | ChEMBL | PDB | Tractable? |
|---|---|---|---|---|
| **TNF-α** | `P01375` (233 aa) | `CHEMBL1825` | 52 | **Biologic** in practice (see note) |
| **IL-17A** | `Q16552` (155 aa) | `CHEMBL3390822` | 44 | **Biologic**, but PPI small molecules now exist |
| IL-17F | `Q96PD4` | — | — | **Biologic** (bimekizumab is IL-17A/F) |
| IL-17RA | `Q96F46` (866 aa) | `CHEMBL3580485` (0 acts) | 10 | **Biologic** (brodalumab) |
| IL-23 p19 / p40 | `Q9NPF7` / `P29460` | `CHEMBL2364154` (0 acts) | 15 / 20 | **Biologic** — *and it failed in AS* |
| IL-23R | `Q5VWK5` (629 aa) | `CHEMBL4296013` (595 pts) | — | Emerging small molecule / peptide |
| **JAK1** | `P23458` (1,154 aa) | `CHEMBL2835` (15,005 pts) | 52 | **Small molecule** ✅ |
| **TYK2** | `P29597` (1,187 aa) | `CHEMBL3553` (10,610 pts) | 52 | **Small molecule** ✅ |
| JAK2 / JAK3 | `O60674` / `P52333` | — | — | **Small molecule** (selectivity anti-targets) |
| **ERAP1** | `Q9NZ08` (941 aa) | `CHEMBL5939` (293 pts) | 27 | **Small molecule**, genetically validated, **no approved drug** |
| **RORγt** | `P51449` (518 aa) | `CHEMBL1741186` (16,623 pts) | **162** | **Small molecule**, data-rich, **no approved drug** |
| HLA-B27 | `P01889` | — | — | Not directly druggable (genetic risk factor) |

#### The negative results matter more than the positive ones

Two large, well-powered failures define this indication, and any honest write-up must name
them:

- **Ustekinumab (anti-IL-12/23 p40) failed phase III in axial spondyloarthritis** — three
  RCTs, reported in *Arthritis & Rheumatology* 2019
  ([PMID 30225992](https://europepmc.org/article/MED/30225992), DOI
  [10.1002/art.40728](https://doi.org/10.1002/art.40728)), with an accompanying editorial
  explicitly titled *"Ustekinumab Fails to Show Efficacy in a Phase III Axial
  Spondyloarthritis Program: The Importance of Negative Results"*
  ([PMID 30600936](https://europepmc.org/article/MED/30600936)).
- **Risankizumab (anti-IL-23 p19) failed a phase II proof-of-concept in AS** — Baeten et al.,
  *Annals of the Rheumatic Diseases* 2018 ([PMID 29945918](https://europepmc.org/article/MED/29945918),
  DOI [10.1136/annrheumdis-2018-213328](https://doi.org/10.1136/annrheumdis-2018-213328)).

So **IL-23 is a validated target in psoriasis and a *refuted* one in axial disease**, even
though IL-17 blockade works in both. If we were ever to rank "IL-23 pathway" molecules and
present them as AS hypotheses, that would be a claim the clinical record contradicts. Keep
IL-23 out of the AS target list; keep it in the psoriasis one if we ever add that.

#### Verified complexes

| Target | Ligand | ChEMBL | Comp | PDB |
|---|---|---|---|---|
| JAK3 | Tofacitinib (CP-690,550) | `CHEMBL221959` (4) | `MI1` | **3LXK** (2.0 Å) |
| JAK1 | Tofacitinib | — | `MI1` | **3EYG** (1.9 Å) |
| TYK2 | Tofacitinib | — | `MI1` | **3LXN** (2.5 Å) |
| JAK1 | Upadacitinib | `CHEMBL3622821` (4) | *no PDB comp found* | approved for AS; no co-structure resolved in this pass |
| TYK2 | Deucravacitinib | `CHEMBL4435170` (4) | *no PDB comp found* | allosteric JH2 binder — pseudokinase domain, a different pocket |
| TNF-α | SPD304 (tool cpd) | — | — | **2AZ5** (2.1 Å) — *"Crystal Structure of TNF-alpha with a small molecule inhibitor"* |
| IL-17A | macrocyclic antagonists | — | — | **5HI4**, **5HI5** (both 1.8 Å, *"Binding site elucidation and structure guided design of macrocyclic IL-17A antagonists"*) |
| IL-17A | oral PPI inhibitors | — | — | **8DYG** (1.49 Å, "Compound 7"), **9SQX** (1.41 Å, "compound 19") |
| ERAP1 | bestatin analogue | — | `I88` | **7Z28** (1.55 Å) |
| ERAP1 | benzothiazinone inhibitor | — | — | **9GK6** (1.33 Å) |
| RORγt | 25-OH-cholesterol + allosteric FM26 | — | — | **6T4X** (1.48 Å) |

**2AZ5 and 5HI4/5HI5/8DYG/9SQX are genuinely useful.** They show that even the "biologic"
cytokine targets have real small-molecule PPI-inhibitor structures, including very recent
high-resolution ones (9SQX at 1.41 Å). That is a legitimate, hypothesis-generating use of a
docking pipeline — *but* with 57 pChEMBL points for IL-17A, it is a few-shot problem, not a
training problem. Frame it as "can we retrieve the known IL-17A PPI chemotypes from a
library", not "we predict IL-17A affinity".

Approved AS biologics, for completeness, all phase 4 in ChEMBL and all **outside our
model's domain**: secukinumab `CHEMBL1743068`, ixekizumab `CHEMBL1743034`, bimekizumab
`CHEMBL4297700`, brodalumab `CHEMBL1742996`, infliximab `CHEMBL1201581`, adalimumab
`CHEMBL1201580`, etanercept `CHEMBL1201572`, golimumab `CHEMBL1201833`, certolizumab pegol
`CHEMBL1201831`. Approved small molecules: tofacitinib `CHEMBL221959`, upadacitinib
`CHEMBL3622821`.

**Recommendation for AS:** make **TYK2 and JAK1** the primary modelling targets (together
~25,600 pChEMBL points — enough for a real train/test split), **RORγt** the data-rich
exploratory target (16,623 points, 162 structures, no approved drug — the most honest
"hypothesis generation" story in this brief), and **ERAP1** the genetically-validated
low-data stretch goal. Report IL-17A/IL-23/TNF as explicitly out of scope.

---

### A.4 Asthma

Same shape as AS: the modern drugs are biologics, the small molecules are older and mostly
symptomatic. But asthma has **more data-rich small-molecule targets** than AS.

| Target | UniProt | ChEMBL | PDB | Tractable? |
|---|---|---|---|---|
| **β2-adrenergic receptor** | `P07550` (413 aa) | `CHEMBL210` (3,180 pts) | **149** | **Small molecule** ✅ — best-characterised GPCR in existence |
| **PDE4B / PDE4D** | `Q07343` / `Q08499` | `CHEMBL275` (2,974) / `CHEMBL288` (2,160) | 41 / — | **Small molecule** ✅ |
| **DP2 / CRTH2** | `Q9Y5Y4` (395 aa) | `CHEMBL5071` (4,431 pts) | 6 | **Small molecule** ✅ (but see failure note) |
| **CysLT1** | `Q9Y271` (337 aa) | `CHEMBL1798` (276 pts) | — | **Small molecule** ✅ (montelukast) |
| **5-LOX** | `P09917` (674 aa) | `CHEMBL215` (3,563 pts) | 9 | **Small molecule** ✅ (zileuton) |
| **FLAP (ALOX5AP)** | `P20292` (161 aa) | `CHEMBL4550` (3,994 pts) | 4 | **Small molecule** ✅ |
| Glucocorticoid receptor | `P04150` (777 aa) | — | 57 | **Small molecule** ✅ (ICS backbone of therapy) |
| Muscarinic M3 | `P20309` (590 aa) | — | — | **Small molecule** ✅ (tiotropium) |
| **IL-4Rα** | `P24394` (825 aa) | `CHEMBL3580490` (**0**) | 10 | **Biologic** (dupilumab) |
| IL-4 / IL-13 | `P05112` / `P35225` | — | — | **Biologic** |
| **IL-5** | `P05113` (134 aa) | `CHEMBL1169600` (**3**) | 5 | **Biologic** (mepolizumab, reslizumab) |
| **IL-5Rα** | `Q01344` (420 aa) | `CHEMBL3580483` (**0**) | — | **Biologic** (benralizumab) |
| **TSLP** | `Q969D9` (159 aa) | `CHEMBL3712931` (50 pts) | 5 | **Biologic** (tezepelumab) |
| TSLPR / CRLF2 | `Q9HC73` | — | — | **Biologic** |
| IL-33 / ST2 | `O95760` / `Q01638` | — | — / 1 | **Biologic** |
| IgE | `P01854` | — | — | **Biologic** (omalizumab) |

#### Verified complexes

| Target | Ligand | ChEMBL | Comp | PDB |
|---|---|---|---|---|
| ADRB2 | Salmeterol | `CHEMBL1263` (4) | `K5Y` | **6MXT** (2.96 Å, + Nb71) |
| ADRB2 | Salbutamol/albuterol | `CHEMBL714` (4) | `68H` | **7DHI** (3.26 Å cryo-EM, partial-agonist–Gs complex) |
| ADRB2 | Formoterol | `CHEMBL1256786` (4) | `H98` | **8JJ8**, **7BZ2** |
| ADRB2 | (reference apo/inverse-agonist) | — | — | **2RH1**, **1GQ4** |
| ADRB2 | Vilanterol / indacaterol | `CHEMBL1198857` / `CHEMBL1095777` (4) | — | |
| PDE4B | Roflumilast | `CHEMBL193240` (4) | `ROF` | **1XMU** (2.3 Å) |
| PDE4D | Roflumilast | — | `ROF` | **1XOQ** (1.83 Å), **3G4L** |
| PDE3/4 | Ensifentrine | `CHEMBL4594287` (4) | — | dual PDE3/4; approved (COPD) |
| DP2/CRTH2 | Fevipiprant | `CHEMBL3137332` (3) | `FSY` | **6D26** (2.80 Å) |
| DP2/CRTH2 | CAY10471 | — | — | **6D27** (2.74 Å) |
| CysLT1 | Zafirlukast | `CHEMBL603` (4) | `ZLK` | **6RZ5** (2.53 Å XFEL), 7YPZ, 9R07 |
| CysLT1 | Pranlukast | `CHEMBL21333` (4) | — | **6RZ4** (2.7 Å) |
| CysLT1 | Montelukast | `CHEMBL787` (4) | `MTK` | **no CysLT1 co-structure** — `MTK` appears in only **1** PDB entry (2NNI), not CysLT1 |
| 5-LOX | — | — | `FE2` | **7TTK** (1.98 Å, "Stable-5-LOX"), 3V98/3V99 |
| FLAP | DG-031 | — | — | **6VGC** (2.37 Å), 6VGI, 2Q7R |
| GR | Dexamethasone | — | `DEX` | **1M2Z** (2.5 Å), 4UDC, 7KRJ, 1P93 |
| IL-4Rα | nanobody HuNb103 | — | — | **8K4Q** (2.59 Å) — a *biologic* complex |
| TSLP | Tezepelumab Fab | `CHEMBL3707229` (4) | — | **5J13** (2.30 Å), 5J11, 5J12 — a *biologic* complex |

The **montelukast** finding is a good example of why these pairings must be checked rather
than assumed: montelukast is the most-prescribed oral asthma small molecule in the world, and
there is **no CysLT1–montelukast co-crystal in the PDB**. If a docking benchmark silently
substitutes zafirlukast's 6RZ5 pose for it, that is a fabricated result.

#### DP2/CRTH2 is the cautionary tale of this section

CRTH2 looks ideal on paper: 4,431 pChEMBL points, a 2.8 Å co-structure with the
clinical-stage ligand, clean pharmacology. And **fevipiprant failed both phase III
exacerbation trials** — LUSTER-1 and LUSTER-2, *Lancet Respiratory Medicine* 2021
([PMID 32979986](https://europepmc.org/article/MED/32979986), DOI
[10.1016/S2213-2600(20)30412-4](https://doi.org/10.1016/S2213-2600(20)30412-4)); see also
the SPIRIT long-term safety trial ([PMID 34895218](https://europepmc.org/article/MED/34895218))
and a 2021 meta-analysis ([PMID 34387775](https://europepmc.org/article/MED/34387775)).
Setipiprant (`CHEMBL2386081`, phase 3) also did not reach approval.

This is the cleanest available illustration of the gap our capability must never elide:
**fevipiprant is a potent, selective, orally bioavailable DP2 antagonist with a crystal
structure — and it does not work in patients.** Our models predict *affinity* and *ADMET*.
They do not predict clinical efficacy, and DP2 is the example to cite when saying so.

Approved asthma biologics (all phase 4, all out of our domain): dupilumab `CHEMBL2108675`,
mepolizumab `CHEMBL2108429`, reslizumab `CHEMBL2107884`, benralizumab `CHEMBL1742991`,
tezepelumab `CHEMBL3707229`, omalizumab `CHEMBL1201589`.

**Recommendation for asthma:** **ADRB2** (149 structures — the right target for a
docking-power check), **PDE4B/D** (two paralogues, 5,100 combined points — the right target
for a *selectivity* task), **DP2** and **FLAP/5-LOX**. Treat the Th2 cytokine axis as out of
scope.

---

### A.5 Target shortlist (what to actually build on)

Ranked by (pChEMBL data volume × structural support × small-molecule tractability):

| Tier | Targets | Why |
|---|---|---|
| **1 — train and benchmark here** | JAK1 `CHEMBL2835`, RORγt `CHEMBL1741186`, TYK2 `CHEMBL3553`, PPARγ `CHEMBL235`, ADRB2 `CHEMBL210`, HIV-1 RT `CHEMBL247`, HIV-1 IN `CHEMBL3471`, HIV-1 PR `CHEMBL243`, DPP-4 `CHEMBL284` | ≥5,000 pChEMBL points each, multiple approved drugs with co-crystals |
| **2 — usable, smaller** | PDE4B/PDE4D, DP2, SGLT2, 5-LOX, FLAP, GCK, FFAR1, hERG `CHEMBL240` (13,766 pts — safety anti-target) | 1,300–4,500 points |
| **3 — low-data, interesting, wide error bars** | ERAP1 (293), HIV-1 capsid (184), IL-17A PPI (57), IL-23R (595), CysLT1 (276) | few-shot / transfer-learning tests only |
| **Out of scope** | IL-4Rα, IL-5, IL-5Rα, IL-17RA, IL-23 complex, TSLP, IL-33/ST2, IgE, insulin receptor, CD4 | biologic targets; 0–50 small-molecule points |

Note the deliberate inclusion of **hERG (`CHEMBL240`, `Q12809`, 13,766 pChEMBL points)** —
it is not a therapeutic target but it is the highest-value *safety* endpoint available, it is
data-rich, and it connects section A to the ADMET half of section B.

---

## B. Training data

**Verification convention used below.** ✅ = confirmed on 2026-10-04 by direct API call or
HTTP HEAD. ⚠️ = taken from a page we read but not independently cross-checked — spot-check
before publishing a number. ❌ = could not verify; do not cite.

### B.0 The headline, before the detail

Three findings dominate this section and should change the plan:

1. **Fingerprints + gradient-boosted trees are a competitive baseline, not a straw man.**
   They hold or share #1 on roughly half of TDC's 22 ADMET tasks, and Deng et al. 2023
   (62,820 models trained, 30 seeds per split) found a Random Forest on fixed fingerprints
   **beat learned representations on most MoleculeNet datasets under scaffold split**. This
   trains in minutes on CPU. Build it first; it is the honest floor and may be the ceiling.
2. **Three of the most-used benchmarks are measurably broken** — DUD-E (ligand-only models
   hit 0.98 AUC), LIT-PCBA (a zero-parameter memorisation baseline matches published SOTA),
   and PDBbind's own general/refined/core split (cross-contaminated). Details in §B.7, B.6,
   B.3 and section D.
3. **Two licensing problems are real and one is potentially serious.** PDBbind data after
   v2020 is commercial and its derivatives cannot be redistributed; and **Enamine's terms of
   use appear to contain a clause prohibiting ML training on Enamine data**. See §B.11.

---

### B.1 ChEMBL — the backbone

✅ Live from `https://www.ebi.ac.uk/chembl/api/data/status.json` on 2026-10-04:

```json
{"activities": 24527044, "chembl_db_version": "ChEMBL_37",
 "chembl_release_date": "2026-05-01", "compound_records": 3824604,
 "disinct_compounds": 2921148, "publications": 101100,
 "status": "UP", "targets": 18552}
```

(the typo `disinct_compounds` is the API's own.)

| Item | ChEMBL 37 |
|---|---|
| Distinct compounds | **2,921,148** ✅ |
| Compound records (non-unique) | **3,824,604** ✅ |
| Activities | **24,527,044** ✅ |
| Activities with a pChEMBL value | **4,969,278** ✅ |
| Assays | **1,970,438** ✅ |
| Targets (all types) | **18,552** ✅ |
| Targets typed `SINGLE PROTEIN` | **11,055** ✅ |
| Approved drugs (`max_phase = 4`) | **4,225** ✅ |
| Molecules at `max_phase ≥ 1` | **16,784** ✅ |
| Publications | **101,100** ✅ |

**Cite it as "ChEMBL 37 (May 2026)".** The release date is genuinely ambiguous: the API says
`2026-05-01`, FTP file timestamps say `2026-05-29`, and the release-notes header still says
"January 5, 2026" (stale).

New in 37 that matters here: an `ACTIVITIES.MODALITY` field annotating targeted protein
degradation (~29,000 points) ⚠️; ~7,300 antimicrobial assays remapped to organism-specific
targets ⚠️; ~400 tautomer duplicates merged ⚠️; >3,000 BindingDB structures curated ⚠️; 169
compounds upgraded to `max_phase=4` on regulator evidence only ⚠️.

**License: CC BY-SA 3.0 Unported** ⚠️ ([FTP README](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/README),
[REQUIRED.ATTRIBUTION](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/REQUIRED.ATTRIBUTION)).
Cite Mendez et al., *NAR* 2019, 47(D1):D930–D940, DOI
[10.1093/nar/gky1075](https://doi.org/10.1093/nar/gky1075).

> **Share-alike matters.** If we redistribute a *derived database* (a cleaned training set),
> CC BY-SA 3.0 propagates to it. Whether trained **weights** are a derivative is an untested
> grey area. Shipping only weights is much lower risk than shipping a dataset.

#### Downloads — ✅ sizes verified by HTTP HEAD on 2026-10-04

Base: `https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/`

| File | Size ✅ |
|---|---|
| `chembl_37_sqlite.tar.gz` | **5.76 GB** (5.37 GiB) |
| `chembl_37_postgresql.tar.gz` | **2.09 GB** |
| `chembl_37_mysql.tar.gz` | **2.10 GB** |
| `chembl_37.sdf.gz` | **0.93 GB** |
| `chembl_37.h5` | **0.33 GB** (FPSim2 format) |
| `chembl_37_chemreps.txt.gz` | **0.29 GB** (SMILES + InChI + InChIKey) |
| `chembl_37.fps.gz` | **0.18 GB** (Morgan fingerprints) |
| `chembl_uniprot_mapping.txt` | ~1.2 MB |

```bash
B=https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest
wget -c $B/chembl_37_chemreps.txt.gz        # 0.29 GB — SMILES only, usually enough
wget -c $B/chembl_37_sqlite.tar.gz          # 5.76 GB -> ~20+ GB unpacked; watch Colab disk
wget    $B/checksums.txt && sha256sum -c --ignore-missing checksums.txt
tar xzf chembl_37_sqlite.tar.gz             # -> chembl_37/chembl_37_sqlite/chembl_37.db
```

**On Colab, take `chemreps.txt.gz` plus targeted REST pulls, not the 5.76 GB SQLite** — the
unpacked DB with indexes will exhaust the disk quota before you train anything.

#### Per-target activity pull

REST (what we used for every count in section A):

```
https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL243&standard_type__in=IC50,Ki,Kd,EC50&pchembl_value__isnull=false&limit=1000
```

Paginate on `page_meta.next`. Useful filters: `standard_relation==`, `assay_type=B`
(binding assays only), `pchembl_value__isnull=false`.
[Web-services docs](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services).

Python client:

```python
# pip install chembl_webresource_client
from chembl_webresource_client.new_client import new_client
import pandas as pd

act = new_client.activity.filter(
    target_chembl_id="CHEMBL2835",        # JAK1
    standard_type="IC50",
    standard_relation="=",
    pchembl_value__isnull=False,
).only(["molecule_chembl_id", "canonical_smiles", "standard_value",
        "standard_units", "pchembl_value", "assay_chembl_id",
        "document_chembl_id"])            # keep document id — you need it for a time split
df = pd.DataFrame(list(act))
```

Keep `document_chembl_id` and `assay_chembl_id`. You cannot build a **time split** (section
D) or detect **assay-batch leakage** without them, and they are easy to drop by accident.

SQL on the dump, per target:

```sql
SELECT md.chembl_id, cs.canonical_smiles, act.standard_value, act.pchembl_value,
       a.chembl_id AS assay_id, d.year
FROM activities act
JOIN assays a               ON a.assay_id   = act.assay_id
JOIN target_dictionary td   ON td.tid       = a.tid
JOIN molecule_dictionary md ON md.molregno  = act.molregno
JOIN compound_structures cs ON cs.molregno  = md.molregno
JOIN docs d                 ON d.doc_id     = act.doc_id
WHERE td.chembl_id = 'CHEMBL2835' AND act.standard_type = 'IC50'
  AND act.pchembl_value IS NOT NULL;
```

Approved-drug subset (4,225 molecules ✅):

```sql
SELECT md.chembl_id, md.pref_name, md.max_phase,
       cs.canonical_smiles, cs.standard_inchi_key
FROM molecule_dictionary md
JOIN compound_structures cs USING (molregno)
WHERE md.max_phase >= 4;   -- use >= : some installs type max_phase as numeric
```

Related tables worth joining: `drug_indication` (MeSH/EFO + `max_phase_for_ind`),
`drug_mechanism` (`mechanism_of_action`, `action_type`, join `target_dictionary` on `tid`),
`molecule_atc_classification`.

---

### B.2 BindingDB

⚠️ Stats dated 2026-09-26 on the [home page](https://bindingdb.org/rwd/bind/index.jsp) and
[download page](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp):

| Item | Count |
|---|---|
| Binding measurements | **3,248,380** |
| Compounds | **1,446,269** |
| Protein targets | **11,516** |
| BindingDB-curated subset | 1.6 M measurements / 778 K compounds / 4.8 K targets |
| Patent-derived | 8,587 patents, 1,317,148 measurements, 660,903 compounds, 3,110 targets (2026-02-28) |

**License** ⚠️ ([info.jsp](https://www.bindingdb.org/rwd/bind/info.jsp)): BindingDB-curated
data **CC BY 3.0**; **ChEMBL-imported data CC BY-SA 3.0 Unported**. Both permit commercial
use with attribution, but **the full TSV mixes the two, so treat the whole file as
share-alike.** Cite Liu et al., *NAR* 2025, 53:D1633–D1644, "BindingDB in 2024".

Downloads — ✅ URLs and sizes HEAD-verified; build **202610**, files updated 2026-09-29;
cadence roughly monthly. **Note the underscore before the extension** —
`BindingDB_All_202610.tsv.zip` is a 404:

| File | URL | Size ✅ |
|---|---|---|
| Full TSV | `.../downloads/BindingDB_All_202610_tsv.zip` | **566.8 MiB** |
| 2D SDF | `.../downloads/BindingDB_All_2D_202610_sdf.zip` | **1.49 GiB** |
| 3D SDF | `.../downloads/BindingDB_All_3D_202610_sdf.zip` | 3.07 GB ⚠️ |
| ChEMBL subset TSV | `.../downloads/BindingDB_ChEMBL_202610_tsv.zip` | **328.8 MiB** |
| UniProt mapping | `https://www.bindingdb.org/rwd/bind/BindingDB_UniProt.txt` | **573 KiB** |

```bash
wget -O BindingDB_All_202610_tsv.zip \
  https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202610_tsv.zip
unzip BindingDB_All_202610_tsv.zip      # -> BindingDB_All.tsv
wget https://www.bindingdb.org/rwd/bind/BindingDB_UniProt.txt
```

The TSV is ~190 ragged columns; load selectively. **Affinity cells carry `>` / `<`
qualifiers as strings** — censored data, which section D says must be handled explicitly
rather than coerced to a float:

```python
import pandas as pd
df = pd.read_csv("BindingDB_All.tsv", sep="\t", on_bad_lines="skip", low_memory=False,
    usecols=["Ligand SMILES", "UniProt (SwissProt) Primary ID of Target Chain 1",
             "IC50 (nM)", "Ki (nM)", "Kd (nM)", "EC50 (nM)"])   # ⚠️ verify column names
```

There is a REST API ([docs](https://www.bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp)),
e.g. `GET http://bindingdb.org/rest/getLigandsByUniprots?uniprot=P00176,P00183&cutoff=10000`
⚠️ untested — not for bulk pulls.

> **Dedup warning.** BindingDB *imports* ChEMBL. If you use both, either drop
> `BindingDB_ChEMBL_*` or dedup on (InChIKey, target UniProt, measurement type). Failing to
> do this inflates your training-set size and leaks across splits.

---

### B.3 PDBbind and PDBbind+ — a licensing minefield

| Version | General (PL) | Refined | Core / CASF |
|---|---|---|---|
| v2016 | ~13,308 ❌ | 4,057 ❌ | **285** (CASF-2016) ✅ |
| v2019 | 17,679 ❌ | 4,852 ❌ | 285 ❌ |
| **v2020** | **19,443** ⚠️ | **5,316** ⚠️ | 285 ⚠️ |
| v2021 | 22,920 ⚠️ | ❌ | ❌ |
| v2024 (PDBbind+) | 27,385 PL, 4,594 PP, 1,440 P-NA, 234 NA-L ⚠️ | ❌ | ❌ |

**The problem.** The old site `pdbbind.org.cn` offered free registration plus a click-through
licence open to academic *and* industrial users
([CCL 2015 announcement](https://server.ccl.net/chemistry/resources/messages/2015/12/17.001-dir/index.html));
that covers everything up to **v2020**. The new site
[`pdbbind-plus.org.cn`](http://www.pdbbind-plus.org.cn/) resolves but is a JavaScript SPA —
`curl` returns a 664-byte Vue shell with no readable terms, so **the licence text could not be
read without a browser**. Secondary sources describe a paid subscription with a free demo
([TargetMol listing](https://www.targetmol.com/services/pdb)). ❌ No price list found;
academic vs commercial tiers unverified.

**Redistribution is prohibited.** The HiQBind paper states plainly that *"the user's
agreement of PDBbind prohibits any distribution of any derivative dataset"* — which is why
its authors shipped a **workflow** instead of a cleaned PDBbind
([arXiv:2411.01223v2](https://arxiv.org/html/2411.01223v2)). The same paper says PDBbind
*"ceased free distribution for data curated after 2020"*.

Two cautions on sourcing: **❌ there is no dated primary announcement** that v2020 is the
cutoff — every statement found is secondary. And a widely-mirrored v2020 copy
([Zenodo 7014096](https://zenodo.org/records/7014096), 25.8 GB, labelled CC BY 4.0) is a
re-upload of Schrödinger-prepared structures whose authorisation is unclear. **A CC BY label
on a re-upload does not override the original click-through.**

#### Use these instead

| Dataset | What | Size | License |
|---|---|---|---|
| **HiQBind / HiQBind-WF** | Standalone high-quality set from BioLiP + Binding MOAD + BindingDB affinities, with an artifact-fixing workflow | **18,160 PDB entries / 32,275 complexes** ⚠️; `hiqbind.tar.gz` ✅ **8.07 GB** (8.09 GB with metadata) | code **MIT** ✅; **dataset CC BY 4.0** ✅ ⭐ |
| **PLINDER** | Large PDB-derived system set with splits and an eval framework | **449,383 protein-ligand systems**, 98,473 apo + 205,300 AF2 structures; PDB as of 2024-04-09 | ❗ code **GPL-2.0** ✅ — see caution below |
| **LP-PDBBind** | Leak-proof re-split of v2020: train/test protein-sequence similarity capped at 0.5, ligand similarity at 0.99 | 14,324 cleaned: 11,513 train / 2,422 val / 4,860 test; plus **BDB2020+**, a 115-record independent test set | arXiv CC BY 4.0; repo license ❌ |
| **Exscientia low-sim PDBbind** | CCDC-prepared v2020 (18,310 of 19,443 survived prep) | 18,310; `pdbbind_ccdc_structures.zip` ✅ **3.95 GB**, `pdbbind_ccdc_ligands.zip` 29 MB, `pdbbind_dataset.csv` 2 MB | ✅ **CC BY 4.0** |

- HiQBind: [GitHub](https://github.com/THGLab/HiQBind) (**MIT** ✅, 87 stars) ·
  *Digital Discovery* 2025, [10.1039/D4DD00357H](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00357h).
  **Two artifacts, two licences, and they are easy to confuse:**
  [Zenodo 10.5281/zenodo.14903380](https://doi.org/10.5281/zenodo.14903380) is only the
  **code release** (`HiQBind-v0.1.zip`, ✅ 4 MB, CC BY 4.0) — the **dataset** is on Figshare,
  [10.6084/m9.figshare.27430305](https://doi.org/10.6084/m9.figshare.27430305), as
  `hiqbind.tar.gz` ✅ **8.07 GB** plus `hiqbind_{sm,poly,}_metadata.csv`, licensed ✅
  **CC BY 4.0** (not MIT — MIT is the workflow code).
- PLINDER (VantAI + MIT CSAIL + NVIDIA): [GitHub](https://github.com/plinder-org/plinder) ·
  [PyPI](https://pypi.org/project/plinder/) ·
  [bioRxiv 10.1101/2024.07.17.603955](https://www.biorxiv.org/content/10.1101/2024.07.17.603955).
  ❗ **Correction worth noting: the PLINDER repository is licensed GPL-2.0** ✅ (confirmed via
  the GitHub API, 313 stars) — *not* Apache 2.0, as is sometimes reported. GPL-2.0 on the
  tooling is a real consideration for a commercial product, since the loader/eval code is what
  you would link against. The *data* is separately described as mixed by provenance
  (PLINDER-curated permissive, BindingDB-derived CC BY 4.0, ChEMBL-derived CC BY-SA 4.0) ⚠️ —
  **read the repo's `LICENSE` and the data terms directly before depending on either.**
- LP-PDBBind: [GitHub](https://github.com/THGLab/LP-PDBBind/) ·
  [arXiv:2308.09639](https://arxiv.org/abs/2308.09639)
- Exscientia: [Zenodo 13772124](https://zenodo.org/records/13772124) ·
  [GitHub](https://github.com/Exscientia/low-sim-pdbbind)

**Quality critiques to design around:** structural artifacts (covalent binders, missing
atoms, wrong bond orders/protonation, sub-2 Å steric clashes, aromaticity errors — HiQBind,
[PMC11967698](https://pmc.ncbi.nlm.nih.gov/articles/PMC11967698/)); cross-split leakage
between general/refined/core, where **models retrained on the leak-proof split score
*better*** (LP-PDBBind); and memorisation dominance (Volkov et al., *J Med Chem* 2022,
65(11):7946–7958 — explicit interaction descriptors give **no advantage** over ligand-only or
protein-only descriptors, and nearest-neighbour baselines already match the networks;
[lab page](https://lab.drugdesign.unistra.fr/publications/2022_volkov/)).

> **Decision: build on HiQBind (data CC BY 4.0, 8.07 GB) and use LP-PDBBind's splits.** Treat
> PLINDER as a reference and a source of split methodology rather than a dependency, until its
> GPL-2.0 tooling licence has been reviewed. Do not touch PDBbind+ data. If v2020 is used at
> all, no derivative of it can be redistributed.

---

### B.4 BindingNet

**What it is:** *comparative-modelling*-based protein-ligand complexes, not physics-only
docking — it takes ChEMBL ligands with measured affinity and transfers a pose from an
analogous crystal complex by constrained docking against a similar-ligand template. v1 used
ChEMBL v28 activities with PDBbind v2019 templates. ⚠️ Exact similarity thresholds not
retrieved.

- **v1:** **69,816 modelled complexes** ⚠️ · [GitHub hnlab/BindingNet](https://github.com/hnlab/BindingNet) ·
  http://bindingnet.huanglab.org.cn/ · Li, Shen, Zhu et al., *JCIM* 2024.
- **v2:** **689,796 complexes across 1,794 protein targets** ⚠️ (~10× v1) — "Augmented
  BindingNet dataset for enhanced ligand binding pose predictions using deep learning",
  *npj Drug Discovery* 2025,
  [s44386-024-00003-0](https://www.nature.com/articles/s44386-024-00003-0). Paired with
  Uni-Mol Docking it is reported to beat AlphaFold3 on low-similarity ligands
  ([DeepModeling blog](https://blogs.deepmodeling.com/Uni-Mol_22_01_2025/)).

Access: moderate subset mmCIF at
[Zenodo 10.5281/zenodo.14561479](https://doi.org/10.5281/zenodo.14561479) (~26.4 GB);
crystal templates at [Zenodo 10.5281/zenodo.11218293](https://doi.org/10.5281/zenodo.11218293).
Easiest single source is the HF mirror ✅ **`chenbq18/BindingNet_v2_fixed`** — **97.1 GiB**,
all three quality tiers plus quality tables.

> ⚠️ **License conflict, unresolved.** The v1 site says **CC BY-NC 4.0**, the v2 Zenodo
> records say **CC BY 4.0**, the HF mirror says **MIT**. Three different answers. At 97 GiB
> it also does not fit our Drive budget. **Skip BindingNet for the first run** and resolve
> the licence only if we later need pose-level data at that scale.

---

### B.5 Papyrus — the best starting point for multi-target affinity

- **Version 2024.2** (internally "05.7"), published **2024-10-24**, built on ChEMBL v34 +
  IUPHAR/BPS Guide to PHARMACOLOGY + Pickett et al. MMP-12. ❌ No newer release found.
- **License: CC BY-SA 4.0**; the `papyrus-scripts` code is **MIT**.
- Changes in 2024.2: fixed multi-pChEMBL metadata, removed mixtures, and **tightened the
  Papyrus++ reproducibility cutoff from 2 → 0.5 log units**, which shrank `++` substantially.

Size (⚠️ paper-era ChEMBL v30 figures; ❌ 05.7 totals not retrievable;
[PMC9824924](https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/)):

- Full: **59,775,087** compound–protein data points, **1,270,570** unique 2D compounds,
  **6,926** proteins across 499 organisms.
- Quality tiers: high **1,238,835** / medium **335,661** / low **58,200,591**. Most of the raw
  volume is ExCAPE-DB and is low quality — **for triage, use `++`.**

Files — ✅ the two that matter, HEAD-verified:

| File | Size ✅ |
|---|---|
| `05.7++_combined_set_without_stereochemistry.tsv.xz` | **54.1 MiB** ← start here |
| `05.7_combined_set_without_stereochemistry.tsv.xz` | **717 MiB** |
| `05.7_combined_set_protein_targets.tsv.xz` | 1.78 MB ⚠️ |

```bash
curl -L -O "https://zenodo.org/records/13987985/files/05.7%2B%2B_combined_set_without_stereochemistry.tsv.xz"
curl -L -O "https://zenodo.org/records/13987985/files/05.7_combined_set_without_stereochemistry.tsv.xz"
```

```bash
pip install papyrus-scripts
papyrus download -V latest
```

```python
from papyrus_scripts import PapyrusDataset
ds = PapyrusDataset(version='2024.09.2', plusplus=True)
df = ds.keep_source(['chembl']).keep_quality('high').to_dataframe()
# target ids are UniProt accessions with a _WT suffix; join against
# 05.7_combined_set_protein_targets.tsv.xz
```

⚠️ The repo calls the release `2024.09` while Zenodo calls it `2024.2 / 05.7`; the naming
could not be reconciled. DOIs: Zenodo
[10.5281/zenodo.13987985](https://doi.org/10.5281/zenodo.13987985) (concept DOI
[10.5281/zenodo.13787633](https://doi.org/10.5281/zenodo.13787633)); 4TU
[10.4121/16896406](https://doi.org/10.4121/16896406) holds **only early versions** (latest
there is v3, 2022) — later releases are **Zenodo only**. Paper: Béquignon et al.,
*J Cheminform* 2023, 15:3, DOI
[10.1186/s13321-022-00672-x](https://doi.org/10.1186/s13321-022-00672-x).

> **Papyrus++ at 54 MiB is the single best starting point for our affinity model**:
> pre-curated, pre-standardised, quality-tiered, UniProt-keyed (so it joins straight onto
> section A's target list), and it fits in RAM on a free Colab instance.

**❗ There is no HF mirror of the real Papyrus bioactivity set.** `Leash-Biosciences/papyrus-decoy-eval`
is a decoy-eval set, not Papyrus. Get it from Zenodo.

---

### B.6 LIT-PCBA — and the 2025 audit that changes how you use it

Tran-Nguyen, Jacquemard & Rognan, *JCIM* 2020, 60(9):4263–4273, DOI
[10.1021/acs.jcim.0c00155](https://doi.org/10.1021/acs.jcim.0c00155).

**Two official counts that disagree:**

| Source | Targets | Actives | Inactives |
|---|---|---|---|
| Paper abstract | 15 | **7,844** | **407,381** |
| [Download site](https://drugdesign.unistra.fr/LIT-PCBA/) | 15 | **7,761** | **382,674** (unique) |

Use the **site** numbers for the distributed files — they are the deduplicated counts.

Targets: ALDH1, IDH1, PKM2, ESR1_ago, ESR1_ant, OPRK1, TP53, VDR, PPARG, FEN1, GBA, KAT2A,
MAPK1, MTORC1, **ADRB2**. Note that **PPARG and ADRB2 are both on our section-A shortlist** —
LIT-PCBA gives us a ready-made screening benchmark for two of our chosen targets. Per-target
range ⚠️: actives 24 (PPARG) → 5,363 (ALDH1); inactives 3,345 (TP53) → 350,718 (FEN1).

How it was built: start from **149 dose-response confirmatory** PubChem BioAssays (not
single-point primary screens) → remove false positives and assay artifacts → keep actives and
inactives in matched molecular-property ranges → keep only targets with a ligand-bound PDB
structure → a feasibility pre-screen (2D fingerprint + 3D shape + docking) retaining only
targets where some method gives ≳2× top-1% enrichment → **AVE** train/validation splitting.

```bash
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/full_data.tgz        # 51.3 MiB ✅
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/AVE_unbiased.tgz     # 54.7 MiB ✅
wget https://drugdesign.unistra.fr/LIT-PCBA/Files/LIT-PCBA_bioactivities.xlsx
```

**License: ❌ none stated** — the site carries only a copyright notice. Treat as "ask before
redistributing".

#### ⚠️ The important finding

**Huang, Knight & Naprienko (SieveStack), "Data Leakage and Redundancy in the LIT-PCBA
Benchmark", [arXiv:2507.21404](https://arxiv.org/abs/2507.21404) (July 2025)**; audit code
[github.com/sievestack/LIT-PCBA-audit](https://github.com/sievestack/LIT-PCBA-audit).
Non-peer-reviewed; these are the authors' claims:

- **2,491 unique inactives appear in both training and validation.**
- 2,945 inactives duplicated *within* training sets; 789 within validation.
- **>350 active-labelled analogue pairs** across train/validation (ECFP2 Tanimoto ≥ 0.6),
  **323 of them in ALDH1**.
- For some targets, **>80% of query ligands are near-duplicates** (Tanimoto ≥ 0.9).
- **A memorisation baseline with zero learnable parameters matches or beats published SOTA**
  deep-learning and 3D-similarity results.
- The authors deliberately provide **no corrected version**.

> **Rule: deduplicate and re-split LIT-PCBA ourselves** (scaffold- or similarity-clustered).
> **Do not trust the shipped AVE split**, and always report a nearest-neighbour memorisation
> baseline next to the model.

HF mirrors of docked poses: `Kingldore/lit-pcba-vs` ✅ (**~939 GiB** of Gnina-1.0 poses from
the Koes lab — far too large for us), `THU-ATOM/litpcba_airdd` ✅.

---

### B.7 DUD-E — and precisely why not to use it

Mysinger, Carchia, Irwin & Shoichet, *J Med Chem* 2012, 55(14):6582–6594, DOI
[10.1021/jm300687e](https://doi.org/10.1021/jm300687e) ([PMC3405771](https://pmc.ncbi.nlm.nih.gov/articles/PMC3405771/)).

✅ **102 targets**; raw ligands **66,695**; **clustered ligands 22,886**; experimental decoys
**9,219**; clustered ligands × unique charge states **28,377**; and — ✅ **measured by counting
every distributed `.ism` record, an exact match to the paper's Table 1 — 1,411,214
computational decoys.**

> ❗ **The most-misquoted fact in the field, and we nearly got it wrong too.** "50 decoys per
> active" is **per unique ligand protonation/charge state** (28,377 of them), **not** per
> clustered ligand. So the real total is **1.41 M**, not 22,886 × 50 = 1.14 M, and the real
> ratio is a **mean of 61.9 decoys per active** (range 50.0 for KITH to 104.1 for FPPS). 91 of
> the 102 per-target decoy counts are exact multiples of 50; the 11 that are not are where the
> 9,219 experimental decoys were pooled in. **Cite 1,411,214 and ~1:62.**

Selection: ChEMBL09, affinity better than 1 µM, MW < 600, <20 rotatable bonds; clustered by
**Bemis–Murcko atomic frameworks**; decoys drawn from ZINC by matching **6 properties** (MW,
logP, rotatable bonds, HBA, HBD, net charge) over 7 window-widening steps; candidates sorted
by max **ECFP4** Tanimoto to any ligand with **the most dissimilar 25% retained**; then up to
50 picked at random per charge state. One X-ray structure per target.

```bash
wget https://dude.docking.org/db/subsets/all/all.tar.gz   # 2.59 GiB ✅
```

✅ Note the `Last-Modified` header on that archive: **15 July 2012.** It has never been
regenerated. The site footer now redirects visitors to **dudez.docking.org**.

#### The full case against it

1. **Analogue bias + artificial enrichment, by construction.** Actives are congeneric series;
   decoys are property-matched but **topologically dissimilar to every active by design**. A
   model scores well by learning "looks like the actives' chemotypes" vs "looks like a ZINC
   decoy", with **zero receptor information**.

2. **Chen et al. 2019**, "Hidden bias in the DUD-E dataset leads to misleading performance of
   deep learning in structure-based virtual screening", *PLoS ONE* 14(8):e0220113, DOI
   [10.1371/journal.pone.0220113](https://doi.org/10.1371/journal.pone.0220113)
   ([PMC6701836](https://pmc.ncbi.nlm.nih.gov/articles/PMC6701836/)):
   - Receptor+ligand CNN **AUC 0.98 ± 0.02**. **Ligand-only, no receptor at all: also 0.98**,
     with **R² = 0.98** between the two across targets.
   - **All 102 targets** gave ligand-only AUC > 0.9.
   - **5 actives + 5 decoys** sufficed for high AUC on many targets.
   - Replacing DUD-E decoys with other targets' actives collapsed a multi-target model from
     ~0.80 AUC to **near random** → **decoy bias is the main driver**.

3. **Sieg, Flachsenberg & Rarey 2019**, "In Need of Bias Control: Evaluating Chemical Data for
   Machine Learning in Structure-Based Virtual Screening", *JCIM* 59(3):947–961, DOI
   [10.1021/acs.jcim.8b00712](https://doi.org/10.1021/acs.jcim.8b00712) — **"bias is learned
   implicitly and unnoticed from standard benchmarks"**.

4. **Wallach & Heifets 2018**, "Most Ligand-Based Classification Benchmarks Reward
   Memorization Rather than Generalization", *JCIM* 58(5):916–932,
   [arXiv:1706.06619](https://arxiv.org/abs/1706.06619), DOI
   [10.1021/acs.jcim.7b00403](https://doi.org/10.1021/acs.jcim.7b00403) — introduced **AVE**.
   Across ✅ **seven** benchmarks (not nine — a figure worth correcting, since it is widely
   misquoted), AVE bias correlates with model performance *"irrespective of the predicted
   property, chemical fingerprint, similarity measure, or previously applied unbiasing
   techniques. Therefore, it may be the case that the previously reported performance of most
   ligand-based methods can be explained by overfitting to benchmarks rather than good
   prospective accuracy."* **The split's AVE score drives the reported score, not the
   chemistry learned.** See §D.6 for why AVE is a good *diagnostic* and a bad *remedy*.

5. **Stein et al. 2021** adds *physical* bias: **35% of DUD-E decoys are neutral while AmpC's
   ligands are mostly anionic**, so charge/desolvation weighting alone separates them.

6. **Scantlebury, Brown, Von Delft & Deane**, *JCIM* 2020, 60(8):3722–3730, DOI
   [10.1021/acs.jcim.0c00263](https://doi.org/10.1021/acs.jcim.0c00263) — DUD-E-trained
   models **fail on unseen target classes** unless data is augmented to force use of protein
   information.

❌ The "Yang et al. hidden bias" paper named in the brief could not be located; the Chen 2019
*PLoS ONE* paper above is almost certainly the intended reference.

#### Better benchmarks

**DEKOIS 2.0** — Bauer, Ibrahim, Vogel & Boeckler, *JCIM* 2013, 53(6):1447–1462, DOI
[10.1021/ci400115b](https://doi.org/10.1021/ci400115b). ✅ **81 benchmark sets**, with
enhanced physicochemical matching and better removal of **LADS** (latent actives in the decoy
set). ✅ **The site is live** — `dekois.com` 301-redirects to
[uni-tuebingen.de/…/prof-dr-f-boeckler/dekois/](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/pharmaziebiochemie/teilbereich-pharmazie-pharmazeutisches-institut/pharmazeutische-chemie/prof-dr-f-boeckler/dekois/)
(verified 2026-10-04 — an earlier fetch failed on SSL against the bare domain).

✅ **Composition, measured by downloading and counting every record** — so the widely-repeated
"40 actives + 1,200 decoys" figure is **correct after all**, and an earlier note here calling
it unsourced is withdrawn: **81 targets**; **40 actives/target** for 80 targets and **39** for
SARS-HCoV → **3,239 actives**; **1,200 decoys/target** (1,170 for SARS-HCoV) → **97,170
decoys**; **exactly 30 decoys per active**. Actives come from **BindingDB**, not ChEMBL, with
a PAINS filter added in 2.0.

Decoy selection matches **8** descriptors rather than DUD-E's 6 — MW, logP, HBA, HBD,
rotatable bonds, plus **positive charges, negative charges and aromatic rings** (the last
three new in 2.0) — from a 15 M ZINC subset, then combines two scores: **PSS**
(physicochemical similarity) and **LADS** (Latent Actives in the Decoy Set), which penalises
decoys sharing bioactive FCFP_6 fragments with the actives, weighted by fragment frequency and
heavy-atom size. Set quality is judged by the **DOE score** (Deviation from Optimal
Embedding); Bauer et al. report DOE < 0.2 for **89% of the 81 targets**.
⚠️ **There is no "Latent Hit Enrichment" metric in DEKOIS** — the two set-quality metrics are
**LADS** and **DOE**.

⚠️ **Download trap:** the four ZIPs (`ligands.zip` 2.78 MB, `decoys_part_1/2/3.zip`
35.8/39.2/16.8 MB) are served behind **TYPO3 signed `/securedl/sdl-<JWT>/…` URLs with ~25 h
expiry** — you cannot hard-code them, you must load the page each time. **The archives contain
ligands and decoys only; no receptor structures** (PDB codes are in the paper's SI).

**DUDE-Z** — Stein, Yang, Balius, O'Meara, Lyu, Young, Tang, Shoichet & Irwin,
**"Property-*Unmatched* Decoys in Docking Benchmarks"**, *JCIM* 2021, 61(2):699–714, DOI
[10.1021/acs.jcim.0c00598](https://doi.org/10.1021/acs.jcim.0c00598)
([PMC7913603](https://pmc.ncbi.nlm.nih.gov/articles/PMC7913603)).
✅ [dudez.docking.org](https://dudez.docking.org) resolves. What changes:

- Charge matching on **actual 3D dockable protomer states**, not 2D SMILES.
- Ligands clustered at ECFP4 Tanimoto 0.7 to de-weight congeneric series.
- Decoys must have **ECFP4 similarity ≤ 0.35** to the ligands.
- **"Extrema"** — molecules at extreme net charges (−2…+2), exposing
  electrostatics/desolvation mis-weighting.
- **"Goldilocks"** — ~1.1 M lead-like molecules representing library properties *without*
  matching to ligands. (Available on HF as `introvoyz041/ZINC_goldilocks_SMILES`.)
- Current size: **43 targets, 2,312 ligands, 69,994 decoys**, plus **732,309 extrema** and
  **1,145,472 Goldilocks** molecules.
- Ships a bootstrap script for CIs — **the average 95% CI is ~9.4 adjusted-logAUC units**,
  i.e. **most published enrichment differences are noise.** This single number is the most
  useful thing in the DUDE-Z paper for us.

**MUV** — Rohrer & Baumann, *JCIM* 2009, 49(2):169–184, DOI
[10.1021/ci8002649](https://doi.org/10.1021/ci8002649). PubChem-derived, removes unselective
compounds, uses nearest-neighbour + topological optimisation to control both analogue bias and
artificial enrichment. ⚠️ 17 targets, 30 actives / 15,000 decoys each (unverified). So hard
and so active-poor that it is often unusable for ML.

> **Decisions:** (1) **Never** train or evaluate on raw DUD-E decoys. (2) Report **EF1% with
> bootstrap CIs**, and treat a <1-unit logAUC gap as nothing. (3) For prospective-realistic
> negatives, use **DUDE-Z Goldilocks** or sample unlabelled property-matched ZINC ourselves.

---

### B.8 Therapeutics Data Commons — the ADMET Benchmark Group

This is **the** ADMET target for this project: 22 tasks, a fixed test set, a published 5-seed
protocol, and a live leaderboard to be measured against.

Every task is a **scaffold split**; TDC fixes **one test set** per task and re-splits the
`train_val` portion per seed. ❌ Exact fractions are not stated on the pages (commonly cited
as a 20% scaffold test set, unverified).

Naming trap: `group.get()` takes Title_Case (`'Caco2_Wang'`); `group.evaluate()` returns
lowercase keys (`{'caco2_wang': {'mae': 0.234}}`).

#### All 22 tasks ⚠️ (from [the overview page](https://tdcommons.ai/benchmark/admet_group/overview/))

| # | Dataset | Molecules | Type | Metric | Page slug |
|---|---|---|---|---|---|
| 1 | `Caco2_Wang` | 906 | Regression | MAE | `01caco2` |
| 2 | `HIA_Hou` | 578 | Binary | AUROC | `02hia` |
| 3 | `Pgp_Broccatelli` | 1,212 | Binary | AUROC | `03pgp` |
| 4 | `Bioavailability_Ma` | 640 | Binary | AUROC | `04bioav` |
| 5 | `Lipophilicity_AstraZeneca` | 4,200 | Regression | MAE | `05lipo` |
| 6 | `Solubility_AqSolDB` | 9,982 | Regression | MAE | `06aqsol` |
| 7 | `BBB_Martins` | 1,975 | Binary | AUROC | `07bbb` |
| 8 | `PPBR_AZ` | 1,797 | Regression | MAE | `08ppbr` |
| 9 | `VDss_Lombardo` | 1,130 | Regression | Spearman | `09vdss` |
| 10 | `CYP2C9_Veith` | 12,092 | Binary | **AUPRC** | `10cyp2c9i` |
| 11 | `CYP2D6_Veith` | 13,130 | Binary | **AUPRC** | `11cyp2d6i` |
| 12 | `CYP3A4_Veith` | 12,328 | Binary | **AUPRC** | `12cyp3a4i` |
| 13 | `CYP2C9_Substrate_CarbonMangels` | 666 | Binary | **AUPRC** | `13cyp2c9s` |
| 14 | `CYP2D6_Substrate_CarbonMangels` | 664 | Binary | **AUPRC** | `14cyp2d6s` |
| 15 | `CYP3A4_Substrate_CarbonMangels` | 667 | Binary | **AUROC** ← inconsistent with 13/14 | `15cyp3a4s` |
| 16 | `Half_Life_Obach` | 667 | Regression | Spearman | `16halflife` |
| 17 | `Clearance_Hepatocyte_AZ` | 1,020 | Regression | Spearman | `17clhepa` |
| 18 | `Clearance_Microsome_AZ` | 1,102 | Regression | Spearman | `18clmicro` |
| 19 | `LD50_Zhu` | 7,385 | Regression | MAE | `19ld50` |
| 20 | `hERG` | 648 | Binary | AUROC | `20herg` |
| 21 | `AMES` | 7,255 | Binary | AUROC | `21ames` |
| 22 | `DILI` | 475 | Binary | AUROC | `22dili` |

Tally: 5 MAE, 6 Spearman, 11 classification. `CYP1A2_Veith` and `CYP2C19_Veith` exist as TDC
datasets but are **not** in the 22-task group. Units: hepatocyte clearance
µL·min⁻¹·10⁶ cells⁻¹; Caco2 cm/s (log scale); solubility log mol/L.

**Note the scale.** Nine of the 22 tasks have **fewer than 1,200 molecules**, and three have
fewer than 700. These are small-data problems where a 300 M-parameter transformer has nothing
to learn from — which is exactly what the leaderboard shows.

#### Current leaderboard top-3 (served 2026-10-04; ⚠️ the pages carry **no date or version stamp**)

Lower is better for MAE, higher for the rest; `±` is std over 5 seeds.

| # | Task (metric) | #1 | #2 | #3 |
|---|---|---|---|---|
| 1 | Caco2 (MAE) | **CaliciBoost 0.256** ±0.006 | XG Boost 0.274 | MapLight 0.276 |
| 2 | HIA (AUROC) | **MiniMol 0.993** ±0.005 | DeepMol 0.990 | MapLight+GNN 0.989 |
| 3 | Pgp (AUROC) | **MapLight+GNN 0.938** ±0.002 | ZairaChem 0.935 | MapLight 0.930 |
| 4 | Bioav (AUROC) | **MiniMol 0.942** ±0.002 | MapLight+GNN 0.938 | ZairaChem 0.935 |
| 5 | Lipo (MAE) | **MiniMol 0.456** ±0.008 | Chemprop-RDKit 0.467 | Chemprop 0.470 |
| 6 | AqSol (MAE) | **MiniMol 0.741** ±0.013 | Chemprop-RDKit 0.761 | DeepMol 0.775 |
| 7 | BBB (AUROC) | **MiniMol 0.924** ±0.003 | CFA 0.920 | MapLight 0.916 |
| 8 | PPBR (MAE) | **Gradient Boost 7.440** ±0.024 | MapLight+GNN 7.526 | MapLight 7.660 |
| 9 | VDss (Spearman) | **MapLight+GNN 0.713** ±0.007 | MapLight 0.707 | CFA 0.628 |
| 10 | CYP2C9 inh. (AUPRC) | **MapLight+GNN 0.859** ±0.001 | ContextPred 0.839 | AttrMasking 0.829 |
| 11 | CYP2D6 inh. (AUPRC) | **MapLight+GNN 0.790** ±0.001 | ContextPred 0.739 | MapLight 0.723 |
| 12 | CYP3A4 inh. (AUPRC) | **MapLight+GNN 0.916** ±0.000 | ContextPred 0.904 | AttrMasking 0.902 |
| 13 | CYP2C9 subs. (AUPRC) | **MiniMol 0.474** ±0.025 | ZairaChem 0.441 | MapLight+GNN 0.437 |
| 14 | CYP2D6 subs. (AUPRC) | **ContextPred 0.736** ±0.024 | DeepMol 0.731 | MapLight+GNN 0.720 |
| 15 | CYP3A4 subs. (AUROC) | **CFA 0.667** ±0.019 | MiniMol 0.663 | CNN (DeepPurpose) 0.662 |
| 16 | Half-life (Spearman) | **CFA 0.576** ±0.025 | MapLight 0.562 | MapLight+GNN 0.557 |
| 17 | Hepatocyte CL (Spearman) | **CFA 0.536** ±0.020 | MapLight+GNN 0.498 | MapLight 0.466 |
| 18 | Microsome CL (Spearman) | **MapLight+GNN 0.630** ±0.010 | MiniMol 0.628 | MapLight 0.626 |
| 19 | LD50 (MAE) | **BaseBoosting 0.552** ±0.009 | ADMETrix 0.573 | MiniMol 0.585 |
| 20 | hERG (AUROC) | **MapLight+GNN 0.880** ±0.002 | CFA 0.875 | SimGCN 0.874 |
| 21 | AMES (AUROC) | **ZairaChem 0.871** ±0.002 | ADMETrix 0.870 | MapLight+GNN 0.869 |
| 22 | DILI (AUROC) | **MiniMol 0.956** ±0.006 | ZairaChem 0.925 | AttrMasking 0.919 |

**Read this table correctly.** The winners are **MiniMol** (a ~10 M-parameter pretrained
embedding plus a small head) and **MapLight / MapLight+GNN** (CatBoost on fingerprints +
descriptors, optionally with GIN embeddings). **Gradient-boosted trees on fingerprints hold
or share #1 on roughly half of these 22 tasks.** A big GPU model is not required to be
competitive here — which is the best news in this document for a 24 h L4 budget.

**Treat the leaderboard as an upper bound on claims, not a target.** Entries are
**self-submitted and unaudited**. The clearest tell: **0.942 AUROC on a 640-molecule
bioavailability task** (re-fetched and confirmed as served) is implausible against the
historical ~0.75 ceiling for that endpoint.

#### TDC-2 / PyTDC

- `pip install PyTDC` — [PyPI](https://pypi.org/project/PyTDC/) shows **1.1.15, released
  2025-03-31**, MIT. ⚠️ Run `pip index versions PyTDC` to confirm.
- [github.com/mims-harvard/TDC](https://github.com/mims-harvard/TDC) (MIT) ·
  [tdc.readthedocs.io](https://tdc.readthedocs.io/) · [tdcommons.ai](https://tdcommons.ai)
- Papers: original TDC, NeurIPS 2021 Datasets & Benchmarks (Huang et al.); *Nature Chemical
  Biology* 2022, "Artificial intelligence foundation for therapeutic science"; **TDC-2:
  "PyTDC: A multimodal machine learning training, evaluation, and inference platform for
  biomedical foundation models"**, Velez-Arce, Caraballo & Zitnik,
  [arXiv:2505.05577](https://arxiv.org/abs/2505.05577), ICML 2025.
- TDC-2 added an API-first architecture over continuously-updated sources, a model
  server/hub, and single-cell therapeutics tasks. **The ADMET group is unchanged — same 22
  tasks.**

**Licensing:** code **MIT**; **all 22 ADMET datasets listed CC BY 4.0** ⚠️ on the
[ADME](https://tdcommons.ai/single_pred_tasks/adme/) and
[Tox](https://tdcommons.ai/single_pred_tasks/tox/) pages. ❗ **But that is TDC's re-license of
its own curated copies.** Upstream sources carry their own terms — AqSolDB, the AstraZeneca
ChEMBL depositions (**CC BY-SA via ChEMBL**), FDA NCTR DILIrank, the Veith PubChem data. ❌ No
upstream licence verified, and `tdcommons.ai/faq/` returns **404**. Get provenance reviewed
before commercial deployment. (`HydrationFreeEnergy_FreeSolv` is **CC BY-NC-SA 4.0** — but it
is not in the ADMET group.)

#### Official evaluation loop — use exactly this

Protocol requires **≥5 runs with seeds [1,2,3,4,5]**; each seed gets a different train/valid
split, the test set is constant.

```python
from tdc.benchmark_group import admet_group
group = admet_group(path='data/')
predictions_list = []

for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get('Caco2_Wang')
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train, valid = group.get_train_valid_split(
        benchmark=name, split_type='default', seed=seed)
    # train on `train`, early-stop on `valid`
    # frames have columns: Drug_ID, Drug (SMILES), Y
    predictions_list.append({name: y_pred_test})

results = group.evaluate_many(predictions_list)   # -> {dataset: [mean, std]}
```

Iterate `group.dataset_names` for all 22. **For binary tasks submit probabilities, not
labels** — the metrics are AUROC/AUPRC.

---

### B.9 MoleculeNet — useful, but the SOTA table is a trap

Row counts below were **measured by `wc -l` on the DeepChem S3 CSVs** ⚠️. DeepChem's doc page
gives only rounded figures (HIV "40,000", Tox21 "8,000") — ignore those.

| Dataset | Measured rows | Paper Table 1 | Tasks | Type / metric | **Paper's** split | DeepChem loader |
|---|---|---|---|---|---|---|
| BACE | **1,513** | 1,522 | 1 | clf / ROC-AUC | **scaffold** | `load_bace_classification` (also `load_bace_regression`, pIC50) |
| BBBP | **2,050** | 2,053 | 1 | clf / ROC-AUC | **scaffold** | `load_bbbp` |
| HIV | **41,127** | 41,913 | 1 | clf / ROC-AUC | **scaffold** | `load_hiv` |
| Tox21 | **7,831** | 8,014 | **12** | clf / ROC-AUC (task mean) | **random** | `load_tox21` |
| ClinTox | **~1,484** | 1,491 | **2** | clf / ROC-AUC | **random** | `load_clintox` |
| ESOL (Delaney) | **1,128** | 1,128 | 1 | reg / RMSE | random | `load_delaney` |
| FreeSolv | **642** | 643 | 1 | reg / RMSE | random | `load_freesolv` |
| Lipophilicity | **4,200** | 4,200 | 1 | reg / RMSE | random | `load_lipo` |

**Two corrections to the brief as written:**

1. **The MoleculeNet paper recommends *random* splits for Tox21 and ClinTox**, not scaffold.
   Scaffold is recommended only for BACE, BBBP and HIV. Modern papers (GROVER, GEM, Uni-Mol,
   ChemBERTa-2) apply scaffold to everything — one reason their numbers do not match the
   original paper's.
2. **ClinTox is ~1,484**, and the HF mirror `scikit-fingerprints/MoleculeNet_ClinTox` reports
   exactly **1,477**.

```bash
B=https://deepchemdata.s3.us-west-1.amazonaws.com/datasets
wget $B/bace.csv $B/BBBP.csv $B/HIV.csv $B/tox21.csv.gz $B/clintox.csv.gz \
     $B/delaney-processed.csv $B/SAMPL.csv $B/Lipophilicity.csv
# ⚠️ freesolv.csv (lowercase) returns 403 — FreeSolv is SAMPL.csv
```

#### Scaffold-split "SOTA" — why not to quote it

ROC-AUC for classification, RMSE for regression.

| Dataset | MoleculeNet 2018 (best, **random**) | GROVER_large (own paper) | GROVER_large (**re-run in GEM**) | GEM | Uni-Mol | MoLFormer-XL | ChemBERTa-2 MTR-77M |
|---|---|---|---|---|---|---|---|
| BACE | 0.867 (RF) | 0.894 | **0.810** | 0.856 | 0.857 | 0.882 | — |
| BBBP | 0.729 (KernelSVM) | 0.940 | **0.695** | 0.724 | 0.729 | **0.937** | 0.728 |
| HIV | 0.792 (KernelSVM) | — | — | 0.806 | 0.808 ❌ | 0.822 | — |
| Tox21 | 0.829 (GraphConv) | 0.831 | **0.735** | 0.781 | 0.796 ❌ | 0.847 | — |
| ClinTox | 0.832 (Weave) | 0.944 | **0.762** | 0.901 | 0.919 | 0.948 | 0.563 |
| ESOL | 0.58 (MPNN) | 0.831 | 0.895 | 0.798 | 0.788 ❌ | 0.279 (random) | 0.889 |
| FreeSolv | 1.15 (MPNN) | 1.544 | 2.272 | 1.877 | 1.480 ❌ | 0.231 (random) | — |
| Lipo | 0.655 (GraphConv) | 0.560 | 0.823 | 0.660 | 0.603 ❌ | 0.529 (random) | 0.798 |

Sources: [MoleculeNet, PMC5868307](https://pmc.ncbi.nlm.nih.gov/articles/PMC5868307/) /
[10.1039/C7SC02664A](https://doi.org/10.1039/C7SC02664A) ·
[GEM, arXiv:2106.06130](https://arxiv.org/abs/2106.06130) (scaffold 8:1:1, 4 runs) ·
[GROVER, arXiv:2007.02835](https://arxiv.org/abs/2007.02835) (scaffold 8:1:1, 3 runs) ·
[Uni-Mol, OpenReview](https://openreview.net/forum?id=6K2RM6wVqKu) ·
[MoLFormer, arXiv:2106.09553](https://arxiv.org/abs/2106.09553) ·
[ChemBERTa-2, arXiv:2209.01712](https://arxiv.org/abs/2209.01712).

**Why to distrust all of it:**

- **GROVER_large on BBBP is 0.940 in GROVER's own paper and 0.695 in GEM's re-run of the same
  model.** ClinTox: 0.944 vs 0.762. **Same model, same benchmark, 24 points apart.**
- MoLFormer's BBBP 0.937 sits ~20 points above Uni-Mol's and GEM's 0.72–0.73 — almost
  certainly a different scaffold-split implementation.
- MoLFormer's **regression** numbers use **random** splits and are not RMSE-comparable to the
  scaffold work beside them.
- ❌ The Uni-Mol HIV / Tox21 / ESOL / FreeSolv / Lipo values above are from recall, not the
  paper's table — **re-check before citing.**
- ❌ **Uni-Mol2 ([arXiv:2406.14969](https://arxiv.org/abs/2406.14969)) evaluates only QM9 and
  COMPAS-1D — there are no Uni-Mol2 MoleculeNet numbers at all.**

**The citation to use when saying "MoleculeNet comparisons are unreliable":**
**Deng et al., "A systematic study of key elements underlying molecular property
prediction", *Nature Communications* 14, 6395 (2023)**, DOI
[10.1038/s41467-023-41948-6](https://doi.org/10.1038/s41467-023-41948-6)
([PMC10575948](https://pmc.ncbi.nlm.nih.gov/articles/PMC10575948/),
[arXiv:2209.13492](https://arxiv.org/abs/2209.13492)). They trained **62,820 models**,
repeating each split 30× with significance tests, and found:

- *"Discrepancies in the actual data split across the literature can entail unfair performance
  comparison"* — gains without repeat-and-test analysis can be **"mere statistical noise"**.
- Scaffold split produces **larger label-distribution gaps** between train and test than
  random split.
- **A Random Forest on fixed fingerprints matched or beat the learned representations on most
  datasets** — RF was *best* on BACE, BBBP, ESOL and Lipo under scaffold split.

#### A better-maintained alternative: OGB

From [ogb.stanford.edu/docs/graphprop](https://ogb.stanford.edu/docs/graphprop/):

| | ogbg-molhiv | ogbg-molpcba |
|---|---|---|
| Graphs | **41,127** | **437,929** |
| Tasks | 1 binary | 128 binary |
| Split | scaffold (OGB's own) | scaffold |
| Metric | ROC-AUC | Average Precision |

**ogbg-molhiv test ROC-AUC:** Multi-RF Fusion + Multi-GNN **0.8476 ±0.0002** (2026-03-21),
HyperFusion 0.8475 ±0.0003 (2024-02-24), PAS+FPs 0.8420, HIG 0.8403, DeepAUC 0.8352,
FingerPrint+GMAN 0.8244, Neural FingerPrints 0.8232, Graphormer+FPs 0.8225.
**ogbg-molpcba test AP:** HyperFusion **0.3204**, HIG 0.3167, Graphormer 0.3140, PDF 0.3031.

Two things to take from this: the molhiv top three are **all fingerprint-fusion methods**, and
**the 2026 #1 beats the 2024 #2 by 0.0001** — the benchmark is saturated. OGB's scaffold split
is not DeepChem's, so these numbers are not comparable to the table above.

---

### B.10 Hugging Face — dataset IDs verified to exist

Every ID below was confirmed by fetching `https://huggingface.co/api/datasets/<id>`.

#### Adjudication of the IDs named in the brief

| Candidate | Verdict |
|---|---|
| `jglaser/binding_affinity` | ✅ exists |
| `jglaser/pdbbind_complexes` | ✅ exists |
| `sagawa/ZINC-canonicalized` | ✅ exists |
| `zpn/zinc20` | ✅ exists |
| `zpn/pubchem_selfies` | ✅ exists |
| `HoangHa/ChEMBL` | ❌ **does not exist**. Real: `HoangHa/chembl-smiles-pretrain`, `HoangHa/chembl-selfies-pretrain` |
| `katielink/moleculenet` | ❌ **does not exist**. Real: **`katielink/moleculenet-benchmark`** |
| `scikit-fingerprints/MoleculeNet_*` | ✅ **all exist** — plus TDC, MoleculeACE, LRGB and MolPILE sets. **The single best find.** |
| `agentlans/*smiles*` | ❌ none. 47 datasets under `agentlans`, zero chemistry |
| `facebook/esm*` (datasets) | ❌ none. `facebook` publishes ESM *models*, not datasets |
| `Hack90/*` | ✅ exists but **DNA/genomics only** |
| `tattabio/*` | ✅ 43 protein/DNA sets (`OMG`, `swissprot`, `ec_classification`) — **no small molecules** |
| `InstaDeepAI/*` | ✅ 31 sets, **genomics/proteomics only** |
| `ChEMBL/*` | ❌ **no such org** on HF datasets |
| `osunlp/*` | ✅ exists, 39 NLP/agent sets; only chemistry one is **`osunlp/SMolInstruct`** |

#### ⭐ `scikit-fingerprints/*` — the best find

One org mirroring MoleculeNet, TDC, MoleculeACE and LRGB as plain CSVs with HF-native parquet
viewers. Works with bare `pandas`/`datasets`; no DeepChem or PyTDC install needed. All
`license: unknown` unless noted.

**MoleculeNet mirrors** (✅ row counts verified):
`MoleculeNet_ESOL` **1,128** · `MoleculeNet_FreeSolv` **642** · `MoleculeNet_Lipophilicity`
**4,200** · `MoleculeNet_BACE` **1,513** · `MoleculeNet_BBBP` **2,039** · `MoleculeNet_HIV`
**41,127** (1.83 MB) · `MoleculeNet_ClinTox` **1,477** · `MoleculeNet_Tox21` **7,831**
(661 KB) · `MoleculeNet_SIDER` **1,427** · `MoleculeNet_MUV` (2.92 MB) ·
`MoleculeNet_ToxCast` (3.04 MB) · `MoleculeNet_PCBA` (365 MB).

**TDC mirrors** — 37 of them, all ✅: `TDC_caco2_wang` (**910 rows**, 30 KB), `TDC_hia_hou`
(**578**), `TDC_bioavailability_ma`, `TDC_pgp_broccatelli`, `TDC_bbb_martins`, `TDC_ppbr_az`,
`TDC_vdss_lombardo`, `TDC_solubility_aqsoldb` (251 KB), `TDC_half_life_obach`,
`TDC_clearance_microsome_az`, `TDC_cyp1a2_veith` (**12,579**, `cc-by-4.0`), `TDC_cyp2c9_veith`
(**12,092**, `cc-by-4.0`), `TDC_cyp2c19_veith` (**12,665**, `cc-by-4.0`), `TDC_cyp2d6_veith`
(✅ **13,130 rows / 657,778 B** — re-verified via the HF datasets-server `size` endpoint; an
earlier reading of 1,513 was wrong), `TDC_cyp3a4_veith` (**12,328**,
`cc-by-4.0`), `TDC_cyp{2c9,2d6,3a4}_substrate_carbonmangels`, `TDC_ames` (**7,278**),
`TDC_dili`, `TDC_herg`, `TDC_herg_karim` (**13,445**), `TDC_herg_central_{at_1um,at_10um,inhib}`,
`TDC_ld50_zhu` (**7,385**), `TDC_carcinogens_lagunin` (**280**), `TDC_skin_reaction` (**404**),
`TDC_pampa_ncats`, `TDC_pampa_approved_drugs`, `TDC_b3db_{classification,regression}`,
`TDC_hlm`, `TDC_rlm`, `TDC_sarscov2_{3clpro_diamond,vitro_touret}`.

**⭐ ~30 `scikit-fingerprints/MoleculeACE_chembl*_{ki,ec50}` activity-cliff sets** —
`chembl204_ki`, `214_ki`, `218_ec50`, `219_ki`, `228_ki`, `231_ki`, `233_ki`, `234_ki`,
`235_ec50`, `236_ki`, `237_{ec50,ki}`, `238_ki`, `239_ec50`, `244_ki`, `262_ki`, `264_ki`,
`287_ki`, `1862_ki`, `1871_ki`, `2034_ki`, `2047_ec50`, `2147_ki`, **`2835_ki`** …
**These are exactly the right test for whether a triage model handles activity cliffs** — and
note that **`chembl2835_ki` is JAK1** (✅ 615 rows) and **`chembl235_ec50` is PPARγ** (✅ 2,349
rows), both tier-1 targets from section A. ✅ These mirrors carry `license: cc-by-sa-3.0`,
correctly inherited from ChEMBL (unlike the `license: unknown` on most of the org's other
sets). Row counts and licences re-verified via
`https://datasets-server.huggingface.co/size?dataset=<id>`.

Also: `scikit-fingerprints/MolPILE` (**9.3 GiB**, 100M–1B molecules from UniChem + PubChem +
Mcule + ChemSpace, with 100k→10M subsets); `LRGB_Peptides-func` / `LRGB_Peptides-struct`
(**15,535 rows each**, `cc-by-nc-4.0`).

#### Protein-ligand affinity / complexes

| ID | Size | License | Notes |
|---|---|---|---|
| **`jglaser/binding_affinity`** | **19.3 GiB** | ❌ unspecified | *The* HF affinity set. Parquet configs per source: `bindingdb`, `biolip`, **`dude`**, `moad`, `pdbbind`, `dcoid`. **Contains DUD-E — strip that config (§B.7).** |
| **`alessandronascimento/clustered_jglaser_binding_affinity`** | — | — | **Clustered re-split of the above. Prefer this over the raw one.** |
| `jglaser/pdb_protein_ligand_complexes` | **16.3 GiB** | ❌ | ~36,000 unique (protein sequence, ligand SMILES) pairs + coordinates; PDB additives filtered |
| `jglaser/protein_ligand_contacts` | 2.5 GiB | ❌ | affinity **and** contact prediction |
| `jglaser/pdbbind_complexes` | — | ❌ | |
| `chenbq18/BindingNet_v2_fixed` | **97.1 GiB** | **MIT** | all three BindingNet v2 tiers (§B.4) |
| **`ibm-research/otter_uniprot_bindingdb_chembl`** | **1.4 GiB** | **MIT** ⭐ | **573,227 SwissProt proteins, 6,207,654 triples** joining UniProt + BindingDB + ChEMBL. [arXiv:2306.12802](https://arxiv.org/abs/2306.12802) |
| `ibm-research/otter_uniprot_bindingdb` | — | MIT | |
| `ibm-research/otter_dude` | — | MIT | DUD-E as a KG — same bias caveats |

**PDBbind mirrors** (note §B.3 — a CC label on a re-upload is not a licence grant):
`THU-ATOM/PDBbind` (per-PDB dirs with `*_ligand.{mol2,sdf}` + `*_protein_*fix.pdb`, incl.
ESMFold-aligned variants, ❌ no license), `THU-ATOM/PDBbind_test`, `photonmz/pdbbindpp-2020`,
`HUBioDataLab/pdbbind_full`, `DaInternet12/pdbbind_{affinities,refined,refined_with_graph_dict}`,
`lawrenceteng/PDBbind_v2019`, and a dozen more.

**PLINDER mirrors:** `StructureCloud/PLINDER` (**56 GiB**, preprocessed `.pt` + parquet
annotations + YAML splits `train_{000..006,apo,pred}` / `val_*` / `test_*`),
`linbc20/plinder_rmsd_split_after_20210930_alltest`, `Synthyra/PLINDER`,
`Synthyra/PLINDER-selfies`.

**Pose validity:** `Jianwen/protein_ligand_cofolding_posebusters`,
`THU-ATOM/posebusters_docking`.

#### ZINC

| ID | Rows | Size | License |
|---|---|---|---|
| `chandar-lab/ZINC_22` | **1.49 B** train / 2.99 M val / 2.99 M test | **608.5 GB** | ❌ |
| | *Stratified HAC 4–49 sample of ZINC-22, deduped + canonicalised, in SMILES / DeepSMILES / SELFIES / SAFE. [arXiv:2508.13408](https://arxiv.org/abs/2508.13408)* | | |
| `zpn/zinc20` | **804,925,861** / 100 M / 101 M | **274 GiB** | **MIT** |
| `sagawa/ZINC-canonicalized` | 90/10 | **4.6 GiB** | **Apache-2.0** |
| **`graphs-datasets/ZINC`** | **220,011 / 24,445 / 5,000** | 427 MB | ❌ |
| | *The benchmarking-GNNs ZINC-12k/250k graph-regression set; PyG-ready* | | |
| **`yairschiff/zinc250k`** | **224,568 / 24,887** | **65 MB** | ❌ |
| | *Cleanest ZINC250k on HF — `smiles, logP, qed, SAS, canonical_smiles` + ring features. [arXiv:2202.02514](https://arxiv.org/abs/2202.02514)* | | |
| `introvoyz041/ZINC_goldilocks_SMILES` | — | — | — |
| | *← the **DUDE-Z Goldilocks** set, i.e. the unbiased-negatives source from §B.7* | | |
| `IrwinLab/LSD_mPro_Fink_2023_noncovalent_Screen2_ZINC22`, `IrwinLab/LSD_NSP3_Mac1_Gahbauer_2022_Everted_ZINC22_screen` | — | — | — |
| | *Real ultra-large-scale docking screens from the Irwin lab* | | |

⚠️ `raphavlas/zinc20` claims 1.9 B rows but is only **2.59 MB** — bogus metadata, treat as a
small sample (its column set is rich, though: `smiles, zinc_id, inchikey, mwt, logp, reactive,
purchasable, tranche_name, generic_scaffold, scaffold, rdkit_*`).

#### PubChem / ChEMBL SMILES corpora

| ID | Rows | Size | License |
|---|---|---|---|
| **`hheiden/PubChem-124M-SMILES-SELFIES-InChI-IUPAC`** | **~124 M** | ~18.3 GB, 124 parquet shards | **CC0-1.0** ⭐ |
| | *PubChem as of **Jan 2026**; raw + RDKit-canonical SMILES, SELFIES, InChI, IUPAC names. **Best-licensed large SMILES corpus found.*** | | |
| **`maykcaldas/smiles-transformers`** | **908,086,717** / 50.5 M / 50.6 M | **151.49 GB** | **MIT** ⭐ |
| | *`text` (SMILES), `formula`, **plus precomputed `NumHDonors, NumHAcceptors, MolLogP, NumHeteroatoms, RingCount, NumRotatableBonds, NumAromaticBonds, NumAcidGroups, NumBasicGroups, Apol`** — saves an RDKit pass over 900 M molecules* | | |
| `MolGen/PubChem-raw` | **109,802,011** | 65.9 GiB | ❌ |
| `zpn/pubchem_selfies` | ~100 M | ~147.1 GB | **OpenRAIL** |
| `sagawa/pubchem-10m-canonicalized` | ~10 M | 3.47 GB | **Apache-2.0** |
| **`HauserGroup/ChEMBL36-SELFIES`** | **2,390,314 / 24,228** | **565 MB** | **CC-BY-4.0** ⭐ |
| | *ChEMBL 36 drug-like, 3–100 heavy atoms, MW ≤ 1000, deduped by InChIKey with deterministic hash splitting* | | |
| **`martinakaduc/ChEMBL_activities`** | 10M–100M band | **2.1 GiB** | **CC-BY-SA-4.0** ⭐ (correctly licensed) |
| `fabikru/chembl-2025-smiles-cleaned` | **2,323,987 / 50,000** | 136 MB | ❌ |
| `molssiai-hub/pubchemqc-pm6`, `molssiai-hub/pubchemqc-b3lyp` | — | — | *DFT/PM6 quantum properties* |

#### MoleculeNet with ready-made splits

**`katielink/moleculenet-benchmark`** ✅ — **Apache-2.0**, ~198 MB, **10 configs with pre-made
train/test/validation CSVs**: BACE, BBBP, ClinTox, ESOL, FreeSolv, HIV, LIPO, QM9, SIDER,
Tox21. ⭐ Use this if you want someone else's splits (but see section D on why you should make
your own). Also `leon14159/moleculenet-unimol-lmdb` (**Uni-Mol-ready LMDB**).

#### Protein sequences, if a protein tower is needed

UniRef: `agemagician/uniref50`, `agemagician/uniref90`, `zpn/uniref90`, `bloyal/uniref50`,
`westlake-repl/AF2_UniRef50`. UniProt/SwissProt: `damlab/uniprot`, `tattabio/swissprot_2025_07_25`,
`jablonkagroup/uniprot_binding_single`. Binding sites: `AmelieSchreiber/{600K,1111K,2600K,12M}_binding_sites`,
`ronig/protein_binding_sequences`, `biomap-research/metal_ion_binding`.

#### ❌ Not on HF, despite searching

A real **Papyrus** mirror (use Zenodo); a recognisable **DUD-E** mirror (only
`ibm-research/otter_dude`); any `ChEMBL/*` org; any `facebook/esm*` **dataset**; a **DEKOIS**
mirror; an **Enamine REAL** mirror beyond two SELFIES pretraining sets
(`HoangHa/enamine-{diverse,nature}-selfies-pretrain` — ⚠️ see the terms problem in §B.11).

---

### B.11 Screening compound libraries

#### ZINC22

Tingle, Tang, Castanon, Gutierrez, Khurelbaatar, Dandarchuluun, Moroz & Irwin, "ZINC-22 — A
Free Multi-Billion-Scale Database of Tangible Compounds for Ligand Discovery", *JCIM* 2023,
63(4):1166–1176, DOI [10.1021/acs.jcim.2c01253](https://doi.org/10.1021/acs.jcim.2c01253)
(open access, [PMC9976280](https://pmc.ncbi.nlm.nih.gov/articles/PMC9976280/)).

**Size at publication (Feb 2023):** *"over **37 billion** enumerated, searchable, commercially
available compounds in 2D, over **4.5 billion** of which have been built in ready-to-dock 3D
formats."* These are **not alternatives** — 37 B is the 2D count, 4.5 B the 3D-built subset.
❌ No newer live count (cartblanche22.docking.org renders nothing to a fetcher); the
`chandar-lab/ZINC_22` HF card says "~70 B synthesizable compounds as of Sept 2024".
Generations are `zinc-22a` … `zinc-22z`; **`g` is "ZINC20 in stock"**, `n` is the largest.

**Tranches** are split by **heavy-atom count × calculated logP**, and 3D tranches further by
**net charge**. Codes look like `H##P###` (`H17` = 17 heavy atoms; a leading `M`/`P` marks
minus/plus logP, e.g. `H05P035`); full filenames add a charge bin, `H05P035M400-0`. ⚠️ Not
every field's meaning was verified.

**Formats:** 2D `.smi`/`.smi.gz`; 3D mol2, sdf, pdbqt, and DOCK `db2` flexibase. **From the
paper: ~1 TB per billion molecules for each of mol2/sdf/pdbqt, and ~200 TB per billion for
db2.** → **On Colab, 2D SMILES only. There is no second option.**

```bash
# rsync one HAC slice of 2D SMILES (documented on the wiki)
rsync -L -a --progress --prune-empty-dirs --delete-excluded \
  --include="*/" --include="H06P???-?.smi.gz" --exclude="*" \
  rsync://files.docking.org/ZINC22-3D/H06 .
```

Otherwise: [cartblanche22.docking.org](https://cartblanche22.docking.org) → Tranches → 2D →
pick a preset → it generates a curl/wget/rsync/Globus/S3 script. Search UIs: **Arthor**
(substructure, arthor.docking.org) and **SmallWorld** (similarity, sw.docking.org). ❌ No
dedicated `zinc22` CLI exists.

**License:** the paper says ZINC-22 is "freely available to everyone" under **CC BY 4.0** ⚠️
(from a paper summary, not a licence page). Some subsets are password-protected to preserve
patentability. ⚠️ **Enamine REAL molecules appear inside ZINC22, and whether Enamine's terms
follow them in is unresolved** — see below.

#### ZINC20

Irwin, Tang, Young, Dandarchuluun, Wong, Khurelbaatar, Moroz, Mayfield & Sayle, *JCIM* 2020,
60(12):6065–6073, DOI [10.1021/acs.jcim.0c00675](https://doi.org/10.1021/acs.jcim.0c00675)
([PMC8284596](https://pmc.ncbi.nlm.nih.gov/articles/PMC8284596/)).

**1.4 billion** compounds, **1.3 billion purchasable**, from 310 catalogs across 150
companies; **736 million lead-like** (rule-of-4), of which **509 million are available in 3D,
ready for docking**. ❌ The download URL pattern could not be confirmed from live pages
(zinc20.docking.org redirects to a captcha). **The practical route to ZINC20 today is
ZINC22's generation `g`.** ❌ No formal licence text retrievable.

**ZINC250k provenance**, since it is the most-cited subset and is routinely mis-attributed: it
is **not** an official ZINC release — it is a random 250,000-molecule drug-like sample of the
~2012-era ZINC that became a benchmark through generative-model papers. ✅ The canonical file
exists in the ChemVAE repo (Gómez-Bombarelli et al., *ACS Cent Sci* 2018, DOI
[10.1021/acscentsci.7b00572](https://doi.org/10.1021/acscentsci.7b00572)):

```bash
wget https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
```

⚠️ Columns are `smiles, logP, qed, SAS`, and **each SMILES cell has a trailing newline** — a
classic gotcha. ⚠️ Provenance between ChemVAE, GrammarVAE and JT-VAE is unresolved; **cite the
file you actually download.** Easiest clean source: `yairschiff/zinc250k` on HF.

#### Enamine REAL / REAL Space — ⚠️⚠️ the biggest legal risk in this document

Three different numbers, all on Enamine's own pages today:

| Thing | Figure | Source |
|---|---|---|
| **REAL Space** (combinatorial synthon space, *not* enumerated) | **94.5 billion** make-on-demand molecules, **169 protocols**, **202,620 in-stock building blocks**; the whole space compresses to "a few hundred megabytes" | [real-space-navigator](https://enamine.net/compound-collections/real-compounds/real-space-navigator) |
| **REAL Database** (enumerated, downloadable) | ✅ **"over 13.6B molecules"**, filtered to MW ≤ 500, SlogP ≤ 5, HBA ≤ 10, HBD ≤ 5, RotB ≤ 10, TPSA ≤ 140 | ✅ [real-database](https://enamine.net/compound-collections/real-compounds/real-database) |
| Stale snapshots | 9.6 B database; 137,000 building blocks / 167 protocols | older pages |

The commonly-quoted "6 B / 8 B / 38 B / 48 B" figures are all **stale snapshots** of one or the
other. The two current anchors are **94.5 B (Space)** and **13.6 B (Database)**.

Subsets ([real-database-subsets](https://enamine.net/compound-collections/real-compounds/real-database-subsets),
all CXSMILES): REAL samples **0.1% = 13.6 M**, **1% = 136 M**, **10% = 1.4 B**; lead-like
**11.3 B** (and a "350/3" subset of **3.1 B**); fragments **57 M** (+1.96 M strict);
natural-like **842.4 M**; PPI modulators **1.53 B**.

❌ **On the "free REAL diversity set": no Enamine page for a set by that name, with a size or
a URL, could be found.** The closest things are the **0.1% REAL sample (13.6 M)** — the
smallest download — and the **Discovery Diversity Set DDS-50K (50,240 compounds)**, which is a
*physical plate set*. **Do not assume a free 20 K/50 K "REAL diversity set" file exists.**

Download mechanics: a [gist for scripted downloads](https://gist.github.com/matteoferla/b1eee8656079d006835f2d8dc159fbb5)
("May 2026" file list) shows base URL **`https://ftp.enamine.net/download/REAL/`** requiring a
**username + password** (an Enamine account), **9 files** of compressed CXSMILES split by HAC,
e.g. `2026.01_Enamine_REAL_HAC_29_38_2.9B_Part_2_CXSMILES.cxsmiles.bz2`. ❌ Total GB not found;
expect several hundred GB for 13.6 B SMILES.

##### ❗ The terms problem — read this before touching REAL

[enamine.net/terms-of-use](https://enamine.net/terms-of-use) is a general site/purchaser terms
document, not a REAL-specific licence, **and it contains a Section 15 "Enamine Data
Non-Use"**. ⚠️ **As summarised by a page-reading model — not read verbatim by us** — it states
the Purchaser is *"strictly prohibited from, directly or indirectly, using, processing,
analyzing, modelling, or incorporating any Enamine Data into any computational system,
including but not limited to AI/ML Systems"* without explicit written authorisation (15.2), and
prohibits sharing data with any third party or system *"for the purpose of processing, analysis,
modelling, or training"* (15.3). "Enamine Data" is defined broadly — structures, outputs,
derivatives — and the summary says this covers training, testing, benchmarking and QSAR, with
**no academic or free exemption found**.

On this reading, training an in-silico triage model on REAL structures — or selling one —
requires written authorisation from Enamine; and the same question propagates to **REAL
molecules reached indirectly via ZINC22** and to the `HoangHa/enamine-*-selfies-pretrain` HF
datasets.

> **❗ Highest-value follow-up in this entire document.** Do **not** rely on the summary above.
> Read Sections 2 and 15 of the terms directly, then ask Enamine **in writing**: (a) does the
> free REAL Database download carry these terms? (b) is model training on it permitted?
> (c) what do commercial/startup licences cost? ❌ No published price found. **Until there is
> an answer, do not train on REAL, and do not treat ZINC22's REAL-derived entries as clean.**

#### ChEMBL approved drugs

✅ **4,225** molecules at `max_phase = 4` in ChEMBL 37 (§B.1), CC BY-SA. This covers the
approved-drug use case and is the recommended source.

#### DrugBank — skip it

- ✅ **Version 5.1.22, dated 2026-06-27**
  ([go.drugbank.com/releases/latest](https://go.drugbank.com/releases/latest)).
- ⚠️ **Bulk academic download is currently PAUSED** — that page shows "Temporarily
  unavailable" against all academic downloads while DrugBank updates its distribution system.
- Files ⚠️: complete database XML ~204 MB; structure SDFs 11–13.9 MB; protein identifier CSVs;
  FASTA; external-resource link CSVs. A free account is required.
- Licensing ⚠️: most datasets **CC BY-NC 4.0**; open-data resources described as **CC0** —
  which **contradicts** the "CC BY-NC open data subset" framing in the brief. ❌ The live
  licence page could not be reached, so **which subset is CC0 vs CC BY-NC is unresolved.**
  [Wikipedia](https://en.wikipedia.org/wiki/DrugBank): *"use and re-distribution of content
  from DrugBank Online or the underlying DrugBank Data, in whole or part, and for any purpose
  requires a license."*
- **Commercial pricing: ❌ none published** (`/pricing` returns 403; quote-based).
- Counts ⚠️: Wikipedia reports DrugBank 6.0 (2024) as **11,891 drug entries**, with **4,030
  approved small-molecule** and **1,601 approved biotech** drugs. ⚠️ Note the oddity — the
  current release is numbered **5.1.22** while Wikipedia calls **6.0** the latest. **Cite
  neither as authoritative for 2026.**

> **For a commercial product, DrugBank is not worth the licensing friction.** ChEMBL's
> `max_phase=4` subset covers the same ground under CC BY-SA.

#### PubChem — the cleanest licensing of the lot

✅ `CID-SMILES.gz` = **1.39 GiB**, Last-Modified **2026-09-27**:

```bash
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz   # 1.39 GiB ✅
```

Other `Extras/` ⚠️: `CID-InChI-Key.gz` 6.9 GB, `CID-IUPAC.gz` 1.7 GB, `CID-Title.gz` 1.8 GB,
`CID-Patent.gz` 6.1 GB; each has an `.md5`. ❌ Current compound count not retrievable (the
statistics page renders nothing to a fetcher); best live proxy is
`hheiden/PubChem-124M-...`'s ~124 M as of Jan 2026.

**License:** NCBI *"places no restrictions on the use or distribution of the data"*
([policies](https://www.ncbi.nlm.nih.gov/home/about/policies/)) — with the caveat from the same
page that some content is contributed or licensed by third parties and may carry copyright
which NCBI *"has no rights to transfer"*. Effectively unrestricted in practice.

---

### B.12 Licensing summary, in priority order

1. **Enamine REAL §15** — get written clarity before training on anything REAL-derived,
   including via ZINC22. **Blocks that work until resolved.**
2. **PDBbind** — do not use ≥ v2021; derivatives of v2020 cannot be redistributed. Use
   HiQBind (MIT) / PLINDER (Apache 2.0) instead.
3. **ChEMBL CC BY-SA propagates to derived *databases*** — as does Papyrus (CC BY-SA 4.0) and
   BindingDB's ChEMBL half. Much lower risk if we ship only weights.
4. **BindingNet's licence is stated three different ways** (CC BY-NC / CC BY / MIT). Pin it
   down or skip it.
5. **TDC's CC BY 4.0 is TDC's own re-license**; upstream terms (incl. AstraZeneca-via-ChEMBL =
   CC BY-SA) unverified.
6. **DrugBank** — skip.
7. **PLINDER's tooling is GPL-2.0** ✅, not Apache 2.0. Fine for research, a real decision for
   a shipped product.
8. **Cleanest for commercial use:** PubChem (effectively unrestricted),
   `hheiden/PubChem-124M-...` (**CC0-1.0**), `maykcaldas/smiles-transformers` (**MIT**, with
   descriptors precomputed), **HiQBind** (code MIT, data CC BY 4.0),
   **Exscientia low-sim PDBbind** (CC BY 4.0), `ibm-research/otter_*` (**MIT**),
   `HauserGroup/ChEMBL36-SELFIES` (**CC BY 4.0**), and the
   `scikit-fingerprints/MoleculeACE_*` sets (**CC BY-SA 3.0**, correctly inherited).

---

## C. Pretrained models we could fine-tune or distil from

All Hugging Face IDs below were checked live against `https://huggingface.co/api/models/<id>`
on 2026-10-04 (exact `safetensors.total` parameter counts and `cardData.license` where
published). ✅ = verified; ⚠️ = reported but not independently confirmed; ❌ = does not exist
or could not be verified.

### C.0 Five findings that reshape the plan

1. **The L4 is not a small A100 — it is bandwidth-starved.**
   [NVIDIA's spec page](https://www.nvidia.com/en-us/data-center/l4/): **24 GB, 300 GB/s,
   30.3 TFLOPS FP32, 242 TFLOPS BF16 *with sparsity* (≈121 dense), 72 W**. That 300 GB/s is
   **~1/5 of an A100's 1,555 GB/s**, so large-model inference is memory-bandwidth-bound, not
   FLOP-bound. Colab also reports only **~22.5 GiB usable** — **budget 21 GB, not 24.**
2. **"77M", "10M", "1.1B" in molecular model names are *corpus* sizes, not parameter
   counts.** ✅ `DeepChem/ChemBERTa-77M-MLM` is **~3.4 M parameters** (hidden 384, 3 layers,
   vocab 600). ✅ `DeepChem/MoLFormer-c3-1.1B` is **46,805,760 parameters**. Sizing a GPU
   budget off these names overestimates by 2–3 orders of magnitude. **Nothing in the
   molecular half of this brief is VRAM-constrained.**
3. **Boltz-2 is the directly relevant affinity model, it is MIT, and its affinity head is
   fine-tunable on one L4.** See §C.3 and §C.4 — this is the single best fit for our
   constraint in the whole landscape.
4. **Three licences would block a commercial product.** **Ankh (all variants) ✅
   `cc-by-nc-sa-4.0`**, ProGen3 ⚠️ `cc-by-nc-sa-4.0`, MolE ⚠️ CC-BY-NC-4.0, plus AlphaFold 3's
   weights terms. A separate, subtler class is *unlicensed-on-HF but widely used* — every
   ProtT5/ProtBert repo, every `seyonec` ChemBERTa.
5. **No SMILES transformer appears in the top 10 of any TDC ADMET leaderboard.** The winners
   are ECFP/descriptor + gradient boosting and two ~10 M-parameter GNN foundation models
   (**MiniMol**, **CheMeleon**). ChemBERTa, MoLFormer, Uni-Mol, SMI-TED and ChemFM are absent
   from every board checked.

**VRAM arithmetic used throughout.** Inference = params × 2 (bf16) or × 4 (fp32). Full
fine-tune with AdamW = **16 bytes/param** (bf16 compute 2 + fp32 master 4 + `exp_avg` 4 +
`exp_avg_sq` 4 + grad 2). 8-bit Adam ≈ 6 B/param. LoRA ≈ frozen weights only. Add ~0.6–0.9 GB
CUDA context. **Parameter counts are measured; activation figures are formula-derived
estimates (±40%, ±60% for pair-representation models).**

---

### C.1 Molecular models

#### ChemBERTa — the naming is a trap

| HF ID | Params | Downloads | Licence |
|---|---|---|---|
| `DeepChem/ChemBERTa-77M-MLM` | ✅ **~3.43 M** | 60,534 | ❌ **none declared** |
| `DeepChem/ChemBERTa-77M-MTR` | ~3.5 M (+199 heads) | **166,087** | ❌ none |
| `DeepChem/ChemBERTa-10M-MLM` / `-MTR` | ~3.4 / ~3.5 M | 3,884 / 11,615 | ❌ none |
| `DeepChem/ChemBERTa-5M-MLM` / `-MTR` | ~3.4 / ~3.5 M | 519 / 750 | ❌ none |
| **`DeepChem/ChemBERTa-100M-MLM`** | ✅ **92,136,436** | 25,613 | ✅ **MIT** ⭐ |
| `DeepChem/ChemBERTa-100M-MTR` | — | — | ❌ **does not exist** |
| `seyonec/ChemBERTa-zinc-base-v1` | ✅ ~44.7 M | 140,823 | ❌ none |

The 5M/10M/77M family all share `hidden 384 / 3 layers / vocab 600` — **a 3.4 M-parameter
3-layer model with a 600-token vocabulary is not a foundation model in any useful sense.**
`ChemBERTa-100M-MLM` is the **only** genuinely RoBERTa-base-sized ChemBERTa (`hidden 768 /
12 layers / vocab 7924`, ZINC20-100M) **and the only one with a declared licence**.

MTR = multi-task regression on **199 computed RDKit descriptors** — supervised pretraining on
free labels, which transfers better to property prediction than MLM
([arXiv:2209.01712](https://arxiv.org/abs/2209.01712)). The 166k vs 60k download split says
practitioners agree. **MTR is also the intellectual ancestor of CheMeleon**, which does the
same trick with 1,613 Mordred descriptors on a D-MPNN — and wins.

**L4 fit:** 3.4 M variants ~0.8 GB inference / ~1.0–1.5 GB full FT ✅ (batch 1024 feasible);
100M-MLM ~1.2 GB / ~4–5 GB ✅. **VRAM is a non-issue; you are data-bound.**

#### MoLFormer

- ✅ **`ibm-research/MoLFormer-XL-both-10pct`** — **46,805,760 params**, 237,560 downloads,
  **Apache-2.0**. Config `hidden 768 / 12 layers / intermediate 768 / vocab 2362`; the
  unusually small FFN is why 12×768 layers come to only 47 M. Corpus: 1.1 B molecules from
  PubChem + ZINC ([arXiv:2106.09553](https://arxiv.org/abs/2106.09553), *Nature Machine
  Intelligence*) — but **`both-10pct` is 10% of both ≈ 110 M molecules**; IBM never released
  the 100% checkpoint.
- ⭐ **The find nobody has noticed:** ✅ **`DeepChem/MoLFormer-c3-1.1B`** — **46,805,760
  params (identical size), MIT, only 2,467 downloads.** It is DeepChem's re-pretrain of the
  same architecture on the **full 1.1 B corpus** that IBM never shipped. Strictly more
  pretraining data, same size, cleaner licence. ⚠️ No benchmark numbers on the card, so the
  gain is unmeasured — worth a head-to-head on our own held-out set.

**Two mandatory flags:** `trust_remote_code=True` (vendored `modeling_molformer.py`) and
**`deterministic_eval=True`**, or the linear-attention path is nondeterministic and embeddings
will not reproduce run to run. **L4: ~0.9 GB inference / ~2.5–3.5 GB full FT @64 ✅**, batch
256 reachable; linear attention removes the `batch × heads × seq² × layers` term entirely.
⚠️ The vendored implementation is plain PyTorch, not fused — **test 10 steps under
`torch.autocast(bf16)` before a 24 h run.**

#### Uni-Mol / Uni-Mol2 — weights exist, but skip them here

✅ **`dptech/Uni-Mol2`** (**MIT**): all five checkpoints present — `84M` 336.6 MB, `164M`
659.4 MB, `310M` 1.24 GB, `570M` 2.28 GB, `1.1B` **4.52 GB**. ✅ **`dptech/Uni-Mol-Models`**
(**MIT**): `mol_pre_{all_h,no_h}_220816.pt` 190.5 MB each ≈ **~47 M params** — **pick `no_h`**
unless conformers carry explicit hydrogens. GitHub
[deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol), **MIT**.

| Model | Params | Inference | Full FT @32–64 |
|---|---|---|---|
| Uni-Mol original | ~47 M | ~1 GB ✅ | ~4–7 GB ✅ |
| Uni-Mol2 84M / 164M | 84 / 164 M | ~1.2 / ~1.5 GB ✅ | ~6–8 / ~8–10 GB ✅ |
| Uni-Mol2 310M | 310 M | ~2.0 GB ✅ | ~12–16 GB ⚠️ fits @32 |
| Uni-Mol2 570M | 570 M | ~2.9 GB ✅ | ~18–22 GB ❌ |
| Uni-Mol2 1.1B | 1.1 B | 2.3 GB bf16 ✅ | ❌ Adam states alone = **17.6 GB** |

> **The 3D conformer tax is the real reason to skip it.** Uni-Mol consumes 3D conformers, not
> SMILES. RDKit ETKDG + MMFF94 is ~0.05–1 s per conformer per core; the pipeline uses ~10
> conformers/molecule; Colab gives 2–8 vCPUs. **100 k molecules × 10 conformers × 0.2 s ÷ 4
> cores ≈ 14 h wall clock** — most of the 24 h window, on CPU, with the L4 idle and not one
> gradient step taken. And Uni-Mol2's headline gains are on **QM9 and COMPAS-1D**
> ([arXiv:2406.14969](https://arxiv.org/abs/2406.14969)), quantum-chemistry sets where
> geometry *is* the signal. For ADMET and affinity ranking, a gas-phase minimum of a flexible
> ligand is a weak and expensive proxy for the bioactive conformer.

#### GraphMVP, MolCLR, SMILES-BERT, SELFIES — mostly no weights

| Thing | HF weights? |
|---|---|
| **GraphMVP** | ❌ **none** — search empty. [GitHub only](https://github.com/chao1224/GraphMVP) |
| **MolCLR** | ❌ **none** — search empty. [GitHub only](https://github.com/yuyangw/MolCLR) |
| **SMILES-BERT** (2019) | ❌ no official weights. Community: `unikei/bert-base-smiles` (1,423 dl), `JuIm/SMILES_BERT` (218 dl), and a 2025 set `fabikru/*modern_smiles_bert_{4.5M,15.2M,111.0M}*` ⚠️ configs unverified |
| **SELFIES** | ✅ `selfies` pip **v2.2.0, 2025-01-15, Apache-2.0** ([repo](https://github.com/aspuru-guzik-group/selfies)). But every HF SELFIES tokenizer has **0 downloads** |

> **Skip SELFIES.** It solves *generative validity* — every SELFIES string decodes to a valid
> molecule. That is worthless for triage, where we score molecules that already exist. The
> download counts show the field agrees for discriminative tasks.

#### MolE, MolGPS and the paper-only tier

| Name | Finding |
|---|---|
| **MolE** (Recursion × Novo Nordisk) | ❌ no HF weights. [Code](https://github.com/recursionpharma/mole_public) under ⚠️ **CC-BY-NC-4.0 — non-commercial.** Published [*Nat. Commun.* 10.1038/s41467-024-53751-y](https://doi.org/10.1038/s41467-024-53751-y). **The NC licence alone rules it out.** |
| **MolGPS** (Valence/Recursion/Graphcore) | ❌ no HF weights, search empty. 3 B params, SOTA on 26/38 tasks, [arXiv:2404.11568](https://arxiv.org/abs/2404.11568), NeurIPS 2024 — **never released.** MiniMol is the public 300×-smaller sibling from the same group, and it **beats MolGPS on several TDC boards.** |
| **Transformer-M** | ❌ no HF weights |
| **"MolmirNet"** | ❌ **does not exist** — empty search, no paper, no repo |
| **"OpenMol"** | ❌ nothing identifiable; the real org is **OpenADMET** (below) |
| **GeminiMol** | ⚠️ `AlphaMWang/GeminiMol` exists, **0 downloads** |

> **A structural fact worth internalising:** searching Hugging Face for 2025–2026 molecular
> foundation models **systematically misses them**. `?search=molecular+foundation` returns
> empty, and huggingface.co/papers showed **zero** molecular-FM papers on 2026-10-04.
> Chemistry FMs publish on arXiv q-bio / JCIM / *Nat. Commun.*, and their weights land on
> **Zenodo or in a pip package**, not the Hub.

#### Real, downloadable and relevant

| Model | Location | Params | Licence |
|---|---|---|---|
| **CheMeleon** ⭐ | [Zenodo 10.5281/zenodo.15426600](https://doi.org/10.5281/ZENODO.15426600) + built into Chemprop ≥2.2.0; [repo](https://github.com/JacksonBurns/chemeleon) | **12.9 M** | **MIT** / CC0 |
| **MiniMol** ⭐ | `pip install minimol`; [repo](https://github.com/graphcore-research/minimol) | **10 M**, 512-d embeddings | **MIT** |
| **ChemFM-1B** | ✅ `ChemFM/ChemFM-1B` | ✅ **970,287,104** | ✅ **MIT** |
| **ChemFM-3B** | `ChemFM/ChemFM-3B` | ✅ **3,004,357,632** | **MIT** |
| **SMI-TED** | `ibm-research/materials.smi-ted` | ~289 M | **Apache-2.0** |
| **ChemDFM** | `OpenDFM/ChemDFM-v2.0-14B`, `-v1.5-8B` | 8–14 B | ⚠️ — |
| **nach0** | `insilicomedicine/nach0_base` / `_large` | ⚠️ ~250 M / ~800 M | ⚠️ — |

❌ `TheLuoFengLab/ChemFM-1B` does not exist — the real home is **`ChemFM/ChemFM-1B`**.

**L4 fit:** CheMeleon / MiniMol (10–13 M) ~0.8 GB / ~1–2 GB ✅ trivially, batch 512+. SMI-TED
(289 M) ~2 GB / ~6–7 GB @32 ✅ — ⚠️ `model_type: SMI-TED` is **not a stock `transformers`
model**; needs IBM's `smi_ted_light` loader. **ChemFM-1B: full FT ❌** (Adam states 15.5 GB);
bf16 + 8-bit Adam + checkpointing ≈ 9 GB @32 ✅; **LoRA r=16–32 ≈ 4–5 GB @64 ✅**.
**ChemFM-3B: full FT impossible** (48 GB of states); LoRA bf16 ≈ 10–13 GB @16–32 ✅ tight;
QLoRA NF4 ≈ 6–8 GB ✅. ChemDFM 14B bf16 = 28 GB ❌ — and these are chat LLMs; **a generated
string is not a calibrated scalar you can rank on.**

#### ⭐ OpenADMET — someone has already done much of our ADMET job

An **Open Molecular Software Foundation** project —
[OpenADMET/openadmet-models](https://github.com/OpenADMET/openadmet-models) (MIT), docs at
docs.openadmet.org. **All models Apache-2.0.** These are **CheMeleon fine-tuned on real ADMET
endpoints, trained on ChEMBL, with ensemble uncertainty:**

| HF ID | Endpoints |
|---|---|
| `openadmet/permeability-logd-ppb-chemeleon-baseline` | Caco-2 A→B and B→A, LogD, mouse PPB, human PPB (multitask) |
| `openadmet/cyp1a2-cyp2d6-cyp3a4-cyp2c9-chemeleon-v1` | 4× CYP inhibition |
| `openadmet/herg-chemeleon-baseline` | **hERG** |
| `openadmet/microsomal-clearance-chemeleon-v1` | microsomal clearance |
| `openadmet/pxr-chemeleon-{baseline,v1}` | PXR |
| `openadmet/{mpbp,logd}-chemprop_foundation_minimol-v1` | MPBP, LogD — **MiniMol** backbone |
| `openadmet/{mpbp,logd}-chemprop_foundation_surrogate-v1` | MPBP, LogD — surrogate backbone |

Note they shipped **both CheMeleon *and* MiniMol backbones for the same two endpoints** — a
free, pre-run A/B for exactly the backbone choice we face. **~0.8 GB inference ✅, zero
training.** This is a day-one calibrated baseline, and it covers hERG and all four CYPs from
section A/B.

#### Toolkits to use instead of a foundation model

**Chemprop v2** — ✅ PyPI **2.3.1, released 2026-08-04**, MIT. ⚠️ **The GitHub README still
says "v2.0.0" and is stale — trust PyPI.** D-MPNN; v2 is a Lightning rewrite.
[Docs](https://chemprop.readthedocs.io/en/main/), citation
[10.1021/acs.jcim.5c02332](https://doi.org/10.1021/acs.jcim.5c02332).
**`--from-foundation CheMeleon` requires `chemprop>=2.2.0` and is the single highest-value
flag for this project.** ⚠️ Training time is an *estimate*: a D-MPNN on a TDC-sized set
(500–7,000 molecules), 50 epochs → **seconds to a few minutes**; on 100 k ChEMBL molecules →
**~1–3 min/epoch, so 1–2.5 h for 50 epochs**, ±3×. The bottleneck is **RDKit featurisation on
2–8 vCPUs, not the GPU** — which means within 24 h we can afford **dozens of hyperparameter
trials and 5-fold × 5-seed ensembles**, worth more than a bigger model.

**scikit-fingerprints** — ✅ PyPI **2.1.0, released 2026-08-27, MIT**
([repo](https://github.com/scikit-fingerprints/scikit-fingerprints)). **30+ fingerprints**
(ECFP, Avalon, MACCS, Mordred, PubChem), **30+ filters** (Lipinski Ro5, PAINS, REOS), **14
similarity metrics**, **11 applicability-domain checks**, and **built-in loaders for
MoleculeNet, TDC, MoleculeACE and LRGB**, all scikit-learn `Pipeline`-native. **The
applicability-domain checks and the Ro5/PAINS/REOS filters are the parts no foundation model
gives you** — and a ranking you cannot bound by domain of applicability is a ranking you
cannot act on.

**The ECFP + GBM evidence — read this before choosing a backbone.**

- **[arXiv:2310.00174](https://arxiv.org/abs/2310.00174)**, *"ADMET property prediction
  through combinations of molecular fingerprints"*, **Notwell & Wood**. Opens: *"While
  investigating methods to predict small molecule potencies, we found random forests or
  support vector machines paired with extended-connectivity fingerprints (ECFP) consistently
  outperformed recently developed methods."* Best config: **CatBoost + {ECFP, Avalon, ErG} +
  200 molecular properties**, validated across **all 22 TDC ADMET benchmarks**. **This is the
  "MapLight" entry that tops the leaderboards in §B.8 — same author.**
- ⭐ **[arXiv:2011.12203](https://arxiv.org/abs/2011.12203)**, *"Making GNNs Worth It for
  Low-Data Molecular ML"* — **the fingerprint baseline beats Chemprop below ~1,024
  datapoints**, and Chemprop only becomes competitive above it. **Every TDC ADMET regression
  set except AqSol is in or near that regime** (Caco-2 906, hERG 648, DILI 475, HIA 578).
  **This single number is the most decision-relevant fact in this whole document.**

**ADMET-AI** — [*Bioinformatics* 40(7) btae416](https://academic.oup.com/bioinformatics/article/40/7/btae416/7698030),
Swanson et al. 2024, CC-BY. Chemprop-RDKit over 41 TDC datasets. Cited speed: 1,000 molecules
in ~40 s on their server; **1 M molecules in 3.1 h locally with 32 CPU cores + GPU (≈90
mol/s)**. ⚠️ **Scale that down hard for Colab:** 90 mol/s assumes 32 cores; we get 2–8, and
RDKit featurisation is the serial bottleneck. **Expect ~10–25 mol/s → 1 M molecules ≈ 11–28 h,
a whole session.** Chunk and cache.

---

### C.2 Protein language models

> **The engineering point that dominates every number below:** for triage we almost never need
> the PLM in the training loop. The protein set is small (section A's shortlist is ~20
> targets); the ligand set is large. So **run the PLM once, inference-only, bf16, dump
> embeddings to disk, and train the head on cached embeddings plus a ligand tower.** This
> decouples model size from the 24 GB *training* budget entirely — ESMC-6B becomes viable at
> 12.7 GB embed-once, whereas fine-tuning it is impossible.

> ⚠️ **THE SINGLE BIGGEST TRAP.** With `attn_implementation="eager"`, HF materialises the
> `heads × L²` attention matrix and keeps it for backward. At ESM-2 650M, L=4096:
> 20 × 4096² × 4 B × 33 layers ≈ **44 GB — instant OOM**. **Always pass
> `attn_implementation="sdpa"` or `"flash_attention_2"`.** This one flag is the difference
> between L≈1024 and L≈8000, and it explains most "ESM OOMs on my 24 GB card" reports.

Also: `max_position_embeddings: 1026` in every ESM-2 config is vestigial — ESM-2 uses
`position_embedding_type: "rotary"`, so 1024 is the *training crop*, not an architectural cap.
Quality degrades past ~1024–2048.

#### ESM-2 — all six sizes ✅ verified, all ✅ MIT

| HF ID | Params | Layers / d_model | Downloads | Disk fp32 |
|---|---|---|---|---|
| `facebook/esm2_t6_8M_UR50D` | ✅ **7,841,868** | 6 / 320 | 559,717 | 0.03 GB |
| `facebook/esm2_t12_35M_UR50D` | ✅ **33,995,044** | 12 / 480 | 220,482 | 0.14 GB |
| `facebook/esm2_t30_150M_UR50D` | ✅ **148,798,300** | 30 / 640 | 241,473 | 0.60 GB |
| **`facebook/esm2_t33_650M_UR50D`** | ✅ **652,358,616** | 33 / 1280 | **3,374,182** | 2.61 GB |
| `facebook/esm2_t36_3B_UR50D` | ⚠️ ~2.84 B (derived) | 36 / 2560 | 217,857 | 11.37 GB |
| `facebook/esm2_t48_15B_UR50D` | ⚠️ ~15.14 B (derived) | 48 / 5120 | 1,842 | 60.54 GB |

NVIDIA mirrors all six in BioNeMo-compatible format, also MIT
(`nvidia/esm2_t33_650M_UR50D` = 651,042,593 params) — **relevant since Talindrew already runs
BioNeMo NIM.**

| Size | Inf. bf16 | Full FT (weights+Adam) | + activations @L=1024,B=1 | Verdict | LoRA bf16 |
|---|---|---|---|---|---|
| 8M | 0.6 GB | 0.13 GB | ~0.5 GB | ✅ trivial | 0.6 GB |
| 35M | 0.7 GB | 0.54 GB | ~1.0 GB | ✅ trivial | 0.7 GB |
| 150M | 1.1 GB | 2.38 GB | ~3.5 GB | ✅ easy | 1.1 GB |
| **650M** | **2.2 GB** | **10.44 GB** | **~13.7 GB** | ✅ **the sweet spot** | **1.9 GB** |
| **3B** | **6.7 GB** | 45.4 GB | 50+ GB | ❌ no configuration | **~11.5 GB ✅** |
| 15B | 31 GB ❌ | 242 GB | — | ❌ | 30.3 GB ❌ |

**ESM-2 650M full FT, SDPA mandatory:** L=512 B=8 ≈ 19 GB ✅ | **L=1024 B=2 ≈ 16 GB ✅
recommended default** | L=1024 B=4 ≈ 21 GB ⚠️ | L=2048 B=1 ≈ 16 GB ✅ | L=4096 B=1 ≈ 20 GB ⚠️
(✅ checkpointed) | L=8192 B=1 ❌ (✅ ~14 GB checkpointed).

**ESM-2 3B LoRA is a real and under-used option:** 5.7 GB bf16 weights + ~5 GB activations
@L=1024 B=1 ≈ **11.5 GB comfortable**.

**Max sequence length on 24 GB:** VRAM is almost never binding for *inference* — 650M with
SDPA is memory-linear and would go ⚠️ ≫30,000; with **eager** it caps at ~13,500. For full FT
it is ~4,096 (~8,192 checkpointed). **Practical answer: cap at 1024.** For a 1,154-aa JAK1,
**domain-crop to the kinase domain** rather than extending context — which is also the
biologically correct thing to do.

⚠️ **Throughput sanity check (order-of-magnitude):** train FLOPs ≈ 6·N·tokens, L4 ≈ 121
TFLOPS bf16 dense at ~30% MFU → 25–40 TFLOPS effective. 650M full FT ≈ **7,700 tok/s → ~660 M
tokens/24 h** ≈ 1.3 M sequences of 512 aa. **The 24 h wall is not the constraint for a 650M
protein tower.**

#### ESM-3 — ✅ now MIT, and the org moved

✅ **`biohub/esm3-sm-open-v1`** — the old `EvolutionaryScale/esm3-sm-open-v1` **307-redirects
here** (I confirmed the redirect myself). `?author=EvolutionaryScale` returns **zero models**;
**EvolutionaryScale is now Chan Zuckerberg Biohub.** 2,629 downloads, 335 likes, ungated.
⚠️ Params **1.4 B** (derived: `esm3_sm_open_v1.pth` = 2,803,646,068 B ÷ 2 bf16). The repo also
ships structure encoder (62 MB), structure decoder (1.24 GB) and function decoder (1.30 GB) —
**5.40 GB total**, because ESM3 is multi-track.

**Licence — the direct answer to the brief: no, it is no longer non-commercial.**
[`LICENSE.md`](https://github.com/Biohub/esm/blob/main/LICENSE.md) is verbatim MIT,
*"Copyright 2026 Chan Zuckerberg Biohub, Inc."* The Cambrian Open License / ESM3 Community
License has been **removed from the repo**.

> ⚠️ **Two caveats I verified independently and would act on.** (a) ✅ `biohub/esm3-sm-open-v1`'s
> HF `cardData` has **no `license` field at all** — I confirmed this; the MIT grant reaches the
> *weights* only via prose linking to a file whose text says "the Software". (b) GitHub's API
> reports the repo licence as **`NOASSERTION`**. The grant looks genuine, but **archive today's
> `LICENSE.md` and model card as evidence of what we relied on** before shipping ESM3-derived
> weights commercially — the 2024–25 release genuinely *was* non-commercial.

**VRAM:** inference bf16 **2.8 GB** (LM) / ~6 GB with decoders ✅. **Full FT: 1.40 B × 16 =
22.4 GB ❌.** LoRA ≈ 6–7 GB ✅. **Relevance to triage: low** — ESM3 is a *generative*
multi-track designer; for affinity/ADMET regression you want a representation model or a
co-folder.

#### ESM C / ESM Cambrian — ✅ all MIT, and the 6B is **not** API-only

| HF ID | Params (exact ✅) | Layers / d / ff | Disk fp32 | Downloads | Licence |
|---|---|---|---|---|---|
| `biohub/ESMC-300M` | **332,997,184** | 30 / 960 / 2560 | 1.33 GB | 29,972 | ✅ MIT |
| `biohub/ESMC-600M` | **575,036,992** | 36 / 1152 / 3072 | 2.30 GB | 6,691 | ✅ MIT |
| `biohub/ESMC-6B` | ✅ **6,352,005,184** | 80 / 2560 / 6912 | 25.41 GB | **1,173,989** | ✅ `['mit','other']` |

✅ I re-verified ESMC-6B's parameter count and licence directly. **The `"other"` is a red
herring** — `license_link` points at a dependency table
([`THIRD_PARTY_NOTICE.md`](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md):
flash-attn BSD, PyTorch BSD, xformers BSD, einops MIT, lightning Apache-2.0). **ESM C is
cleanly MIT.** The 6B's **25.41 GB of weights are public and ungated with 1.17 M downloads**
— it is not API-only.

**Context: `max_position_embeddings: 2048`**, and the card documents the schedule — 512 for
the first 1 M steps, **2048 for the final 500 k**. So 2048 is genuinely trained, better than
ESM-2's 1024.

| Model | Inf. bf16 | Full FT | LoRA bf16 |
|---|---|---|---|
| ESMC-300M | **1.3 GB** ✅ | **5.3 GB ✅** (~7 GB w/ act) | 1.3 GB ✅ |
| ESMC-600M | **1.9 GB** ✅ | **9.2 GB ✅** (~12 GB @B=1) | 1.9 GB ✅ |
| **ESMC-6B** | **12.7 GB ✅** (~14–15 GB w/ act) | 102 GB ❌ | ~19 GB ⚠️ very tight |

**Recommendation: ESMC-600M to fine-tune, ESMC-6B to embed-once.** ESMC-6B bf16 inference fits
on an L4 — the strongest embed-once option under MIT. Load with `dtype=torch.bfloat16,
attn_implementation="flash_attention_2"`. Also available: ~45 sparse-autoencoder repos
(`biohub/ESMC-SAE-Overview`) giving interpretable features — potentially valuable for
*explaining* a triage score.

#### ESMFold — ✅ verified, and the brief's param split is inverted

✅ **`facebook/esmfold_v1`** — **MIT**, **1,713,682 downloads**, ungated. **It is ESM-2 3B as
the LM plus a ~690 M folding trunk ≈ 3.53 B total**, not "690 M LM + trunk". The arithmetic
closes exactly: `pytorch_model.bin` = 8.44 GB; config is verbatim `esm2_t36_3B`
(`hidden 2560 / 36 layers / ff 10240`); `fp16_esm` defaults true so the LM is 2.84 B × 2 =
5.67 GB; remainder 8.44 − 5.67 = 2.77 GB which at fp32 is **692 M** = the trunk. Trunk:
`num_blocks 48, sequence_state_dim 1024, pairwise_state_dim 128, max_recycles 4`, and
**`chunk_size: null` — chunking is OFF by default, which is what OOMs people.**

**The `chunk_size` trick.** `model.trunk.set_chunk_size(n)` chunks axial/triangle attention
over the pair representation, taking the pair-attention peak from **O(L²) → O(L)**. The
[upstream README](https://github.com/facebookresearch/esm) recommends **128, 64, 32**, and
documents `--cpu-offload` for longer sequences.

| Setting | Max L | Source |
|---|---|---|
| `chunk_size=None` (default) | **~400 aa** | ⚠️ community reports |
| `chunk_size=64` | **~700–900 aa** | ⚠️ community reports |
| `chunk_size=32` + `--cpu-offload` | ~1000–1200 aa | ⚠️ extrapolation |
| **honest planning number** | **~1024 aa** | — |

⚠️ All third-party — **Meta publishes no per-GPU table.** Corroborating reports: ~400 aa
unchunked on 24 GB cards, 16 GB stated minimum; memory errors above L≈1400 even at
`chunk_size=128`; on an 80 GB H100, 1792 residues reserves 76.8 GiB and 2048 OOMs.
**The weights are never the problem; the L² pair representation across 48 trunk blocks is.**
**Fine-tuning ESMFold on 24 GB: ❌** (3.53 B × 16 = 56 GB before activations).

#### ESMFold2 — ✅ new, MIT, and directly on-point

- ✅ **`biohub/ESMFold2`** — **6,579,755,843 params** (exact), 26.32 GB, **MIT**, **302,719
  downloads**, last modified 2026-09-16. Built on a **frozen ESMC-6B**. Card: *"Unlike
  ESMFold, ESMFold2 is able to predict structures for all biomolecules, including small
  molecules, DNA, RNA, and modified amino acids"*, with pLDDT/pAE/pTM/**iPTM** outputs. Paper:
  [biorxiv 2026.06.03.729735](https://www.biorxiv.org/content/10.64898/2026.06.03.729735).
- ✅ `biohub/ESMFold2-Fast` — **6,533,751,875 params**, single-sequence inference-optimised.

**Why it matters here:** it co-folds protein + small-molecule ligand and emits **iPTM** — a
structure-based affinity *prior* we would not have to build. **VRAM: ~13.2 GB bf16 weights ✅,
plus an unknown pair-representation cost — ⚠️ measure before relying on it. Fine-tuning ❌**
(6.58 B × 16 = 105 GB). Note from §C.4 that **iPTM alone is a *poor* affinity predictor**
(Boltz-2's own ipTM scores R = 0.04), so treat this as a feature, not a score.

#### ProtT5 / ProstT5 / Ankh / SaProt

| Model | Params | Inf. fp16 | Full FT | Licence |
|---|---|---|---|---|
| `Rostlab/prot_t5_xl_half_uniref50-enc` | ⚠️ ~1.21 B (enc only) | **2.4 GB ✅** | 19.4 GB ❌ | ❌ **none on HF** |
| `Rostlab/prot_t5_xl_uniref50` | ⚠️ ~2.82 B | 5.6 GB ✅ | 45 GB ❌ | ❌ none |
| `Rostlab/ProstT5` / `ProstT5_fp16` | ⚠️ ~2.82 B | 5.6 GB ✅ | 45 GB ❌ | ✅ **MIT** |
| `Rostlab/prot_bert` | ⚠️ ~420 M | 0.9 GB bf16 ✅ | **6.7 GB ✅** | ❌ none |
| `ElnaggarLab/ankh-base` | ⚠️ ~738 M | 1.5 GB ✅ | **11.8 GB ✅** | ❌ ✅**`cc-by-nc-sa-4.0`** |
| `ElnaggarLab/ankh-large` | ⚠️ ~1.88 B | 3.8 GB ✅ | 30 GB ❌ | ❌ `cc-by-nc-sa-4.0` |
| `westlake-repl/SaProt_650M_AF2` | ~650 M | 2.0 GB ✅ | **13 GB ✅** | ✅ **MIT** |
| `chandar-lab/AMPLIFY_350M` | ✅ **354,008,667** | 0.9 GB ✅ | **~7 GB ✅** | ✅ **MIT** ⭐ |

> ⚠️ **LICENCE FINDING — act on this.** `cardData.license` is **absent on every ProtT5 and
> ProtBert repo**, and grepping their READMEs for `licen|academic|commercial` returns zero
> hits. The upstream repo [`agemagician/ProtTrans`](https://github.com/agemagician/ProtTrans)
> **is MIT**, so the intent is permissive — but **the artefact we would download carries no
> grant.** Prefer ESM-2 / ESM C, which are explicitly MIT on the artefact itself.

> ⚠️ **DO NOT BUILD ON ANKH.** ✅ I verified `ElnaggarLab/ankh-base` is **`cc-by-nc-sa-4.0`** —
> **non-commercial *and* share-alike.** NC blocks a paid product (Talindrew has Free/Pro/Max
> plans) and SA would try to pull our fine-tuned derivative under the same terms. Every one of
> the 9 `ElnaggarLab` models carries it. **This is the clearest licensing landmine in section
> C.**

**ProstT5's non-obvious use:** it can **predict 3Di tokens from sequence alone**
(`<AA2fold>` / `<fold2AA>`), trained on 17 M AlphaFold DB structures with
[Foldseek](https://github.com/steineggerlab/foldseek) 3Di tokens. That **removes the
Foldseek/structure dependency from SaProt** — structure-aware tokens for a target with no PDB
entry, far cheaper than running ESMFold.

> ⚠️ **SaProt's catch, straight from its card:** *"SaProt requires structural (SA token) input
> for optimal performance. AA-sequence-only mode works but must be finetuned — **frozen
> embeddings work only for SA, not AA sequences!**"* So **SaProt's embed-once advantage
> evaporates unless you supply 3Di tokens.** For section A's ~20-target list that is
> tractable, but it is a pipeline dependency, not a drop-in.

**⭐ AMPLIFY_350M is the best full-fine-tuning value on this list.** ✅ 354,008,667 params,
MIT. Modern architecture (SwiGLU, RMSNorm, RoPE, no biases, pre-activation LN;
`hidden 960 / 32 layers / ff 3840`). Non-`_base` variants have
`max_position_embeddings: 2048`. Trained on UR100P (UniRef100 + **OAS antibody data** +
SCOP) — the OAS part matters if we ever triage biologics.
[Paper](https://www.biorxiv.org/content/10.1101/2024.09.23.614603v1). **Full FT ≈ 7 GB ✅ with
huge headroom — L=2048 B=4, or L=512 B=16, inside 24 GB.** ⚠️ A newer official home appears
to be `flair-bio/amplify-350m` (354,018,272 params, MIT) — **check which is canonical before
pinning.**

#### The 2025–2026 cohort, briefly

- **ESM++ (Synthyra)** — MIT, but it is a **repackaging of ESM C**: `ESMplusplus_small` =
  332,997,184 (= ESMC-300M), `_large` = 575,036,992 (= ESMC-600M), `_6B` = 6,352,005,184
  (= ESMC-6B). ⚠️ **Dependency trap:** the card requires *"Python 3.11-3.14, PyTorch 2.13,
  Transformers 5.13"* plus `trust_remote_code=True`. Colab's default stack won't match.
  **Since ESM C is now natively `model_type: "esmc"` with MIT weights direct from `biohub`,
  prefer `biohub/*` and skip ESM++.**
- **ProGen2** — ⚠️ **Salesforce does not host these on HF.** De-facto weights are community
  conversions on an individual account (`hugohrban/progen2-*`, bsd-3-clause). **Vendor them**
  rather than depending on a re-upload.
- **ProGen3 (Profluent)** — ⚠️ all `cc-by-nc-sa-4.0`, **non-commercial.** But
  **`Profluent-Bio/E1-600m`** is *"designed to be drop-in replacement for ESM family"*,
  commercially usable under a clickthrough licence with ⚠️ **mandatory attribution and a
  termination clause** — unlike MIT, **someone can revoke it**. Inf. bf16 1.3 GB ✅,
  full FT ≈ 13 GB ✅.
- **Evo 2** — ✅ Apache-2.0, and ⚠️ **it is a genomic/nucleotide model, not a protein model** —
  it will not give a protein representation for an affinity head. Measured: `evo2_1b_base`
  2.70 GB ✅, `evo2_7b` 13.77 GB ✅ inference only, `evo2_20b` 47.95 GB ❌, `evo2_40b`
  82.25 GB ❌.
- **Others found:** `westlake-repl/Evolla-10B-hf` (MIT, 51,632 dl; 10 B bf16 = 20 GB ⚠️ at the
  very edge), `ProTrek_650M` (MIT, tri-modal sequence/structure/text retrieval — useful for
  target annotation), `ChatterjeeLab/PepMLM-650M` (MIT, ESM-2 650M fine-tuned for peptide
  binders), `airkingbd/dplm2_650m` (Apache-2.0).
- ❌ **Claims that do not check out:** **"ESM-2 small distillations"** — `?search=distill+esm2`
  returns **zero results**; there is no established distilled-ESM-2 family. **Lobster
  (Prescient Design/Genentech)** — `?author=prescient-design` returns **zero models**; weights
  live on individual accounts with **no licence tags**. ⚠️ Treat Lobster as unlicensed and
  unusable commercially.

**Non-negotiables for the L4:** (1) `attn_implementation="sdpa"`/`"flash_attention_2"`, never
`"eager"`; (2) **bf16, not fp16** — Ada handles bf16 natively, no loss-scaling; (3)
`gradient_checkpointing_enable()` roughly halves peak for ~30% speed; (4) 8-bit Adam takes
ESM-2 650M full FT from 10.4 → **6.5 GB**; (5) **cache embeddings.**

---

### C.3 Structure / complex prediction

#### ⭐ Boltz-2 — MIT, and the affinity head is the headline

✅ **`boltz-community/boltz-1`** and ✅ **`boltz-community/boltz-2`**, both **MIT** (I verified
boltz-2's licence field directly). ❌ There is no `boltz-1x` HF repo — boltz-1x is a
code-level steering variant. GitHub [jwohlwend/boltz](https://github.com/jwohlwend/boltz) —
**MIT**, 4,232★. README verbatim: *"Our model and code are released under MIT License, and
can be freely used for both academic and commercial purposes."* ⚠️ Size **~0.5 B params**.

**The affinity module, in detail** (from the
[technical report](https://jeremywohlwend.com/assets/boltz2.pdf) §B.5): it takes Boltz-2's
`s_inputs` and the final pair representation `z_ij` **after five recycling iterations**, with
coordinates from **the top-ranked of 5 samples over 200 diffusion steps**, ranked by
protein–ligand ipTM. Core is a **PairFormer that masks out intra-protein interactions** to
focus on the interface, then **mean-pools over all pairwise interactions**, then **two MLP
heads**. It is an **ensemble of two models** (8 vs 4 PairFormer layers), both trained across
**128 A100 GPUs**.

**Two outputs, and the docs are emphatic about not confusing them:**

- **`affinity_probability_binary`** (0–1) — *"Should be used to detect binders from decoys,
  for example in a hit-discovery stage."*
- **`affinity_pred_value`** — **log₁₀(IC₅₀) in μM**. *"Should be used in ligand optimization
  stages such as hit-to-lead and lead-optimization."* Convert to kcal/mol with
  **`(6 − y) × 1.364`**. The report is careful: the value head is supervised on a *mixture* of
  Ki, Kd and IC₅₀, so *"the predicted value should be viewed as a general measure of binding
  strength that supports ranking."*

```yaml
properties:
    - affinity:
        binder: B        # a single LIGAND chain id
```

Flags: `--affinity_mw_correction` (applies a fitted molecular-weight correction
`ŷ = C₀·(y₁+y₂) + C₁·MW + C₂`), `--sampling_steps_affinity` (default 200),
`--diffusion_samples_affinity` (default 5).

**Constraints: the binder must be a single ligand chain, ≤128 heavy atoms after RDKit
H-removal, and "ligands significantly larger than 56 atoms are not recommended."** The docs
warn that with an RNA/DNA/co-factor target *"the code will not crash but the output will be
unreliable."*

**Training data** (report §4): ChEMBL v34 + BindingDB continuous values — **1.2 M binders over
2 k targets / 600 k compounds**; plus PubChem HTS (200 k binders / 1.8 M decoys), CeMM
Fragments, MIDAS Metabolites, and 1.2 M **synthetic decoys**; a structural quality filter
drops anything with **ipTM < 0.75**. **Note the overlap with our own data plan** — Boltz-2 has
already seen ChEMBL v34 and BindingDB, which matters for leakage (section D).

**Does it fit on one L4?**

- **(a) Inference: ✅ yes, to ~1000 residues.** Empirical, from the maintainers: *"We've run
  tests and are able to pass 1800AA on a 40G GPU"*
  ([#104](https://github.com/jwohlwend/boltz/issues/104)); and on a 24 GB 4090, *"the maximum
  prediction size is about 1000 residues. Using 16-bit floating point model weights and
  activations (bfloat16) … allows it to predict about 1400 residues"* with *"no reduction in
  accuracy"* ([#263](https://github.com/jwohlwend/boltz/issues/263)). **So ~1000 aa fp32,
  ~1400 aa bf16.** A low-memory fork exists at
  [tlitfin/lmi4boltz](https://github.com/tlitfin/lmi4boltz).
- **(b) Fine-tuning: ✅ yes — only the affinity heads, and that is exactly what we want.**
  ⭐ [`molecularinformatics/Boltz2_affinity`](https://github.com/molecularinformatics/Boltz2_affinity)
  (**MIT**, 64★) publishes the recipe from *"Affinity Fine-Tuning of Boltz-2: An Open
  Framework for Protein-Ligand Potency Prediction in Drug Discovery"*, Amini, Sciabola & Wang,
  2026, DOI [10.64898/2026.05.26.727958](https://doi.org/10.64898/2026.05.26.727958). The
  flow: (1) one YAML per complex with an affinity `value`; (2)
  `boltz predict --write_embeddings --recycling_steps 5 --sampling_steps 200
  --diffusion_samples 5` to **cache embeddings to disk once**; (3) `data_collector.py`
  consolidates; (4) `trainv2.py` trains. The docs are explicit: *"Freeze all structure
  prediction modules (trunk, diffusion, confidence) / Train only the affinity prediction
  heads."* Shipped config: **`devices: 1`, `precision: bf16-mixed`, `batch_size: 1`,
  `accumulate_grad_batches: 16`, `max_tokens: 512`, `max_epochs: 10`,
  `samples_per_epoch: 500`, `max_lr: 1e-4`.** Since the trunk is frozen and training runs on
  **cached embeddings**, the training step is a 4–8-layer PairFormer plus two MLPs — **well
  under 5 GB. ✅ This is the single best-fitting option in this entire document for one L4.**
  - ⚠️ Two practical warnings: `--write_embeddings` produces **large files** (Colab's *disk*
    becomes the constraint, not VRAM), and **pre-compute MSAs** — the ColabFold MSA server
    will rate-limit you into tens of minutes per prediction otherwise.
  - ⚠️ Reported gains of **0.2–0.4 Pearson R from as few as 50–200 project-specific
    compounds** are **single-sourced and UNVERIFIED** (bioRxiv returned 429/403). The
    *qualitative* claim — *"in some cases reaches performance competitive with free energy
    perturbation (FEP) methods"* — is verified from the abstract.

⚠️ **Throughput estimate on one L4 (extrapolation, not published):** the report quotes **20
GPU-seconds** per affinity prediction on A100/H100-class hardware, and issue
[#612](https://github.com/jwohlwend/boltz/issues/612) reports *"5-10 seconds"* on a 4090. An L4
is ~2–3× slower (121 vs ~165 dense bf16 TFLOPS, and **300 vs 1,008 GB/s** — the bandwidth gap
dominates). **Estimate ~30–60 s per complex → ~1,500–2,500 complexes per 24 h.** **Batch the
YAMLs into one directory** — `boltz predict <dir>` loads the network once (~20 s) instead of
per-molecule.

**Boltz-2's own stated limitations** (report §6, verbatim): it *"does not significantly
deviate from the structure prediction performance of its predecessors"* and *"still often
fails to capture large conformational changes, such as those that can be induced by
binding."* On affinity: *"If the model fails to identify the correct pocket or inaccurately
reconstructs the binding interface … downstream affinity predictions are unlikely to be
reliable… the affinity module does not explicitly handle such cofactors, including ions,
water, or multimeric binding partners."* And: *"performance varies strongly between assays."*
[Rowan's FAQ](https://www.rowansci.com/blog/boltz2-faq) adds: **~40% of predicted "hits" are
false positives**; no protein–protein support; **poor on GPCRs, transporters and ion
channels** (sparse training data).

> ❗ **That last point hits our target list directly.** Section A's shortlist includes
> **ADRB2 (GPCR), DP2 (GPCR), CysLT1 (GPCR), SGLT2 (transporter) and hERG (ion channel)** —
> precisely the classes Boltz-2 is documented to handle poorly. Plan for Boltz-2 on the
> **enzymes and kinases** (JAK1, TYK2, DPP-4, PDE4, HIV PR/RT/IN, 5-LOX) and expect the GPCRs
> and transporters to need the ligand-based model instead.

#### AF3-class models: what is actually available

| Model | Code | Weights | Weights licence | **Commercial?** | 24 GB |
|---|---|---|---|---|---|
| **AlphaFold 3** | Apache-2.0 | ✅ **direct download** | AF3 Model Parameters ToU | ❌ **non-commercial, no redistribution, derivatives inherit** | ⚠️ ~1k tokens w/ unified memory, unsupported |
| **Boltz-2 / Boltz-1** | MIT | ✅ | ✅ **MIT** | ✅ **yes, explicitly** | ✅ ~1000 aa (1400 bf16) |
| **OpenFold3** | Apache-2.0 | ✅ `OpenFold/OpenFold3` | ✅ **Apache-2.0** | ✅ **yes** — training data + code public too | ⚠️ unmeasured |
| **Chai-1** | Apache-2.0 | ✅ `chaidiscovery/chai-1` | ✅ **Apache-2.0** | ✅ **yes, "including for drug discovery"** | ⚠️ small complexes |
| **Chai-2** | — | ❌ | — | ❌ not released | — |
| **Protenix v1.0/v0.5/Mini** | Apache-2.0 | ✅ `pip install protenix` | Apache-2.0 | ✅ | ⚠️ unpublished |
| **Protenix-v2** | — | ✅ shipped | ⚠️ **proprietary**, no redistribution | ❌ | — |
| **HelixFold3** (Baidu) | — | ✅ | ⚠️ **non-commercial** | ❌ | ❌ **"32 GB minimum"** |
| `lucidrains/alphafold3-pytorch` | MIT | ❌ **code only** | — | n/a | n/a |
| `Ligo-Biosciences/AlphaFold3` | Apache-2.0 | ❌ **code only**, stale (2024-10) | — | n/a | n/a |

**AlphaFold 3 — the weights policy changed.** There is **no request form any more**; the
README says verbatim: *"You can download the AlphaFold 3 model parameters from
`https://storage.googleapis.com/alphafold3/af3.bin.zst`"*, while still saying *"You may only
use AlphaFold 3 model parameters if received directly from Google."* Commercial use is now
offered via the **Gemini Enterprise Agent Platform on Google Cloud**. But
[`WEIGHTS_TERMS_OF_USE.md`](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md)
remains restrictive:

- *"only available for non-commercial use by, or on behalf of, non-commercial organizations"*
- *"must not publish or share AlphaFold 3 model parameters, except … within your organization"*
- *"must not… use AlphaFold 3 output to train machine learning models or related technology
  for biomolecular structure prediction similar to AlphaFold 3"*
- **"Model Parameters" is defined to include** *"modifications to those weights…, works based
  on those weights…, or other code or machine learning models which incorporate… those
  weights"* — **so a fine-tune inherits every restriction.**
- Output *can* be published and adapted.

Hardware ([docs/performance.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md)):
officially **1× A100 80GB or H100 80GB**, max **5,120 tokens**. A100 80GB: 1,024 tok = 62 s;
2,048 = 275 s; 4,096 = 1,434 s; 5,120 = 2,547 s. A100 40GB reaches 4,352 tokens with unified
memory; V100 32GB reaches 1,280; P100 reaches 1,024. **There is no 24 GB entry.** ⚠️
Extrapolating, a 24 GB L4 with unified memory tops out around **~1,000 tokens with large host
spill** — painful, and **commercially forbidden anyway.**

**OpenFold3 is real and shipped** — ⚠️ the repo is
[aqlaboratory/openfold-3](https://github.com/aqlaboratory/openfold-3) (**not** `openfold3`,
which 404s), **Apache-2.0**, 887★, last push 2026-10-04. The consortium announced on
2026-03-13 that **training datasets (AWS Registry of Open Data), model weights, training and
inference code, and evaluation scripts** are all under permissive licences. ✅
`OpenFold/OpenFold3` is **Apache-2.0** — ⚠️ note it is **`gated: auto`** (auto-approved, but
it is a gate). Two parameter sets: **OpenBind-0** (default from v0.5.0) and
**OpenFold3-preview2**. Also on NVIDIA NIM
([docs](https://docs.api.nvidia.com/nim/reference/openfold-openfold3)) — relevant given
Talindrew's BioNeMo wiring. **It is the only AF3-class model with Apache-2.0 code AND weights
AND published training data AND training code** — so the only one we could legitimately
*fine-tune* and ship. **But it has no affinity head**, so Boltz-2 remains the better fit.

**Chai-1's licence changed too.** The README now states: *"Chai-1 is released under an Apache
2.0 License (both code and model weights), which means it can be used for both academic and
commerical purposes, including for drug discovery."* ✅ I verified `chaidiscovery/chai-1` is
`apache-2.0`. **There is no "Chai Discovery Community License" in the repo any more** — it
launched non-commercial in 2024 and was relicensed. ⚠️ **Archive the LICENSE and README as of
today**, same as for ESM3. **Chai-2 is not open** — API/partnership only. VRAM: README
recommends A100/H100 80GB / L40S 48GB; **24 GB = small complexes only ⚠️, measure.** **No
affinity head, no training code.**

#### NeuralPLexer, DiffDock

- **NeuralPLexer** — [zrqiao/NeuralPLexer](https://github.com/zrqiao/NeuralPLexer),
  **BSD-3-Clause-Clear**. ⚠️ **NeuralPLexer3 is PAPER-ONLY**
  ([arXiv:2412.10743](https://arxiv.org/abs/2412.10743), Iambic Therapeutics) — **no code or
  weights repo exists.** Treat as unavailable.
- **DiffDock** — [gcorso/DiffDock](https://github.com/gcorso/DiffDock), **MIT** (code *and*
  weights), 1,579★. **DiffDock-L** (Feb 2024) is the improved version. ❌ No official HF
  weights — only third-party re-uploads. Confidence guide: >0 high, −1.5 to 0 moderate, <−1.5
  low. **Inference on 24 GB ✅** (it even runs on CPU). See below for why not to centre a
  pipeline on it.

#### Docking throughput on one GPU

| Tool | Licence | Published throughput | → one L4 ⚠️ |
|---|---|---|---|
| **Uni-Dock** | Apache-2.0 | *"more than 2000-fold speed-up on V100 … compared with AutoDock Vina running in single CPU core"*; **~0.1 s/molecule**; **38.2 M molecules vs KRAS G12D in 12 h on 100 V100s** = **31,800 mol/h/GPU** ([JCTC 10.1021/acs.jctc.2c01145](https://doi.org/10.1021/acs.jctc.2c01145)) | **~10,000–25,000 mol/h** |
| **AutoDock-GPU** | **GPL-2.0** ⚠️ | **~2 ligands/s on an RTX 3090 = 7,200 mol/h** ([arXiv:2605.01681](https://arxiv.org/abs/2605.01681)) | **~2,500–4,000 mol/h** |
| **DiffDock** | MIT | **1–2 s/ligand on an A100** = 1,800–3,600 mol/h; *"4 to 8 times [more] computer power … compared to AutoDock, just to deliver inferior precision"* | **~500–1,200 mol/h** |
| **gnina** | ⚠️ **dual GPL + Apache** | No absolute timing. `refine` is *"10x slower than rescore when using a GPU"*; `--cnn_scoring all` is *"extremely computationally intensive and not recommended."* **Pretrained CNN weights ship with it.** | use as a **rescorer**, not a docker |
| **Vina-GPU 2.0** | Apache-2.0 | ⚠️ **README publishes no speedup numbers, GPU spec, per-ligand time or VRAM.** Last push **2023-10-27** — effectively unmaintained | ❌ unverified |
| **AutoDock Vina 1.2.x** | Apache-2.0 | CPU baseline (what Uni-Dock's 2000× is measured against) | — |

⚠️ **Extrapolation basis:** L4 at 121 dense bf16 TFLOPS and **300 GB/s** vs RTX 3090's
~936 GB/s and A100's 1,555 GB/s. Docking kernels are memory- and branch-bound, so discount the
L4 to ~1/3 of a 3090. **These are derived numbers, not published ones.**

⚠️ **gnina's GPL comes from OpenBabel**; Apache-only use requires removing every OpenBabel
reference — **a real packaging constraint for a commercial product.**

> ⭐ **The result that should shape our docking choice** —
> [arXiv:2605.01681](https://arxiv.org/abs/2605.01681), Abo-Dahab, Xiang, Chun & Zhao (UCSF),
> *"Benchmarking Single-Pose Docking, Consensus Rescoring, and Supervised ML on the LIT-PCBA
> Library"*: 578,295 ligand–target pairs, 15 targets, 10,008 actives / 568,287 inactives.
>
> - **AutoDock-GNINA was the best single method: median EF1% = 2.14**, precision **1.85%**.
>   Their own verdict: *"it was only slightly better than random screening."*
> - DiffDock-GNINA: **EF1% = 0.84**. Raw AutoDock 1.10. Raw DiffDock 1.17. Consensus 1.8–1.9.
> - **AutoDock-GNINA uniquely rescued OPRK1 (EF1% = 12.5) where every DiffDock variant
>   scored 0.**
> - **Supervised ML re-ranking on the six docking scores gave by far the largest gains:**
>   Wide NN (512-256-128-1) **EF1% = 4.49 (+110%)**, **Random Forest 4.10 (+92%)**, XGBoost
>   3.80 (+77%), LightGBM 3.735 (+75%). **A *deeper* 256-128-64-1 MLP got 1.792, −16% —
>   depth hurt.**
> - Verbatim: *"no single docking technique works on all targets… employing a rigorously
>   tested docking technique, whose limitations are well understood, is preferable to relying
>   on an unpredictable novel method that may yield inconsistent results."*
>
> **Two takeaways.** (1) **A classical docker + GNINA rescore + a shallow supervised
> re-ranker beats every diffusion docker tested, at 4–8× lower cost** — and Random Forest at
> 4.10 is within 0.4 of the best neural net, the same ECFP+GBM lesson as §C.1. (2) **An EF1%
> of 2–4 on LIT-PCBA is the honest ceiling for structure-based virtual screening.** Do not
> plan around CASF's EF1% ≈ 20–28.

---

### C.4 Affinity / scoring models

| Model | Repo | Licence | Weights | 24 GB inf / FT |
|---|---|---|---|---|
| **PIGNet2** | [ACE-KAIST/PIGNet2](https://github.com/ACE-KAIST/PIGNet2) | **MIT** | ✅ | ✅ / ✅ |
| PIGNet | [ACE-KAIST/PIGNet](https://github.com/ACE-KAIST/PIGNet) | ⚠️ **none** | ✅ | ✅ / ✅ |
| **TankBind** | [luwei0917/TankBind](https://github.com/luwei0917/TankBind) | **MIT** | ✅ | ✅ / ✅ |
| PLANET | [ComputArtCMCG/PLANET](https://github.com/ComputArtCMCG/PLANET) | ⚠️ **none** | ✅ | ✅ / ✅ |
| OnionNet-2 | [zchwang/OnionNet-2](https://github.com/zchwang/OnionNet-2) | ⚠️ **GPL-3.0** | ✅ | ✅ / ✅ |
| **GIGN** | [guaguabujianle/GIGN](https://github.com/guaguabujianle/GIGN) | **MIT** | ✅ | ✅ / ✅ |
| **RTMScore** | [sc8668/RTMScore](https://github.com/sc8668/RTMScore) | **MIT** | ✅ | ✅ / ✅ |
| **GenScore** | [sc8668/GenScore](https://github.com/sc8668/GenScore) | **MIT** | ✅ | ✅ / ✅ |
| EquiScore | [.../EquiScore](https://github.com/Intelligent-Drug-Discovery-Lab/EquiScore) | MIT | ✅ | ✅ / ✅ |
| KarmaDock | [schrojunzhang/KarmaDock](https://github.com/schrojunzhang/KarmaDock) | Apache-2.0 | ✅ | ✅ / ✅ |
| CarsiDock | [carbonsilicon-ai/CarsiDock](https://github.com/carbonsilicon-ai/CarsiDock) | Apache-2.0 | ✅ | ✅ / ⚠️ |
| SurfDock | [.../SurfDock](https://github.com/Intelligent-Drug-Discovery-Lab/SurfDock) | MIT | ✅ | ✅ / ⚠️ |
| Interformer | [tencent-ailab/Interformer](https://github.com/tencent-ailab/Interformer) | Apache-2.0 | ✅ | ✅ / ⚠️ |
| SIGN | PaddleHelix | ⚠️ NOASSERTION | ✅ | ✅ / ✅ |
| **KDeep** | Acellera | ⚠️ **commercial/web only** | ❌ | n/a |

**All of these are small GNNs or 3D CNNs — tens of millions of parameters at most. Every one
fits comfortably on a 24 GB L4 for both inference and full fine-tuning. VRAM is not the
constraint anywhere in this section.** The constraints are (a) whether the pipeline runs at
all and (b) whether reported accuracy survives non-crystal inputs — both covered in section D.

#### CASF-2016, with actual numbers

From Table 1 of **[GenScore, *Chem. Sci.* 2023, DOI 10.1039/d3sc02044d](https://pmc.ncbi.nlm.nih.gov/articles/PMC10395315/)**:

| Method | Type | Docking SR1 | Screening fwd SR1 | **EF1%** | **Scoring R_p** | **Ranking R_s** |
|---|---|---|---|---|---|---|
| AutoDock Vina | classical | 0.846 | 0.298 | 7.70 | 0.604 | 0.528 |
| ChemPLP@GOLD | classical | 0.832 | 0.351 | 11.91 | 0.614 | 0.633 |
| GlideScore-SP | classical | 0.846 | 0.368 | 11.44 | 0.513 | 0.419 |
| KORP-PL | classical | 0.856 | 0.421 | 22.23 | 0.447 | 0.570 |
| **KDEEP** | 3D CNN | 0.291 | — | — | **0.701** | 0.528 |
| **AKScore** | ML | 0.360 | — | — | **0.812** | 0.670 |
| ΔVinaRF20 | ML Δ | 0.849 | 0.456 | 12.36 | 0.739 | 0.635 |
| ΔVinaXGB | ML Δ | — | 0.368 | 13.14 | 0.796 | 0.647 |
| **ΔLin_F9^XGB** | ML Δ | — | 0.404 | 12.61 | **0.845** | **0.704** |
| AEScore | ML | 0.358 | — | — | **0.830** | 0.640 |
| **PIGNet** | GNN | 0.870 | 0.554 | 19.60 | 0.761 | **0.682** |
| DeepDock | GNN | 0.870 | 0.439 | 16.41 | 0.460 | 0.425 |
| **RTMScore** | GNN | **0.934** | **0.667** | **28.00** | 0.455 | 0.529 |
| **GenScore GT_ft_1.0** | GNN | **0.940** | **0.719** | **28.12** | **0.802** | **0.684** |
| GenScore GatedGCN_1.0 | GNN | 0.912 | 0.585 | 19.78 | **0.837** | 0.682 |
| GenScore GatedGCN_ft_1.0 | GNN | 0.926 | 0.661 | 23.54 | 0.834 | **0.686** |

**Leaders 2025–2026:** scoring power ⚠️ **TopoFormer 0.881**, ΔLin_F9^XGB 0.845, GenScore
GatedGCN 0.837, AEScore 0.830, iScore 0.814
([*JCIM* 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11938276/) — R = 0.814, RMSE = 1.34,
ρ = 0.705), AKScore 0.812. **Ranking power ceiling is ~0.684–0.704.** Newest: **BA-Pred**
([*JCIM* 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13080981/)) — *"state-of-the-art
scoring power on the CASF-2016 benchmark with a root-mean-squared error of 1.10 pKd"*, 96%
top-1 docking power, EF1% 21.1, **ranked 2nd in CASP16's ligand-binding-affinity category** —
and it improved AutoDock-GPU's LIT-PCBA EF1% from **2.18 to 3.19, which is the honest
real-world number.**

> ⚠️ **Note the striking anti-correlation: RTMScore has the best docking (0.934) and screening
> (EF1% 28.00) power in that table and nearly the *worst* scoring power (0.455).**
> Docking/screening power and scoring power are **different skills**, and a model tuned for
> one is routinely bad at the other. GenScore's whole contribution is balancing them.
> **For triage we want ranking (R_s) and screening (EF1%), not scoring R_p** — so the
> GenScore `_ft` variants and RTMScore are the relevant leaders, **not** the scoring-power
> champions.

#### Boltz-2's affinity numbers vs FEP — the actual tables

From the [technical report](https://jeremywohlwend.com/assets/boltz2.pdf), Tables 11–15.

**Table 12 — FEP+ 4-target subset (CDK2, TYK2, JNK1, p38):**

| Method | Type | Time | Pearson R | Kendall τ | PMAE (kcal/mol) |
|---|---|---|---|---|---|
| **Boltz-2** | ML | **20 GPU sec** | **0.66** | **0.48** | **0.85** |
| Boltz-2 ipTM | ML | 5 GPU sec | **0.04** | 0.09 | — |
| GAT | ML | 0.18 GPU ms | 0.40 | 0.28 | 1.07 |
| BACPI | ML | 0.48 GPU ms | 0.14 | 0.09 | 1.18 |
| **OpenFE** | physics | **6–12 GPU hours** | **0.66** | 0.51 | 1.09 |
| **FEP+** | physics | — | **0.78** | **0.63** | **0.77** |
| ABFE | physics | **>20 GPU hours** | 0.75 | 0.54 | 0.95 |
| FMO | physics | 2–10 CPU min | 0.55 | 0.38 | — |
| MM/PBSA | physics | 10–15 CPU min | 0.18 | 0.16 | — |

> **The honest statement: Boltz-2 matches the open-source FEP implementation (OpenFE,
> R = 0.66) at ~1/1000 the cost, and remains clearly behind commercial FEP+ (0.78).** Note
> also that **Boltz-2's own ipTM scores R = 0.04** — which is why the dedicated affinity head
> exists, and why ESMFold2's iPTM (§C.2) is a feature, not a score.

- **Table 11 — OpenFE subset (876 hit-to-lead measurements):** Boltz-2 **R = 0.62**, τ 0.46 |
  OpenFE 0.63 | FEP+ 0.72 | GAT 0.28 | BACPI 0.29 | **Boltz-2 ipTM −0.07**.
- **Table 14 — CASP16 blind affinity challenge** (data released after Boltz-2's training
  cutoff; max Tanimoto to training 0.41–0.59): **Boltz-2 R = 0.65, τ = 0.45**, beating **every
  one of the top-6 human CASP16 entrants** (best LG016 at 0.54). ⭐ **This is the most
  credible Boltz-2 number, because it is genuinely prospective.**
- **Table 13 — MF-PCBA hit discovery:** Boltz-2 **AP 0.0248, EF@0.5% 18.39, EF@1% 13.95,
  AUROC 0.812**, versus Chemgauss4 docking (AP 0.0051, EF@0.5% 2.00, AUROC 0.545) — **~5× the
  average precision and ~9× the EF@0.5% of docking.**
- **Table 15 — hit-to-lead validation:** Boltz-2 **R = 0.425**. The report's own candour: on
  private Recursion assays **R = 0.39**, and *"centered MAE = 1.36 kcal/mol is significantly
  worse compared to the validation set (MAE = 0.86)"*; on one target R fell 0.634 → 0.42.

**Independent third-party evaluation — and it is harsh.** Wan, Zhang, Xue & Coveney (UCL),
*"On the Reliability of AI Methods in Drug Discovery: Evaluation of Boltz-2 for Structure and
Binding Affinity Prediction"*, [arXiv:2603.05532](https://arxiv.org/abs/2603.05532),
2026-03-02. 3CLPro (16,780 compounds) and TNKS2 (21,702):

- Boltz-2 vs **ESMACS** physics-based ΔG: **3CLPro r = 0.24, ρ = 0.25; TNKS2 r = 0.45,
  ρ = 0.46.**
- **On the top-100 compounds, correlations "collapsed to near zero."**
- Reproducibility between independent runs was high (r = 0.913 / 0.962) — **the noise is
  systematic, not stochastic.**
- **"Regression to the mean":** predictions bunch between −5 and −8 kcal/mol; **all ligands
  scored >0.80 confidence** (overconfident).
- **Weak internal consistency: r = 0.21–0.25 between the affinity value and the binding
  probability** from the same embeddings.
- **28–30 of the top 100 had incorrect bonding topology.**
- Verdict: Boltz-2 *"lacks the energetic resolution required for lead identification"* though
  it offers *"substantial speed for initial screening."*

> ⚠️ **One methodological caveat worth weighing:** that critique correlates Boltz-2 against
> **ESMACS predictions, not experimental data.** Disagreement between two computational
> methods does not establish which is wrong, so the global r = 0.24 is weaker evidence than it
> looks. **The findings to act on regardless are (a) do not treat
> `affinity_probability_binary` as calibrated — everything scores >0.80 — and (b) expect rank
> correlation to vanish within an already-enriched top slice, which is precisely the regime
> triage cares about.**

---

### C.5 What this means for one L4, 24 h

**Trainable / fine-tunable on 24 GB:** PIGNet2, TankBind, PLANET, GIGN, OnionNet-2, RTMScore,
GenScore, EquiScore, SIGN, ESM-2 ≤650M, ESMC ≤600M, AMPLIFY_350M, SaProt 650M, every molecular
model in §C.1, **plus Boltz-2's affinity heads** via the frozen-trunk recipe.
**Inference-only on 24 GB:** AlphaFold 3 (and licence-barred), Chai-1, HelixFold3,
ESMFold/ESMFold2, Boltz-2's structure trunk, ESMC-6B, ESM-2 3B, KDeep (closed).

**The realistic accuracy ceiling, stated plainly:**

| Task | Honest expectation |
|---|---|
| Congeneric-series / lead-opt ranking | Pearson **R ≈ 0.6–0.7** per target with Boltz-2 off-the-shelf; ⚠️ plausibly 0.7–0.8 after affinity-head fine-tuning on 50–200 of our own compounds (single-sourced). **FEP+ at 0.78 is the bar; OpenFE at 0.66 is what is reachable for free.** |
| Hit discovery / enrichment | **EF1% ≈ 2–4** on realistically-curated libraries (LIT-PCBA), **not** the EF1% ≈ 20–28 of CASF-2016. Precision at the top 1% is ~2%. |
| Cross-target generalisation | Expect **ΔPCC ≈ −0.2** versus any crystal-structure benchmark the moment you use docked or predicted inputs (section D). |
| ADMET | TDC-leaderboard-competitive is achievable with ECFP+GBM or CheMeleon/MiniMol — but ⚠️ the leaderboards are leakage-contaminated, so **our own time-split holdout is the only number worth believing.** |

**Licence summary for a commercial product:**

- ❌ **Avoid: Ankh / Ankh2 / Ankh3 (✅ `cc-by-nc-sa-4.0`), ProGen3, MolE** — non-commercial,
  and the share-alike would try to pull our derivative under the same terms.
- ❌ **Avoid: AlphaFold 3 weights, HelixFold3, Protenix-v2** — non-commercial or proprietary;
  AF3's terms explicitly extend to derivatives and forbid using its output to train a
  competing structure predictor.
- ⚠️ **Unlicensed-on-HF but widely used:** all ProtT5/ProtBert repos, all `seyonec`
  ChemBERTas, the small DeepChem ChemBERTas, all `chandar-lab/Sa*`, all ProtGPT3, all Lobster,
  `Synthyra/FastESM2_650`. Upstream intent is usually permissive — **but the artefact carries
  no grant.** Prefer something explicitly MIT.
- ⚠️ **Profluent E1** is commercially usable with **mandatory attribution and a termination
  clause** — unlike MIT, **it can be revoked.**
- ⚠️ **gnina** is dual GPL/Apache, GPL via OpenBabel. **AutoDock-GPU is GPL-2.0.**
  **OnionNet-2 is GPL-3.0.**
- ✅ **Clean MIT/Apache all the way down:** Boltz-1/Boltz-2, OpenFold3, Chai-1, ESM-2 (all
  sizes), ESM C, ESMFold/ESMFold2, AMPLIFY, SaProt, ProstT5, PepMLM, CheMeleon, MiniMol,
  ChemFM, OpenADMET, `DeepChem/ChemBERTa-100M-MLM`, `DeepChem/MoLFormer-c3-1.1B`, MoLFormer-XL
  (Apache), Uni-Mol/Uni-Mol2, Chemprop, scikit-fingerprints, DiffDock, Uni-Dock,
  AutoDock-Vina, PIGNet2 / TankBind / GIGN / RTMScore / GenScore.

---

## D. Evaluation that is actually honest

Citations in this section were resolved against Crossref / OpenAlex / Europe PMC rather than
quoted from memory. ⚠️ marks a claim taken from an abstract or secondary source; ❌ marks
something that does not exist. **A useful technique worth recording: the Europe PMC REST API
(`ebi.ac.uk/europepmc/webservices/rest/search?...&resultType=core` and
`/PMC<id>/fullTextXML`) returns verbatim abstracts and OA full text past publisher 403s** —
it is how most of this was verified, and I re-verified the twelve load-bearing citations
myself.

### D.0 Four facts that drive the whole protocol

1. **Your split is your result.** On the same 60 NCI-60 datasets and the same random forest,
   "hit rate in the top 100" goes **93% → 91% → 84% → 43%** as you move random → scaffold →
   Butina → UMAP. **Nothing about the model changed.**
2. **Scaffold split does not do what people think.** Under the standard MoleculeNet scaffold
   splits, **76% of ESOL, 56% of HIV, 78% of TDC `CYP2D6_Veith` and 88% of MolData test
   molecules still have a training neighbour at ECFP4 Tanimoto > 0.4.** Two molecules
   differing by one atom can have different Bemis–Murcko scaffolds at 0.66 Tanimoto.
3. **For structure-based affinity, the single most important number you can report is the
   ligand-only ablation.** A GNN trained on unfiltered PDBbind scores **RMSE 1.424** on
   CASF-2016 *with all protein information deleted* — **better than AutoDock Vina's 1.73** and
   better than several published complex-based deep scoring functions.
4. **Your error floor is ~0.4–0.7 log units and your R² ceiling is ~0.81** on heterogeneous
   public bioactivity data. **An RMSE below ~0.5 log units is not an achievement; it is a
   leakage detector that has gone off.**

A sceptical computational chemist will believe our numbers if, and realistically only if, we
show **(a)** a panel of splits with the random-split number included as an acknowledged
ceiling, **(b)** a tuned ECFP4+LightGBM baseline that we did not beat by much, **(c)**
ligand-only / protein-only / label-shuffled ablations, and **(d)** the train–test
nearest-neighbour similarity distribution for every split we report.

---

### D.1 Split strategies

#### The methods

**Random split.** Reference only. Its value is as a *stated upper bound*. Keep it in the table
so the reader can see the size of the gap.

**Bemis–Murcko scaffold split** — scaffold = ring systems plus the linker atoms between them
(Bemis & Murcko 1996, *J Med Chem* 39:2887–2893, DOI
[10.1021/jm9602928](https://doi.org/10.1021/jm9602928)). **Two implementations differ in ways
that matter:**

- **DeepChem `ScaffoldSplitter`** — RDKit `MurckoScaffoldSmiles(includeChirality=False)`;
  scaffold sets sorted by size descending, assigned sequentially to train → valid → test. It
  is **deterministic — one split, no seed, so no error bars from splitting.** Invalid SMILES
  are silently dropped. ✅ **There is no "balanced" variant in DeepChem** (the source was
  checked).
- **Chemprop `SCAFFOLD_BALANCED`** is the "balanced" variant people mean (Yang et al. 2019,
  *JCIM* 59:3370–3388, DOI [10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237)),
  verbatim: *"Molecules are partitioned into bins based on their Murcko scaffold... Any bins
  larger than half of the desired test set size are placed into the training set... All
  remaining bins are placed randomly into the training, validation, and test sets."* **Because
  the remainder is randomised, you can generate several scaffold splits and get error bars** —
  which is the whole reason to use it over DeepChem's. In Chemprop v2 it is implemented via
  [`astartes`](https://github.com/JacksonBurns/astartes) (JOSS DOI
  [10.21105/joss.05996](https://doi.org/10.21105/joss.05996)), which also offers
  `RANDOM_WITH_REPEATED_SMILES` (handles duplicate-SMILES leakage) and `KENNARD_STONE`.

**Cluster / Butina split** — Taylor–Butina clustering (Butina 1999, *J Chem Inf Comput Sci*
39:747–750, DOI [10.1021/ci9803381](https://doi.org/10.1021/ci9803381)): Morgan fingerprints,
Tanimoto distance, leader-based clustering, whole clusters assigned to a fold. DeepChem's
`ButinaSplitter` defaults to **cutoff 0.6**. This is Pat Walters' recommended default
([*Practical Cheminformatics*, 2024-11-16](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html))
because, unlike scaffold, **it directly bounds train–test similarity.**

**Time / temporal split — the gold standard.** Sheridan 2013, *JCIM* 53(4):783–790, DOI
[10.1021/ci400084k](https://doi.org/10.1021/ci400084k) (✅ PMID 23521722, re-verified). From
the abstract, verbatim: *"time-split selection gives an R² that is **more like that of true
prospective prediction** than the R² from random selection (**too optimistic**) or from our
analog of leave-class-out selection (**too pessimistic**). Time-split selection should be used
**in addition to** random selection as a standard for cross-validation."* Usual construction:
first-75%-assayed → train, last-25% → test. It is the gold standard because its distribution
shift *is the shift we will actually face*: compounds get larger, more polar and more
synthetically complex over a project's life, and **nobody designed the test set.**

> ❗ **This is why section B.1 insists on keeping `document_chembl_id`.** Without the document
> year you cannot build a time split at all, and it is the one split a sceptic will ask for.

**SIMPD — the answer when you have no dates.** Landrum, Beckers, Lanini, Schneider, Stiefl &
Riniker 2023, *J Cheminform* 15:119, DOI
[10.1186/s13321-023-00787-9](https://doi.org/10.1186/s13321-023-00787-9). A multi-objective
genetic algorithm that manufactures a *simulated* time split, with objectives fitted to **138
real Novartis lead-optimisation projects**. The targets it optimises toward are themselves the
best published description of what a temporal shift *is*:

| Objective (test − train medians) | Target | Direction |
|---|---|---|
| Δ median SA_Score (synthetic accessibility) | 0.28 | test > train |
| Δ median heavy-atom count | 3.1 | test > train |
| Δ median TPSA | 13.2 | test > train |
| Δ median benzene fraction / 1000 heavy atoms | −8.8 | test < train |
| spatial statistic ΣG − ΣF′ | 10–30 | test clusters |
| spatial statistic ΣG | > 70 | test compounds similar to each other |

Ships **99 SIMPD-split ChEMBL32 datasets** (Ki assays, 300–1000 compounds) at
[rinikerlab/molecular_time_series](https://github.com/rinikerlab/molecular_time_series).

**UMAP split** — Guo, Hernandez-Hernandez & Ballester 2025, *J Cheminform* 17:94, DOI
[10.1186/s13321-025-01039-8](https://doi.org/10.1186/s13321-025-01039-8). Morgan FP (2048
bits, r=2) → UMAP → agglomerative clustering → 7 clusters over 33,118 NCI-60 molecules;
leave-one-cluster-out.

**Lo-Hi** — Steshin 2023, *Lo-Hi: Practical ML Drug Discovery Benchmark*, NeurIPS 2023
Datasets & Benchmarks, [arXiv:2310.06399](https://arxiv.org/abs/2310.06399). Two tasks matched
to two real jobs:

- **Hi (Hit Identification)** — binary classification where **no test molecule exceeds ECFP4
  Tanimoto 0.4 to any training molecule**, posed as a Balanced Vertex Minimum k-Cut on the
  similarity graph. Far less wasteful than greedy: on DRD2 it discards **97 molecules (1.5%)**
  vs greedy's 1,066 (17.0%); on HIV, **1,598 (3.8%)** vs 5,851 (14.2%).
- **Lo (Lead Optimisation)** — a per-cluster *ranking* task: isolate clusters with intracluster
  similarity ≥ 0.4 and ≥ 5 members, put the cluster in test, leave **exactly one** member in
  train (the known hit). They verify intracluster activity variance exceeds experimental noise,
  so the task is learnable in principle.
- Code: [lohi_splitter](https://github.com/SteshinSS/lohi_splitter).

**SPECTRA — the conceptually right answer.** Ektefaie, Shen, Bykova, Marin et al. 2024,
*Nature Machine Intelligence*, DOI
[10.1038/s42256-024-00931-6](https://doi.org/10.1038/s42256-024-00931-6)
([code](https://github.com/mims-harvard/SPECTRA)). Instead of one split, generate a *series*
of splits spanning decreasing **cross-split overlap**, plot performance against it, and report
the **area under that spectral performance curve**. Verbatim: traditional metadata- or
similarity-based splits *"mischaracteri[se] model generalizability by failing to consider the
full spectrum of cross-split overlap."* Applied to 18 datasets (protein–ligand binding among
them) and 19 models. **If we adopt one upgrade over a split panel, this is it — it makes the
"which split is right" argument disappear.**

**ASAP / Polaris blind challenge** — the closest thing the field has to a true prospective
ADMET+potency test: MacDermott-Opeskin et al. 2026, *JCIM*, DOI
[10.1021/acs.jcim.5c02106](https://doi.org/10.1021/acs.jcim.5c02106) — 66 teams, **381
submissions**, held-out *undisclosed* lead-optimisation data from the ASAP Discovery
pan-coronavirus programme (SARS-CoV-2 and MERS-CoV Mpro). Reported: top potency models reached
**~0.5 log unit mean error** (i.e. the experimental noise floor); lipophilicity and
permeability modelled well; **solubility and liver clearance remained hard**; co-folding hit
the right pose in **>80%** of cases; winning recipes were pretraining + multi-task +
ensembles. ⚠️ Per-task leaderboard tables unverified (paywalled).

#### The drop, with numbers

**Same model, same data, four splits** (Guo et al. 2025, 60 NCI-60 datasets, 7-fold CV × 5
seeds = 300 evaluations per cell; metric = % of true actives among the top-100 predictions):

| Model | Random | Scaffold | Butina | UMAP |
|---|---|---|---|---|
| Linear regression | ~83% | ~81% | ~51% | **~15%** |
| **Random forest** | **~93%** | **~91%** | **~84%** | **~43%** |
| GEM (pretrained GNN) | ~94% | ~90% | ~82% | **~67%** |
| Transformer-CNN | ~92% | ~81% | ~72% | **~50%** |

And the mechanism — **mean maximum Tanimoto from a test molecule to its nearest training
neighbour**, averaged over 60 datasets:

| Random | Scaffold | Butina | UMAP |
|---|---|---|---|
| **0.717** | **0.625** | **0.417** | **0.375** |

> ❗ **Note that the model *ranking flips*.** Random and Butina splits say random forest is
> best; the UMAP split says GEM is, **by 24 points**. They also found **ROC-AUC correlates only
> ρ = 0.368 with top-100 hit rate** — *"similar ROC AUC scores can correspond to vastly
> different hit rates."* For a triage product, top-100 hit rate *is* the deliverable and
> ROC-AUC is nearly uninformative about it.

**Walters' Biogen solubility experiment** (LightGBM, R², 5×5-fold): random **0.71** →
scaffold **0.66** → Butina **0.61** → UMAP **0.55**. Tukey HSD: UMAP differs significantly
from the rest. **His preference is Butina** — UMAP's 2-D projection discards high-dimensional
similarity structure and its variable cluster sizes inflate metric variance.

**Random vs true temporal, via SIMPD** (138 NIBR project datasets, RF classification, median
ΔAUC vs the temporal split):

| Split | Median ΔAUC vs temporal |
|---|---|
| Random | **+0.04** (too optimistic) |
| Neighbour | **−0.06** (too pessimistic) |
| SIMPD | **+0.02** (closest) |

So on real medicinal-chemistry project data the random-vs-prospective AUC gap is about
**0.04**, and a naive similarity split **overshoots in the other direction by 0.06** — exactly
Sheridan's 2013 finding reproduced on modern industrial data.

⭐ **Random CV vs cluster split on real ADME data, 2026** — Seal, Zalte, Araripe, Gomes,
Korani, Shekhar, Siramshetty, Patra et al. 2026, bioRxiv DOI
[10.64898/2026.08.19.745868](https://doi.org/10.64898/2026.08.19.745868), a cross-industry
consortium framework on the Expansion Therapeutics ADMET set (7,608 molecules, 9 ADME
endpoints, temporally ordered; the top 3 Butina clusters hold **62.5%** of the molecules):

- Random 5-fold CV: **MA-RAE ≈ 0.47**, with "remarkably small" SDs of **0.003–0.012**.
- Butina cluster split (Jaccard 0.7): **MA-RAE ≈ 0.67**, with honestly wide ranges.
- The ~0.20 gap is **~7× the within-cluster per-endpoint split SD**.
- **Cross-series generalisation: R² goes negative for 5 of 8 endpoints under XGBoost and 6 of
  8 under CheMeleon** when training on one chemical series and testing on a structurally
  distinct one — despite strong in-distribution performance.

> ❗ **That last line is the single most important number in this section for a triage
> project.** Train on one series, test on another, and a well-performing model becomes **worse
> than predicting the mean.** Since triage exists precisely to evaluate *new* chemistry, a
> series-holdout split is not optional.

**MoleculeNet scaffold-split contamination** (Steshin 2023, Table 6) — fraction of test
molecules with an ECFP4 Tanimoto > 0.4 neighbour in train: QM9 **0.99**, QM8 0.98, QM7 0.93,
MUV **0.96**, PCBA >0.93, FreeSolv 0.80, BACE 0.77, Lipophilicity 0.67, Tox21 0.52, SIDER
0.48.

**The 2026 caveat — don't assume, measure.** Reimer, Do & Green 2026, *"Chemical space
separation: Scaffold and UMAP splits are closer to random than we think"*, ChemRxiv DOI
[10.26434/chemrxiv.15007205](https://doi.org/10.26434/chemrxiv.15007205): across 8
MoleculeNet datasets, 6 classical models and Chemprop, **"in 3 out of 8 datasets, scaffold and
UMAP splits do not consistently produce more challenging tasks than random splits."** So
**name-dropping a split proves nothing — report the similarity distribution.**

**And a temporal split can fail to test generalisation at all on structural data.** Corso,
Deng, Fry, Polizzi, Barzilay & Jaakkola 2024 (DockGen), ICLR,
[arXiv:2402.18396](https://arxiv.org/abs/2402.18396): PDBbind pre-2019 (~17k complexes) spans
only **487 ECOD t-groups**, and the post-2019 time-split test set adds **8 new clusters / 15
complexes**. A time split there is almost entirely in-distribution.

---

### D.2 CASF-2016 and the four powers

**Su, M.; Yang, Q.; Du, Y.; Feng, G.; Liu, Z.; Li, Y.; Wang, R.** "Comparative Assessment of
Scoring Functions: The CASF-2016 Update." *J. Chem. Inf. Model.* **2019**, 59(2), 895–913. DOI
[10.1021/acs.jcim.8b00545](https://doi.org/10.1021/acs.jcim.8b00545). ✅ PMID 30481020
(re-verified). From the abstract, verbatim: *"CASF is designed as a 'scoring benchmark', where
the **scoring process is decoupled from the docking process**... A new test set is compiled,
which consists of **285 protein–ligand complexes**... **A panel of 25 scoring functions** are
tested... the performance of current scoring functions is **more promising in terms of docking
power than scoring, ranking, and screening power**."*

**Composition: 285 complexes = 57 targets × 5 ligands each**, sampled non-redundantly from
PDBbind. Plus, for the dynamic tests: **docking power** uses a decoy set of up to 100 docked
poses per complex; **screening power** cross-docks each of the **57 proteins with the other
280 ligands** (so the matrix is 57 × 285, and cross-binders mean a target may have more than 5
true binders).

Predecessors: **CASF-2013** = 195 complexes (protocol paper: Li, Su, Liu & Wang et al. 2018,
*Nature Protocols* 13:666–680, DOI
[10.1038/nprot.2017.114](https://doi.org/10.1038/nprot.2017.114)); **CASF-2007** = 195
complexes from 65 clusters (Cheng et al. 2009, DOI
[10.1021/ci9000053](https://doi.org/10.1021/ci9000053)).

| Power | What it asks | Metric |
|---|---|---|
| **Scoring** | Do predicted scores correlate linearly with measured affinity across all 285? | **Pearson R** (and RMSE in pK units) |
| **Ranking** | Within each of the 57 targets, is the order of its 5 ligands right? | **Spearman ρ** (also Kendall τ, predictive index), averaged over 57 clusters |
| **Docking** | Is the top-scored pose within 2 Å RMSD of the crystal pose? | **SR1 (%)** top-1 success rate |
| **Screening** | Can you pull a target's true binders out of the 285-ligand cross-docked pool? | **EF1%** plus **SR1%** = % of targets whose best binder is in the top 1% |

#### Current best reported numbers

From Tables 4–5 of Jiang, Zhang & Hou et al. 2024 (*DeepRLI*,
[arXiv:2401.10806](https://arxiv.org/abs/2401.10806)); the first rows are Su et al.'s own 2019
values.

| Method | Scoring RMSE | Scoring Rp | Ranking ρ | Docking SR1 (%) | Screening SR1% | Screening EF1% |
|---|---|---|---|---|---|---|
| **AutoDock Vina** | 1.73 | 0.604 | 0.528 | 90.2 | 29.8 | 7.70 |
| Glide SP | 1.89 | 0.513 | 0.419 | 87.7 | 36.8 | 11.44 |
| Glide XP | 1.95 | 0.467 | 0.257 | 83.9 | 26.3 | 8.83 |
| X-Score | 1.69 | 0.631 | 0.604 | 63.5 | 7.0 | 2.68 |
| Δvina RF20 | 1.26 | 0.816 | **0.750** | 89.1 | 42.1 | 11.73 |
| ΔLin_F9 XGB | 1.24 | 0.845 | 0.704 | 86.7 | 40.4 | 12.6 |
| AEScore | 1.22 | 0.830 | 0.640 | 35.8 | — | — |
| PLANET | 1.25 | 0.824 | 0.682 | 71.8 | — | — |
| PIGNet | — | 0.749 | 0.668 | — | 50.0 | 18.5 |
| **GenScore** | — | 0.829 | 0.673 | **93.3** | **57.3** | **18.58** |
| **DeepRLI** | **1.18** | **0.849** | 0.730 | 90.9 | 26.3 | 11.42 |

**Best per power:** scoring **Rp 0.849** (DeepRLI; ΔLin_F9 XGB 0.845), ranking **ρ 0.750**
(Δvina RF20), docking **SR1 93.3%** (GenScore), screening **EF1% 18.58** (GenScore).

> ⚠️ **Two warnings on that table, and they are serious.** First, **reported EF1% for the same
> method varies by protocol** — PIGNet2's table gives GenScore EF1% = **28.2** / SR1% = 71.4
> where DeepRLI's gives **18.58** / 57.3. (Section C.4's GenScore table, drawn from the
> GenScore paper itself, gives 28.12 — all three are "GenScore's EF1%".) Second, **almost
> every row is a number the method's own authors computed**; it is not a controlled
> comparison. Note too that **ΔLinF9_XGB's scoring Rp moves from 0.853 to 0.839 to 0.813**
> depending only on whether poses were locally optimised, flexibly re-docked or
> ensemble-docked.

#### ❌ There is no CASF-2024, and no "CBScore"

✅ A Crossref search of the CASF series confirms **CASF-2016 is the latest official release**;
nothing titled CASF-2024 exists. ❌ **No benchmark called "CBScore" exists in Crossref — do
not cite it.** The *de facto* successors are dataset-hygiene and split projects, not a new
CASF: **PDBbind CleanSplit / GEMS**, **LP-PDBbind**, **PDBbind-Opt / HiQBind**, **BindingNet
v2**, **PLINDER** (all in §B.3–B.5), plus:

- **CASP16 ligand category** — Gilson, Eberhardt, Škrinjar, Durairaj, Robin & Kryshtafovych
  2026, *Proteins* 94(1):249–266, DOI [10.1002/prot.70061](https://doi.org/10.1002/prot.70061):
  30 groups, 229 pose targets, 140 affinity targets, real pharma compounds. Best groups mean
  **LDDT-PLI 0.69**; ❗ **the automated AlphaFold 3 baseline scored 0.80, beating the best
  human predictor.** Affinity: **max Kendall τ = 0.42 against a noise-imposed ceiling of
  ~0.73**, and **giving predictors the experimental structures in stage 2 did not improve
  affinity — the scoring function is the bottleneck, not the structure.**

#### The critique: CASF scoring power is inflated — now quantified

⭐ **Graber, Stockinger, Meyer, Mishra, Horn & Buller 2025, "Resolving data bias improves
generalization in binding affinity prediction", *Nature Machine Intelligence*, DOI
[10.1038/s42256-025-01124-5](https://doi.org/10.1038/s42256-025-01124-5)** (✅ PMID 41143208,
re-verified; [code](https://github.com/camlab-ethz/GEMS)). They built a filter combining
**TM-align** protein similarity, **count-Morgan Tanimoto** ligand similarity and
**pocket-aligned ligand RMSD**, excluding a training complex if *all* of TM-score > 0.8,
(Tanimoto + (1 − RMSD)) > 0.8 and |ΔpK| ≤ 1:

- ❗ **Only 144 of the 285 CASF-2016 complexes (≈51%) are independent of PDBbind.** For
  CASF-2013 it is **101 of 195**. **About half of CASF has a near-duplicate in the training
  set.**
- A trivial memorisation baseline — *average the labels of the 5 most similar training
  complexes* — reaches **Pearson R = 0.716, RMSE = 1.517** on CASF-2016, **competitive with
  published deep scoring functions.** The **ligand-only** version reaches **R = 0.707,
  RMSE = 1.539**.
- On CleanSplit those collapse to **R = 0.653** and **R = 0.625**.
- Retraining published models on CleanSplit: **Pafnucy drops "substantially, approaching the
  level of the simple search algorithms"** (RMSE 1.484); **GenScore is more robust** (1.362);
  GEMS 1.308.
- ⭐ **Less training data, better true generalisation:** GEMS trained on raw PDBbind scores
  RMSE **1.223 on all 285 but 1.483 on the independent 144**; trained on CleanSplit it scores
  **1.367** on that same subset.

Also: **Zhu, Yang & Huang 2022**, *JCIM*, DOI
[10.1021/acs.jcim.2c01149](https://doi.org/10.1021/acs.jcim.2c01149) evaluated **12 ML scoring
functions** under Random-CV vs protein-sequence-CV vs **pocket-Pfam-CV**: *"all of the tested
models showed decreased performances from Random-CV to Seq-CV to Pfam-CV... not showing
satisfactory generalization capacity."* Their mechanistic finding: cross-target predictions
were driven by **buried solvent-accessible surface area** — bigger interface, bigger predicted
affinity. And **Kopko, Graber et al. 2025**,
[arXiv:2512.05386](https://arxiv.org/abs/2512.05386): *"commonly used benchmarks do not
reflect the true challenge of generalizing to novel targets."*

---

### D.3 The ligand-only / target-memorisation failure mode

#### The evidence

**Yang, Shen & Huang 2020, "Predicting or Pretending", *Frontiers in Pharmacology* 11:69, DOI
[10.3389/fphar.2020.00069](https://doi.org/10.3389/fphar.2020.00069)** — an atomic CNN on the
PDBbind refined set, Pearson R²:

| Input | R² (random split) | R² (ligand-scaffold split) | R² (protein-sequence split) |
|---|---|---|---|
| **Ligand only** | **0.71** | **0.48** | — |
| **Protein only** | **0.84** | — | **0.63** |
| Full complex | **0.80** | — | **0.35** |

Their words: the models *"did not require learning the essential protein-ligand
interactions"*. ❗ **Protein-only (0.84) beat the full complex (0.80)** — the model had learned
"which target is this" and looked up that target's mean affinity.

> This is the paper the brief was reaching for with "Yang et al. 2020 hidden bias". ❌ There is
> **no paper with the author set "Chen, Cruz, Paul, Ash, Durrant"** — that is a mis-merge of
> Chen et al. 2019 (*PLoS ONE*, DUD-E) and Yang, Shen & Huang 2020 (*Front Pharmacol*). Both
> exist and are normally cited together.

**Graber et al. 2025's ablation** — trained on unfiltered PDBbind, *"all tested GEMS model
variants achieved competitive CASF2016 performance even after removing all protein information
from the input data (**r.m.s.e. = 1.424**)... these models — when trained only on ligands —
make **more accurate predictions than the scoring function of AutoDock Vina** and even
outperform several published deep-learning-based scoring functions."* Trained on CleanSplit,
the same ablation degrades to **1.572**. **That contrast *is* the test: if deleting the protein
barely hurts you, you have not learned binding.**

**Boyles, Deane & Morris 2020, "Learning from the ligand", *Bioinformatics* 36(3):758–764, DOI
[10.1093/bioinformatics/btz665](https://doi.org/10.1093/bioinformatics/btz665)** (✅ PMID
31598630, re-verified) — RF-Score v3 + RDKit descriptors reaches Pearson **0.836 / 0.780 /
0.821** on the PDBbind 2007/2013/2016 core sets vs **0.790 / 0.746 / 0.814** for RF-Score v3
alone. The mechanistic reading matters more than the numbers: a **ligand-only RF is
"predictive at a level similar to classical scoring functions" and appears to be predicting the
ligand's *mean* affinity across its binding partners**; excluding similar train proteins and
ligands *"has a significant effect... but does not remove the predictive power of ligand-based
features."* Their 2022 follow-up is also worth having — *"Learning from Docked Ligands:
Ligand-Based Features Rescue Structure-Based Scoring Functions When Trained on Docked Poses"*,
*JCIM* 62, ✅ PMID 34469150, DOI
[10.1021/acs.jcim.1c00096](https://doi.org/10.1021/acs.jcim.1c00096).

**Volkov, Turk, Drizard, Martin, Hoffmann, Gaston-Mathé & Rognan 2022, *J Med Chem*
65(11):7946–7958, DOI
[10.1021/acs.jmedchem.2c00487](https://doi.org/10.1021/acs.jmedchem.2c00487)** (✅ PMID
35608179, re-verified — this settles the DOI that §B.3 flagged as uncertain): *"an explicit
description of protein-ligand noncovalent interactions does not provide any advantage with
respect to ligand or protein descriptors"*, and *"simple models inferring binding affinities
from the closest ligands or proteins in the training set already exhibit good performances,
suggesting that **memorization largely dominates true learning in the deep neural networks**."*

#### The classifier-learns-target-identity version

**Chen, Cruz, Ramsey, Dickson, Duca, Hornak, Koes & Kurtzman 2019, *PLoS ONE* 14(8):e0220113,
DOI [10.1371/journal.pone.0220113](https://doi.org/10.1371/journal.pone.0220113)** (✅ PMID
31430292, re-verified) is the cleanest demonstration, and **its controls are exactly the
controls to copy:**

- Per-target CNNs on all 102 DUD-E targets. **Receptor–ligand and ligand-only (receptor
  replaced by a single dummy atom) both reach average AUC 0.98**, correlating at R² = 0.98,
  **mean absolute AUC difference 0.001**; AUC > 0.9 on all 102 targets for both.
- Taking the *trained* receptor–ligand model and removing the receptor **at test time**:
  0.98 → 0.98, mean difference **0.0006**.
- A plain **1-NN fingerprint KNN** reached AUC > 0.90 on **97/102** targets.
- Models trained on **5 actives and 5 decoys** still scored high AUCs.
- **Cross-target:** 74 targets scored AUC > 0.9 under a model trained on *other targets'*
  ligands. Functionally unrelated targets: mean AUC **0.618 ± 0.176**.
- ⭐ **The decisive control — the "Actives as Decoys" (AD) set**, where DUD-E decoys are
  replaced by actives of other targets: functionally different targets fall to **mean AUC
  0.500 (chance)**, and a 10-target multi-target model that scored **0.80 ± 0.13** on 92
  held-out DUD-E targets **fell to chance**.
- Verdict: *"the high performance of CNN models trained on DUD-E is **not attributable to
  having learned the features of protein-ligand interactions** but rather to analogue and
  decoy bias."* Recommendation: *"DUD-E should be used with extreme caution... and ideally
  should be avoided."*

Reference numbers from the same paper: **AutoDock Vina mean AUC 0.725** on DUD-E; gnina 0.709;
Pafnucy 0.632.

#### ⭐ The controls to run — be exact

Run **all** of these, report every number, and present them as **one table beside the headline
result.** Each catches a different failure.

**A. Input ablations** (retrain from scratch each time, same hyperparameter budget):

1. **Ligand-only** — delete all protein atoms/nodes/sequence. **Pass criterion: materially
   worse than the full model.** If it is within ~0.1 log units / ~0.03 R², the model is a
   ligand QSAR with extra steps.
2. **Protein-only** — delete the ligand. Catches target-identity lookup (Yang 2020's 0.84).
3. **Wrong-receptor / scrambled pocket** — score each ligand against a *different* target's
   pocket. A real interaction model must degrade.
4. **Pose-scrambled** — re-score with the ligand randomly rotated/translated, or with a decoy
   pose. **Sensitivity to pose is a necessary condition for having learned geometry.**

**B. Memorisation baselines** — not ablations, **competitors you must beat**:

5. **k-NN(5) on complex similarity** — predict the mean label of the 5 most similar training
   complexes. Graber's version gets **R = 0.716**.
6. **k-NN(5) on ligand similarity alone** — **R = 0.707**.
7. **Per-target mean predictor** — the "learned target identity" null on multi-target data.
8. **Global mean predictor** — Graber et al. put this as the bottom row of their RMSE table;
   so should we.

**C. Label controls:**

9. **y-randomisation with the entire pipeline inside the permutation loop** (§D.6).
10. **Shuffled pairing** — keep both proteins and ligands but permute which goes with which,
    retaining the label distribution. Isolates "did it learn the pairing" from "did it learn
    the marginals".

**D. Leakage diagnostics** (report alongside every split):

11. **Train–test nearest-neighbour similarity distribution** — ECFP4 Tanimoto, and for
    complexes the TM-score / Tanimoto / pocket-RMSD triple. Report the median and the **% above
    0.6 and above 0.9**.
12. **Exact- and near-duplicate audit on standardised structures including stereochemistry and
    tautomer** — §D.6 explains why canonical-SMILES matching is not enough.
13. **Independent-subset score** — report the metric on the leakage-free subset separately.
    Graber's **144/285** CASF-2016 split is ready-made.

**E. Decoy-bias controls** (screening/classification):

14. **Actives-as-decoys** (Chen 2019's AD set).
15. **Re-generated decoys** — regenerate with [DeepCoy](https://github.com/oxpig/DeepCoy) and
    re-measure. On DUD-E that improves the DOE score from **0.166 → 0.032** and drops Vina's
    ROC-AUC **0.70 → 0.63** (Imrie, Bradley & Deane 2021, *Bioinformatics* 37(15):2134–2141,
    DOI [10.1093/bioinformatics/btab080](https://doi.org/10.1093/bioinformatics/btab080)).

---

### D.4 Screening benchmarks, and which metric each one speaks

See §B.6–B.7 for DUD-E, DEKOIS 2.0 and LIT-PCBA composition and downloads. Two additions:

**MUV** — Rohrer & Baumann 2009, *JCIM* 49(2):169–184, DOI
[10.1021/ci8002649](https://doi.org/10.1021/ci8002649) (✅ PMID 19434821). ✅ **17 targets, not
18** (a figure Réau et al. 2018 get wrong; the canonical MoleculeNet CSV has exactly 17
`MUV-xxx` columns). Design target: **30 actives and 15,000 decoys per target (500 decoys per
active)**. Measured from the DeepChem CSV: **93,087 unique compounds**, 17 tasks, **24–30
actives per task**, **489 active labels / 249,397 inactive labels**, **positive rate 0.20%**.
⚠️ Each compound is labelled for only ~2.7 of 17 tasks, so an all-tasks matrix is ~84%
missing.

The design idea is worth understanding because nothing else does it: distances are computed in
a space of simple 1-D descriptors, then two spatial point-pattern statistics — **G(t)**, the
nearest-neighbour function (self-similarity *within* the actives) and **F(t)**, the
empty-space function (separation *between* actives and decoys) — are forced to satisfy
**ΣG ≈ ΣF**, so the actives form a random-like point pattern *embedded in* the decoys. That
removes **both** artificial enrichment and analogue bias by construction. **It is so hard, and
so active-poor, that it is often unusable for ML:** best reported **AUC-PRC 0.184**; graph
conv / Weave only **0.109**. ⚠️ **Quote the PRC-AUC, not the ROC-AUC** — ROC-AUC on MUV
commonly looks like 0.6–0.8 while the top of the list is useless.

**Pose and structure successors:**

- **PoseBusters** — Buttenschoen, Morris & Deane 2024, *Chem Sci* 15(9):3130–3139, DOI
  [10.1039/d3sc04185a](https://doi.org/10.1039/d3sc04185a) (✅ PMID 38425520). ⚠️ **Astex
  Diverse = 85 cases; PoseBusters Benchmark = 428 as released, but the journal paper reports on
  a 308-structure subset** after crystal contacts were found during review. **Cite 428 for the
  data, 308 for the paper's numbers** — this is routinely conflated. All complexes released
  2021+, so absent from PDBbind v2020. `pip install posebusters`.
- **DockGen** ([arXiv:2402.18396](https://arxiv.org/abs/2402.18396), no DOI) — splits by
  **ECOD t-group of the ligand-contacting domain**. Top-1 RMSD<2Å (PDBbind / DockGen-full /
  DockGen-clusters): GNINA 22.9 / 14.3 / 9.4; DiffDock 35.0 / 7.1 / 6.1; DiffDock-L 43.0 /
  22.6 / 27.6. **Note DiffDock falls from 35.0 to 6.1.**
- **Runs N' Poses** — Škrinjar, Eberhardt, Studer, Tauriello, Schwede & Durairaj 2026, *Nat
  Struct Mol Biol* 33(5):782–794, DOI
  [10.1038/s41594-026-01797-5](https://doi.org/10.1038/s41594-026-01797-5): 2,600
  post-training-cutoff systems; *"current cofolding approaches largely **memorize ligand poses
  from their training data**."*

#### Metrics: which is native, and the pitfall of each

| Benchmark | Native metric |
|---|---|
| DUD-E | **adjusted logAUC₀.₀₀₁**, EF1 (decoy-based) |
| DEKOIS 2.0 | **pROC-AUC** |
| MUV | **AUC-PRC** |
| LIT-PCBA | **EF1%** |
| DUDE-Z | enrichment **with bootstrapped CIs** |
| PoseBusters | **% top-1 RMSD<2Å *and* PB-valid** (joint) |
| CASF-2016 | Rp / ρ / SR1 / EF1% |
| CASP16 ligand | **LDDT-PLI**; **Kendall τ** for affinity |

Definitions people get wrong:

- **adjusted logAUC₀.₀₀₁** (DUD-E) spans three decades and **subtracts the random curve's
  14.462%**, so **random = 0%** and negative is worse than random. Anyone quoting "logAUC 14%"
  as random is using the unadjusted form.
- ❗ **DUD-E's EF1 is "the percent of ligands found when 1% of the *decoys* have been found"** —
  **not** the ordinary list-based EF1%. Decoy-based and list-based EF1% are different numbers;
  **say which you mean.**
- **pROC-AUC** (DEKOIS) has **random = 0**, not 0.5.
- **BEDROC(α=20)** (Truchon & Bayly 2007, *JCIM* 47(2):488–508, DOI
  [10.1021/ci600426e](https://doi.org/10.1021/ci600426e)) puts **~80% of the weight in the top
  8%**. **Saturation is the trap** — its maximum is < 1 and depends on n/N, so BEDROC is **not
  comparable across benchmarks with different active:decoy ratios** (DUD-E ~1:62, DEKOIS 1:30,
  MUV 1:500, LIT-PCBA 1:4,000–1:18,000).
- **EF at x%** — *"not sensitive to ranking performance before and after the cutoff"*: an
  active at rank 1 and at rank 99/100 score identically. Capped at 1/x and by n/N. Swings
  wildly with few actives — **LIT-PCBA's ADRB2 has 17 actives, so EF1% moves in ~6%-of-actives
  steps.** ❗ **Always with bootstrapped CIs.**
- **AUC-ROC** — integrates the 90% of the list nobody will buy, and is the metric most inflated
  by analogue/decoy bias. **Sanity floor only.**
- **AUC-PRC** — prevalence-dependent, so not comparable across targets with different hit
  rates; random baseline = the positive rate (0.002 for MUV).
- **% RMSD<2Å** — a hard threshold hides the distribution, and **a sub-2Å pose can be
  physically impossible.** Report **"PB-valid AND <2Å"** as one joint number.
- **LDDT-PLI** — contact-based, so more forgiving of a globally shifted pose than RMSD. And
  note CASP16's affinity ceiling: **τ = 0.42 against τ_max ≈ 0.73 is 58% of attainable**, not
  42% of perfect.

⚠️ **LIT-PCBA's class imbalance makes per-target metrics fragile:** **ADRB2 has 17 actives
against 311,748 inactives (0.005%)**; IDH1 39 vs 358,757; ESR_ago 13 vs 4,378. Several targets
have fewer than 30 actives — **plan to bootstrap, and never report a bare mean EF1% across
targets**, because ALDH1 (5,363 actives) and ADRB2 (17) cannot be averaged.

---

### D.5 Activity cliffs / MoleculeACE

**van Tilborg, D.; Alenicheva, A.; Grisoni, F.** "Exposing the Limitations of Molecular
Machine Learning with Activity Cliffs." *J. Chem. Inf. Model.* **2022**, 62(23), 5938–5951.
DOI [10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073). ✅ PMID 36456532
(re-verified). Open access.

> ❗ **Two published corrections, and the second is substantive — I verified both exist.**
> 1. DOI [10.1021/acs.jcim.3c00423](https://doi.org/10.1021/acs.jcim.3c00423) (2023, ✅ PMID
>    36995229) — early-stopping wording.
> 2. DOI [10.1021/acs.jcim.3c01576](https://doi.org/10.1021/acs.jcim.3c01576) (2023-10-18, ✅
>    PMID 37851546) — **"a software bug affected the splitting between training and test sets,
>    due to incorrect labeling of certain activity cliff pairs."** Table 1 and Figures 3–5 were
>    reprinted after full retraining. The conclusions held, but cliff percentages moved
>    materially (factor X 44→48%, ABL1 32→42%, JAK2 12→17%, GHSR 48→52%).
>
> **Any MoleculeACE baseline published before October 2023 used the buggy splits and is not
> strictly comparable.** Use the corrected splits and say so.

**Construction:** ChEMBL **v29**, *Homo sapiens*; **30 targets**; **48,707 molecules** summed
per target (**35,632 unique** — molecules recur across targets); **Ki and EC50 only** (22 Ki
datasets, 8 EC50), **IC50 not used**; labels as pKi/pEC50; sizes **615 (JAK1) to 3,657
(D3R)**; cliff fraction **7%–52%** as published, ~9–54% corrected.

> ✅ **Independent cross-check:** summing the 30 `scikit-fingerprints/MoleculeACE_*` HF mirrors
> (§B.10) gives **48,714 rows** — matching the paper's 48,707 to within 7 records. The mirrors
> also carry `license: cc-by-sa-3.0`, correctly inherited from ChEMBL. And **`chembl2835_ki`
> is JAK1 (615 rows — the paper's smallest set, exactly)** while `chembl235_ec50` is PPARγ
> (2,349 rows) — both tier-1 targets from section A, so MoleculeACE doubles as a
> target-specific cliff test for us.

**How a cliff is defined — three criteria combined by OR** (the paper calls this a *"soft
consensus"*): a pair is "highly similar" if **at least one** of (1) ECFP (1024 bits, r=2)
Tanimoto ≥ 0.9; (2) Tanimoto ≥ 0.9 on the ECFP of the Bemis–Murcko **scaffold**; (3)
**Levenshtein** SMILES-string similarity ≥ 0.9. Then the potency criterion: **>10-fold (>1 log
unit)** difference. A molecule in ≥1 such pair is an **activity-cliff compound** (mean 2.7 ±
0.9 partners). **Chirality cliffs excluded.** MMPs were evaluated but not used for the main
definition (they nearly doubled the cliff count while covering 86.6% of the OR-definition's
cliffs); MMPs *are* supported as an option in the package.

**Splitting:** per target, spectral clustering on ECFP into **5 clusters**, then **80/20
stratified by the binary cliff label within each cluster** — cliffs are deliberately *kept* in
test, proportionally. **The splits are fixed and shipped**, so every paper uses the same
partition. Consequence worth noting: even so, mean maximal test→train similarity is **0.80 ±
0.03 (substructure) / 0.93 ± 0.02 (scaffold) / 0.95 ± 0.01 (SMILES)**.

**Metrics: RMSE** over all test molecules, and **RMSE_cliff** = RMSE restricted to test
molecules in ≥1 cliff pair. `calc_rmse()` and `calc_cliff_rmse()`.

**Headline results** (24 strategies × 30 datasets = **720 models**; Bayesian/GP search, 50
configs, 5-fold CV):

| Quantity | Value |
|---|---|
| RMSE_cliff range, traditional ML | **0.62 – 1.60** log units |
| RMSE range, traditional ML | 0.41 – 1.35 |
| RMSE_cliff range, deep learning | 0.68 – 1.44 |
| Mean Pearson r(RMSE, RMSE_cliff) | **0.81** (r > 0.70 on 25/30 datasets) |
| Max over-estimation if you look at RMSE only | **up to 0.54 log units** |
| SVM+ECFP mean (RMSE_cliff − RMSE) | **0.094** (worst 0.39 on JAK1) |
| Correlation of (RMSE_cliff − RMSE) with training-set size | **r = −0.15** (i.e. none) |

**The finding: SVM + ECFP was the best single method on cliffs on average**, with **no
statistically significant difference between SVM, GBM and RF when all used ECFP** — one
indistinguishable top tier. ❗ **The descriptor mattered more than the algorithm** (p < 0.05):
ECFP best, MACCS noisier, **WHIM 3D descriptors and the 11 physicochemical properties
considerably worse.** Among deep models, **MLP+ECFP beat every SMILES- or graph-based
network**; SMILES beat graphs; **LSTM was the best deep model, and ChemBERTa pretrained on 10 M
SMILES did not beat an LSTM pretrained on 36,281.** GNN self-supervised transfer learning gave
no improvement and was dropped.

⚠️ The paper does **not** print a single averaged RMSE/RMSE_cliff pair. Reproductions give
SVM+ECFP **0.671 / 0.742** (+10.6% relative), and across modern methods a consistent **+0.10
to +0.11 log units (+14–16%) cliff penalty for every method.**

**Package:** [molML/MoleculeACE](https://github.com/molML/MoleculeACE), `pip install
MoleculeACE` (v3.0.1, MIT) — ships the 30 datasets, the fixed splits, the per-molecule
`cliff_mols` flag, and pre-optimised hyperparameters for all 24 strategies × 30 datasets. For
third-party models, `calc_cliff_rmse(..., similarity=0.9, potency_fold=10)` computes cliffs on
the fly.

#### The 2023–2026 follow-ups

**Current state of the leaderboard** — Banaszewski & Fitzgibbon 2026 (*Monroe*,
[arXiv:2608.18982](https://arxiv.org/abs/2608.18982)), mean over the 30 MoleculeACE tasks:

| Model | Win % | RMSE ↓ | RMSE_cliff ↓ |
|---|---|---|---|
| MolFormer | 0.0 | 0.748 ± 0.018 | 0.854 ± 0.027 |
| MiniMol | 10.0 | 0.703 ± 0.018 | 0.794 ± 0.031 |
| **GBM (count-ECFP4 + RDKit2D → LightGBM, tuned)** | 16.7 | **0.680 ± 0.020** | **0.769 ± 0.023** |
| CheMeleon | 36.7 | 0.662 ± 0.016 | 0.760 ± 0.026 |
| MiniMol_PFN | 70.0 | 0.642 ± 0.004 | 0.760 ± 0.008 |
| CheMeleon_PFN | 80.0 | 0.641 ± 0.013 | 0.741 ± 0.020 |
| **Monroe** | 86.7 | **0.629 ± 0.004** | **0.737 ± 0.006** |

❗ **The tuned LightGBM-on-ECFP4 classical baseline beats MolFormer and MiniMol outright and
sits within 0.02 log units of CheMeleon.** And the authors' own verdict: *"the cliff error still
sits well above the overall error, as it does for every method, indicating that the benchmark
is not solved."*

- **Burns, Zalte, Abreu, Sieg, Feldmann, Mathea & Green 2026, *JCIM* 66(17):10692–10701, DOI
  [10.1021/acs.jcim.6c01546](https://doi.org/10.1021/acs.jcim.6c01546)** (CheMeleon). 58
  datasets, **with no hyperparameter optimisation for any model on any dataset** — which
  removes the "they tuned the classical models harder" objection. Win rates on MoleculeACE
  (whole test set / cliff subset): **CheMeleon 97% / 100%**; **Random Forest (Morgan) 50% /
  67%**; minimol 40% / 67%; MoLFormer 17% / 33%; **Chemprop 0% / 13%**; MolCLR 0% / 7%.
  Verbatim: *"Many-parameter, well-known foundation models such as MoLFormer and MolCLR fare
  surprisingly poorly... Chemprop falls below all but one of the tested methods, including the
  baseline Random Forest."*
- **Zhao et al. 2026, *J Cheminform* 18:95, DOI
  [10.1186/s13321-026-01217-2](https://doi.org/10.1186/s13321-026-01217-2)** — R² on
  MoleculeACE: KPGT 0.678, **XGBoost+MorganCount 0.668**, TabPFNv2+RDKit2D 0.649; GNNs from
  scratch "performed poorly". **Cliff split: XGBoost R² 0.51 on cliff vs 0.69 on non-cliff.**
  Activity-cliff-pair recall: MOLMCL 0.130, KPGT 0.128, Uni-Mol 0.098 — **everything under
  0.14.** Verdict: *"neither end-to-end deep learning nor advanced tabular foundation models
  have fundamentally solved the challenge of activity cliff identification."*
- **Cliff-aware losses** — Cui, Su, Zhang, Velez-Arce, Wang, Shi, Zhang, Wu, Chen, Zitnik &
  Shen 2026, *Nat Commun* 17:9376, DOI
  [10.1038/s41467-026-75713-2](https://doi.org/10.1038/s41467-026-75713-2) (**ACANet**):
  regression loss + **triplet soft-margin loss**, backbone-agnostic. **RMSE_all 0.646 vs
  SVM+ECFP's 0.671; RMSE_cliff 0.711 vs 0.742**; wins 23/30 and 21/30. Ships a **Molecular
  Property Cliff** task in TDC ([tdcommons.ai/single_pred_tasks/MPC](https://tdcommons.ai/single_pred_tasks/MPC/)).
  The counterweight: Surendran & Miranda-Quintana 2026 (bioRxiv DOI
  [10.64898/2026.09.08.750257](https://doi.org/10.64898/2026.09.08.750257)) find contrastive
  losses improve embedding separation but *"their impact on absolute predictive accuracy
  remains highly dataset dependent"* — **the gains are interpretability, not RMSE.**
- **Is it a tuning or data-size artefact?** ❌ No paper asserts that directly. Adjacent:
  Deng et al. 2023 (*Nat Commun* 14:6395 — the 62,820-model study from §B.9) concluded
  *"dataset size is essential for representation learning models to excel"* — i.e. **the
  deficit is a low-data phenomenon.** Tetko, van Deursen & Godin 2024 (*J Cheminform* 16:139,
  DOI [10.1186/s13321-024-00934-w](https://doi.org/10.1186/s13321-024-00934-w)) show **HPO
  itself overfits on small sets** — which is why CheMeleon ran with no HPO, and the ranking
  did **not** qualitatively flip. And Praski, Adamczyk & Czech 2025
  ([arXiv:2508.06199](https://arxiv.org/abs/2508.06199)), 25 pretrained models × 25 datasets
  with hierarchical Bayesian testing: *"nearly all neural models show negligible or no
  improvement over the baseline ECFP molecular fingerprint."*
- **Independent replication, different cliff definition.** Tamura, Miyao & Bajorath 2023,
  *J Cheminform* 15:4, DOI
  [10.1186/s13321-022-00676-7](https://doi.org/10.1186/s13321-022-00676-7) — 100 activity
  classes, MMP-based: *"prediction accuracy did not scale with methodological complexity but
  was significantly influenced by **memorization of compounds shared by different ACs or
  nonACs**"*; *"no detectable advantage of deep learning"*. And the sharpest statement in the
  literature — Abe, Miyao & Bajorath 2026, *J Cheminform* 18:72, DOI
  [10.1186/s13321-026-01210-9](https://doi.org/10.1186/s13321-026-01210-9): ❗ *"**in the
  absence of data leakage from training to test sets, AC prediction accuracy based on chemical
  structure is low or moderate at best.**"*
- ⭐ **Where the deficit actually lives** — Özçelik & Grisoni 2025, *Digital Discovery*
  4(2):316–325, DOI [10.1039/d4dd00311j](https://doi.org/10.1039/d4dd00311j): in **regression
  with structurally similar test molecules, "neither deep models nor XGBoost is statistically
  superior"**; in **classification with dissimilar test molecules (Tanimoto < 0.4), the CNN
  beat XGBoost on all datasets.** **So the deep-learning deficit is specific to the
  similar-test-set regression regime — exactly where cliffs live. The model class we want
  depends on whether our deployment is lead-op (cliffs → descriptors win) or hit-ID (novelty →
  deep models win).** This is the most actionable nuance in the cliff literature.

⚠️ ❌ **No paper titled "MoleculeACE revisited" exists.** And **MoleculeACE "win rate" is
pool-relative** — CheMeleon's reported 97% becomes 36.7% in a different comparison pool.
**Never quote a win rate across papers.**

**The general "does DL beat RF/SVM" reference:** Jiang, Wu, Hsieh, Chen, Liao, Wang, Shen,
Cao, Wu & Hou 2021, *J Cheminform* 13:12, DOI
[10.1186/s13321-020-00479-8](https://doi.org/10.1186/s13321-020-00479-8) — 11 datasets, 8
algorithms: *"**on average the descriptor-based models outperform the graph-based models**...
**SVM generally achieves the best predictions for the regression tasks.**... some of the
graph-based models... can yield outstanding performance **for a fraction of larger or
multi-task datasets**... **XGBoost and RF... only need a few seconds to train**."* The nuance:
graph models win **only on larger or multi-task datasets** — the same data-size boundary Deng
et al. later quantified.

---

### D.6 Other honesty issues

#### Assay heterogeneity when pooling ChEMBL

❗ **What `pchembl_value` actually is, and two consequences almost everyone misses.** Per
ChEMBL's docs, pChEMBL = −log(molar IC50, XC50, EC50, AC50, Ki, Kd or Potency), computed
**only** when `standard_relation = '='`, `standard_units = 'nM'`, `standard_value > 0` and
`data_validity_comment` is null or 'Manually validated'. Therefore:

1. **`pchembl_value` *is* the act of pooling IC50 with Ki with Kd with EC50.** Filtering to
   `pchembl_value IS NOT NULL` does not protect you from assay heterogeneity — **it performs it
   silently and discards the type label.**
2. **`pchembl_value` is null for every censored record** (because of `standard_relation = '='`).
   **The standard "give me pChEMBL" query is also the standard way to silently delete all your
   inactives.**

> ❗ This applies directly to every count in section A, which used exactly that filter. Those
> numbers are correct as "potency records with a pChEMBL value", but they are **not** the full
> activity picture, and the censored records they exclude are the ones a triage model most
> needs to see.

**The noise floor.** Kramer, Kalliokoski, Gedeck & Vulpetti 2012, *J Med Chem*
55(11):5165–5173, DOI [10.1021/jm300131x](https://doi.org/10.1021/jm300131x) — verbatim:
*"**Models that yield errors smaller than the experimental uncertainty are necessarily
overtrained.** ... Careful filtering of the data was required because ChEMBL contains
unit-transcription errors, undifferentiated stereoisomers, and **repeated citations of single
measurements (90% of all pairs)**. The experimental uncertainty is estimated to yield a **mean
error of 0.44 pKi units, a standard deviation of 0.54 pKi units, and a median error of 0.34
pKi units. The maximum possible squared Pearson correlation coefficient (R²) on large data
sets is estimated to be 0.81.**"*

⚠️ **Correction to a common framing:** 0.44 is the **mean absolute error** between independent
measurements of the same pair; **0.54 is the standard deviation** — not "0.54 for mixed". And
the **90%-duplicate finding is the mechanism**: if 90% of apparently independent pairs are
re-citations of one measurement, a naive split puts the identical number on both sides.

**IC50 pooling and conversion.** Kalliokoski, Kramer, Vulpetti & Gedeck 2013, *PLoS ONE*
8(4):e61007, DOI [10.1371/journal.pone.0061007](https://doi.org/10.1371/journal.pone.0061007):
mixed IC50 scatter **σ = 0.68 log units**, Ki σ = 0.47; *"the standard deviation of IC50 data
is only 25% larger than the standard deviation of Ki data, suggesting that mixing IC50 data
from different assays... only adds a moderate amount of noise."* Mean pKi−pIC50 offset **0.355
log units** → *"a **Ki–IC50 conversion factor of 2** was found to be the most reasonable"*;
**R² between pKi and pIC50 only 0.46.** ⚠️ The 0.68 and 0.47 figures are on different footings
(pairwise-difference vs per-measurement SD) — **do not divide one by the other.**

⭐ **The paper that reframes this for ML.** Landrum & Riniker 2024, *"Combining IC50 or Ki
Values from Different Sources Is a Source of Significant Noise"*, *JCIM* 64(5):1560–1567, DOI
[10.1021/acs.jcim.4c00049](https://doi.org/10.1021/acs.jcim.4c00049) — same compound, two
different assays, same target, ChEMBL 32:

| IC50 | % differ > 0.3 log | % differ > 1.0 log | Kendall τ | R² |
|---|---|---|---|---|
| minimal curation | **64%** | **27%** | 0.51 | 0.31 |
| maximal curation | **48%** | **13%** | 0.71 | 0.63 |

| Ki | % > 0.3 log | % > 1.0 log | Kendall τ |
|---|---|---|---|
| minimal curation | **67%** | **30%** | 0.43 |
| maximal curation | **58%** | **25%** | 0.67 |

⭐ **Their "maximal curation" recipe is the operational answer to "what do I filter on":**
require identical `assay_type`, `assay_organism`, `assay_category`, `assay_tax_id`,
`assay_strain`, `assay_tissue`, `assay_cell_type`, `assay_subcellular_fraction`, `bao_format`
and `variant_id`; `confidence_score = 9`; document-associated assays only; drop descriptions
containing "mutant"/"mutation"/"variant"; **one assay per document per target**; **drop assays
with > 100 compounds** (review/aggregation records). Code:
[rinikerlab/overlapping_assays](https://github.com/rinikerlab/overlapping_assays).

> **State the tension honestly.** Kalliokoski 2013 says mixing IC50 adds *"a moderate amount of
> noise"*; Landrum & Riniker 2024 says a quarter to a third of such pairs disagree by **more
> than a log unit**. Both are right (SD vs tail fraction) — but **the 2024 framing is the one
> that matters for ML, because a 27% tail of >1-log disagreements is a 27% label-corruption
> rate.**

**Practical rule.** Keep `standard_type` as a column and report per-type performance; pool
**Ki + Kd** freely; fold **IC50 into pKi with a +0.35 log (factor 2) offset**, accepting ~0.68
log residual scatter; **keep EC50 separate** (functional readout; ❌ no published offset
analogous to the Ki/IC50 factor of 2 exists); never pool biochemical with cell-based;
`confidence_score = 9`; de-duplicate **at the measurement level before splitting**; and
**report the noise floor next to the RMSE.**

Background: Papadatos, Gaulton, Hersey & Overington 2015, *J Comput Aided Mol Des*
29(9):885–896, DOI [10.1007/s10822-015-9860-5](https://doi.org/10.1007/s10822-015-9860-5) —
ChEMBL standardises **133 concentration units** to nM; log conversion and pChEMBL calculation
touched **20.7%** of records; *"data preparation and filtering is equally, if not more
important, than the analysis or model itself"*, and assay annotation functions *"as a removal
filter for clearly incomparable data rather than a guarantee of comparability."* Also
Tiikkainen, Bellis, Light & Franke 2013, *JCIM* 53(10):2499–2505, DOI
[10.1021/ci400099q](https://doi.org/10.1021/ci400099q): error-rate ordering is **structure
representation > target assignment > activity value > activity type**. ⚠️ ❌ The commonly
quoted "~5% error rate" for this paper is **not in the abstract — do not cite a number for
it.**

#### Censored data

**How much there is:** on 15 internal AstraZeneca assays, **0–63% of labels censored** per
assay, overwhelmingly left-censored in pActivity terms (Svensson, Friesacher, Winiwarter,
Mervin, Arany & Engkvist 2025, *Artificial Intelligence in the Life Sciences* 7:100128, DOI
[10.1016/j.ailsci.2025.100128](https://doi.org/10.1016/j.ailsci.2025.100128)).

**Three options:**

- **(a) Drop them** — the default behaviour of any `standard_relation = '='` or
  `pchembl_value IS NOT NULL` filter. ❗ **The bias is structural and one-directional: you
  remove the weak and inactive compounds, truncating the label distribution from below. The
  model never sees what inactivity looks like, and will confidently predict mid-range potency
  for compounds whose true answer is ">100 µM" — which is precisely the prediction a triage
  pipeline acts on. For a triage model this is the single worst curation decision available.**
- **(b) Impute at the threshold** — cheap, but creates an artificial mass spike that biases the
  mean and, worse, the variance: it tells the model those compounds are *exactly* 30 µM when
  all you know is ≥30 µM.
- **(c) Censored / Tobit likelihood** — ⭐ what the literature recommends, and it is a **loss
  function change, not an architecture change.** Observed labels contribute the usual Gaussian
  log-density; censored labels contribute the log of the Gaussian **CDF** over the region the
  true value is known to lie in. For non-probabilistic ensembles, a one-sided "CensoredMSE"
  penalises the prediction only if it lands on the wrong side of the threshold.

Svensson et al.'s results (5 temporal folds by experiment date; MSE, NLL, **ENCE**): models
trained **with** censored labels **significantly outperformed** the same models trained on
observed data only, *"particularly for epistemic uncertainty estimation"*, with the advantage
holding **above ~35% censoring**. And a finding worth repeating: ❗ **aleatoric uncertainty did
not correlate with observed replicate error — it tracked activity-cliff structure instead.**
So "aleatoric = noise floor" is not what these models learn.

**Classification is a legitimate fallback**, since a censored label is perfectly informative
for a binary task at or below the censoring threshold. The cost is discarding the quantitative
signal from the observed half.

#### Near-duplicate / analogue leakage

**AVE** — Wallach & Heifets 2018, *JCIM* 58(5):916–932, DOI
[10.1021/acs.jcim.7b00403](https://doi.org/10.1021/acs.jcim.7b00403). Formula:
**AVE = (AA − AI) + (II − IA)**, areas under the cumulative NN-similarity distributions from
validation→training, n = 100 neighbours; close to zero = "fair". Davis, McLoughlin, Allen &
Ellingson 2020 (ICCS, DOI
[10.1007/978-3-030-50420-5_44](https://doi.org/10.1007/978-3-030-50420-5_44)) measure
**Pearson r = 0.80 between mean AVE bias and PR-AUC**, and on AVE-minimised splits **~26% of
models scored below random.**

⭐ **The critique, and it matters.** Sundar & Colwell 2020, *JCIM* 60(1):56–62, DOI
[10.1021/acs.jcim.9b00415](https://doi.org/10.1021/acs.jcim.9b00415) introduce **far-AUC** (AUC
on ligands at Jaccard distance ≥ 0.4 from every training ligand) and find that **both MUV and
AVE debiasing typically *reduce* far-AUC**: *"model performance measured after debiasing is not
representative of the ability of a model to generalize"* — **because debiasing works by
deleting training data.**

> **The honest synthesis: AVE is a good diagnostic and a bad remedy.** Measure the bias and
> report it; evaluate on a genuinely distant held-out set; **do not subsample the training set
> to drive a bias metric to zero.**

⭐ **How much of a benchmark test set is actually novel — measured.** Schuh, Daniluk & Sieber
2026, *"Auditing widely used biomolecular benchmarks reveals systematic data
inconsistencies"*, *Chem Sci* 17(36):17455–17467, DOI
[10.1039/d6sc01799a](https://doi.org/10.1039/d6sc01799a), audited **51 benchmark
configurations** (Polaris 7, **TDC 22**, MoleculeNet 9, DTI 13). Averaged over 28 datasets:

| Test-set category | Share |
|---|---|
| audit-clean | **59.2%** |
| shares a Bemis–Murcko scaffold with training | **31.5%** |
| near-training analogue | **7.1%** |
| **exact train/test molecular leak** | **1.3%** |
| duplicate or conflicting labels | 0.9% |

Mean cross-split NN similarity: **Polaris 0.494**, **TDC 0.559**, **MoleculeNet 0.589**, **DTI
0.779**. Worst TDC cross-split contamination **4.95% (`Clearance_Hepatocyte_AZ`)**; PDBbind
"Leak Proof" splits showed **no** exact contamination. And the noise-injection experiment that
quantifies what leakage buys you (41 datasets):

| Perturbation | Δ ROC-AUC | Δ RMSE |
|---|---|---|
| label conflicts | −0.317 | +6.53 |
| random label noise | −0.329 | +4.97 |
| activity-cliff perturbation | −0.023 | +3.61 |
| **train–test contamination** | **+0.177 (inflation)** | **−11.37 (inflation)** |

Leaderboard stability: the original top model stayed rank 1 in 23/28 datasets on audit-clean
panels, but **Kendall τ rank stability fell to 0.54** at observed near-train composition and
**0.33** on conflict-enriched panels.

> ⚠️ **Correction to the brief's framing:** this is **not** a "Polaris critique of
> MoleculeNet". Polaris is audited **alongside** the others and comes out **best** on leakage
> (0.494). [polarishub.io](https://polarishub.io) is better-curated, not flawless.

⭐⭐ **And the TDC-specific audit — this is the most important single finding in section D for
anyone planning to quote a TDC leaderboard.** Koleiev, Stratiichuk, Shevchuk, Melnychenko,
Nyporko, Todoryshyn, Husak, Starosyla et al. 2026, *JCIM* 66(14):8045–8058, DOI
[10.1021/acs.jcim.6c00819](https://doi.org/10.1021/acs.jcim.6c00819):

- 10 architectures from the top-3 of all 22 TDC ADMET leaderboards, audited as an end user with
  only the public artefacts. ❗ **Only 3 of 10 (CaliciBoost, MapLight, MapLight+GNN) passed all
  stages and reproduced their reported performance.**
- **Leakage found in 3 models.** **MiniMol**: molecules highly similar to the TDC test set
  survived pretraining filtering because ❗ **canonical-SMILES matching misses stereoisomers and
  tautomers.** **GradientBoost and XGBoost**: tuning-stage validation sets overlapped the public
  test sets via an incorrect resplit with a hard-coded `random_state=42`. Fixing it moved
  **GradientBoost 1st → 3rd** and **XGBoost 2nd → 8th**.
- ⭐ **The deliberate-overfitting experiment:** honest in-house models reached top-3 on **2/22**
  endpoints; **the same models tuned against the public test set reached top-3 on 10/22,
  including 5 first places** (`bbb_martins` 8→1, `clearance_microsome_az` 14→1, `ames` 13→1).
  ❗ **Leaderboard position is, to a first approximation, a measurement of how hard someone
  looked at the test set.**

> ❗ **This directly qualifies §B.8's and §C.1's leaderboard tables.** MiniMol — the model that
> tops HIA, BBB, Lipophilicity, AqSol, DILI and more — is one of the three with documented
> leakage. Treat those tables as **"numbers published"**, not **"performance achieved"**, and
> note that the implausible `Bioavailability_Ma` 0.942 flagged in §B.8 now has a mechanism.

**The practical consequence for deduplication:** matching on canonical SMILES (or even the
full InChIKey) **under-detects** leakage when stereo/tautomer annotation is inconsistent — which
it routinely is. Kramer 2012 lists "undifferentiated stereoisomers" as one of three defects he
had to filter, and Walters' MoleculeNet BACE audit found **71% of molecules have at least one
undefined stereocentre** (222 with three or more). ⭐ **Dedupe on a standardised structure
including stereo and tautomer.**

**Should you dedupe by scaffold?** Only if scaffold hopping is the deployment question — and
then say that is what you are measuring. Two molecules differing by a single atom can have
**different Bemis–Murcko scaffolds while sharing 0.66 Tanimoto**, which the Schuh audit's
**31.5%** scaffold-sharing figure confirms empirically *on benchmarks that already use scaffold
splits.*

#### y-randomisation

**Golbraikh & Tropsha 2002, "Beware of q²!", *J Mol Graph Model* 20(4):269–276, DOI
[10.1016/S1093-3263(01)00123-1](https://doi.org/10.1016/S1093-3263(01)00123-1)** — verbatim:
*"Often, a high value of this statistical characteristic (q² > 0.5) is considered as a proof of
the high predictive ability of the model. In this paper, we show that **this assumption is
generally incorrect.** ... **No correlation between the values of q² for the training set and
predictive ability for the test set was found for any of the datasets.** ... **We emphasize
that the external validation is the only way to establish a reliable QSAR model.**"*

**Rücker, Rücker & Meringer 2007, *JCIM* 47(6):2345–2357, DOI
[10.1021/ci700157b](https://doi.org/10.1021/ci700157b)** test the four combinations of
(original vs permuted response) × (original vs random pseudo-descriptors) and find **variants
using random pseudo-descriptors produce higher mean random r²** than those using the original
descriptors, because of intercorrelation among real descriptors. **The pseudo-descriptor null
is the stricter one.**

**What to report:**

1. ❗ **The whole pipeline inside the permutation loop** — shuffle y, then redo descriptor
   selection, hyperparameter search, scaling and fitting from scratch. **Shuffling only at the
   final fit is the standard way to get a reassuring, meaningless null.**
2. **The full null distribution**, not pass/fail: number of permutations (≥100; ≥1000 for a
   usable p-value), **mean and maximum** random metric, empirical p-value.
3. **Both Rücker variants** where possible.
4. **External-test metrics, not q².** Golbraikh & Tropsha's criteria: q² > 0.5 (necessary
   only), external R² > 0.6, (R² − R₀²)/R² < 0.1, and 0.85 ≤ k ≤ 1.15.
5. **Classification: permute within strata** so prevalence is preserved.
6. ❗ **y-randomisation and leakage diagnostics are orthogonal and you need both** — a
   scrambling control catches a model fitting noise; it does **not** catch a model memorising
   near-duplicates, **because the duplicate's label is shuffled too.**

#### Applicability domain

**The five OECD principles** (adopted Nov 2004; OECD Series on Testing and Assessment No. 69):
(1) a defined endpoint; (2) an unambiguous algorithm; (3) **a defined domain of
applicability**; (4) appropriate measures of goodness-of-fit, robustness and predictivity;
(5) a mechanistic interpretation, if possible. ⚠️ Every OECD URL returns 403 to automated
fetch; the wording is quoted via a peer-reviewed secondary source (Czub, Pacławski, Szlęk &
Mendyk 2022, *Pharmaceutics* 14(7):1415, DOI
[10.3390/pharmaceutics14071415](https://doi.org/10.3390/pharmaceutics14071415)).

**Sahigara, Mansouri, Ballabio, Mauri, Consonni & Todeschini 2012, *Molecules* 17(5):4791–4810,
DOI [10.3390/molecules17054791](https://doi.org/10.3390/molecules17054791)** — verbatim: *"**the
reliable predictions are generally limited to query chemicals structurally similar to the
training compounds**... while **defined thresholds contribute significantly to the
extrapolations**."* Methods: range/geometric (bounding box, PCA, convex hull); distance-based
(Euclidean, city-block, Mahalanobis, **leverage/Williams plot**, k-NN with k=5); probability
density. **The Williams-plot warning leverage is h\* = 3p′/n**; plotting standardised residual
vs leverage with h\* and ±3σ separates **structural** outliers (high h, low residual) from
**response** outliers (low h, high residual). Findings: **no single approach is universally
best**; probability-density and convex-hull methods flagged **~28–43%** out-of-domain; the
**95th-percentile** distance threshold gave a reasonable balance; and **some well-predicted
compounds are wrongly excluded** by aggressive ADs.

> ⭐ **The observation worth internalising: an AD metric and a leakage metric are the same
> number read in two directions.** High Tanimoto-to-nearest-training-neighbour means "in
> domain" *and* "possibly leaked". **Report the distribution once and use it for both.**

#### Calibrated uncertainty / conformal prediction

The argument the field has accepted (Norinder, Carlsson, Boyer & Eklund 2014, *JCIM*
54(6):1596–1603, DOI [10.1021/ci5001168](https://doi.org/10.1021/ci5001168)): a heuristic AD
gives a binary in/out label with an arbitrary threshold and no guarantee; **conformal
prediction gives a per-compound interval with a finite-sample coverage guarantee**, which
subsumes the AD's job and is falsifiable.

⚠️ **On "find the exact venue" for Cortés-Ciriano & Bender: it is an RSC book chapter, not a
journal article** — "Concepts and Applications of Conformal Prediction in Computational Drug
Discovery", Chapter 5 in *Artificial Intelligence in Drug Discovery* (RSC Drug Discovery
Series), 2020, pp. 63–101, DOI
[10.1039/9781788016841-00063](https://doi.org/10.1039/9781788016841-00063); preprint
[arXiv:1908.03569](https://arxiv.org/abs/1908.03569). Verbatim: *"at a confidence level of 90%
the true value will be within the predicted confidence intervals in at least 90% of the cases.
**This so called validity of conformal predictors is guaranteed by the robust mathematical
foundation underlying CP.** The versatility of CP relies on its minimal computational
footprint, as it can be easily coupled to any machine learning algorithm at little
computational cost."*

For the imbalanced case: Sun, Carlsson, Ahlberg, Norinder, Engkvist & Chen 2017, *JCIM* 57(7),
DOI [10.1021/acs.jcim.7b00159](https://doi.org/10.1021/acs.jcim.7b00159) — **Mondrian
cross-conformal prediction** on 18 datasets at 1:10 to 1:1000 imbalance *"produces valid
predictions for the minority class"*.

❗ **The headline negative result.** Hirschfeld, Swanson, Yang, Barzilay & Coley 2020, *JCIM*
60(8):3770–3780, DOI
[10.1021/acs.jcim.0c00502](https://doi.org/10.1021/acs.jcim.0c00502) — verbatim: *"**none of
the methods we tested is unequivocally superior to all others, and none produces a particularly
reliable ranking of errors across multiple data sets.** While we believe that these results
show that **existing UQ methods are not sufficient for all common use cases**..."* ⚠️ Their
specific practical recommendation is in the body, which could not be retrieved — **do not state
which method they recommend.** The honest framing: **as of 2020 the best evidence was that
neural-network UQ does not reliably rank its own errors**, which is why conformal prediction —
whose guarantee does not depend on correct error ranking — took over.

**2025–2026.** The most useful recent prescription is Jeliazkova, Kochev, Iliev & Jeliazkov
2026, *Chem Res Toxicol* 39(7):1357–1376, DOI
[10.1021/acs.chemrestox.6c00065](https://doi.org/10.1021/acs.chemrestox.6c00065): >100 VEGA
models plus EPA CompTox (~500,000 chemicals, 22 M predictions). At 90% nominal: regression
empirical coverage **85–95%** (52 models, all but two on target); ⭐ **relative interval width
0.052–1.016 — a 19-fold spread**, which is the single best argument for reporting *efficiency*
alongside validity; coverage stratified by AD index shows **undercoverage at low AD (88.8%)**
and **overcoverage at high AD (99.2%)**; on EPA CompTox, **ρ = −1.00** between relative
interval width and AD index, and **50.3% of predictions landed in the "very low AD" bin** with
mean relative interval width **47% of the training response range**.

Also: Parrondo-Pizarro, Lanini & Rodríguez-Pérez 2026, *JCIM*, DOI
[10.1021/acs.jcim.5c02381](https://doi.org/10.1021/acs.jcim.5c02381) — benchmarks UQ for ADME
**under data shift** using the UNIQUE framework; finds **data-based (chemical distance) and
model-based (variance) UQ metrics capture complementary aspects**, and that combining them
through an **error model** (a model that predicts the base model's error) gives the
highest-quality estimates and is robust across shift types.

**What to report:** **Validity** — empirical coverage at each claimed nominal level (sweep
0.70/0.80/0.90/0.95, not one point), plus **class-conditional (Mondrian) coverage** and
**AD-stratified coverage bins**, because marginal coverage hides systematic undercoverage on
the hard tail. **Efficiency** — mean interval width *and* **relative interval width normalised
by the training response range**; for classification, **singleton rate** plus empty and
multi-class rates. **Calibration** — **ECE/ENCE**, **NLL**, a calibration curve, and
**Spearman ρ between predicted uncertainty and realised absolute error** (report this
precisely because it is the metric Hirschfeld et al. show neural UQ fails). **Assumption
diagnostic** — a **KS test comparing calibration-set and test-set nonconformity-score
distributions**; ❗ **under temporal or cluster shift exchangeability breaks and the guarantee
goes with it.** **Process** — reserve the calibration set at development time and keep it out
of hyperparameter tuning (see the `random_state=42` failure above).

#### Retrospective enrichment vs prospective hit rate

**The prospective numbers, verified from the papers:**

| Campaign | Target | Library docked | Tested | Hits | Hit rate | Best affinity |
|---|---|---|---|---|---|---|
| [Lyu 2019 *Nature*](https://doi.org/10.1038/s41586-019-0917-9) (✅ PMID 30728502) | AmpC | 99 M | 44 made | 5 | **11%** | Ki 1.3 µM → **77 nM** |
| Lyu 2019 | D4 | 138 M | **549** | **122** | **22%** | Ki **180 pM** |
| [Stein 2020 *Nature*](https://doi.org/10.1038/s41586-020-2027-0) | MT1 melatonin | >150 M | 38 made | 15 | **39%** | **470 pM** |
| [Gorgulla 2020 *Nature*](https://doi.org/10.1038/s41586-020-2117-z) (VirtualFlow) | KEAP1 | >1 B | 590 | 23 | ~3.9% | Kd **114 nM** |
| [Alon 2021 *Nature*](https://doi.org/10.1038/s41586-021-04175-x) | σ2 | 490 M | **484** | 127 <1 µM | **26%** | 3–48 nM |
| [Sadybekov 2022 *Nature*](https://doi.org/10.1038/s41586-021-04220-9) (V-SYNTHES) | CB2 | >11 B REAL, docking <0.1% | — | 14 submicromolar | **33%** | Ki **0.9 nM** |
| [Kaplan 2022 *Nature*](https://doi.org/10.1038/s41586-022-05258-z) | 5-HT2A | 75 M | 17 | 4 | ~24% | EC50 41 nM |
| [Lyu 2024 *Science*](https://doi.org/10.1126/science.adn6354) | σ2, **AF2 model** | — | 119 | **64** | **54%** | Ki 1.6–84 nM |
| Lyu 2024 | σ2, **crystal** | — | 138 | 70 | **51%** | Ki 1.8 nM– |
| Lyu 2024 | 5-HT2A, **AF2** | — | 161 | 42 | **26%** | EC50 42 nM– |
| Lyu 2024 | 5-HT2A, **cryo-EM** | — | 223 | 52 | **23%** | EC50 246 nM– |
| [Brezinová 2026 *Mol Syst Biol*](https://doi.org/10.1038/s44320-025-00159-5) | Aβ42 nucleation | 539 M | 35 | 19 | **54%** | low-nM Kd |
| [Gutkin 2024 *Chem Sci*](https://doi.org/10.1039/d3sc06880c) | LRRK2 WDR | 4.1 B | 59 made | 5 | **8.5%** | — |

⚠️ Caveats: Lyu 2019 D4 is **122/549 = 22%**, not the widely-quoted 24%. Gorgulla's 23/590 is
second-hand. V-SYNTHES' 33% is in the abstract but **the denominator is not**. Kaplan's 4/17 is
arithmetic; the paper does not call it a hit rate.

**The sanity check:** Zhu, Zhang, Li & Huang 2022, *IJMS* 23(24):15961, DOI
[10.3390/ijms232415961](https://doi.org/10.3390/ijms232415961) surveyed **419 prospective SBVS
case studies** over 15 years. Only **113 (~27%)** produced a best hit better than 1 µM; *"the
activities of identified hits were better than 100 µM in most studies."* And: ❗ *"**We did not
find any association between the docking software and the resulting hit potency and hit
rate.**"* ⚠️ The survey gives no aggregate median, so **there is no citable "typical" hit rate**
— the 2–55% spread above is the honest answer, and it is dominated by threshold and selection
choices, not method.

⭐ **The retrospective–prospective gap, quantified two ways:**

**(1) Decoy provenance.** Lacour, Agha, Hirsch & Volkamer 2026, *J Cheminform* 18:135, DOI
[10.1186/s13321-026-01287-2](https://doi.org/10.1186/s13321-026-01287-2) ran **one pipeline
(DockM8) across both benchmarks:**

| Benchmark | Median relative EF1% | Best-case median |
|---|---|---|
| **DUD-E** (synthetic decoys) | **73.80%** | 82.59% |
| **LIT-PCBA** (real screening inactives) | **4.93%** | 8.85% |

❗ **A ~15-fold drop in enrichment from the same software, purely from swapping synthetic
decoys for experimentally confirmed inactives.** ⚠️ These are **relative** EF1% (% of maximum
achievable), not raw. Derived (flagged as arithmetic): at DUD-E's ~2% active fraction,
EF1%max ≈ 50, so **raw EF1% ≈ 37**; at LIT-PCBA's 0.1–2%, **raw EF1% ≈ 2–5**. So the common
framing "retrospective EF1% of 10–50× vs real hit rates of a few % to 20–30%" is well
supported, with the refinement that **the 10–50× range is specifically the synthetic-decoy
regime.**

**(2) Retrospective enrichment can be *anti-correlated* with prospective usefulness.** Lyu,
Kapolka, Gumpper, Alon et al. 2024, *Science* 384:eadn6354, DOI
[10.1126/science.adn6354](https://doi.org/10.1126/science.adn6354) — same targets, same
protocol, two structures each:

| | retrospective | prospective |
|---|---|---|
| σ2, crystal | **logAUC 39** | **51%** (70/138) |
| σ2, AlphaFold2 model | **logAUC 16** | **54%** (64/119) |

❗ **The AF2 model is 2.4× worse retrospectively and slightly better prospectively.** Same for
5-HT2A — and *"the most potent, subtype-selective agonists were discovered via docking against
the AF2 model, not the experimental structure."* **This is the strongest available evidence
that retrospective enrichment measures recovery of already-known ligands, i.e. a memorisation
task in the §D.3 sense.**

**Three framing points the field keeps getting wrong:**

**(a) "Hit rate" is not a property of a method.** It is a function of the activity threshold,
the number tested, and whether a human looked at the list — and all three are visible above.
*Threshold:* Alon et al. count Ki < 1 µM and get 26%; at < 50 nM the same 484 compounds give
31/484 = **6.4%**. *Number tested:* Lyu 2019 found for D4 that *"hit-rates fell almost
monotonically with score"* — **22–26% among the top-ranked, 12% by −54 kcal/mol, zero below
−43**. *Human inspection:* Lyu 2019, verbatim — **visual inspection yielded 44% submicromolar
vs 27% for docking-score-only selection.** ❗ **A reported hit rate that includes a medicinal
chemist's eye is not a measurement of the algorithm.**

**(b) Docking score predicts P(binding) much better than it predicts affinity — but the old
slogan overstates it.** Moesgaard, Shoichet & Mailhot 2026, *JCIM* 66(14):8325–8335, DOI
[10.1021/acs.jcim.6c01066](https://doi.org/10.1021/acs.jcim.6c01066) fit Pearson ρ between
DOCK score and measured affinity: **AmpC −0.70, D4 −0.40, σ2 −0.64**. Their gloss: *"As weak as
the correlations between docking score and binding affinity are, a simple quantitative model
demonstrates a definitive correlation that matches both hit rate and affinity patterns...
**This challenges the view of docking as merely a classifier.**"* They also quantify the
economics: **a 0.1 increase in that correlation would justify a ~10-fold increase in compute
per molecule.** The accurate statement: **docking score is a weak-but-usable rank predictor of
affinity (|ρ| ≈ 0.4–0.7, R² ≈ 0.16–0.49) and a much better predictor of whether a compound
binds at all.** Lyu 2019's own words are worth quoting alongside: *"our raw docking scores
remain off-set from true binding energies, and we cannot confidently even rank-order molecules
for activity."*

**(c) Library size buys affinity, not hit rate — and buys artefacts.** Lyu, Irwin & Shoichet
2023, *Nat Chem Biol* 19:712–718, DOI
[10.1038/s41589-022-01234-w](https://doi.org/10.1038/s41589-022-01234-w): *"the bias toward
bio-like molecules in the tangible library **decreases 19,000-fold** versus those
'in-stock'"*; *"better-fitting molecules are found as the library grows, with the **score
improving log-linearly with library size**"*; and ❗ *"as library size increases, **so too do
rare molecules that rank artifactually well.** Although the nature of these artifacts changes
from target to target, the expectation of their occurrence does not."*

**And a caution against over-correcting:** Tran-Nguyen & Ballester 2023, *"Beware of Simple
Methods for Structure-Based Virtual Screening"*, *JCIM* 63(6):1401–1405, DOI
[10.1021/acs.jcim.3c00218](https://doi.org/10.1021/acs.jcim.3c00218) — on ACHE-DEKOIS2.0 a
simple interaction-fingerprint method scored PR-AUC **0.167** against target-specific ML
scoring functions at **0.518–0.566**; on HMGR-DEKOIS2.0, **0.125 vs 0.791–0.968**. **Don't
swing so far toward "simple baselines are enough" that you skip the broad comparison.**

---

### D.7 Reporting standards

The two documents to follow are both 2025/2026 and both from cross-industry consortia:

**Ash, Wognum, Rodríguez-Pérez, Aldeghi, Cheng, Clevert, Engkvist, Fang et al.
"Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule
Drug Discovery", *JCIM* 2025**, DOI
[10.1021/acs.jcim.5c01609](https://doi.org/10.1021/acs.jcim.5c01609): *"statistically rigorous
method comparison protocols and domain-appropriate performance metrics are essential to ensure
replicability and ultimately the adoption of ML in small molecule drug discovery."* ⚠️ The
specific guideline list could not be retrieved (paywalled).

**Seal, Zalte, Araripe, Gomes, Korani, Shekhar, Siramshetty, Patra et al. "Model Validation
Protocols for Machine Learning in Small Molecule Drug Discovery", bioRxiv 2026**, DOI
[10.64898/2026.08.19.745868](https://doi.org/10.64898/2026.08.19.745868)
([code](https://github.com/srijitseal/polaris)). ⭐ **Its five recommendations are the cleanest
checklist available:**

1. **Characterise the dataset before splitting** — audit chemical-series structure, endpoint
   coverage and temporal ordering.
2. **Identify the deployment scenario** — choose the split that creates the shift you will
   actually face.
3. **Use distance-aware splits with multiple repeats** — *"Report confidence intervals; **avoid
   using random splits as evidence of deployment performance when structural novelty is the
   use case**."*
4. **Verify the split creates the intended shift** — fold balance, endpoint distributions,
   nearest-neighbour distances, structural overlap, **adversarial validation**.
5. **Probe specific failure modes** — performance-over-distance curves, in/out-of-distribution
   series tests, activity-cliff analysis, molecular-variant consistency checks.

And its four failure modes with diagnostics:

| Failure mode | Diagnostic |
|---|---|
| **Extrapolation** (structurally distant compounds) | performance-over-distance curve; R² vs NN distance |
| **Interpolation** (activity cliffs) | cliff vs non-cliff subset error |
| **Representation** (fingerprints miss stereo, resonance) | stereoisomer consistency test; resonance-form sensitivity |
| **Evaluation** (validation mismatched to deployment) | scaffold vs random comparison; between-repeat variance |

Also: Bender, Schneider, Segler, Walters, Engkvist & Rodrigues 2022, *"Evaluation guidelines
for machine learning tools in the chemical sciences"*, *Nature Reviews Chemistry* 6:428–442,
DOI [10.1038/s41570-022-00391-9](https://doi.org/10.1038/s41570-022-00391-9); and Bender &
Cortés-Ciriano's two-part realism series (*Drug Discov Today* 2021, DOI
[10.1016/j.drudis.2020.12.009](https://doi.org/10.1016/j.drudis.2020.12.009) and
[10.1016/j.drudis.2020.11.037](https://doi.org/10.1016/j.drudis.2020.11.037)) — *"**the quality
of decisions regarding which compound to take forward... are more important than speed or
cost**."*

⚠️ ❌ **There is no Walters piece titled "Does your model beat a random forest?"** — do not
cite it. The right substitutes are his
[*"We Need Better Benchmarks for Machine Learning in Drug Discovery"* (2023-08-03)](https://practicalcheminformatics.blogspot.com/2023/08/we-need-better-benchmarks-for-machine.html)
— **MoleculeNet BBBP has 11 unparseable SMILES, 59 duplicate structures and 10 pairs labelled
both penetrant and non-penetrant**; **BACE has 71% of molecules with ≥1 undefined
stereocentre** (222 with ≥3), aggregated from 55 papers with same-compound IC50s differing by
>0.3 log; **ESOL spans >13 logs where pharma work spans 2.5–3** (so its R² is inflated by
range, not skill); **HIV has 70% of "confirmed actives" triggering structural alerts** and 68
azo dyes — plus
["Some Thoughts on Splitting Chemical Datasets" (2024-11-16)](https://practicalcheminformatics.blogspot.com/2024/11/some-thoughts-on-splitting-chemical.html).

> ❗ **Note that four of the eight MoleculeNet datasets in §B.9 are named in that audit.** Our
> §B.9 table should be read with it.

#### The baselines you must beat

**Non-negotiable, for everything:**

1. ⭐ **Count-ECFP4 (2048 bits, r=2) + RDKit 2D descriptors → LightGBM, tuned per task.** This
   is the live state of the art *as a baseline*: it beats MolFormer and MiniMol on MoleculeACE
   (RMSE 0.680 / cliff 0.769) and ties the best 2026 foundation models within 0.02 log units.
   **XGBoost currently sits #2 on the TDC Caco2 leaderboard.**
2. **Random forest on Morgan fingerprints** — wins 50% of MoleculeACE tasks and 67% of the
   cliff subsets **with no HPO at all.**
3. **1-NN / k-NN(5) on Tanimoto distance** — matches complex ML on ChEMBL-scale potency
   (Janela & Bajorath 2022, *Nat Mach Intell* 4:1246–1255, DOI
   [10.1038/s42256-022-00581-6](https://doi.org/10.1038/s42256-022-00581-6)); Chen 2019's 1-NN
   reached AUC > 0.9 on 97/102 DUD-E targets.
4. **The trivial predictor** — global mean (regression) or prevalence (classification).
5. **The noise floor** — state **0.44–0.68 log units and R²max ≈ 0.81** next to the metric.

**Additionally for binding affinity:**

6. **A ligand-only model** (ECFP + LightGBM on pKi, no protein).
7. **AutoDock Vina** (Eberhardt, Santos-Martins, Tillack & Forli 2021, *JCIM*, DOI
   [10.1021/acs.jcim.1c00203](https://doi.org/10.1021/acs.jcim.1c00203)). Reference line:
   **Rp 0.604 / RMSE 1.73 / SR1 90.2% / EF1% 7.70.**
8. **The nearest-neighbour memorisation baselines** (complex R 0.716, ligand R 0.707).

#### Numbers to report per TDC task

TDC's protocol: **minimum five independent runs** at seeds 1–5, reported as **mean and standard
deviation**; **the test set is fixed** at 20% by scaffold split for all 22 datasets. ❗ **Note
the limitation: because the test set is fixed, TDC error bars reflect *training* variance, not
split variance.** Report both — add our own repeated scaffold-balanced and Butina splits for
split variance. See §B.8 for the full 22-task leaderboard.

Three honest observations about that leaderboard: **gradient boosting holds #1 on PPBR and #2
on Caco2 — the baseline is on the leaderboard**; the clearance and half-life Spearmans of
**0.54–0.58** are the real ceiling on the hardest endpoints, **matching the ASAP challenge's
independent finding that solubility and clearance resist modelling**; and per Koleiev et al.
2026, **only 3 of the 10 top-ranked architectures reproduce at all and three had traceable
leakage.**

⭐ **A reporting idea worth adopting at our exact scale: a learning curve, not a single
number.** Lee 2026, *"When Do Models Win? A Learning Curve Benchmark for Molecular Property
Prediction in Low-Data Regimes"*, ChemRxiv DOI
[10.26434/chemrxiv.15001253](https://doi.org/10.26434/chemrxiv.15001253) varies training size
from **50 to 3,000** under scaffold splits and finds: model optimality is
**regime-dependent** (deep learning leads on continuous physicochemical properties;
**Gaussian process regression leads on bioactivity**); **pretraining raises the low-data floor
rather than the mean** (lifting minimum QM9 R² at N≤200 by ~0.6); **neither parameter count nor
pretraining-corpus size predicts ranking** (3.4 M-parameter ChemBERTa-2 beats 44.5 M MoLFormer
on ESOL, and the order reverses elsewhere); and **under data scarcity, model selection beats
averaging.**

---

### D.8 The held-out protocol that would make our numbers credible

The 24 h budget is **an advantage** here: a tuned LightGBM-on-ECFP4 baseline trains in minutes,
so there is no excuse for not having it. **Spend the GPU on one model and the CPU on all the
controls. Controls are cheap; credibility is the expensive part.**

#### Data preparation (both tasks)

1. **Standardise structures** — RDKit sanitisation, neutralisation, salt/mixture removal,
   canonical tautomer, **stereochemistry preserved.** Record how many molecules have undefined
   stereocentres.
2. ⭐ **Deduplicate on the standardised structure including stereo and tautomer**, not on bare
   canonical SMILES or InChIKey — **this is exactly where MiniMol's leak got through.**
3. **Curate maximally** (Landrum & Riniker's recipe above). Accept the smaller dataset.
4. **Keep `standard_type`.** Pool Ki+Kd; fold IC50 in with a **+0.35 log offset**; **keep EC50
   separate.** Report per-type performance.
5. ⭐ **Keep censored records** and train with a **censored/Tobit likelihood.** Never let
   threshold-imputed points into a test-set RMSE. Record the censoring fraction per endpoint.
6. **Write down the noise floor** for each endpoint *before* training anything.

#### (a) ADMET regression / classification

**Report a panel of splits, not one number.** Every row of the results table is a split:

| Split | Role | Implementation |
|---|---|---|
| **Random** (5 seeds) | Stated ceiling. **Label it as such.** | — |
| **Scaffold-balanced** (5 seeds) | Comparability with TDC/MoleculeNet | Chemprop `SCAFFOLD_BALANCED` / `astartes` — **not** DeepChem's deterministic splitter, so you get error bars |
| **Butina cluster, Jaccard 0.7** (5 repeats) | ⭐ **Primary headline number** | DeepChem `ButinaSplitter` |
| **Temporal** (where dates exist) | Prospective realism | first 75% / last 25% by `document_chembl_id` year |
| **SIMPD** (where they don't) | Simulated temporal | `rinikerlab/molecular_time_series` |
| ⭐ **Series-holdout** | **The failure mode that kills triage** | train on one chemical series, test on another — Seal et al. saw **R² go negative on 5–6 of 8 endpoints** |
| **Lo-Hi Hi** (optional) | Pure novelty, if hit-ID is a use case | `lohi_splitter`, Tanimoto ≤ 0.4 |

If we adopt one upgrade over the panel, make it **SPECTRA**: performance as a curve against
decreasing cross-split overlap, with the area under it as the headline.

**For every split, publish the split's own diagnostics:** train→test nearest-neighbour ECFP4
Tanimoto distribution with the median and the **% above 0.6 and above 0.9**; % of test
molecules sharing a Bemis–Murcko scaffold with train; exact-duplicate count; and an
**adversarial-validation AUC** (can a classifier tell train from test?). **A reader cannot
calibrate our number without these**, and Reimer et al. 2026 show a named split can be no
harder than random.

**Controls:** y-randomisation with the **full pipeline inside the loop**, ≥100 permutations,
reporting the mean *and maximum* null metric plus an empirical p-value; the near-duplicate
audit; a **learning curve from N=50 upward**; and **per-endpoint censoring fractions**.

**Baselines (all five, every time):** count-ECFP4+RDKit2D→LightGBM tuned; RF on Morgan; 1-NN
Tanimoto; global mean/prevalence; and — given Lee 2026's finding that **GPR leads on
bioactivity in the low-data regime** — a Tanimoto-kernel GP for any bioactivity-like endpoint.

**Metrics:** the TDC metric per task **plus** RMSE with the noise floor stated alongside, plus
**Spearman** (triage is a ranking problem — report the metric that matches the decision), plus
AUPRC whenever the positive rate is below 2%. Report **mean ± SD over ≥5 seeds**, and use
**paired statistical tests** across seeds when claiming one method beats another — not a
comparison of two point estimates.

**Uncertainty:** wrap everything in **inductive/Mondrian conformal prediction** with an
explicitly reserved calibration set kept out of hyperparameter tuning. Report a **coverage
sweep** (0.70/0.80/0.90/0.95), **relative interval width normalised by the training response
range**, **class-conditional and AD-stratified coverage bins**, **ENCE**, a calibration curve,
**Spearman ρ(predicted uncertainty, realised absolute error)**, and a **KS test** comparing
calibration- and test-set nonconformity distributions — and **say explicitly that
exchangeability breaks under our Butina and temporal splits.**

**Applicability domain:** report the Tanimoto-to-nearest-training-neighbour distribution with
a **95th-percentile** threshold and quote the fraction of the prediction set that falls out of
domain. **State explicitly that triage rankings for out-of-domain compounds are not
supported.**

#### (b) Binding-affinity prediction

**If structure-based:**

1. ⭐ **Train on PDBbind CleanSplit, not raw PDBbind** ([camlab-ethz/GEMS](https://github.com/camlab-ethz/GEMS)),
   or LP-PDBbind / HiQBind. **Say which, and say why.** Training on unfiltered PDBbind and
   reporting CASF-2016 is, in 2026, a self-inflicted credibility wound.
2. **Evaluate on CASF-2016, all four powers**, **and separately on the 144-complex leakage-free
   subset**, **and on CASF-2013** (and its 101-complex independent subset). **Report the
   full-set and independent-subset numbers side by side; the gap is the honest measure of what
   the benchmark was giving you.**
3. ⭐ **Run every control in §D.3 and put them in one table beside the headline:**

| Row | Expected if the model is real |
|---|---|
| Full complex | best |
| **Ligand only** | **much worse** — if within 0.1, stop and fix the data |
| **Protein only** | much worse (Yang 2020 saw 0.84 > 0.80) |
| Wrong-receptor / scrambled pocket | ≈ chance |
| Pose-scrambled | materially worse |
| k-NN(5) complex memorisation | you beat it clearly (R 0.716) |
| k-NN(5) ligand memorisation | you beat it clearly (R 0.707) |
| Per-target mean | you beat it clearly |
| Global mean | floor |
| **AutoDock Vina** | reference (Rp 0.604 / RMSE 1.73 / SR1 90.2% / EF1% 7.70) |
| y-randomised | ≈ chance, mean and max over ≥100 permutations |

4. **If we produce poses:** report **"PoseBusters-valid AND RMSD < 2 Å"** as a single joint
   number, never RMSD alone. Add DockGen if claiming cross-domain generalisation.
5. **If we screen:** **LIT-PCBA as primary**, reported as **per-target EF1% with bootstrapped
   95% CIs** — never a bare mean across targets. Add DUD-E and DEKOIS 2.0 in their native
   metrics for legacy comparability **with the bias caveat stated**, ideally with a
   **DeepCoy-regenerated-decoy arm**, plus DUDE-Z's **Extrema + Goldilocks** sets to catch
   electrostatic over-fitting. **Expect the ~15× DUD-E→LIT-PCBA enrichment drop and present it
   as a feature of the report, not a failure.**

**If ligand-based (per-target ChEMBL affinity — probably the right choice at our compute
scale):**

1. **Per-target models**, with the split panel above applied per target (Butina primary;
   temporal where dates exist).
2. ⭐ **MoleculeACE as the lead-optimisation stress test** — its 30 targets, its fixed splits,
   **both RMSE and RMSE_cliff**. Reference points: tuned LightGBM-on-ECFP4 **0.680 / 0.769**;
   SVM+ECFP **0.671 / 0.742**; best 2026 foundation model **0.629 / 0.737**. **Use the
   post-October-2023 corrected splits and say so.** Expect a cliff penalty of **+0.10 to +0.11
   log units** — ❗ **if we report no cliff penalty, something is wrong.**
3. **Report RMSE_cliff even on our own data.** A model selected on RMSE alone can be chosen to
   be bad exactly where triage decisions get made.
4. ❗ **Ligand-only is not a failure mode here — it is the model.** But the honest framing
   changes: **say explicitly that we are predicting per-target potency from structure, not
   modelling the interaction**, and that the ceiling is ~0.4–0.7 log units of assay noise.

#### What to claim, and what not to

**Claim:** a per-split performance panel with error bars over ≥5 seeds; the margin over a tuned
ECFP4+LightGBM baseline, with a paired test; the ablation table; the conformal coverage and
interval width; the applicability-domain fraction; and the noise floor.

**Do not claim:** that retrospective enrichment or a CASF number predicts prospective hit rate
(Lyu 2024 shows it can be *anti*-correlated); that the model ranks affinity *within* hits (even
DOCK manages only |ρ| ≈ 0.4–0.7, and Lyu 2019's authors say *"we cannot confidently even
rank-order molecules for activity"*); a hit rate without stating the threshold, the number
tested and whether a human curated the list; or **any therapeutic implication whatsoever.**

> ⭐ **The defensible framing for a triage model:** *"reduces the number of compounds that must
> be assayed to find N actives, under the stated applicability domain, with the stated
> coverage"* — **a ranking claim with a calibrated uncertainty attached, not an affinity
> claim, and never an efficacy claim.**

---

### D.9 Things that do not exist — do not cite them

Collected so nobody re-researches them:

| Claim | Status |
|---|---|
| **"CASF-2024"** | ❌ does not exist; CASF-2016 is the latest official release |
| **"CBScore"** benchmark | ❌ not in Crossref |
| **"Latent Hit Enrichment"** as a DEKOIS metric | ❌ the metrics are **LADS** and **DOE** |
| **"Chen, Cruz, Paul, Ash, Durrant"** | ❌ a mis-merge of Chen et al. 2019 (*PLoS ONE*) and Yang, Shen & Huang 2020 (*Front Pharmacol*) |
| Walters, **"Does your model beat a random forest?"** | ❌ no such piece |
| **"MoleculeACE revisited"** | ❌ no such paper |
| **"MolmirNet"** | ❌ no paper, no repo, empty search (§C.1) |
| `DeepChem/ChemBERTa-100M-MTR` | ❌ only `-MLM` exists (§C.1) |
| `TheLuoFengLab/ChemFM-1B` | ❌ the real home is `ChemFM/ChemFM-1B` (§C.1) |
| **"ESM-2 small distillations"** | ❌ no such family (§C.2) |
| AVE evaluated on **nine** benchmarks | ❌ it is **seven** |
| MUV has **18** targets | ❌ it is **17** |
| DUD-E has **~1.14 M** decoys at 50/active | ❌ **1,411,214** decoys, ~**62** per active (§B.7) |
| LIT-PCBA's **7,844 / 407,381** | ⚠️ superseded online by **7,761 / 382,674** |
| PoseBusters Benchmark = 428 in the paper's results | ⚠️ **428 as released, 308 in the paper's numbers** |
| Kramer 2012's "0.54 for mixed data" | ⚠️ 0.44 is the **mean error**, 0.54 the **SD** |
| Tiikkainen 2013's "~5% error rate" | ❌ no percentage in the abstract |

---

## Recommended minimal data plan for 24h on one L4

**Design principle.** The L4 is 24 GB (≈21 GB usable) and bandwidth-starved at 300 GB/s.
Nothing in sections B, C or D says we are short of GPU. Everything says we are short of
**clean data, honest splits, and CPU for featurisation**. So this plan downloads **under
300 MB on day 1**, spends the GPU on ensembles and controls rather than on one big model, and
adds the 8 GB structure tier only once the ligand-based half is beating its own baseline.

All sizes below were verified by HTTP HEAD or an API `size` call on 2026-10-04.

### Tier 0 — the floor, before any GPU work (≈270 MB, CPU-only)

Build the baseline that §B.8, §B.9, §C.1 and §D.7 independently say may be the ceiling. **If
the deep model does not beat this, it has not earned its place.**

| What | Source | Size ✅ |
|---|---|---|
| **Papyrus 05.7++** — multi-target affinity, pre-curated, CC BY-SA 4.0 | `zenodo.org/records/13987985` | **56.8 MB** |
| **Papyrus protein-target table** (joins onto section A's accessions) | same record | **1.8 MB** |
| **TDC ADMET group**, all 22 tasks | `pip install PyTDC` → `admet_group(path='data/')` | **~10 MB** |
| **TDC mirrors** as a loader-free fallback — ✅ **20 of the 22** exist under `scikit-fingerprints/TDC_*`, totalling **3.16 MB / 66,680 rows**. ❌ `TDC_lipophilicity_astrazeneca` and `TDC_clearance_hepatocyte_az` are **not mirrored** — take those two from PyTDC | HF | **3.2 MB** |
| **MoleculeACE** — ✅ all **30** sets, **48,714 rows**, `cc-by-sa-3.0`. Includes JAK1 (`chembl2835_ki`, 615 rows) and PPARγ (`chembl235_ec50`, 2,349) | `scikit-fingerprints/MoleculeACE_*` | **3.75 MB** |
| **MoleculeNet** 8 canonical CSVs (`SAMPL.csv` for FreeSolv — lowercase `freesolv.csv` 403s) | DeepChem S3 | **~3 MB** |
| **MoleculeNet mirrors**, 12 sets, 600,976 rows | `scikit-fingerprints/MoleculeNet_*` | **126.5 MB** |
| **Per-target ChEMBL 37 pulls** for section A's tier 1+2 targets | ChEMBL REST | **~50 MB** |
| **ZINC250k** (`yairschiff/zinc250k` — the cleanest copy) for smoke tests | HF | **65 MB** |

**Tier 0 total ≈ 270 MB.** Trains on CPU in minutes:
`count-ECFP4 + Avalon + ErG + 200 RDKit descriptors → CatBoost/LightGBM` (this *is* MapLight)
via `scikit-fingerprints` 2.1.0, plus `chemprop>=2.2.0 --from-foundation CheMeleon` on the GPU
at ~1–2 GB. Also pull **`openadmet/*-chemeleon-v1`** (Apache-2.0, ~0.8 GB inference, zero
training) for a day-one calibrated comparison on hERG, four CYPs, microsomal clearance,
Caco-2, LogD and PPB.

❗ **Two things to get right in the ChEMBL pull, both from §D:**

```python
# Keep these columns or you cannot build a time split or detect assay-batch leakage:
#   document_chembl_id, assay_chembl_id, standard_type, standard_relation
# And DO NOT filter on pchembl_value__isnull=False for the training set —
#   that filter silently deletes every censored ('>' / '<') record, i.e. all your inactives.
```

Apply **Landrum & Riniker's maximal-curation recipe** (§D.6) — matching `assay_type` /
`assay_organism` / `bao_format` / etc., `confidence_score = 9`, one assay per document per
target, drop assays with >100 compounds — and **accept the smaller dataset.** It cuts the
>1-log disagreement rate from 27% to 13%.

### Tier 1 — the real affinity training set (≈890 MB, or 1.65 GB with full Papyrus)

| What | Size ✅ | Note |
|---|---|---|
| **ChEMBL 37 `chemreps.txt.gz`** | **292.5 MB** | SMILES + InChIKey for all 2,921,148 compounds. **Not** the 5.76 GB SQLite — unpacked it will exhaust Colab's disk |
| **BindingDB `BindingDB_All_202610_tsv.zip`** | **594.3 MB** | 3,248,380 measurements. ❗ **Dedup against ChEMBL on (standardised structure incl. stereo/tautomer, UniProt, measurement type)** — BindingDB imports ChEMBL, and §D.6 shows canonical-SMILES matching under-detects this |
| **BindingDB UniProt mapping** | 0.6 MB | |
| **Papyrus 05.7 full** — only if `++` proves too small | **751.5 MB** | |

### Tier 2 — screening and cliff benchmarks (≈110 MB)

| What | Size ✅ |
|---|---|
| **LIT-PCBA `full_data.tgz`** | **53.8 MB** |
| **LIT-PCBA `AVE_unbiased.tgz`** — download it, then ❗ **re-split it yourself** (§B.6: a zero-parameter memorisation baseline matches published SOTA on the shipped split) | **57.4 MB** |

LIT-PCBA includes **PPARG and ADRB2**, two of section A's tier-1 targets, so it doubles as a
target-specific screening benchmark. ⚠️ **Do not download DUD-E for training or evaluation**
(§B.7, §D.3) — only as a *legacy comparability* arm with the bias caveat stated.

### Tier 3 — structure, only once the ligand half works (8.1 GB)

| What | Size ✅ | Licence |
|---|---|---|
| **HiQBind** `hiqbind.tar.gz` + metadata | **8.07 GB** + 16 MB | ✅ data **CC BY 4.0**; code MIT |
| *or* **Exscientia low-sim PDBbind** `pdbbind_ccdc_structures.zip` | **3.95 GB** | ✅ **CC BY 4.0** |
| **PDBbind CleanSplit / GEMS split files** | <10 MB | §D.8's recommended training split |
| **LP-PDBBind split files** (splits only) | <5 MB | ⚠️ repo licence unstated |
| **The ~45 verified section-A co-crystals** (`.cif` from RCSB) | ~50 MB | public domain |

Use **HiQBind, not PDBbind** (§B.3). Do **not** download PDBbind ≥ v2021, and do not plan to
redistribute any derivative of v2020. ⚠️ Treat **PLINDER** as a source of split methodology
rather than a dependency until its **GPL-2.0** tooling licence has been reviewed.

### Tier 4 — skip on the first run

- **PubChem `CID-SMILES.gz` (1.49 GB)** and `hheiden/PubChem-124M-...` (18.3 GB, CC0) — only
  if we ever pretrain from scratch, which §B.8, §B.9 and §C.1 all argue against.
- **`chenbq18/BindingNet_v2_fixed` (97.1 GiB)** — too large, and its licence is stated three
  different ways.
- **`Kingldore/lit-pcba-vs` (~939 GiB)** — absurd for this budget.
- **ZINC22 / Enamine REAL** — ZINC22 3D is ~1 TB per *billion* molecules; and ❗ **Enamine REAL
  is blocked pending the §B.11 terms question.**

### Totals

| Plan | Size |
|---|---|
| **Day 1 (tier 0)** | **≈ 270 MB** |
| **Day 1 + affinity + benchmarks (tiers 0–2)** | **≈ 1.3 GB** |
| **Full ligand-based plan (incl. full Papyrus)** | **≈ 2.1 GB** |
| **Adding structure (tier 3, HiQBind)** | **≈ 10.2 GB** |
| **Adding structure (tier 3, Exscientia instead)** | **≈ 6.1 GB** |

❗ **Everything the ligand-based model needs fits in 2.1 GB.** That is the most important number
in this plan: **the data that matters is small**, and the 8 GB structure tier is optional.

### What to cache to Google Drive

Colab's local disk is ephemeral, and its **disk** — not its VRAM — is what the Boltz-2
embedding step exhausts. Mount Drive at `/content/drive` and cache in this order of value:

**Cache (high value — expensive to regenerate):**

1. ⭐ **RDKit featurisations** — ECFP4 / Avalon / ErG / Mordred matrices as `.npy` or parquet,
   keyed by **standardised InChIKey including stereo**. This is the single biggest win:
   featurisation is the CPU bottleneck on 2–8 vCPUs, and ADMET-AI's 90 mol/s assumes 32 cores,
   so expect **~10–25 mol/s** here.
2. ⭐ **Frozen embeddings** — MiniMol 512-d vectors, ChemBERTa/MoLFormer pooled outputs, and
   especially **protein-tower embeddings** (ESM-2 650M or ESMC-600M/6B, run once in bf16).
   **This is what makes a 6 B protein model usable on an L4 at all.**
3. **Boltz-2 `--write_embeddings` output**, if we take the affinity-head route. ⚠️ Large —
   budget generously and prune per target.
4. **Pre-computed MSAs for section A's targets** — the Boltz docs insist on this; the ColabFold
   MSA server will otherwise rate-limit us into tens of minutes per prediction.
5. ⭐ **The split assignments themselves** — the exact standardised-InChIKey → fold mapping for
   **every** split (random, scaffold-balanced × 5 seeds, Butina, temporal, series-holdout), as
   a committed CSV, **plus the reserved conformal calibration set**. **A number you cannot
   reproduce is not a result**, and §D.6's `random_state=42` failure is what happens when the
   calibration set is not pinned.
6. **The split diagnostics** — per-split nearest-neighbour Tanimoto distributions and
   adversarial-validation AUCs. These are cheap to compute once and are what a sceptic reads
   first.
7. **Model checkpoints and the per-seed metric tables.**

**Do not cache:** raw ChEMBL SQLite, raw ZINC tranches, anything in tier 4. Re-download is
cheaper than Drive quota, and Drive I/O is slow enough that reading a multi-GB archive per
session costs more than fetching it.

⚠️ **One operational warning:** Drive I/O through FUSE is slow for many small files. Write
**one archive or one parquet per logical artifact**, not a directory of 100k files — a
per-molecule cache directory on Drive will be slower than recomputing.

### The 24 hours, allocated

| Hours | What | GPU |
|---|---|---|
| 0–1 | Tier 0 download; standardise (stereo + tautomer), dedupe, RDKit featurise and cache | CPU |
| 1–2 | ⭐ **count-ECFP4 + RDKit2D → LightGBM across all 22 TDC tasks, 5 seeds** — the floor | CPU |
| 2–3 | Build and **freeze** the split panel: random / scaffold-balanced ×5 / Butina 0.7 / temporal by document year / series-holdout; compute and save every split's NN-similarity distribution and adversarial-validation AUC | CPU |
| 3–5 | `chemprop --from-foundation CheMeleon`, 5-fold × 5-seed, on the frozen panel | ~2 GB |
| 5–7 | MiniMol frozen embeddings → LightGBM heads; compare against OpenADMET's shipped models | ~1 GB |
| 7–11 | Affinity: Papyrus++ per-target models on section A tier-1 targets, **with the ligand-only, per-target-mean, 1-NN and y-randomisation controls from §D.3** | ~4 GB |
| 11–13 | MoleculeACE run (corrected splits): **RMSE and RMSE_cliff**, against the SVM+ECFP and LightGBM reference points | ~2 GB |
| 13–16 | Conformal wrapper: coverage sweep, relative interval width, AD-stratified bins, ENCE, ρ(uncertainty, error), KS exchangeability test | CPU |
| 16–21 | *Optional* Boltz-2 affinity-head route on 2–3 **enzyme** targets (JAK1, DPP-4, HIV-1 PR — ❗ **not** the GPCRs, transporters or ion channels, which §C.3 documents as its weak classes) | ~20 GB embed / <5 GB train |
| 21–24 | Learning curves from N=50, bootstrap CIs, paired tests, write-up | CPU |

Note what is **not** in that table: training a large transformer from scratch; generating 3D
conformers for Uni-Mol (≈14 h of CPU alone, §C.1); and any docking campaign at scale. Each
would consume the whole window for less return than the ensembles and controls above.

### Five things to carry into the next document

1. ❗ **The floor is the headline.** Tuned LightGBM-on-ECFP4 tops or ties roughly half of TDC's
   22 ADMET tasks, **beats MolFormer and MiniMol outright on MoleculeACE**, and beat learned
   representations on most MoleculeNet sets under scaffold split (Deng et al. 2023, 62,820
   models). **Report it beside every deep result, always.**
2. ❗ **The split is the result.** Random → Butina costs ~0.20 MA-RAE on real ADME data (~7× the
   split SD), and **series-holdout drives R² negative on 5–6 of 8 endpoints.** Since triage
   exists to evaluate *new* chemistry, the series-holdout number is the one that describes the
   product.
3. ❗ **The leaderboards we quote are contaminated.** Only **3 of 10** top-ranked TDC
   architectures reproduce; **three had traceable leakage**, including MiniMol; and tuning
   against the public test set moved honest models from top-3 on 2/22 endpoints to **10/22
   with 5 first places.** Our own frozen, diagnosed splits are the only numbers worth
   believing.
4. **Boltz-2's affinity head is the one place a 24 GB card buys something a CPU cannot** —
   MIT, frozen trunk, cached embeddings, `<5 GB` to train, and genuinely prospective evidence
   (CASP16 R = 0.65, beating every human entrant). But its documented weak classes are
   **GPCRs, transporters and ion channels**, which is five of section A's targets, and its
   binary confidence is uncalibrated (everything >0.80).
5. ❗ **Two licence questions must be answered before any commercial use**: Enamine's §15
   AI/ML non-use clause (§B.11) and the provenance of TDC's CC BY 4.0 re-licence (§B.8).
   Neither is a technical problem and both are cheap to resolve by asking. A third to watch:
   **Ankh is CC-BY-NC-SA** and **PLINDER's tooling is GPL-2.0**.

---

## Appendix — what could not be verified

Collected so the next person does not re-research it. ❌ = does not exist; ⚠️ = exists but the
specific figure is unconfirmed.

**Highest-value open items, in order of what they would cost us:**

1. ⚠️ **Enamine REAL terms §15** (AI/ML non-use) — read verbatim; get written authorisation
   terms and a price. **Blocks any REAL-based work, including via ZINC22.**
2. ⚠️ **PDBbind+ actual terms and pricing** — the site is a JS-only SPA; composition,
   versioning and licence unconfirmed. Also ❌ **no primary dated statement that v2020 is the
   free-distribution cutoff** exists; every source is secondary.
3. ⚠️ **BindingNet's licence** — stated three different ways (CC BY-NC / CC BY / MIT).
4. ⚠️ **TDC upstream dataset provenance** — the per-dataset CC BY 4.0 is TDC's own re-license;
   `tdcommons.ai/faq/` is a 404.
5. ⚠️ **`biohub/esm3-sm-open-v1` has no `license` field in its HF cardData** (✅ I confirmed
   this myself) and GitHub reports the repo as `NOASSERTION`. The MIT grant is real in
   `LICENSE.md` but reaches the *weights* only by prose reference. **The one licensing item
   worth a lawyer's eye before shipping ESM3-derived weights.**
6. ⚠️ **Boltz-2 affinity fine-tuning "+0.2–0.4 Pearson R from 50–200 compounds"** —
   single-sourced; bioRxiv blocked. The *qualitative* claim is verified.

**Numbers that are derived, not published:**

- All **L4 throughput extrapolations** (Boltz-2 ~30–60 s/complex, docking mol/h, PLM tok/s) —
  arithmetic from published figures on other GPUs scaled by the 300 GB/s bandwidth gap.
  Order-of-magnitude only.
- All **activation-memory estimates** (±40%, ±60% for pair-representation models). Parameter
  counts are measured; activation figures are formula-derived.
- **Raw (vs relative) EF1%** conversions in §D.6 — derived from assumed active fractions.
- ⚠️ **ESMFold max-length figures** (~400 aa unchunked, ~700–900 at `chunk_size=64`) are
  third-party community reports; Meta publishes no per-GPU table.
- ⚠️ Parameter counts marked "derived" (ESM-2 3B/15B, ProtT5/ProstT5/Ankh/ProtBert, Uni-Mol,
  ESM3) come from measured blob bytes ÷ bytes-per-param, not a safetensors header.

**Specific figures that could not be confirmed:** ASAP/Polaris per-task tables; Ash et al.
2025's guideline list; Hirschfeld 2020's practical recommendation; Tiikkainen 2013's "~5%"
error rate; Gorgulla 2020's 23/590 (second-hand); V-SYNTHES' 33% denominator; Lyu 2023's
log-linear slope; Lo-Hi's quantitative drops vs other splits; PLINDER's split sizes and a
peer-reviewed DOI; GEMS/CleanSplit's per-model before/after Pearson R; PoseBusters' headline
percentages; Vina-GPU 2.0's speedup factors (its README publishes none; last push 2023-10-27);
the identity of the 51-benchmark audit as reported by Chemistry World (we did locate the
underlying paper — Schuh, Daniluk & Sieber 2026, *Chem Sci*, DOI
[10.1039/d6sc01799a](https://doi.org/10.1039/d6sc01799a)); Uni-Mol's MoleculeNet values for
HIV/Tox21/ESOL/FreeSolv/Lipo; ChEMBL 37's exact release date (API 2026-05-01, FTP timestamps
2026-05-29, release-notes header "January 5, 2026"); DrugBank's CC0-vs-CC-BY-NC subset split
and the 5.1.22-vs-6.0 numbering; a current ZINC22 2D count; Enamine REAL Database total GB and
whether a free "REAL diversity set" exists at all; the current PubChem compound count; and
`flair-bio` vs `chandar-lab` as AMPLIFY's canonical home.

**❌ Confirmed not to exist** — the full list is in **§D.9**, and it includes CASF-2024,
"CBScore", "MolmirNet", `DeepChem/ChemBERTa-100M-MTR`, `TheLuoFengLab/ChemFM-1B`,
`HoangHa/ChEMBL`, `katielink/moleculenet`, a `ChEMBL/*` HF org, any `facebook/esm*` **dataset**,
distilled-ESM-2 weights, Prescient Design's Lobster weights, a Papyrus HF mirror, a DEKOIS HF
mirror, "MoleculeACE revisited", and Walters' "Does your model beat a random forest?".
