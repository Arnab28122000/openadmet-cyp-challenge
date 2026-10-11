# 02 — Drug discovery: targets, datasets, models, benchmarks

Research notes, 2026-10-04. Web research only; nothing here was run or verified computationally.

**Scope and honesty frame.** The goal is a pipeline that, on a single Colab L4 (24 GB VRAM,
~24 h sessions), trains small models on public bioactivity data and emits *ranked candidate
molecules per target with predicted affinity and predicted ADMET*. Everything such a pipeline
produces is an **in silico hypothesis** that needs wet-lab confirmation. The calibration
section below gives the numbers to quote so we never oversell: in the only large blind
prospective benchmark of ML virtual screening (CACHE #1, LRRK2 WDR), 23 teams' combined
1,955 predictions yielded 73 binders at K_D < 150 µM — a **~3.7 % hit rate, with the best
affinities at 18–140 µM**
([RSC Chem. Sci. 2024](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc06880c)).
That is the realistic ceiling for this class of work, and it is still useful — it is 10–100×
better than random — but a "predicted nanomolar hit" from a model like ours is a ranking,
not a measurement.

---

## 1. Targets per disease

UniProt accessions and PDB IDs below were each verified live against
`rest.uniprot.org/uniprotkb/<acc>.json` and `data.rcsb.org/rest/v1/core/entry/<id>` on
2026-10-04; the PDB resolution is given so we can judge whether a structure is docking-grade.

### 1.1 HIV-1

The four clinically validated small-molecule enzyme/assembly targets, plus two entry targets.

| Target | UniProt | PDB (res. Å) | Reference ligands (approved unless noted) |
|---|---|---|---|
| Protease (PR) | P03367 / P04585 / P03366 (Gag-Pol polyprotein; PR = residues 501–599 of P03367) | **1HSG** (2.0) — indinavir | darunavir, atazanavir, lopinavir, indinavir, saquinavir, tipranavir |
| Reverse transcriptase (RT) | same Pol polyprotein | **1RTD** (3.2) — catalytic complex | NNRTIs: efavirenz, rilpivirine, doravirine. NRTIs: tenofovir, emtricitabine, lamivudine, abacavir, zidovudine. NRTTI (investigational): islatravir |
| Integrase (IN) | same Pol polyprotein | **5U1C** (3.9) — tetrameric HIV-1 strand-transfer complex; **3OYA** (2.65) — PFV intasome + raltegravir (the classic surrogate) | dolutegravir, bictegravir, cabotegravir, raltegravir, elvitegravir |
| Capsid (CA) | P04591 / P12493 (Gag polyprotein) | **6V2F** (2.0) — CA hexamer + GS-6207 (lenacapavir); **4XFZ** (2.7) — full-length CA + PF-74 | lenacapavir; PF-74 (tool compound) |
| CCR5 (host coreceptor) | P51681 | **4MBS** (2.71) — maraviroc | maraviroc |
| gp120 / Env (attachment) | Env polyprotein (strain-specific) | **5U7O** (3.03) — BG505 SOSIP + BMS-626529 | fostemsavir (prodrug of BMS-626529) |

Notes that matter for modelling:

- **1HSG's deposited title says "HIV II protease", which is wrong/historical.** The SIFTS
  mapping for 1hsg is to **P03367 (POL_HV1BR, HIV-1 BRU/LAI)**, residues 501–599, 100 %
  identity, both chains — so it is HIV-1 PR. Confirmed via
  `https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/1hsg`.
- HIV targets are *mutating* targets, so affinity alone is a weak objective. The
  **Stanford HIV Drug Resistance Database** publishes genotype→phenotype (PhenoSense
  fold-change) datasets per gene — PR, RT, IN, and now CAI (capsid) —
  <https://hivdb.stanford.edu/pages/genotype-phenotype.html>, downloadable per-gene with a
  reference regression R script. This gives us a second, cheap, honest task:
  *predict fold-resistance from sequence*, which is a tabular/sequence problem that trains
  in minutes, not hours, and is directly publishable as a feature.
- MoleculeNet's **HIV** dataset (41,127 compounds, binary active/inactive) is *not* a target
  dataset — it is the NCI AIDS Antiviral Screen, a whole-cell phenotypic screen. Useful as a
  pretraining/aux task, misleading if labelled "HIV protease activity".

### 1.2 Type 2 diabetes

T2D is the best-served disease here: large public SAR, several genuinely small-molecule
targets, and a 2026 proof that even a peptide-receptor target is small-molecule druggable.

| Target | UniProt | PDB (res. Å) | Reference ligands |
|---|---|---|---|
| DPP-4 | P27487 | **1X70** (2.1); **2RGU** (2.6) | sitagliptin, linagliptin, saxagliptin, alogliptin, vildagliptin |
| SGLT2 (SLC5A2) | P31639 | **7VSI** (2.95) — human SGLT2–MAP17 + empagliflozin | empagliflozin, dapagliflozin, canagliflozin, ertugliflozin |
| GLP-1R | P43220 | **6X18** (2.1) — GLP-1 peptide bound | **orforglipron** — a non-peptide oral small-molecule GLP-1R agonist; in ACHIEVE-3 (n=1,698, 52 wk) it beat oral semaglutide: HbA1c −2.2 % at 17.2 mg vs −1.4 %, weight −8.9 kg vs −5.0 kg ([Lilly](https://investor.lilly.com/news-releases/news-release-details/lillys-oral-GLP-1-orforglipron-delivered-superior-blood-sugar), [Pharmacy Times, Feb 2026](https://www.pharmacytimes.com/view/orforglipron-outperforms-oral-semaglutide-in-head-to-head-t2d-trial)). Also danuglipron (discontinued) |
| PPARγ | P37231 | **2PRG** (2.3) — rosiglitazone | pioglitazone, rosiglitazone; lobeglitazone |
| Glucokinase (GCK) | P35557 | **1V4S** (2.3) | dorzagliatin (approved in China); piragliatin (tool) |
| 11β-HSD1 | P28845 | **2BEL** (2.11) — NADP + carbenoxolone | carbenoxolone; many failed clinical candidates (rich SAR, no approvals) |
| PTP1B (PTPN1) | P18031 | many | no approved drug; classic "undruggable-ish" phosphatase — good negative control |
| FFAR1 / GPR40 | O14842 | — | fasiglifam (withdrawn, hepatotox) — good ADMET-failure case study |
| GPR119 | Q8TDV5 | — | several discontinued agonists |
| DGAT1 | O75907 | — | discontinued candidates |
| AMPK α1 (PRKAA1) | Q13131 | — | metformin acts upstream, not as a direct AMPK ligand |
| K_ATP: KCNJ11 / ABCC8 | Q14654 / Q09428 | — | sulfonylureas (glibenclamide, glimepiride), repaglinide |

### 1.3 Type 1 diabetes

Honest read: **T1D is largely not a small-molecule target space yet.** The approved
disease-modifying agent is teplizumab (anti-CD3 antibody). The tractable small-molecule
hypothesis space is the autoimmune-signalling one, shared with §1.4:

| Target | UniProt | PDB (res. Å) | Rationale / reference |
|---|---|---|---|
| TYK2 (JH2 pseudokinase) | P29597 | **4WOV** (1.8) — BMS-066; **6NZP** (2.35) | *TYK2* loss-of-function variants protect against T1D; deucravacitinib validates allosteric JH2 binding |
| JAK1 / JAK2 | P23458 / O60674 | **3EYG** (1.9) | baricitinib preserved β-cell function in the BANDIT trial (phase 2) |
| PTPN22 | Q9Y2R2 | — | top non-HLA T1D risk gene; phosphatase, hard |
| PTPN2 | P17706 | — | T1D GWAS, β-cell apoptosis |
| IL-2RA (CD25) | P01589 | — | low-dose IL-2 / Treg axis; biologic, not small molecule |
| Ca_V1.2/1.3 (verapamil) | — | — | verapamil preserved C-peptide in a repurposing trial; a repurposing-screen anchor |

Recommendation: treat T1D as a **repurposing + immune-kinase** problem, and be explicit in
the output that affinity at JAK/TYK2 is a mechanism hypothesis, not a T1D efficacy claim.

### 1.4 Ankylosing spondylitis / axial spondyloarthritis

Approved therapy is dominated by biologics plus two JAK inhibitors. **Bimekizumab**
(dual IL-17A/IL-17F) is FDA-approved for active AS and nr-axSpA at 160 mg q4w
([Healio, Sep 2024](https://www.healio.com/news/rheumatology/20240923/fda-oks-bimzelx-for-psoriatic-arthritis-ankylosing-spondylitis-nonradiographic-axspa));
**tofacitinib** was the first JAK inhibitor approved for AS (2021) and **upadacitinib** is
approved for AS and nr-axSpA. That asymmetry is the opportunity: *the validated biology is
cytokine-level, and the small-molecule entry points are the intracellular transducers and
the nuclear receptor upstream of IL-17.*

| Target | UniProt | PDB (res. Å) | Reference ligands / notes |
|---|---|---|---|
| JAK1 | P23458 | **3EYG** (1.9) | upadacitinib (JAK1-selective), tofacitinib (pan-JAK) — **approved for AS** |
| TYK2 (JH2) | P29597 | **4WOV** (1.8), **6NZP** (2.35) | deucravacitinib (approved in psoriasis); IL-23/IL-17 axis is TYK2-dependent — the single best small-molecule AS hypothesis |
| RORγt (RORC) | P51449 | **4NB6** (2.85) — T0901317 | no approval; several clinical inverse agonists; master TF for Th17/IL-17 |
| IL-17A | Q16552 | **4HSA** (3.15) — IL-17A/IL-17RA; **5HI4** (1.8) — macrocyclic antagonist | secukinumab, ixekizumab, bimekizumab (all antibodies). **5HI4 is the structure to dock against** — it is an actual small-molecule/macrocycle site on a cytokine |
| IL-23 p19 | Q9NPF7 | — | risankizumab **failed** in axSpA — a documented negative, worth encoding |
| TNF-α | P01375 | **2AZ5** (2.1) — small-molecule inhibitor | adalimumab, etanercept, infliximab, golimumab, certolizumab; 2AZ5 is the rare small-molecule PPI site |
| ERAP1 | Q9NZ08 | **6RQX** (1.68) — 10-mer phosphinic peptide | HLA-B27-epistatic AS risk gene; aminopeptidase, genuinely druggable, **no approved drug** → highest unmet-need/novelty score in this set |
| HLA-B27 | P03989 | **1HSA** (2.1) | the causal allele; not a ligand target, but the pocket is characterised |

NICE recommends secukinumab, ixekizumab and upadacitinib for active AS unresponsive to
conventional therapy ([NICE TA918 scope](https://www.nice.org.uk/guidance/ta918/documents/final-scope)).

### 1.5 Asthma

| Target | UniProt | PDB (res. Å) | Reference ligands |
|---|---|---|---|
| β2-adrenoceptor (ADRB2) | P07550 | **3NY8** (2.84) — ICI 118,551 | salbutamol, formoterol, salmeterol, vilanterol, indacaterol |
| CysLT1 (CYSLTR1) | Q9Y271 | **6RZ4** (2.7) — pranlukast | montelukast, zafirlukast, pranlukast |
| CRTH2 / DP2 (PTGDR2) | Q9Y5Y4 | **6D26** (2.80) — fevipiprant | fevipiprant (phase 3 failed in asthma), timapiprant |
| Glucocorticoid receptor (NR3C1) | P04150 | **1M2Z** (2.5) — dexamethasone | budesonide, fluticasone, mometasone, ciclesonide |
| PDE4B | Q07343 | **1XM4** (2.31) — piclamilast | roflumilast (COPD), ensifentrine (COPD, dual PDE3/4) |
| 5-LOX (ALOX5) | P09917 | **3O8Y** (2.39) — "Stable-5-LOX" | zileuton |
| FLAP (ALOX5AP) | P20292 | **2Q7M** (4.25 — too low-res for docking; use a newer FLAP entry) | MK-591, MK-886 (tool compounds) |
| TSLP | Q969D9 | **5J11** (2.56) — TSLP/TSLPR/IL-7Rα | tezepelumab (antibody) — small-molecule space is **open** |
| IL-4Rα | P24394 | — | dupilumab (antibody) |
| IL-5 | P05113 | — | mepolizumab, reslizumab; benralizumab (IL-5Rα) |
| JAK1 | P23458 | 3EYG | investigational inhaled JAK1 inhibitors |

Unmet-need reading: the whole **T2-high biologics** space (TSLP, IL-4Rα, IL-5, IL-33/ST2) is
antibody-only, i.e. injectable and expensive. An oral small molecule at TSLP or ST2 is the
highest-value, highest-risk hypothesis in this disease.

### 1.6 Other unmet-need diseases worth adding

Cheap to add because the data already exists in the same sources, and each has a clear
"no good oral drug" story: **IPF** (αvβ6 integrin, LPA1/LPAR1, autotaxin/ENPP2),
**NASH/MASH** (THR-β, FGF21 axis, ACC), **sickle cell** (PKR/PKLR activators),
**chronic cough** (P2X3), **endometriosis** (GnRH-R antagonists), and
**drug-resistant TB** (DprE1, MmpL3) — TB in particular has a large, well-curated public
phenotypic dataset and is the canonical "ML virtual screening actually helped" story.

---

## 2. Datasets

### 2.1 Primary bioactivity sources

| Source | Size | License | URL |
|---|---|---|---|
| **ChEMBL 36** (rel. 2025-10-15) | 2.8 M distinct compounds, 17,803 targets; BindingDB-sourced patent assays tripled 4,117 → 13,847 | CC BY-SA 3.0 | <https://www.ebi.ac.uk/about/news/updates-from-data-resources/chembl-36/> |
| ChEMBL 35 (for comparison) | ~2.5 M compounds, 21.1 M bioactivities, 1.7 M assays, ~16,000 targets | CC BY-SA 3.0 | <https://www.ebi.ac.uk/chembl/> |
| **BindingDB** | 3.2 M measurements, 1.4 M compounds, 11.4 k targets; of which 1.6 M / 748 k / 4.8 k curated in-house | in-house curation CC BY 3.0; ChEMBL-derived rows CC BY-SA 3.0 | <https://bindingdb.org/rwd/bind/info.jsp>, paper [PMC11701568](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11701568/) |
| **Papyrus 05.x** | 59,775,087 activity values, 1,270,570 unique 2D structures, 6,926 proteins (ChEMBL 30 + ExCAPE-DB + smaller high-quality sets, standardised for ML) | open (4TU / Zenodo) | <https://zenodo.org/records/10944245>, <https://data.4tu.nl/datasets/ca10bf7d-f508-4d54-9c9a-5a9e9c1adef9>, paper [PMC9824924](https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/) |
| **PDBbind v2020** | general 14,108 complexes / refined 5,050 / core 285 | free registration + licence agreement, academic **and** commercial; **PDBbind+ (2024/2025) is a paid subscription** | <http://www.pdbbind.org.cn/> |
| **Stanford HIVdb** genotype–phenotype | per-gene PR / RT / IN / CAI PhenoSense fold-change sets | public, cite-and-use | <https://hivdb.stanford.edu/pages/genotype-phenotype.html> |
| **BELKA** (Leash) | ~3.6 B DEL binding measurements over 133 M small molecules × 3 targets (BRD4, sEH, HSA) — >1000× BindingDB | Kaggle competition terms; **not clearly open-licensed — check before use** | <https://www.kaggle.com/competitions/leash-BELKA>, <https://leashbio.substack.com/p/introducing-belka> |
| **KinDEL** | DEL dataset for 2 kinases (DDR1, MAPK14), with held-out on-DNA + off-DNA validation | open | [arXiv 2410.08938](https://arxiv.org/pdf/2410.08938) |
| **LIT-PCBA** | 15 targets, 7,844 confirmed actives, 407,381 inactives, from dose–response PubChem bioassays | open | <https://drugdesign.unistra.fr/LIT-PCBA/> |

### 2.2 Hugging Face dataset IDs (row counts and licences pulled live from the HF API, 2026-10-04)

| HF dataset ID | Rows | On-disk | Licence | What it is |
|---|---|---|---|---|
| `martinakaduc/ChEMBL_activities` | **20,334,684** | 1.61 GB | **cc-by-sa-4.0** | the full ChEMBL activity table, ML-ready. Best single starting point. |
| `jglaser/binding_affinity` | **3,673,401** | 456 MB | not declared (BindingDB-derived → treat as CC BY / CC BY-SA) | protein sequence + ligand SMILES + affinity. The default DTA training set. |
| `jglaser/pdbbind_complexes` | **16,079** | 712 MB | not declared | PDBbind complexes with coordinates |
| `THU-ATOM/PDBbind` / `THU-ATOM/PDBbind_test` | — (high download counts: 6.9 k / 19.5 k per 30 d) | — | not declared | widely used PDBbind mirror |
| `katielink/moleculenet-benchmark` | bace 1,512 · bbbp 2,038 · clintox 1,476 · esol 1,127 · freesolv 641 · **hiv 41,126** · sider 1,426 · tox21 7,830 | small | **apache-2.0** | MoleculeNet, one config per task |
| `scikit-fingerprints/MoleculeNet_*` | e.g. `MoleculeNet_HIV` **41,127** rows / 2.0 MB; also ESOL, FreeSolv, Lipophilicity, BACE, BBBP, ClinTox, MUV, SIDER, Tox21, ToxCast | small | `unknown` (declared) | cleanest per-task MoleculeNet split; ~2 k downloads/30 d each |
| `liuganghuggingface/admet` | **52,665** | 5.1 MB | **mit** | aggregated ADMET endpoints |
| `introvoyz041/admetica`, `dbn4/anpdb-admet-ai` | — | — | — | ADMET model-output / platform mirrors |
| `haydn-jones/ZINC20` | **1,922,925,837** (1.92 B) | **174 GB** | **mit** | full ZINC20 SMILES — screening library, far too big for Colab; stream a slice |
| `zpn/zinc20`, `Pixelatory/ZINC20-Druglike`, `YinkaiW/ZINC20-filtered`, `duongttr/zinc20-1M` | 1 M–1 B | — | mit (zpn) | practical ZINC20 subsets |
| `jablonkagroup/*_tox21` (`sr_are_tox21`, `nr_ahr_tox21`, `nr_ar_tox21`, `nr_aromatase_tox21`, …) | per-assay | small | — | ChemNLP-style per-endpoint Tox21 |
| `ibm-research/otter_uniprot_bindingdb_chembl`, `ibm-research/otter_uniprot_bindingdb` | — | — | — | knowledge-graph triples behind Otter-Knowledge, the **current TDC DTI-DG leader** |
| `HUBioDataLab/DrugGEN-chembl-smiles`, `HUBioDataLab/pdbbind_full`, `HUBioDataLab/tox21` | — | — | — | DrugGEN's curated ChEMBL SMILES for generative pretraining |
| `MolGen/ChEMBL-raw`, `MolGen/GuacaMol-raw`, `jxie/guacamol`, `katielink/GuacaMol` | — | — | — | generative-model pretraining corpora |
| `BALM/BALM-benchmark` | — | — | — | binding-affinity LM benchmark incl. a filtered BindingDB config |
| `williamTLmiller/matrixnorm-bindingdb-chembl-ccbysa` | — | — | CC BY-SA (in the name) | explicitly-licensed BindingDB+ChEMBL join |

**Licence hygiene.** ChEMBL and anything derived from it is **CC BY-SA** — share-alike. If we
ship model weights trained on ChEMBL-derived rows, get a view on whether weights are a
derivative work before we attach a licence to them. BindingDB's own curation is CC BY 3.0
(no share-alike), which is the cleaner subset. ZINC20 on HF is declared MIT. BELKA's terms
are Kaggle-competition terms and are **not** a clear open licence — I could not find one.

### 2.3 TDC (Therapeutics Data Commons) — the benchmark harness

TDC ships **66 AI-ready datasets across 22 tasks**, and the **ADMET Benchmark Group** is 22
datasets with fixed scaffold splits and a live leaderboard. Install `PyTDC`; it downloads and
splits for you, which removes most of the data-engineering risk from the project.

| Group | Dataset | n | Task | Metric |
|---|---|---|---|---|
| Absorption | Caco2_Wang | 906 | reg | MAE |
| | HIA_Hou | 578 | bin | AUROC |
| | Pgp_Broccatelli | 1,212 | bin | AUROC |
| | Bioavailability_Ma | 640 | bin | AUROC |
| | Lipophilicity_AZ | 4,200 | reg | MAE |
| | Solubility_AqSolDB | 9,982 | reg | MAE |
| Distribution | BBB_Martins | 1,975 | bin | AUROC |
| | PPBR_AZ | 1,797 | reg | MAE |
| | VDss_Lombardo | 1,130 | reg | Spearman |
| Metabolism | CYP2C9_Veith | 12,092 | bin | AUPRC |
| | CYP2D6_Veith | 13,130 | bin | AUPRC |
| | CYP3A4_Veith | 12,328 | bin | AUPRC |
| | CYP2C9_Substrate | 666 | bin | AUPRC |
| | CYP2D6_Substrate | 664 | bin | AUPRC |
| | CYP3A4_Substrate | 667 | bin | AUROC |
| Excretion | Half_Life_Obach | 667 | reg | Spearman |
| | Clearance_Hepatocyte_AZ | 1,020 | reg | Spearman |
| | Clearance_Microsome_AZ | 1,102 | reg | Spearman |
| Toxicity | LD50_Zhu | 7,385 | reg | MAE |
| | hERG | 648 | bin | AUROC |
| | AMES | 7,255 | bin | AUROC |
| | DILI | 475 | bin | AUROC |

Source: <https://tdcommons.ai/benchmark/admet_group/overview/>, TDC NeurIPS 2021 paper
<https://zitniklab.hms.harvard.edu/publications/papers/TDC-neurips21-main.pdf>.

**The headline fact for planning: the median ADMET endpoint has ~1,000 labels.** These are
small-data problems. A 24 h L4 session is enormous overkill for fitting any one of them; the
binding constraint is *data*, not compute. Budget the GPU for (a) pretraining/representation,
(b) docking and rescoring, (c) generative sampling — not for ADMET heads.

---

## 3. Benchmarks and SOTA numbers

| Benchmark | Metric | SOTA | Source |
|---|---|---|---|
| TDC **Caco2_Wang** (906, scaffold split) | MAE ↓ | **0.256 ± 0.006** (CaliciBoost); XGBoost 0.274, MapLight 0.276, BaseBoosting 0.285, MolMapNet-D 0.287. 24 entries; worst listed 6.328 | <https://tdcommons.ai/benchmark/admet_group/01caco2/> |
| TDC **ADMET group**, overall | mean rank | **ADMET-AI** holds the best average rank across the group | [PMC11226862](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11226862/) |
| TDC **DTI-DG / BindingDB_Patent** (domain generalisation across patent years) | PCC ↑ | **0.588 ± 0.002** (Otter-Knowledge-Ensemble); 2nd: OrangeBalls-ProtBertMorgan **0.538 ± 0.008**; 9 entries, floor ≈ 0.284 | <https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/> |
| **ogbg-molhiv** (41,127 mols) | ROC-AUC ↑ | **0.8476 ± 0.0002** (Multi-RF Fusion + Multi-GNN, 2026-03-21) | <https://ogb.stanford.edu/docs/leader_graphprop/> |
| **ogbg-molpcba** | AP ↑ | **0.3204 ± 0.0001** (HyperFusion) | same |
| **FEP+ benchmark** (hit-to-lead affinity) | Pearson r ↑ | **Boltz-2 = 0.62** vs **full FEP physics = 0.72**, at ~**1000× lower cost**; r = 0.66 on the FEP+ 4-target subset | <https://boltz.bio/boltz2>, <https://jclinic.mit.edu/boltz-2-towards-accurate-and-efficient-binding-affinity-prediction/>, [R&D World](https://www.rdworldonline.com/open-source-boltz-2-can-speed-binding-affinity-predictions-1000-fold/) |
| **CASP16 blind affinity** | Pearson r ↑ | **Boltz-2 = 0.65 out-of-the-box, won the challenge** | same |
| **CASF-2016 scoring power** | 285 complexes (161 of them outside LP-PDBbind train) | retrained-on-LP-PDBbind models beat stock ones; **InteractionGraphNet (IGN)** recommended for scoring/ranking on novel systems | [PMC10462179](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10462179/), [arXiv 2308.09639](https://arxiv.org/html/2308.09639v3) |
| **CACHE #1** (LRRK2 WDR), prospective | confirmed-hit rate | **73 / 1,955 ≈ 3.7 %** at K_D < 150 µM; 7 diverse series, 18–140 µM; best single team 5 hits from 59 synthesised = **8.5 %** | [Chem. Sci. 2024](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc06880c) |
| **CACHE #5** (MCHR1 antagonists) | — | R1: 576 compounds from 15 teams → 75 analogues of 16 hits with pK_i > 5; R2: 32 tested → 18 confirmed antagonists | <https://cache-challenge.org/node/3000> |

### 3.1 The benchmark traps — read before quoting any number

- **DUD-E is biased.** CNN enrichment on DUD-E is attributable to *analogue and decoy bias*,
  not learned molecular recognition ([PMC6701836](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6701836/)).
  LIT-PCBA was built to fix this (15 targets, 7,844 actives / 407,381 inactives) but
  **bias lingers there too**, and LIT-PCBA targets "proved difficult for all methods tested"
  ([benchmark study, 2026](https://arxiv.org/pdf/2605.01681)). Training an ML scoring
  function on these sets risks a model "taught to the test".
- **PDBbind leaks.** Standard random/temporal PDBbind splits put sequence- and
  structure-similar proteins and near-identical ligands on both sides. Use **LP-PDBbind**
  (leak-proof) splits, or report both and say which is which.
- **Pearson r on a congeneric series is not Pearson r across chemotypes.** Boltz-2's 0.62 is
  on FEP+-style congeneric sets. Nothing in the literature says a small model will reproduce
  that across our targets.

---

## 4. Models that actually fit a 24 GB L4 in 24 h

An L4 is 24 GB VRAM, ~120 TFLOP/s BF16 dense, no NVLink, and Colab will interrupt it. Design
for *checkpoint-and-resume* and for runs that finish in 6–12 h so a disconnect costs one run,
not the project.

### 4.1 Pretrained backbones to fine-tune (all fit comfortably)

| HF model ID | Size | Downloads/30 d | Use |
|---|---|---|---|
| `DeepChem/ChemBERTa-77M-MTR` | 77 M params, multi-task-regression pretrained | 166,087 | **first choice** for molecular property/ADMET heads; MTR variant beats MLM on regression |
| `DeepChem/ChemBERTa-77M-MLM` | 77 M | 60,534 | masked-LM variant, better for generative/embedding work |
| `DeepChem/ChemBERTa-10M-MTR` / `-10M-MLM` / `-5M-*` | 5–10 M | 11.6 k / 3.9 k | ablations; trains in minutes |
| `ibm-research/MoLFormer-XL-both-10pct` | ~47 M, pretrained on 10 % of ZINC+PubChem (1.1 B mols) | **237,560** | strongest general SMILES encoder available openly |
| `facebook/esm2_t33_650M_UR50D` | 650 M | 3.37 M | protein side of a DTA model; fine-tune with LoRA on an L4, or freeze and cache embeddings |
| `facebook/esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t6_8M_UR50D` | 8–150 M | 220 k–560 k | **use these** — for ~5 k targets, a 150 M ESM-2 frozen encoder is almost always enough and leaves VRAM for the ligand branch |
| `clefourrier/graphormer-base-pcqm4mv2` | — | 1,198 | graph-transformer baseline |
| `unikei/bert-base-smiles`, `JuIm/SMILES_BERT` | — | 1.4 k / 218 | alternative SMILES tokenisers |
| `dptech/Uni-Mol-Models`, `dptech/Uni-Mol2` | — | — | 3D-aware molecular representation; heavier |

**Critical L4 tactic: freeze the protein encoder and pre-compute embeddings once.** Our
target list is ~40 proteins, not 5,000. One forward pass of ESM-2 650M over 40 sequences is
seconds; caching it turns a DTA model into a small ligand-side network that trains in under
an hour, and frees the whole 24 GB for batch size.

### 4.2 Model families, matched to the output we want

1. **Ligand-only QSAR per target** (the honest workhorse). Chemprop-style D-MPNN or
   ChemBERTa fine-tune on ChEMBL actives for each UniProt in §1. Trains in minutes. Beats
   fancier things whenever the target has >500 measured compounds — which is true for DPP-4,
   PPARγ, JAK1/2, PDE4B, CYSLTR1, ADRB2, HIV PR/RT/IN, and false for ERAP1, TSLP, GPR119.
2. **Cross-target DTA** (`jglaser/binding_affinity`, 3.67 M rows). Ligand encoder +
   cached ESM-2 target embedding → pK_d. This is what gives predictions on the
   *data-poor* targets, which are exactly the unmet-need ones. Benchmark against the TDC
   DTI-DG number (0.588 PCC) and expect to land well below it.
3. **Structure-based rescoring.** Dock with GPU docking, rescore with a learned function.
   **Uni-Dock** reports >1000× speedup over single-core AutoDock Vina on a V100 32G, and
   >37,000 dockings per GPU-hour in production screens
   ([Uni-Dock paper](https://cdn.dp.tech/home/paper/yu-et-al-2023-uni-dock-gpu-accelerated-docking-enables-ultralarge-virtual-screening.pdf),
   <https://github.com/dptech-corp/Uni-Dock>). At that rate an L4 session can dock ~500 k–900 k
   compounds — enough for a real screen of a curated library per target. `Vina-GPU 2.1`
   is the other option ([biorxiv](https://www.biorxiv.org/content/10.1101/2023.11.04.565429.full.pdf)).
   GNINA 1.0 gives CNN rescoring ([PMC8659095](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659095)).
4. **Co-folding / affinity oracle, not trained by us.** **Boltz-2** is open and is the only
   open model near FEP accuracy (r = 0.62 vs 0.72, 1000× faster; won CASP16 affinity at
   r = 0.65). Use it as an *expensive reranker on the top ~200 per target*, not as a screen.
   DiffDock-L is the ML-docking counterpart with confidence bootstrapping.
5. **Generative / optimisation.** **REINVENT 4** (Apache-2.0, <https://github.com/MolecularAI/REINVENT4>)
   does RL-driven de novo design, R-group replacement, linker design and scaffold hopping with
   multi-component scoring — i.e. it can optimise *our* QSAR + ADMET predictors jointly. Runs
   on one GPU. Pretrain or load priors from `MolGen/ChEMBL-raw` / GuacaMol corpora.
6. **ADMET.** Do not reinvent: **ADMET-AI** (Chemprop-RDKit — a Chemprop D-MPNN plus 200 RDKit
   descriptors, Chemprop v1.6.1) has the best average rank on the TDC ADMET leaderboard and
   scores **1 M molecules in 3.1 h locally**
   ([PMC11226862](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11226862/),
   [Zenodo v1.0.0](https://zenodo.org/record/10372418)). Retrain it on TDC splits as our
   baseline and only try to beat it on the endpoints we care about (hERG, CYP3A4, BBB, LD50).

---

## 5. Suggested pipeline (one L4 session per stage)

1. **Build the target card set.** 40 targets from §1, each with UniProt, best PDB, cleaned
   ChEMBL actives (pChEMBL ≥ 5, single-protein assay, `standard_relation = '='`), and a
   scaffold split. Deliverable: a parquet per target + a count table. No GPU.
2. **Ligand QSAR per target** — ChemBERTa-77M-MTR fine-tune, plus an XGBoost-on-ECFP4
   baseline. *Always ship the baseline number*: on small ADMET-scale data, boosted trees
   win the TDC Caco2 leaderboard's 2nd–4th places.
3. **Cross-target DTA** on `jglaser/binding_affinity` with cached ESM-2 embeddings, for the
   data-poor targets. Report PCC on a *target-disjoint* split, not a random one.
4. **ADMET panel** — retrain ADMET-AI/Chemprop-RDKit on the 22 TDC endpoints; report per-
   endpoint metric against the leaderboard so the gap is visible.
5. **Screen** — Uni-Dock a curated library (ZINC20 drug-like slice + ChEMBL + DrugBank for
   repurposing) against the docking-grade structures (skip 2Q7M at 4.25 Å and 5U1C at 3.9 Å).
6. **Rerank** top 200/target with Boltz-2 affinity + GNINA rescoring; require agreement
   between orthogonal scorers before anything is called a candidate.
7. **Rank and publish** — per target, a table of molecule, predicted pAffinity with
   uncertainty, each ADMET flag, nearest known active (Tanimoto) and whether it is a known
   drug. **Report the applicability domain**: a prediction on a scaffold far from training
   data is a guess and must be labelled as one.

---

## 6. Calibration: what to promise

- Quote **CACHE #1's 3.7 %** as the expected prospective hit rate, and 18–140 µM as the
  expected potency of a first hit. Nanomolar hits do not come out of a first screen.
- The best open affinity model (Boltz-2) hits r ≈ 0.62–0.66. A model we train on an L4 in a
  day will be worse. Report r, Spearman *and* MAE in log units, on a leakage-controlled split.
- Say which split was used, every time. LP-PDBbind vs random PDBbind changes the headline
  number more than any architecture choice.
- **Never** present output as a drug, a treatment, or clinical advice. The honest framing is:
  "ranked, uncertainty-quantified in-silico hypotheses against a named structure, for
  experimental triage."
- Prefer targets where a negative is informative: PTP1B (no approved drug), ERAP1 (none),
  TSLP (antibody-only) are better science than re-finding sitagliptin at DPP-4 — though
  *re-finding sitagliptin should be the validation test* that the pipeline works at all.

---

## 7. Gaps I could not close

- **BELKA's licence.** Kaggle competition terms only; no CC licence found. 3.6 B measurements
  is transformative for this project if usable — worth an email to Leash.
- **PDBbind+ pricing.** The 2024/2025 editions are a paid subscription; v2020 remains free
  with registration. Size numbers for v2020 differ slightly between sources
  (14,108 vs 14,127 general; 5,050 vs 5,316 refined) — pin the exact release before quoting.
- **Full TDC ADMET SOTA table.** I pulled Caco2 in full; the other 21 leaderboards need one
  fetch each (`https://tdcommons.ai/benchmark/admet_group/<nn><name>/`).
- **A newer FLAP structure** than 2Q7M (4.25 Å) — needed before ALOX5AP can be docked.
- **T1D small-molecule precedent** is thin by nature; I found no approved small molecule, so
  §1.3 is mechanism-hypothesis only and should be presented that way.

---

### Sources

- ChEMBL 36: <https://www.ebi.ac.uk/about/news/updates-from-data-resources/chembl-36/>
- BindingDB 2024 paper: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11701568/> · stats <https://bindingdb.org/rwd/bind/info.jsp>
- Papyrus: <https://zenodo.org/records/10944245> · <https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/>
- TDC: <https://tdcommons.ai/benchmark/admet_group/overview/> · <https://tdcommons.ai/benchmark/dti_dg_group/bindingdb_patent/> · <https://zitniklab.hms.harvard.edu/publications/papers/TDC-neurips21-main.pdf>
- OGB leaderboard: <https://ogb.stanford.edu/docs/leader_graphprop/>
- ADMET-AI: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11226862/> · <https://zenodo.org/record/10372418>
- Boltz-2: <https://boltz.bio/boltz2> · <https://jclinic.mit.edu/boltz-2-towards-accurate-and-efficient-binding-affinity-prediction/> · <https://www.rdworldonline.com/open-source-boltz-2-can-speed-binding-affinity-predictions-1000-fold/>
- LP-PDBbind: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10462179/> · <https://arxiv.org/html/2308.09639v3>
- DUD-E bias: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6701836/> · LIT-PCBA benchmarking <https://arxiv.org/pdf/2605.01681>
- GNINA 1.0: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8659095>
- Uni-Dock: <https://github.com/dptech-corp/Uni-Dock> · <https://cdn.dp.tech/home/paper/yu-et-al-2023-uni-dock-gpu-accelerated-docking-enables-ultralarge-virtual-screening.pdf> · Vina-GPU 2.1 <https://www.biorxiv.org/content/10.1101/2023.11.04.565429.full.pdf>
- REINVENT 4: <https://github.com/MolecularAI/REINVENT4> · <https://chemrxiv.org/engage/chemrxiv/article-details/65463cafc573f893f1cae33a>
- CACHE: <https://pubs.rsc.org/en/content/articlehtml/2024/sc/d3sc06880c> · <https://cache-challenge.org/node/3000>
- BELKA: <https://www.kaggle.com/competitions/leash-BELKA> · <https://leashbio.substack.com/p/introducing-belka> · KinDEL <https://arxiv.org/pdf/2410.08938>
- Stanford HIVdb: <https://hivdb.stanford.edu/pages/genotype-phenotype.html>
- Orforglipron: <https://investor.lilly.com/news-releases/news-release-details/lillys-oral-GLP-1-orforglipron-delivered-superior-blood-sugar> · <https://www.pharmacytimes.com/view/orforglipron-outperforms-oral-semaglutide-in-head-to-head-t2d-trial>
- AS therapy: <https://www.healio.com/news/rheumatology/20240923/fda-oks-bimzelx-for-psoriatic-arthritis-ankylosing-spondylitis-nonradiographic-axspa> · <https://www.hcplive.com/view/bimekizumab-approved-psoriatic-arthritis-ankylosing-spondylitis-axspa> · <https://www.nice.org.uk/guidance/ta918/documents/final-scope>
- Structures/accessions verified via <https://data.rcsb.org/rest/v1/core/entry/>, <https://rest.uniprot.org/uniprotkb/>, <https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/1hsg>
