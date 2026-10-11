# More ADMET data — what exists, verified against the live APIs

Research date **2026-10-04**. Every count below was retrieved by a real call to PubChem PUG-REST /
NCBI E-utilities, the ChEMBL REST API, the Hugging Face dataset + datasets-server APIs, Europe PMC,
or by downloading the file and counting its rows. Numbers are **retrieved, not estimated**. Anything
I did *not* re-fetch is tagged `[unverified]`.

**§10 is an addendum added 2026-10-05** that closes every `≈` and `[unverified]` the first pass left,
and contains the worst leakage result in the document (§10.1) plus a correction to a premise
`PROGRESS.md` is currently acting on (§10.8). Read §0, §10.1 and §10.8 first.

No repo code was changed by this task.

---

## 0. The baseline, re-measured

From `talindrew-drugrx/data/admet_multitask.parquet` as it stands today:

| endpoint | n labels | train | test | endpoint | n labels | train | test |
|---|---|---|---|---|---|---|---|
| caco2 | 906 | 519 | 181 | cyp2d6_inhib | 13,130 | 5,469 | 2,626 |
| bioavailability | 640 | 163 | 128 | cyp3a4_inhib | 12,328 | 4,872 | 2,467 |
| lipophilicity | 4,200 | 2,685 | 420 | cyp2c9_sub | 666 | 172 | 135 |
| solubility | 9,982 | 6,472 | 1,997 | cyp2d6_sub | 663 | 171 | 134 |
| hia | 578 | 263 | 117 | cyp3a4_sub | 667 | 173 | 134 |
| pgp | 1,212 | 699 | 245 | half_life | 665 | 236 | 134 |
| bbb | 1,965 | 1,036 | 394 | cl_micro | 1,102 | 485 | 221 |
| ppbr | 1,614 | 791 | 324 | ld50 | 7,342 | 4,710 | 1,466 |
| vdss | 1,111 | 670 | 221 | herg | 645 | 380 | 128 |
| cyp2c9_inhib | 12,092 | 4,844 | 2,419 | ames | 7,255 | 4,700 | 1,453 |
| | | | | dili | 475 | 171 | 96 |

**79,238 labels / 46,061 rows / 21 endpoints.** Train 26,972, valid 5,776, test 13,313.

### Finding 0 — we already have a leak, and it is 441 rows wide

Before adding anything, I standardised our own 46,061 SMILES with RDKit MolStandardize
(Normalizer → LargestFragmentChooser → Uncharger → InChIKey):

```
46,061 rows  →  44,561 distinct full InChIKeys  →  41,324 distinct InChIKey skeleton blocks
```

So 1,500 of our "unique molecules" are the same compound under a different salt or charge form, and
3,237 more are tautomer/stereo variants of one another. Consequences, measured:

- **441 rows currently in `split == 'train'` have the same standardised InChIKey as a molecule we
  hold out as test.** By raw SMILES the split is perfectly clean (0 violations) — the leak only
  appears after salt stripping. 442 full-InChIKey groups straddle the test boundary.
- 25 cases are a straight leaderboard contamination: the same standardised molecule is both *test
  and train for the same endpoint*. By endpoint: `solubility` 8, `herg` 6, `cyp2c9_inhib` 3,
  `cyp2d6_inhib` 3, `cyp3a4_inhib` 3, `bbb` 1, `ames` 1. At skeleton level, 26.
- 6 of those are on `herg`, whose test fold is 128 molecules — **4.7% of the hERG test set has an
  exact-molecule twin in training.** That is roughly the size of the gap we have been chasing with
  architecture changes.

This must be fixed before any new data arrives, because every recipe in §6 assumes the existing
boundary is sound.

---

## 1. PubChem BioAssay

### 1.1 The encoding, from NCBI's own spec

Fetched `https://ftp.ncbi.nlm.nih.gov/pubchem/specifications/pubchem.asn`, `PC-AssayResults`:

```
outcome  INTEGER { inactive (1), active (2), inconclusive (3),
                   unspecified (4), probe (5) }  DEFAULT unspecified
```

Three traps in those five lines:

1. **`DEFAULT unspecified`.** A row with no deposited outcome reads as 4, not 1. It is an absent
   measurement, not a negative.
2. **`inconclusive(3)` is not `inactive(1)`.** The qHTS curve fit failed, or the compound was
   cytotoxic/fluorescent/insoluble at the read concentration. Treating it as inactive injects
   systematic noise that correlates with *physicochemistry* — exactly the signal the model is trying
   to learn — so it does not average out. Measured incidence in the assays below: **AID 1851 is
   22.4% inconclusive (19,186 of 85,715 rows); the NCATS PAMPA assay AID 1508612 is 19.6% (495 of
   2,530); the Tox21 CYP3A4 assay AID 2284096 is 29.2% (2,820 of 9,668); the Tox21 hERG assay
   AID 2284432 is 21.9% (2,118 of 9,668).** Dropping them is the only defensible default.
3. **`probe(5)` is not an inactive either** — it marks a designated MLP probe, i.e. a strong active.

Rule: keep `Active` and `Inactive`, drop `Inconclusive`, `Unspecified`, blank; treat `Probe` as
active or drop it (there are very few).

### 1.2 The CYP qHTS panel — already fully mined, but with free neighbours

`AID 1851` — "Cytochrome panel assay with activity outcomes", NCATS/NCGC, deposited 2009-07,
**17,143 SIDs / 16,560 CIDs**. Its concise CSV is 85,715 rows = 17,143 substances × 5 isozymes:

| GeneID | isozyme | Active | Inactive | Inconclusive | usable (A+I) | our TDC n |
|---|---|---|---|---|---|---|
| 1544 | CYP1A2 | 6,000 | 7,256 | 3,887 | 13,256 | *not used* |
| 1557 | CYP2C19 | 5,913 | 7,532 | 3,698 | 13,445 | *not used* |
| 1559 | CYP2C9 | 4,119 | 8,782 | 4,242 | 12,901 | 12,092 |
| 1565 | CYP3A4 | 2,771 | 11,139 | 3,233 | 13,910 | 12,328 |
| 1576 | CYP2D6 | 5,266 | 7,751 | 4,126 | 13,017 | 13,130 |
| | **total** | 24,069 | 42,460 | 19,186 | 66,529 | |

TDC's `cyp{2c9,2d6,3a4}_veith` are this assay, Active+Inactive, minus a few hundred to CID/SMILES
dedup. **There is nothing left to take from AID 1851 for our three endpoints.** What *is* left is
the two columns we never used: **CYP1A2 (13,256) and CYP2C19 (13,445) are 26,701 labels on
molecules that are already in our table and already featurised.** Zero new molecules, zero new
leakage surface, two new heads. This is the cheapest gain in the whole report (§5).

Individual qHTS assays behind the panel, all NCATS, all verified:

| AID | isozyme | SIDs | CIDs | Active | Inactive | Inconclusive | CIDs not in 1851 |
|---|---|---|---|---|---|---|---|
| 410 | CYP1A2 | 9,198 | 8,354 | 4,446 | — | — | 487 |
| 883 | CYP2C9 | 10,320 | 9,385 | 1,327 | — | — | 838 |
| 884 | CYP3A4 | 14,155 | 13,076 | 3,571 | 7,657 | 2,927 | **4,526** |
| 891 | CYP2D6 | 10,320 | 9,385 | 1,776 | — | — | 838 |
| 899 | CYP2C19 | 10,320 | 9,385 | 1,964 | — | — | 838 |

AID 884 carries 6,846 rows with an AC50 in µM — a *regression* target for cyp3a4 inhibition where
TDC gives only a binary one. Its 4,526 CIDs outside AID 1851 are the only material increment here
and need a standardised-InChIKey check before being believed.

```bash
# the panel (CSV endpoint refuses it — see the trap below)
curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1851/concise/CSV' -o aid1851.csv
# a single assay under 10,000 SIDs, with the depositor's own result columns
curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/883/CSV' -o aid883.csv
```

> **Trap — the 10,000-SID cap.** `/rest/pug/assay/aid/<AID>/CSV` answers
> `400 PUGREST.BadRequest / "Too many SIDs" / "Assay record retrieval is limited to 10000 SIDs"`
> for AID 1851 (17,143) and AID 884 (14,155). The `/concise/CSV` variant has no such cap but drops
> the depositor's result columns (AC50, HITC, …) and returns one row per (SID, target). For the full
> matrix on a large assay, page with explicit `?sid=` lists, or take the bulk files:
> `https://ftp.ncbi.nlm.nih.gov/pubchem/Bioassay/CSV/Data/` — **2,015 zips, 1,000 AIDs each**
> (`0000001_0001000.zip` … `2284001_2285000.zip`).

### 1.3 The Tox21 / EPA ToxCast re-deposit of September 2026 — the biggest new thing in PubChem

EPA ToxCast deposited the Tox21 assay suite into PubChem in **2026-09**: 494 assays with ≥5,000
SIDs. Each carries **9,667 SIDs / 7,668 CIDs**, and — unlike the 2009 panel — ships `AC50` (µM),
`HITC` (continuous hit-call, −1…1) and `BMD` (µM) columns. The ones that hit our endpoints:

| AID | assay | Active | Inactive | Inconclusive | our endpoint |
|---|---|---|---|---|---|
| 2284391 | TOX21_p450_CYP2C9_Antagonist | 3,655 | 5,710 | 302 | cyp2c9_inhib |
| 2284394 | TOX21_p450_CYP2D6_Antagonist | 3,894 | — | — | cyp2d6_inhib |
| 2284096 | TOX21_p450_CYP3A4_Antagonist | 2,713 | 4,134 | 2,820 | cyp3a4_inhib |
| 2284393 | TOX21_p450_CYP1A2_Antagonist | 5,104 | — | — | *auxiliary* |
| 2284395 | TOX21_p450_CYP2C19_Antagonist | 3,066 | — | — | *auxiliary* |
| 2284432 | TOX21_hERG_U2OS_Antagonist | 1,289 | 6,260 | 2,118 | herg |

**The library is almost disjoint from the 2009 MLSMR panel.** Measured at CID level: only **419** of
the Tox21 library's 7,668 CIDs appear in AID 1851 — **7,249 new molecules**. Tox21 is environmental
chemicals, pesticides, food additives and marketed drugs; MLSMR is screening-deck med-chem. That
disjointness is the point: it extends the applicability domain rather than densifying it.

> **Trap — the same data deposited twice.** `AID 2284432` (EPA ToxCast, 2026-09) and `AID 1671200`
> (Tox21, 2021-09) are both "Tox21 hERG U2OS antagonist", both 9,667 SIDs, and share **7,400 CIDs**.
> They are one experiment under two AIDs. Merging both inflates your hERG count by ~7k and makes a
> duplicate look like agreement between independent assays.

### 1.4 hERG

| AID | name | SIDs | Active | source |
|---|---|---|---|---|
| 1511 | Primary cell-based HTS … hERG (thallium flux) | 305,679 | 1,552 | Johns Hopkins ICCB |
| 2648 / 2642 / 1672 | KCNQ1 / KCNQ1 / Kcnj2 counterscreens | 305,679 | 1,082 / 3,880 / 2,598 | Johns Hopkins |
| 720551 / 720553 | KCNH2 **3.1** wildtype / mutant qHTS | 343,909 / 342,311 | 1,267 / 3,260 | NCATS |
| 2284432 | TOX21_hERG_U2OS_Antagonist | 9,667 | 1,289 | EPA ToxCast |
| 1671200 | Tox21 hERG fluorescence cell-based | 9,667 | 704 | Tox21 (dup of above) |
| 588834 | qHTS Assay for Inhibitors of the Human hERG Channel | 5,381 | 664 | NCATS, U2OS, KCNH2 |

AID 1511 is the 300k-compound screen behind TDC's `herg_central` (§3). **AIDs 720551/720553 are
KCNH2 *3.1*, a brain-selective isoform, not cardiac hERG1** — 686k rows that look like the jackpot
and are a different protein. Do not merge them into `herg`.

### 1.5 NCATS Tier-I ADME profiling — permeability, solubility, microsomal stability

Siramshetty et al. 2021, *SLAS Discovery*, PMID **34176369**. All continuous, all NCATS-deposited:

| AID | assay | rows | Active | Inactive | Inconclusive | result column | our endpoint |
|---|---|---|---|---|---|---|---|
| 1645848 | Kinetic Aqueous Solubility Profiling | 2,532 | 1,056 | 1,400 | 76 | `Kinetic Aqueous Solubility (ug/mL)` | solubility |
| 1508612 | PAMPA Profiling (pH 7.4) | 2,530 | 1,740 | 295 | 495 | `Permeability` | caco2 (proxy) |
| 1508591 | Rat Liver Microsome Stability | 2,531 | 754 | 1,777 | 0 | `Half-life (minutes)` | cl_micro |
| 1963597 | Human Liver Microsomal (HLM) Stability | 900 | 542 | 358 | 0 | `Half-life` | cl_micro |
| 1845228 | PAMPA — Blood-Brain Barrier | 438 | 267 | 171 | 0 | `Permeability` | bbb |
| 1645871 | PAMPA in pH 5 buffer | 486 | 443 | 43 | 0 | `Permeability` | caco2 (proxy) |
| 1508603 | Human Liver Cytosol Stability | 189 | 80 | — | — | | cl_micro |
| 1508604 | Mouse Liver Cytosol Stability | 57 | 32 | — | — | | — |

TDC mirrors the *expanded* versions of two of these as `hlm` (6,013) and `rlm` (5,590) and the PAMPA
set as `pampa_ncats` (2,034) — all three larger than the PubChem deposits, so take them from TDC
(§3) and use PubChem only for the 1845228 / 1645871 / 1508603 extras.

### 1.6 Finding the large assays at all

PubChem's Entrez index exposes `TSC` (Total Sid Count) as a **range-searchable** field, which is the
only sane way to find big assays among 2.28M AIDs:

```bash
# every hERG-mentioning assay with ≥3,000 substances
curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pcassay&retmode=json&retmax=200&term=hERG+AND+3000:100000000[TSC]'
# then esummary for name / activesidcount / totalsidcount / activityoutcomemethod / proteintargetlist
curl -s 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pcassay&retmode=json&id=588834,2284432'
```

Other usable fields from `einfo.fcgi?db=pcassay`: `AC` (ActiveSidCount), `ACMD`
(ActivityOutcomeMethod: screening/confirmatory/summary/other), `GSYM`, `UPAC`, `SNME`, `DTMD`.
Searching plain text (`"Tox21"`, `"PAMPA permeability"`) returns thousands of three-compound ChEMBL
literature deposits and buries the real assays; always add the `[TSC]` range.

### 1.7 Where PubChem has nothing useful

- **Caco-2**: two assays with ≥300 substances, and neither is permeability (`AID 1409598` is a
  SARS-CoV-2 cytotoxicity screen in Caco-2 cells; `AID 377` is PDSP MDR-1 binding, 779 substances).
  Caco-2 lives in ChEMBL, not PubChem (§2.3).
- **Ames**: 2,178 text matches, almost all three-compound ChEMBL literature deposits. The Tox21
  genotoxicity surrogates (`2283922` / `2283923` DT40 DNA-damage, 10,496 substances each;
  `sr_atad5`, `sr_p53`) are *auxiliary* tasks, not Ames labels.
- **dili, bioavailability, hia, half_life, vdss**: nothing at scale. These are clinical/in-vivo
  endpoints and PubChem is an HTS archive.

**Licence**: PubChem/NCBI data is US-government public domain; depositor-asserted rights are
possible but none of the AIDs above carry one. Safe for commercial use.

---

## 2. ChEMBL

**Licence first: ChEMBL data is CC BY-SA 3.0 Unported** (verified on
`chembl.gitbook.io/chembl-interface-documentation/about`). *Share-alike* is the live commercial
question — see §9.

### 2.1 The censoring trap, quantified on hERG

This is real and it is large. `CHEMBL240` (hERG):

| query | count |
|---|---|
| all activities | 41,078 |
| `standard_type=IC50` | 19,130 |
| `standard_type=IC50 & pchembl_value__isnull=false` | 11,709 |
| `standard_type=IC50 & standard_relation='='` | 12,021 |
| `standard_type=IC50 & standard_relation='>'` | **4,773** |
| `standard_type=IC50 & standard_relation='<'` | 290 |
| `standard_type=IC50 & standard_relation` null | 2,022 |
| `standard_type=IC50 & pchembl_value__isnull=false & standard_relation='>'` | **0** |

ChEMBL only computes `pchembl_value` for an exact relation. So **filtering on `pchembl_value`
deletes all 4,773 `IC50 > x` records, and those are precisely the clean non-blockers** — a compound
tested to 30 µM with no inhibition. For a hERG *classifier* that filter removes the negative class
and leaves you training on actives versus nothing. Pull `standard_relation`, `standard_value`,
`standard_units` and `standard_upper_value` and censor explicitly; a `>` at 30 µM is a label
("inactive at 30 µM"), not a missing value.

The same arithmetic holds elsewhere: ChEMBL ADME `PPB` has 866 `>` and 3,702 rows with **no relation
at all**; ADME `Papp` has 539 `<` and 161 `>`; human ADME `T1/2` has 1,887 `>`; human ADME `CL` has
1,566 `<`.

And the loss is measurable in molecules, not just rows. Standardising the 19,130 hERG IC50 rows:

| | distinct IK | already in ours | collides with our 128-molecule `herg` test fold | **new molecules** |
|---|---|---|---|---|
| all relations | **15,937** | 1,520 | 53 | **14,417** |
| relation `=` only | 9,764 | 673 | 46 | 9,091 |

**Insisting on an exact relation throws away 5,326 hERG molecules** — more than eight times the size
of TDC's entire `herg` dataset (645). Those 5,326 are the compounds somebody tested to 10 or 30 µM
and found clean, i.e. the class a safety model most needs to see.

Note the other number in that table: **53 of our 128 `herg` test molecules (41%) are in ChEMBL's
hERG IC50 rows**, because TDC's `herg` and `herg_karim` are themselves ChEMBL-derived. §6.3 is not
optional here.

### 2.2 How TDC built its ChEMBL-derived sets — and exactly what it left behind

`CHEMBL3301361` — "Experimental in vitro DMPK and physicochemical data on a set of publicly
disclosed compounds" (AstraZeneca). **11,687 activities over 15 assays.** Per-assay counts,
retrieved:

| assay | description | activities | TDC |
|---|---|---|---|
| CHEMBL3301363 | octanol/water logD pH 7.4, shake flask | **4,200** | = `lipophilicity_astrazeneca` |
| CHEMBL3301365 | % bound, **human** plasma, equilibrium dialysis | **1,614** | = `ppbr_az` |
| CHEMBL3301370 | CLint, **human liver microsomes** | **1,102** | = `clearance_microsome_az` |
| CHEMBL3301371 | CLint, **rat hepatocytes** | 837 | — |
| CHEMBL3301372 | CLint, **human hepatocytes** | 408 | ≈ `clearance_hepatocyte_az` (1,213) |
| CHEMBL3301364 | solubility in pH 7.4 buffer, solid start | **1,763** | — |
| CHEMBL3301366 | % bound, rat plasma | 717 | — |
| CHEMBL3301367 | % bound, dog plasma | 244 | — |
| CHEMBL3301368 | % bound, mouse plasma | 162 | — |
| CHEMBL3301369 | % bound, guinea pig plasma | 91 | — |
| CHEMBL3301362/73/74/75/76 | pKa B1/B2/B3/A1/A2 | 261/40/2/221/25 | — |

Three of our endpoints are *exactly* one AZ assay each — 4,200, 1,614 and 1,102 match our
`lipophilicity`, `ppbr` and `cl_micro` counts to the row. So the construction is: take the assay,
keep every activity, done. Which means **what TDC excluded is a precise, enumerable list:
1,763 + 717 + 244 + 162 + 91 + 837 + 549 (pKa) = 4,363 activities from a document we already use**,
same depositor, same protocol family, same compounds. Zero new curation risk.

Reproduce the whole document in two calls:

```bash
curl -s 'https://www.ebi.ac.uk/chembl/api/data/assay.json?document_chembl_id=CHEMBL3301361&limit=50'
curl -s 'https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id=CHEMBL3301364&limit=1000'
```

ChEMBL holds **465 documents with `doc_type=DATASET`** (`/document.json?doc_type=DATASET&limit=100`);
`CHEMBL3301361` is the only large ADME one. The rest are MMV/DNDi/GSK phenotypic boxes, DrugMatrix,
GSK PKIS, DailyMed labels and FDA approval packages.

### 2.3 `assay_type=A` — Caco-2, human half-life and human clearance, measured

`assay_type=A` (ADME) is the filter that makes ChEMBL tractable: **1,168,142 activities**. By
`standard_type`:

| standard_type | activities (assay_type=A) | distinct molecules | notes |
|---|---|---|---|
| `Papp` | 20,058 | 14,042 | downloaded in full |
| `CL` | 81,830 | — | **`target_organism='Homo sapiens'` → 11,725 rows / 9,550 molecules** |
| `T1/2` | 87,243 | — | **`target_organism='Homo sapiens'` → 9,279 rows / 7,440 molecules**, units `hr` 9,261 |
| `Vdss` | 16,637 | 11,333 | organism: 7,175 null / 5,554 rat / 1,453 mouse — human share is tiny |
| `PPB` | 14,606 | 8,790 | organism: 9,649 null / **2,220 human** / 1,696 rat |
| `Solubility` | 1,787 | 1,357 | small; AqSolDB is better |
| `LogD7.4` | 4,557 (all types) | 4,523 | 4,200 are the AZ assay; **units null on all 4,557 rows** and relation null on 4,202 |

**The human-PK filter changes the picture for two of our weakest endpoints.** `half_life` has 665
labels today and ChEMBL holds **7,440 distinct molecules with a human `T1/2` in hours**; `cl_micro`
has 1,102 and ChEMBL holds **9,550 distinct molecules with a human `CL`**. Censoring is heavy and
must be kept: human `T1/2` is 7,123 `=` / **1,887 `>`** / 189 `<`; human `CL` is 9,541 `=` /
**1,566 `<`** / 316 `>` / 280 with no relation.

> **Trap — `standard_type='CL'` is two different quantities under one name.** Of the 11,725 human
> rows, **6,213 are `mL.min-1.g-1`** (intrinsic clearance normalised to protein mass — a *microsomal*
> quantity, which is our `cl_micro`) and **4,263 are `mL.min-1.kg-1`** (in-vivo plasma clearance
> normalised to body weight — a different endpoint entirely), plus 268 `L/hr` and 282 unitless.
> Merging them into one `cl_micro` head mixes an in-vitro rate with an in-vivo one; they differ by
> orders of magnitude and by mechanism. Split on `standard_units`, not on `standard_type`.

Measured against our table after identical standardisation:

| source | endpoint | distinct IK | in ours | anyTest | **epTest** | **new** |
|---|---|---|---|---|---|---|
| ChEMBL human `T1/2` (all relations) | half_life | 7,402 | 454 | 207 | 31 | **6,948** |
| ChEMBL human `T1/2`, relation `=` only | half_life | 5,560 | 420 | 198 | 30 | 5,140 |
| ChEMBL human `CL`, `mL.min-1.g-1` (intrinsic) | cl_micro | 5,766 | 543 | 183 | **76** | **5,223** |
| ChEMBL human `CL`, `mL.min-1.kg-1` (plasma) | *(new endpoint)* | 3,125 | 535 | 330 | 17 | 2,590 |

Two things fall out of those four rows.

**The censoring filter costs 26% of the new data.** Restricting human `T1/2` to `standard_relation
= '='` drops 6,948 new molecules to 5,140 — **1,808 molecules deleted**, almost all of them real
measurements reported as a bound (1,887 rows `>`, 189 `<`, 38 with no relation). On a 665-label
endpoint that is not a rounding error; it is nearly three times the whole current dataset.

**ChEMBL's human intrinsic clearance contains 76 of our 221 `cl_micro` test molecules — 34% of that
test fold.** This is not a coincidence: `clearance_microsome_az` *is* ChEMBL assay CHEMBL3301370,
so pulling `standard_type=CL` re-imports a third of the benchmark we score on. Merge this source
without the `HELD_OUT` diff in §6.3 and the `cl_micro` number becomes meaningless.

Breaking the 20,058 `Papp` rows down by assay system:

| system | activities | distinct SMILES | standardised InChIKeys |
|---|---|---|---|
| Caco-2 | 10,397 | 6,930 | **6,911** |
| MDCK | 6,210 | 4,659 | 4,656 |
| PAMPA | 1,017 | 916 | — |
| other | 2,430 | — | — |

Against our table, after identical RDKit standardisation:

| source | distinct IK | already in ours | collides with **any** test fold | collides with **caco2** test fold | **new molecules** |
|---|---|---|---|---|---|
| ChEMBL Caco-2 `Papp` | 6,911 | 420 | 156 | 25 | **6,491** |
| ChEMBL MDCK `Papp` | 4,656 | 160 | 72 | 5 | **4,496** |

**TDC `caco2_wang` is 910 molecules (906 after our canonical dedup). ChEMBL has 6,491 Caco-2
molecules we have never seen.** caco2 is our smallest regression endpoint and this is a 7.6×
increase.

> **Trap — the unit field is free text and it is a swamp.** The 20,058 `Papp` rows carry **37
> distinct `standard_units` strings**, including `10^-6 cm/s` (5,785), `10'-6 cm/s` (4,327 — a
> different string, same unit), `ucm/s` (4,090), `nm/s` (2,874), `10'6cm/s` (183),
> `cm/s * 10E6` (116), `cm s-1` (101), `10^-8cm/s` (54), `cm2/sec` (1), plus **1,540 rows with no
> units at all**. A naïve `standard_value` column is three orders of magnitude wide for no physical
> reason. You must write an explicit harmonisation map (`ucm/s` ≡ `10^-6 cm/s`; `nm/s` ≡
> `10^-7 cm/s`; drop `cm2/sec` and `microg/cm2` as a different quantity), and drop the unitless rows
> rather than guess.
>
> **Trap — ChEMBL ADME PK is mostly rodent.** `Vdss` is 5,554 rat + 1,453 mouse + 7,175 with no
> organism recorded; `PPB` has only 2,220 human rows. Our `vdss`, `ppbr` and `half_life` endpoints
> are human. An unfiltered merge trains a human head on rat data — and `target_organism` is null
> more often than it is "Homo sapiens", so the filter you need is `assay_description` text, not the
> organism field.

### 2.4 Ames, LD50, DILI in ChEMBL

`assay_type=T` (toxicity) holds 1,603,013 activities but `standard_type=LD50` is only 6,981 of them
(14,215 across all assay types) and is overwhelmingly non-mammalian or non-oral. ChEMBL is not the
route for `ld50`, `ames` or `dili` — §4.3 is.

---

## 3. Hugging Face — verified by API

Row counts from `datasets-server.huggingface.co/size`, licences from the dataset card's `cardData`.
Where the server reported `rows=1` (a parse failure on the repo's plain CSV) I downloaded the CSV
and counted it; those are marked †.

### 3.1 `scikit-fingerprints/*` — the TDC mirror, and it contains more TDC than we are using

| dataset | rows | licence on card | relevance |
|---|---|---|---|
| `TDC_herg_central_inhib` | **306,893** † | not declared | herg — 293,149 neg / 13,744 pos |
| `TDC_herg_central_at_1um` | **306,893** † | not declared | herg — **continuous % inhibition** |
| `TDC_herg_central_at_10um` | **306,893** † | not declared | herg — continuous % inhibition |
| `TDC_herg_karim` | **13,445** | `unknown` | herg — balanced 6,727 / 6,718 |
| `TDC_b3db_classification` | **6,167** † | not declared | bbb |
| `TDC_b3db_regression` | 942 † | not declared | bbb (logBB) |
| `TDC_hlm` | **6,013** † | not declared | cl_micro |
| `TDC_rlm` | **5,590** † | not declared | cl_micro |
| `TDC_pampa_ncats` | 2,034 | `unknown` | caco2 proxy |
| `TDC_pampa_approved_drugs` | 142 † | not declared | caco2 proxy |
| `TDC_cyp1a2_veith` | **12,579** | `cc-by-4.0` | auxiliary |
| `TDC_cyp2c19_veith` | **12,665** | `cc-by-4.0` | auxiliary |
| `MoleculeNet_Tox21` | 7,831 × 12 = **77,946** labels | `unknown` | auxiliary |
| `MoleculeNet_ToxCast` | 8,576 × 617 = **1,533,660** labels | `unknown` | auxiliary |
| `MoleculeNet_PCBA` | 437,929 | `unknown` | auxiliary / pretraining |
| `MolPILE` | **238,050,487** | not declared | unlabelled pretraining corpus |

The org holds **105 datasets** (36 of them `TDC_*`), confirmed by
`/api/datasets?author=scikit-fingerprints`. Off-topic for ADMET but relevant to the *other* half of
this programme: it also mirrors 30 `MoleculeACE_chembl*` activity-cliff benchmarks, including
`MoleculeACE_chembl2835_ki` — **JAK1**, a Tier-1 target in `TARGETS.md`.

Our own endpoints' mirrors reproduce our counts, which confirms the mirror is faithful:
`TDC_caco2_wang` 910, `TDC_ames` 7,278, `TDC_ld50_zhu` 7,385, `TDC_vdss_lombardo` 1,130,
`TDC_hia_hou` 578, `TDC_pgp_broccatelli` 1,218, `TDC_bioavailability_ma` 640,
`TDC_solubility_aqsoldb` 9,982, `TDC_cyp2c9_veith` 12,092, `TDC_cyp2d6_veith` 13,130,
`TDC_cyp3a4_veith` 12,328.

**`tdcommons` does not exist as a Hugging Face org** — the search returns zero datasets. TDC ships
through its pip package and Harvard DataVerse; `scikit-fingerprints` is the de-facto HF mirror.

### 3.2 `openadmet/*` — the best new data in this report

| dataset | configs / rows | licence |
|---|---|---|
| `openadmet/cyp-challenge-train-test` | `default` train **4,905** / test 750 (blinded); `tdi` 6,145; `emax` 6,145 (card says 6,146); `single_concentration` **17,504** measurements over 4,376 compounds; `structure` 20 | **apache-2.0** |
| `openadmet/pxr-challenge-train-test` | 29,763 across 6 configs (`default` 4,652; counter_assay 2,859; htchem 456+96; phase 1/2 unblinded 253/260) | `cc-by-4.0` |
| `openadmet/openadmet-expansionrx-challenge-data` | 7,608 + 7,618 raw | `cc-by-4.0` |
| `openadmet/openadmet-expansionrx-challenge-train-data` | 5,326 + 5,331 raw | `cc-by-4.0` |
| `openadmet/Octant_CYP_inhibition_reactivity_blog_release` | inhibition 1,340 / inhibition_detailed 16,931 / reactivity 2,446 / reactivity_detailed 19,344 / will_it_fly 11,353 | `cc-by-4.0` |
| `openadmet/ASAP_Polaris_OpenADMET_challenge` | files not parsed by the server (0 reported) | `mit` |

The CYP challenge train file gives **pIC50, with 95% CI and std**, for four CYPs — a regression
target where TDC gives us a binary one. Non-null per enzyme: CYP1A2 1,412, CYP2C9 1,285,
CYP2D6 1,493, CYP3A4 2,335 = **6,525 pIC50 labels on 4,905 compounds**.

Measured against our table: **4,618 of the 4,905 are new molecules**; 287 are already in ours and
**167 of those collide with one of our test folds** (drop or hold out). The blinded 750-compound
test set overlaps our table by 2 molecules, so it is genuinely blind to us.

> **The challenge is live.** The card states the challenge period is **2026-08-17 → 2026-11-03**.
> Today is 2026-10-04. The blinded test set exists and is scored externally — which makes it the
> only *prospective* evaluation available to this programme, and worth more than any number of
> retrospective leaderboard points.

### 3.3 `scikit-fingerprints` mirrors of the pharma releases

**ExpansionRx (via OpenADMET), all `cc-by-4.0`** — and, measured, **100% novel chemistry**:

| dataset | rows | our endpoint | already in ours | new |
|---|---|---|---|---|
| `ExpansionRx_OpenADMET_LogD` | 7,309 | lipophilicity | 0 | **7,309** |
| `ExpansionRx_OpenADMET_KSOL` | 7,298 | solubility | 0 | **7,298** |
| `ExpansionRx_OpenADMET_MLM_CLint` | 5,692 | cl_micro | 0 | **5,692** |
| `ExpansionRx_OpenADMET_HLM_CLint` | 4,541 | cl_micro | 0 | **4,541** |
| `ExpansionRx_OpenADMET_Caco-2_Permeability_Papp_AB` | 3,773 | caco2 | 0 | **3,773** |
| `ExpansionRx_OpenADMET_Caco-2_Permeability_Efflux` | 3,777 | pgp | — | ~3,777 |
| `ExpansionRx_OpenADMET_MPPB` | 1,756 | ppbr | 0 | **1,756** |
| `ExpansionRx_OpenADMET_MBPB` | 1,426 | ppbr | — | ~1,426 |
| `ExpansionRx_OpenADMET_RLM_CLint` | 633 | cl_micro | — | ~633 |
| `ExpansionRx_OpenADMET_MGMB` | 431 | ppbr | — | ~431 |

Not one of the 30,369 molecules I checked across six of these files appears in our 44,561. This is
a previously-proprietary med-chem series released whole; it is the single highest-novelty source in
the report.

**ASAP Discovery (via OpenADMET), all `cc0-1.0`** — small but public domain: `ASAP_OpenADMET_LogD`
478, `ASAP_OpenADMET_KSOL` 477, `ASAP_OpenADMET_MDR1-MDCKII` 551, `ASAP_OpenADMET_HLM` 407,
`ASAP_OpenADMET_MLM` 425.

### 3.4 Others, confirmed to exist

| dataset | rows | licence | verdict |
|---|---|---|---|
| `liuganghuggingface/admet` | 52,665 | `mit` | one flat table; provenance per row not documented — treat as a convenience mirror, not a source |
| `scbirlab/fang-2023-biogen-adme` | 7,042 (2 configs × 3,521) | `mit` | same data as the Biogen GitHub (§4.1), with a scaffold split attached |
| `dbn4/anpdb-admet-ai` | 11,448 | **none declared** | ADMET-AI *predictions* on a natural-product DB — model output, not measurements. Do not train on it. |
| `introvoyz041/admetica` | 96 | `mit` | too small |
| `agenticx/DILIrank` | 1,036 | **none declared** | matches DILIrank 1.0's size (§4.3) |
| `jablonkagroup/*` (ChemNLP, **315 datasets**) | `ld50_catmos` **9,032** mols; `ames_mutagenicity` 7,278; `herg_central_inhib` 306,893; `herg_karim` 13,445 | **none declared** | instruction-tuning text (`"The molecule with the SMILES … displays an acute oral toxicity rat LD50 of 2751.0 mg/kg."`). SMILES and value are regex-extractable. `ld50_catmos` at 9,032 is the only place CATMoS is on HF. |
| `ml-jku/tox21` | — | — | the original Tox21 challenge split |

**ADMET-AI's own training data is not published as a dataset.** `swansonk14/admet_ai` (MIT) is
Chemprop trained on TDC; `data/admet_merged.csv` is a 404. The trained model is a distillation
teacher, not a data source.

---

## 4. Public pharma ADME releases

### 4.1 Biogen — exact, and the brief's "~3000 each" is only true for four of six

Fang et al. 2023, *JCIM*, PMID **37216672**, DOI `10.1021/acs.jcim.3c00160`.
`github.com/molecularinformatics/Computational-ADME`, **MIT licence** (verified in the repo's
`LICENSE`). One file, `ADME_public_set_3521.csv`, **3,521 molecules**:

| column | non-null | our endpoint | new to us (standardised) |
|---|---|---|---|
| `LOG HLM_CLint (mL/min/kg)` | **3,087** | cl_micro | 3,003 (83 in ours, 35 hit a test fold) |
| `LOG RLM_CLint (mL/min/kg)` | **3,054** | cl_micro (rat) | ~3,000 |
| `LOG MDR1-MDCK ER (B-A/A-B)` | **2,642** | pgp | 2,570 (71 in ours, 34 hit a test fold) |
| `LOG SOLUBILITY PH 6.8 (ug/mL)` | **2,173** | solubility | 2,114 (58 in ours, 24 hit a test fold) |
| `LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)` | **194** | ppbr | 177 |
| `LOG PLASMA PROTEIN BINDING (RAT) (% unbound)` | **168** | — | ~160 |
| | **11,318 labels** | | |

hPPB and rPPB are 194 and 168, not ~3,000. Everything is already log-transformed; `% unbound` is the
*inverse* of our `ppbr` (% bound) — `ppbr = 100 − 10^value`, and getting that backwards silently
inverts the endpoint.

```bash
curl -sL https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv -o biogen_adme.csv
```

### 4.2 AstraZeneca, NCATS, ExpansionRx, ASAP, Octant

- **AstraZeneca** — `CHEMBL3301361`, 15 assays, 11,687 activities, fully enumerated in §2.2. CC BY-SA 3.0.
- **NCATS** — PubChem AIDs in §1.5, plus TDC's larger `hlm` (6,013) / `rlm` (5,590) /
  `pampa_ncats` (2,034). PMID 34176369. Public domain.
- **ExpansionRx** — §3.3, 10 endpoints, CC BY 4.0, zero overlap with us.
- **ASAP Discovery** — §3.3, 5 endpoints, CC0.
- **Octant Bio** — `openadmet/Octant_CYP_inhibition_reactivity_blog_release`, CC BY 4.0;
  `inhibition_detailed` 16,931 rows, plus a `reactivity` assay and a `will_it_fly` set (11,353).
- **Roche** — the only Roche deposit in ChEMBL's 465 `DATASET` documents is `CHEMBL3431459`
  (carnitine palmitoyltransferase modulators). **No public Roche ADME release found.**
- **Novartis** — none in the `DATASET` list; the Genomics Institute deposits are siRNA screens.
  **No public Novartis ADME release found.** Do not spend time looking again.

### 4.3 The endpoints with no pharma release — ld50, ames, dili, half_life, vdss

These need named academic/regulatory datasets, all verified in Europe PMC:

| endpoint | dataset | size | reference | availability |
|---|---|---|---|---|
| ld50 | **CATMoS** (NICEATM/ICCVAM acute oral toxicity) | **11,992 chemicals** (abstract) | PMID **33929906**, DOI `10.1289/ehp8495` | `jablonkagroup/ld50_catmos` has **9,032** molecules; primary data via NIEHS ICE / the OPERA repo |
| ames | **Ames/QSAR International Challenge** (Honma, NIHS Japan) | **12,140** (confirmed on the NIHS page) | PMID **30357358**, DOI `10.1093/mutage/gey031` | **confidential** — participants signed an NDA. Only the Class A strong-positive PDFs are public: 615 SMILES, 564 new, no negatives (§10.2). HF `ames_mutagenicity` (7,278) is the same Hansen/Xu set TDC uses |
| dili | **DILIst** | **1,279 drugs** (abstract) | PMID **31669330** | supplementary table |
| dili | **DILIrank 2.0** | **1,336 compounds, 981 non-ambiguous** (v1 = 1,037; 300 new, 49 revised) | PMID **41005561**, DOI `10.1016/j.drudis.2025.104485`, 2025 | `fda.gov/media/113052/download?attachment` — xlsx, **no SMILES** (§10.3) |
| half_life, vdss, cl | **Lombardo & Jing IV PK database** | **1,352 drugs** (abstract) | PMID **30115648**, DOI `10.1124/dmd.118.082966` | supplementary; TDC took **only VDss** (1,130) from it |

**The Lombardo observation is the free lunch here.** TDC's `vdss_lombardo` is one column of a
1,352-drug table that also carries human CL, terminal t½, MRT and fu. Our `half_life` endpoint has
665 labels; the same paper we already depend on for `vdss` holds roughly double that for t½, on
molecules we have already featurised.

> **Trap — CATMoS is not a drug set.** Its chemical space is industrial and agrochemical, and it
> contains inorganics and organometallics. The first `ld50_catmos` row I pulled was
> `B12B3B4B1C432` — a boron carbide cluster. RDKit will parse some of these and produce a feature
> vector that means nothing, and they are far outside the applicability domain of a model that is
> going to be asked about drug candidates. Filter on organic-element whitelist + heavy-atom count
> before merging.

---

## 5. Auxiliary tasks — what actually regularises a shared encoder

Ranked by labels-per-unit-of-risk, with an honest note on each.

### Tier A — free, and expected to help

1. **CYP1A2 + CYP2C19 from AID 1851 — 26,701 labels, zero new molecules.** Same assay, same plate,
   same compounds as three heads we already train. The isozymes are correlated through shared
   pharmacophore and through compound promiscuity, which is the signal a shared encoder is for. No
   new leakage surface at all, because the molecule set is identical to one we already split. If
   anything in this report is worth doing first, it is this. TDC ships them as `cyp1a2_veith`
   (12,579) and `cyp2c19_veith` (12,665), both `cc-by-4.0`.
2. **Tox21's 12 assays — 77,946 labels on 7,831 molecules** (`MoleculeNet_Tox21`). The canonical
   auxiliary set in this literature; it is in nearly every multitask ADMET paper for a reason. It is
   nuclear-receptor and stress-response activity, i.e. it teaches the encoder about reactive and
   promiscuous chemistry — which is what `ames`, `dili` and `herg` are also about.
3. **The AZ document's leftovers — 4,363 labels** (§2.2): rat/dog/mouse/guinea-pig PPB, rat
   hepatocyte CLint, buffer solubility, five pKa columns. Cross-species PPB is a near-duplicate task
   for human PPB and a classic transfer win; **pKa is the most mechanistically informative auxiliary
   in the list** — it drives logD, permeability and solubility, and we currently make the model
   infer it.

### Tier B — large, probably helps, needs care

4. **ToxCast's 617 assays — 1,533,660 labels on 8,576 molecules** (`MoleculeNet_ToxCast`). 19× our
   entire current label count. But the label matrix is sparse and extremely imbalanced per assay,
   many assays are cytotoxicity artefacts, and with 617 heads against 21 you will drown the tasks
   you care about unless the auxiliary loss is down-weighted hard and the gradient balancing (FAMO
   or similar, already in the plan) is actually watched. Treat as a **pretraining** stage with a
   discarded head, not a co-trained objective.
5. **PCBA — 437,929 molecules × 128 assays** (`MoleculeNet_PCBA`). Pretraining only; it is target
   bioactivity, not ADME, so it teaches scaffold chemistry rather than property.
6. **hERG Central's two continuous columns** (`at_1um`, `at_10um`, 306,893 each). These are percent
   inhibition from a single screen, so they are noisy and the distribution is a spike at zero — but
   they are 613,786 *continuous* labels for a 645-label endpoint, and a regression auxiliary on the
   same protein as a classification target is about the most favourable transfer setup available.

### Tier C — do not bother

7. **Computed RDKit descriptors as auxiliary targets.** We already feed RDKit2D as input features.
   Asking the model to predict a number it was handed is a no-op at best; if the descriptor head
   gets any loss weight it is pure capacity burned. (Descriptors as *inputs* are already in the
   baseline and are why it is strong.)
8. **PubChem computed properties (XLogP, TPSA, complexity, fingerprint bit counts).** Same argument
   — deterministic functions of the graph. The one exception worth testing is **experimental** pKa
   (Tier A3), which is a measurement, not a calculation.
9. **Tox21 viability/cytotoxicity counter-screens** (`*_viability`, ~30 assays in the 2026-09
   deposit). These measure "did the cells die", which is dominated by concentration and assay
   artefact; in the DeepTox-era literature they are the assays that add label noise rather than
   signal. Not re-measured on our own setup — but §10.7 gives the published method for deciding
   this before spending a training hour (PGM, PMID 38632308) instead of adding them and watching.
10. **KCNH2 3.1 (AIDs 720551/720553, 686k rows).** Wrong protein (§1.4).

---

## 6. Deduplication and leakage

### 6.1 The standardisation that must precede any merge

Measured on our own 46,061 molecules, RDKit 2026.03.6, 8 processes:

| step | pipeline | distinct keys | cost |
|---|---|---|---|
| raw SMILES | — | 46,061 | — |
| Normalizer → LargestFragmentChooser → Uncharger → InChIKey | the recipe below | **44,561** | 2.3 s total |
| + TautomerEnumerator().Canonicalize | adds the tautomer step | 44,487 (extrapolated: **8 extra merges per 5,000**) | **+4.7 min** |
| InChIKey skeleton block (first 14 chars) | tautomer- *and* stereo-insensitive | 41,324 | free |

The value ordering is unambiguous: **salt/fragment stripping and uncharging find 1,500 duplicates;
the tautomer step finds ~74 more.** Do the first two always; do the tautomer step too (5 minutes is
nothing), but do not believe anyone who tells you tautomers are the main problem — salts are.

```python
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')

_norm = rdMolStandardize.Normalizer()          # functional-group / charge normalisation
_frag = rdMolStandardize.LargestFragmentChooser()  # drop counter-ions, solvates
_unch = rdMolStandardize.Uncharger()           # neutralise where chemically sensible
_taut = rdMolStandardize.TautomerEnumerator()  # canonical tautomer

def key(smiles: str) -> tuple[str, str] | None:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None                            # unparseable: log it, never silently drop
    m = _norm.normalize(m)
    m = _frag.choose(m)
    m = _unch.uncharge(m)
    m = _taut.Canonicalize(m)
    ik = Chem.MolToInchiKey(m)
    if not ik:
        return None
    return ik, ik.split('-')[0]                # (full key, skeleton block)
```

Order matters. `Uncharger` before `LargestFragmentChooser` can neutralise a counter-ion into a
neutral fragment of comparable size and change which fragment "largest" picks. Normalise first,
choose the parent, then uncharge.

### 6.2 The merge rule

- **Identity for merging is the full InChIKey.** Two rows with the same full key are the same
  compound; combine their labels.
- **The skeleton block is a review flag, not an identity.** It collapses enantiomers, so merging on
  it would fuse (S)- and (R)-thalidomide. Use it to *find* the 1,099 skeleton groups in our table
  that straddle the test boundary, then decide each stereo pair on the chemistry. For an endpoint
  where stereochemistry is irrelevant (aqueous solubility, logD), merging is right; for anything
  enzyme-mediated (`cyp*`, `pgp`, `herg`), it is not.
- **Conflicting labels.** Where a merged molecule has two values for the same endpoint: for
  regression, keep the median and record the spread (a spread wider than the endpoint's own
  reproducibility is a signal the assays are not comparable — that is how you discover a unit bug);
  for classification, drop the molecule if the two sources disagree rather than voting. Disagreement
  between two curated sources is nearly always a protocol difference, and a majority vote launders
  it into a confident wrong label.
- **Source precedence.** Where the same molecule-endpoint pair exists in TDC and in a new source,
  **TDC wins** — not because it is better, but because our test folds are defined on it and
  overwriting a test molecule's value redefines the benchmark.

### 6.3 Not contaminating the official folds

Our rule — *a molecule in ANY endpoint's official test fold is never trained on* — needs one
amendment, because the measurement in §0 shows it is currently enforced on raw SMILES:

1. Standardise **everything**: our 46,061 and every incoming row, with the same function, in the
   same process.
2. Build `HELD_OUT = { full InChIKey : [endpoints for which it is test] }` from the standardised
   keys, not the SMILES. Today that set is **13,159 keys** (from 13,313 test rows).
3. **Fix the existing 441.** Every `split == 'train'` row whose standardised key is in `HELD_OUT`
   moves out of train. The 25 same-endpoint collisions are the urgent ones.
4. For each incoming row, if its key is in `HELD_OUT`, it may contribute **no training label for any
   endpoint** — because the shared encoder is the leak path. It may still be used to *extend the
   test fold's label coverage*, which is sometimes useful and always optional.
5. Re-derive `<endpoint>__fold` only from TDC's own test molecule list. Never from a new source.
   Growing a test fold with new data breaks leaderboard comparability — the mistake PROGRESS.md
   records as split-attempt #2.
6. Record, per source, the four numbers I report in the table below: distinct keys, already-present,
   test-collisions, new. If a source's test-collision count is high relative to its size, that is
   evidence it shares provenance with a TDC set and you are about to re-import the benchmark.

### 6.4 Measured test-collisions for every source I checked

This is the table to act on. `anyTest` = collides with some endpoint's test fold (must not be
trained on at all); `epTest` = collides with *this* endpoint's test fold (would be a direct
leaderboard leak).

| source | endpoint | distinct IK | in ours | anyTest | epTest | **new** |
|---|---|---|---|---|---|---|
| ChEMBL `Papp` Caco-2 | caco2 | 6,911 | 420 | 156 | 25 | **6,491** |
| ChEMBL `Papp` MDCK | caco2/pgp | 4,656 | 160 | 72 | 5 | **4,496** |
| ChEMBL human `T1/2` | half_life | 7,402 | 454 | 207 | 31 | **6,948** |
| ChEMBL human `CL` `mL.min-1.g-1` | cl_micro | 5,766 | 543 | 183 | **76** | **5,223** |
| ChEMBL human `CL` `mL.min-1.kg-1` | *new endpoint* | 3,125 | 535 | 330 | 17 | 2,590 |
| ChEMBL hERG IC50 (all relations) | herg | 15,937 | 1,520 | 635 | **53** | **14,417** |
| ChEMBL hERG IC50 (relation `=` only) | herg | 9,764 | 673 | 231 | 46 | 9,091 |
| **ChEMBL `PPB` (all organisms)** | ppbr | 8,728 | 2,008 | 747 | **324 — all 324** | 6,720 |
| ChEMBL `PPB` (human only) | ppbr | 2,025 | 197 | 106 | 13 | 1,828 |
| CATMoS LD50 | ld50 | 8,936 | 6,487 | 1,706 | **762 of 1,466** | 2,449 |
| B3DB regression (logBB) | bbb | 942 | 527 | 273 | **85 of 394** | 415 |
| PubChem AID 884 CYP3A4 (CIDs absent from 1851) | cyp3a4_inhib | 3,589 | 1,541 | 447 | 16 | 2,048 |
| Ames Class A (NIHS PDFs, **positives only**) | ames | 614 | 50 | 9 | 2 | 564 |
| ExpansionRx Caco-2 efflux | pgp | 3,777 | **0** | **0** | **0** | **3,777** |
| ExpansionRx MBPB | ppbr | 1,426 | **0** | **0** | **0** | **1,426** |
| ExpansionRx MGMB | ppbr | 431 | **0** | **0** | **0** | **431** |
| ExpansionRx RLM CLint | cl_micro | 633 | **0** | **0** | **0** | **633** |
| ASAP MDR1-MDCKII | pgp | 490 | 7 | 1 | 0 | 483 |
| ASAP KSOL | solubility | 418 | 4 | 1 | 0 | 414 |
| ASAP LogD | lipophilicity | 417 | 4 | 1 | 0 | 413 |
| ASAP HLM | cl_micro | 351 | 5 | 2 | 0 | 346 |
| ASAP MLM | cl_micro | 369 | 6 | 3 | 0 | 363 |
| Biogen HLM CLint | cl_micro | 3,086 | 83 | 35 | 3 | **3,003** |
| Biogen solubility pH 6.8 | solubility | 2,172 | 58 | 24 | 4 | **2,114** |
| Biogen MDR1-MDCK ER | pgp | 2,641 | 71 | 34 | 1 | **2,570** |
| Biogen hPPB | ppbr | 194 | 17 | 9 | 0 | 177 |
| B3DB classification | bbb | 7,802 | 3,356 | 1,237 | **359** | 4,446 |
| TDC hERG Central (inhib) | herg | 306,650 | 10,257 | 3,940 | 23 | **296,393** |
| TDC hERG Karim | herg | 12,966 | 1,422 | 485 | **95** | **11,544** |
| PubChem Tox21 hERG 2284432 | herg | 5,890 | 4,028 | 1,282 | 23 | 1,862 |
| PubChem Tox21 CYP2C9 2284391 | cyp2c9_inhib | 6,998 | 4,743 | 1,414 | **96** | 2,255 |
| PubChem Tox21 CYP2D6 2284394 | cyp2d6_inhib | 7,075 | 4,795 | 1,452 | **136** | 2,280 |
| PubChem Tox21 CYP3A4 2284096 | cyp3a4_inhib | 5,342 | 3,656 | 1,116 | 44 | 1,686 |
| PubChem NCATS KSOL 1645848 | solubility | 2,454 | 164 | 70 | 7 | 2,290 |
| PubChem NCATS PAMPA 1508612 | caco2 | 2,034 | 144 | 55 | 4 | 1,890 |
| PubChem NCATS RLM 1508591 | cl_micro | 2,529 | 182 | 74 | 2 | 2,347 |
| PubChem NCATS HLM 1963597 | cl_micro | 900 | 313 | 141 | 2 | 587 |
| TDC `hlm` | cl_micro | 6,013 | 98 | 42 | 4 | **5,915** |
| TDC `rlm` | cl_micro | 5,590 | 231 | 95 | 3 | **5,359** |
| OpenADMET CYP challenge train | cyp*_inhib | 4,905 | 287 | 167 | — | **4,618** |
| OpenADMET CYP challenge test (blinded) | — | 750 | 2 | 1 | — | 748 |
| ExpansionRx Caco-2 Papp A>B | caco2 | 3,773 | **0** | **0** | **0** | **3,773** |
| ExpansionRx KSOL | solubility | 7,298 | **0** | **0** | **0** | **7,298** |
| ExpansionRx LogD | lipophilicity | 7,309 | **0** | **0** | **0** | **7,309** |
| ExpansionRx HLM CLint | cl_micro | 4,541 | **0** | **0** | **0** | **4,541** |
| ExpansionRx MLM CLint | cl_micro | 5,692 | **0** | **0** | **0** | **5,692** |
| ExpansionRx MPPB | ppbr | 1,756 | **0** | **0** | **0** | **1,756** |

Read the two extremes. **B3DB collides with 359 of our 394 `bbb` test molecules** — because
`bbb_martins` *is* a B3DB subset, so B3DB is partly our own test set wearing a new name; take only
the 4,446 new rows and nothing else. **ExpansionRx collides with nothing at all** — 30,369 rows of
chemistry this programme has never seen.

And note `TDC hERG Karim` at 95 and `Tox21 CYP2D6` at 136 epTest collisions. Those are the rows that
would have bought us a free leaderboard jump and an unreproducible model.

### 6.5 The failure mode this prevents

The classic version is not subtle and it is worth naming: `CC(=O)Oc1ccccc1C(=O)O`,
`CC(=O)Oc1ccccc1C(=O)[O-].[Na+]`, `CC(=O)OC1=CC=CC=C1C(O)=O` and `CC(=O)Oc1ccccc1C(=O)O.O` are four
SMILES for one drug. A `set(smiles)` dedup keeps all four; they land in different folds; the model
memorises aspirin's solubility from train and is scored on it in test. Our own table has 1,500 such
pairs and 441 of them cross the boundary — which is the measurement, not the hypothetical.

The subtler version is **same-source re-import**: B3DB and `bbb_martins`, or AID 2284432 and
AID 1671200, or TDC's `caco2_wang` inside ChEMBL's Caco-2 rows. The defence is the same —
standardise, then diff against `HELD_OUT` *before* counting how much new data you think you have.

---

## 7. Realistic ceiling

"Plausible additional n" below is **new molecules measured by standardised InChIKey against our
table and against every test fold**, except where marked ≈ (not individually measured, estimated
from the file's row count minus a typical overlap). Labels, not molecules, where an endpoint can
take more than one source.

| endpoint | current | plausible additional | → total | principal sources |
|---|---|---|---|---|
| caco2 | 906 | **+12,154** | 13,060 | ChEMBL Caco-2 `Papp` 6,491; ChEMBL MDCK 4,496 (proxy); ExpansionRx Caco-2 3,773 → minus proxy discount; NCATS PAMPA 1,890 |
| bioavailability | 640 | +0 | 640 | nothing found at scale |
| lipophilicity | 4,200 | **+7,722** | 11,922 | ExpansionRx LogD 7,309; ASAP LogD 413 |
| solubility | 9,982 | **+12,116** | 22,098 | ExpansionRx KSOL 7,298; NCATS KSOL 2,290; Biogen 2,114; ASAP KSOL 414 |
| hia | 578 | +0 | 578 | nothing found |
| pgp | 1,212 | **+6,830** | 8,042 | Biogen MDR1-MDCK ER 2,570; ExpansionRx Caco-2 efflux 3,777; ASAP MDR1-MDCKII 483 |
| bbb | 1,965 | **+5,128** | 7,093 | B3DB classification 4,446; B3DB regression (logBB) 415; NCATS PAMPA-BBB 267 |
| ppbr | 1,614 | **+5,618** | 7,232 | ExpansionRx MPPB 1,756 + MBPB 1,426 + MGMB 431; ChEMBL human `PPB` 1,828; Biogen hPPB 177; AZ rat/dog/mouse PPB (auxiliary) |
| vdss | 1,111 | +0 real, ≈+11,000 rodent | 1,111 | ChEMBL `Vdss` is 11,333 molecules but 7,175 null-organism / 5,554 rat — **auxiliary only** |
| cyp2c9_inhib | 12,092 | **+3,540** | 15,632 | Tox21 2284391 2,255; OpenADMET CYP challenge 1,285 pIC50 |
| cyp2d6_inhib | 13,130 | **+3,773** | 16,903 | Tox21 2284394 2,280; OpenADMET 1,493 pIC50 |
| cyp3a4_inhib | 12,328 | **+6,069** | 18,397 | Tox21 2284096 1,686; OpenADMET 2,335 pIC50; **AID 884 2,048 new** (plus 6,846 AC50 values for regression) |
| cyp2c9_sub | 666 | +0 | 666 | nothing found |
| cyp2d6_sub | 663 | +0 | 663 | nothing found |
| cyp3a4_sub | 667 | +0 | 667 | nothing found |
| half_life | 665 | **+6,948** | 7,613 | **ChEMBL human `T1/2` 6,948 new** (1,808 of them censored — keep the bound); Lombardo & Jing t½ column (PMID 30115648) is a subset |
| cl_micro | 1,102 | **+28,520** | 29,622 | **ChEMBL human intrinsic `CL` 5,223** (watch the 76 test collisions); TDC `hlm` 5,915 + `rlm` 5,359; ExpansionRx HLM 4,541 + MLM 5,692; Biogen HLM 3,003; NCATS 2,934 |
| *(new)* plasma CL | — | +2,590 | 2,590 | ChEMBL human `CL` in `mL.min-1.kg-1` — an in-vivo endpoint we do not currently have a head for |
| ld50 | 7,342 | **+2,449** | 9,791 | CATMoS 9,030 extracted → 8,936 distinct, of which 6,487 already ours and **762 of our 1,466 test molecules** |
| herg | 645 | **+312,672** | 313,317 | **ChEMBL IC50 14,417 new, curated dose-response** (supersedes `herg_karim`'s 11,544, which is ChEMBL-derived); Tox21 2284432 1,862; hERG Central 296,393 *screen-grade* (+613,786 continuous labels) |
| ames | 7,255 | **+564 positives only** | 7,819 | Honma's 12,140-chemical set is **confidential**; only the Class A strong-positive PDFs are public (615 SMILES recovered, 564 new, **no negatives**) |
| dili | 475 | **≈+500** | ≈1,000 | **DILIrank 2.0 = 1,336 compounds, 981 non-ambiguous** — but names only, no SMILES, so the yield depends on name→structure resolution. The only remaining estimate in this table. |
| **total labels** | **79,238** | **≈+417,000** | **≈496,000** | |

Honest reading of that table:

- **Four endpoints go from "a number on a few hundred molecules" to real datasets**: `caco2` ×14,
  `half_life` ×11, `cl_micro` ×26, `herg` ×486. Those are four of the five endpoints where the
  baseline is weakest, and in every case the data was simply sitting in a source we already use.
- **Five endpoints cannot be improved with public data**: `bioavailability`, `hia` and the three
  `cyp*_sub` sets. They are small because the underlying measurement is rare (clinical F%, human
  intestinal absorption, a definitive substrate call). For these the only honest lever is transfer
  from the endpoints that *are* large, which is the argument for the multi-task architecture in the
  first place.
- **`ames` is a sixth, nearly**: Honma's 12,140-chemical Ames/QSAR set is behind a confidentiality
  agreement (§10.2), and the 564 new molecules recoverable from the public Class A PDFs are **all
  strong positives**. Adding positives alone to a 7,255-label set with no matching negatives shifts
  the prior without adding discriminative signal — useful as an enrichment check, not as training
  data.
- The ≈496,000 total is dominated by hERG Central. **Excluding it, the realistic figure is
  ≈200,000 labels — 2.5× what we have.** That is the number to plan against, because 296k
  single-concentration percent-inhibition readings from one 2009 thallium-flux screen are not worth
  296k curated dose-response measurements, and `herg`'s real increment is the **14,417 ChEMBL IC50
  molecules**, not the 296k.
- **Add the auxiliary tasks and the label count is a different order of magnitude**: +26,701 (CYP1A2
  /2C19, free), +77,946 (Tox21), +1,533,660 (ToxCast), +613,786 (hERG Central continuous). Which is
  the real answer to "we are data-bound": we are not short of *labels*, we are short of labels **on
  the 21 endpoints we score**. The multi-task encoder is the bridge, and §5 Tier A is where to test
  whether that bridge carries weight.

---

## 8. Acquisition plan, ordered by labels-per-hour-of-work

| # | source | curated labels | hours | labels/hour | licence risk |
|---|---|---|---|---|---|
| 1 | TDC datasets we never loaded | ~54,400 (+920k screen-grade) | 1 | **~54,000** | low |
| 2 | ExpansionRx + ASAP via OpenADMET | ~32,700 | 1 | **~33,000** | CC BY 4.0 / CC0 |
| 3 | Biogen ADME | 11,318 | 0.5 | **~23,000** | MIT |
| 4 | PubChem Tox21 CYP/hERG + NCATS ADME | ~43,000 | 3 | **~14,000** | public domain |
| 5 | ChEMBL ADME + hERG | ~33,000 | 5 | **~6,600** | **CC BY-SA 3.0** |

### 1. TDC's own datasets we never loaded — ~54,400 curated labels (+920k screen-grade), ~1 hour

Highest ratio in the report by an order of magnitude. Same mirror, same file format
(`SMILES,Y` + a splits JSON), same loader we already have. No curation, no unit harmonisation, no
new provenance to audit.

```bash
B=https://huggingface.co/datasets/scikit-fingerprints
for d in TDC_cyp1a2_veith:tdc_cyp1a2_veith TDC_cyp2c19_veith:tdc_cyp2c19_veith \
         TDC_hlm:tdc_hlm TDC_rlm:tdc_rlm TDC_pampa_ncats:tdc_pampa_ncats \
         TDC_b3db_classification:tdc_b3db_classification TDC_herg_karim:tdc_herg_karim \
         TDC_herg_central_inhib:tdc_herg_central_inhib \
         TDC_herg_central_at_1um:tdc_herg_central_at_1um \
         TDC_herg_central_at_10um:tdc_herg_central_at_10um; do
  r=${d%%:*}; f=${d##*:}
  curl -sL "$B/$r/resolve/main/$f.csv"         -o "$f.csv"
  curl -sL "$B/$r/resolve/main/${f}_splits.json" -o "${f}_splits.json"
done
```

Yields: CYP1A2 12,579 + CYP2C19 12,665 (auxiliary, zero new molecules) · `hlm` 5,915 new +
`rlm` 5,359 new for `cl_micro` · `pampa_ncats` 1,890 new for `caco2` · B3DB **4,446 new only —
discard the 3,356 already present and the 359 that hit our `bbb` test fold** · `herg_karim` 11,544
new, **minus 95 epTest collisions** · hERG Central 296,393 new molecules and 613,786 continuous
labels, minus 23 epTest.

Licence: `cc-by-4.0` on the two `veith` cards; the rest declare nothing, so treat them as the
upstream TDC licence (MIT code, per-dataset source terms) and keep the provenance note.

### 2. ExpansionRx + ASAP via OpenADMET — ~32,700 labels of 100%-novel chemistry, ~1 hour

Zero overlap with our table, measured — 30,369 rows across the six ExpansionRx files I checked
and not one molecule in common with our 44,561. Clean CC-BY-4.0 / CC0. Three of our weakest
regression endpoints (`caco2`, `cl_micro`, `ppbr`) and one strong one (`lipophilicity`). The four
ExpansionRx files I did not individually diff (Caco-2 efflux 3,777, MBPB 1,426, RLM 633, MGMB 431)
would add ~6,300 more.

```bash
B=https://huggingface.co/datasets/scikit-fingerprints
curl -sL "$B/ExpansionRx_OpenADMET_Caco-2_Permeability_Papp_AB/resolve/main/caco_2_permeability_papp_a_b.csv" -o erx_caco2.csv
curl -sL "$B/ExpansionRx_OpenADMET_KSOL/resolve/main/ksol.csv"            -o erx_ksol.csv
curl -sL "$B/ExpansionRx_OpenADMET_LogD/resolve/main/logd.csv"            -o erx_logd.csv
curl -sL "$B/ExpansionRx_OpenADMET_HLM_CLint/resolve/main/hlm_clint.csv"  -o erx_hlm.csv
curl -sL "$B/ExpansionRx_OpenADMET_MLM_CLint/resolve/main/mlm_clint.csv"  -o erx_mlm.csv
curl -sL "$B/ExpansionRx_OpenADMET_MPPB/resolve/main/mppb.csv"            -o erx_mppb.csv
# and the raw challenge release, which carries the assay metadata the mirrors drop
python -c "import datasets;datasets.load_dataset('openadmet/openadmet-expansionrx-challenge-data','raw').save_to_disk('erx_raw')"
# ASAP Discovery, CC0: HLM 407, KSOL 477, LogD 478, MDR1-MDCKII 551, MLM 425
for d in HLM:hlm KSOL:ksol LogD:logd MDR1-MDCKII:mdr1_mdckii MLM:mlm; do
  curl -sL "$B/ASAP_OpenADMET_${d%%:*}/resolve/main/${d##*:}.csv" -o "asap_${d##*:}.csv"
done
```

Also take the **OpenADMET CYP inhibition challenge** (apache-2.0) while you are here — 4,905
compounds, **4,618 new to us**, with pIC50 instead of a binary label, and 167 train molecules that
must be dropped because they hit one of our test folds:

```bash
python - <<'PY'
import pandas as pd
B="hf://datasets/openadmet/cyp-challenge-train-test/"
pd.read_csv(B+"cyp-challenge-TRAIN_inhibition.csv").to_csv("cyp_challenge_train.csv", index=False)
pd.read_csv(B+"cyp-challenge-TEST-BLINDED.csv").to_csv("cyp_challenge_test_blinded.csv", index=False)
PY
```

The challenge closes **2026-11-03** and its test set is externally scored — the only prospective
evaluation available to this programme.

### 3. Biogen ADME — 11,318 labels on 3,521 molecules, ~30 minutes

The cleanest file in the report: one CSV, MIT, already log-transformed, six endpoints, and 3,003 +
2,114 + 2,570 new molecules for `cl_micro`, `solubility` and `pgp`.

```bash
curl -sL https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/main/ADME_public_set_3521.csv -o biogen_adme.csv
```

Remember `% unbound` inverts against our `ppbr` — `ppbr = 100 − 10^value` (§4.1) — and that
hPPB/rPPB are 194/168 rows, not 3,000.

### 4. PubChem — the Tox21 CYP + hERG panel and the NCATS ADME tier — ~43,000 labels, ~3 hours

Needs the outcome filter (§1.1) and the dedup pass; the CSVs come with `AC50`/`HITC`/`BMD` so you
get a regression target alongside the class label.

```bash
for aid in 2284391 2284394 2284096 2284393 2284395 2284432 \
           1645848 1508612 1508591 1963597 1845228 1645871; do
  curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/$aid/CSV" -o "aid$aid.csv"
  sleep 0.3   # PUG-REST: ≤5 req/s, and be polite
done
# AID 1851 and 884 exceed the 10,000-SID cap on /CSV — use /concise/CSV
curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/1851/concise/CSV' -o aid1851.csv
curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/884/concise/CSV'  -o aid884.csv
```

Drop every row whose `PUBCHEM_ACTIVITY_OUTCOME` is not `Active`/`Inactive` — that is 22.4% of
AID 1851, 29.2% of AID 2284096 and 19.6% of AID 1508612. Deduplicate AID 2284432 against
AID 1671200 before counting.

### 5. ChEMBL ADME + hERG — ~33,000 new molecules across four endpoints, ~5 hours

The largest absolute increment in the report and the most work per label, entirely because of the
units and the censoring. Yields `caco2` +6,491, `half_life` +6,948, `cl_micro` +5,223,
`herg` +14,417, plus 4,496 MDCK and 2,590 plasma-CL molecules and the AZ document's 4,363
unexploited activities.

```bash
C=https://www.ebi.ac.uk/chembl/api/data/activity.json
F=molecule_chembl_id,canonical_smiles,standard_value,standard_units,standard_relation,standard_upper_value,standard_type,assay_description,assay_chembl_id,document_chembl_id,target_organism

page () {  # $1 = query string, $2 = output prefix, $3 = total rows
  for off in $(seq 0 1000 "$3"); do
    curl -s "$C?$1&limit=1000&offset=$off&only=$F" -o "$2_$off.json"
  done
}
page 'assay_type=A&standard_type=Papp'                                   papp   20000   # 20,058
page 'assay_type=A&standard_type=T1%2F2&target_organism=Homo%20sapiens'  thalf   9000   #  9,279
page 'assay_type=A&standard_type=CL&target_organism=Homo%20sapiens'      clear  11000   # 11,725
page 'target_chembl_id=CHEMBL240&standard_type=IC50'                     herg   19000   # 19,130

# the AstraZeneca document's 4,363 activities TDC left behind (§2.2)
for a in 3301364 3301366 3301367 3301368 3301369 3301371 3301362 3301373 3301374 3301375 3301376; do
  curl -s "$C?assay_chembl_id=CHEMBL$a&limit=1000&only=$F" -o "az_$a.json"
done
```

Then, in this order:

1. **Never filter on `pchembl_value`** (§2.1) — it deletes 4,773 hERG `>` rows and 5,326 hERG
   molecules outright. Keep `standard_relation` and model the censoring.
2. Split `standard_type=CL` on `standard_units`: `mL.min-1.g-1` → `cl_micro`, `mL.min-1.kg-1` → a
   separate plasma-clearance head, everything else dropped.
3. Classify `Papp` rows by `assay_description` (`caco` / `mdck` / `pampa`), harmonise the 37 unit
   strings, drop the 1,540 unitless rows.
4. Diff against `HELD_OUT` *before* counting. This source alone collides with 53 of 128 `herg` test
   molecules, 76 of 221 `cl_micro`, 31 of 134 `half_life` and 25 of 181 `caco2`.

### And before any of the five: the two-hour fix that costs nothing to acquire

Standardise the existing table and move the **441 leaking train rows** out of train (§0, §6.3). It
is the only item here that *improves* the honesty of every number we have already published, and the
25 same-endpoint collisions — 6 of them on a 128-molecule hERG test fold — are currently inflating
the baseline we are measuring every hypothesis against.

---

## 9. Licences, and the ones that would restrict commercial use

| source | licence | commercial |
|---|---|---|
| PubChem / NCBI BioAssay (all AIDs here) | US-government public domain | **yes** |
| B3DB (`theochem/B3DB`, `LICENSE` = CC0 1.0) | **CC0 1.0** | **yes** |
| Biogen (`molecularinformatics/Computational-ADME`, `LICENSE` = MIT) | **MIT** | **yes** |
| ASAP Discovery via OpenADMET (`cc0-1.0`) | **CC0 1.0** | **yes** |
| OpenADMET CYP challenge (`apache-2.0`) | **Apache-2.0** | **yes** |
| OpenADMET PXR / ExpansionRx / Octant (`cc-by-4.0`) | **CC BY 4.0** | yes, **with attribution** |
| TDC `cyp*_veith` mirrors (`cc-by-4.0`) | **CC BY 4.0** | yes, with attribution |
| TDC code (`mims-harvard/TDC`, `LICENSE` = MIT) | MIT | yes (code only — datasets keep source terms) |
| ADMET-AI (`swansonk14/admet_ai`, `LICENSE.txt` = MIT) | MIT | yes (model, not data) |
| **ChEMBL (incl. the AstraZeneca deposit)** | **CC BY-SA 3.0 Unported** | **flag — see below** |
| `scikit-fingerprints/TDC_*` cards marked `unknown` / undeclared | **unknown** | **flag** |
| `jablonkagroup/*` (315 datasets, no licence on the cards I checked) | **unknown** | **flag** |
| `dbn4/anpdb-admet-ai` | none declared — and it is model *output*, not data | **do not train on it** |

Three things to decide, not to assume:

1. **ChEMBL's CC BY-SA 3.0 share-alike is the one real commercial question in this report.** It
   covers the whole of acquisition item 5 — Caco-2 permeability, human half-life, human clearance,
   hERG IC50 and the AstraZeneca DMPK document — which is also the largest single increment in the
   report (~33,000 molecules). It reaches further than that: `herg_karim` and TDC's `herg` are
   ChEMBL-derived too, so item 1 is partly affected. Whether model *weights* trained on a CC BY-SA database are a
   derivative work of that database is unsettled and jurisdiction-dependent; EMBL-EBI's own position
   is that the licence attaches to the data and to databases derived from it. The cheap mitigation
   is to keep the ChEMBL-derived rows in a separately-flagged partition so a model can be retrained
   without them, and to get an actual answer before anything ChEMBL-trained ships in a paid
   product. Do not resolve this by reading the licence again; it needs a decision.
2. **"Licence unknown" is not "licence permissive."** Several `scikit-fingerprints` cards and all of
   `jablonkagroup` declare nothing. The *underlying* data is usually fine (TDC, PubChem, published
   supplementary tables), but the mirror asserts no grant. For anything commercial, re-derive from
   the primary source — which for most of them is one of the public-domain or MIT routes above.
3. **Do not train on predictions.** `dbn4/anpdb-admet-ai` is ADMET-AI output on a natural-product
   database. Training on it is distillation dressed as data acquisition, it inherits ADMET-AI's
   errors as ground truth, and it will silently contaminate any comparison against ADMET-AI.

---

## 10. Addendum (2026-10-05) — the loose ends, closed

Everything this report originally left as `≈` or `[unverified]`, resolved by measurement. One of
these is the worst leakage result in the whole document.

### 10.1 ChEMBL's `PPB` contains **100% of our `ppbr` test fold**

| query | distinct IK | in ours | collides with our 324-molecule `ppbr` test fold | new |
|---|---|---|---|---|
| `assay_type=A&standard_type=PPB`, all organisms | 8,728 | 2,008 | **324 — every single one** | 6,720 |
| the same, `target_organism=Homo sapiens` | 2,025 | 197 | **13** | 1,828 |

`ppbr_az` **is** ChEMBL assay `CHEMBL3301365` (§2.2), so pulling `standard_type=PPB` re-imports the
entire benchmark. Not a third of it, not most of it — all 324 molecules.

And the fix that looks obvious is the wrong one. Adding `target_organism=Homo sapiens` drops the
collisions from 324 to 13, but only because **the AstraZeneca human-plasma assay has no
`target_organism` recorded at all** — the filter throws away our own source along with the leak, and
takes the new-molecule yield from 6,720 down to 1,828. Species must be read off
`assay_description`, not `target_organism`; the organism field is null for most ADME assays because
an ADME assay has no molecular target to attach an organism to.

Use the `HELD_OUT` diff, keep the 6,720, and get species from the description text.

### 10.2 Honma's Ames set — confidential, and the public slice is positives only

Verified on the primary source (`https://www.nihs.go.jp/dgm/amesqsar.html`), which states:

> "Because most of the Ames data are confidential, the QSAR builders/vendors participating in the
> project must execute a confidentiality agreement."

The page confirms the **12,140 chemicals** figure across three phases, and publishes exactly five
files — `ClassA183.pdf`, `ClassA236.pdf`, `ClassA253.pdf`, `ICEM2017Honma.pdf`, `paper2019.pdf`.
The three `ClassA*` PDFs are the **strong positives only** (>1,000 revertants/mg in at least one
strain), and they *do* carry SMILES in a table column.

```bash
for f in ClassA183 ClassA236 ClassA253; do
  curl -sL "https://www.nihs.go.jp/dgm/$f.pdf" -o "$f.pdf" && pdftotext "$f.pdf" "$f.txt"
done
```

`pdftotext` + reassembling the line-wrapped cells + RDKit validation recovers **615 of the 672**
(91.5%); 564 are new to us, 2 hit our `ames` test fold. **There are no negatives**, so this is an
enrichment/sanity set, not training data — see the §7 bullet. Class B (positive, not strong) and
Class C (negative) were never published.

Verdict: `ames` cannot be grown from the modern literature. Treat 7,255 as its ceiling.

### 10.3 DILIrank 2.0 — downloadable, 1,336 compounds, and no structures

The FDA LTKB page links one spreadsheet, and it is *not* the `media/129717` URL that appears in
search results (that is a JPEG):

```bash
curl -sL 'https://www.fda.gov/media/113052/download?attachment' -o dilirank.xlsx
```

Two sheets. "version 2" has **1,336 compounds**, "version 1" has 1,037. Composition of v2:

| vDILI-Concern | n |
|---|---|
| vNo-DILI-concern | 413 |
| Ambiguous-DILI-concern | 354 |
| vLess-DILI-concern | 351 |
| vMost-DILI-concern | 217 |
| | **981 non-ambiguous** |

`Comment` column: 987 Unchanged, **300 New**, 49 Revised against v1.

Two traps:

- **No SMILES.** Columns are `LTKBID, CompoundName, SeverityClass, LabelSection, vDILI-Concern,
  Comment`. Every molecule needs name→structure resolution (PubChem `/compound/name/...`), which
  fails on salts, mixtures and biologics and silently resolves some names to the wrong parent. This
  is why `dili` is the one row in §7 that is still an estimate. The HF mirror `agenticx/DILIrank`
  (1,036 rows) has the same gap — it is v1 and also names-only.
- **The label strings are not case-consistent**: `vMost-DILI-concern` (215) vs `vMOST-DILI-concern`
  (2) vs `vNo-DILI-concern` (413) vs `vNo-DILI-Concern` (1). A `groupby` on the raw column yields
  six classes instead of four. Normalise case before anything.

`jablonkagroup/drug_induced_liver_injury` is **475 rows** — exactly TDC's set, no gain.

### 10.4 CATMoS LD50 — extractable, and half our test fold is in it

The only public machine-readable copy I found is the ChemNLP mirror, as instruction text:

```python
import pandas as pd, re
B = "https://huggingface.co/api/datasets/jablonkagroup/ld50_catmos/parquet/completion_0/%s/0.parquet"
df = pd.concat(pd.read_parquet(B % s) for s in ("train", "val", "test"))   # 9,032 rows
pat = re.compile(r"SMILES(?: representation of)?\s+(\S+)\s+"
                 r"(?:displays|shows|exhibits|has)\s+an acute oral toxicity.*?of\s+([\d.]+)\s*mg/kg")
rows = [m.groups() for t in df["text"] if (m := pat.search(t))]              # 9,030 of 9,032
```

Two rows have an empty SMILES field. Note the regex: a naive `[^0-9]*` before the value cannot cross
the literal "LD50" and matches nothing — the `.*?of` form is required.

Measured: 8,936 distinct standardised keys, 6,487 already in our table, **762 of our 1,466 `ld50`
test molecules (52%)**, and **2,449 new**. `ld50_zhu` and CATMoS draw on the same acute-toxicity
compilations, so half the benchmark is inside the "new" source. My ≈+2,500 estimate was right for
the wrong reason — the overlap is far larger than assumed and the novelty is far smaller.

### 10.5 PubChem AID 884 — 2,048 new CYP3A4 molecules, not 4,526

The 4,526 figure in §1.2 was CID-level and included inconclusives. Restricting to
`Active`/`Inactive` and resolving to standardised keys:

| | n |
|---|---|
| AID 884 CIDs with a usable outcome and absent from AID 1851 | 3,618 |
| distinct standardised InChIKeys | 3,589 |
| already in our table | 1,541 |
| collides with our `cyp3a4_inhib` test fold | 16 |
| **new molecules** | **2,048** |

Plus 6,846 of AID 884's rows carry an AC50 in µM, i.e. a regression target for an endpoint TDC gives
us only as a binary label. Getting the SMILES needs a CID→SMILES POST, because the concise CSV has
no structure column:

```bash
curl -s -d "cid=$(paste -sd, cids.txt)" \
  'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/property/SMILES/TXT'
```

(2,000 CIDs per POST is comfortable; the GET form hits the URL-length limit well before that.)

### 10.6 The four remaining ExpansionRx files and all five ASAP files

Completing §3.3. ExpansionRx's novelty holds at exactly zero overlap across all ten files:

| source | endpoint | distinct IK | in ours | new |
|---|---|---|---|---|
| ExpansionRx Caco-2 efflux | pgp | 3,777 | **0** | **3,777** |
| ExpansionRx MBPB | ppbr | 1,426 | **0** | **1,426** |
| ExpansionRx MGMB | ppbr | 431 | **0** | **431** |
| ExpansionRx RLM CLint | cl_micro | 633 | **0** | **633** |
| ASAP MDR1-MDCKII | pgp | 490 | 7 | 483 |
| ASAP KSOL | solubility | 418 | 4 | 414 |
| ASAP LogD | lipophilicity | 417 | 4 | 413 |
| ASAP HLM | cl_micro | 351 | 5 | 346 |
| ASAP MLM | cl_micro | 369 | 6 | 363 |

ExpansionRx total across all ten files: **36,636 molecules, not one of them in our 44,561.** ASAP
adds 2,019 more. That keeps acquisition item 2 where it is.

B3DB's regression half (logBB, 942 rows) adds only **415** new molecules and collides with **85 of
our 394 `bbb` test molecules (22%)** — same provenance problem as the classification half.

### 10.7 Which auxiliary tasks help — the §5 Tier-C claim, now with evidence

My Tier-C note was tagged `[unverified]`. Three papers address it directly, all confirmed in
Europe PMC:

| paper | PMID / DOI | what it establishes |
|---|---|---|
| **MTGL-ADMET**: ADMET property prediction via multi-task graph learning under adaptive auxiliary task selection, *iScience* 2023 | **38026198** / `10.1016/j.isci.2023.108285` | Adopts "**one primary, multiple auxiliaries**" and *selects* the auxiliary set per primary task (status theory + maximum flow). Beats both single-task and naive all-tasks MTL. So the right question is not "which auxiliaries help" but "which help *this* endpoint" — the answer is per-head, not global. |
| **PGM / transferability map**: Fast and effective molecular property prediction with transferability map, *Communications Chemistry* 2024 | **38632308** / `10.1038/s42004-024-01169-4` | Gives an **optimization-free** transferability measure — the distance between the principal gradients of the source and target datasets — and builds a quantitative map for source selection. Explicitly motivated by avoiding **negative transfer**. This is a cheap pre-training screen for whether ToxCast's 617 heads would help or hurt, computable before spending a single L4 hour. |
| **ACS**: Molecular property prediction in the ultra-low data regime, *Communications Chemistry* 2025 | **40629077** / `10.1038/s42004-025-01592-1` | "**Imbalanced training datasets often degrade [MTL] efficacy through negative transfer**"; adaptive checkpointing with specialization mitigates inter-task interference while keeping the MTL benefit, down to 29 labelled samples. |

The ACS finding is the one that matters most here, because **merging the sources in this report makes
our label matrix far more imbalanced, not less**: `herg` would go from 645 to ~310,000 while `dili`
stays near 1,000, a 300:1 ratio against today's 650:1 worst case but spread over endpoints we
actually score. The order of operations that follows from these three papers:

1. Run PGM (or any cheap gradient-distance proxy) over the candidate auxiliary sets **before**
   adding them. It is the published answer to the Tier-B/Tier-C question and it costs no training.
2. Add auxiliaries per primary head rather than globally — MTGL-ADMET's result says a fixed shared
   auxiliary set is leaving performance on the table.
3. Expect the imbalance to bite, and plan for ACS-style per-task checkpointing or the FAMO-style
   gradient balancing already in `PLAYBOOK.md`. Do not add hERG Central at full weight.

That also sharpens §5 Tier A: CYP1A2/CYP2C19 from AID 1851 are the ideal auxiliary under all three
papers' logic — same assay, same plate, same molecules, correlated mechanism, and **no imbalance
introduced at all**, because the molecule set is identical to one we already split.

### 10.8 The CYP challenge test set is still blind — the reveal was PXR's, not CYP's

`PROGRESS.md` (commit 39368cf) records an entry rule premised on "the 25 September intermediate
leaderboard was a one-time full test-set reveal". **The dataset repo does not support that for the
CYP challenge.** Checked directly:

| | CYP challenge | PXR challenge |
|---|---|---|
| blinded test file | `cyp-challenge-TEST-BLINDED.csv` — **2 columns, `Molecule_Name,SMILES`, 750 rows** | `pxr-challenge_TEST_BLINDED.csv` |
| unblinded test file | **none exists** | `pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`, `pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`, `pxr-challenge_DRC_plot_data_TEST_UNBLINDED.parquet` |
| last commit to `main` | **2026-09-23** ("Add structure blinded set") | **2026-09-28** (DRC raw data split) |
| README CHANGELOG's last entry | 2026-09-23 | 2026-09-28 |

`https://huggingface.co/api/datasets/openadmet/cyp-challenge-train-test/commits/main` lists seven
commits, none in the week of 25 September and none adding labels. **It is the PXR challenge that
runs staged reveals**, and it has the `*_UNBLINDED` files to prove it; CYP has no such file and no
commit that could have carried one.

What I can and cannot verify: the authoritative artefact — the dataset — still ships a blinded CYP
test set. The leaderboard Space (`openadmet/cyp-challenge`, docker, last modified 2026-09-29) is a
scoring app (`leaderboards.py`, `head_to_head.py`, `submission.py`) whose backing data I cannot
read, so I cannot rule out that a leaderboard view once exposed per-compound truth. But the premise
as recorded — that the full test set was released — is **not evidenced**, and it was almost
certainly read across from PXR.

This matters because the entry rule derived from it ("treat the revealed test set as single-use
evidence") is strictly more conservative than necessary. If the CYP test set is still blind, a
submission before **2026-11-03** is a genuine prospective evaluation — the one measurement this
programme's retrospective numbers cannot reach. Worth re-checking against the Space before the
rule is allowed to stand.

### 10.9 Revised totals

| | labels |
|---|---|
| current | 79,238 |
| additional, measured (§7) | **≈+417,000** |
| of which hERG Central, screen-grade single-concentration | 296,393 |
| **additional excluding hERG Central** | **≈+121,000** |
| realistic plan-against total | **≈200,000 (2.5×)** |

Nothing in §8's ordering changes. The one practical consequence of this addendum is that
**acquisition item 5 (ChEMBL) now needs the `HELD_OUT` diff more than any other item**: it collides
with 324 of 324 `ppbr`, 76 of 221 `cl_micro`, 53 of 128 `herg`, 31 of 134 `half_life` and 25 of 181
`caco2` test molecules. Five of our twenty-one benchmarks are substantially inside it.

---

## Appendix — things that cost me time, so they do not cost it twice

- `tdcommons` is **not** a Hugging Face org (0 datasets). TDC ships via pip and Harvard DataVerse.
- `swansonk14/admet_ai`'s `data/admet_merged.csv` is a **404**. ADMET-AI's training set is not
  published; only the trained Chemprop models are.
- `ncats/ADME` has no `ADME_public_sets.csv` on `main` and no `LICENSE`. Take NCATS data from
  PubChem (§1.5) or from TDC's `hlm`/`rlm`/`pampa_ncats`.
- PubChem Entrez full-text search for a *property* name ("PAMPA permeability" → 3,254 hits, "Ames
  mutagenicity" → 2,178) returns almost entirely 2–13-substance ChEMBL literature deposits.
  Always constrain with `N:100000000[TSC]`.
- The `datasets-server` `/size` endpoint reports `num_rows=1` for several `scikit-fingerprints`
  repos (plain CSV it did not parse). The CSV is fine; count it yourself. Affected here:
  `TDC_herg_central_*` (really 306,893), `TDC_b3db_classification` (6,167), `TDC_hlm` (6,013),
  `TDC_rlm` (5,590), `TDC_pampa_approved_drugs` (142), `TDC_dili`, `TDC_herg`, `TDC_ppbr_az`,
  `TDC_half_life_obach`, `TDC_bbb_martins`, `TDC_cyp*_substrate_carbonmangels`,
  `TDC_clearance_microsome_az`.
- `multiprocessing.Pool` on macOS Python 3.12 defaults to **spawn**, which cannot re-import a script
  fed on stdin — a `python - <<'EOF'` heredoc that uses a Pool hangs forever with no output. Write
  the script to a file. (Cost me two background jobs.)
- Searching Europe PMC for a dataset's *description* finds papers that *use* it, not the paper that
  *published* it. Search the exact title: `TITLE:"CATMoS: Collaborative Acute Toxicity Modeling
  Suite"` found PMID 33929906 immediately where the descriptive query did not.
- No public **Roche** or **Novartis** ADME release exists in ChEMBL's 465 `doc_type=DATASET`
  documents. The brief's expectation of one is not met; do not re-research it.

Added on the second pass (2026-10-05):

- **`fda.gov/media/129717/download` is a JPEG**, not the DILIrank spreadsheet, despite ranking well
  for it. The real file is `fda.gov/media/113052/download?attachment`, linked once from the LTKB
  page.
- **ChemNLP's instruction templates vary per row**, so one regex misses most of a dataset: my first
  pattern recovered 765 of 9,032 `ld50_catmos` rows. The verbs alternate
  (`displays|shows|exhibits|has`), the SMILES is sometimes preceded by "representation of", and a
  `[^0-9]*` guard cannot cross the literal "LD50". The working pattern is in §10.4.
- **PubChem's CID→SMILES lookup must be a POST.** The GET form
  (`/compound/cid/<list>/property/SMILES/TXT`) exceeds the URL limit at a few hundred CIDs; POST the
  `cid=` body instead, 2,000 per call. Note the property is `SMILES`, not `CanonicalSMILES`.
- **`openpyxl` is not in `talindrew-drugrx/.venv`.** Parsing a one-sheet xlsx with `zipfile` +
  `xml.etree` (sharedStrings → sheet rows) takes ten lines and avoids touching the project's
  dependencies for a research task.
- **A `tail`/`grep` pipe swallows a background job's output.** `cmd | tail -40` run with
  `run_in_background` buffers everything until EOF, so the task's output file stays at 0 bytes and
  the job looks hung. Redirect to a file and poll the file.
- **ChEMBL's `target_organism` is null for most ADME assays** — including AstraZeneca's human plasma
  protein binding. Filtering on it is not a species filter; it is a filter for "assays that happen
  to have a molecular target attached", which ADME assays mostly do not (§10.1).
