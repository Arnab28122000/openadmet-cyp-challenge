# Data provenance and the weights

## The decision

**Weights will be published to the Hugging Face Hub**, recorded in git and in memory (owner's
instruction, 2026-10-04). That is a deliberate change from the earlier assumption that weights would
stay behind the FoldRx API, and **it is the variable the licence question turns on**: ShareAlike
triggers on *distribution of an adaptation*, so a weights release puts the question live where an
API-only deployment never would have.

## What the audit changed about the ChEMBL question

The licence was never the gating risk. **Leakage was**, and measuring it shrank the problem:

| pull | own-test hits | residual new & safe | mechanism |
|---|---|---|---|
| lipophilicity `CHEMBL3301363` | 418 / 420 (99.5%) | **3** | **A** |
| ppbr `CHEMBL3301365` | 324 / 324 (100%) | **1** | **A** |
| cl_micro `CHEMBL3301370` | 221 / 221 (100%) | **0** | **A** |
| hERG `CHEMBL240` IC50 | 53 / 123 (43.1%) | **14,406** | **B** |
| half_life `T1/2` | 30 / 134 (22.4%) | **11,930** | **B** |
| caco2 `Papp` | 36 / 178 (20.2%) | **13,327** | **B** |

**Two mechanisms, two different exclusion rules.** Conflating them would be a costly mistake in
either direction.

- **Mechanism A — the ChEMBL document *is* the benchmark.** TDC built `lipophilicity_astrazeneca`,
  `ppbr_az` and `clearance_microsome_az` from these AstraZeneca deposits essentially whole. Importing
  them re-imports 100% of three test folds for a combined **4 new molecules**. Correct action:
  **drop the document entirely**. Do not plumb per-record provenance for 4 molecules.
- **Mechanism B — ChEMBL independently indexed the same upstream literature** (Obach, Lombardo, Wang,
  the AqSolDB constituents). The residual is real and large; removing 53 colliding molecules from a
  15,934-molecule hERG pull leaves **90% intact**. Correct action: **molecule-level exclusion**, not
  document-level. The mechanism-A result must not be allowed to argue for discarding these.

## Provenance model

**Per record, not per partition.** A partition flag on a chunk gives a retrain path; per-record
provenance answers *"which obligations attach to this trained artefact"*, which is the question
actually asked.

| column | meaning |
|---|---|
| `source` | where this measurement was taken from |
| `source_license` | the licence of that source |
| `preferred_provenance` | for a measurement reachable from several sources, the most permissive one |

Where the same measurement exists in BindingDB (CC BY 3.0) or PubChem (public domain), that
provenance is preferred and the ChEMBL copy drops out of the obligation set. **Attribution is owed
either way**, including to anything that survives dedup to ChEMBL.

## Open, and not for me to settle

**ChEMBL is CC BY-SA 3.0 Unported.** With weights now being published, the ShareAlike question is
live rather than moot. This is unsettled in the field rather than resolved:

- The common position is that weights are not a copyright derivative of factual measurements, and
  published ADMET weight releases state exactly that while still crediting ChEMBL under CC BY-SA.
- Against that: **CC 3.0 Unported handles sui generis database rights ambiguously** where CC 4.0 is
  explicit.

**I am not a lawyer and this is not legal advice.** What the programme does about it:

1. Mechanism-A documents are dropped, so they attach no obligation at all.
2. Mechanism-B ChEMBL rows are kept with per-record provenance, so a ChEMBL-free model can be
   retrained on demand and the obligation set for any artefact can be stated exactly.
3. **Before the first public weights release**, either train the released artefact on the
   ChEMBL-free partition, or take a deliberate decision to release under ShareAlike terms with
   attribution. That is an owner decision, and it should be made with the residual numbers in hand
   rather than in the abstract.

## Clean sources (no share-alike)

PubChem (public domain) · B3DB (CC0) · ASAP (CC0) · Biogen ADME (MIT) · OpenADMET CYP (Apache-2.0) ·
ExpansionRx / PXR / Octant (CC BY 4.0) · BindingDB in-house curation (CC BY 3.0).

**Flagged unusable:** Enamine's terms §15 appears to prohibit ML use without written authorisation,
which propagates to ZINC22's REAL-derived entries; BELKA has no clear open licence; `ChemFM` is
CC BY-NC despite an HF tag saying MIT; MolE is non-commercial; ChemBERTa's weight repos declare no
licence at all; TabPFN 3.5 defaults are non-commercial.
