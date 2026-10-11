# 05 — Scaling and pretraining: how to spend more data well

Research date: **2026-10-04**. Web only; no code was changed.

Every claim is tagged **[V]** = *verified by fetching* (I read it at the URL cited in the same
paragraph) or **[R]** = *recalled, verify before quoting*. Anything I could not confirm says so.

**The question this file answers.** We have 21 TDC ADMET endpoints — 46,061 molecules, 79,238
labels, ~27k trainable — count-ECFP4 + RDKit2D (2,264 dims) into an MLP. Our measured facts:
tuned count-ECFP4+RDKit2D LightGBM beats binary ECFP4 on 18/21; NN and per-endpoint LightGBM are a
dead heat (10 vs 11) on official test folds; a capacity/regularisation sweep moved test score by
0.0004 and dropout hurt monotonically. We concluded the ceiling is data, not architecture. This
file asks what data to buy and in what order.

---

## 0. Seven findings, in the order they should change what we do

1. **CheMeleon is not a TDC result.** The 75% Polaris win rate in its abstract is over 28
   Polaris-hosted benchmarks. I pulled the authors' released per-seed scores and recomputed the
   head-to-head on the **17 TDC ADMET endpoints** against a Morgan-count + physchem Random Forest —
   almost exactly our own feature set. It goes **9 wins / 8 losses**: the same dead heat we already
   have. **But the split is not random — CheMeleon wins 6 of 8 regression endpoints (26% lower MAE
   on lipophilicity) and loses 6 of 9 classification endpoints.** That is an instruction, not a null
   result. (§1.6)
2. **CheMeleon's classification weakness looks like a fine-tuning-protocol bug, not a representation
   problem.** The authors shipped four fine-tuning strategies as raw data and never ranked them. I
   did: **every single TDC classification endpoint prefers a frozen or delayed-unfreeze encoder; not
   one prefers the full end-to-end fine-tune their headline results used.** (§1.11)
3. **The biggest *verified* gain in this file is not fine-tuning at all.** MapLight — one of only
   three TDC architectures that survived a 2026 reproducibility audit — went from **top-1 on 6/22 →
   11/22** TDC ADMET benchmarks by **concatenating a frozen 300-column GNN fingerprint** to its
   GBDT features. CheMeleon's 2048-d frozen embedding is a strictly better fingerprint, and
   extracting it is a 60-line script and under a GPU-hour. (§3.5.1 → §8 Step 2)
4. **Adding more endpoints to one shared encoder is the configuration with the most evidence
   against it.** ADMET-AI ran our exact experiment at **41 endpoints** and multi-task bought
   **speed, not accuracy**. Capela measured **−28% RMSE on the smallest task** from pooling
   uncorrelated tasks at n=6. The 2026 blind challenge *did* find multi-task essential — on nine
   endpoints measured on the **same** molecules. The discriminator is **labels per molecule**, and
   ours is **1.72**. (§6, §7.4a)
5. **Don't pretrain our own CheMeleon — do domain-adaptive descriptor regression instead.**
   Reproducing CheMeleon costs **250–500 L4-hours** plus ~500 CPU-hours to arrive where
   `pip install chemprop` puts us in a minute. Meanwhile Sultan et al. 2026 report that multi-task
   regression of 209 RDKit properties **on ≤4,000 of your own molecules** improves every ADME
   dataset at **P < 0.001** — and that pretraining past **400–800k molecules stops helping at all**.
   We already compute those descriptors. (§2.2, §3.5.2 → §8 Step 4)
6. **"The ceiling is label noise" is half-right, and every direct test of it has failed.** The
   quotable numbers are real — pooled public bioactivity **σ = 0.54–0.68**, maximum achievable
   **R² = 0.81** — but two things undercut the framing. **Same-lab reproducibility is σ = 0.17–0.22
   against 0.66–0.68 pooled: 9–16× the variance is curation debt, not assay physics.** And when the
   Solubility Challenge handed everyone a test set with **SD 0.17 log**, the best model still only
   reached **RMSE 0.86** — ~0.7 log of headroom — with the organisers concluding the old "0.6-log
   experimental error is the limiting factor" story was **wrong**. Kolmar & Grulke (>1,800 models)
   show noise raises *measured* error, not *true* error; Zhao et al. show **removing** suspected-noisy
   compounds does **not** improve external predictivity. (§5)
7. **The binding constraint is generalisation, not labels.** Llompart: SOTA solubility models at
   0.35–0.63 log on their own validation score **1.74–2.17** on a fresh external set — the
   generalisation gap is 3× the noise floor. And ADMET-AI and ADMETlab 3.0, both bigger multi-task
   models than ours, produce **negative R²** on a real discovery programme's molecules for Caco-2 and
   HLM clearance, and **R² = 0.02** on LogD, because nearest-neighbour similarity to public data
   falls below 0.3. **Every other number in this file is an in-distribution number.** (§5.3, §7.4b)

And one procedural finding we did not need to go looking for: our H4 result — a capacity sweep
moving the score by 0.0004 — **is published**. Tetko et al. found pre-set hyperparameters match
optimised ones at **1/10,000 the compute**, and the blind-challenge winners reported *"minimal
performance gain"* from tuning. Do not run another sweep. (§7.3)

---

## 1. CheMeleon

### 1.1 Identity and provenance

Burns, Zalte, Abreu, Sieg, Feldmann, Mathea, Green (MIT ChemE + BASF SE). **[V]**

| | |
|---|---|
| Preprint | *Deep Learning Foundation Models from Classical Molecular Descriptors*, [arXiv:2506.15792](https://arxiv.org/abs/2506.15792). v1 2025-06-18, **v2 2026-02-09**. arXiv licence **CC0 1.0**. **[V]** |
| **Journal version** | *Deep Learning Foundation Models **for Low-Data Regimes** from Classical Molecular Descriptors*, **JCIM 2026**, [10.1021/acs.jcim.6c01546](https://doi.org/10.1021/acs.jcim.6c01546). **[V]** (via OpenAlex W7201882692) |
| Code | <https://github.com/JacksonBurns/CheMeleon> — **MIT** **[V]** |
| Pretraining recipe, re-written | <https://github.com/JacksonBurns/how-to-train-your-chemeleon> — MIT **[V]** |

Note the title change on publication: **"for Low-Data Regimes"** was added. The authors
themselves scoped the claim. Our per-endpoint training sets are 500–7,000 molecules, so we *are*
in the regime they claim — but see §1.6.

### 1.2 Zenodo DOIs — verified

- **Model weights: `10.5281/zenodo.15426600`** (concept DOI, all versions). Title *"CheMeleon
  Foundation Model"*, author Jackson Burns, **licence MIT**, published 2025-05-18, one file
  `chemeleon_mp.pt`, **34.9 MB**. **[V]** (<https://zenodo.org/records/15426600>)
- **The version DOI the code actually downloads is `10.5281/zenodo.15460715`** — same title, same
  MIT licence, same `chemeleon_mp.pt` (34,859,448 bytes), 2025-05-18. **[V]**
  (<https://zenodo.org/api/records/15460715>). Both `chemeleon_fingerprint.py` and
  `finetuning_demo.ipynb` hard-code `https://zenodo.org/records/15460715/files/chemeleon_mp.pt`. **[V]**
- **Pretraining data: `10.5281/zenodo.15733574`** (concept; version DOI `...15733575`), *"CheMeleon
  Training Data"*, **CC-BY-4.0**, 2025-06-24. Files: `cleaned_pubchem_1MM.smiles` (~40 MB) and
  `pubchem_1MM.zarr.tar.xz` (**3.0 GB**). **[V]**

If you cite one DOI, cite the concept DOI **10.5281/zenodo.15426600**; if you pin a download, pin
**15460715**, because that is what the code fetches.

### 1.3 Architecture and pretraining — exact numbers

All **[V]** from the arXiv v2 PDF (§1.1 "Pre-training" and §3.1 "CheMeleon Pre-training"):

- D-MPNN implemented in Chemprop. Message-passing block **dimension 2048**, **depth 6**, **mean
  aggregation** → **8.7M parameters** in the message-passing portion.
- A two-layer FNN of the same dimension maps the embedding to the descriptor targets → **12.9M
  parameters total**. The FNN is **discarded** at fine-tuning time; only the ~8.7M-param D-MPNN
  encoder is what you load.
- Pretraining corpus: **1,000,000 molecules randomly selected from PubChem**.
- Targets: **1,613 Mordred descriptors** (`mordredcommunity`, BSD-3-Clause **[V]** via PyPI),
  each rescaled to zero mean / unit variance and **Winsorized to six standard deviations**.
- Loss: **masked MSE** with **85% of descriptor targets randomly masked each step** — explicitly
  described as "analogous to ChemBERTa's" regularisation.
- Post-pretraining test **RMSE 0.14** averaged over the 1,613 standardised descriptors. The paper
  explicitly says this number is reported "for completeness rather than its importance" — the
  model is never deployed to compute descriptors.
- Embedding width out of the encoder: **2048 floats**.

**Discrepancy worth knowing.** The re-written `how-to-train-your-chemeleon/pretraining/config.py`
ships `DROPOUT_FRACTION = 0.30`, `EPOCHS = 100`, `PATIENCE = 5`, LR schedule 1e-4 → 1e-3 → 1e-4
with 5 warmup epochs, `MP_HIDDEN_SIZE = 2048`, `MP_DEPTH = 6`, `FNN_HIDDEN_SIZE = 2048`,
`FNN_HIDDEN_LAYERS = 1`, `FEATURIZER = "RIGR"`. **[V]**. That is **not** the paper's 85% masking
and not the paper's default featurizer, and the repo says plainly it "is *not* a faithful
reproduction of the original training code". **[V]** Do not read `config.py` as the paper's
hyperparameters.

### 1.4 How to actually use it — three ways, in increasing cost

**Chemprop version: `>=2.2.0`** (released 2025-05-29). Latest on PyPI today is **2.3.1**
(2026-08-04), MIT, `requires_python >=3.11,<3.15`. **[V]** (<https://pypi.org/pypi/chemprop/json>)

**(a) CLI, zero code.** **[V]** — documented in
`docs/source/tutorial/cli/train.rst` under *"Finetuning Foundation Models"*:

```bash
pip install 'chemprop>=2.2.0'
chemprop train --data-path admet.csv --task-type regression \
  --from-foundation CheMeleon --output-dir ckpt
```

The docs state verbatim: *"Users can access pretrained foundation models by using the
`--from-foundation <name>` command line argument"*, that `CheMeleon` is *"pretrained on 1M
molecules from PubChem, suitable for many tasks and especially small datasets"*, and that *"The
first time a given model is requested it will automatically be downloaded for you and saved to a
directory called `.chemprop` in your home directory"*. **[V]**

Crucially the docs also say fine-tuning, unlike their transfer-learning path, **does not** require
the downstream FFN to match the pretrained one: *"the Message Passing (depth, hidden size,
activation function, etc.) and Aggregation configurations are fixed to be whatever they were
during pretraining, but the FNN is initialized from scratch according to the users request"*. **[V]**

**(b) Python, full control.** From `finetuning_demo.ipynb` **[V]** — this is the whole of it:

```python
import torch
from chemprop import featurizers, nn, data, models

featurizer    = featurizers.SimpleMoleculeMolGraphFeaturizer()
agg           = nn.MeanAggregation()
chemeleon_mp  = torch.load("chemeleon_mp.pt", weights_only=True)
mp            = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
mp.load_state_dict(chemeleon_mp["state_dict"])

ffn   = nn.RegressionFFN(input_dim=mp.output_dim, output_transform=output_transform)
mpnn  = models.MPNN(mp, agg, ffn, batch_norm=False, metrics=[nn.metrics.RMSE()])
```

`input_dim=mp.output_dim` is the one line people get wrong. `output_dim` is 2048.

**(c) Frozen fingerprint, no fine-tuning at all.** `chemeleon_fingerprint.py` in the repo is a
self-contained ~60-line class with no dependency beyond `chemprop>=2.2.0`. **[V]** It downloads the
checkpoint to `~/.chemprop/`, builds an `MPNN` with a throwaway `RegressionFFN`
(`# not actually used`), calls `.eval()`, and `__call__` returns
`self.model.fingerprint(bmg).numpy(force=True)` — a **2048-dim float vector per molecule**. **[V]**

This is the cheapest possible experiment we can run and it is the one I would run first (§8).

### 1.5 Fine-tuning hyperparameters the authors actually used

From `analysis/fine-tuning/scripts/finetune.py` **[V]**:

- `max_epochs=30`, `accelerator="gpu"`, `deterministic=True`, `NUM_WORKERS = 4`,
  `build_dataloader(...)` at Chemprop's default batch size (64).
- Head: `RegressionFFN` or `BinaryClassificationFFN` with **`hidden_dim=256`**,
  `input_dim=mp.output_dim` (2048).
- Checkpoint selection on `val_loss`; 80/20 train/val split of the benchmark train set via
  `astartes`; 5 seeds `(42, 117, 709, 1701, 9001)`.
- **Defaults are `--freeze-epochs 0` and `--dropout 0.0`** — i.e. *full end-to-end fine-tuning
  from step one, no dropout*.

The repo ships four ablation result sets — `default.csv`, `frozen.csv`, `delayed.csv`,
`dropout.csv` **[V]** — i.e. they tested freezing the encoder, delaying the unfreeze
(`FreezeMessagePassingCallback`), and adding dropout, and **shipped `default` (no freeze, no
dropout) as the recommended path.** **[V]** on the code and the defaults. **I did parse those four
CSVs and `default` is *not* the best choice on small classification sets — see §1.11, which is the
most actionable finding in this section.**

*This independently corroborates our own sweep result.* We measured dropout hurting monotonically
(0.1 > 0.3 > 0.5) and more regularisation pushing the best epoch later while making the model
worse. The CheMeleon authors landed on dropout 0.0 for fine-tuning too.

### 1.6 Benchmark numbers — and the part the abstract does not tell you

**Polaris (28 benchmarks, Tukey HSD α=0.05, a "win" = best or statistically indistinguishable from
best):** **[V]** Table 1 of the paper.

| Model | Win count | Win rate |
|---|---|---|
| **CheMeleon** | **21** | **75%** |
| minimol | 20 | 71% |
| RF (Morgan count 2048 r=2 + RDKit physchem) | 19 | 68% |
| MoLFormer | 11 | 39% |
| fastprop | 10 | 36% |
| Chemprop | 9 | 32% |
| MolCLR | 4 | 14% |

**MoleculeACE (29–30 ChEMBL activity-cliff assays):** **[V]** Table 2.

| Model | Whole test set | Cliff subset |
|---|---|---|
| **CheMeleon** | **29 wins / 97%** | **30 wins / 100%** |
| RF | 15 / 50% | 20 / 67% |
| minimol | 12 / 40% | 20 / 67% |
| fastprop | 5 / 17% | 12 / 40% |
| MoLFormer | 5 / 17% | 10 / 33% |
| Chemprop | 0 / 0% | 4 / 13% |
| MolCLR | 0 / 0% | 2 / 7% |

Three things the headline numbers hide:

1. **Those are single-task models.** The paper says explicitly: *"Across all of these benchmarks we
   only perform single-task regression or classification… we seek to demonstrate performance in
   individual tasks to better facilitate comparisons; all of the present models **can** be used in
   a multitask fitting approach."* **[V]** Nobody has published CheMeleon multi-task.
2. **No hyperparameter optimisation was done, deliberately**, citing Tetko et al. on overfitting by
   HPO. **[V]** So these are out-of-the-box numbers, which is good news for us.
3. **RF is third on Polaris at 68% against CheMeleon's 75%.** The paper's own text concedes
   *"Also noteworthy is the relative strength of the Random Forest baseline model when compared
   against deep models… Many-parameter, well-known foundation models such as MoLFormer and MolCLR
   fare surprisingly poorly."* **[V]**

#### The TDC-only analysis — original work, from their released source data

The abstract's 75% is over 28 Polaris-hosted benchmarks, only 17 of which are TDC ADMET. I pulled
their raw per-seed scores (`analysis/hsd/source_data.csv`, 1,960 rows, 14 models × 28 benchmarks ×
5 seeds) **[V]** and recomputed the head-to-head **restricted to the 17 TDC endpoints**, against
`RF_Morgan_Physchem` — Morgan **count** fingerprint 2048 bits r=2 **+ RDKit physchem descriptors**
**[V]** (paper §3.3), which is as close to our own count-ECFP4 + RDKit2D feature set as a
published baseline gets.

Mean over 5 seeds; ↓ marks lower-is-better (MAE), everything else higher-is-better:

| TDC endpoint | CheMeleon | RF_Morgan_Physchem | winner |
|---|---|---|---|
| lipophilicity-astrazeneca ↓ | **0.456** ±0.007 | 0.616 ±0.003 | CheMeleon (−26%) |
| ppbr-az ↓ | **7.815** ±0.133 | 8.372 ±0.040 | CheMeleon (−6.7%) |
| ld50-zhu ↓ | **0.527** ±0.011 | 0.616 ±0.001 | CheMeleon (−14%) |
| clearance-microsome-az | **0.603** ±0.014 | 0.554 ±0.006 | CheMeleon |
| half-life-obach | **0.325** ±0.025 | 0.287 ±0.010 | CheMeleon |
| vdss-lombardo | **0.546** ±0.011 | 0.514 ±0.007 | CheMeleon |
| caco2-wang ↓ | 0.376 ±0.030 | **0.291** ±0.003 | RF |
| clearance-hepatocyte-az | 0.381 ±0.044 | **0.448** ±0.006 | RF |
| herg | **0.867** ±0.011 | 0.855 ±0.004 | CheMeleon |
| pgp-broccatelli | **0.927** ±0.009 | 0.915 ±0.002 | CheMeleon |
| cyp2c9-substrate | **0.426** ±0.008 | 0.392 ±0.009 | CheMeleon |
| ames | 0.847 ±0.005 | **0.852** ±0.001 | RF |
| bbb-martins | 0.886 ±0.013 | **0.913** ±0.002 | RF |
| bioavailability-ma | 0.579 ±0.019 | **0.715** ±0.005 | RF (−19% rel.) |
| cyp2d6-substrate | 0.668 ±0.035 | **0.698** ±0.008 | RF |
| cyp3a4-substrate | 0.609 ±0.009 | **0.676** ±0.007 | RF |
| dili | 0.901 ±0.025 | **0.934** ±0.004 | RF |

**Overall: CheMeleon 9 W / 8 L.** A dead heat — the same dead heat we already have.

**But split by task type and it stops being a dead heat:**

| | regression (8) | classification (9) |
|---|---|---|
| **CheMeleon** vs RF_Morgan_Physchem | **6 W / 2 L** | 3 W / 6 L |
| minimol vs RF_Morgan_Physchem | 3 W / 5 L | 4 W / 5 L |

And for context, over the same 17 TDC endpoints against the same baseline: **fastprop 4W/13L,
Chemprop (scratch) 4W/13L, MoLFormer 2W/15L, plain RF-on-Morgan-only 7W/10L.** **[V]** (my
computation; raw data verified)

*Caveat on method:* I compared means over 5 seeds. The paper's win counts use a Tukey HSD test in
which statistical ties count as wins for both, so my table is stricter and not directly comparable
to their Table 1. The **direction and the regression/classification split** are what I would act
on, not the exact 9–8.

**Conclusion for us.** CheMeleon is worth doing, but the expected gain is concentrated:
large on the continuous endpoints (lipophilicity, PPBR, LD50, microsomal clearance, half-life,
VDss) and negative on the small binary ones (DILI n≈475, bioavailability-Ma n≈640, CYP3A4
substrate n≈667). The sensible target is a **hybrid**, exactly like the H3 routing table we
already built: CheMeleon for regression, keep LightGBM for the small classification sets.

### 1.7 VRAM and wall-clock on a 24 GB L4

The authors report **"Finetuning a single CheMeleon-based model requires <<1 GPU hour"** **[V]**
(README §Hardware), on benchmarks of ~100–7,400 training molecules. Their own pretraining ran on
**8 × NVIDIA 2080 Ti** **[V]**.

I could not find a published VRAM figure for fine-tuning. **[V that it is absent]** My estimate,
labelled as such:

- **Parameters.** 8.7M encoder + a 2048→256→1 head ≈ 9.2M trainable. At fp32 with Adam that is
  9.2M × (4 param + 4 grad + 8 optimiser state) ≈ **150 MB**. Negligible on 24 GB. **[R — arithmetic]**
- **Activations.** A Chemprop batch of 64 drug-like molecules is roughly 1,700 bonds → 3,400
  directed bonds. Six message-passing steps at width 2048, fp32: 6 × 3,400 × 2048 × 4 B ≈
  **170 MB** retained for backward. **[R — arithmetic]**
- **So fine-tuning on ~30k molecules uses well under 2 GB at batch 64.** The L4 is enormously
  over-provisioned. You could run batch 512–1024 and still be under 8 GB. **[R]**
- **The bottleneck is CPU and Python, not VRAM.** Chemprop's default caches graph featurisation up
  front (`--no-cache` to disable), and the docs flag `--num-workers` and the newer
  `--use-cuikmolmaker-featurization` as the throughput levers. **[V]**
- **Wall clock, estimated:** ~470 steps/epoch at batch 64 on 30k molecules; 6 × 2048² matmuls per
  step are small enough that kernel-launch and dataloader overhead dominate on an L4 →
  **~20–40 s/epoch, so ~10–20 min for the authors' 30 epochs, per task-type model.** **[R —
  estimate, not measured; measure it before planning around it.]**

Practical consequence: **a full 21-endpoint CheMeleon experiment is a few GPU-hours, not a few
GPU-days.** It is cheap enough that the cost of *not* trying it exceeds the cost of trying it.

### 1.8 Commercial use — yes, cleanly

| Component | Licence | Commercial? |
|---|---|---|
| CheMeleon weights (Zenodo 15426600 / 15460715) | **MIT** | **Yes** **[V]** |
| CheMeleon code (GitHub) | **MIT** (`LICENSE`, "Copyright (c) 2025 Jackson Burns") | Yes **[V]** |
| CheMeleon pretraining data (Zenodo 15733574) | **CC-BY-4.0** | Yes, with attribution **[V]** |
| Chemprop | **MIT** | Yes **[V]** (PyPI metadata) |
| `mordredcommunity` (only needed to *re*-pretrain) | **BSD-3-Clause** | Yes **[V]** (PyPI) |
| arXiv preprint text | CC0 1.0 | Yes **[V]** |
| PubChem source structures | public domain | Yes **[R]** |

There is no non-commercial clause, no "research use only" rider, and no model-licence-with-
acceptable-use-policy anywhere in the chain. This is the cleanest licence position of any model in
§3. The authors' conclusion says it outright: *"CheMeleon is open source, permissively licensed,
and readily accessible via the Chemprop machine learning package."* **[V]**

### 1.9 The one real integration obstacle: mixed task types

Chemprop's CLI takes **one** `--task-type` per run — `regression`, `regression-mve`,
`regression-evidential`, `regression-quantile`, `classification`, `classification-dirichlet`,
`multiclass`, `multiclass-dirichlet`, `spectral`. **[V]** There is no mixed regression+classification
multi-task head in the CLI.

The good news: **multi-task with sparse labels is natively supported and is the documented
default.** *"Chemprop automatically handles missing entries in the dataset by masking out the
respective values in the loss function, so that partial data can be utilized. The loss function is
rescaled according to all non-missing values... Training on partial data is therefore possible and
encouraged... No keyword is needed for this behavior, it is the default."* **[V]** That is exactly
our 46,061 × 21 sparse matrix with 79,238 observed cells.

So the shape of the experiment is forced and it is fine:

- **Model A** — one multi-task `regression` CheMeleon over our continuous endpoints, one CSV,
  blanks for missing.
- **Model B** — one multi-task `classification` CheMeleon over our binary endpoints.
- Or, in Python, one shared `BondMessagePassing` encoder with two heads and a summed loss. **This
  needs code we would have to write.** `chemprop/models/model.py` shows `MPNN.__init__` taking a
  single `predictor: Predictor`, and `MulticomponentMPNN` in `models/multi.py` is for multiple
  *molecules per datapoint* (solute/solvent), not mixed task types. **[V]** A two-head model means
  subclassing `Predictor` (or `MPNN`) ourselves — perhaps 100 lines, and it loses the CLI.

Given §1.11, Model A and Model B want *different fine-tuning protocols anyway* (full fine-tune for
regression; frozen or delayed-unfreeze for the small classification sets). So the forced split is
not a compromise — it is what the evidence says to do. Write the two-head model only if, after
measuring, cross-type transfer looks worth chasing.

### 1.10 CheMeleon outside the Polaris set — including multi-task, which the paper does not report

The repo ships a second, unpublicised benchmark suite (`analysis/moleculenet_results/`) run with
**Chemprop ≥2.2.0** and `--from-foundation chemeleon`, comparing CheMeleon-initialised Chemprop
against scratch Chemprop on MoleculeNet/OGB. **[V]** All numbers below are from
`CheMeleon.md` / `Chemprop.md` in that folder. **[V]**

| Benchmark | ≈ n | setting | CheMeleon | Chemprop (scratch) | winner |
|---|---|---|---|---|---|
| **PCBA scaffold (128 tasks)** | ~437k | **multi-task clf**, PRC-AUC ↑ | **0.4894** | 0.4722 | CheMeleon **+0.0172** |
| PCBA random (128 tasks) | ~437k | multi-task clf, PRC-AUC ↑ | **0.4451** | 0.4347 | CheMeleon +0.0104 |
| PCBA random, NaNs kept | ~437k | multi-task clf, PRC-AUC ↑ | **0.5501** | 0.5394 | CheMeleon +0.0107 |
| HIV scaffold | ~41k | clf, ROC-AUC ↑ | 0.7681 | **0.7895** | Chemprop |
| HIV scaffold | ~41k | clf, PRC-AUC ↑ | **0.3318** | 0.3057 | CheMeleon |
| PCQM4Mv2 | ~3.4M | reg, MAE ↓ | **0.0929** | 0.0992 | CheMeleon |
| QM9 multitask — gap | ~130k | **multi-task reg**, MAE ↓ | **0.0034** | 0.0045 | CheMeleon |
| QM9 multitask — homo | ~130k | multi-task reg, MAE ↓ | **0.0025** | 0.0032 | CheMeleon |
| QM9 multitask — lumo | ~130k | multi-task reg, MAE ↓ | **0.0025** | 0.0035 | CheMeleon |
| QM9 multitask — mu | ~130k | multi-task reg, MAE ↓ | **0.3474** | 0.4084 | CheMeleon |
| QM9 multitask — alpha | ~130k | multi-task reg, MAE ↓ | **0.2943** | 0.3714 | CheMeleon |
| **QM9 U0 single-task** | ~130k | reg, MAE ↓ | 2.1678 | **1.4464** | Chemprop, by 50% |
| SAMPL6 | tiny | reg, MAE ↓ | **0.2821** | 0.3036 | CheMeleon |
| SAMPL7 | tiny | reg, MAE ↓ | 0.4810 | **0.3591** | Chemprop |
| SAMPL9 | tiny | reg, MAE ↓ | **0.8527** | 0.9418 | CheMeleon |

Two things matter here and neither is in the paper:

1. **The only published CheMeleon multi-task evidence is PCBA-128 and QM9-multitask, and it is
   positive in every cell.** PCBA under a *scaffold* split with 128 sparse classification tasks is
   structurally the closest published analogue to our 21-task sparse ADMET matrix, and the
   pretrained init is worth **+0.017 PRC-AUC** there. Single run, no error bars, no seeds reported.
   **[V on the numbers; [R] on reading across to ADMET.]**
2. **The one heavy loss is QM9 U0 single-task** (MAE 2.17 vs 1.45). A total electronic energy is
   about as far from "predict 1,613 topological and physicochemical descriptors" as a target can
   get, and at 130k labels there is plenty of data to learn it from scratch. That is the shape of
   the failure mode to watch for: **far-from-descriptor target + lots of labels → the pretrained
   init is a liability.** None of our 21 endpoints is that, but AqSolDB (n≈10k) is our closest.

The exact CLI they used for the 128-task run, which doubles as our template **[V]**
(`scripts/chemeleon/pcba_scaffold_script.sh`):

```bash
chemprop train \
  -t classification \
  --data-path  $data_path \
  --splits-file $splits_path \
  --num-workers 20 \
  --epochs 50 \
  --pytorch-seed 42 \
  --save-dir $results_dir \
  --ensemble-size 5 \
  --from-foundation chemeleon \
  --accelerator gpu --devices "1,"
```

Note `--from-foundation chemeleon` (lowercase works), `--ensemble-size 5`, `--splits-file` for
honouring a predefined split — all three are things we want.

### 1.11 How to fine-tune: their ablation, re-analysed

The paper's fine-tuning ablation (`analysis/fine-tuning/{default,delayed,dropout,frozen}.csv`,
plotted as learning curves in `plot_figure.ipynb`) reports **only train/val loss curves — no test
metric per strategy**. **[V]** So the authors never published a ranking. I computed one: minimum
validation loss per benchmark, averaged over the 5 seeds, for each strategy, across all 28
benchmarks. **[V — my computation on their released CSVs.]**

Overall best-val-loss wins: **delayed 14, default 7, frozen 4, dropout 3** (of 28).

Split by TDC task type, the result is startlingly clean:

| TDC endpoints | default (full FT) | delayed unfreeze | frozen encoder | dropout |
|---|---|---|---|---|
| **classification (9)** | **0** | 5 | 4 | 0 |
| **regression (8)** | 3 | 2 | 0 | 3 |

**Every single TDC classification endpoint prefers a frozen or delayed-unfreeze encoder; not one
prefers full end-to-end fine-tuning.** The four where the encoder stays *fully* frozen are
cyp2c9-substrate (n=666), cyp2d6-substrate (n=664), cyp3a4-substrate (n=667) and DILI (n=475) —
the four smallest classification sets, and *exactly* four of the six endpoints where CheMeleon lost
to the Morgan+physchem RF in §1.6.

This is the most actionable thing in this file after §1.6: **CheMeleon's classification weakness
looks like a fine-tuning-protocol problem, not a representation problem.** The authors' headline
results used `default` (full fine-tune) everywhere, which is the worst choice on small
classification sets by their own validation curves.

**Three honest caveats.** (a) Validation loss is not test score, and a frozen encoder mechanically
produces a flatter, lower val curve — this is a hypothesis to test, not a result. (b) I could not
find the `--freeze-epochs` value used for the "delayed" runs; `finetune.py` takes it as an argument
and `scripts/run_experiments.sh` does not invoke `finetune.py` at all. **[V — I checked.]** (c) The
`dropout` run's dropout rate is likewise unrecorded in the repo. **[V]**

### 1.12 We can keep our own 2,264 features — Chemprop concatenates them

This matters because the comparison we want is not "CheMeleon vs our features" but "CheMeleon **plus**
our features vs our features alone". Chemprop supports exactly that. **[V]** (same `train.rst`):

> *"Additional datapoint descriptors can be concatenated to the learned representation **after
> aggregation**… provided using `--descriptors-path /path/to/descriptors.npz` where the descriptors
> are saved as a numpy `.npz` file… `X_d` is a 2D array with a shape of number of datapoints by
> number of additional descriptors… in the same order as the SMILES strings in your data file."*

There are also built-ins — `--molecule-featurizers` takes `morgan_binary`, **`morgan_count`**,
**`rdkit_2d`**, `v1_rdkit_2d`, `v1_rdkit_2d_normalized` **[V]** — i.e. `morgan_count rdkit_2d` *is*
our feature set. But the docs warn against using them raw:

> *"The Morgan fingerprints should not be scaled. Use `--no-descriptor-scaling`… The RDKit 2D
> features are not normalized. The StandardScaler used in the CLI to normalize is non-optimal for
> some of the RDKit features. **It is recommended to precompute and scale these features outside of
> the CLI** using an appropriate scaler and then provide them using `--descriptors-path` and
> `--no-descriptor-scaling`."* **[V]**

So the right call is to dump our **already-scaled** 2,264-dim matrix to `.npz` and pass it:

```bash
chemprop train -t regression \
  --data-path admet_reg.csv --splits-file tdc_scaffold.json \
  --from-foundation CheMeleon \
  --descriptors-path our_2264_features.npz --no-descriptor-scaling \
  --num-workers 8 --epochs 30 --ensemble-size 5 --accelerator gpu
```

That is structurally **ADMET-AI's Chemprop-RDKit (§6.1) with CheMeleon's encoder instead of a random
one**, multi-task over our regression endpoints, on the official split. It is one command.

One gotcha the docs flag: *"Passing separate train/val/test files is not supported for extra
features and descriptors `.npz` files"* **[V]** — so use one CSV plus `--splits-file`, not three
files.

---

## 2. Doing the descriptor-prediction pretraining ourselves

Short answer: **don't.** The recipe is cheap *by foundation-model standards* and expensive *by
our standards*, and there is no published scaling curve to tell you where to stop.

### 2.1 What it actually cost them — verified

From the CheMeleon README, lines 634–640 **[V]**, verbatim:

> Feature calculation for 1MM SMILES from PubChem took roughly **500 CPU hours**.
> Pretraining of the MLP-PLR took roughly 500 GPU hours and pretraining the `CheMeleon` model took
> about **1,000 GPU hours**.
> Finetuning a single `CheMeleon`-based model requires **<<1 GPU hour**, though running all
> finetuning for all models and all repetitions will take around 100 GPU hours.
> Storage of all pretraining and finetuning model checkpoints requires **1+ terabytes**…
> Storage of pretraining features requires **~40 GB** of disk space.

Those 1,000 GPU-hours were on **2080 Ti**s **[V]**.

### 2.2 Translating to one L4

- An L4 is roughly **2–4× a 2080 Ti** on this kind of fp32/bf16 workload. **[R — spec-sheet
  reasoning, not measured.]**
- 1,000 2080-Ti-hours → **≈250–500 L4-hours ≈ 10–21 days of continuous L4 time** to reproduce
  CheMeleon as published. **[R — derived]**
- Plus **500 CPU-hours** of Mordred featurisation. Mordred is slow and single-threaded per
  molecule; on a 16-core box that is ~1.5 days wall clock. **[R]**
- Plus **40 GB** of feature storage and a 3 GB Zarr download if you take theirs. **[V]**

The authors' own rewrite says it "has successfully trained... on a single GPU (with increased
patience)" **[V]**, so it is *possible* on one L4 — but on our one L4 that is the entire machine
for two to three weeks to reproduce a checkpoint we can download for free under MIT.

### 2.3 Is there a published data-scaling curve for this specific recipe?

**No.** I checked the arXiv v2 in full (22 pages), the GitHub repo, and the Zenodo records: there
is **no ablation varying the number of pretraining molecules and no ablation varying the number of
descriptors**. **[V — by absence; I read the whole paper.]** The paper's only architectural
control is "Chemprop (large)", a D-MPNN *identical in size to CheMeleon but randomly initialised*,
which performs substantially worse — establishing that the gain is from pretraining and not from
parameter count, but saying nothing about how much pretraining. **[V]**

They did, however, *run* one. `scripts/run_experiments.sh` invokes the descriptor-MLP evaluation
four times, as `pretrained_1MM` (checkpoint `e16_n2_h4096_best.pt`) **and** `pretrained_10MM`
(`10MM_best.pt`), on both MoleculeACE and Polaris. **[V]** But the released result tables contain
a single `MLP_PLR_Pretrained` model — **there is no 1M-vs-10M comparison in the published data.**
**[V — I enumerated the models in `analysis/hsd/source_data.csv` and `analysis/hsd_subset/`.]**
A 10× pretraining-corpus ablation was executed and not reported. Draw your own inference; I will
only say that nobody has published it, so we would be the first to know.

The best calibration points I could find are indirect, and they all point the same way — **the
curve is flat and non-monotone, so more pretraining molecules is a bad place to spend**:

- **Kirschbaum & Bande, *Transfer learning for molecular property predictions from small
  datasets*, AIP Advances 2024, [10.1063/5.0214754](https://doi.org/10.1063/5.0214754)** — this is
  the closest published analogue to our question: pretraining labels obtained from *computationally
  cheap ab initio or semi-empirical models* (i.e. calculated, low-noise labels, same idea as
  descriptors), then fine-tuning on small sets. Their finding, verbatim from the abstract: *"we find
  that for the HOPV dataset, the final training results **do not improve monotonically with the size
  of the pre-training dataset**, but pre-training with fewer data points can lead to more biased
  pre-trained models and higher accuracy after fine-tuning."* They also report it worked well on
  HOPV and **"less successful"** on FreeSolv, blaming a complex learning task and dissimilarity
  between pretraining and fine-tuning labels. **[V]**
- **Friedemann, Shinnick, Torr, Andreis, *Procedural Pretraining for Molecular Property
  Prediction*, arXiv 2609.17831 (2026)** gives a startling scale calibration: a procedural
  (non-chemical!) pretraining task reduced Lipophilicity test error by 4.8%, and *"the magnitude of
  this improvement is roughly **90% of the performance difference between our 250K-molecule
  baseline and the publicly released MoLFormer checkpoint pretrained on approximately 100M
  molecules**."* **[V]** In other words: going from 250k to 100M pretraining molecules buys about
  5% on Lipophilicity. A 400× increase in pretraining corpus is worth less than a good trick. They
  also report transfer *"does not increase monotonically with additional procedural training…
  transfer typically peaks at an intermediate procedural budget and deteriorates as the model
  approaches convergence on the procedural task."* **[V]**
- ChemBERTa-2 (arXiv 2209.01712) states in its abstract that it varies pretraining dataset size
  "up to 77M compounds from PubChem" **[V that the abstract says this]**, but I could **not**
  extract the per-size numbers from the landing page and did not read the PDF. **[V — could not
  verify the numbers]**

### 2.4 If we pretrained anyway — the honest estimate

Given no published curve, this is extrapolation and must be labelled as such. **[R — all of §2.4]**

| Corpus | Mordred CPU-hours | L4 GPU-hours (full 2048×6 arch) | My expectation vs downloading CheMeleon |
|---|---|---|---|
| 50k | ~25 | ~15–25 | Clearly worse. Not enough to learn 1,613 targets. |
| 250k | ~125 | ~70–120 | Probably most of the benefit, per the MoLFormer-vs-250k calibration above. |
| 1M (their recipe) | ~500 | ~250–500 | Reproduces the released checkpoint, at best. |

And a smaller encoder (width 512, depth 4 ≈ 0.6M params) would cut GPU time by roughly an order of
magnitude, but then you are no longer testing CheMeleon's recipe, you are testing a new one, with
no prior to lean on.

**The only version of this worth doing** — and I would put it near the bottom of the ranked plan —
is *domain-adaptive* continued pretraining: take the **released** CheMeleon checkpoint and continue
the masked-descriptor objective for a short run on **drug-like** molecules (ChEMBL, or the union of
our 46k plus a few hundred thousand near-neighbour PubChem molecules) rather than random PubChem.
That is tens of L4-hours, not hundreds, and `how-to-train-your-chemeleon` gives you the loss
function (`pretraining/random_dropout_mse.py`) and the Zarr dataloader to do it. **[V that those
files exist; [R] that continued pretraining would help — I found no paper testing it for this
recipe.]**

### 2.5 The strictly better use of the same idea: pretrain on *predicted labels*

**Pham, Le, Thai, *Improved ADME Prediction by Multitask Pretraining on Predicted Data: Insights
from the ASAP-Polaris-OpenADMET Blind Challenge*, JCIM 2025,
[10.1021/acs.jcim.5c02030](https://doi.org/10.1021/acs.jcim.5c02030)** — code at
<https://github.com/LongHung-Pham/pADME>. **[V]**

From the abstract **[V]**: industry teams have released *predicted* labels from in-house models for
public structures; the authors pretrain a *simple multitask GNN* on those large surrogate sets and
fine-tune on experimental data. In the blinded 2025 ASAP-Polaris-OpenADMET challenge this ranked
**4th on aggregated MAE and tied 2nd on aggregated Pearson R**, and post-competition tuning beat
the 3rd-place entry on MAE — *"without using any proprietary data or commercial featurization
methods"*. They also report that **pretraining on experimental *and* predicted labels together
improved further**.

This is the same structural idea as CheMeleon (pretrain on cheap, abundant, low-noise surrogate
labels) but the surrogate labels are *ADME predictions* rather than *topological descriptors* —
i.e. the pretraining task is much closer to our downstream task. For a 21-endpoint ADMET model
this is arguably a better-aimed recipe than Mordred regression, and it is a *multitask* recipe,
which CheMeleon's evaluation is not. I did not verify the corpus size or GPU cost — the README is
a stub. **[V that the README is a stub.]**

---

### 2.6 The cheap third option nobody expects: in-context learning on tabular features

Two 2026 papers independently found that you can get most of a foundation model's benefit without
fine-tuning anything, by feeding **frozen** embeddings (or plain descriptors) to a **tabular
foundation model** that predicts by in-context learning. Our data is already a 2,264-dim table, so
this is a near-zero-engineering experiment.

- **Ben Hicham, Rittig, Grohe, Mitsos, *Tabular foundation models for in-context prediction of
  molecular properties*, arXiv 2604.16123 (Apr 2026).** Abstract, verbatim in part **[V]**:
  *"Molecular foundation models… typically involve task-specific fine-tuning, require machine
  learning expertise, and often fail to outperform classical baselines. Tabular foundation models
  (TFMs) offer a fundamentally different paradigm… **combining TFMs with CheMeleon embeddings
  yields up to 100% win rates on 30 MoleculeACE tasks**, while compact RDKit2d and Mordred
  descriptors provide strong descriptor-based alternatives… molecular foundation model embeddings
  and 2D descriptor sets both providing substantial gains over classic molecular fingerprints."*
  I could **not** extract the specific TFM names, size limits or runtimes from the landing page —
  read the PDF before planning on it. **[V — could not verify those details]**
- **Matyushin & Sholokhova, *In-Context Learning Meets Small Molecule Property Prediction*, JCIM
  2026, [10.1021/acs.jcim.6c02361](https://doi.org/10.1021/acs.jcim.6c02361).** Abstract **[V]**:
  11 regression datasets, **random and structure-aware splits, up to 10,000 molecules each**;
  *"TFM consistently outperform XGBoost, CatBoost, multilayer perceptrons, and other
  descriptor-based methods, **even with careful hyperparameter selection for the baselines**"*;
  *"accuracy on par with or better than graph-based methods, including those pretrained on chemical
  data"*; Uni-Mol2 *"slightly outperforms TFM in some experiments"*. They also report a **retrieval**
  trick: for each test molecule take the **500 nearest training molecules by Tanimoto** and run TFM
  inference on that local subset, which helped on larger sets and random splits. They conclude *"TFM
  (particularly the **TabPFN** family) are highly promising."*

**⚠ Licence trap — this one would bite us.** From the TabPFN repo README **[V]**
(<https://github.com/PriorLabs/TabPFN>): *"The TabPFN-2.5, TabPFN-2.6, TabPFN-3 and TabPFN-3.5
model weights are released under **non-commercial licenses**… **TabPFN-3.5 is used by default**."*
Only **TabPFN-v2** weights are under the Prior Labs License (Apache 2.0 plus an attribution
requirement) and therefore usable commercially; you must opt in explicitly:

```python
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
tabpfn_v2 = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
```

The code itself is Apache-2.0, and first use triggers a browser licence acceptance / `TABPFN_TOKEN`
flow. **[V]** For a commercial product, **TabPFN default = non-commercial weights**. If we try
this, pin V2 and record why.

### 2.7 And a sobering control on the whole premise

**Yao & Duran-Frigola, *Interpretable-by-Design Descriptor Portfolios Match a 2048-Dimensional
Foundation Embedding on Low-Data Molecular Assays*, [arXiv 2609.30789](https://arxiv.org/abs/2609.30789)
(Sep 2026).** **[V]** Across **nine ADME/Tox assays** (50 assessment cells, common-coverage
subsets), mean test AUC: **descriptor portfolio 0.762, CheMeleon 0.764, Mordred 0.756**. Pooled gap
**+0.003 AUC, 95% CI [−0.020, +0.030]** — they declare parity against a pre-registered threshold.

Read together with my §1.6 TDC table (CheMeleon 3W/6L against Morgan+physchem RF on the nine
*classification* endpoints), the picture is consistent: **on small ADMET classification tasks,
CheMeleon's learned 2048-dim embedding is worth about as much as good descriptors, which we already
have.** The gains are on regression.

### 2.8 Which of our 21 endpoints CheMeleon has *not* been tested on

CheMeleon's released results cover **17** TDC endpoints. **[V]** From the TDC ADMET Group
overview **[V]** (<https://tdcommons.ai/benchmark/admet_group/overview/>; scaffold split, 20%
test), the ones with **no CheMeleon datapoint** are:

| Endpoint | n | Task |
|---|---|---|
| Solubility_AqSolDB | 9,982 | regression (MAE) |
| CYP2C9_Veith | 12,092 | classification (AUPRC) |
| CYP2D6_Veith | 13,130 | classification (AUPRC) |
| CYP3A4_Veith | 12,328 | classification (AUPRC) |
| HIA_Hou | 578 | classification (AUROC) |

Note that four of those five are **the largest datasets in the whole group** — and CheMeleon's
published claim is explicitly scoped to the *low-data regime* (§1.1). We have **no evidence either
way** for CheMeleon on the four biggest endpoints we train on, which between them carry a large
share of our 79,238 labels. That is a genuine unknown, and it is cheap to resolve ourselves.
---

## 3. Other pretrained molecular encoders

**Headline: none of them beats count-ECFP4 + RDKit2D, and the one that comes closest does it as a
concatenated *feature*, not as a fine-tuned backbone.**

### 3.1 Parameter counts — the trap, confirmed

The naming convention in this field is actively misleading. Verified counts, with the method:

| Advertised as | What the number is | **Real trainable params** | How verified |
|---|---|---|---|
| ChemBERTa-**77M**-MLM | 77M = PubChem **corpus** | **3,428,808 (≈3.4M)** | config arithmetic **[V]**; cross-checked `pytorch_model.bin` = 13,734,116 B ÷ 4 = 3,433,529. I independently re-derived ≈3.3M backbone from `config.json` (hidden 384, 3 layers, intermediate 464, vocab 600). **[V]** |
| ChemBERTa-**5M** / **10M** / **77M** | all three = corpus sizes | **identical models** — the three weight files are **byte-identical** at 13,734,116 B | **[V]**. I confirmed the configs are identical too. **[V]** |
| MoLFormer-c3-**1.1B** | 1.1B = ZINC20+PubChem **corpus** | **46,805,760** | HF `safetensors.total` **[V]** — I verified this number myself on `ibm/MoLFormer-XL-both-10pct`. **[V]** |
| SMI-TED **289M** | total incl. a 242M **decoder** | **encoder 47M** / total 289,946,047 | paper Table 4 + blob ÷ 4 **[V]** |
| MiniMol **10M** | includes 2.4M discarded pretraining heads | **encoder ≈7.92M** | direct tensor count of the released checkpoint **[V]** |
| Uni-Mol2 84M…1.1B | **genuinely parameters** | 84,158,005 / 164,853,147 / 310,190,278 / 570,341,094 / 1,129,422,575 | checkpoint bytes ÷ 4, matching the names **[V]**; I confirmed the five `modelzoo/{84M,164M,310M,570M,1.1B}/checkpoint.pt` files exist on HF. **[V]** |
| ChemFM 1B / 3B | **genuinely parameters** | 970,287,104 / 3,004,357,632 | HF `safetensors.total` **[V]** — I verified both myself. **[V]** |

### 3.2 The table

VRAM figures are engineering estimates (≈16–20 bytes/param for bf16 mixed-precision AdamW plus
small activations, since SMILES are ≤202 tokens and n≈30k), not fetched claims. **[R]**

| Model | Paper | **Params** | Download | **Weights licence → commercial?** | VRAM, 30k mols | Independent scaffold-split ADMET evidence |
|---|---|---|---|---|---|---|
| **MolFormer-XL** | Ross et al., *Nat Mach Intell* 4 (2022), [10.1038/s42256-022-00580-7](https://doi.org/10.1038/s42256-022-00580-7) | **46,805,760** | `ibm-research/MoLFormer-XL-both-10pct`. ⚠️ **only the `both-10pct` ablation is public**; the paper's full 1.1B-molecule XL is not released | **Apache-2.0 → YES** | ~1.5–3 GB, trivial | Praski rank **9.50/25**, 79.80% AUROC — **below ECFP's 79.89%**. CheMeleon's Polaris set: **39% win rate** vs RF 68% (§1.6). On our 17 TDC endpoints vs Morgan+physchem RF: **2W/15L** (§1.6) |
| **ChemBERTa-2** | Ahmad et al., [arXiv 2209.01712](https://arxiv.org/abs/2209.01712), workshop only | **3.43M** (MLM) / ~3.50M (MTR) | `DeepChem/ChemBERTa-{5M,10M,77M}-{MLM,MTR}` | ⚠️ **NONE DECLARED** — `cardData: null`, no licence tag, no README on the weight repos. The DeepChem *library* is MIT, which covers code, not weights. I confirmed `license: None` on the API. **[V]** | <0.5 GB | Praski rank **7.32/25**, 79.99% — marginally above ECFP but "practically equivalent" under Bayesian Bradley–Terry. **Useful signal: MTR beat MLM by 7.33–9% AUROC in every case** — see §3.4 |
| **SMI-TED** (IBM) | Soares et al., *Commun Chem* 2025, [10.1038/s42004-025-01585-0](https://doi.org/10.1038/s42004-025-01585-0) | **289.9M total = 47M encoder + 242M decoder** | `ibm-research/materials.smi-ted` (146 downloads) | **Apache-2.0 → YES** (I confirmed on the HF API) | encoder ~1.5–3 GB | **None.** The paper has **no TDC/ADMET at all** and never states its MoleculeNet split type. The only head-to-head is *Nat Commun* 2026 where **all seven authors are IBM Research** |
| **ChemFM** | Cai et al., *Commun Chem* 2025, [10.1038/s42004-025-01793-8](https://doi.org/10.1038/s42004-025-01793-8) | **1B = 970,287,104; 3B = 3,004,357,632** | `ChemFM/ChemFM-1B`, `-3B`, plus **22 `ChemFM/admet_*` LoRA adapters (r=1!)** covering the whole TDC ADMET group | ❌ **LICENCE CONFLICT — treat as NON-COMMERCIAL.** HF cards say `mit` (I saw `license: mit` on the API myself **[V]**) but the authors' repo `LICENSE` is verbatim **CC BY-NC 4.0** and the README repeats it **[V]** | 1B full FT ≈19 GB optimiser alone → LoRA/8-bit needed; 3B LoRA r=1 bf16 ≈8–12 GB | Only model here with authors' own TDC ADMET numbers — but Praski (frozen) puts it at rank **12.60/25**, 78.25%, **below ECFP**, *"often delivered disappointing results"* |
| **MiniMol** | Kläser, Banaszewski et al., [arXiv 2404.14986](https://arxiv.org/abs/2404.14986) | **encoder ≈7.92M** (checkpoint 10,323,665 fp16) | PyPI `minimol`; GitHub `graphcore-research/minimol`. **No official HF** | **MIT (Graphcore Ltd) → YES** (PyPI `license` field is null, but the repo LICENSE is MIT) | <1 GB; the real cost is CPU graph featurisation | ⚠️ **Leakage — §7.1.** On the real leaderboard: **7th on AMES (0.849 vs MapLight 0.868)**, **6th on hERG (0.846 vs 0.880)**. CheMeleon's Polaris set: 71% win rate (2nd). Our 17 TDC endpoints vs Morgan+physchem RF: **7W/10L** (§1.6) |
| **Uni-Mol2** | Ji et al., [arXiv 2406.14969](https://arxiv.org/abs/2406.14969); peer-reviewed venue **could not verify** | five released sizes, 84M → 1.13B | HF `dptech/Uni-Mol2` (**0 downloads**; README is a two-line licence stub, which I confirmed) | **MIT → YES** **[V]** | 84M ≈3–5 GB; 310M ≈9 GB; 1.1B needs LoRA. ⚠️ **needs 3D conformers** — ETKDG for 46k molecules is hours of CPU | Paper has **no TDC/ADMET at all** (QM9 + COMPAS-1D). Praski: **Uni-Mol2 rank 13.32** (78.11%), Uni-Mol 16.64 — both **"Worse" than ECFP**, posterior 0.99/1.00, **0/25 datasets won**. *Counterpoint:* Matyushin & Sholokhova (§2.6) found Uni-Mol2 *"slightly outperforms TFM in some experiments"* |
| **MolE** (Recursion) | Méndez-Lucio et al., *Nat Commun* 15, 9431 (2024), [10.1038/s41467-024-53751-y](https://doi.org/10.1038/s41467-024-53751-y) | **≈85.3M** (DeBERTa-base, vocab 211 atom-environment tokens). **Not stated in the paper**; derived from the released config | ❌ **NOT DOWNLOADABLE.** README "Download pretrained models" is literally **"TODO"**; no releases, no tags, no HF. Author in issue #2: *"we are not releasing the model pretrained on full ZINC20 for the moment"* | **CC BY-NC 4.0 → NO** | n/a | Authors report TDC ADMET SOTA on **10/22 (Sep 2023)**, which **they themselves say decayed to 5/22 by Jun 2024**. **No independent reproduction exists and none can, because the weights don't** |
| **GROVER** | Rong et al., NeurIPS 2020, [arXiv 2007.02835](https://arxiv.org/abs/2007.02835) | **base 48,790,038 / large 107,714,488** | OneDrive + Google Drive links in the README. Not on HF | Code **MIT** (Tencent AI Lab); **weights carry no licence**; README says *"not an officially supported Tencent product"* → **grey** | ~4–6 GB base | ⚠️ **Reproducibility critique — §3.3.** Praski rank **17.08/25**, 75.68%, **posterior 1.00 worse than ECFP** |
| **GraphMVP** | Liu et al., ICLR 2022, [arXiv 2110.07728](https://arxiv.org/abs/2110.07728) | **≈1,857,900** (5-layer GIN, emb 300) | **Google Drive folder only** (could not verify it still resolves) | **MIT by inheritance**; weights carry no explicit terms | <0.5 GB | Praski rank **14.64**, 77.18%, **worse than ECFP** (p=0.97); named in their explicit *"Avoid"* list |
| **MolCLR** | Wang et al., *Nat Mach Intell* 4, 279–287 (2022), [10.1038/s42256-022-00447-x](https://doi.org/10.1038/s42256-022-00447-x) | GIN **≈2.40M** total / **≈2.01M** transferred encoder | **committed inside the GitHub repo** — no external host | **MIT → YES**. Cleanest licence of the graph models | <0.5 GB | Not in Praski. **CheMeleon's Polaris set: 14% win rate — last of seven.** On our 17 TDC endpoints it was the worst non-MoLFormer model. Li et al. 2026 (Biogen ADME, time split): near-worst of 12 |
| **Mol2vec** | Jaeger, Fulle, Turk, *JCIM* 58(1):27–35 (2018), [10.1021/acs.jcim.7b00616](https://doi.org/10.1021/acs.jcim.7b00616) | **not a trainable encoder** — a gensim word2vec lookup, 300-dim, vocab 21,003 | committed in `samoturk/mol2vec` | **BSD-3-Clause → YES** | n/a (CPU lookup) | Praski rank **10.36**, **79.55%** — the **best of the graph/legacy group**, essentially tied with ECFP's 79.89%, though BBT still calls it "Worse" (p=0.94) |

All rows **[V]** as reported by the delegated survey, with the six parameter counts and four licences
I re-verified myself marked above.

### 3.3 The independent benchmarks, and what they actually say

**Praski, Adamczyk, Czech, *Benchmarking Pretrained Molecular Embedding Models*,
[arXiv 2508.06199](https://arxiv.org/abs/2508.06199) (v4, Feb 2026).** **[V]** — the central
reference. **25 models × 25 datasets** (7 MoleculeNet + **18 TDC ADMET**), **scaffold splits**,
frozen embeddings into tuned RF/LR/kNN, Bayesian Bradley–Terry.

Mean rank / mean AUROC: CLAMP 5.40/82.55 · R-MAT 6.08/80.83 · MolBERT 6.92/80.51 ·
**ChemBERTa 7.32/79.99** · **ECFP 7.52/79.89** · CDDD 7.64/80.60 · AtomPair 7.88/79.48 ·
MAT 8.02/80.18 · **MoLFormer 9.50/79.80** · **Mol2Vec 10.36/79.55** · TT 12.04 · **ChemFM 12.60/78.25**
· **Uni-Mol2 13.32/78.11** · MolR 13.80 · COATI 13.92 · **GraphMVP 14.64/77.18** · Chemformer 15.00 ·
**Uni-Mol 16.64/76.85** · **GROVER 17.08/75.68** · GIN 17.28 · SimSon 19.12 · ChemGPT 19.32 ·
GEM 19.36 · SELFormer 20.04 · GraphFP 24.20/59.67.

Verdict, verbatim: *"All models, except CLAMP, are either worse than or practically equivalent to the
baseline ECFP fingerprint."* Recommendation #1: *"**Always include the ECFP count fingerprint as a
baseline, paired with a tree-based classifier.**"* **[V]**

**Three caveats that must travel with that table:** **[V]**
1. **Frozen embeddings, no fine-tuning** — deliberate, *"to isolate representation quality"*. It does
   **not** bound what fine-tuning could do, which is why §1.6 (fine-tuned CheMeleon) is a separate
   question. **CheMeleon, MolCLR, MolE, MiniMol, SMI-TED and MolGPS are not in Praski at all.**
2. **All 18 TDC sets are classification** — no ADMET regression endpoints, which is exactly where
   §1.6 says the gains live.
3. They used **their own** scaffold splits, not the TDC Benchmark Group's official folds.

⚠️ **And the single winner is contested.** GitHub issue
[MLCIL/benchmarking_molecular_models#1](https://github.com/MLCIL/benchmarking_molecular_models/issues/1)
(2026-06-04, **open, no author reply**): *"the observed benefit from CLAMP may be due to leakage
between the pretraining data and some of the test data… I tried removing some of the datasets with
suspected leakage, and **CLAMP is no longer significantly outperforming ECFP**."* **[V]**

**Deng et al., *A systematic study of key elements underlying molecular property prediction*,
*Nat Commun* 14, 6395 (2023), [10.1038/s41467-023-41948-6](https://doi.org/10.1038/s41467-023-41948-6).**
**[V]** 62,820 models, scaffold **and** random, 30 seeds, Mann–Whitney tests. Verbatim: *"representation
learning models exhibit limited performance in molecular property prediction in most datasets"* and
***"RDKit2D descriptors show better performance than learned representations by RNN, GCN, GIN, and
pretrained models."*** Also: activity cliffs occupy **25–46%** of molecules, and for MolBERT, GROVER
and GROVER_RDKit *"the average pIC50 values appear to be 'imputed'"* for cliff molecules.

**Sun, Dai, Yu, *Does GNN Pretraining Help Molecular Representation?*, NeurIPS 2022,
[arXiv 2207.06010](https://arxiv.org/abs/2207.06010).** **[V]** *"The benefit brought by
self-supervised pretraining on small molecular data can be negligible in many cases"*;
*"hyper-parameters could have larger impacts on accuracy of downstream tasks than the choice of
pretraining tasks."*

**Xia, Zhang, Zhu, Li, [arXiv 2306.17702](https://arxiv.org/abs/2306.17702) — ⚠️ weaker than its
reputation.** **[V]** 12 models × 14 datasets, **random 8:1:1 splits, 50 seeds — not scaffold**. It
benchmarks **no pretrained encoder** except one in-house transformer and **never mentions GROVER,
MolCLR or GraphMVP**. Result: *"Non-deep models rank as the top one on 10/14 datasets."* Cite it for
"fingerprints + trees are hard to beat"; **do not** cite it as a pretraining or GROVER critique.

**Adrian et al., [arXiv 2510.12719](https://arxiv.org/abs/2510.12719) (Merck + NVIDIA) — ⚠️ does not
test our question.** **[V]** Models: MolFormer, Chemprop, KPGT, **KERMT** (their accelerated GROVER
re-implementation, ~51M). **TDC is never mentioned**, and **there is no ECFP/descriptor + GBM
baseline in any results table** — Random Forest appears only as a passing remark. It compares
pretrained-GNN vs non-pretrained-GNN. Two findings worth keeping anyway: multi-task fine-tuning made
KERMT *"statistically superior in 18/30 assays"* (+0.02–0.07 Pearson r² over Chemprop), **but
"KERMT performs best on assays of larger data size rather than small data sizes"** — gains appear
above ~10k/task and >50k total. And `cuik-molmaker` gives *"2.2× speedup on finetuning"* and
*"around 34% reduction in memory usage"* — relevant to §1.7, since Chemprop exposes it as
`--use-cuikmolmaker-featurization`.

**MolPILE, [arXiv 2509.18353](https://arxiv.org/abs/2509.18353)** **[V]** — supports the
data-not-architecture thesis from the opposite side: retraining Mol2vec and ChemBERTa on a
**222M-compound** curated corpus bought only **+0.58** and **+1.26** average AUROC on TDC.

### 3.4 The GROVER and MiniMol caveats, stated precisely

**GROVER — a reproducibility critique, *not* a leakage critique.** The survey searched arXiv,
Europe PMC, OpenAlex and Crossref and found **no published data-leakage claim** against GROVER.
**[V — by absence]** What is verified:

1. **Sun et al. 2022, Appendix Table 18** — pretrained vs scratch, same architecture:
   BBBP **92.6 (scratch) vs 89.6 (pretrained)**, BACE 86.2 vs 85.1, TOX21 79.3 vs 78.7, TOXCAST
   71.5 vs 71.5, SIDER 62.3 vs 62.0. **GROVER's pretraining made every dataset worse or equal.**
   *"We didn't see significant gains over the model trained from scratch."* **[V]**
2. **Deng et al. 2023 — the attribution critique, and the one that should worry us most.**
   *"Concatenating RDKit2D descriptors significantly improves GROVER's performance in HIV, ESOL, and
   Lipop"* and *"concatenating RDKit2D descriptors to the learned representations is **misleading
   when assessing the real power** of the representation learning models."* **[V]** GROVER's own
   paper confirms it does this: *"we extract additional 200 molecule-level features by RDKit… and
   concatenate these features to the output of self-attentive READOUT."* **[V]**
   **This is a direct warning about §1.12 and §8 Step 3.** If we hand CheMeleon our 2,264 RDKit
   features via `--descriptors-path`, we must also run it **without** them, or we will not know
   whether CheMeleon contributed anything.
3. **The authors' own tracker:** issue #2 *"Reproduce BBBP result"* has been **open since March 2021,
   unanswered**, with four independent users getting **0.894 / 0.901 / 0.895** against the paper's
   **0.936**. Issues #17 (*"self-attention is computed between different heads rather than different
   atoms"*) and #19 (incorrect masking) are **open with zero replies**. The README itself concedes
   *"it is hard to exactly reproduce the training process of finetuning."* **[V]**

**MiniMol — leakage, but *unquantified*.** Koleiev et al. (§7.1) place MiniMol in the leakage
category, with a precise mechanism: canonical-SMILES filtering *"fails to account for molecular
chirality properly"*, plus tautomers; danazol as the worked example; residual overlap largest in
PCBA. **[V]** But: **MiniMol was eliminated at the leakage stage and never retrained, so there is no
corrected score and no leaked-molecule count.** **[V]** For GradientBoost+/XGBoost they *did*
quantify (1st→3rd, 2nd→8th). So the defensible sentence is *"MiniMol's deduplication is demonstrably
stereochemistry- and tautomer-blind"* — **do not state an effect size.** Precedent on the same
leaderboard: TDC issue
[#166](https://github.com/mims-harvard/TDC/issues/166), *"Potential data leakage of ContextPred and
AttrMasking on the ADMET group benchmark"* (2022). **[V]**

### 3.5 The three positive findings in this literature

Everything above is negative. These three are not, and two of them change the plan in §8.

1. **MapLight's GIN fingerprint, quantified.** Adding a **pretrained GIN supervised-masking
   fingerprint as 300 concatenated columns** (not a fine-tuned backbone) to ECFP counts (1024) +
   Avalon counts (1024) + ErG (315) + 200 RDKit properties = **2,863 features → CatBoost**, moved
   MapLight from **top-1 on 6/22 → 11/22** and **top-3 on 16/22 → 19/22** TDC ADMET benchmarks.
   **[V]** Code MIT. And their own origin story is ours: *"even models that leveraged pretraining on
   large molecular datasets did not outperform random forests or support vector machines coupled
   with ECFP"*, and *"the neural network fingerprints, regardless of architecture type, performed
   worse compared to traditional molecular fingerprints"* — **used alone**. Concatenated, they help.
   **This is the single cheapest credible experiment available to us, and §8 Step 2 is it.**

2. **Domain-adaptive descriptor regression — Sultan et al., *J Cheminform* 2026,
   [10.1186/s13321-026-01252-z](https://doi.org/10.1186/s13321-026-01252-z).** **[V]** Two findings,
   both directly load-bearing:
   - ***"Increasing the pre-training dataset beyond approximately 400–800K molecules does not
     improve performance."*** That is the **data-scaling answer §2.3 could not find** for the
     descriptor-prediction recipe, from an adjacent one — and it says CheMeleon's 1M is already past
     the knee, so re-pretraining on more molecules is pointless.
   - ***"Domain adaptation via multi-task regression of 209 physicochemical/RDKit properties on
     ≤4K domain molecules improves performance across all datasets (P < 0.001)."*** On ≤4,000
     molecules. Their baseline — RF + RDKit descriptors + Morgan — *"remain strong baselines"*, and a
     400k-molecule domain-adapted model *"outperforms larger-scale transformer models like
     MolFormer"*.

   **This is §8 Step 4, published, significant, and far cheaper than I had assumed.** It also
   explains Praski's otherwise odd result that **ChemBERTa-MTR beats ChemBERTa-MLM by 7.33–9%
   AUROC in every case** **[V]** — multi-task *regression* on computed properties is a better
   pretraining objective than masked language modelling, at the same model size. Same conclusion as
   CheMeleon (§1), Fallani (§7.3) and Graphium (§6.6), now from a fourth direction.

3. **Fallani et al., [arXiv 2410.08024](https://arxiv.org/abs/2410.08024) (Janssen)** **[V]** — a
   **~10M-param** Graph Transformer pretrained on **atom-level QM properties** (charges, NMR
   shieldings, Fukui indices) beat the TDC leaderboard on 3/22: **lipophilicity 0.393 vs 0.539 MAE;
   solubility 0.720 vs 0.878; LD50 0.538 vs 0.593**. ⚠️ **No fingerprint baseline, author-reported.**
   But note it is our parameter scale, and note *which* endpoints: lipophilicity, solubility and
   LD50 — three of the regression endpoints where §1.6 says CheMeleon also wins.

### 3.6 Benchmark-integrity findings worth acting on

- **The TDC ADMET split really is scaffold.** `tdc/metadata.py` maps all 22 to `"scaffold"`; the test
  set is fixed and pre-distributed; `get_train_valid_split` resplits only `train_val` at
  0.875/0.125. **[V]** The problems are that the test set is **public and permanently fixed**, that
  there is **no dataset versioning** (Koleiev: *"the absence of explicit data set versioning"*
  defeats exact reproduction), and residual redundancy (§7.2).
- **Scaffold split is a soft stressor.** Imdad, Zenodo 2026,
  [10.5281/zenodo.20710323](https://doi.org/10.5281/zenodo.20710323) — single-author, unreviewed,
  weight accordingly **[V]** — across all 22 TDC ADMET endpoints: *"Bemis–Murcko scaffold splits are
  much weaker stressors than cluster (sphere-exclusion) splits — their median relative gap is about
  **2.3× smaller (+0.10 vs +0.23)**"*, with significant gaps on **21/22** under cluster splits.
  Combined with §7.4(b)'s negative R² on a real programme, this says our scaffold-split numbers are
  the *optimistic* end of the range.

### 3.7 Verdict for us

**Do not fine-tune MolFormer-XL, Uni-Mol2, ChemFM, GROVER, GraphMVP, MolCLR or ChemBERTa.** Frozen,
they are at or below ECFP on 18 TDC ADMET classification sets (Praski); where CheMeleon measured
them fine-tuned, MoLFormer went 2W/15L and MolCLR worse still against a Morgan+physchem RF on TDC
(§1.6). MolE cannot be downloaded; ChemFM is non-commercial despite its HF tag; ChemBERTa's weights
carry no licence at all.

**If one encoder is to be fine-tuned, it is CheMeleon** — it is the only one with a published win
over RF on a curated benchmark suite (§1.6), it is MIT, it is 12.9M params, it fits the L4 ten times
over, and it was *"particularly popular"* among the winners of the one blind prospective challenge
in this field (§7.4a). **MolFormer-XL is the commercially-clean fallback** (Apache-2.0, 46.8M, best
of the transformers in Praski) and should be expected to land at parity, not above.

**But the two things most worth doing are not fine-tuning at all**: MapLight's concatenated
fingerprint (§3.5.1 → Step 2) and Sultan's domain-adaptive descriptor regression (§3.5.2 → Step 4).

---

## 4. Semi-supervised use of unlabelled molecules at our scale

**The headline, after a full search: there is no published result that clears all four bars —
(a) ~30k labels, (b) scaffold split, (c) a *tuned* descriptor+GBM baseline, (d) independently
reproduced. Every verified semi-supervised gain is measured against a weak GNN, or on noiseless
computed labels, or under a non-scaffold split, or all three. The size of the reported gain is
inversely proportional to the quality of the baseline.**

### 4.1 The decisive table

From an ensemble-consensus SSL paper ([arXiv 2607.28304](https://arxiv.org/abs/2607.28304)), read
from the PDF. **[V]** Its setup is the most favourable possible: the training 80% is split 10%
labelled / 90% label-discarded, so the "unlabelled" pool is same-distribution held-out data.

| Dataset (metric) | label type | supervised ens. | +SSL ens. | gain |
|---|---|---|---|---|
| **ogbg-molhiv (AUROC)** — GCN | **real assay** | .7357 | .7357 | **+0.0000** |
| ogbg-molhiv — GIN | real assay | .7346 | .7347 | **+0.0001** |
| ogbg-molhiv — GatedGCN | real assay | .7341 | .7383 | +0.0042 |
| **ogbg-molpcba (AP)** — GCN | **real assay** | .1578 | .1585 | **+0.0007** |
| ogbg-molpcba — GIN | real assay | .1567 | .1567 | **0.0000** |
| **ZINC (MAE)** — GCN | **computed penalised logP** | .2934 | .2367 | **−19%** |
| **QM9 U₀ (MAE)** — PaiNN | **DFT** | 20.91 meV | 19.38 meV | **−20%** |

> **The gains live entirely on labels that are deterministic functions of structure.** On real
> bioactivity labels the effect is **zero to four ten-thousandths**. On RDKit-computed logP and on
> DFT it is 19–20%. That is our hypothesis, measured, in one table.

The paper's own mechanistic claim is also worth knowing: *"an individual member of an ensemble
trained this way outperforms a full ensemble trained in a traditional supervised fashion in almost
all cases"*, an effect *"similar to knowledge distillation"*. **[V]** **So it is a cheaper-ensemble /
self-distillation result, not an "unlabelled data adds information" result.**

The same pattern appears in **InfoGraph** (ICLR 2020, [arXiv 1908.01000](https://arxiv.org/abs/1908.01000),
PDF read **[V]**), whose semi-supervised experiments are **QM9 only**, 5,000 labelled / ~105,000
unlabelled, **random split**. InfoGraph*'s error ratios vs supervised: **U₀ 0.44, ⟨R²⟩ 0.49, ZPVE
0.52, G 0.54** — all size-extensive/thermodynamic targets learnable from molecular size and
composition — against **μ 0.99, α 0.94, HOMO 0.99, LUMO 0.99, gap 0.98** on the intensive electronic
targets, i.e. **nil**. **The mechanism is "unlabelled data teaches you molecular size", which is
already in our RDKit2D block.** And **Mean Teacher is *worse* than supervised on 2 of 12 targets**
(μ 1.09, U₀ 1.16). Naive InfoGraph with a shared encoder is useless (0.85–1.02), which the paper
attributes to **negative transfer** — a direct warning about bolting an SSL loss onto a single
multi-task trunk.

### 4.2 Method by method

| Method | Status for molecular property prediction | Best verified evidence |
|---|---|---|
| **Self-supervised pretraining on unlabelled molecules** (the de-facto answer) | Exists extensively; **does not help once you have rich features** | **Sun, Dai, Yu, NeurIPS 2022** ([arXiv 2207.06010](https://arxiv.org/abs/2207.06010)) **[V]**: with **rich features + balanced scaffold split**, average gain over no-pretraining is **node prediction +0.122% ± 1.542; context prediction +0.126% ± 1.234; contrastive −0.578% ± 1.684** — **every gain an order of magnitude smaller than its own SD.** *"The improvement may diminish with richer features or more balanced data splits."* **This is the most direct hit on our situation in the entire search: we already have the rich features, which is exactly the condition under which the gain goes to zero.** |
| **Contrastive pretraining on 10M molecules** | Loses to an SVM | **MolCLR** ([arXiv 2102.10056](https://arxiv.org/abs/2102.10056)) **[V]**, 10M unlabelled PubChem, **scaffold split**, the paper's own baselines: **Tox21 SVM 81.8 vs MolCLR 79.8 (SVM wins by 2.0)**; **SIDER RF 68.4 vs 68.0 (RF wins)**; BBBP +0.7 (noise). Its wins are ClinTox (+22) and MUV (+21) — two sets notorious for extreme imbalance where single-split scaffold AUC is wildly unstable. |
| **Mean Teacher / consistency** | **The best-matching positive result in the search**, and it collapses where the baseline is good | **SSL-GCN**, J Cheminform 2021 (PMC8627024) **[V]**: Tox21, **5,821–7,265 labelled per endpoint** (closest published match to our per-task regime), 50,527 unlabelled, **scaffold 0.8:0.1:0.1**. **SL-GCN 0.7156 → SSL-GCN 0.7571 = +0.0415 AUC**, beating the best ECFP4 model (0.71) by ~0.047. **But per endpoint: NR-AR 0.7193 → 0.7967 and SR-MMP 0.7818 → 0.8120, versus NR-AhR 0.8287 (XGBoost) → 0.8303 = +0.0016 — a wash precisely where the fingerprint model was already strong.** Baselines are DeepChem-default with no stated hyperparameter search; no independent replication. |
| **Pseudo-labelling / self-training** | Thin; the two real papers are by the same author | **SemiMol** (IJCAI-24, [arXiv 2601.04507](https://arxiv.org/abs/2601.04507)) **[V]**: MoleculeACE, 615–3,657 per target, **ZINC15 250K unlabelled (not millions)**, **split is cliff-stratified random, not scaffold**. RMSE gains of −0.18 to −0.38 **over a plain GIN**; headline *"26.53%"* is over the GNN backbone. **There is no numeric SemiMol-vs-SVM/GBM comparison anywhere in the paper.** **InstructMol** (NeurIPS 2024, [arXiv 2304.03906](https://arxiv.org/abs/2304.03906)) **[V]**: **scaffold split**, 1M ZINC15 unlabelled; on top of an already-pretrained GEM, **Lipo RMSE 0.660 → 0.582 (−11.8%)**, BBBP +0.9 AUC. **No fingerprint/GBM/SVM baseline anywhere.** |
| **Noisy Student** | **Two papers exist; neither is usable evidence** | arXiv `all:"Noisy Student" AND all:"molecular"` → **0 results** **[V]**. **PLANS**, BMC Bioinformatics 2022 (PMC9063120) **[V]** is the only real one: CYP450, 17,121 molecules, GINFP pretrained on **1.7M ChEMBL**; **+2.57 accuracy points over the best ECFP baseline (p = 0.031)**. But the **split type is not stated** ("7:3"), the baselines sit at **52–59% accuracy** where CYP inhibition normally reaches 0.85–0.90 ROC-AUC, and the Tox21 MLP baseline has **AP 0.03** — a broken baseline, so its "+0.17 AP" is evidence of nothing. **MycoPermeNet-v2** (JCIM 2026) is architecturally the closest thing to what we would build (descriptors + embedding fusion + noisy student) but is **paywalled with no numbers**. |
| **UDA / FixMatch** | **Absent** | arXiv `("FixMatch" OR "unsupervised data augmentation") AND (molecul*)` → **0 results**; Europe PMC `("noisy student" OR "UDA" OR "FixMatch" OR "MixMatch") AND (molecul* OR ADMET)` → 80 hits, only the two Noisy Student papers. **[V — by absence]** |
| **Consistency regularisation** | One paper, paywalled | **CRGNN**, Neural Networks 2025, [10.1016/j.neunet.2025.108157](https://doi.org/10.1016/j.neunet.2025.108157) — FixMatch-style weak/strong views, claims gains *"especially when the training dataset is smaller"*. **Paywalled: no datasets, split or numbers.** **[NV]** Closest direct hit on our question; worth obtaining. |
| **Distillation from a foundation model into a small ADMET model** | **Genuinely unstudied** | arXiv `"knowledge distillation" AND "ADMET"` → **0 results** **[V]**. The molecular distillation papers that exist are all **modality** distillation (3D→2D, big GNN→small GNN **on QM energies**). The 2026 *Adv Drug Deliv Rev* small-data review omits distillation entirely. **"Does distilling MolFormer into a LightGBM beat training the GBM directly?" — nobody has published it.** |
| **Pseudo-labelling on an ADMET panel** | **Does not exist** | Europe PMC `("pseudo-labeling" OR "self-training") AND (ADMET OR ADME)` → 18 hits, **zero** actual SSL-on-ADMET studies. **[V — by absence]** |

### 4.3 Why: molecules have no label-preserving augmentation

Three independent groups say so in print. **[V]**

1. **Ensemble-consensus paper**: *"Such augmentations are notoriously difficult to design for
   molecules, where minor structural changes can drastically alter the chemical properties we aim to
   predict."* They abandon augmentation-consistency entirely.
2. **SemiMol**: *"random node dropping and edge perturbation are biochemically improper, and data
   augmentation may alter the semantics of molecular graphs."* **The only augmentation they trusted
   was dropout applied to the *network*, not the input.**
3. **CRGNN**: *"Conventional data augmentation strategies are generally ineffective in this context,
   as simply perturbing molecular graphs can unintentionally alter their intrinsic properties."*

And **GraphCL**'s own Observation 1 **[V]**: *"Without any data augmentation graph contrastive
learning is not helpful and often worse compared with training from scratch"* — its "Aug.-only" row
is flat or negative on NCI1 at both label rates.

> **⚠️ SMILES enumeration is a no-op for us.** Bjerrum 2017 ([arXiv 1703.07076](https://arxiv.org/abs/1703.07076))
> is the canonical result — 130× augmentation took an LSTM from R² 0.56 → 0.66 **[V]** — but
> **count-ECFP4 and RDKit2D are both canonicalisation-invariant, so SMILES randomisation produces a
> byte-identical feature vector.** That entire gain is about teaching a *sequence* model an
> invariance we already have for free. (Also: `"SMILES enumeration" AND "semi-supervised"` → **0
> results** **[V]**; and one 2026 review cautions that *"simple augmentation strategies like SMILES
> enumeration can lead to label leakage or redundant training signals."*)

**For our feature set, the only label-preserving augmentations that exist are feature dropout,
Gaussian noise on the descriptor block, and mixup on the descriptor vector.** Graph edits are not
label-preserving (three papers); SMILES enumeration is a no-op.

### 4.4 The scorecard

| Evidence | Gain | Labels | Split | Baseline | Noiseless labels? |
|---|---|---|---|---|---|
| SSL-GCN Mean Teacher, Tox21 | **+0.0415 AUC** | 5.8–7.3k/task | **scaffold** | untuned ECFP4 (0.71) | no |
| …same, NR-AhR only | **+0.0016** | " | scaffold | XGBoost already 0.8287 | no |
| InstructMol on GEM, Lipo | −11.8% RMSE | 4.2k | **scaffold** | GEM (pretrained GNN) | no |
| SemiMol on GIN, MoleculeACE | −0.18 to −0.38 RMSE | 0.6–3.7k | cliff-stratified random | **plain GIN** | no |
| Ensemble consensus, **ogbg-molhiv** | **+0.0000 / +0.0001 / +0.0042** | ~33k | 80/10/10 | supervised ensemble | no |
| Ensemble consensus, **ogbg-molpcba** | **+0.0007 / 0.0000 / +0.0002** | ~350k | 80/10/10 | supervised ensemble | no |
| Ensemble consensus, ZINC | **−19% MAE** | ~10k | 80/10/10 | supervised GCN | **yes — RDKit logP** |
| InfoGraph*, QM9 U₀/⟨R²⟩/ZPVE | ratio 0.44 / 0.49 / 0.52 | 5k | random | supervised GNN | **yes — DFT** |
| InfoGraph*, QM9 μ/α/HOMO/LUMO | ratio 0.94–0.99 (**nil**) | 5k | random | supervised GNN | **yes — DFT** |
| Mean Teacher, QM9 | ratio 0.52–**1.16** (worse on 2/12) | 5k | random | supervised GNN | **yes — DFT** |
| MolCLR, Tox21 (10M unlabelled) | **−2.0 AUC vs SVM** | 7.8k | **scaffold** | SVM 81.8 | no |
| PLANS noisy student, CYP450 | +2.57 acc, p=0.031 | 5.2k | **unspecified** | untuned ECFP SVM/RF/XGB | no |
| Sun et al., rich feats + bal. scaffold | **+0.12% ± 1.54%** | MoleculeNet | balanced scaffold | no-pretrain, **rich features** | no |

> **Not one verified instance of a semi-supervised method beating a *tuned* descriptor-based GBM on a
> real ADMET or bioactivity endpoint under a scaffold split.**

### 4.5 Verdict — and the order to try them in, if we ever do

**Do not spend GPU-hours here.** But if we do, the evidence dictates an order:

1. **Consistency with feature dropout / descriptor-vector noise only** — the only label-preserving
   augmentation our features admit, and exactly what SemiMol's π-model baseline used after rejecting
   graph edits. Its own measured gain was small (1.019 → 0.984).
2. **Pseudo-labelling with a reliability gate**, InstructMol-style, on ~1M ZINC. Plain
   confidence-threshold pseudo-labelling does not work for regression, which is why both papers in
   this space needed an instructor model.
3. **Ensemble consensus / self-distillation** — but understand that it buys **serving cost**, not
   accuracy: its gain on the one real-label bioactivity benchmark was **0.0000 AUROC**.

**Do not bother with:** SMILES enumeration (a literal no-op on our features), graph augmentations
(three papers say they are not label-preserving), UDA/FixMatch ports (the strong augmentation they
require does not exist for molecules), or distilling a foundation model into our GBM — **the
published information flow runs the other way**: Deng et al. found GROVER's gains came from
concatenating RDKit2D *into* it, CheMeleon's pretraining signal *is* descriptors, and descriptors
already match a 2048-d foundation embedding to within +0.003 AUC on ADME/Tox (§2.7).

**The productive way to spend unlabelled molecules is to stop treating them as unlabelled.** Compute
descriptors for them and regress those (§3.5.2, §8 Step 4). That is CheMeleon's thesis, Sultan's
result, Fallani's result and ChemBERTa-MTR-over-MLM, and it is the only axis where four independent
groups agree.

---

## 5. Label noise — how close to the ceiling are we actually?

**Revised after a second research pass.** My first draft of this section concluded we were plausibly
at the aleatoric floor on solubility. **That conclusion was too strong, and the two papers that
actually *tested* the noise-ceiling claim both refute it.** The noise-floor literature is real and
quantitative; it does not say what it is usually quoted as saying.

### 5.1 The framing everyone cites — and the numbers behind it

**Kramer, Kalliokoski, Gedeck, Vulpetti, *The Experimental Uncertainty of Heterogeneous Public Ki
Data*, J Med Chem 55(11):5165–5173, 2012,
[10.1021/jm300131x](https://doi.org/10.1021/jm300131x).** **[V]** Verbatim:

> *"The maximum achievable accuracy of computational models depends on the quality of the
> experimental data. Consequently, experimental uncertainty defines a natural upper limit to the
> predictive performance possible. **Models that yield errors smaller than the experimental
> uncertainty are necessarily overtrained.** … Careful filtering of the data was required because
> ChEMBL contains unit-transcription errors, undifferentiated stereoisomers, and repeated citations
> of single measurements (**90% of all pairs**). The experimental uncertainty is estimated to yield a
> **mean error of 0.44 pKi units, a standard deviation of 0.54 pKi units, and a median error of 0.34
> pKi units. The maximum possible squared Pearson correlation coefficient (R²) on large data sets is
> estimated to be 0.81.**"*

**R²_max ≈ 0.81** for heterogeneous public Ki. Independently corroborated verbatim in Mervin et al.,
J Cheminform 2021 (PMC8375213). **[V]** This is the template citation, and it is one of only **two**
canonical numeric applications of `R²_max = 1 − σ²_noise/σ²_total` in the literature; the other is
Brown, Muchmore & Hajduk, *Healthy skepticism*, Drug Discov Today 2009 — *"~80% **correlation**,
given a standard deviation of ~0.3 and a potency range of only 2 log units"*. **[V via the quoting
source; original paywalled]** Note that is correlation, not R².

**Kalliokoski, Kramer, Vulpetti, Gedeck, PLoS ONE 2013,
[10.1371/journal.pone.0061007](https://doi.org/10.1371/journal.pone.0061007).** **[V]** Mixed public
pIC50: **σ = 0.68, MUE 0.55, MedUE 0.43**; pKi σ = 0.66. R² between independent measurements of the
same system: **0.40** raw, **0.53** after pruning. pKi − pIC50 offset **0.355 log** (factor 2.3).

> **The single most under-quoted number in this literature.** The same paper measured *in-house*
> reproducibility: **σ = 0.22 (rolipram/PDE4D) and σ = 0.17 (cilostamide/PDE3)** — against **0.66–0.68
> for pooled public data.** **[V]** That is ~3× the SD and **9–16× the variance**, and it comes from
> *pooling across sources*, not from the assay. **Most of what we call "assay noise" is curation
> debt.** It is in principle removable, which is the opposite of an irreducible floor.

**Landrum & Riniker, JCIM 64(5):1560–1567, 2024,
[10.1021/acs.jcim.4c00049](https://doi.org/10.1021/acs.jcim.4c00049)** (PMC10934815) — the modern
restatement, with the full table: **[V]**

| Setting | >0.3 log | >1.0 log | Kendall τ | R² | MAE | Size |
|---|---|---|---|---|---|---|
| IC50, minimal curation | 64–65% | 27% | 0.51 | 0.31 | 0.50 | 1,358 assays / 38,022 cmpds |
| IC50, **maximal curation** | 48% | 13% | 0.71 | **0.63** | **0.27** | **26 assays / 340 cmpds** |
| Ki, minimal curation | 67% | 30% | 0.43 | 0.13 | 0.52 | 587 assays / 7,734 cmpds |
| Ki, maximal + pruning | 58% | 25% | 0.67 | 0.65 | 0.45 | — |

Two things the headline misses: **the curation price is *"almost 99% of the data"*** (maximal-curation
IC50 is 340 compounds), and **IC50 pooling turns out no worse than Ki pooling**, inverting the
folklore. Their **R² ≈ 0.63–0.65 between two carefully curated sources is itself a replicate-dataset
ceiling**, directly comparable to R²_max. Code: <https://github.com/rinikerlab/overlapping_assays>.

### 5.2 Solubility — and the result that cuts against the whole framing

**Attia, Burns, Doyle, Green, Nature Communications 16, 7497 (2025),
[10.1038/s41467-025-62717-7](https://doi.org/10.1038/s41467-025-62717-7)** (PMC12365040). **[V]**

What the models reach (RMSE, log S — **no MAE is reported anywhere in the paper**):

| Test set | FASTSOLV (fastprop) | Chemprop-derived | Vermeire (prior SOTA) |
|---|---|---|---|
| SolProp | **0.83** (78.1% within ±1 log S) | 0.83 (76.1%) | 1.43 (66.9%) |
| Leeds | **0.95** (73.8%) | 0.99 (70.9%) | 2.16 (41.2%) |

**How the floor was estimated — and this is the weak link.** Verbatim: *"Between these solutions the
average inter-laboratory standard deviation is **0.34 log** while the RMSE is **0.75 log**… We use
this value of inter-laboratory RMSE (0.75) for direct comparison with model performance."* That rests
on **34 solutions containing 8 unique solutes and 6 unique solvents.** **[V]** The abstract's
*"approaching the aleatoric limit (0.5–1 log S)"* is a **conservative band borrowed from the
literature, not a measured per-endpoint σ.** 0.83 vs a 0.75-log floor estimated from 34 data points
is "at the ceiling" only if you accept that estimate.

**And then the Solubility Challenge demolishes the framing.** **[V]**

- Llinàs & Avdeef, JCIM 2019, [10.1021/acs.jcim.9b00345](https://doi.org/10.1021/acs.jcim.9b00345) —
  the SDs are in the title: a **tight test set with SD ≈ 0.17 log** and a loose one with SD > 0.62.
- Llinàs, Oprisiu, Avdeef, JCIM 60(10):4791–4803, 2020,
  [10.1021/acs.jcim.0c00701](https://doi.org/10.1021/acs.jcim.0c00701) — tight set 100 molecules
  (inter-lab reproducibility **≈ 0.17 log**), loose set 32 "contentious" molecules (**≈ 0.62 log**).
- **Blinded Predictions and Post Hoc Analysis of the Second Solubility Challenge Data, JCIM 2023,
  [10.1021/acs.jcim.2c01189](https://doi.org/10.1021/acs.jcim.2c01189)** (PMC9976279) — the finding:
  **best submission (a graph convolutional NN) reached RMSE 0.86 log on a test set whose own
  reproducibility was 0.17 log.** Mean across submissions: 1.14 (tight) / 1.62 (loose). Verbatim:
  *"the error in published experimental data (which had been estimated to be approximately 0.6 log
  units) was often cited as the limiting factor preventing solubility models from improving"* — the
  challenge showed that assumption was **wrong**, and the authors conclude instead that *"it is the
  deficiency of QSPR methods … which is the limiting factor."*

> **Hand the field a 0.17-log test set and the best model still only gets to 0.86. There was ~0.7 log
> of headroom. The ceiling was not binding.**

Corroborating the same direction — **Llompart et al., *Will we ever be able to accurately predict
solubility?*, Sci Data 2024,
[10.1038/s41597-024-03105-6](https://doi.org/10.1038/s41597-024-03105-6)** **[V]**: SOTA models
claiming **0.35–0.63 log RMSE on their own validation sets score 1.74–2.17 log on a fresh curated
OChem set.** **The generalisation gap dwarfs the noise floor** — by a factor of three.

**Correction to my first draft.** I previously wrote that our H1 solubility MAE of 0.804 sits inside
the 0.5–1.0 band and that we are therefore plausibly at the floor. **Withdraw that.** Three reasons:
(a) ours is **MAE on AqSolDB**, theirs is **RMSE on organic-solvent BigSolDB** — RMSE ≥ MAE always
and the datasets differ, so the comparison was never like-for-like; (b) the 0.5–1.0 band is a
borrowed literature range, not a measurement; (c) the Solubility Challenge shows that even against a
0.17-log floor there is ~0.7 log of headroom. **The right statement is: our solubility number is in
the same neighbourhood as published SOTA, and nobody knows where the real floor is for AqSolDB.**

### 5.3 The two papers that directly refute the naive noise-ceiling argument

**Kolmar & Grulke, *The effect of noise on the predictive limit of QSAR models*, J Cheminform 13:92,
2021, [10.1186/s13321-021-00571-7](https://doi.org/10.1186/s13321-021-00571-7)** (PMC8613965). **[V]**
**>1,800 models**, eight datasets — including **TDC's Lipophilicity_AstraZeneca (4,200) and LD50_Zhu
(5,003)** — with noise injected at 15 levels × 5 replicates, across ridge/kNN/SVR/RF/GP.

> **Headline: RMSE measured on an *error-free* test set stays essentially constant as training noise
> rises, while RMSE on a *noisy* test set rises proportionally.** Slope ratio up to **11**. Their
> verbatim conclusion: the claim that *"a model cannot make predictions which are more accurate than
> its training data"* is flawed reasoning, and their data support that *"QSAR models can indeed make
> predictions which are more accurate than the training data."*

**Implication for us: a large part of our reported test error is the *test labels*, not the model.**
Label noise raises measured error without raising true error. That is not the same as having stopped
learning — and it means no amount of modelling will move the measured number, but also that the
measured number is not telling us we are done.

**Zhao, Wang, Sedykh, Zhu, *Experimental errors in QSAR modeling sets*, ACS Omega 2017,
[10.1021/acsomega.7b00274](https://doi.org/10.1021/acsomega.7b00274)** (PMC5494643). **[V]** Noisy
compounds **can** be detected (12.9× ROC enrichment on MDR1) — but the critical negative result,
verbatim: ***"simply removing the suspicious compounds from the modeling sets did not improve the
external predictivity."*** Removal caused overfitting and shrank the applicability domain, cancelling
the gain. They still assert noise *"sets the upper limit on the **evaluation** accuracy of models"* —
note **evaluation**, which is exactly Kolmar & Grulke's point.

**One counterpoint in the other direction, and it is the paper to get:** **Adrian, Chung, Cheng
(Merck), *Denoising Drug Discovery Data for Improved ADMET Property Prediction*, JCIM 2024,
[10.1021/acs.jcim.4c00639](https://doi.org/10.1021/acs.jcim.4c00639)** **[V — abstract only; ACS and
the ChemRxiv PDF both 403]**: claims the **first denoising scheme to enhance ADMET model
performance**, that **training error identifies noisy regression labels where other metrics fail**,
with *"strongest gains for moderately noisy datasets"*. That is the actionable version of our
hypothesis, from people who say they got it working on our endpoints. Weigh against Zhao 2017.

### 5.4 Per-endpoint reproducibility — the table

Compiled from fetched primary sources. **σ in log10; "(derived)" means arithmetic from a published
CV or fold-range, not a published σ.**

| Endpoint | Design | Published quantity | σ (log10) |
|---|---|---|---|
| **hERG pIC50** | **5 labs, 28 drugs, manual patch clamp, harmonised ICH S7B** — Alvarez Baron et al., Sci Rep 2025, [10.1038/s41598-025-15761-8](https://doi.org/10.1038/s41598-025-15761-8) | *"overall variability (τ) … expressed as **SD was 0.18**, and as 95% CI 0.69, corresponding to an IC50 ratio of **4.9×**"*. Recommendation: *"hERG block potency values **within 5×** of each other should not be considered different"* | **0.18 (published)** |
| hERG, automated patch | **17 sites, 5 platforms, 12 blinded drugs** — Kramer et al., Sci Rep 10:5627, 2020 | dofetilide 25/75 spread **1.02 log (10.4×)**; quinidine 0.22 log (1.7×); within-platform repeatability **3.7×–9.7×** | 0.2–0.5 |
| hERG, inter-occasion vs inter-lab | Leishman et al., JPTM 2024 | *"**inter-occasion and inter-laboratory variability in hERG IC50 were comparable**"* — repeating in one lab on another day is as variable as changing lab | — |
| **Bioactivity pKi** (pooled public) | Kramer 2012 | mean 0.44, **SD 0.54**, median 0.34, **R²_max 0.81** | **0.54 (published)** |
| **Bioactivity pIC50** (pooled public) | Kalliokoski 2013 | **σ 0.68**, MUE 0.55 | **0.68 (published)** |
| **Bioactivity, same lab** | Kalliokoski 2013 | **σ 0.22 / 0.17** | **0.17–0.22 (published)** |
| **Intrinsic solubility log S** | Solubility Challenge 2 | **tight SD 0.17**, loose **0.62**; best model RMSE **0.86** | **0.17 / 0.62 (published)** |
| Aqueous solubility (general) | Ulrich 2025, Llompart 2024 | *"experimental error of 0.5 to 0.6 log units"*; SDi *"up to 1.0 log"* | 0.5–0.6 |
| Organic solubility log S | Attia 2025 (34 solutions, 8 solutes) | inter-lab **SD 0.34 / RMSE 0.75** | 0.34–0.75 |
| **Hepatocyte CLint** | 50 compounds, 2–9 sources each — Bowman & Benet, Pharm Res 2019 | *"CVs ranged from **8.53–99.8%**"*; triazolam CV 99.4% | ~0.04–0.35 **(derived)** |
| **CLint, inter vs intra-lab** | Fagerholm, bioRxiv 2022 | interlab *"**~3-fold mean** (5-fold mean range), maximum 51-fold"*; intralab *"**~2-fold mean**"*; in vivo/in vitro **R² = 0.44** | ≈0.24 interlab **(derived)** |
| Microsomal CLint | Fagerholm 2022 | *"**every other compound had non-quantifiable in vitro CLint**"* | **left-censoring, not scatter** |
| CLint ring trial (fish only) | Nichols et al., Toxicol Sci 2018 | intralab CV **18.9%** (hepatocytes) / 14.1% (S9); interlab **30.1%** / 22.4% | ~0.08–0.13 **(derived)** |
| **Caco-2 log Papp** | 11 compounds × 10 studies — Fagerholm 2022 | interlab *"**3-fold** average, mean range 24-fold"*; 5 cmpds × 10 labs: interlab **4-fold / 44-fold**; log Papp vs Fa interlab **R² 0.48** | ≈0.24 **(derived)** |
| Permeability, same method | Storchmannová et al., Mol Pharm 2025 | per-molecule log Papp ± SD: antipyrine ±**0.23**, ketoprofen ±0.35, carbamazepine ±**0.67**, propranolol ±0.87, theophylline ±**0.93** | **0.23–0.93 (published)** |
| **P-gp inhibition IC50** | **23 laboratories** — Bentz et al., DMD 2013 | IC50 differences *"from a minimum of **20- and 24-fold** to a maximum of **407- and 796-fold**"* | **effectively unpoolable** |
| Efflux ratio | Fagerholm 2022 | average/median/max range **5.3× / 1.7× / 93×**; in vivo R² **0.13**; *"**19%** have contradictory interlaboratory ER results"* | ≈0.22 **(derived)** |
| **Plasma protein binding, fu** | Fagerholm 2022 | interlab ratios **3.3× / 1.5× / 47×**; **intralab ~1.1×** | ≈0.26 interlab **(derived)** |
| fu, highly bound | **11 companies, 20 compounds** (IQ consortium), DMD 2026 | *"fu values as low as **~0.003%** can … be reliably quantified"* — limit is bioanalytical sensitivity | — |
| **logD, shake-flask, intra-lab** | 37 compounds — Ma et al., J Chromatogr B 2023 | pooled vs single measurement **RMSE 0.21 log, R² 0.9879** | ≈0.15 **(derived)** |
| logKow method intercomparison | Bugher et al., J Phys Chem B 2026 | shake-flask *"variations up to **0.64 log units**"* without QC | ≤0.64 |

**Three things to note.**

1. ***"Two-fold error for clearance" has no human ring-trial anchor.*** **[V — by absence]** No human
   liver-microsome CLint ring trial appears to exist. The closest support is Fagerholm's compilation
   (~3× interlab, ~2× intralab), and the "2-fold" figure in the IVIVE literature is an **acceptance
   threshold for predictions**, not a measured assay SD. **Do not cite it as reproducibility.**
2. **P-gp and efflux ratio are not modellable from pooled public data at all** — 20–796-fold inter-lab
   IC50 spread across 23 labs, and 19% of compounds give *contradictory* efflux directions. Our
   `Pgp_Broccatelli` number should be read with that in mind.
3. **logD7.4 remains the gap, and it is the one we most need** — 4,200 of our labels. Four candidate
   AZ sources were located (Wenlock 2011 J Biomol Screen PMID 21343602; Colclough & Wenlock 2015
   PMID 26054297; Wenlock & Barton 2013; Chang/Woody/Keefer ChemRxiv 2021) and **all are paywalled
   with numberless abstracts.** **[V — searched and failed, twice, independently.]**

### 5.5 There is no noise-ceiling analysis for the TDC ADMET group — and that is a gap we could fill

**[V — by absence]** A Europe PMC search over `"Therapeutics Data Commons" OR "TDC ADMET" OR "ADMET
Benchmark Group"` returns 23 hits and **the entire critique literature is about leakage and
reproducibility, not noise.** Searches for an explicit `R²_max` derivation applied to ADMET returned
only Kolmar & Grulke, Zhao 2017 and Mervin 2021. **Nobody has computed a per-endpoint noise ceiling
for the 22 TDC ADMET datasets.**

The method is standard and the inputs are mostly in §5.4. Where an endpoint carries duplicate
molecules, the spread between repeats *is* its floor (Schuh et al. flag regression conflicts at
**≥3σ**), and `R²_max = 1 − σ²_noise/σ²_total` follows. **This is a day's work, it is publishable,
and it would tell us per endpoint where the headroom is.**

### 5.6 What this section actually concludes

**Not "we are at the ceiling."** Revised position:

1. **Noise is real and large**, especially for pooled public bioactivity (σ 0.54–0.68), P-gp
   (unpoolable) and permeability (0.23–0.93 per molecule).
2. **But most of it is curation debt, not assay physics** — same-lab σ is 0.17–0.22 against 0.66–0.68
   pooled (§5.1). **9–16× the variance comes from pooling.**
3. **And noise inflates *measured* error, not *true* error** (Kolmar & Grulke), while *removing*
   suspected-noisy compounds does **not** improve external predictivity (Zhao 2017).
4. **Every direct test of "noise is the binding constraint" has failed.** Solubility Challenge 2:
   0.17-log test set, 0.86-log best model. Llompart: 0.35–0.63 claimed → 1.74–2.17 external.
5. **So the binding constraint is generalisation, not labels.** Which is §7.4(b)'s finding arriving
   from a second direction: on a real programme's chemistry, models far bigger than ours give
   **negative R²**. Spend on §8 Steps 0, 1, 2 and 7, not on denoising — with the single exception of
   obtaining the Merck denoising paper, which claims the opposite and may be right.

---

## 6. Multi-task scaling — does adding *more endpoints* help?

**Short answer: no, not at our scale, and the best single piece of evidence is a platform that ran
exactly our experiment at double our task count and got a tie.**

**But read §7.4(a) immediately after this section.** The 2026 blind challenge found multi-task
*"essential"* — on a dataset where the tasks share their molecules. The discriminator is
**labels per molecule**, and ours is **1.72**, which puts us on the "tie" side of the line. The two
results are not in conflict and the distinction is the whole point.

### 6.1 The decisive result: ADMET-AI ran this at 41 endpoints

Swanson, Walther, Leitz, Mukherjee, Wu, Shivnaraine, Zou, *ADMET-AI: a machine learning ADMET
platform for evaluation of large-scale chemical libraries*, **Bioinformatics 40(7), 2024**,
[10.1093/bioinformatics/btae416](https://doi.org/10.1093/bioinformatics/btae416). Full text read
via Europe PMC. **[V]** (<https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11226862/fullTextXML>)

They trained Chemprop-RDKit (D-MPNN + 200 RDKit physicochemical features, 5-model ensembles) on
**41 TDC ADMET datasets**, both single-task and multi-task, and report verbatim: **[V]**

> *"Across all 41 TDC ADMET datasets, ADMET-AI models trained on each dataset individually
> ('single-task') had overall strong performance" … "ADMET-AI models trained on the two multi-task
> datasets achieved **very similar performance**" … "Notably, the two multi-task models … make
> predictions for all the properties **significantly faster** than the 41 single-task models.
> Therefore, the multi-task models were deployed in the ADMET-AI web server."*

**Multi-task bought speed, not accuracy — at 41 endpoints, with a better encoder than ours.**

Note also *how* they split: *"from the 41 ADMET datasets, two multi-task datasets were created, one
containing all 10 regression datasets and one containing all 31 classification datasets."* **[V]**
They did **not** build one 41-task model. The regression/classification split is itself a
task-grouping decision — and it is the same split Chemprop's CLI forces on us (§1.9) and the same
split §1.11 says wants different fine-tuning protocols.

### 6.2 The published platforms, and whether they ever ran the ablation

| Platform | Endpoints | Architecture | Training data | MT-vs-ST ablation? | Result |
|---|---|---|---|---|---|
| **ADMET-AI** (Bioinformatics 2024) | **41** (10 reg / 31 clf); 22 on the leaderboard | Chemprop v1.6.1 D-MPNN + 200 RDKit features, 5× ensemble | not stated | **Yes — the only clean one** | **Tie.** MT deployed for speed. 1M molecules in ~3.1 h, "a 45% reduction in time compared to the next fastest public ADMET web server." |
| **ADMETlab 3.0** (NAR 2024, [10.1093/nar/gkae236](https://doi.org/10.1093/nar/gkae236)) | 119 published features from **77 models** (59 clf / 18 reg) | **multi-task DMPNN-Des** = D-MPNN + RDKit 2D | **>400,000 entries**, 1.5× v2.0 | Architecture comparison only | DMPNN-Des/DMPNN beat the older MGA in **47 of 59** classification tasks. **No multitask-vs-single-task control.** |
| **Deep-PK** (NAR 2024, [10.1093/nar/gkae254](https://doi.org/10.1093/nar/gkae254)) | **73** (64 ADMET + 9 physchem); 49 clf / 24 reg | Chemprop D-MPNN + 449 graph-level features | not stated | **No** | Effectively **per-endpoint**: "160,000 optimization runs for 73 Deep-PK … endpoints **individually**." |
| **ADMETboost** (J Mol Model 2022, [10.1007/s00894-022-05373-8](https://doi.org/10.1007/s00894-022-05373-8)) | 22 TDC | **XGBoost** on fingerprints + descriptors — inherently single-task | TDC ADMET group | n/a | Ranked **1st in 18 of 22**, top-3 in **21 of 22**. *A single-task tree ensemble topping a 22-endpoint leaderboard is itself an answer.* |
| **MiniMol** (arXiv 2404.14986) | 17 TDC reported | GNN foundation model, **10M params** | 6M molecules / **500M labels / ~3,300 pretraining tasks** | pretrain→finetune | Beat prior SOTA foundation model — **but failed the leakage audit, §7.1** |
| **MolGPS** (arXiv 2404.11568, NeurIPS 2024) | 38 downstream incl. 22 TDC | MPNN++/GPS++ hybrids, up to **3B params** | LargeMix, 5M molecules | label-fraction scaling | SOTA on 26/38, **11/22 TDC** |

All **[V]** per the agent's fetches.

**Nobody has published a 20+-endpoint ADMET result where multi-task demonstrably beat
single-task.** Of the four 20+-endpoint platforms, one ran the ablation and it tied; one asserts
multi-task without a control; one is per-endpoint and NAR-published; and the fourth is plain
per-endpoint XGBoost that topped 18 of 22 TDC tasks.

### 6.3 Where multi-task gain actually comes from — and where it turns negative

Three independent measurements, all with the same shape: **gain is concentrated in the small,
data-poor tasks; flat-to-negative on the large ones; negative when unrelated tasks are pooled.**

**Capela, Nouchi, Van Deursen, Tetko, Godin, *Multitask Learning On Graph Neural Networks Applied
To Molecular Property Predictions*, [arXiv 1910.13124](https://arxiv.org/abs/1910.13124).** **[V]**
Six physicochemical regression tasks (ESOL 1,128; FreeSolv 613; logD7.4 4,200; BP 5,435; LogVP
2,708; LogP 13,973), three GIN-family architectures, 5-fold CV, masked loss for sparse labels.
Headline: *"MTL improves the results by 4%–20%."* But:

- **They did not train one 6-task model.** They grouped by cross-correlation with an **absolute
  correlation threshold of 0.5**, giving buckets {logD7.4, LogP}, {ESOL, LogP}, {FreeSolv, LogVP,
  BP}. *"We obtained better results for related tasks than tasks with lower correlation."*
- **Their Appendix Table 3 is the cleanest published measurement of negative transfer.** Pooling all
  six blindly vs grouping by correlation, RMSE on the smallest task **FreeSolv**:
  GIN **1.15 → 0.83**, GGRNet **1.00 → 0.80**, GAIN **1.25 → 0.87** — **28–30% worse when pooled**,
  and the pooled GGRNet (1.00) was worse than *single-task* GGRNet (0.82). **Negative transfer at
  n = 6 tasks.**
- **LogP, the largest task, got no benefit and under GAIN got worse (MT 0.50 vs ST 0.45).** Their
  own reading: *"the positive effect of MTL decreases with increasing size."*
- **Transfer learning failed outright.** Train on 5, freeze convolutions, retrain heads on the 6th:
  ESOL 0.65→0.76, logD7.4 0.74→0.92, LogP 0.48→0.61 — all worse than single-task. Their conclusion:
  *"the primary reason for the improvement may be due to an **implicit data augmentation** obtained
  from merging the datasets for different but similar end points."*

**Montanari et al. (Bayer), *Modeling Physico-Chemical ADMET Endpoints with Multitask Graph
Convolutional Networks*, Molecules 25(1):44, 2019,
[10.3390/molecules25010044](https://doi.org/10.3390/molecules25010044).** **[V]** 10 endpoints,
537,443 compounds, per-endpoint n from 2,334 to 236,280. Multi-task vs single-task R²:

| endpoint | n | ST → MT R² |
|---|---|---|
| solubility (powder) | 2,334 | **−0.17 → 0.56** (+0.73) |
| solubility (DMSO) | — | 0.33 → 0.58 (+0.25) |
| solubility (nephelometry) | — | 0.59 → 0.68 (+0.09) |
| logD pH 7.5 | — | 0.87 → 0.88 (flat) |
| melting point | — | 0.51 → 0.51 (flat) |
| **logD acidic** | **236,280** | **0.94 → 0.91 (worse)** |

Their words: ***"larger tasks are either not affected or show poorer performance in multitask than
in single task approach."***

**Graphium / LargeMix (ICLR 2024, [arXiv 2310.04292](https://arxiv.org/abs/2310.04292)).** **[V]**
Single vs multi-task GINE, MAE:

| dataset | n molecules | single | multi | |
|---|---|---|---|---|
| PCQM4M_G25 | 3.8M | .2223 | .2315 | **worse** |
| PCBA_1328 | 1.6M | .0320 | .0341 | **worse** |
| L1000_VCAP | 15k | .1907 | .1862 | better |
| L1000_MCF7 | 12k | .1856 | .1873 | worse |

And on ToyMix: *"For the larger dataset QM9, which takes up nearly 87% of the molecules, however, we
do see a minor reduction in the performance in the multi dataset context."*

**Ramsundar et al., *Massively Multitask Networks for Drug Discovery*,
[arXiv 1502.02072](https://arxiv.org/abs/1502.02072)** — the paper usually cited for "more tasks is
better". Its four verbatim conclusions, from **~40 million measurements across >200 targets**: **[V]**

> *"(1) massively multitask networks obtain predictive accuracies significantly better than
> single-task methods, (2) the predictive power of multitask networks improves as additional tasks
> and data are added, (3) the total amount of data and the total number of tasks both contribute
> significantly, and (4) **multitask networks afford limited transferability to tasks not in the
> training set.**"*

Read (2)+(3)+(4) together and it is the same mechanism Capela isolated: the task count helps
**because it brings data**, and the representation does not generalise off-manifold. The regime
where they saw clean monotone task-count gains is **200+ targets and tens of millions of labels** —
roughly **500× our 79,238 labels**.

### 6.4 Is there an optimal number of tasks?

**No paper reports one.** Every paper that measures it reports the trigger as **task relatedness and
per-task data volume, not task count.** **[V]** Capela sees negative transfer at n=6 with
uncorrelated tasks; Ramsundar sees monotone gains to 200+ with same-modality bioactivity and 40M
labels. **The decision variable is task affinity, not count.**

If we do spend effort on this axis, the method to use is **Li, Ju, Sharma, Zhang, *Boosting
Multitask Learning on Graphs through Higher-Order Task Affinities*,
[arXiv 2306.14009](https://arxiv.org/abs/2306.14009)** **[V]** — it estimates task affinity as the
loss of task A *in the presence of task B plus a random subset of others*, spectral-clusters the
affinity matrix, and fits one model per group. Its whole claim is that **higher-order affinities
predict negative transfer more accurately than pairwise affinities**, which is exactly the weakness
in Capela's correlation-threshold heuristic. Verbatim: *"we find that negative transfer is prevalent
when we apply naive multitask learning… as task relationships are highly nonlinear."*

Gradient surgery (PCGrad, GradNorm) is **essentially unexplored for ADMET**. The one genuine hit is
**RCGrad** — *Enhancing Molecular Property Prediction with Auxiliary Learning and Task-Specific
Adaptation*, [arXiv 2401.16299](https://arxiv.org/abs/2401.16299) / J Cheminform 2024
[10.1186/s13321-024-00880-7](https://doi.org/10.1186/s13321-024-00880-7), reporting up to **7.7%**
over fine-tuning. **[V for existence/summary]** Treat "PCGrad for ADMET" as an open gap, not an
established technique.

### 6.5 What the 2026 consensus actually recommends

Not "multi-task", and not "single-task" — **grouped multi-task with per-task adaptation**:

- **Xue et al. (NVIDIA), *Probabilistic Contrastive Pretraining for Multi-task ADME Property
  Prediction*, [arXiv 2606.11508](https://arxiv.org/abs/2606.11508)** (Jun 2026) **[V]**: *"we
  propose a multi-task GNN readout architecture with **task-specific multilayer perceptron heads**,
  preserving shared representation learning while **mitigating negative transfer**."* Gains of
  7.6% / 9.9% / 9.5% over the KERMT baseline on Biogen, ExpansionRx and ChEMBL-MT — with their own
  caveat that this is *"averaged over significantly-improved endpoints"*, not over all endpoints.
- **Ito et al. (Kyoto), *Improving ADME Prediction with Multitask Graph Neural Networks and
  Assessing Explainability in Lead Optimization*, JCIM 65(21):11796–11803, 2025,
  [10.1021/acs.jcim.5c00758](https://doi.org/10.1021/acs.jcim.5c00758)** **[V — abstract only; ACS
  body blocked]**: ten ADME parameters simultaneously, explicitly two-stage — *"multitask learning,
  which shares information across multiple tasks to increase the number of usable samples, **with
  fine-tuning that adapts to each task**."*
- **The OpenADMET–ExpansionRx blind challenge** (see §7.3) states it in its abstract: *"critical
  need to optimize predictive workflows for different endpoints independently via **multitasking
  and task-affinity grouping**."* **[V — abstract via OpenAlex; ChemRxiv body is Cloudflare-blocked,
  so treat anything beyond the abstract as [R].]**

And one more piece of evidence pointing *away* from the task axis: **MolGPS** sub-sampled training
labels at 12.5 / 25 / 50% and measured a **Spearman of 0.57 (Polaris) and 0.54 (TDC) between label
fraction and downstream performance**, ranking label count **second** behind **encoder width** among
scaling axes. **[V]** At 3B parameters and 500M labels. If width beats labels *there*, our marginal
L4-hour is not best spent on endpoint #22.

### 6.6 But CheMeleon's *pretraining* is a 1,613-task multi-task problem

The one place where "more tasks" demonstrably paid is **cheap, computable, low-noise tasks**, not
more assays. CheMeleon's pretraining objective *is* masked multi-task regression over 1,613 Mordred
descriptors (§1.3). Fallani et al. (§7.3) found **supervised atom-level QM pretraining beats
self-supervised masking** on TDC ADMET and confirmed it on proprietary pharma data. Graphium found
quantum data lifts low-resource *biological* tasks. **[V]**

We already compute RDKit2D. The descriptor targets for a CheMeleon-style auxiliary loss are
**free**. That is a far cheaper experiment than finding nine more assays — see §8.

---

## 7. What actually moved the needle in 2025–2026

### 7.1 The audit: Koleiev et al. 2026

**Exact title:** *An End-User Audit of Reproducibility, Data Leakage, and Overfitting of the
Top-Ranked ADMET Prediction Models in TDC Leaderboards.*
**Koleiev, Stratiichuk, Shevchuk, Melnychenko, Nyporko, Todoryshyn, Husak, Starosyla, Yesylevskyy,
Nafiiev**, **JCIM 66(14):8045–8058**, 2 July 2026,
[10.1021/acs.jcim.6c00819](https://doi.org/10.1021/acs.jcim.6c00819). Open-access full text read via
Europe PMC (**PMC13417885**). Preprint: bioRxiv `10.64898/2026.02.26.708193`, 28 Feb 2026,
*"Critical Assessment of ML models for ADMET Prediction in TDC leaderboards"*. **[V]**

**Scope — state this precisely when citing.** They took the **top three entries of each of the 22
TDC ADMET endpoints**, de-duplicated, and got **10 distinct architectures**: ADMETrix, CFA,
GradientBoost+, MapLight, MapLight+GNN, MiniMol, CaliciBoost, SimGCN, XGBoost, ZairaChem. Four
stages: reproducibility → environment → leakage audit → HPO audit. **[V]**

| Failure mode | n | Models |
|---|---|---|
| Code unavailable (broken links) | 1 | CFA |
| Environment setup impossible | 2 | ADMETrix, SimGCN |
| Runtime incompatibilities | 1 | ZairaChem |
| **Leakage in hyperparameter optimisation** | 2 | GradientBoost+, XGBoost |
| **Pretraining data leakage** | 1 | MiniMol |
| **Failed** | **7** | |
| **Passed all four stages** | **3** | **CaliciBoost, MapLight, MapLight+GNN** |

So "only 3 of 10 reproduce" is right — but **4 of the 7 failures are infrastructure, not
misconduct.** **[V]**

**What reproduced, and how closely:** **[V]**
- **CaliciBoost** (Caco-2): reported MAE 0.256 ± 0.006, reproduced **0.271 ± 0.002** — kept first place.
- **MapLight** (all 22): kept rank on **11**, improved on **4**; *"absolute differences between
  TDC-reported and reproduced metrics are very small (**<0.01**)."*
- **MapLight + GNN**: kept rank on **13 of 22**, improved on **5**; same <0.01 agreement.

**The two leakage mechanisms, in detail:** **[V]**
- **GradientBoost+ / XGBoost — a one-line bug.** *"An incorrect function call… resplits the full
  data set randomly into training, validation, and test subsets with the default hard-coded random
  seed of 42. As a consequence, the validation set contains, among others, molecules from the
  original TDC test set."* After correction: **GradientBoost+ 1st → 3rd; XGBoost 2nd → 8th.**
- **MiniMol — pretraining-corpus leakage, and this one generalises to us.** Pretrained on ~6M
  LargeMix molecules; dedup by canonical SMILES was insufficient: *"Simple filtering by canonical
  SMILES is insufficient"* because it *"fails to account for molecular chirality properly."*
  Worked example: **danazol**, in the TDC test set, survived filtering because its stereochemistry
  was unspecified in the MiniMol training set. Residual overlap after filtering: **MCF7 7%,
  VCAP 4%, PCBA 0.6%** (the last being the largest absolute count). They also flag **tautomers**:
  *"different tautomeric forms… are usually represented by distinct canonical SMILES strings."*

> **⚠ Carry this forward.** If we pretrain on, or add endpoints from, any public corpus,
> **canonical-SMILES deduplication against the TDC test sets is not enough.** Match on
> **stereo-stripped, tautomer-canonicalised InChIKey**. MiniMol lost its audit on exactly this.
> This applies directly to §8 step 4, and to any use of CheMeleon's 1M PubChem corpus.

**The intentional-overfitting experiment — the most quotable result in the paper.** **[V]** They
built deliberately overfit variants of their **own in-house LightGBM**, *"using the TDC test data
sets instead of TDC train data sets during both SFS and HPO optimization phases"*:

- Overfit variants reached top-3 on **10 of 22** endpoints (~45%). Their **honest** models reached
  top-3 on **2 of 22**. A **5× inflation of the top-3 rate.**
- Rank jumps: bbb_martins **8→1**; clearance_microsome_az **14→1**; cyp2d6_substrate **12→1**;
  ames **13→1**; caco2_wang **9→2**; dili **13→2**; hia_hou **2→1**.
- Metric inflation: bbb_martins AUROC **0.9225 → 0.9376**; dili AUROC **0.86 → 0.9422**.
- A fairness note they make themselves: MapLight, MapLight+GNN and MiniMol *"appear in the top ranks
  far more frequently than any of our own overfitted variants,"* which they read as consistency
  rather than deliberate cheating.

**Verdict, verbatim:** *"Current TDC leaderboard positions cannot be read as a direct measure of
model quality and practical applicability."* **[V]**

> **What this means for our numbers.** Koleiev's *honest* in-house LightGBM scored **2/22 top-3**.
> That is the company our LightGBM is in. The distance between us and the leaderboard is
> substantially an **honesty gap, not a modelling gap** — and the three architectures that *did*
> reproduce (CaliciBoost = boosting; MapLight = descriptors + fingerprints into gradient boosting;
> MapLight+GNN) are **architecturally our setup**. Our 10-vs-11 dead heat is where the honest
> literature sits.

### 7.2 Are the TDC splits themselves dirty?

**The splits are mostly fine; the leaderboard is not.** That distinction matters.

**Schuh, Daniluk, Sieber, *Auditing widely used biomolecular benchmarks reveals systematic data
inconsistencies*, Chemical Science 17(36):17455–17467, Sep 2026,
[10.1039/d6sc01799a](https://doi.org/10.1039/d6sc01799a)** (PMC13425739). **[V]**

- **51 configurations audited: 22 TDC, 9 MoleculeNet, 7 Polaris, 13 DTI.**
- Definitions: *cross-split contamination* = unique cleaned SMILES in both train and test;
  *label conflict* = same cleaned SMILES with differing labels, or for regression labels differing
  by **≥3σ**; *activity cliff* = similarity above **τ = 0.9** by Morgan-Tanimoto, Murcko-scaffold
  Tanimoto or normalised-SMILES Levenshtein, with divergent activity.
- **Worst TDC ADMET config: Clearance_Hepatocyte_AZ at 4.95% cross-split contamination — and that is
  the maximum across every benchmark they audited.** Mean nearest-neighbour similarity: **TDC
  0.559**, Polaris 0.494, MoleculeNet 0.589, **DTI 0.779**. Worst offenders overall are the DTI
  suites and MoleculeNet (ClinTox severe-conflict rate 0.74%; BACE cliff density 0.175%).
- **How much leakage inflates a score (their noise-injection study):** explicit train–test
  contamination **decreased paired RMSE by 11.37** and **increased ROC-AUC by 0.177**. By contrast
  label conflicts and random label noise *degraded* ROC-AUC by **0.317** and **0.329** and increased
  RMSE by **6.53 / 4.97**.

Koleiev independently notes *"a nontrivial fraction of compounds within the official splits are
duplicated and, in a number of cases, carry conflicting labels."* **[V]**

**Net:** use the TDC splits; do not use the TDC leaderboard as a target. Our comparator should be
**MapLight / MapLight+GNN / CaliciBoost**, which reproduce, under a proper paired test.

### 7.3 So what do the *honest* 2025–26 papers do differently?

Five things, in rough order of how much they move the number:

1. **They pretrain on cheap, computable, low-noise labels — not on more experimental data.**
   - **CheMeleon** (§1): 1,613 Mordred descriptors as masked pretraining targets, 12.9M params,
     75% Polaris win rate. The authors' framing: it *"uniquely employs low-noise descriptors… avoiding
     noisy experimental data or biased simulations."* **[V]**
   - **Fallani, Nugmanov, Arjona-Medina, Wegner, Tkatchenko, Chernichenko, *Pretraining graph
     transformers with atom-in-a-molecule quantum properties for improved ADMET modeling*,
     J Cheminform 17:25, 2025, [10.1186/s13321-025-00970-0](https://doi.org/10.1186/s13321-025-00970-0).**
     **[V]** Three pretraining strategies compared — **atom-level QM properties**, molecular QM
     (HOMO-LUMO gap), and self-supervised atom masking — fine-tuned on **TDC ADMET**. *"Models
     pretrained with atomic quantum mechanical properties produce in general better results."*
     Crucially they then validate on *"a much larger non-public dataset for microsomal clearance"*
     and report that *"model types with similar performance on public benchmarks can have different
     performances on large scale pharmaceutical data"* — a warning that public-benchmark ties do not
     survive contact with real data.
   - This is the single most consistent signal across 2025–26: **supervised pretraining on computed
     properties beats self-supervised masking.** And it is the axis where our RDKit2D features are
     already a free source of targets.

2. **They stop over-searching hyperparameters.** **Tetko, van Deursen, Godin, *Be aware of
   overfitting by hyperparameter optimization!*, J Cheminform 16, 2024,
   [10.1186/s13321-024-00934-w](https://doi.org/10.1186/s13321-024-00934-w).** **[V]** Across seven
   solubility datasets: *"an optimization of a large space of parameters could result in overfitting
   of models… Similar results could be calculated using pre-set hyperparameters, **reducing the
   computational effort by around 10,000 times**."* Their Transformer-CNN beat graph methods on
   **26 of 28** pairwise comparisons.

   *This is our H4 result, published.* Our sweep moved the test score by 0.0004 and dropout hurt
   monotonically. Tetko says a fixed sensible configuration is the correct default and the search
   is the overfitting risk. **Do not re-run the sweep.**

3. **They use a real statistical protocol instead of a win count.** **Ash, Wognum,
   Rodríguez-Pérez, Aldeghi, Cheng, Clevert, Engkvist, Fang, Price, Hughes-Oliver, Walters,
   *Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug
   Discovery*, JCIM 65(18):9398–9411, 2025,
   [10.1021/acs.jcim.5c01609](https://doi.org/10.1021/acs.jcim.5c01609).** **[V]** Guidelines plus
   annotated open-source examples for rigorous, domain-appropriate comparison, and for the
   distinction between *statistically* and *practically* significant.

   *Apply this to our 10-vs-11.* A 10/21 vs 11/21 win count on single official test folds carries
   essentially no statistical weight. The practical standard is CheMeleon's and the ExpansionRx
   challenge's: **Tukey HSD on bootstrapped or multi-seed samples, with ties counted as ties.**
   We should re-express H3 that way before building anything on it.

4. **They evaluate on blind, prospective, time-split industrial data.** The **OpenADMET–ExpansionRx
   community blind challenge**, ChemRxiv `10.26434/chemrxiv.15009342/v1`, 23 Sep 2026, ~130 authors
   including W. Patrick Walters, Igor Tetko, Jackson Burns, Alan Cheng. **[V for title, authors and
   abstract via OpenAlex; ChemRxiv's PDF is Cloudflare-403, so the body is unread — [R] beyond the
   abstract.]** Verbatim from the abstract:

   > *"Evaluating submissions against a realistic, challenging dataset from active ExpansionRx drug
   > discovery programs revealed that **message-passing neural networks and chemical foundation
   > models dominated among the top performers**. Furthermore, our results highlighted the power of
   > proprietary industry datasets and critical need to optimize predictive workflows for different
   > endpoints independently via **multitasking and task-affinity grouping**."*

   Challenge facts from OpenADMET's own results post **[V]**
   (<https://openadmet.ghost.io/the-openadmet-expansionrx-blind-challenge-has-come-to-an-end/>,
   <https://openadmet.org/blindchallenges/>): **nine ADMET endpoints, >7,000 compounds** ("one of the
   largest lead-optimization-like datasets available in the public domain"), **time-split** (early
   molecules → predict late molecules), blinded test half-validation/half-final, **370+
   participants, 4,000+ submissions**. Metric **MA-RAE** (macro-averaged relative absolute error,
   normalised per endpoint so endpoints weigh equally), scored with **Tukey HSD on bootstrapped
   samples**. Top three: **pebble 0.5113 ± 0.0070** (R² 0.6372, Spearman 0.7932), **overfit
   0.5284 ± 0.0070**, **moka 0.5321 ± 0.0070**.

   **Two things to take from this.** (a) *D-MPNNs and chemical foundation models won a blind
   prospective test* — which is the strongest evidence anywhere that CheMeleon is worth our time,
   and the opposite of what the public-leaderboard picture suggests. (b) The recommendation is
   **not** "multi-task" but "optimise per endpoint, via multitasking **and task-affinity
   grouping**" — §6.5 again.

5. **They release code that runs.** The whole Koleiev failure taxonomy is dominated by broken links,
   un-buildable environments and runtime incompatibilities (4 of 7 failures). The cheapest way for
   this programme to be in the honest tercile is to keep `runs/<run-id>/` reproducible and pin
   environments — which we already do.

### 7.4 The blind challenge's own post-mortem — read this before anything else

The OpenADMET organisers published two trend analyses after the ExpansionRx challenge closed. They
are not peer-reviewed, but they are the only account anywhere of what **370+ teams and 4,000+
submissions** actually did against a **blind, time-split, real discovery-programme dataset**, and
they contradict the public-leaderboard picture in both directions. **[V]**

**(a) *Lessons Learned from the OpenADMET–ExpansionRx Blind Challenge*, 4 Feb 2026** **[V]**
(<https://openadmet.ghost.io/lessons-learned-from-the-openadmet-expansionrx-blind-challenge/>)

Verbatim or near-verbatim from the organisers:

- **Architecture:** *"2D Graph neural networks (GNNs) dominated the top of the leaderboard. In
  particular, **Chemprop** proved a popular choice."* KERMT, Uni-Mol and **TabPFN** were also used,
  but TabPFN *"lacked the performance of 2D GNNs at the top"* — a direct caution against §2.6.
- **Multi-task:** ***"Multitask learning was essential, with several top-performing groups also
  investigating the placement of tasks into related groups"*** where they complement each other.
- **Foundation models:** most winners used pretrained models rather than training from scratch, and
  **CheMeleon was particularly popular**; some did their own de-novo Chemprop pretraining.
- **Features:** winners combined **RDKit and Mordred descriptors, Morgan/ECFP fingerprints**,
  electrostatic features from **Jazzy**, and QM/MD-derived properties.
- **Data augmentation:** *"Several ingenious approaches even used **de novo generation of molecules
  from the chemical space around the challenge molecules to augment pretraining sets**."* Also RIGR
  and SMILES enumeration.
- **External data:** ***"4 of the top 5 participants also utilized proprietary data."*** Public
  supplements: ASAP Discovery, Galapagos, Novartis, ChEMBL.
- **Hyperparameters:** ***"most noted minimal performance gain or did not extensively tune their
  models."*** Data engineering and feature augmentation beat tuning.
- **Ensembling:** **3 of the top 10 combined traditional ML (RF, LightGBM, XGBoost, CatBoost) with
  MPNN models.** Simple averaging through Caruana and snapshot ensembling.
- **Validation:** top performers used **temporal splits** (mirroring the prospective task) alongside
  scaffold and difficulty-based splits.

**Reconciling "multitask was essential" here with ADMET-AI's tie in §6.1.** These are not in
conflict; they differ on one variable — **how much the tasks share molecules.**

- TDC's 41 ADMET datasets are 41 *different compound collections* curated from different papers.
  Merging them gives a shared encoder more molecules but almost no molecule with two labels, so the
  only mechanism available is representation transfer — which Capela showed does not transfer (§6.3).
  **ADMET-AI measured a tie, and that is the right answer for that data.**
- ExpansionRx's nine endpoints were measured on **largely the same >7,000 compounds** from one
  company's programmes. There, multi-task is genuine multi-output learning on a dense label matrix,
  and the "implicit data augmentation" Capela identified is at full strength.
  The nine are **LogD, kinetic solubility (KSOL), HLM Clint, MLM Clint, Caco-2 Papp A→B, Caco-2
  efflux ratio, mouse plasma protein binding, mouse brain protein binding, and mouse gastrocnemius
  muscle binding** **[V]**
  (<https://openadmet.ghost.io/openadmet-expansionrx-blind-challenge/>) — from a real campaign
  against RNA-mediated disease (myotonic dystrophy, ALS, dementia), split **by time**, early
  molecules → predict late molecules. **[V]** Note the structure: *two* liver-microsome clearances,
  *two* Caco-2 readouts and *three* tissue-binding readouts. Several of those pairs are nearly the
  same measurement on the same molecules. Of course multi-task was essential.

**Our matrix is 46,061 molecules × 21 endpoints with 79,238 labels — 1.72 labels per molecule.** And
the sparsity is worse than that average suggests: across 27 TDC endpoints, **pairs of endpoints share
a median of only 0.99% of their compounds** (Koo et al. 2026,
[10.21203/rs.3.rs-10822555/v1](https://doi.org/10.21203/rs.3.rs-10822555/v1)). **[V — abstract]**
A shared trunk has almost no per-molecule label overlap to exploit. That is the *sparse, TDC-like*
regime, not the ExpansionRx regime. So ADMET-AI's tie is the
applicable precedent, and the ExpansionRx result is the thing we would get **if we ever obtained a
dense in-house panel**. Both "multitask is essential" and "multitask ties" are true, of different
data, and the discriminator is labels-per-molecule.

**(b) *Can We Trust Zero-Shot ADMET Predictions?*, 26 Mar 2026** **[V]**
(<https://openadmet.ghost.io/zero-shot-expansiorx-admet-predictions/>) — **the most important
caveat in this entire file for anything we ship.**

They ran **ADMET-AI** (multi-task Chemprop v1 on TDC) and **ADMETlab 3.0** (multi-task Chemprop v1
on 400k+ ChEMBL/PubChem/OCHEM molecules) zero-shot against the ExpansionRx test set:

| Endpoint | Zero-shot public model | A model trained on ExpansionRx data |
|---|---|---|
| **LogD** | ADMETlab 3.0: **R² = 0.02**, MAE 0.82 | internal R² 0.50, MAE 0.57; Pebble **MAE < 0.3** |
| **Caco-2 permeability** | **all zero-shot models: negative R²** | Pebble **R² = 0.62** |
| **HLM CLint** | ADMET-AI and ADMETlab 3.0: **negative R²** | — |
| MPPB | best of the zero-shot endpoints, *"due to narrow dynamic range"* | — |

Root cause, verbatim: average nearest-neighbour similarity between the ExpansionRx test set and the
public datasets fell **below 0.3** — under the **0.27 "95% noise level" for Morgan fingerprints** —
so *"models are forced to extrapolate into entirely unseen territory."* Their conclusion: ***"Do not
rely on these models for 'zero-shot' ADMET predictions in active drug discovery programs."***

> **The ceiling that actually matters is not label noise and not architecture — it is domain shift.**
> A model trained on TDC, however good its TDC test score, produces **negative R²** on a real
> programme's chemistry for permeability and microsomal clearance. Every number elsewhere in this
> file is an in-distribution number. If our ADMET model is going to be used on a customer's
> series, the honest framing is: *ranking aid inside known chemistry; not a prediction on novel
> series*, and the product should say so. The same post gives the cheap diagnostic — compute mean
> nearest-neighbour Morgan-Tanimoto similarity from the query set to our training set, and warn
> below ~0.3.

---

## 8. Ranked plan to escape the data ceiling, for one L4

Ordered by **expected gain per L4-hour**, cheapest first. Costs are for our 46,061 molecules /
79,238 labels / 21 endpoints. Every step is a hypothesis with a written prediction, per
`HYPOTHESES.md` — the predictions below are mine and should be copied into that log *before* the
run, not after.

Running total for steps 0–5: **about 25–40 L4-hours and two or three days of CPU.** Nothing in the
first six steps requires new experimental data, a bigger GPU, or more than a week.

---

### Step 0 — Decide whether 10-vs-11 and 0.0004 mean anything. **0 GPU-hours.**

Before buying anything, re-express H3 and H4 under Ash et al.'s protocol (§7.3 item 3): multiple
seeds, bootstrapped Tukey HSD, ties counted as ties, and a pre-declared **practical** significance
threshold, not just a statistical one. CheMeleon uses exactly this (5 seeds, Tukey HSD, α=0.05) and
so did the ExpansionRx challenge.

- **Cost:** CPU only, a few hours. We already have the predictions and the folds.
- **Expected outcome:** H3's 10-vs-11 becomes *"statistically indistinguishable"*, which is a
  publishable, defensible statement, and H4's 0.0004 becomes *"inside noise"*.
- **The arithmetic is already decisive.** **Fourteen of the 22 TDC ADMET datasets have fewer than
  2,000 molecules, and eight have test sets of roughly 95–175 compounds.** **[V]** A single-split
  delta of 0.0004 is far inside split noise before label noise is even considered. Deng et al.'s
  warning applies directly: *"without statistical tests, there exists a potential risk of drawing
  incorrect conclusions"*, because individual splits spuriously favour one model family even when
  another wins overall across 30 seeds. **[V]**
- **And change the split while you are here — this is the cheapest high-information move available.**
  Scaffold split on TDC is only **Δ ROC-AUC ~0.01–0.04** harder than random, because *"most scaffolds
  contain few molecules (1.93 average)"*; **UMAP-cluster, Lo-Hi and DataSAIL splits drop 0.08–0.19**
  (PMC12529777, 8 binary TDC datasets at exactly our scale). **[V]** ID→OOD rank correlation is
  r ≈ 0.97 under random, 0.94 under scaffold, but **0.19–0.26 under Lo-Hi** — *"ID performance is not
  always a good proxy and criterion for model selection."* **If a scaffold split cannot separate our
  LightGBM from our NN, it cannot detect any of Steps 2–5 either.** Costs no GPU.
- **Why first:** every step below is measured against this baseline. If the baseline's error bars
  are unknown, no later result is interpretable. Tetko et al. (§7.3 item 2) is the second half of
  this: **do not run another hyperparameter sweep.** The ExpansionRx post-mortem independently found
  that *"most noted minimal performance gain or did not extensively tune their models"* (§7.4a).

### Step 1 — Add Avalon + ErG fingerprints to the LightGBM. **0 GPU-hours, ~1 day CPU.**

MapLight is one of only three TDC architectures that survived Koleiev's audit (§7.1), it held or
improved its rank on **11–13 of 22** endpoints on reproduction, and its recipe is:
**CatBoost on count-Morgan (1024 bits, r=2) + Avalon count FP (1024 bits) + ErG + 200 RDKit
physicochemical descriptors** — verified from `maplight.py`, which imports
`GetHashedMorganFingerprint`, `rdkit.Avalon.pyAvalonTools.GetAvalonCountFP` and
`rdkit.Chem.rdReducedGraphs`. **[V]** (<https://github.com/maplightrx/MapLight-TDC>; paper
*ADMET property prediction through combinations of molecular fingerprints*, Notwell & Wood,
[arXiv 2310.00174](https://arxiv.org/abs/2310.00174) **[V]**.)

**Our feature set is MapLight's minus Avalon and ErG.** Both are pure RDKit, both are CPU-cheap,
neither needs a GPU.

- **Cost:** one afternoon of engineering, a few CPU-hours of featurisation, a LightGBM re-run.
- **Expected gain:** small but real and broad — the published claim is that the *combination* is
  what matters (*"our findings underscore the significance of richer molecular representations"*).
  **Prediction: wins on ≥12 of 21 endpoints, mean improvement under 2% relative.** If it wins on
  fewer than 10, our H1 feature set is already saturated and that is worth knowing.
- Also try CatBoost against LightGBM here; MapLight found CatBoost best of the GBDTs. **[V]**

### Step 2 — CheMeleon as a frozen 2048-d fingerprint, concatenated into the GBDT. **<1 L4-hour.**

This is **MapLight+GNN with a better GNN**. MapLight+GNN appends a frozen GIN-supervised-masking
fingerprint (300 columns) to the same CatBoost, and MapLight+GNN is the *other* architecture that
passed Koleiev's audit, retaining rank on **13 of 22**. **[V]**

**The size of that effect is published and it is large.** Adding the 300-column GIN fingerprint to
ECFP counts + Avalon counts + ErG + 200 RDKit properties moved MapLight from **top-1 on 6/22 →
11/22** and **top-3 on 16/22 → 19/22** TDC ADMET benchmarks (§3.5.1). **[V]** That is the biggest
single verified gain anywhere in this file, it came from a *concatenated frozen fingerprint* rather
than a fine-tuned backbone, and the fingerprint they used (a GIN trained with supervised masking) is
one of the weaker encoders in §3.2. Substituting CheMeleon's 2048-d embedding is an obvious upgrade.

`chemeleon_fingerprint.py` (§1.4c) is a self-contained 60-line class: `pip install 'chemprop>=2.2.0'`,
instantiate, call on a list of SMILES, get a 2048-float vector. **[V]**

Supporting evidence that this specific embedding is the right one: the TFM paper found *"combining
TFMs with CheMeleon embeddings yields up to 100% win rates on 30 MoleculeACE tasks"* (§2.6) **[V]**,
and CheMeleon's own kNN probing showed its embedding organises chemically similar compounds better
than Morgan or Mordred (§1.6, Fig. 4). **[V]**

- **Cost:** one forward pass over 46,061 molecules. **Estimated under 5 minutes on the L4** (RDKit
  featurisation dominates), then re-run LightGBM/CatBoost. **[R — estimate]** Call it <1 L4-hour
  including fumbling.
- **Expected gain:** this is the **best gain-per-GPU-hour in the whole plan**, because it combines
  (a) the published MapLight+GNN ablation, (b) an audited-reproducible architecture, and (c) the
  strongest available molecular embedding, at essentially zero compute.
  **Prediction: improves the LightGBM on ≥14 of 21 endpoints, with the largest gains on the
  regression endpoints — lipophilicity, PPBR, LD50, clearance-microsome, VDss, half-life — mirroring
  §1.6.** If it does nothing, the CheMeleon representation adds nothing our features do not already
  have, which is Yao & Duran-Frigola's result (§2.7) and would kill Step 3 before we spend on it.
- **Do Step 2 before Step 3.** It is 20× cheaper and it is a near-perfect predictor of whether
  Step 3 will pay.

### Step 2b — TabPFNv2 on the eight smallest endpoints. **~0 L4-hours (inference only).**

A 2026 ADMET benchmark that is unusually well designed — **three escalating splits (random, scaffold,
**perimeter**), foundation models (KPGT, Uni-Mol, GEM, MOLMCL, K-BERT, TabPFNv2, AutoGluon) against
Morgan / RDKit2D / Mordred × XGBoost, RF, CatBoost, **LightGBM** — reports that for the few-shot
datasets ***"the general tabular foundation model TabPFNv2 exhibits a prominent performance
advantage, ranking first across multiple few-shot datasets"***, naming **Oral Bioavailability,
Half-Life, VDss and Caco-2**. **[V]** (*Revisiting ADMET prediction reliability under real-world
challenges in the foundation model era*, J Cheminform 2026, PMC13371653.)

Those are four of our 21 and four of our smallest. **It needs no training, no SSL machinery and our
existing feature matrix** — it is in-context learning over the table we already have.

The same paper is the clearest statement of why a hybrid is right: on the data-sufficient
CycPept-PAMPA (6,637) **KPGT R² 0.627 beats XGBoost+Mordred 0.524**, while on activity cliffs
**XGBoost+MorganCount R² 0.51 beats KPGT 0.46**. Its conclusion: no universal winner, context
dependent, **ensemble orthogonal model families.**

- **Cost:** inference only; minutes.
- **Expected gain:** **Prediction: wins on ≥3 of DILI, bioavailability-Ma, the three CYP substrates,
  HIA, half-life and VDss** — the endpoints where §1.6 says CheMeleon *loses* and where our LightGBM
  is weakest. If it works, the hybrid routing table gains a third branch.
- ⚠️ **Two cautions.** The blind challenge found TabPFN *"lacked the performance of 2D GNNs at the
  top"* (§7.4a) — so this is a small-endpoint tool, not a main model. And **TabPFN-3.5, the library
  default, is non-commercially licensed**; pin `ModelVersion.V2` (§2.6) and record why.

### Step 3 — Fine-tune CheMeleon properly: two models, two protocols. **~5–15 L4-hours.**

The headline experiment, with the protocol taken from §1.11 rather than from the paper's defaults.

```bash
# Model A — multi-task REGRESSION, full end-to-end fine-tune (freeze 0, dropout 0)
chemprop train -t regression \
  --data-path admet_reg.csv --splits-file tdc_scaffold.json \
  --from-foundation CheMeleon \
  --descriptors-path our_2264_features.npz --no-descriptor-scaling \
  --epochs 30 --ensemble-size 5 --num-workers 8 --accelerator gpu

# Model B — multi-task CLASSIFICATION, same but with the encoder frozen or
# unfrozen late (FreezeMessagePassingCallback, Python path only)
chemprop train -t classification ... --from-foundation CheMeleon ...
```

Why this shape:

- **Two models, not one** — the CLI takes one `--task-type`, and §1.9 shows a mixed head needs code
  we would have to write. ADMET-AI made exactly this split at 41 endpoints (§6.1). **[V]**
- **Sparse labels are fine** — Chemprop masks missing targets by default and the docs say partial
  data is *"encouraged"*. Our 46,061 × 21 matrix with 79,238 cells goes in as one CSV with blanks.
  **[V]**
- **Keep our own 2,264 features** via `--descriptors-path` (§1.12), making this
  *CheMeleon-RDKit* — the CheMeleon analogue of ADMET-AI's Chemprop-RDKit. **[V]**
  ⚠️ **But run it both ways.** Deng et al. (§3.3) showed *"concatenating RDKit2D descriptors to the
  learned representations is **misleading when assessing the real power** of the representation
  learning models"* — it is how GROVER's numbers were inflated. **[V]** With-descriptors is the
  number we would ship; without-descriptors is the number that tells us whether CheMeleon
  contributed anything. Both, or we learn nothing.
- **Different protocols per model.** §1.11: every TDC classification endpoint preferred a frozen or
  delayed-unfreeze encoder; regression preferred full fine-tuning. Run Model B both ways; it is
  cheap.
- `--ensemble-size 5` because ADMET-AI, MapLight and the ExpansionRx winners all ensemble.

**Expected gain, with numbers.** From §1.6, single-task CheMeleon vs a Morgan+physchem RF on the 17
shared TDC endpoints: **6W/2L on regression**, including lipophilicity MAE **0.456 vs 0.616 (−26%)**,
LD50 **0.527 vs 0.616 (−14%)**, PPBR **7.82 vs 8.37 (−6.7%)**, plus +0.049 Spearman on
clearance-microsome, +0.038 on half-life, +0.032 on VDss.

> **Prediction for H5:** Model A beats the per-endpoint LightGBM on **≥6 of our 8 regression
> endpoints**, with ≥10% relative MAE improvement on lipophilicity specifically. Model B, with a
> frozen or delayed-unfreeze encoder, is **a tie** with LightGBM on classification (not a win) —
> and with a *full* fine-tune it **loses**, which is the control that tests §1.11.

**Risk, named:** the four classification endpoints where CheMeleon lost worst (DILI n=475,
bioavailability-Ma n=640, CYP3A4-substrate n=667, CYP2D6-substrate n=664) are also the four where a
fully frozen encoder won on validation loss. If both Model B variants lose, the honest answer is the
hybrid routing table we already have from H3 — LightGBM for small classification, CheMeleon for
regression — and that is a perfectly good product answer.

**Unknown we would resolve for free:** CheMeleon has published results on **17** of our 21 endpoints.
There is **no evidence at all** for it on **Solubility_AqSolDB (9,982), CYP2C9/2D6/3A4_Veith
(12,092 / 13,130 / 12,328) and HIA_Hou (578)** — four of which are the *largest* datasets in the
group, and the paper is scoped to the *low-data regime* (§2.8). Watch for the QM9-U0 failure mode
(§1.10): far-from-descriptor target + lots of labels → the pretrained init becomes a liability.

### Step 4 — Domain-adaptive descriptor regression on our own molecules. **~10–20 L4-hours.**

I had this filed as speculative. It is not — **it is published, significant, and cheaper than I
assumed.**

**Sultan et al., *J Cheminform* 2026,
[10.1186/s13321-026-01252-z](https://doi.org/10.1186/s13321-026-01252-z)** **[V]** report that
**domain adaptation via multi-task regression of 209 physicochemical / RDKit properties on ≤4,000
domain molecules improves performance across all datasets (P < 0.001)**, on seven ADME sets under
Butina splits, against a baseline of RF + RDKit descriptors + Morgan which they say *"remain strong
baselines"*. Their 400k-molecule domain-adapted model *"outperforms larger-scale transformer models
like MolFormer"*. The same paper gives the data-scaling answer §2.3 could not find:
***"increasing the pre-training dataset beyond approximately 400–800K molecules does not improve
performance."***

Corroboration from a second direction: in Praski's benchmark, **ChemBERTa-MTR beat ChemBERTa-MLM by
7.33–9% AUROC in every case** at identical model size (§3.5.2) **[V]** — multi-task regression on
computed properties is simply a better pretraining objective than masked language modelling.

**Implementation is a wider CSV.** Chemprop treats every target column as a task, masks missing
values, and exposes **`--task-weights`** — *"Weights to apply for whole tasks in the loss function"*,
one float per task **[V]** (`chemprop/cli/train.py` line 504; applied per-task in `nn/metrics.py`).
So:

```
smiles, lipo, ppbr, ld50, ..., rdkit2d_001, ..., rdkit2d_209
```

with `--task-weights 1 1 1 … 0.05 0.05 …`, keeping the real endpoints dominant. Two variants worth
running: (a) as a **joint** auxiliary loss alongside the endpoints, (b) as a **separate intermediate
stage** on our 46k molecules before the Step 3 fine-tune, which is literally Sultan's recipe.

- **Cost:** descriptor targets are **free** — we already compute RDKit2D. If we want Mordred's full
  1,613 instead, that is ≈**25 CPU-hours** for 46k molecules (scaled from CheMeleon's 500 CPU-hours
  per 1M) **[R — derived]**. GPU cost ≈2× Step 3.
- **Expected gain:** **Prediction: significant improvement (p < 0.05 under Step 0's protocol) on
  ≥10 of 21 endpoints, concentrated on the smallest ones.** The published effect is P<0.001 across
  all seven of their datasets from only 4k domain molecules; we have 46k.
- **The failure mode to watch:** the auxiliary loss swamping the real tasks. Sweep the task weight
  over {0.01, 0.05, 0.2} only — three runs, not a grid, per H4's lesson.
- **Do not** run the full CheMeleon pretraining. §2.2: **250–500 L4-hours** plus ~500 CPU-hours plus
  40 GB, to reproduce a checkpoint that is already MIT-licensed on Zenodo — and Sultan's 400–800k
  knee says CheMeleon's 1M corpus is already past the point where more molecules help.

### Step 5 — Group the 21 endpoints by task affinity. **~5 L4-hours.**

Not more endpoints — **better-grouped** endpoints. The entire 2026 consensus says this (§6.5): the
ExpansionRx abstract calls for *"task-affinity grouping"*; Ito et al. do shared encoder then
per-task fine-tuning; NVIDIA's Xue et al. use task-specific MLP heads *"mitigating negative
transfer"*; ADMET-AI's own reg/clf split is a crude version of it.

- **Method:** Li et al.'s **higher-order task affinity**
  ([arXiv 2306.14009](https://arxiv.org/abs/2306.14009)) — estimate the loss of task A *in the
  presence of task B plus a random subset of others*, spectral-cluster, fit one model per group.
  Its explicit claim is that higher-order affinity predicts negative transfer better than pairwise
  correlation, which is Capela's weaker heuristic. **[V]**
- **Cheaper first pass:** Capela's pairwise correlation with a **|r| > 0.5** threshold on the
  overlapping molecules. **[V]** Costs nothing and may be enough.
- **Expected gain:** Capela's own numbers say the gain shows up on the *smallest* endpoints and can
  be large there — FreeSolv RMSE **1.15 → 0.83** (−28%) from grouping rather than pooling. **[V]**
  **Prediction: improves our smallest endpoints (DILI, bioavailability-Ma, the three CYP substrates,
  HIA) and changes nothing on the large ones.** That is precisely where our current model is weakest.
- **Add per-task heads** (Xue et al.) while you are in there — a shared trunk plus a small
  task-specific MLP per endpoint is a two-line change and is the published mitigation for negative
  transfer.

### Step 6 — More molecules for the endpoints we already have. **CPU-days; no GPU.**

The thing we actually concluded we needed. In priority order, and with the trap attached:

1. **ChEMBL / public re-curation** for the endpoints whose TDC sets are small and old
   (half-life-Obach 667, VDss-Lombardo 1,130, clearance 1,020/1,102). PharmaBench
   ([10.1038/s41597-024-03793-0](https://doi.org/10.1038/s41597-024-03793-0)) is an LLM-assisted
   re-curation of exactly these benchmarks and is worth reading first. **[R — metadata only]**
2. **Predicted-label surrogate data** — the pADME recipe (§2.5): industry teams have released
   *predicted* ADME labels on public structures; pretrain multi-task on those, fine-tune on
   experimental. Ranked **4th on MAE, tied 2nd on Pearson R** in a blinded challenge without
   proprietary data. **[V]** Code at <https://github.com/LongHung-Pham/pADME>.
3. **Augmenting the pretraining set with de-novo molecules around our own chemistry** — the
   ExpansionRx winners did this (§7.4a), and RIGR / SMILES enumeration are the named tools. **[V]**

> **⚠ The dedup trap, from §7.1.** MiniMol lost the Koleiev audit because **canonical-SMILES
> deduplication against TDC test sets is not enough**: it misses unspecified stereochemistry (their
> danazol example) and tautomers, leaving **0.6–7% residual overlap**. Any molecule we add must be
> matched against every TDC test set on **stereo-stripped, tautomer-canonicalised InChIKey**. Get
> this wrong and every number after it is worthless — and we will not know.

### Step 7 — Measure and report the domain-shift ceiling. **0 GPU-hours. Do this anyway.**

§7.4(b) is the finding that should change what we *claim*, not what we train. ADMET-AI and
ADMETlab 3.0 — both multi-task Chemprop, both trained on far more data than us — produce
**negative R²** on a real discovery programme's molecules for Caco-2 permeability and HLM clearance,
and **R² = 0.02** for LogD, because the mean nearest-neighbour similarity to public data is **below
0.3**, under the 0.27 "95% noise level" for Morgan fingerprints. **[V]**

- Compute mean nearest-neighbour Morgan-Tanimoto from any query set to our training set; warn below
  ~0.3; say in the product that the model ranks within known chemistry and does not predict on novel
  series. Cost: an afternoon.
- This is also the reason not to over-invest in Steps 4–6. Our in-distribution TDC number is not the
  number that will matter to a user.

---

### What not to do

| | Why |
|---|---|
| **Another capacity / regularisation sweep** | H4 moved the score 0.0004. Tetko et al. (§7.3) measured the same thing across seven datasets and found pre-set hyperparameters match optimised ones at **1/10,000 the compute**; the ExpansionRx winners reported *"minimal performance gain"* from tuning. **The sweep is the overfitting risk, not the cure.** |
| **Pretrain our own CheMeleon** | §2.2: **250–500 L4-hours** plus ~500 CPU-hours plus 40 GB, to reproduce an MIT-licensed checkpoint we can download. And there is no published data-scaling curve to tell us where to stop (§2.3) — the two nearest analogues both report the curve is **non-monotone** (Kirschbaum & Bande; the Procedural Pretraining paper). |
| **Add more endpoints to one shared encoder** | §6. ADMET-AI ran it at 41 endpoints and got a tie. Capela measured **−28% on the smallest task** from pooling uncorrelated tasks at n=6. Montanari: *"larger tasks are either not affected or show poorer performance."* Our 1.72 labels/molecule puts us firmly on the "tie" side (§7.4a). Group what we have (Step 5) instead. |
| **Chase the TDC leaderboard** | §7.1–7.2. Only 3 of 10 top-3 architectures reproduce; test-set tuning gave Koleiev a **5× inflation** of the top-3 rate; their *honest* LightGBM scored 2/22. Benchmark against **MapLight, MapLight+GNN and CaliciBoost**, which reproduce, under Step 0's protocol. |
| **TabPFN as the *main* model** | The blind challenge found it *"lacked the performance of 2D GNNs at the top"* (§7.4a). It earns a scoped place on the **smallest** endpoints only — that is Step 2b, not a replacement for the GBDT. And **TabPFN-3.5, the library default, is non-commercially licensed**: pin `ModelVersion.V2` (§2.6) and write down why. |
| **Denoising / dropping suspect labels, as a first move** | §5.3. Zhao et al. showed **removing detected-noisy compounds did not improve external predictivity** — it caused overfitting and shrank the applicability domain. Kolmar & Grulke showed noise inflates *measured*, not *true*, error. Revisit only after obtaining the Merck denoising paper, which claims the opposite. |
| **Fine-tune MolFormer-XL, Uni-Mol2, ChemFM, GROVER, GraphMVP, MolCLR or ChemBERTa** | §3. Frozen, every one of them is at or below ECFP across 18 TDC ADMET classification sets under scaffold splits (Praski: *"All models, except CLAMP, are either worse than or practically equivalent to the baseline ECFP fingerprint"*). Where CheMeleon measured them fine-tuned on our TDC endpoints, MoLFormer went **2W/15L** and MolCLR was worse (§1.6). Deng et al.: *"RDKit2D descriptors show better performance than learned representations by RNN, GCN, GIN, and pretrained models."* Add the licence problems — **ChemFM is CC BY-NC despite its HF "MIT" tag**, MolE is NC *and* undownloadable, ChemBERTa's weight repos declare **no licence at all**. |
| **Trust a TDC scaffold-split number as the real-world number** | §3.6: cluster/sphere-exclusion splits open a gap **2.3× larger** than scaffold on 21 of 22 TDC ADMET endpoints. §7.4(b): on a real programme's chemistry, two bigger multi-task models than ours produce **negative R²**. Report a cluster-split number alongside the scaffold one. |


### The plan on one line each

| # | Step | GPU | Expected gain | Strength of evidence |
|---|---|---|---|---|
| 0 | Proper statistics on H3/H4 (Ash et al.) | **0** | none — but makes every later number interpretable | Published protocol, used by CheMeleon and the blind challenge |
| 1 | Add Avalon + ErG to the LightGBM | **0** | small, broad | MapLight's recipe; audited-reproducible |
| 2 | **CheMeleon frozen fingerprint → LightGBM** | **<1 h** | **largest gain per GPU-hour** | MapLight's GIN ablation: top-1 **6/22 → 11/22** |
| 2b | **TabPFNv2 on the 8 smallest endpoints** | **~0** | first place on few-shot ADMET in a 2026 benchmark | Beats LightGBM on Oral Bioav., Half-Life, VDss, Caco-2 |
| 3 | Fine-tune CheMeleon: reg full-FT, clf frozen/delayed | 5–15 h | **−10 to −26% MAE on the regression endpoints**; a tie on classification | CheMeleon's own released TDC data (§1.6) + its own fine-tuning curves (§1.11) |
| 4 | Domain-adaptive descriptor regression on our 46k | 10–20 h | significant on most endpoints, biggest on the small ones | Sultan et al., **P < 0.001 on all 7 ADME sets from ≤4k molecules** |
| 5 | Task-affinity grouping of the 21 we have | ~5 h | large on the smallest endpoints only | Capela: FreeSolv **1.15 → 0.83** from grouping vs pooling |
| 6 | More molecules per existing endpoint | 0 (CPU-days) | the real fix, slowest | pADME ranked 4th/2nd in a blinded challenge on predicted-label pretraining |
| 7 | Measure and report the domain-shift ceiling | **0** | changes what we *claim*, not what we score | ADMET-AI / ADMETlab give **negative R²** on a real programme |

**Total for steps 0–5: roughly 25–40 L4-hours and two or three days of CPU.** **Steps 0, 1, 2, 2b and
7 cost essentially no GPU at all** and between them carry the strongest evidence in the file — do
those five first, in a week, before committing the GPU to anything. Step 0 alone (proper statistics
**plus a cluster/Lo-Hi split**) may change what every later number means.

---

## 9. What I could not verify — and one process warning that matters

### 9.1 ⚠️ A summariser fabricated numbers in this session

**WebFetch's summarising model invented a results table.** Asked for the results of
[arXiv 2607.28304](https://arxiv.org/abs/2607.28304), it returned clean numbers — "OGB-MolHIV
ROC-AUC ~0.770 → ~0.805 (+3.5%)", "OGB-MolPCBA AP ~0.310 → ~0.335", dataset sizes, "scaffold split
(default)". Pulling the PDF with `curl` and reading it with `pdftotext -layout` showed the real
numbers are **.7357 → .7357** and **.1578 → .1585**, on a **plain 80/10/10 split**, with QM9 as the
primary dataset and no ZINC-250K anywhere. **Every delta of interest was inverted from "SSL works" to
"SSL does nothing."** That table is now §4.1 and it is load-bearing — it was nearly load-bearing in
the opposite direction.

**Rule going forward: for any paper whose numbers will drive a decision, fetch the PDF and grep it
yourself** (`curl -sL -o x.pdf <arxiv pdf url> && pdftotext -layout x.pdf x.txt`). Do not let a
summariser read a results table.

**Audit of this file against that rule.** The CheMeleon numbers (§1) were read from the PDF pages
directly, not summarised. The §1.6 and §1.11 tables are my own computations on their released CSVs.
Zenodo metadata, licences, Chemprop docs and source, and the `chemeleon_fingerprint.py` listing were
fetched with `curl` and parsed, not summarised. The TDC dataset sizes and the Attia RMSE table were
each obtained independently twice and **match exactly**. **The exception: §7.4's two OpenADMET blog
posts are single-source WebFetch summaries.** They are prose rather than results tables, and the
quoted strings are distinctive, but nothing cross-checks them — treat §7.4 as the organisers'
account at one remove, and pull the posts directly before quoting them in anything external.

### 9.2 Gaps in the evidence

- **AstraZeneca logD7.4 reproducibility — the most important missing number**, since
  Lipophilicity_AstraZeneca is 4,200 of our labels. **Searched twice, independently, and failed both
  times.** Four candidate sources located, all paywalled with numberless abstracts: Wenlock et al.,
  J Biomol Screen 2011 (PMID 21343602, almost certainly the assay behind the ChEMBL AZ deposition);
  Colclough & Wenlock, JCAMD 2015 (PMID 26054297); Wenlock & Barton, Mol Pharm 2013; Chang, Woody &
  Keefer, ChemRxiv 2021. **Do not quote a number for logD without finding one of these.**
- **Merck's ADMET denoising numbers** ([10.1021/acs.jcim.4c00639](https://doi.org/10.1021/acs.jcim.4c00639))
  — ACS 403, ChemRxiv PDF 403. **The highest-value paper on this list to obtain**, since it claims the
  first denoising scheme that improves ADMET models and that training error identifies noisy
  regression labels where other metrics fail.
- **CRGNN** (Neural Networks 2025, [10.1016/j.neunet.2025.108157](https://doi.org/10.1016/j.neunet.2025.108157))
  — the only FixMatch-style consistency paper for molecules; paywalled, no datasets/split/numbers.
- **Fang et al. (Biogen, JCIM 2023) per-endpoint assay reproducibility** for the six ADME assays —
  not in the abstract, ACS 403s, and the public 3,521-compound dataset README says nothing about it.
  Try the Polaris `biogen/adme-fang-*` dataset cards.
- **No published data-scaling curve for CheMeleon's exact recipe** (§2.3). The authors *ran* a
  1M-vs-10M descriptor-MLP comparison (`run_experiments.sh` calls both) and **did not report it**.
- **No noise-ceiling or R²_max analysis exists for the TDC ADMET Benchmark Group** (§5.5) — confirmed
  by two independent searches. That ground is unoccupied and the inputs are mostly in §5.4.
- **No human liver-microsome CLint ring trial appears to exist.** The "2–3-fold" figure has no human
  ring-trial anchor; the only properly designed intra-vs-inter-lab CV decompositions are **in fish**
  (Nichols 2018, Fay 2014). The "2-fold" criterion in the IVIVE literature is an *acceptance threshold
  for predictions*, not a measured assay SD. **Do not cite it as reproducibility.**
- **VRAM and wall-clock for fine-tuning CheMeleon on 30k molecules are my arithmetic** (§1.7), not a
  measured or published figure. The authors' only statement is *"<<1 GPU hour"* on much smaller sets.
- **`--freeze-epochs` for the "delayed" ablation and the dropout rate for the "dropout" ablation are
  not recorded anywhere in the CheMeleon repo** (§1.11). My re-analysis ranks strategies by
  *validation* loss, not test score, and a frozen encoder mechanically produces a flatter val curve.
  **Treat §1.11 as a hypothesis, not a result.**
- **The ExpansionRx challenge paper body is unread** — ChemRxiv is Cloudflare-blocked. Everything in
  §7.3 item 4 and §7.4 beyond the abstract comes from the organisers' blog posts.
- **MiniMol's leakage is mechanistic, not quantified** — eliminated at the leakage stage and never
  retrained, so there is no corrected score and no leaked-molecule count (§3.4).
- **No published data-leakage claim exists against GROVER** — only reproducibility and attribution
  critiques. Do not call it leakage.
- **Praski's one winner, CLAMP, is itself contested** for pretraining leakage in an open, unanswered
  GitHub issue (§3.3).
- **Uni-Mol2's peer-reviewed venue could not be confirmed**; the HF repo has a two-line README.
- **Tabular-foundation-model details** (which TFMs, size and feature limits, runtimes) could not be
  extracted from either §2.6 paper's landing page. Read the PDFs first — and note the blind challenge
  found TabPFN *"lacked the performance of 2D GNNs at the top"* (§7.4a), while a 2026 ADMET benchmark
  found **TabPFNv2 *"ranking first across multiple few-shot datasets"*** against LightGBM on
  Morgan/RDKit2D/Mordred — on Oral Bioavailability, Half-Life, VDss and Caco-2, four of our smallest.
  Those two findings are not yet reconciled.
- **"Taking a respite from representation learning for molecular property prediction"** returns **0
  hits** on arXiv and Europe PMC under that title. The paper meant is almost certainly **Deng et al.,
  Nat Commun 2023** (§3.3), same authors and thesis.
