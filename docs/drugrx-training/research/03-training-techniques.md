# 03 — Training techniques for small models on one Colab L4

Research note, 2026-10-04. Web research only; nothing here has been run.

**Budget this is written against.** 1M–300M parameters, one Colab L4 (Ada, `sm_89`, 24 GB GDDR6,
300 GB/s, ~121 dense fp16/bf16 TFLOPS, PCIe Gen4, 72 W), ~24 h per run, checkpoint to Drive every
epoch so a run survives a session kill. Task: molecular property prediction and protein–ligand
binding-affinity regression/ranking.

**Two numbers that shape every decision below.** The L4's roofline ridge is ~400 FLOP/byte
(121 TFLOPS ÷ 300 GB/s). A 1–300M model is **memory-bandwidth and kernel-launch bound, not
FLOP bound** — so fusion, fewer bytes moved and fewer kernels matter more than arithmetic
cleverness. And 24 h on an L4 is roughly **1/10 of an A100-day**: enough for a full training run
of a 10–50M model many times over, not enough to pretrain anything from scratch on a billion
molecules.

**Confidence markers.** `[V]` = a specific number read off the source page during this research.
`[S]` = from a search snippet or secondary summary, not the paper itself. `[M]` = recalled, URL
plausible but the number was not re-read — **check before quoting**. Unmarked prose is reasoning,
not a citation.

---

## 0. Reality check: read this before picking a technique

Three findings from the literature constrain what "expected gain" can even mean here.

**0a. The benchmarks are leaky.** Schuh et al. (TU Munich, *Chemical Science* 2026) audited 51
benchmark configurations — 7 Polaris, 22 TDC, 9 MoleculeNet, 13 DTI (incl. BindingDB, PDBbind) —
and found train/test leakage and contradictory labels in most of them. DTI benchmarks averaged
**0.779 nearest-neighbour similarity between train and test**, vs 0.494 for Polaris. Enriching a
test set with near-duplicates of training molecules boosted scores, and changing the mix of
familiar compounds was **enough to flip which model ranked first**. They released `BenchAudit`. `[S]`
- https://www.chemistryworld.com/news/benchmark-flaws-may-be-distorting-ai-drug-discovery-leaderboards/4024084.article
- Also: 71% of molecules in MoleculeNet's BACE have an undefined stereocentre `[S]`;
  LIT-PCBA has documented leakage/redundancy — https://arxiv.org/abs/2507.21404
- **Consequence:** any "+3 ROC-AUC" you read, including in this document, may be leakage. Build your
  own scaffold- and time-split evaluation first. A technique that only shows up on a random split
  is not a technique.

**0b. Random Forest on fingerprints is a genuinely strong baseline, and wins outright on activity
cliffs.** van Tilborg et al. (JCIM 2022, MoleculeACE) benchmarked 24 methods on 30 targets and
found **descriptor-based ML beat the deep methods on activity-cliff compounds** `[S]` —
https://github.com/molML/MoleculeACE. The 2026 survey (arXiv:2604.16586, *JCTC* 2026) found "no
universal winner": on MoleculeNet, foundation models win but differences are "modest, with many
approaches clustering within a few percentage points of ROC-AUC"; under **time-based splits on ADME
benchmarks, four different models claim the best score across six endpoints**, with AttentiveFP
beating foundation models on some `[V]`.
- https://arxiv.org/abs/2604.16586 · repo: https://github.com/Zongru-Li/Survey-and-Benchmarks-of-DL-for-Molecular-Property-Prediction-in-the-Foundation-Model-Era
- **Consequence:** your scratch/RF baseline must be tuned as hard as the fancy model, or every
  comparison below is meaningless. Budget a day for the baseline.

**0b′. Does scale help at all for molecular property prediction?** Mixed, and the "no" side is
strong at your size. **For:** Uni-Mol2 (arXiv:2406.14969) trains a two-track transformer up to
**1.1B params on 800M conformations** and reports **power-law relations between validation loss and
model size, dataset size and compute**, with claimed gains of **27% on QM9 and 14% on COMPAS-1D** —
but on quantum/3D-style tasks `[V]`. Ramsundar et al. found task count and data volume both still
helping with no plateau `[V]`. **Against:** ChemBERTa-2 shows **no clear gain going from 5M to 77M
pretraining compounds** (BACE RMSE 1.451 → 1.509, Lipo 0.986 → 0.987 for MLM; Lipo 0.758 → 0.798 for
MTR — i.e. flat or *worse*) `[V]`; Sun et al. found little benefit from pretraining on small data
`[V]`; and CheMeleon leads Polaris/MoleculeACE at **8.7M params on 1M molecules** `[V]`. The 2026
survey asserts **sub-linear scaling** for molecular foundation models `[S]`.
**Read: scale buys quantum-property accuracy, not ADMET or affinity accuracy. Your 1M–300M ceiling
is not the binding constraint — data quality is.**
(Frey et al., "Neural Scaling of Deep Chemical Models", *Nat. Mach. Intell.* 2023, would be the
other source here; it could not be fetched, so no numbers from it are quoted.)

**0c. There is a noise floor you cannot train through.** Inter-lab IC50/Ki variability in ChEMBL
is ~0.4–0.5 log units `[M]` (Kramer et al., *J. Med. Chem.* 2012); Landrum & Riniker (JCIM 2024,
"Combining IC50 or Ki Values from Different Sources Is a Source of Significant Noise") make the
same point `[M]` — https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049. A model reporting RMSE well
below ~0.5 log on pooled ChEMBL data is fitting leakage.

---

## The catalogue

Twenty techniques. Each: what it is → expected gain → cost → when it fails → one-line PyTorch note.

---

### 1. Descriptor-prediction pretraining (the cheap pretraining that actually works)

**What it is.** Pretrain the encoder to regress *computed* descriptors (Mordred, RDKit, or ECFP
bits) from the molecular graph, on unlabelled molecules. No experimental labels, no contrastive
machinery, no 3D.

**Expected gain.** This is the strongest cost-adjusted result I found.
- **CheMeleon** (Burns et al., arXiv:2506.15792): D-MPNN, message-passing dim 2048, depth 6,
  **12.9M params total** (8.7M in message passing), pretrained on **1M PubChem molecules** to predict
  **1,613 Mordred descriptors**, with **85% of descriptor targets randomly masked in the loss** and
  MSE on the rest. Pretraining test RMSE 0.14 (standardised). On **28 Polaris tasks** win rate
  **75%** (21/28) vs minimol 71%, Random Forest 68%, MoLFormer 39%, fastprop 36%, Chemprop
  (scratch) 32%, MolCLR 14%. On **30 MoleculeACE assays**: **97%** win rate overall and **100% on
  the activity-cliff subset**, vs RF 50%/67%, minimol 40%/67%, MoLFormer 17%/33%, Chemprop
  0%/13%, MolCLR 0%/7% `[V]`.
  - Paper https://arxiv.org/abs/2506.15792 · code https://github.com/JacksonBurns/CheMeleon ·
    weights Zenodo `10.5281/zenodo.15426600` · **shipped inside Chemprop ≥ 2.2.0** `[V]`
  - It beats MoLFormer, which saw 1.1B SMILES, using 1M molecules and 12.9M params.
- **ECFP-prediction pretraining** (Money-Kyrle et al., Oxford/OPIG, arXiv:2605.10722, May 2026):
  pretrain a GIN to predict hashed ECFP bits. Config: **embedding dim 512, 3 message-passing layers,
  3-layer GIN MLPs, 2-layer head**, tuned with Optuna over 50 trials. Evaluated on **40 tasks**
  (Biogen, MoleculeNet, ChEMBL) with **Butina-cluster OOD splits, 200×5-fold CV and Wilcoxon tests** —
  unusually rigorous. **Best mean R² on 5 of 6 Biogen regression tasks** (human/rat PPB, human/rat
  CLint, efflux, solubility), with rank-biserial effect size ≈1.00 over scratch GIN, i.e. it won on
  essentially every CV repeat `[V]`. Repo: https://github.com/oxpig/topological-pretraining
  - **But read the negative results, because they are your task:** PT-GIN **did worse than classical
    fingerprints on binding affinity (DRD2, Factor Xa) and on virtual screening (MUV)**, and lost to
    Sort&Slice ECFP on FreeSolv/Lipo. Scratch GIN had the *lowest* MAPE on both CLint tasks and on
    efflux despite worse R². Substructure-level leakage between pretraining and benchmark sets had
    **minimal** impact `[V]`. **The paper's own conclusion is that 2D-fingerprint pretraining does
    not help binding affinity** — so use this for the ADMET half, not the affinity half.
  - Absolute R² deltas are not given in the text (results are reported as significance and effect
    size), and the pretraining corpus size could not be located `[V]`.
- **ChemBERTa-2 is the controlled experiment for "descriptors beat masked-LM"**, and the margin is
  not small. Same architecture, same corpus, two objectives — MTR (regress 200 RDKit descriptors)
  vs MLM `[V]`:

  | Task | MLM-77M | **MTR-77M** | scratch baseline |
  |---|---|---|---|
  | BACE RMSE ↓ | 1.509 | **1.363** | RF 1.318, GCN 1.645, D-MPNN 2.253 |
  | Clearance RMSE ↓ | 52.8 | **48.5** | D-MPNN 49.8 |
  | Delaney RMSE ↓ | 1.025 | **0.889** | GCN 0.885 |
  | Lipo RMSE ↓ | 0.987 | **0.798** | GCN 0.781, RF 0.962 |
  | BACE AUC ↑ | 0.735 | **0.799** | |
  | BBBP AUC ↑ | 0.698 | **0.728** | |
  | ClinTox AUC ↑ | 0.239 | **0.563** | |
  | SR-p53 AUC ↑ | 0.749 | **0.817** | |

  Note two things: **MLM never beat a scratch GCN on any regression task** (and its ClinTox AUC of
  0.24 is *below chance*), while MTR roughly ties GCN. And **"77M" is the SMILES count, not the
  parameter count** — the paper sweeps 5M–46M params `[V]`. The paper's own advice: use the cheap
  MLM loss to pick an architecture, then train MTR (slower, because of the 200-element label
  vector). Both checkpoints are live on HF today — `DeepChem/ChemBERTa-77M-MTR` (166k downloads last
  month) and `DeepChem/ChemBERTa-77M-MLM` (60.5k) `[V]`. https://arxiv.org/abs/2209.01712
- Historical precedent: **MolBERT**'s PHYSCHEM (RDKit descriptor) objective helped more than its
  masked-LM and SMILES-equivalence objectives `[M]` — https://arxiv.org/abs/2011.13230.

**Cost.** 1M molecules × 1,613 descriptors is a few GB of targets; Mordred featurisation is the
expensive part and is **CPU-bound** (do it once, offline, cache to disk — Colab's L4 runtime is a
`g2-standard-4`, 4 vCPU `[V]` https://docs.cloud.google.com/colab/docs/default-runtimes-with-gpus).
Pretraining a 12.9M D-MPNN on 1M molecules is comfortably inside one 24 h L4 session.
CheMeleon's own compute cost is not stated in the paper `[V]`.

**When it fails.** Descriptors are deterministic functions of the graph, so the task is noise-free
but also *closed* — the model learns no chemistry the descriptors don't contain. OOD binding
affinity is where ECFP pretraining broke down. Substructure leakage between pretraining corpus and
downstream test set inflates the gain.

**PyTorch note.** `loss = ((pred - y)**2 * mask).sum() / mask.sum()` with `mask = torch.rand_like(y) > 0.85`
— or just `pip install chemprop>=2.2.0` and load the CheMeleon weights.

---

### 2. Self-supervised pretraining: masked atoms, contrastive 2D/3D, coordinate denoising

**What it is.** The classic family. Masked atom/token modelling (ChemBERTa-2, SMILES-BERT,
Mole-BERT), contrastive with graph augmentations (MolCLR), 2D↔3D mutual information (GraphMVP,
3D Infomax), coordinate denoising (Uni-Mol, "Pre-training via Denoising").

**Expected gain — and the negative result that matters more.**
Sun et al., **"Does GNN Pretraining Help Molecular Representation?"** (NeurIPS 2022) is the paper
to read before spending a day on this. Findings `[V]`: the benefit of self-supervised pretraining
on small molecular data **can be negligible in many cases**; it **often has no statistically
significant advantage** over non-pretrained methods; supervised pretraining does help but the gain
**diminishes with richer input features or more balanced (non-scaffold) splits**; and
**hyperparameters can matter more than the choice of pretraining task**.
- https://arxiv.org/abs/2207.06010 ·
  https://papers.neurips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Paper-Conference.pdf

Per-method, for orientation:

| Method | Size / corpus | Objective | Note |
|---|---|---|---|
| MolCLR | GIN, 512-d readout + 256-d projection head / **10M PubChem** `[V]` | contrastive: atom masking, bond deletion, subgraph removal `[V]` | **see the table below** — real gains over scratch GIN, but most of them are a weak baseline. Pretrained 50 epochs, batch 512, 10 warmup epochs at lr 5e-4 then cosine `[V]`. Param count and GPU time not stated `[V]`. Scored 14% / 0% win rate in CheMeleon's Polaris / MoleculeACE comparison `[V]` — https://arxiv.org/abs/2102.10056 |
| Mole-BERT | GIN / 2M ZINC15 | VQ-VAE atom tokenizer → masked atom modelling + triplet masked contrastive | explicitly motivated by AttrMask's "small and unbalanced atom vocabulary" and its **negative transfer** `[V]`; ICLR 2023 — https://github.com/junxia97/Mole-BERT |
| GraphMVP | GEOM ~50k w/ conformers | contrastive + generative 2D↔3D; 3D only at pretraining | cheapest 3D-aware option — https://arxiv.org/abs/2110.07728 |
| 3D Infomax | — | MI between 2D and 3D embeddings | gains concentrate on quantum targets — https://arxiv.org/abs/2110.04126 |
| **Uni-Mol** | molecule model on **209M conformations**, pocket model on **3M pockets**; checkpoints ~181 MB each ⇒ **~45M params** (inferred from file size, not stated) `[V]` | masked atom type + coordinate & pair-distance denoising | **has a protein-pocket branch — the most relevant pretrained model for affinity.** Claims best on **14 of 15** molecular property tasks `[V]`. Repo is **https://github.com/deepmodeling/Uni-Mol** (the `dptech-corp` URL redirects). Weights on GitHub Releases v0.1; pretraining data is 114.76 GB (molecules) + 8.6 GB (pockets), plus a 263 MB protein–ligand binding set. Needs Uni-Core + `rdkit==2022.9.3`; `pip install unimol-tools`. Siblings: Uni-Mol+, Uni-Mol Docking V2, Uni-Mol2. **GPU-days and PDBbind/CASF numbers could not be verified** (OpenReview captcha) `[V]` |
| Denoising-as-force-learning | GNS-TAT + Noisy Nodes / PCQM4Mv2 (3.4M DFT structures, labels unused) | predict the added Gaussian coordinate noise ≡ learn a force field | **new SOTA on 10 of 12 QM9 targets.** Scratch → pretrained: HOMO 17.3 → **14.9 meV**, LUMO 17.1 → **14.7**, gap 25.7 → **22.0** (≈14–15% better); U0 6.39 → **5.76 meV**, G 7.41 → **6.90**; μ 0.021 → **0.016 D**; α 0.047 → **0.040** `[V]`. **But it gave *no* benefit on OC20 validation** — only pretraining on OC20 itself helped there. Trained on **16 TPU v3** `[V]` — far outside one L4. https://arxiv.org/abs/2206.00133 |
| GROVER | base ~48M / large ~100M; **11M** unlabelled molecules `[V]` | contextual property + motif prediction | claim verified: *"overall relative improvement is 6.1%"* over then-SOTA across 11 benchmarks `[V]`. Its own with/without-pretraining table (GROVER_large): FreeSolv RMSE 1.987 → **1.544**, ESOL 0.911 → **0.831**, Lipo 0.643 → **0.560**, QM7 MAE 89.4 → **72.6**, QM8 0.017 → **0.013**; classification AUC averaged over BBBP/SIDER/BACE 0.802 → **0.817** (GROVER-MPNN) and 0.784 → **0.812** (GROVER-GIN) `[V]`. So **~1.5–3 AUC points on classification**, more in relative terms on tiny regression sets like FreeSolv — https://github.com/tencent-ailab/grover |

**MolCLR's actual deltas, and why they overstate the case** `[V]`:

| Task | supervised GIN | **MolCLR_GIN** | supervised D-MPNN |
|---|---|---|---|
| FreeSolv RMSE ↓ | 2.76 | 2.20 | **2.18** |
| ESOL RMSE ↓ | 1.45 | 1.11 | **0.98** |
| Lipo RMSE ↓ | 0.85 | **0.65** | **0.65** |
| QM7 MAE ↓ | 124.8 | **87.2** | 105.8 |
| QM8 MAE ↓ | 0.0371 | 0.0174 | **0.0143** |
| QM9 MAE ↓ | 4.741 | **2.357** | 3.241 |
| BBBP AUC ↑ | 65.8 | 73.6 | |
| Tox21 AUC ↑ | 74.0 | 79.8 | |
| ClinTox AUC ↑ | 58.0 | 93.2 | |
| BACE AUC ↑ | 70.1 | 89.0 | |
| MUV AUC ↑ | 71.8 | 88.6 | |

The paper's own summary is **+12.4% (GCN) / +16.8% (GIN) on classification and +27.6% / +33.5% on
regression** over scratch `[V]` — which sounds decisive until you read the right-hand column:
**a plain tuned supervised D-MPNN ties MolCLR on Lipo, beats it on ESOL and QM8, and ties on
FreeSolv.** Most of the "pretraining gain" is the weakness of the scratch-GIN baseline — exactly
Sun et al.'s point, and exactly why §0b matters. (Dataset-size ablation: even **10K** pretraining
molecules gave +2.5% on HIV; the full 10M gave 75.3 → 80.6 `[V]` — the curve saturates early.)

**Cost.** This is the decisive point. **MoLFormer-XL took 208 h on 16 V100 for 4 epochs over 1.1B
molecules** `[V]` ≈ 3,300 V100-hours. **GROVER used 250 V100 GPUs for 2.5 days (base) and 4 days
(large)** `[V]` ≈ **15,000 and 24,000 V100-hours** — a thousand times your budget. Graph models on
2–10M molecules (MolCLR, Mole-BERT) are the only ones in this family you could pretrain yourself,
and they are precisely the ones Sun et al. found least convincing.

**When it fails.** Small pretraining corpus (~2M) plus small downstream label count (<10k) is the
documented failure regime. Balanced splits and rich input features both erase the gain. 3D methods
need conformers at inference, which is CPU time you don't have.

**PyTorch note.** Don't pretrain this yourself — see technique 3. If you must,
`x[torch.rand(x.size(0)) < 0.15] = MASK_IDX` in the forward is the whole of masked-atom modelling.

---

### 3. Start from a downloadable checkpoint (the dominant move)

**What it is.** Skip pretraining; fine-tune someone else's. At your budget this is not a shortcut,
it is the only way to get pretraining's benefit at all.

**Expected gain / what's available.**
- **CheMeleon** — 12.9M, in Chemprop ≥2.2.0, Zenodo weights. Best Polaris/MoleculeACE win rates
  above `[V]`. **Start here.**
- **MiniMol** (arXiv:2404.14986, ICML 2024 workshop) — **10M params**, GNN, pretrained on Graphium
  LargeMix: **~6M molecules, 526M labels, >3,300 sparsely-defined graph- and node-level tasks**
  trained jointly. Beats the previous best foundation model on **17/22 TDC ADMET tasks with 10×
  fewer parameters** `[S]`. `pip install minimol` `[V]` — https://pypi.org/project/minimol
- **MoLFormer-XL** — 12 layers, 12 heads, d=768, **46.8M params**, 1.1B PubChem+ZINC molecules,
  rotary + linear attention. HF `ibm/MoLFormer-XL-both-10pct` `[V]`
  (https://huggingface.co/ibm/MoLFormer-XL-both-10pct); the 10pct variant is only slightly worse
  than full `[V]`. Needs `trust_remote_code=True` and a pinned `transformers` version `[M]`.
- **Uni-Mol** — for the protein–ligand half; molecule and pocket checkpoints in the repo's releases.
- **ESM-2** — 8M/35M/150M/650M/3B for the protein side (see technique 8).

**Cost.** Fine-tuning a 10–47M encoder on a few hundred thousand labelled rows: hours, not days,
on an L4. Multiple full runs per 24 h session.

**When it fails.** The checkpoint's pretraining corpus may overlap your test set — this is the
substructure-leakage problem of §0a, and it is *worse* for a 1.1B-molecule corpus than a 1M one.
Pretrained SMILES models saw **canonical** SMILES, so randomised-SMILES augmentation (technique 15)
can be out-of-distribution for them. Licence-check anything commercial.

**PyTorch note.** `AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True)`,
or `from minimol import Minimol`, or `chemprop train --from-foundation CheMeleon`.

---

### 4. Multi-task learning across ChEMBL targets + ADMET endpoints

**What it is.** One shared encoder, many heads; a masked loss so missing labels cost nothing. This
is how the current ADMET leaders are built.

**Expected gain.**
- **ADMET-AI** (Swanson et al., *Bioinformatics* 2024): Chemprop-RDKit multi-task over **41 TDC
  ADMET datasets**; ranked first on the TDC ADMET leaderboard by average rank at publication `[M]`.
  Trains on a single GPU in minutes to hours — the closest published analogue to your workload.
  https://doi.org/10.1093/bioinformatics/btae416 · https://github.com/swansonk14/admet_ai
- **MiniMol** is multi-task pretraining taken to 3,300 tasks (technique 3) `[S]`.
- **Ramsundar et al. 2015**, massively multitask: **259 datasets** (128 PCBA, 17 MUV, 102 DUD-E,
  12 Tox21), **37.8M data points, 1.6M compounds**. Median 5-fold AUC `[V]`:

  | Model | PCBA (128) | MUV (17) | Tox21 (12) |
  |---|---|---|---|
  | **pyramidal multitask NN** | **.873** | **.841** | **.818** |
  | 1-hidden-layer multitask NN | .842 | .797 | .785 |
  | pyramidal single-task NN | .809 | .745 | .740 |
  | random forest | .800 | .774 | .790 |
  | logistic regression | .801 | .752 | .738 |
  | best of four single-task models | .824 | .781 | .790 |

  So **+0.05 AUC on PCBA, +0.06 on MUV, +0.03 on Tox21** over the best single-task model. Data
  volume *and* task count both contribute and **had not plateaued**; transfer to held-out tasks is
  limited `[V]`. Caveat: large models trained 50M steps — not a small-compute result.
  https://arxiv.org/abs/1502.02072
- **Chemprop v2** supports multi-task with missing labels natively, and v2 is ~**2× faster and 3×
  less memory** than v1 `[S]` — https://github.com/chemprop/chemprop

**Cost.** ~1×. Strictly cheaper than training N single-task models.

**When it fails.** Negative transfer between unrelated endpoints; a large task dominating small
ones; wildly different label scales. **Standardise every regression target to zero mean / unit
variance before summing losses** — most reported MTL failures are really unnormalised-target
failures. Group related endpoints (all ADMET together; one model per ChEMBL target family) rather
than pooling everything.

**PyTorch note.** `loss = (mask * (pred - y)**2).sum() / mask.sum().clamp(min=1)` with per-task
standardised `y`.

---

### 5. Multi-task loss balancing: uncertainty weighting vs GradNorm vs PCGrad vs …

**What it is.** Methods that set or adapt per-task loss weights, or surgically de-conflict per-task
gradients.

**Expected gain — mostly zero on vision/language benchmarks, but NOT on many-target regression.
This is the one place the two literatures disagree, and the disagreement is in your favour.**

The sceptical case, which applies to small T and similar-scale tasks:
- **Xin et al., "Do Current Multi-Task Optimization Methods in Deep Learning Even Help?"**
  (NeurIPS 2022): across large-scale language and vision experiments, specialised MTO methods
  **yielded no improvement beyond what traditional approaches achieve**, despite added complexity
  and compute `[V]` — https://arxiv.org/abs/2209.11379
- **Kurin et al., "In Defense of the Unitary Scalarization for Deep Multi-Task Learning"** (2022):
  plain loss summation with standard regularisation performs **comparably to or better than**
  specialised MTO, and several such optimisers can partly be read as forms of regularisation `[V]` —
  https://arxiv.org/abs/2201.04122

The counter-case, from **FAMO**'s own benchmark table (Δm% = average per-task change vs single-task
training, **lower is better**) `[V]`:

| Method | NYUv2 (3) | CityScapes (2) | **QM9 (11 regression targets)** | CelebA (40) |
|---|---|---|---|---|
| Linear scalarization (plain sum) | 5.59 | 14.11 | **177.6** | 6.28 |
| Uncertainty weighting (Kendall) | 4.05 | 5.89 | **108.0** | 5.78 |
| PCGrad | 3.97 | 18.29 | 125.7 | 6.65 |
| CAGrad | 0.20 | 11.64 | 112.8 | 6.20 |
| Nash-MTL | −4.04 | 6.82 | 62.0 | 4.97 |
| **FAMO** | **−4.10** | 8.13 | **58.5** | **4.72** |

**Read QM9 carefully: plain summation scored 177.6% vs FAMO's 58.5% — a 3× difference — on the one
benchmark here that is 11 regression targets at different scales.** That is the closest published
analogue to "ChEMBL pKi + 41 ADMET endpoints", and it says loss balancing *does* matter when target
scales differ. Two readings, both actionable: **(a) standardise every target** (most of the 177.6%
is almost certainly scale, not gradient conflict), and **(b) if you still see a task being crushed
after standardising, use FAMO** — it matches Nash-MTL at a fraction of the cost. Also note PCGrad
and CAGrad are **worse than plain uncertainty weighting** on QM9 and CelebA. These are the authors'
own numbers and have not been independently replicated.

| Method | Mechanism | Per-step cost | Verdict |
|---|---|---|---|
| Unitary scalarization | `sum(w_t * L_t)`, `w` fixed, targets standardised | **1×** | **the baseline to beat** |
| Uncertainty weighting (Kendall & Gal 2018) | learn `s_t`; `sum(exp(-s_t)*L_t + s_t)` | ~1× (T extra scalars) | cheap enough to ablate — https://arxiv.org/abs/1705.07115 |
| GradNorm | equalise per-task grad norms on the shared trunk | ~1.3–2×, grows with T | hyperparameter-sensitive (α) — https://arxiv.org/abs/1711.02257 |
| PCGrad | project away conflicting gradient components | **T backward passes + O(T²) projections**; T grad copies in memory | at 41 ADMET tasks this is ~41× backward. Unaffordable — https://github.com/tianheyu927/PCGrad |
| CAGrad | maximise worst-case task improvement in a ball | T backward passes + small QP | same order as PCGrad — https://arxiv.org/abs/2110.14048 |
| Nash-MTL | Nash bargaining over gradients | T backward passes + iterative solve | heaviest; amortise with `update_weights_every=k` — https://github.com/AvivNavon/nash-mtl |
| **FAMO** | fast adaptive multi-task weighting; the paper claims **O(1) space and time per iteration** vs O(k) for MGDA/PCGrad/CAGrad/IMTL-G/Nash-MTL `[V]`; ~**25× faster than Nash-MTL** on their toy example `[V]` | ~1× | **the only one in this list affordable at T=41, and the best scorer on QM9 and CelebA** — https://arxiv.org/abs/2306.03792 |

**When it fails.** The gradient-surgery methods (PCGrad, CAGrad, Nash-MTL) fail on *cost* at T > ~5
on one L4 — T backward passes per step — and PCGrad/CAGrad also underperform plain uncertainty
weighting on the two many-task benchmarks above. Uncertainty weighting diverges if a task's loss can
be driven to zero, and is misleading when a task's label noise is large (it learns the noise as
variance). Plain summation fails when target scales differ and you forgot to standardise.

**PyTorch note.** `s = nn.Parameter(torch.zeros(T)); loss = (torch.exp(-s) * L + s).sum()` for
uncertainty weighting; `pip install pytorch-pcgrad` then `optimizer.pc_backward(losses)` if you
must try PCGrad on a toy T.

---

### 6. Knowledge distillation from a large teacher into a tiny student

**What it is.** Train the small model to match a large model's outputs (logit/response KD) and/or
its hidden states (feature KD).

**Expected gain.**
- Classic form: `L = α·T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) + (1-α)·CE(y, z_s)`, typically
  **T = 2–4, α = 0.5–0.9**; the `T²` keeps gradient scale constant `[M]` —
  https://arxiv.org/abs/1503.02531. **For regression there are no logits** — use MSE or Huber on
  teacher outputs plus a label term.
- **DistilBERT**: half the layers, initialised from alternate teacher layers, loss
  `L_ce + L_mlm + L_cos` (cosine term aligns hidden states) → **97% of GLUE at 40% fewer params,
  60% faster** `[M]` — https://arxiv.org/abs/1910.01108
- **TinyBERT**: distils embeddings, **attention matrices** and hidden states layer-wise, in two
  stages `[M]` — https://arxiv.org/abs/1909.10351
- Protein-LM specifics:
  - **MTDP** (Shang et al. 2024): teachers ESM2-650M + ProtT5-XL → student a **6-layer T5, ~20M
    params (~3% of the teacher)**; loss = MLM + KL, with an RL selector choosing the teacher per
    sample. **Within ±1.5% of teacher accuracy on classification *and regression*, ~70% less
    compute.** AdamW, lr 3e-4, 15% masking, batch 16, 10 epochs, warmup 0.1 `[V]`.
    https://arxiv.org/html/2405.11735v1 · https://github.com/KennthShang/MTDP —
    **licence CC BY-NC-ND 4.0, which blocks derivative use** `[V]`.
  - **DistilProtBert**: distilled from ProtBert on 43M sequences (vs 216M); half the size and
    runtime, **~98% less pretraining compute**; secondary-structure accuracy 0.72 CASP12 / 0.81
    TS115 / 0.79 CB513; real-vs-shuffled AUC 0.92/0.91/0.87 vs teacher 0.93/0.92/0.87 `[S]` —
    https://www.biorxiv.org/content/10.1101/2022.05.09.491157.full.pdf
  - **Structure-teacher → sequence-student for affinity** (arXiv:2601.03704): student Pearson
    **0.481**, RMSE 2.488 kcal/mol vs sequence-only baseline 0.375 / 2.712; the structure teacher
    scores 0.512, so the student closes **~77% of the gap** `[V]`. This is the most directly
    relevant published result for your task.

**Cost.** Teacher inference over the transfer set, once, offline — then student training is normal.
Feature KD needs a learned projection when widths differ. Gains scale with transfer-set size, so
the real cost is teacher inference, not student training.

**When it fails.** Capacity gap too large (use a teacher-assistant / intermediate size). Teacher
wrong in the region you care about. **Too little unlabelled transfer data** — distillation's gain
comes from the transfer set, not the loss form.

**PyTorch note.** Regression KD is just
`F.huber_loss(s, t_detached) * alpha + F.mse_loss(s, y) * (1 - alpha)`; add
`1 - F.cosine_similarity(proj(h_s), h_t).mean()` for a hidden-state term.

---

### 7. Pseudo-label distillation from a physics or structure teacher (the scale play)

**What it is.** Generate millions of *synthetic* labels with an expensive oracle — Boltz-2, Vina/
Smina/Glide, FEP — and train the small model as a surrogate. Distinct from technique 6 in that the
teacher is a simulator or a structure model, not a same-modality network, and the volume is the point.

**Expected gain.**
- **Boltz-2** (MIT CSAIL / Recursion / Valence, June 2025) is the teacher worth considering.
  Trained on **~5M binding-affinity measurements** plus MD and distillation data (it already
  distils AlphaFold2 and Boltz-1). On the FEP+/OpenFE benchmark with targets held out, Pearson
  **~0.62**, "comparable to OpenFE FEP and **>1000× faster**"; **ranked first on binding affinity
  in CASP16 across 140 complexes**. **Weights, inference and training code are MIT-licensed,
  commercial use allowed** `[S]/[V]` — so using its outputs as pseudo-labels is permitted (check
  the underlying training-data licences separately).
  - https://www.biorxiv.org/content/10.1101/2025.06.14.659707 ·
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12262699/ · https://github.com/jwohlwend/boltz
  - **Catch:** Boltz-2's affinity head sits on the Pairformer and needs 3D input. A
    sequence+SMILES student distilled from Boltz-2 *values* is the realistic target — expect
    roughly the structure→sequence gap in technique 6 (~77% of it closed, Pearson ~0.48).
    And Boltz-2 inference is itself GPU-heavy, so generating 1–10M pseudo-labels is a budget line,
    not a free lunch.
- **Docking surrogates** are the cheaper, better-documented template:
  - **Deep Docking** (Gentile et al.): iterative fingerprint QSAR on docking scores over **1.36B
    ZINC15 molecules** (FRED) across 12 targets → **up to 100× fewer dockings, ~6000× enrichment**
    `[S]` — https://www.biorxiv.org/content/10.1101/2019.12.15.877316.full.pdf ·
    *Nat. Protoc.* 2022 https://www.nature.com/articles/s41596-021-00659-2
  - **Deep Surrogate Docking** (Hosseini 2022): FiLMv2 GNN, **9.5× screening speedup at <3% recall
    error** `[V]` — https://arxiv.org/abs/2211.02720
  - Review: https://arxiv.org/abs/2211.03208
- **FEP-label distillation specifically: I found nothing published.** That is a genuine gap.

**Cost.** Dominated by oracle inference. Vina/Smina on CPU is the affordable oracle; Boltz-2 is not,
at 10M scale, on one L4. The student itself (20–100M params, MSE/Huber on pseudo-labels + a small
real-label term) is one normal 24 h run.

**When it fails.** **You inherit the oracle's errors.** Docking-score artefacts (large ligands,
charged groups, pose failures) mean you learn to predict *docking scores*, not binding — so
validate against experimental assays, never only against the oracle. This is the same failure as
§8's "surrogate exploitation" and it compounds with it.

**PyTorch note.** Two-term loss with a small weight on scarce real labels:
`L = F.huber_loss(pred[pseudo], y_pseudo) + 5.0 * F.huber_loss(pred[real], y_real)`.

---

### 8. LoRA / QLoRA / DoRA fine-tuning of ESM-2 on 24 GB

**What it is.** Freeze the pretrained protein LM, train low-rank adapters (`W + BA`) plus a head.

**Expected gain — and it is large.**
- **Schmirler, Heinzinger & Rost** (*Nat. Commun.* 2024) is the reference, and it ran **on a single
  24 GB A10G** — the same memory as your L4. ESM-2 8M→3B, plus ProtT5/Ankh. **LoRA rank 4, alpha 1,
  on query/key/value/attention-output.** Full fine-tuning was feasible up to 650M; LoRA was used for
  3B. **LoRA ~4.5× faster than full fine-tuning at 3B.** Fine-tuning almost always beat frozen
  embeddings; mutation-landscape tasks (GFP, AAV, GB1) gained a lot, secondary structure ~**+1.2
  points**, disorder ~**+2.2 points**. With mixed precision, gradient accumulation and CPU offload
  most models fine-tuned in ~**8 GB** `[V]` — https://pmc.ncbi.nlm.nih.gov/articles/PMC11358375
- **SeqProFT** (arXiv:2411.11530): ESM-2 35M/150M/650M + ESM-C, 10 tasks. **LoRA r=32, α=32** on
  q/k/v (+dense). ESM-2 35M+LoRA trains 5.97M params (15%). ESM-2 650M+LoRA gained **+15.1 points
  on Fold-fold** and **+11.9 on Fold-superfamily** over frozen. **LoRA-tuned ESM-2 150M beat a
  frozen ESM-2 650M on several tasks** `[V]` — directly relevant: a tuned small model beats a
  frozen big one.
- **"LoRA Without Regret"** (Schulman et al., Thinking Machines, Sept 2025) — the practical tuning
  guide: LoRA matches full fine-tuning when configured right, at **~67% of the compute**; the
  **optimal LoRA lr is ~10× the full-FT lr** (~15× for ~100-step runs); the **1/r scaling makes the
  optimal lr roughly rank-independent**; and **attention-only LoRA underperforms — include the MLP
  layers** `[V]` — https://thinkingmachines.ai/blog/lora/ ·
  https://huggingface.co/docs/trl/main/en/lora_without_regret
- **"LoRA Learns Less and Forgets Less"** (Biderman et al. 2024): LoRA substantially
  **underperforms** full fine-tuning in-domain under standard configs (code/math, ~100k instruction
  pairs, 20B continued-pretraining tokens), but **forgets less**; full FT learns perturbations of
  **10–100× higher rank** than typical LoRA ranks `[V]` — https://arxiv.org/abs/2405.09673.
  Read as: for a narrow downstream regression head LoRA's regularisation helps; for large-scale
  transfer on millions of pseudo-labels, go high-rank or full.
- **Variants give little.** A 2026 unified study concludes well-tuned vanilla LoRA stays
  competitive and variants give **~1–3%**; DoRA is less lr-sensitive `[S]`
  (https://arxiv.org/abs/2601.22708 — treat as approximate). DoRA +1.0–4.4 on LLaMA commonsense
  `[S]` (https://arxiv.org/abs/2402.09353); rsLoRA matters at r ≥ 64 `[S]`
  (https://arxiv.org/abs/2312.03732); LoRA+ ~1–2% `[M]`; PiSSA init 72.9 vs 67.7 GSM8K on
  Mistral-7B `[S]` (https://arxiv.org/abs/2404.02948).
- NVIDIA BioNeMo: LoRA on ESM-2 650M/3B cuts GPU memory **2.5–4×** and raises throughput
  **25–80%**; plain PyTorch OOM'd above batch 16 at 650M `[S]`.

**Does full fine-tuning of ESM-2 650M fit in 24 GB?** Arithmetic: 650M in bf16 weights+grads ≈
2.6 GB; fp32 AdamW states + fp32 master weights ≈ 7.8 GB (≈1.3 GB with 8-bit Adam). With gradient
checkpointing and seq ≤ 512 it should fit with room to spare, and Schmirler's A10G result supports
it. **ESM-2 3B full FT does not fit** — LoRA or QLoRA. (Unverified by measurement.)

**Cost.** LoRA at r=8–32 trains ~1–15% of params; roughly 1.5–3× faster per step than full FT and
far less memory. QLoRA (NF4 base) is slower per step but fits 3B.

**When it fails.** Rank too low for a large-scale transfer task (Biderman). Attention-only targeting
(Thinking Machines). Under-tuned lr — LoRA's optimal lr is an order of magnitude off the full-FT
value, and most "LoRA is worse" results are this.

**PyTorch note.**
`LoraConfig(r=16, lora_alpha=32, target_modules=["query","key","value","dense"], modules_to_save=["classifier"], task_type="SEQ_CLS")`
with adapter lr ~1e-4–5e-4 and head lr ~1e-3.

---

### 9. Recursive / looped depth recycling (TRM / HRM style)

**What it is.** Apply one small block K times with a carried latent instead of stacking K distinct
blocks. Parameters stay flat; compute scales with K.

**Expected gain — and the one controlled comparison that exists.**
- **TRM** (Jolicoeur-Martineau, Samsung SAIL Montréal, 2025): a **2-layer** net does n latent
  updates then one answer update, repeated T times, with **deep supervision up to 16 steps**.
  At **5–7M params**: Sudoku-Extreme 55→**87%**, Maze-Hard 75→**85%**, ARC-AGI-1 40→**45%**,
  ARC-AGI-2 5→**8%**. Hyperparameters: AdamW β=(0.9, 0.95), **lr 1e-4, batch 768, hidden 512,
  EMA 0.999**, wd 1.0 (Sudoku/Maze) or 0.1 (ARC). **Sudoku-Extreme: 60k epochs on one L40S in
  under 36 h** — almost exactly your budget `[V]`.
  https://arxiv.org/abs/2510.04871 · https://github.com/SamsungSAILMontreal/TinyRecursiveModels
  - **The fixed-budget ablation** (Sudoku-Extreme; baseline T=3, n=6, 5M params = 87.4%) `[V]`:

    | Variant | Accuracy |
    |---|---|
    | **baseline: 2 layers, T=3, n=6 (5M)** | **87.4%** |
    | 4 layers with n=3 (**10M params**) | 79.5% |
    | separate high/low-level networks (10M) | 82.4% |
    | with ACT | 86.1% |
    | without EMA | 79.9% |
    | self-attention (7M) | 74.7% |
    | T=2, n=2 | 73.7% |
    | **1-step gradient** | **56.5%** |

    So: **recursion beat parameters here** (2 layers + more loops > 4 layers + fewer, at 2× the
    params), EMA was worth ~7.5 points, and **truncating to a 1-step gradient destroyed it**.
- **Reasoning with Latent Thoughts** (Saunshi et al. 2025): a k-layer transformer looped L times
  nearly matches a kL-layer model, and is far better than a plain k-layer one `[S]` —
  https://arxiv.org/abs/2502.17416
- **Huginn** (Geiping et al. 2025): 3.5B, 800B tokens, recurrent block iterated to arbitrary test-time
  depth; reasoning gains up to the compute-equivalent of a 50B model `[V]` —
  https://arxiv.org/abs/2502.05171
- **Ouro**: 1.4B/2.6B looped LMs with a learned early-exit gate matching up to 12B models — 2–3×
  parameter efficiency, attributed to knowledge *manipulation* not storage `[S]` —
  https://arxiv.org/abs/2510.25741
- **Mixture-of-Recursions** (Bae et al. 2025): at equal FLOPs (16.5e18), MoR Nr=2 gets **43.1%**
  few-shot vs 42.3% vanilla **with half the params**, up to 2.06× inference speedup — but **it
  underperforms vanilla at 135M params**, attributed to a "recursive capacity bottleneck" `[V]`.
  That is a direct warning at your scale. https://arxiv.org/abs/2507.10524 ·
  https://github.com/raymin0223/mixture_of_recursions
- **The deflating result.** ARC Prize's independent HRM analysis `[V]`: verified ARC-AGI-1 **32%**
  semi-private (vs 41% claimed public), ARC-AGI-2 **2%**; a **same-size 27M vanilla transformer came
  within ~5 points with no hyperparameter tuning**, so the "hierarchy" added little. The real driver
  was **the outer refinement loop** — none→one refinement added **13 points**, one→eight **doubled**
  the score — and **data augmentation** (300 augmentations near-maximum; 30 within 4% of peak).
  Training *with* refinement mattered more than refinement at inference. Cross-task transfer was
  limited (31% training on eval tasks only vs 41%), i.e. closer to test-time training than transfer.
  https://arcprize.org/blog/hrm-analysis · HRM https://arxiv.org/abs/2506.21734

**Cost.** K× forward and backward FLOPs for 1× parameters. Memory is the activation stack across
unrolled steps — TRM's answer is deep supervision plus backprop through only the last steps.

**When it fails.** **No one has published this for regression or molecular property prediction** —
all the evidence is on discrete puzzles with iterative structure and ~1,000 training examples,
where a bigger net simply overfits. MoR's 135M regression warns the gain inverts at small scale.
And the ARC analysis says most of HRM's gain was the refinement loop and augmentation, both of which
you can have without any recursion architecture.

**PyTorch note.** `for _ in range(K): z = block(z + x)` with a loss on every step (deep supervision)
and `z = z.detach()` only outside the last few — **never a 1-step gradient**.

---

### 10. Mixture-of-Experts at small scale, with aux-loss-free load balancing

**What it is.** Replace the FFN with N experts and a router that picks top-k per token/node.

**Expected gain.**
- **Aux-loss-free balancing** (DeepSeek, used in V3) is the right mechanism if you do this: add a
  per-expert bias to the routing score **for top-k selection only**, updated
  `b_i += u·sign(avg_load − load_i)` outside autograd, and weight the output by the *raw* score.
  **u = 0.001 was best** (1e-4 too slow, 1e-2 unstable). 1B: perplexity **9.50 vs 9.56** for the
  aux-loss baseline, MaxVio **0.04 vs 0.72**; 3B: **7.92 vs 7.97**, MaxVio 0.04 vs 0.52 `[V]` —
  https://arxiv.org/abs/2408.15664 · https://arxiv.org/abs/2412.19437
- **Fine-grained MoE scaling laws** (arXiv:2402.07871): MoE beats dense at every budget tested and
  the gap widens with scale; experts sized like the dense FFN are suboptimal `[S]`. Their regime is
  far above yours.
- **Expert-choice routing** (experts pick tokens → perfect balance) leaks future tokens
  autoregressively but that's irrelevant for an encoder-style predictor; MoR found expert-choice
  beat token-choice **42.6 vs 40.0** `[V]` — https://arxiv.org/abs/2202.09368
- Molecular MoE exists but is substructure/task experts on a GNN, not token MoE: GNN-MoCE over 24
  property datasets, helping most with scarce/imbalanced data `[S]`
  (https://arxiv.org/abs/2312.03292); ASE-Mol (BRICS substructure experts,
  https://arxiv.org/abs/2504.05844) `[S]`; HierMolMoE `[S]`.

**Cost — the decisive point.** **All experts stay resident in VRAM.** Total params, not active
params, set memory *and* optimizer state. The FLOP saving does not become a wall-clock saving on a
small model at small batch, because dispatch/gather overhead dominates on a 300 GB/s card.

**When it fails.** At your scale, apparently just *fails*: one reported study found that at
**10–20M active parameters MoEs underperform dense** across expert counts and granularities, only
overtaking dense with ~20× more data (4B tokens) `[S]` (likely arXiv:2605.11689 — **unverified,
check before relying on it**). Plus expert collapse, router/expert co-adaptation instability, and
**noisy load estimates at the batch sizes of tens you'll run on an L4**.

**Verdict: skip.** If you want conditional capacity anyway, 4–8 experts with the loss-free bias.

**PyTorch note.** Keep the bias as a non-grad buffer: `idx = (scores + self.bias).topk(k).indices`,
then weight by `scores.gather(-1, idx)`.

---

### 11. Curriculum by assay confidence / scaffold difficulty — or just clean the data

**What it is.** Order or reweight training examples: easy→hard, high-confidence-assay first,
scaffold-difficulty staged.

**Expected gain — close to zero, except in one regime that happens to be yours.**
- **Wu, Dyer & Neyshabur, "When Do Curricula Work?"** (ICLR 2021) `[V]`: on standard benchmarks the
  benefit is **marginal**, and **random ordering matches both curriculum and anti-curriculum**, so
  the apparent benefit is attributed to the dynamic training-set size. Curricula **do** help under a
  **drastically limited step budget** (they test 1760 and 352 steps, gains growing as the budget
  shrinks) and **under noisy labels**. Anti-curricula don't help.
  https://arxiv.org/abs/2012.03107
  - Your 24 h on 1–300M params is **not** step-starved, so the step-budget benefit doesn't apply.
    The **noisy-label** benefit does — ChEMBL noise is real (§0c).
- I found **no strong published molecular curriculum result**. Treat it as an ablation with a stop rule.

**What to do instead — data quality beats ordering.**
- Filter on ChEMBL `confidence_score` (9 = direct single protein target) and
  `standard_relation = '='`; de-duplicate by median per (compound, target, assay type); never pool
  IC50 across heterogeneous assays (§0c).
- Per-assay loss weights from `confidence_score` or replicate variance, combined with a
  heteroscedastic head (technique 14) so noisy labels are down-weighted automatically.
- Small-loss selection / co-teaching for label noise `[M]` — https://arxiv.org/abs/1804.06872
- **Data pruning** (Sorscher et al. 2022, "Beyond neural scaling laws"): with a good pruning metric
  power-law error scaling can be beaten **when data is abundant**, but **with scarce data you should
  keep the easy examples** `[M]` — https://arxiv.org/abs/2206.14486

**Cost.** Near zero for reweighting; the data-cleaning pass is human/CPU time, not GPU time.

**When it fails.** As above — most of the time, on ordering. Hard-example mining amplifies label
noise, which on activity-cliff data is exactly wrong.

**PyTorch note.** `F.mse_loss(pred, y, reduction='none') @ w` with `w` from assay confidence, or a
`WeightedRandomSampler`.

---

### 12. Active learning / acquisition loops for screening

**What it is.** Train the surrogate on a seed set, score the library, acquire a batch, re-dock or
re-assay it, retrain. Greedy / UCB / Thompson / EI over the pool.

**Expected gain — the largest multiplier in this document, but on *oracle* calls, not GPU time.**
- **MolPAL** (Graff, Shakhnovich & Coley, *Chem. Sci.* 2021): **87.9% of the top-50,000 ligands of a
  100M-member library found after testing only 2.4% of it**, with a greedy acquisition strategy;
  the paper sweeps surrogate architectures (RF / FFN / D-MPNN), acquisition functions and batch
  sizes `[V]`. Other reported points: ~94.8% of top-50k from 6% of the library `[M]`.
  https://arxiv.org/abs/2012.07127 · https://github.com/coleygroup/molpal
  Follow-ups: Pareto multi-objective https://arxiv.org/abs/2310.10598, cost-aware
  https://arxiv.org/abs/2109.11683
- **Thompson sampling over combinatorial libraries** (Klarich et al., JCIM 2024): screening **0.1%
  of a 94M quinazoline library found 90 of the top-100** molecules found by exhaustive search;
  recovery **plateaus around 90%** even with concatenated runs `[V]`. Works in *reagent* space with
  Bayesian posteriors per building block and **needs no neural net at all** — but only applies to
  Enamine-style synthesis-on-demand libraries.
  https://www.biorxiv.org/content/10.1101/2024.05.16.594622.full.pdf
- **Deep Docking**: up to 100× fewer dockings, ~6000× enrichment (see technique 7) `[S]`.
- **Active learning in FEP** (Konze et al., JCIM 2019): cuts the number of FEP calculations needed
  to find top binders, but needs a good first batch `[M]` —
  https://pubs.acs.org/doi/10.1021/acs.jcim.9b00829

**Cost.** N retrainings of a small model — cheap. The expensive part is the oracle. Note the
surprising finding across MolPAL and its follow-ups: **the choice of acquisition function mattered
less than having a decent surrogate at all** — greedy is usually strong at a few-% budget.

**When it fails — four distinct ways.**
1. **Surrogate exploitation.** Maximising a predicted score finds regions where the model is
   overconfident. This is why technique 13 is not optional.
2. **Oracle artefacts.** You recover the top *docking scores*, not the top binders.
3. **Cold start.** The first random batch (0.4–1% of the library) must contain a few positives.
4. **No stopping rule.** In a real campaign you never observe recall.

**PyTorch note.** `botorch` (`qUpperConfidenceBound`, `qNoisyExpectedImprovement`) over your feature
space, or plain `topk` on `mu + beta*sigma` from the ensemble, with a Tanimoto/Butina diversity cap
per batch.

---

### 13. Deep ensembles + conformal prediction (not optional, since we rank)

**What it is.** Train N independently-initialised models with variance heads and combine as a
mixture (ensemble); then wrap a distribution-free interval around the predictions on a held-out
calibration set (conformal).

**Expected gain.**
- **Deep ensembles** (Lakshminarayanan et al. 2017, M=5): better NLL *and* RMSE than a single net
  and than MC dropout on UCI regression, and better calibration under dataset shift `[V]` —
  https://arxiv.org/abs/1612.01474. Independent inits explore different loss basins; dropout and
  snapshots stay in one `[M]` (https://arxiv.org/abs/1912.02757).
- **Molecular evidence is more sober.** Hirschfeld et al. (JCIM 2020) compared ensembles, MC
  dropout, mean-variance, GP-on-fingerprints and others on 5 regression sets and concluded **no
  method is unequivocally best and none gives a very reliable ranking of errors** `[V]` —
  https://arxiv.org/abs/2005.10036. Scalia et al. 2020 found ensembles and MC dropout most reliable
  for docking-score regression `[M]` — https://arxiv.org/abs/1911.00782. Chemprop v2 ships
  ensembling, MVE, evidential and conformal calibration.
- **Conformal prediction**: split conformal gives `ŷ ± q̂`, where `q̂` is the (1−α)-quantile of
  calibration residuals, with **marginal coverage ≥ 1−α under exchangeability** and constant width.
  **CQR** (Romano et al. 2019, https://arxiv.org/abs/1905.03222) adapts the width.
  **Normalised conformal** — divide residuals by the ensemble's `σ̂(x)` — is the cheapest adaptive
  option once you already have an ensemble. Tutorial: https://arxiv.org/abs/2107.07511
- **Cheaper ensembles:** snapshot ensembles (https://arxiv.org/abs/1704.00109) and FGE
  (https://arxiv.org/abs/1802.10026) cost ~1× training but give less diverse members; Multi-SWAG
  (https://arxiv.org/abs/2002.08791) beats plain ensembles at equal budget `[M]`; BatchEnsemble
  (rank-1 shared members, ~1× memory) `[M]`.

**Cost.** N× training — but at 10–50M params you can train 5 members **concurrently in one process
on one L4** via `torch.func.stack_module_state` + `vmap`, bringing wall-clock close to 1–2×. At
300M, 5 members at 24 h each is not affordable.

**When it fails — and this is the important bit for a screening model.**
- All members share the same data bias, so epistemic variance stays low exactly where you're
  extrapolating (scaffold shift). Diversity from random init alone is limited.
- **Conformal's guarantee is marginal, not conditional**, and assumes **exchangeability** — which
  a scaffold split deliberately breaks. Expect undercoverage on novel scaffolds, i.e. on precisely
  the molecules a screen cares about. One 2026 study reported standard CP hitting its 90% global
  target while minority-group coverage was **64.8% on BBBP and 4.2%** on clinical-trial toxicity,
  with Mondrian (group-conditional) CP closing the gap `[S, secondary source]`.
- Fixes: **Mondrian/group-conditional** calibration per scaffold cluster; importance-weighted CP
  under covariate shift — **CoDrug** (NeurIPS 2023) reduces the coverage gap by **>35% on average**
  `[V]` (https://arxiv.org/abs/2310.12033); **ACI** (Gibbs & Candès 2021,
  https://arxiv.org/abs/2106.00170) updates α online from observed miscoverage — which fits an
  active-learning loop, where labels do arrive sequentially.
- **Practical rule: calibrate on a scaffold-split calibration set that is as far from train as your
  deployment library will be, and report coverage per scaffold cluster, not just globally.**

**PyTorch note.** `torch.func.stack_module_state` + `vmap` for the ensemble; `crepes`
(https://github.com/henrikbostrom/crepes) for conformal with an arbitrary `σ̂`, or `MAPIE` /
`torchcp`.

---

### 14. Heteroscedastic / β-NLL / evidential regression heads

**What it is.** Predict `(μ, σ²)` instead of `μ`, so the model reports per-molecule aleatoric
uncertainty for free — and down-weights noisy assays automatically.

**Expected gain.** Free aleatoric uncertainty at ~1× cost, and a principled way to handle ChEMBL's
noise floor without a curriculum. But **plain Gaussian NLL has a known pathology**: the network
down-weights high-variance regions, so the **mean fit gets worse exactly there**. Seitzer et al.
2022's **β-NLL** fix weights each point's NLL by `stopgrad(σ^{2β})` and **recovers MSE-level
accuracy while keeping calibration; β = 0.5 is a good default** `[V]` —
https://arxiv.org/abs/2203.09168. This is a few lines of PyTorch and should be the default head.

**Evidential regression** (Amini et al. 2020, NIG head → aleatoric *and* epistemic in one pass,
no ensemble) is tempting but contested: Meinert et al. 2023 argue the **uncertainty decomposition
is not valid** and it behaves like a heteroscedastic regressor `[M]`
(https://arxiv.org/abs/2205.10060). Soleimany et al. 2021 reported encouraging molecular/active-
learning results `[M]` (https://pubs.acs.org/doi/10.1021/acscentsci.1c00546); Hirschfeld's were mixed.

**Cost.** One extra output dimension. ~1×.

**When it fails.** Plain NLL, as above. Evidential as a *sole* source of epistemic uncertainty.
Both need recalibration (isotonic or σ-scaling) on held-out data before you trust the intervals.

**PyTorch note.** `nn.GaussianNLLLoss(reduction='none')` then multiply by
`sigma2.detach().pow(0.5)` for β=0.5.

---

### 15. Data augmentation: SMILES randomisation, SELFIES, conformers, masking

**What it is.** Multiply the effective dataset by exploiting representational redundancy.

**Expected gain.**
- **SMILES enumeration / randomisation** (Bjerrum 2017, https://arxiv.org/abs/1703.07076). The
  actual experiment `[V]`: 756 DHFR inhibitors, 1:9 split (602 train / 71 test), LSTM on one-hot
  SMILES, hyperparameters by Bayesian optimisation. Enumeration took the training set from 602 to
  **79,143 rows — a 130× factor**. Test R² / RMS:

  | Training | Test input | R² ↑ | RMS ↓ |
  |---|---|---|---|
  | canonical | canonical | 0.56 | 0.62 |
  | **enumerated** | canonical | **0.63** | **0.56** |
  | **enumerated** | enumerated, averaged per molecule (**TTA**) | **0.68** | **0.52** |
  | canonical | enumerated | **0.26** | 0.84 |

  Two things to take from that last row. **A model trained on canonical SMILES collapses (R² 0.56 →
  0.26) when shown randomised ones** — which is the quantified version of the warning below about
  pretrained SMILES checkpoints. And **TTA added more than training-time augmentation did**
  (0.63 → 0.68). Caveats the author states: hyperparameters were tuned on the test set with no
  separate validation split, and it is a single dataset — the gain is real but the evidence is thin.
- **Maxsmi** (Kimber, Gagnebin & Volkamer, *AI in the Life Sciences* 2021) is the molecular-specific
  study: CNNs and RNNs on four datasets (solubility, lipophilicity, bioactivity). **Augmentation
  improved accuracy independently of the deep-learning model and of dataset size**, and **the
  standard deviation across per-SMILES predictions correlates with the accuracy of that compound's
  prediction** — so augmentation buys you a free uncertainty estimate as well `[V]`.
  https://volkamerlab.org/projects/maxsmi · https://github.com/volkamerlab/maxsmi
  A related CNN study reports RMSE gains of ~0.5 on challenge datasets and ~0.2 on others
  `[S, secondary summary]` — https://arxiv.org/abs/1812.04439
- **SELFIES** (https://arxiv.org/abs/1905.13741): every string decodes to a valid molecule. Big win
  for *generation* (100% validity); **no consistent gain for property prediction** `[M]`.
- **3D conformer sampling**: sample a different RDKit ETKDG conformer per epoch (Uni-Mol stores ~10
  per molecule). Helps 3D models modestly `[M]`.
- **Atom/subgraph masking** at 10–15% as a supervised regulariser — a structured dropout. MolCLR's
  ablations found subgraph removal + atom masking best for its contrastive objective `[V]`.
- Note the ARC Prize finding (technique 9): **300 augmentations reached near-maximum performance
  and 30 got within 4% of peak** `[V]` — augmentation count saturates fast.

**Cost.** Dataset × N, so one epoch is N× longer — cut epoch count to stay in 24 h. Randomised
SMILES is free per `__getitem__`. **Conformer generation is CPU-bound and will dominate on a 4-vCPU
Colab box — pre-generate once, offline, and cache.**

**When it fails.**
- **Graph models are already permutation-invariant — SMILES randomisation does nothing for them.**
- A **pretrained** SMILES model (MoLFormer, ChemBERTa) saw canonical SMILES; randomised strings may
  be out-of-distribution for it.
- Masking that removes the functional group carrying the label (activity cliffs, toxicophores)
  injects label noise. Keep rates low.
- Conformer sampling adds noise when the label is conformer-independent.

**PyTorch note.** `Chem.MolToSmiles(mol, doRandom=True)` inside `__getitem__` — augment on the fly,
never materialise the augmented set to disk.

---

### 16. Mixup for molecules — specifically C-Mixup for regression

**What it is.** Train on convex combinations of pairs. Standard mixup is wrong for regression
because interpolating labels of structurally dissimilar molecules is chemically meaningless.

**Expected gain.** **C-Mixup** (Yao et al., ICML 2022) fixes this by sampling the mixing partner
with probability decreasing in **label** distance (Gaussian kernel on `|y_i − y_j|`). Across
**eleven datasets**, compared with the best prior approach: **+6.56% in-distribution
generalization, +4.76% task generalization, +5.82% OOD robustness** `[V]` —
https://arxiv.org/abs/2210.05775 · https://github.com/huaxiuyao/C-Mixup.
Graph-specific variants (G-Mixup, https://arxiv.org/abs/2202.07179; GraphMix) target
*classification* and don't transfer to regression `[M]`. Mixing **pooled graph embeddings** rather
than structures (manifold mixup) is the simplest thing that works inside any architecture.

**Cost.** Essentially free — one extra permutation per batch, plus a label-distance sampler built
once.

**When it fails.** **Activity cliffs** are the adversarial case: similar structures with very
different labels means label-space neighbours are *not* structure-space neighbours, which is exactly
the assumption C-Mixup makes. Also fails with heavy label noise (same reason) and on very small
datasets (<500) where the kernel bandwidth can't be set reliably.

**PyTorch note.** `j = torch.multinomial(softmax(-(y[:,None]-y[None,:])**2/(2*s**2)), 1)`, then
`h = lam*h + (1-lam)*h[j]`, `y_mix = lam*y + (1-lam)*y[j]`, `lam ~ Beta(a,a)`.

---

### 17. Test-time compute: augmentation ensembling, conformer ensembling, adaptive halting

**What it is.** Spend inference FLOPs instead of training FLOPs: average predictions over
randomised SMILES, over conformers, or over ensemble members; halt adaptively.

**Expected gain.** **TTA over randomised SMILES is the cheapest accuracy you will buy** for a
sequence model — Maxsmi's whole method is "predict over N SMILES variants, aggregate" `[V]`
(technique 15), and it hands you the uncertainty estimate as a by-product. 10–100 views per molecule.
Conformer ensembling helps 3D models a little; for binding models the **docking pose matters more
than the conformer average** `[M]`. Ensemble spread across TTA views is a good extra feature to feed
the conformal calibrator (technique 13).

**Cost.** N× inference. For a 10–50M model this is trivial — inference is not your constraint.

**When it fails.** **Does nothing for permutation-invariant graph models.** Adaptive halting /
early exit is a compute *saver*, not an accuracy gain, and is not worth the complexity at this
model size. **Test-time training** (https://arxiv.org/abs/1909.13231) has no robust molecular
result I could find — skip it.

**PyTorch note.** Batch the views: `pred = model(views).view(B, N, -1).mean(1)`, keeping
`.std(1)` as an uncertainty feature.

---

### 18. The 24 GB Ada memory/throughput toolkit

**What it is.** The mechanical settings that decide whether your 24 h is 24 h of training or 24 h of
waiting. L4 facts (NVIDIA's table quotes *with sparsity*; dense is ~half) `[V]`: 24 GB / 300 GB/s;
FP16 **and** BF16 tensor 242 sparse → **~121 dense**; TF32 120 sparse → ~60 dense; FP8 485 sparse →
~242 dense; FP32 non-tensor 30.3 TFLOPS; 72 W TDP. **Usable VRAM on Colab ~22.5 GB** `[V]`.
https://www.nvidia.com/en-us/data-center/l4/ · https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/

| Lever | Expected gain | Cost | When it fails | PyTorch |
|---|---|---|---|---|
| **bf16 autocast** | ~2× over fp32; **L4 has native bf16 at the same throughput as fp16** `[V]`, so you get fp16 speed with fp32 exponent range and **no GradScaler** | none | keep loss/softmax/norms and the regression head in fp32 (autocast does this by default) | `torch.autocast("cuda", dtype=torch.bfloat16)` |
| **TF32** | fp32 matmuls at ~60 vs 30 TFLOPS | tiny precision loss | less good than just using bf16 | `torch.set_float32_matmul_precision("high")` |
| **SDPA (not flash-attn)** | FA2-class kernels with no build step. FA2 **does** support Ada per the repo README ("Ampere, Ada, or Hopper") `[V]`; **FA3 is Hopper-only and FA4 Hopper/Blackwell — neither runs on sm_89** `[V]` | none | flash backend needs fp16/bf16 and head_dim ≤ 256; **additive attention bias (graph/distance/pair bias, Graphormer- or Uni-Mol-style) falls back to the mem-efficient kernel**; padding masks likewise. At SMILES lengths of 50–150, attention isn't your bottleneck anyway | `F.scaled_dot_product_attention(q,k,v)`; pin with `torch.nn.attention.sdpa_kernel` |
| **fused AdamW** | collapses many optimizer kernels into one — **real on a launch-bound small model** | none | — | `AdamW(..., fused=True)` |
| **torch.compile** | PyTorch's own tutorial shows 2.46× on an elementwise-dominated case `[V]`; **PyG reports "up to 300%", measured GCN 2.83× (4.81s→1.70s), GraphSAGE 2.24×, GIN 2.29×** `[V]` | tens of seconds to minutes of compile, **paid again on every Colab restart** unless you persist the inductor cache | **this is the big one for molecules: every new `(num_nodes, num_edges)` shape triggers a recompile**, and past `torch._dynamo.config.cache_size_limit` (default 8 `[M]`) it silently falls back to eager. PyG graph breaks come from `global_mean_pool` without `size=` and from `remove_self_loops`/`add_remaining_self_loops` inside the model — move those to transforms. `reduce-overhead` (CUDA graphs) needs static shapes, so it's wrong for ragged graphs; `max-autotune` isn't worth minutes of compile for small GEMMs | `torch.compile(model, dynamic=True)`; bucket node counts to multiples of 64; diagnose with `TORCH_LOGS=recompiles` and `fullgraph=True`. Docs: https://pytorch-geometric.readthedocs.io/en/latest/advanced/compile.html |
| **Gradient checkpointing** | Chen et al. 2016: a 1000-layer ResNet went **48 GB → 7 GB (−85%) for +30% runtime** `[V]` (https://arxiv.org/abs/1604.06174). Transformer figures from a secondary source: 65–75% activation memory saved for 20–30% slower `[S]` | ~+33% compute in theory (one extra forward) | **a 1–300M model at moderate batch rarely needs it** — use it only if you're OOM or want a bigger batch. Net throughput can *win* if the freed memory doubles the batch. Interacts badly with `torch.compile` graph breaks | `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` or `model.gradient_checkpointing_enable()` |
| **8-bit optimizers** | Dettmers 2022: block-wise quantised Adam state "maintaining the performance levels of using 32-bit optimizer states", a "two-line code change" `[V]` (https://arxiv.org/abs/2110.02861); bitsandbytes docs claim 75% less optimizer memory `[V]` | ~none | **the arithmetic says skip it at your scale.** Adam state is 8 B/param → ~2 B/param, so at 300M you save ~1.8 GB of a 22.5 GB budget; below 100M it's noise. Only activations scale with batch, and 8-bit doesn't touch those. Tensors < 4096 elements stay fp32 anyway | `bnb.optim.AdamW8bit(params, lr=...)` |
| **Paged optimizers** | — | — | **irrelevant without a near-OOM huge model**; they page state to host RAM during spikes | `bnb.optim.PagedAdamW8bit` |
| **Gradient accumulation** | effective batch = micro × accum | none | **the late-2024 loss-normalisation bug matters if your loss is masked or per-atom/per-token**: averaging per-micro-batch means over-weights short sequences. Correct is total loss ÷ total items across the whole accumulated window. Found by Unsloth Oct 2024, fixed in HF `[V]` — https://huggingface.co/blog/gradient_accumulation. Also makes BatchNorm stats micro-batch-local | `reduction="sum"` then `/ num_items`; divide by `accum_steps` before `.backward()` |
| **FP8 on Ada** | sm_89 *has* FP8 tensor cores and TransformerEngine says it supports Ada `[V]` (https://github.com/NVIDIA/TransformerEngine); ~242 vs 121 dense TFLOPS | quantisation overhead | **not recommended here.** Gains need large GEMMs with dims multiple of 16; at hidden ≤ 1024 the quantisation is memory-bound on a 300 GB/s card, and low precision is a bad trade for a regression target | `te.Linear` + `fp8_autocast` |
| **DataLoader** | — | — | **the likeliest real bottleneck.** The Colab L4 runtime is a `g2-standard-4` — **4 vCPU** `[V]`. RDKit featurisation, Mordred descriptors and PyG collate will starve the GPU. **Run `os.cpu_count()` / `!nproc` once, precompute and cache featurised tensors, and if the dataset fits in 22.5 GB keep it GPU-resident** | `num_workers=2..4, pin_memory=True, persistent_workers=True, prefetch_factor=4` |

Also: `optimizer.zero_grad(set_to_none=True)` (default in 2.x); wrap only forward+loss in autocast;
**xformers is unnecessary** — SDPA bundles the mem-efficient kernel; `channels_last` is irrelevant
for transformers and GNNs; and at 72 W TDP expect some sustained-clock throttling on a 24 h run `[M]`.

---

### 19. Optimizers and schedules

**What it is.** The update rule and the learning-rate trajectory — where a lot of claimed free
speedups live, and where most of them die under equal tuning.

**The framing result.** **"Fantastic Pretraining Optimizers and Where to Find Them"**
(arXiv:2509.02046) `[V]`: with proper tuning, matrix optimizers (Muon, SOAP) give **~1.4× at 0.1B
but only ~1.1× at 1.2B**, and comparisons based on intermediate checkpoints mislead. **The good
news for you: 0.1B is where the gain is largest.** The paper also found a well-tuned AdamW often
ranks top with low seed variance. Always re-tune lr per optimizer or the comparison is void.

| Optimizer | What / settings | Reported gain | Cost | Fails when |
|---|---|---|---|---|
| **AdamW** | betas (0.9, 0.95), eps 1e-8, wd 0.01–0.1, warmup 1–5%, clip 1.0, no wd on biases/norms/embeddings; lr 3e-4–1e-3 at <100M `[M]` | the baseline | 8 B/param state | — |
| **Muon** | Newton–Schulz orthogonalisation (`ns_steps=5`) of the momentum **for ndim ≥ 2 hidden weights only**; momentum 0.95, Nesterov. **Embeddings, the output/regression head, biases and norm gains must go to AdamW** via `MuonWithAuxAdam` | README: 1.35× on GPT-2 small, CIFAR-10 3.3→2.6 A100-s `[V]`. Moonshot "Muon is Scalable" (https://arxiv.org/abs/2502.16982): **~2× compute efficiency vs AdamW** at compute-optimal, with weight decay and update-RMS matching (~0.2) so AdamW's lr/wd transfer `[V]`. 17M–202M decoders: target loss at **~48–52% of AdamW's FLOPs** `[S]` (https://arxiv.org/abs/2505.02222) | one momentum buffer (4 B/param) + a few % FLOPs for NS | **completely untested on regression, GNNs or molecules** — treat as an ablation. Tiny matrices (hidden < 128) gain little. Split fused QKV. https://github.com/KellerJordan/Muon; PyTorch ≥2.9 may ship `torch.optim.Muon` `[M]` |
| **Lion** | sign-of-momentum; **lr 3–10× smaller and wd 3–10× larger than AdamW**, betas (0.9, 0.99) `[V]` | ViT +up to 2% ImageNet, up to 5× less JFT pretraining compute, diffusion FID at up to 2.3× less compute `[V]` | **half the state** (1 buffer) | needs batch ≥ 64; the sign update has large per-element norm so lr must be small; sensitive for noisy regression targets. "No Train No Gain" found Lion ≈ AdamW on BERT downstream `[S]` — https://arxiv.org/abs/2302.06675, `pip install lion-pytorch` |
| **Sophia** | diagonal-Hessian preconditioner refreshed ~every 10 steps + elementwise clipping | **claimed** 2× on GPT 125M–1.5B | 2nd state + Hessian estimation | **skip.** "No Train No Gain" found no speedup at equal iterations even ignoring Hessian cost `[S]` (https://arxiv.org/abs/2307.06440); Fantastic Optimizers didn't rank it top either |
| **SOAP / Distributed Shampoo** | Adam in Shampoo's eigenbasis (one extra hyperparameter: preconditioning frequency) | SOAP on 360M/660M LMs at large batch: **~40% fewer iterations, >35% less wall-clock** than AdamW, ~20% better than Shampoo `[V]` (https://arxiv.org/abs/2409.11321). **Distributed Shampoo won AlgoPerf 2024's external-tuning track, ~28% faster than the NAdamW baseline** `[S]` | preconditioner factors per dim + periodic eigendecompositions — can be latency-bound on one L4 | wins are reported **at large batch**, which you may not afford; embeddings usually blocked or skipped — https://github.com/nikhilvyas/SOAP, https://github.com/facebookresearch/optimizers |
| **Schedule-Free AdamW** | removes the schedule entirely by combining interpolation and iterate averaging; **no need to know the total step count T** | **won AlgoPerf 2024's self-tuning track, ~8% faster than baseline** `[S]`; https://arxiv.org/abs/2405.15682 | extra buffers | **you must call `optimizer.eval()`/`.train()` around evaluation** or you evaluate the wrong iterate — easy to forget, and it breaks BatchNorm. Checkpoint the averaged state. **Its "no T needed" property is exactly right for multi-session Colab** — https://github.com/facebookresearch/schedule_free |
| **AdEMAMix** | mixes a fast and a slow gradient EMA | a 1.3B LM on 101B tokens matched AdamW on 197B (~95% token efficiency), and forgets more slowly `[V]` | 2nd momentum buffer; needs warmup schedules on α and β₃ | the slow EMA only pays at long horizons (tens of thousands of steps); untested for regression — https://arxiv.org/abs/2409.03137 |

**Schedules.**
- **Cosine** needs T up-front. **Stopping early gives a worse model and you cannot cleanly extend a
  run** — a bad fit for a session that may die at hour 9.
- **WSD (warmup–stable–decay)** is the right default here. MiniCPM: "conducive to continuous
  training and domain adaptation" `[V]` (https://arxiv.org/abs/2404.06395). Hägele et al. 2024
  (NeurIPS spotlight): **constant LR plus a short cooldown "scales predictably and reliably similar
  to cosine"**, and **SWA improves performance "throughout training" at no cost** `[V]`
  (https://arxiv.org/abs/2405.18392). Cooldown ~10–20% of steps; a `1-sqrt` shape tends to beat
  linear `[M]`. **The operational property that matters: you can branch a cooldown off any
  stable-phase checkpoint**, so a session that dies at hour 9 still yields a deliverable model.
  Delta vs cosine: roughly parity, not a win — the win is restartability.
- One-cycle and linear-decay-to-zero also need T. Linear-to-zero roughly matches cosine `[M]`.

**EMA / weight averaging.** `w_ema = d*w_ema + (1-d)*w`, d ≈ 0.999–0.9999. Costs one extra param
copy and a cheap update; gives a small consistent gain and much smoother validation curves. Note
TRM's ablation: **removing EMA cost 7.5 points (87.4% → 79.9%)** `[V]`. SWA gives a similar effect
(Hägele et al.). Model soups (https://arxiv.org/abs/2203.05482) average fine-tunes from a shared
init — ~free once you have the runs, but only works within one basin. **Fails** with BatchNorm
(recompute stats) and with too-high decay on a short run.
`AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))`.

**muP.** Yang & Hu's Tensor Programs V: lr and init transfer across widths; tuning a 13M proxy beat
BERT-large (350M), and a 40M transfer beat GPT-3 6.7B at 7% of its pretraining cost `[V]`
(https://arxiv.org/abs/2203.03466); u-muP adds unit scaling and claims near-optimal defaults plus
out-of-the-box FP8 (https://arxiv.org/abs/2407.17465). **Verdict: only worth the wiring if you will
sweep widths** (tune at 5M, run at 100M). For one fixed-size model, a direct lr sweep is cheaper.

---

### 20. Evolutionary / population approaches — and what is affordable

**What it is.** PBT, hyperparameter search, model merging, NAS.

**Affordable on one L4.**
- **ASHA / Hyperband with early stopping is the practical choice**: a budget of ~5–10 full training
  runs covers 50–100 configurations. Random search with early stopping is a strong baseline and
  Hyperband usually beats plain Bayesian optimisation on wall-clock `[M]`.
  Hyperband https://arxiv.org/abs/1603.06560 · ASHA https://arxiv.org/abs/1810.05934 ·
  BOHB https://arxiv.org/abs/1807.01774. `optuna.pruners.HyperbandPruner` with TPE, or Ray Tune's
  `ASHAScheduler`; run 2–4 trials concurrently for small models.
- **SWA / model soups / EMA** — see technique 19. Free gains from runs you already have.

**Marginal.** **PBT** (https://arxiv.org/abs/1711.09846) needs a population of 10–40 concurrent
trainings. At ~10M params a population of ~8 fits on one L4 — use it only if one member trains in
under ~1 GPU-hour. At 100–300M you get one or two concurrent runs, so it degenerates.

**Not affordable, say so plainly.**
- Full NAS by training candidates.
- PBT at 100M+.
- Any N-fold ensemble of 300M models at 24 h each — a single run already *is* the budget.
- **Evolutionary model merging** (Sakana, https://arxiv.org/abs/2403.13187) needs many full
  evaluations of merged models and strong pretrained parents. TIES
  (https://arxiv.org/abs/2306.01708) and DARE (https://arxiv.org/abs/2311.03099) target merging
  fine-tuned LLMs — probably irrelevant to small from-scratch models.
- **Zero-cost NAS proxies: skip.** NAS-Bench-Suite-Zero (White et al., NeurIPS 2022) found
  **simple baselines (parameter count, FLOPs) are competitive with all existing proxies across most
  settings, and most proxies do not generalise across benchmarks** `[V]` —
  https://arxiv.org/abs/2210.03230. They were designed for vision CNN spaces; nothing validates them
  for molecular GNNs or transformers. Use parameter count as the baseline.

---

## What's new in 2025–2026 that changes the picture

Items not in the original brief, or that materially update it.

1. **Descriptor-prediction foundation models are the small-budget story.** CheMeleon (June 2025,
   12.9M params, 1M molecules) beating MoLFormer (1.1B molecules) on Polaris and MoleculeACE, and
   shipping inside Chemprop ≥2.2.0, is the single most actionable 2025 result for this budget `[V]`.
   ECFP-prediction pretraining (2026) is the same idea, cheaper still `[V]`.
2. **Benchmark auditing went mainstream.** Schuh et al.'s 51-configuration audit (*Chem. Sci.* 2026)
   and `BenchAudit`, plus the 2026 *JCTC* survey's finding that time-based ADME splits reshuffle the
   leaderboard entirely. **This is a methodology change, not a technique: it invalidates a lot of
   the deltas elsewhere in this document.** `[S]`/`[V]`
3. **Boltz-2** (June 2025) made an FEP-quality, MIT-licensed affinity teacher available at >1000×
   FEP speed, which is what makes technique 7 possible at all `[S]`.
4. **"LoRA Without Regret"** (Sept 2025) settled the LoRA tuning questions: ~10× lr, rank-independent
   via 1/r scaling, must include MLP layers, ~67% of full-FT compute `[V]`.
5. **Muon** went from a NanoGPT-speedrun curiosity to a scaled, published optimizer (Moonshot,
   Feb 2025), and the Sept 2025 "Fantastic Optimizers" audit says the gain is **largest at ~0.1B**
   — i.e. at your size — while shrinking to ~1.1× at 1.2B `[V]`. Untested on regression.
6. **Schedule-Free AdamW** won AlgoPerf's self-tuning track and removes the need to know T, which is
   worth more under Colab's session limits than its headline 8% `[S]`.
7. **WSD is now the documented-and-measured default** for restartable training (Hägele et al.,
   NeurIPS 2024), including the finding that SWA helps throughout training at no cost `[V]`.
8. **Tiny recursive models** (TRM, Oct 2025) produced the first clean fixed-budget "recursion beats
   params" ablation — on puzzles, at 5M params, on ~1,000 examples — while the ARC Prize's HRM
   analysis showed most of the family's gain is the outer refinement loop and augmentation, not the
   architecture `[V]`. Mixture-of-Recursions' failure at 135M is the relevant caution `[V]`.
9. **Latent recurrent depth at scale**: Huginn (3.5B, test-time depth scaling) and Ouro (looped
   1.4B/2.6B matching 12B) make test-time depth a real axis — but both are LMs, not predictors `[V]`/`[S]`.
10. **Aux-loss-free MoE load balancing** (DeepSeek) is now the standard mechanism and removes the
    main reason MoE was fiddly — though the small-scale evidence still says don't `[V]`.
11. **Compact-foundation-model work**: pruning JMP-L's interaction blocks + node/edge feature
    distillation gives **160.1M → 108.2M params (−32%) at 1.3× inference throughput** (19.1 → 25.6
    samples/s) with the 5-block variant comparable to the full 6-block model `[V]` —
    https://arxiv.org/abs/2504.19538 · https://github.com/Yasir-Ghunaim/efficient-jmp
12. **Protein-LM distillation matured**: MTDP's ~20M student within ±1.5% of ESM2-650M/ProtT5-XL on
    classification *and regression* at ~70% less compute, and a 2026 "reverse distillation" ICLR
    paper exploiting the fact that **mid-sized PLMs often beat the largest ones** on ProteinGym `[V]`/`[S]`.
13. **Covariate-shift-aware conformal prediction** (CoDrug >35% coverage-gap reduction; Mondrian
    group-conditional CP) is the fix for the scaffold-shift problem that plain conformal has `[V]`/`[S]`.
14. **MiniMol** (2024, 10M params, 3,300 pretraining tasks, beats the prior best FM on 17/22 TDC
    ADMET with 10× fewer params) is the other small checkpoint worth benchmarking against CheMeleon
    `[S]`. **MolE** (arXiv:2211.02657, DeBERTa on molecular graphs, two-stage self-supervised →
    biological-multitask pretraining) was best on **9 of 22** TDC ADMET tasks at publication `[V]`.
15. **FA3/FA4 are Hopper/Blackwell-only**, so on an L4 the attention story is settled: use PyTorch
    SDPA and don't build flash-attn `[V]`.
16. **FAMO** (2023, but it's the one MTL method that survived the 2022 critiques) reframes the
    multi-task question: O(1) overhead, matched Nash-MTL, and the QM9 column of its table is the
    strongest published argument that **many-target regression is the exception where loss balancing
    earns its keep** `[V]`.
17. **Uni-Mol2** (arXiv:2406.14969) is the first molecular model with published **power-law scaling
    of validation loss against size/data/compute**, to 1.1B params on 800M conformations, with 27%
    on QM9 and 14% on COMPAS-1D `[V]` — useful mainly as evidence about *where* scale pays
    (quantum/3D), not as something you can run.

---

## Running long training on Colab reliably

### What the docs actually say

From the official FAQ, **https://research.google.com/colaboratory/faq.html**, quoted `[V]`:

- *"In the version of Colab that is free of charge notebooks can run for at most 12 hours, depending
  on availability and your usage patterns."*
- *"Colab Pro+ supports continuous code execution for up to 24 hours if you have sufficient compute
  units."*
- *"Colab prioritizes interactive compute. Runtimes will time out if you are idle."*
- *"Idle timeouts only apply if code execution terminates."*
- *"You can expect to experience backend termination if you exhaust your available compute units on a
  Pro, Pro+, or Pay As You Go plan."*
- *"Colab does not publish these limits, in part because they can vary over time."*
- *"The types of GPUs and TPUs that are available in Colab vary over time."*
- *"Colab resources are not guaranteed and not unlimited, and usage limits sometimes fluctuate."*
- Prohibited actions include *"file hosting, media serving, or other web service offerings not
  related to interactive compute with Colab; downloading torrents or engaging in peer-to-peer
  file-sharing; connecting to remote proxies; mining cryptocurrency; running denial-of-service
  attacks; password cracking; using multiple accounts to work around access or resource usage
  restrictions; creating deepfakes; employing techniques such as containerization to circumvent
  anti-abuse policies."*
- Drive: *"storing more than approximately ten thousand items in the root directory can cause
  mounting to fail"*; *"Accessing items in any folder containing many items can cause errors like
  `OSError: [Errno 5] Input/output error`."*

**Read it carefully: "at most 12 hours" and "up to 24 hours" are ceilings, not commitments.** The FAQ
declines to publish usage limits and declines to guarantee resources. **Design for preemption at any
moment; treat 24 h as a best case.** There is no FAQ text giving a Pro lifetime number or an exact
idle-minutes figure (community lore says 90 min idle on free — unverified).

### Background execution

- Pro+ "continuous code execution for up to 24 hours" is the sanctioned route, and is described as
  continuing with the tab closed `[S]`.
- **New, and it changes the answer**: Google's developer blog, **2026-09-22**, *"Colab is now part of
  your Google AI plan"* `[V]`: Google AI subscribers get *"priority access to faster accelerators and
  more powerful machines"*, and the **Ultra** tier unlocks *"uninterrupted background execution and
  Premium GPU access, so long training runs can finish without keeping a browser tab open."*
  Existing Colab subscriptions are unchanged and *"Google AI subscription benefits can be stacked."*
  Rolling out "over the next few weeks" from that date.
  https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/
  **It is not confirmed whether Google AI Pro (as opposed to Ultra) gets background execution.**
- **Background execution is reported to be unreliable.** Open `colabtools` issues: **#5950**
  (2026-04-14) reports Pro+ runtimes ending after **40 min to 1 h while running**, and closing the
  tab stopping the run — labelled a bug, no visible Google response; **#4151** and **#5804** are
  similar `[V]`. https://github.com/googlecolab/colabtools/issues/5950
  **Plan resume logic regardless of plan tier.**
- I found no FAQ text banning keep-alive JS, but the FAQ does say runtimes time out when idle, and
  free-tier extras like remote shells and "bypassing the notebook UI" are called out. Keep-alive
  scripts are a grey area; **don't run them on the free tier.**

### Compute units and what an L4-hour costs

Prices and allowances (**not read from the live pricing page — Colab's pricing/signup pages returned
403 or a sign-in stub to every fetch attempt, so verify these in-app**):

- Pro **$9.99/mo for 100 units**; Pro+ **$49.99/mo**; pay-as-you-go **$9.99 per 100 units** `[S]`.
- **Pro+ unit allowance is unresolved**: 2026 secondary sources say **600**, an older page says
  **500** `[S]`. Either way ≈ **$0.083–0.10 per unit**.
- Unused units reportedly carry over up to **90 days** `[S]`, not officially verified.
- **Colab does not publish burn rates.** The resource picker shows the live rate in-app. Everything
  below is **community-measured and the spread is large**:

| GPU | Oct 2024 measured `[S]` | Mar 2026 measured `[S]` | Other 2026 claims `[S]` |
|---|---|---|---|
| T4 | 1.58 CU/h | ~1.19 | 1.5–2.0, 1.76 |
| **L4** | **3.00** | **~1.71** | 2.5–5.0 |
| A100 40 GB | 10.59 | ~5.40 | 13–15 |
| A100 80 GB | — | ~7.52 | — |
| H100 | not offered | not offered | never definitively measured |
| TPU v2-8 | 1.76 | — | — |
| CPU only | 0.07 | — | — |

Sources: https://varlog.info/colab-credit-consumption-comparison-2024/ ·
https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025

- **Implied $/hr** at ~$0.10/unit: T4 ≈ $0.12–0.20; **L4 ≈ $0.17–0.30**; A100 ≈ $0.54–1.06
  (up to ~$1.5 at the 15 CU/h figure).
- **L4-hours per Pro+ month** (500–600 units): **~290–350 h** at the Mar 2026 rate (1.71),
  **~165–200 h** at the Oct 2024 rate (3.0), **~100–240 h** at 2.5–5.0 CU/h. **Even 500 units covers
  a 24 h run comfortably — roughly 7–14 days of L4 per month.** Units, not session length, are the
  binding constraint only if you run near-continuously.
- GPUs on offer: free tier is T4 only `[S]`; paid tiers list T4, L4, A100 40/80 GB and TPU v2-8 `[S]`.
  **No firm evidence of V100, H100, TPU v5e or v6e.** Check the picker on the day.
- **I could not verify Colab Enterprise / Vertex max-runtime or idle numbers** — those pages were
  unreachable. It's billed as GCE/Vertex compute, not compute units: https://cloud.google.com/colab/docs

### The checkpoint-resume pattern

Save **every** piece of state, or resume silently diverges:
model weights, optimizer state, LR-scheduler state, **GradScaler/AMP state** (if fp16),
**RNG states for python/numpy/torch/CUDA**, sampler/dataloader epoch *and* position,
**EMA shadow weights**, **global step**, and the W&B run id.

- **Atomic save, always**: `torch.save(state, "ckpt.tmp"); os.replace("ckpt.tmp", "ckpt.pt")`. A
  non-atomic save that is killed mid-write leaves you with no checkpoint at all.
- **Keep two rotating checkpoints** (`ckpt_a.pt` / `ckpt_b.pt`), so a corrupt newest one isn't fatal.
- **Schedule by step, not epoch.** A cosine schedule resumes correctly only if you restore the
  scheduler state *or* recompute LR from the global step; **WSD or constant-LR is simply easier to
  resume because there's no dependence on a fixed total** (technique 19). Schedule-Free removes T
  entirely.
- **Write to local disk, then copy one file to Drive.** The FAQ's own Drive guidance is the reason:
  many small writes hit per-user and per-file operation and bandwidth quotas and produce
  `OSError: [Errno 5]`, and >10k items in a folder breaks mounting. The FAQ's suggested workarounds
  are archive formats instead of many small reads, and waiting for quota resets `[V]`.
- The Drive mount is FUSE; after long idle or token expiry it may need
  `drive.mount('/content/drive', force_remount=True)` `[S]` — no official statement on token expiry
  was found.
- **Persist the inductor cache to Drive** (`TORCHINDUCTOR_CACHE_DIR`) or you pay `torch.compile`'s
  compile time again on every session.
- **Write a heartbeat file** (timestamp every N steps) rather than relying on signal handlers — I
  found no official Colab text on SIGTERM behaviour, so assume the VM is simply deleted.
- Reference implementations: HF Trainer `resume_from_checkpoint`
  (https://huggingface.co/docs/transformers/main_classes/trainer) · Lightning `ckpt_path`
  (https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html) · nanoGPT
  (https://github.com/karpathy/nanoGPT/blob/master/train.py) · torchtitan
  (https://github.com/pytorch/torchtitan) · `torch.distributed.checkpoint`
  (https://pytorch.org/docs/stable/distributed.checkpoint.html) · W&B resume
  `wandb.init(id=..., resume="allow")` with the id stored *in the checkpoint*
  (https://docs.wandb.ai/guides/runs/resuming).

### How to structure a 24h+ multi-session run

1. **Idempotent resume from step 0.** The notebook's first cell looks for a checkpoint and resumes;
   there is no "first run" code path. Test it by killing the runtime deliberately.
2. **Step-based everything** — schedule, eval, checkpointing, logging.
3. **Checkpoint on a time interval**, not an epoch boundary (every ~15–20 min), since "epoch" may be
   hours. The brief's "every epoch" is too coarse if an epoch is long.
4. **WSD or Schedule-Free**, so any checkpoint can be cooled down into a deliverable model.
5. **Short eval passes** written to Drive as JSONL, so a dead session still leaves a record.
6. **Precompute and cache featurisation** — on 4 vCPU this otherwise dominates, and it's wasted
   work every session.
7. **Log to W&B with a stable resumed run id**, so a 4-session run reads as one curve.

### Is Colab even the cheapest 24 GB card?

2025–26 prices from secondary aggregators, **not provider pages** `[S]` — verify before deciding:
RunPod L4 from **$0.39/h**, RTX 4090 $0.34–0.69/h, A5000 24 GB $0.27/h community cloud
(https://www.runpod.io/gpu-cloud/pricing); Vast.ai RTX 4090 ~$0.31–0.40/h on demand, lower
interruptible; Lightning AI T4 from ~$0.19/h with a free monthly-credit tier incl. L4/A10G/L40S;
Thunder Compute A100 80 GB $1.09/h; Paperspace A6000 $1.89/h; Kaggle free P100 or T4×2 at ~30
GPU-h/week with per-session caps (commonly 9–12 h, unverified).

**Judgement.** At the Mar 2026 community rate, Colab L4 at ~$0.17–0.30/h undercuts RunPod's $0.39
L4 and a $0.34–0.69 4090. At the Oct 2024 rate or 2.5–5 CU/h it's a wash or worse. **The real cost of
Colab is not the price — it's the absence of guarantees plus the open background-execution bug
reports.** If a run must finish on a deadline, rent a 4090 or L4 by the hour; if cost per
experiment is what matters and your resume logic is solid, Colab Pro+ is competitive.

---

## Ranked shortlist: the 7–10 techniques to actually use, in order, with expected benefit for the money

Ordered by expected benefit per unit of effort and GPU time, under this specific budget.

**1. Fix the evaluation before anything else — scaffold *and* time splits, plus a leakage audit.**
*Benefit:* not a model gain but the precondition for every other number being real. Schuh et al.
showed the benchmark mix alone **flips which model ranks first**, and the 2026 survey showed
time-split ADME reshuffles the leaderboard entirely `[V]`. *Cost:* ~1 day, no GPU. *Why first:*
every item below is unmeasurable without it, and a leaky split will tell you MoE and recursion work.

**2. A hard-tuned baseline: Chemprop/D-MPNN + RDKit descriptors, and Random Forest on ECFP.**
*Benefit:* RF wins outright on activity cliffs (MoleculeACE `[S]`) and placed 68% on Polaris vs
CheMeleon's 75% `[V]`. *Cost:* hours, mostly CPU. *Why:* Sun et al.'s central finding is that
**hyperparameters matter more than the pretraining task** `[V]`, so an under-tuned baseline
manufactures fake wins for everything else.

**3. Start from CheMeleon (and benchmark MiniMol / MoLFormer-XL-10pct against it).**
*Benefit:* the best cost-adjusted published result for this budget — **75% Polaris, 97%
MoleculeACE, 100% on the cliff subset, at 12.9M params from 1M molecules** `[V]`, beating a model
pretrained on 1.1B. Weights are downloadable and it's in Chemprop ≥2.2.0. *Cost:* near zero —
`pip install` and fine-tune in hours. *Why third:* it only shows as a gain against a real baseline.
*Important scope limit:* the descriptor/fingerprint-pretraining evidence is strong for **ADMET and
physicochemical** endpoints and **explicitly negative for binding affinity** — the ECFP-pretraining
paper found PT-GIN losing to classical fingerprints on DRD2, Factor Xa and MUV `[V]`. **Use
CheMeleon for the property half; for the affinity half go to Uni-Mol (it has a pocket branch) or
distillation (#9).**

**4. Multi-task heads over ChEMBL + ADMET with a masked loss, standardised targets, and FAMO if a
task still gets crushed.**
*Benefit:* ADMET-AI took the TDC leaderboard this way, training in minutes to hours on one GPU `[M]`;
MiniMol's 3,300-task pretraining is the same idea scaled `[S]`; Ramsundar's +0.03–0.06 median AUC
over the best single-task model `[V]`. *Cost:* ~1×, cheaper than N single-task models.
**Standardising every regression target to zero-mean/unit-variance is the load-bearing step** —
on QM9's 11 differently-scaled regression targets plain summation scored Δm% 177.6 vs FAMO's 58.5
`[V]`, and most of that gap is scale. **Explicitly skip PCGrad/CAGrad/Nash-MTL**: Xin et al. found no
gain over tuned baselines `[V]`, PCGrad and CAGrad were *worse* than plain uncertainty weighting on
QM9 and CelebA `[V]`, and at 41 tasks they cost ~41× backward. **FAMO is the one balancing method
worth wiring in** — O(1) per step, matched Nash-MTL at ~1/25 the cost `[V]`.

**5. β-NLL heteroscedastic head + a 5-member deep ensemble + scaffold-calibrated conformal
prediction.**
*Benefit:* this is the one that makes *ranking* trustworthy, and ranking is the product. β-NLL
recovers MSE-level accuracy while keeping calibration (β=0.5) `[V]`; ensembles beat MC dropout on
NLL and shift-calibration `[V]`; conformal gives distribution-free intervals. *Cost:* ~1× for the
head; the ensemble is 5× nominal but close to **1–2× wall-clock at 10–50M params via
`stack_module_state` + `vmap` on one L4**; conformal costs only a calibration split. *Caveat that
must be respected:* **calibrate on a scaffold split and report per-cluster coverage** — plain
conformal's exchangeability assumption is exactly what a scaffold split breaks, and group-conditional
coverage can collapse (64.8% / 4.2% in one reported case `[S]`). Use Mondrian CP or CoDrug-style
importance weighting (>35% gap reduction `[V]`).

**6. The L4 mechanics: bf16 autocast, SDPA, fused AdamW, WSD (or Schedule-Free) + EMA, cached
featurisation, `torch.compile` with bucketed shapes.**
*Benefit:* bf16 ≈ 2× over fp32 at no precision cost on Ada `[V]`; PyG measured **2.2–2.8× from
`torch.compile`** `[V]`; EMA was worth **7.5 points** in TRM's ablation `[V]`; WSD makes any
checkpoint deliverable `[V]`. *Cost:* a day of engineering, then free forever. *Skip:* gradient
checkpointing (you're not OOM), 8-bit optimizers (saves ~1.8 GB of 22.5 at 300M), paged optimizers,
FP8. *Watch:* dataloader starvation on 4 vCPU and `torch.compile` recompilation on ragged graphs —
these are the two things most likely to silently eat your 24 h.

**7. Randomised-SMILES augmentation at train time + TTA at test time (sequence models only).**
*Benefit:* Maxsmi found augmentation improved accuracy **independently of model and dataset size**,
and the **std across per-SMILES predictions correlates with that compound's error** — accuracy plus
a free uncertainty signal `[V]`. *Cost:* dataset × N at train (cut epochs to compensate); N×
inference at test, which is nothing for a 10–50M model. *Does nothing for graph models* — they're
already permutation-invariant. Note augmentation saturates fast (ARC: 30 augmentations within 4% of
peak at 300 `[V]`).

**8. Active learning for the screening loop — greedy/UCB over the library, with a diversity cap.**
*Benefit:* the biggest single multiplier in the document: **MolPAL recovered 87.9% of the top-50k of
a 100M library from 2.4% of it** `[V]`; Thompson sampling found **90 of the top-100 from 0.1% of a
94M combinatorial library** `[V]`; Deep Docking reports up to 100× fewer dockings `[S]`. *Cost:*
N cheap retrainings; the oracle dominates. *Why it depends on #5:* greedy acquisition on an
uncalibrated surrogate exploits the model's own overconfidence. *And:* validate against experimental
assays, never only against the docking oracle you trained on.

**9. Distillation — pick the cheap oracle, not the glamorous one.**
*Benefit:* MTDP's ~20M student stayed **within ±1.5% of ESM2-650M/ProtT5-XL on classification and
regression at ~70% less compute** `[V]`; a structure-teacher → sequence-student affinity model
closed **~77% of the structure/sequence gap** (Pearson 0.481 vs 0.375 baseline) `[V]`. *Cost:*
dominated by teacher inference over the transfer set. **Start with Vina/Smina pseudo-labels on
millions of pairs, not Boltz-2** — Boltz-2 is MIT-licensed and FEP-competitive at >1000× FEP speed
`[S]`, so it's permitted and attractive, but generating 1–10M of its labels is itself a serious GPU
bill and its affinity head needs 3D input.

**10. LoRA on ESM-2 150M/650M for the protein side — if the protein representation is a bottleneck.**
*Benefit:* **a LoRA-tuned ESM-2 150M beat a frozen ESM-2 650M on several tasks** `[V]`, fine-tuning
almost always beat frozen embeddings, and the whole study ran **on one 24 GB A10G** `[V]`. *Cost:*
r=8–32 trains 1–15% of params; full FT of 650M should also fit with checkpointing + 8-bit Adam
(unverified by measurement). *Tuning that decides the outcome:* lr ~10× your full-FT value, and
**include the MLP layers — attention-only LoRA underperforms** `[V]`.

### Deliberately not in the shortlist, and why

- **Mixture-of-Experts** — all experts stay resident so you pay full memory for sparse FLOPs; one
  study reports MoEs *underperforming dense* at 10–20M active params until ~20× more data `[S]`; and
  routing is unstable at the batch sizes of tens an L4 gives you. **Skip.**
- **Recursive / looped depth** — the only clean fixed-budget win is on 1,000-example puzzles at 5M
  params `[V]`, Mixture-of-Recursions **underperforms vanilla at 135M** `[V]`, and ARC Prize showed a
  same-size vanilla transformer came within ~5 points of HRM `[V]`. **But take the two cheap pieces
  that transfer: deep supervision / iterative refinement during training (none→one refinement was
  worth 13 points `[V]`) and EMA (7.5 points `[V]`).** Those need no architecture change. Treat the
  loop itself as a cheap ablation (`for _ in range(K): z = block(z+x)`, never a 1-step gradient).
- **Curriculum learning** — random ordering matches curriculum and anti-curriculum on standard
  benchmarks `[V]`; the step-budget regime where it helps isn't yours. **Do the data cleaning
  instead** (ChEMBL `confidence_score`, `standard_relation = '='`, de-duplicate by median, never pool
  IC50 across assay types) and let the β-NLL head down-weight noise. That will beat any ordering.
- **PBT, evolutionary merging, zero-cost NAS** — PBT needs 10–40 concurrent trainings; merging needs
  many full evaluations; and NAS-Bench-Suite-Zero found **parameter count competitive with every
  zero-cost proxy** `[V]`. **Use ASHA/Optuna with a Hyperband pruner instead** — 50–100 configs for
  the price of 5–10 runs — plus free SWA/soups over the runs you already have.
- **Sophia, FP8, paged optimizers, test-time training** — no reproducible gain, wrong scale, or no
  molecular evidence.
- **Muon** — genuinely interesting, because the audit says its advantage is **largest at ~0.1B**
  `[V]`, but it is **untested on regression and GNNs**. Worth exactly one ablation once #1–#6 are in
  place, with AdamW retained for embeddings, the regression head and all 1D params.

### Verify before quoting

Everything marked `[M]` above, plus these specific known gaps (GROVER's compute and claim, MolCLR's
deltas, ChemBERTa-2's tables, Zaidi's QM9 numbers, Ramsundar's AUCs, Bjerrum's R², FAMO's Δm% and
Kurin's id have all now been read off the source PDFs and are marked `[V]`):

- **ADMET-AI** — the TDC leaderboard claim, the dataset count (41?) and the training/inference time
  could **not** be verified (bioRxiv 429, PMC captcha). The GitHub README confirms only that it is
  Chemprop models on TDC ADMET data, that the *published* paper is v1 (Chemprop + RDKit
  fingerprints), and that the repo's v2 uses Chemprop v2 **without** RDKit fingerprints.
- **Uni-Mol** — GPU-days and PDBbind/CASF affinity numbers (OpenReview captcha). Its ~45M parameter
  count is **inferred from a 181 MB checkpoint**, not stated. Same for ChemBERTa-77M-MTR's ~3.5M.
- **MolCLR** — GIN parameter count and GPU time are not in the paper.
- **arXiv:2605.10722** (ECFP pretraining) — pretraining corpus size, total parameters, and
  per-benchmark absolute R² deltas are not given in the text.
- **Frey et al., "Neural Scaling of Deep Chemical Models"** (*Nat. Mach. Intell.* 2023) — could not
  be fetched; no scaling exponents quoted anywhere above.
- Kramer et al.'s inter-lab variability figures; Deep Docking's recall; Scalia et al.'s ranking;
  Sorscher's scarce-data conditions; the model-soup gain.
- **Colab:** the Pro+ unit allowance (500 vs 600) and **every** compute-unit burn rate — read the
  live rate in the runtime picker on the day; whether Google AI **Pro** (not just Ultra) now carries
  background execution; and Colab Enterprise / Vertex runtime limits (pages unreachable).
- The arXiv id behind the small-scale-MoE-underperforms-dense claim (possibly 2605.11689).
