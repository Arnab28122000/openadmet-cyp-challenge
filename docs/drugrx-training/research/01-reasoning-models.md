# 01 — Reasoning-model architectures and agent harnesses (research brief)

Compiled 2026-10-04 from web research. Scope: tiny recursive reasoners, neuroevolution and model merging,
adaptive compute, small MoE, diffusion (language and molecular), and ARC-AGI-3 / Terminal-Bench harnesses.
It ends with recommendations for a small drug-discovery model trained on one L4 (24 GB) in about 24 h.

Verification key: **[v]** means fetched and checked in this session. **[k]** means taken from prior knowledge
(well-known paper, arXiv ID believed correct) and not re-fetched today. Check any [k] number before quoting it.
Several 2026 sources are leaderboards and blogs that move weekly. Their numbers are as of the fetch date.

---

## 1. Tiny recursive reasoning models

### 1.1 HRM — Hierarchical Reasoning Model (Sapient, 2025)
- Paper: *Hierarchical Reasoning Model*, Wang et al., arXiv **2506.21734** [k]. Repo `sapientinc/HRM` [k].
- 27M params. Two recurrent transformer modules: a slow high-level H module and a fast low-level L module.
  Uses deep supervision, one-step gradient approximation (no BPTT) and ACT halting trained by Q-learning.
- Trained on about 1000 examples per task, with no pretraining. Claimed results: Sudoku-Extreme 55%,
  Maze-Hard (30×30) 74.5%, ARC-AGI-1 40.3%, ARC-AGI-2 5.0% (numbers as quoted in the TRM paper) [v].
- **ARC Prize's independent analysis** ([arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)) [v]:
  - Verified semi-private scores: ARC-AGI-1 **32%**, ARC-AGI-2 **2%**.
  - The hierarchy barely matters. A plain transformer of the same size lands within about 5 pp.
  - The **outer refinement loop** drives the result. Going from 1 to 2 refinement loops adds about 13 pp,
    and training with refinement matters more than refining at inference.
  - Training only on the evaluation tasks gives 31%, against 41% with the extra data. So most of the
    performance comes from *seeing the eval tasks at train time* (transductive, per-task embeddings).
  - 300 augmentations perform about as well as 1000.
  - Puzzle-ID embeddings tie the model to tasks it has seen, so it cannot be applied zero-shot to a new task.

### 1.2 TRM — Tiny Recursive Model (Samsung SAIL Montréal, Oct 2025)
- *Less is More: Recursive Reasoning with Tiny Networks*, Alexia Jolicoeur-Martineau, arXiv
  **[2510.04871](https://arxiv.org/abs/2510.04871)** [v]. Repo
  [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
- **Architecture.** One 2-layer network (5–7M params) replaces HRM's two modules. It keeps a latent `z`
  and a current answer `y`, and loops:
  `for n steps: z = f(x, y, z)` (latent reasoning), then `y = f(y, z)` (answer update).
  That inner cycle runs T times; only the last cycle gets gradients, the earlier T−1 run under no-grad.
  The whole thing sits inside **deep supervision**: up to N_sup = 16 outer steps, each a fresh supervised
  step that carries (y, z) forward detached.
- **Recipe** (from the paper) [v]:

| Item | Value |
|---|---|
| Optimizer | AdamW, β = (0.9, 0.95) |
| LR | 1e-4 (embeddings 1e-2 on ARC); 2K-step warm-up |
| Weight decay | 1.0 (Sudoku, Maze); 0.1 (ARC) |
| Batch | 768 |
| Recursion | n = 6, T = 3 → 42 effective layers (HRM: 24) |
| Deep supervision | N_sup = 16 |
| Halting | single BCE "is the answer correct" head (drops HRM's Q-learning continue-loss and its 2nd forward pass) |
| EMA | 0.999 (removing it: Sudoku 87.4 → 79.9%) |
| Augmentation | Sudoku 1000 shuffles/example; Maze 8 dihedral; ARC 1000 (colour perms, dihedral, translations) |
| Compute | Sudoku-Extreme 60K epochs, <36 h on **1× L40S**; Maze 60K epochs, <24 h on 4× L40S; ARC 100K epochs, about 3 days on 4× H100 |

- **Results** (TRM vs HRM) [v]: Sudoku-Extreme **87.4%** (5M, MLP-mixer variant) vs 55.0%; Maze-Hard
  **85.3%** vs 74.5%; ARC-AGI-1 **44.6%** vs 40.3%; ARC-AGI-2 **7.8%** vs 5.0%.
- **Ablations worth copying** [v]:
  - On small fixed grids an MLP beats self-attention (87.4 vs 74.7).
  - 2 layers beat 4 (87.4 vs 79.5). More recursion with fewer params generalises better on tiny data.
  - One shared network beats two (87.4 vs 82.4).
  - Full backprop through the last recursion beats HRM's 1-step gradient (87.4 vs 56.5).

### 1.3 Follow-ups (late 2025 – 2026)
| Work | ID | Finding |
|---|---|---|
| Test-time adaptation of TRM | [2511.02886](https://arxiv.org/abs/2511.02886) [v] | Pretrain 7M TRM on 1,280 public ARC tasks (48 h, 4× H100, about 10% public eval). Then **full fine-tuning** (not LoRA, not embeddings-only) for 12,500 steps inside the Kaggle budget: 6.67% ARC-AGI-2 semi-private. |
| TRMs on ARC-AGI-1: inductive biases, identity conditioning, TTC | [2512.11847](https://arxiv.org/abs/2512.11847) [v] | Skeptical audit. A blank or random puzzle ID gives **0%**. 1000-sample augmentation voting adds about 11 pp pass@1. **Most accuracy arrives at the first recursion step** and saturates after a few latent updates. |
| Latent reasoning in TRMs is a policy-improvement operator | [2511.16886](https://arxiv.org/abs/2511.16886) [v] | Frames recursion as policy improvement. RL/diffusion-style training schemes give an **18× cut in forward passes** at equal accuracy, by removing "dead compute" steps. |
| Tiny Recursive Reasoning with Mamba-2 Attention Hybrid | [2602.12078](https://arxiv.org/abs/2602.12078) [v] | 6.83M params. ARC-AGI-1 pass@2 45.88 vs 43.88; +4.75 pp at pass@100; pass@1 unchanged. Better candidate diversity, same single-best answer. |
| Probabilistic TRM (PTRM) | [2605.19943](https://arxiv.org/abs/2605.19943) [v] | Sghaier, Parviz, Jolicoeur-Martineau. Inject **Gaussian noise at each deep-recursion step**, run parallel trajectories, pick one with the existing Q/halt head. No retraining. Sudoku-Extreme 87.4 → **98.75%**; Pencil-Puzzle-Bench 62.6 → 91.2%. |
| Tab-TRM (insurance pricing, tabular) | [2601.07675](https://arxiv.org/abs/2601.07675) [v-title] | TRM recursion applied to tabular regression, the nearest analogue to QSAR. |
| ARC Prize 2025 Technical Report | [2601.10904](https://arxiv.org/abs/2601.10904) [v] | Top Kaggle ARC-AGI-2 private score 24%. The theme of the year was the **"refinement loop"** (per-task iterative optimisation guided by feedback), which covers both program-synthesis loops and 7M-param weight-space refinement (TRM won a paper award). |

Community: an r/MachineLearning thread (Dec 2025) was broadly positive. It stressed the split between a
"reasoning" latent and an "answering" output, and that compute scarcity produced the idea
([snapshot](https://reddit.sentinel-team.org/posts/1pewrwt/snapshots/2025-12-06T03%3A21%3A09.333739Z)) [v].
Reproductions: [AntonioRoye/TinyRecursiveModels](https://github.com/AntonioRoye/TinyRecursiveModels), and ARC
Prize's verification checkpoint on Hugging Face, `arcprize/trm_arc_prize_verification` [v].

### 1.4 Transfer to molecular property and binding prediction
- **Recursion is already common in molecular modelling.** Message passing is a recurrent refinement over
  a graph, and AlphaFold2/Boltz "recycling" (feeding the predicted structure back 3+ times) is TRM's
  outer loop. The cheap experiment is to share weights across K message-passing / pair-update blocks,
  recurse them, and add deep supervision. This is likely to help when labelled data is small (ADMET sets
  of 1k–10k molecules), because that is TRM's regime: little data, so heavy weight-sharing and strong
  regularisation (wd = 1.0, EMA) beat extra layers.
- **What does not transfer.**
  - TRM's ARC wins are **transductive**: per-task puzzle-ID embeddings, training on the eval tasks'
    demonstration pairs, and 1000× augmentation with voting ([2512.11847](https://arxiv.org/abs/2512.11847)).
    Property prediction on new scaffolds is inductive, and that crutch is not available.
  - Augmentation counts are not comparable. Molecules have SMILES randomisation, conformers, and
    E(3)/permutation symmetry, which equivariant models get for free. There is no "colour permutation"
    trick of 1000× strength.
  - The audit finding that most accuracy appears at recursion step 1 means the claimed gain from
    recursion depth needs an ablation on your own data before you rely on it.
- **What does transfer well.**
  - Iterative **pose / structure refinement**: y = coordinates, z = latent, with deep supervision on
    RMSD at each step.
  - **Halting as a calibrated confidence head**: the BCE "is the answer right" head becomes a pose or
    affinity confidence, much like DiffDock's confidence model.
  - **PTRM-style noise + Q-head selection** as cheap test-time compute for generation and docking.
  - **EMA**.

---

## 2. Genotype→phenotype encodings, neuroevolution, evolutionary merging

| Method | Ref | What it is | Small-scale practical value |
|---|---|---|---|
| NEAT | Stanley & Miikkulainen 2002, *Evolutionary Computation* 10(2) [k] | Direct genome of nodes and connections. Historical markings allow crossover; speciation protects new structure. | Useful for tiny control policies, not for million-parameter networks. |
| HyperNEAT / CPPN | Stanley et al. 2009, *Artificial Life* 15(2) [k] | **Indirect** genotype→phenotype mapping: a CPPN paints the connection weights of a large substrate from geometry. | Elegant compression, but beaten by SGD on supervised tasks. Historical interest for "developmental encodings". |
| Weight Agnostic NNs | Gaier & Ha, arXiv 1906.04358 [k] | Search topology with one shared weight. | Shows how strong architectural priors can be. Not a training method. |
| Regularized evolution (AmoebaNet) | Real et al., arXiv 1802.01548 [k] | Aging-evolution NAS. | About 3k GPU-days at the time. Use only as an outer-loop pattern. |
| Evolution Strategies | Salimans et al., arXiv 1703.03864 [k]; ES for LLM fine-tuning, Qiu et al. 2025 (arXiv 2509.24372) [k] | Gradient-free population perturbation. | Good for non-differentiable rewards (docking scores, synthesisability). |
| **Evolutionary model merging (Sakana)** | Akiba, Shing, Tang, Sun, Ha — arXiv **[2403.13187](https://arxiv.org/abs/2403.13187)** [v] | CMA-ES searches (a) per-layer merge weights in **parameter space** (TIES/DARE + task arithmetic) and (b) layer ordering in **data-flow space**. EvoLLM-JP 7B reaches 55.2% on Japanese MGSM and beats some 70B models. | **High practical value and cheap.** Fitness is only an eval forward pass, with no training. Fine-tune 3–6 small specialists from one parent (different assays, seeds or splits), then evolve merge coefficients against a validation set. |
| Darwin Gödel Machine (Sakana) | arXiv 2505.22954 [k] | Open-ended evolution of agent *code* with an archive. | The pattern behind autoresearch loops: keep an archive of variants and select by measured score. |

"Neuro-Genetic Architecture" is not a single named method in the literature. Searches return
genotype→phenotype *biology* networks (omnigenic NN on UK Biobank, G-P Atlas, GenNet) and generic GA+NAS
systems (e.g. arXiv 2503.10869) [v]. Read it as "a GA over an architecture genome, with SGD for weights".
**Verdict at <100M params on one GPU:**
- Do not evolve weights.
- *Do* evolve a small hyper-parameter / architecture genome (depth, recursion n/T, width, expert count,
  augmentation strength) with a cheap proxy run per individual.
- Use Sakana-style merging of fine-tuned siblings as a free final boost.

---

## 3. Dynamic / adaptive compute

| Method | Ref | Key mechanism | Notes for small models |
|---|---|---|---|
| ACT | Graves 2016, arXiv 1603.08983 [k] | Halting unit; stops when the cumulative halting probability passes 1−ε; ponder-cost penalty τ. | Sensitive to τ. TRM's simple BCE halt head is the modern, cheaper version. |
| Universal Transformer | Dehghani et al., arXiv 1807.03819 [k] | Weight-shared depth recurrence + ACT. | The ancestor of TRM and looped transformers. |
| PonderNet | Banino et al., arXiv 2107.05407 [k] | Probabilistic halting (geometric prior λ_p, KL regulariser); unbiased gradient. | More stable than ACT and extrapolates to more steps at test time. |
| Mixture-of-Depths | Raposo et al., arXiv 2404.02258 [k] | Per-block top-k token router; the rest skip the block. | Saves FLOPs at fixed quality. Little benefit for molecules (about 50 heavy atoms), where sequences are short. |
| Mixture-of-Recursions | Bae et al., arXiv 2507.10524 [k] | Per-token recursion depth over a shared block. | MoD crossed with TRM. |
| Early exit (CALM, LayerSkip) | Schuster et al. 2207.07061; Elhoushi et al. 2404.16710 [k] | Confidence-based exit from intermediate layers. | Only pays off at inference volume, such as virtual screening of millions of compounds. |
| Recurrent-depth latent reasoning (Huginn) | Geiping et al., arXiv 2502.05171 [k] | 3.5B model, with test-time compute scaled by unrolling the recurrent core more times. | Shows recursion depth works as a test-time knob. |
| Compute-optimal TTS | Liu et al., arXiv [2502.06703](https://arxiv.org/abs/2502.06703) [v] | Best-of-N / beam search with a PRM. 1B beats 405B on MATH-500; 0.5B beats GPT-4o. | The optimal strategy depends on policy, verifier and difficulty. A good **verifier** is the lever. |
| PTRM | [2605.19943](https://arxiv.org/abs/2605.19943) [v] | Noise + parallel trajectories + learned selector. | The cheapest TTC for a TRM-style model. |

Takeaway: for a small model, test-time compute is only as good as its **selector or verifier**: a halt/Q
head, a confidence model, a physics check (PoseBusters), or a docking score. Train one alongside the main model.

---

## 4. Mixture of Experts at small scale

- **Load balancing with an auxiliary loss.**
  - Switch Transformer (arXiv 2101.03961) [k]: `α · N · Σ f_i · P_i`, with α ≈ 0.01.
  - ST-MoE (arXiv 2202.08906) [k] adds a **router z-loss** (about 1e-3) for numerical stability.
- **Aux-loss-free balancing.**
  - DeepSeek: Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for MoE*, arXiv
    **[2408.15664](https://arxiv.org/abs/2408.15664)** [v].
  - A per-expert bias b_i is added to the routing scores *only for top-K selection*; the gate weights
    use the unbiased scores. After each step, `b_i ← b_i + u · sign(mean_load − load_i)`.
  - It avoids the interference gradients an auxiliary loss adds. Validated up to 3B params / 200B tokens,
    with variants at the 100M scale.
  - DeepSeek-V3 (arXiv 2412.19437) [k] uses an update speed of about 0.001, plus a tiny sequence-wise
    aux loss (α ≈ 1e-4) as a safety net.
  - Theory: arXiv 2512.03915 [v-title]. Alternative: φ-Balancing, arXiv 2605.15403 [v-title].
- **Design patterns.**
  - DeepSeekMoE (arXiv 2401.06066) [k]: fine-grained experts plus 1–2 **shared experts**.
  - Expert-choice routing (arXiv 2202.09368) [k]: balanced by construction, but leaks future tokens
    when decoding autoregressively. That is fine for property prediction.
  - Soft MoE (arXiv 2308.00951) [k]: no discrete routing, so the most stable choice at tiny scale.
  - OLMoE (arXiv 2409.02060) [k]: 64 experts, top-8, 1B active / 7B total. A fully open reference recipe.
- **At under 100M params.** Gains are real but modest, and token-routed MoE mostly pays off with lots of
  diverse data. Typical tiny settings are 4–8 experts, top-1/2, with 1 shared expert. On one L4, MoE costs
  memory, not FLOPs; at 50M total that is negligible.
- **For molecules,** route at the **molecule or task level** rather than per token: one expert per assay
  family or scaffold cluster. Examples:
  - ASE-Mol (substructure-aware experts with BRICS), arXiv [2504.05844](https://arxiv.org/abs/2504.05844).
  - HierMolMoE (atom / motif / global experts), [ICLR 2025](https://iclr.cc/virtual/2025/37801).
  - MI-MoE (multiscale interaction experts), [arXiv 2601.12637](https://www.alphaxiv.org/abs/2601.12637).
  - GNN-MoCE (collaborative experts for multitask ADMET), arXiv [2312.03292](https://arxiv.org/abs/2312.03292).

  All [v].
- Bias-based balancing transfers directly to task- or scaffold-level routing.

---

## 5. Diffusion reasoning, and diffusion for molecules

### 5.1 Diffusion language models
| Model | Ref | Notes |
|---|---|---|
| LLaDA | Nie et al., arXiv 2502.09992 [k] | 8B masked-diffusion LM trained from scratch (about 2.3T tokens); competitive with LLaMA3-8B; beats AR on reversal tasks. |
| Dream 7B | Ye et al., arXiv 2508.15487 [k] | Initialised from Qwen2.5-7B AR weights. The strongest open full-sequence dLLM on MMLU, HellaSwag, MBPP and Sudoku ([analysis](https://arxiv.org/html/2606.19475v2)) [v]. |
| Mercury (Inception Labs) | arXiv 2506.17298 [k] | Commercial dLLM; Mercury Coder at about 1,100 tok/s on an H100. Pitched as a fast reasoning model. |
| d1 / diffu-GRPO | Zhao et al., arXiv 2504.12216 [k] | SFT + diffu-GRPO (one-step log-prob estimate with random prompt masking) on LLaDA-8B. Gains on GSM8K, MATH, Countdown, Sudoku. |
| GDPO | arXiv [2510.08554](https://arxiv.org/abs/2510.08554) (ICLR 2026) [v] | Group Diffusion Policy Optimization; outperforms diffu-GRPO on reasoning, planning and coding. |
| d2 | arXiv 2509.21474 [v-title] | Trajectory-likelihood estimation for RL on dLLMs. |
| Discrete diffusion for planning | Ye et al., *Beyond Autoregression*, arXiv 2410.14157 [k] | Small masked-diffusion models solve Sudoku and planning tasks where AR models of the same size fail. Relevant to TRM-like constraint problems. |
| Tooling | [ZHZisZZ/dllm](https://github.com/ZHZisZZ/dllm) [v] | Simple dLLM training library; diffu-GRPO on LLaDA and Tiny-A2D; Fast-dLLM caching (2026). |

**Link to TRM.** Masked diffusion and TRM both refine a full answer iteratively, and
[2511.16886](https://arxiv.org/abs/2511.16886) borrows diffusion training schemes to make TRM 18× cheaper.
For non-sequential outputs such as molecules or poses, "recursive refinement" and "diffusion denoising" are
the same family. Diffusion adds a principled noise schedule and sampling diversity, which is what PTRM
bolts on.

### 5.2 Diffusion for molecules [k unless noted]
| Model | Ref | Task | Notes |
|---|---|---|---|
| EDM | Hoogeboom et al., arXiv 2203.17003 | E(3)-equivariant 3D molecule generation (QM9, GEOM) | Baseline EGNN denoiser; trains on 1 GPU. |
| GeoDiff | Xu et al., arXiv 2203.02923 | Conformer generation | Roto-translation invariant. |
| DiffDock | Corso et al., arXiv 2210.01776 | Blind docking (diffusion over translation, rotation and torsions) + a confidence model | About 38% top-1 <2 Å on PDBBind time-split. |
| DiffDock-L | Corso et al., arXiv 2402.18396 | Larger, retrained with augmentation; DockGen benchmark | Best blind docking on PDBBind and DockGen at release. |
| PoseBusters | Buttenschoen et al., arXiv [2308.05777](https://arxiv.org/abs/2308.05777) | Physical-validity checks | Many DL docking poses fail physics checks. Use it as the verifier. |
| DiffSBDD | Schneuing et al., arXiv [2210.13695](https://arxiv.org/abs/2210.13695) [v] | Pocket-conditioned ligand generation (EGNN) | Leads on Vina score and diversity in comparisons [v]. |
| TargetDiff | Guan et al., ICLR 2023, arXiv 2303.03543 | Joint coordinates + atom types; target-aware | Leads on "high affinity %" [v]. |
| DecompDiff | Guan et al., ICML 2023 | Decomposed arm/scaffold priors | Better binding than TargetDiff on CrossDocked. |
| MolDiff | Peng et al., ICML 2023, arXiv 2305.07508 | Atom-bond inconsistency fix | Fewer invalid or strained molecules. |
| 2026 frontier | EvoEGF-Mol (arXiv [2601.22466](https://arxiv.org/abs/2601.22466)) [v-title]; ShallowBench (arXiv [2606.06717](https://arxiv.org/abs/2606.06717)) [v-title] | Geodesic flow SBDD; benchmark for shallow pockets | Flow matching is now replacing DDPM in SBDD. |

The standard SBDD dataset is CrossDocked2020 (about 100k training pairs). These EGNN models have 2–10M params
and train on a single 24 GB GPU in about 1–3 days. A 24 h L4 run is enough for a reduced-step reproduction,
but not for full SOTA.

---

## 6. ARC-AGI-3 and Terminal-Bench: what the winners do

### 6.1 ARC-AGI-3 (interactive games, launched 2026)
- **Format** ([technical report, arXiv 2603.24621](https://arxiv.org/html/2603.24621v1)) [v]:
  - Turn-based 64×64 grid games with 16 colours and no instructions. Action space is a subset of 5 keys,
    Undo, and cell click.
  - Score is RHAE (relative human action efficiency): agent actions over the second-best human,
    **squared**, with later levels weighted more. Humans solve 100%; frontier models scored <1% at launch.
- **2025 preview competition.** StochasticGoose 12.58%: a CNN + RL model that predicts which actions
  change the frame. Blind Squirrel 6.71%: a directed state graph built from observed frames. Both rely on
  search and exploration, not LLMs [v].
- **Frontier models** ([ARC Prize blog, Sep 2026](https://arcprize.org/blog/astra)) [v]:
  - GPT-6 Astra scores **62.7%** semi-private with the *Standard* harness ($26K; it may carry forward notes
    it chooses to keep). With the *Provider Adapter* harness (preserved opaque reasoning state + compaction)
    it scores **99.9%** ($19K), runs 3.66× faster and uses 49% fewer tokens.
  - Astra invented its own algebraic shorthand for game state and built game-specific tools (maze solvers).
  - The official leaderboard on 29 Sep 2026 had Astra 62.7%, Claude Opus 5 30.2%, Gemini 3.8 Flash 10.4%
    ([benchlm](https://benchlm.ai/benchmarks/arcagi3)) [v-search].
- **Harness results on the public set** [v]:
  - **Retrodict** ([github.com/ryanbbrown/Retrodict](https://github.com/ryanbbrown/Retrodict)): gpt-5.6-sol
    on the "ThinHarness". **99.86% mean RHAE**, all 183 levels of 25 games, 7,703 actions, 660M tokens,
    **$654**. That is 6× fewer tokens than the same-model baseline (98.97% at 3.97B tokens).
    The only point ahead of it on the cost frontier is "Tycho" (100% at about $2,986).
  - **NVIDIA AVO** (Agentic Variation Operators; Aug 2026) took Claude Opus 5 from 30% to 100%. It adds
    memory controls and a **supervisory agent** that redirects the main agent when it is stuck
    ([report](https://cryptobriefing.com/nvidia-ai-harness-over-model-research/)) [v].
  - NVIDIA's NOOA harness write-up
    ([blog](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/))
    attributes **+8.5 pp** to a world-modelling skill over a hypothesis-driven baseline, and **+11.8 pp**
    to a structured memory subsystem over file-based notes. GPT-5.6-sol fleet: 85.1% at about $13 per game.
  - Schema harness: about 99% public ([schema-harness.github.io](https://schema-harness.github.io/)) [v].
- **Kaggle competition** (open models, single GPU, no internet) [v]:
  - **Milestone #1 (30 Jun 2026)** went to Tufa Labs' **"The Duck"**
    ([write-up](https://tufalabs.ai/research/duck-harness/),
    [code](https://github.com/Tufalabs/duck-harness)). Qwen3.6-27B-FP8 plays inside a **Python REPL**:
    game observations are Python variables, the model writes code, takes an action and repeats. It sees an
    image + ASCII grid + segmentation tool. "**Infinite play via eviction**" drops the oldest messages and
    keeps only the system prompt plus recent turns.
  - Tufa's stated finding: *hand-crafted tools hurt; let the model improvise*. "Solvability depends on the
    model; cost is dictated by the harness."
  - 2nd and 3rd went to Gemma-4-31B vision agents returning JSON actions, with reflection memory and a click
    fallback. The 3rd-place run **disabled most of its own extra machinery**
    ([ARC Prize blog](https://arcprize.org/blog/arc-prize-2026-milestone-1)).
  - Tufa Labs leads the Kaggle board (52.51 at fetch time). Milestone #2 closed 30 Sep 2026; its write-up
    was not yet published when this was compiled.

**Retrodict's "thin harness + lab notebook" pattern, in detail** [v]:
1. **Every observation is appended to `log.txt`.** The agent reasons over the log with code (a sandboxed
   Python with numpy/scipy/networkx), not over images.
2. **Retrodiction.** Every hypothesis must pass a replay against the recorded history before it may spend
   a live action. Being wrong in replay costs nothing.
3. **Forward simulation.** Each action carries its *predicted* board state. The runner checks it and halts
   on a mismatch, returning a diff, so a wrong model costs one action rather than a whole plan.
4. **`playbook.md`** survives context resets. It holds a *working model* (rules, each marked verified-against-log
   or assumed) and *working memory* (current attempt, plan, what has been ruled out). Context is hard-reset
   at 150k tokens and resumed from playbook + log.
5. **Escalation.** After 300 actions on one level, the agent must build an executable simulator and verify it
   against every recorded frame.
6. The system prompt is about 150 lines and game-agnostic. Tools are only read, write, edit, search and python.

### 6.2 Terminal-Bench 2.0 / 2.1
- **Benchmark** ([arXiv 2601.11868](https://arxiv.org/abs/2601.11868v1)) [v]: 89 hard tasks in Docker,
  each with a human-written solution and tests. At release, frontier agents scored <65%. 2.1 is a
  re-validated revision.
- **Leaderboard** ([Snorkel mirror of TB 2.0](https://snorkel.ai/leaderboard/terminal-bench-2-0/),
  fetched 2026-10-04) [v]:
  1. NexAU-AHE + GPT-5.5 — 84.7
  2. LemonHarness — 84.5
  3. Capy + GPT-5.5 — 83.1
  4. Codex CLI + GPT-5.5 — 82.2
  5. Polaris — 82.2
  6. WOZCODE + Opus 4.7 — 80.2
  7. TongAgents + Gemini 3.1 Pro — 80.2
  8. SageAgent — 78.4
  9. **Factory Droid** + GPT-5.3-Codex — 77.3

  There are 142 submissions from about 23 scaffolds, and the same model appears several times with
  different harnesses ([Morph](https://www.morphllm.com/terminal-bench-2)).
- **Terminus 2** is the benchmark's own neutral reference agent: a model driving a single tmux session by
  keystrokes. Gemini 3.1 Pro + Terminus 2 scored 70.7 on TB 2.1 [v-search].
- **Warp** led the early TB 0.1.1 at 52%, 20 pp ahead of the field, with three ingredients [v-search]:
  long-running command control, an **updatable todo list**, and a model fallback.
- **StateM** ([arXiv 2608.15089](https://arxiv.org/abs/2608.15089), Aug 2026) [v, read the PDF]:
  - Reaches **95.3% raw** on TB 2.1 with GPT-5.6 Sol, and 92.1% with GPT-5.5, against an 83.1% reference.
  - Lifts DeepSeek-V4 Flash from 82.7 to 88.1 for less than $38 of adaptation (about $15 total eval spend).
  - Mechanism: a human- and agent-editable **YAML runbook** of states, with valid transitions, exit checks,
    hooks and recovery rules. Each state is a "context-and-contract boundary": entering it refreshes the
    relevant instructions; leaving it requires passing a check.
  - It runs **failure-driven harness optimisation**: after each run, failures are classified and turned
    into versioned runbook rules, which closes the "procedural-memory gap".
- **Tmax** ([arXiv 2606.23321](https://arxiv.org/abs/2606.23321)) [v]: a 9B model reaches **27%** on
  TB 2.0 with a simple, outcome-only RL recipe plus large synthetic environment generation (difficulty
  control, personas, verifier diversity). With small models, data and verifiers beat scaffolding cleverness.

### 6.3 The common pattern, and the transferable lesson for an autonomous research loop
Across Retrodict, Duck, AVO/NOOA, StateM and Warp:
1. **Thin, generic harness; capable model; few tools.**
   - Duck and Retrodict say plainly that hand-built, task-specific tools hurt.
   - Give the agent a shell or a Python REPL plus file read/write, and stop there.
2. **Externalised, durable state rather than chat history.**
   - Use an append-only **raw log** (ground truth) and a curated **playbook / notebook** that marks each
     belief as *verified* or *assumed*.
   - Context is disposable (evict, compact, hard reset); the files are not.
3. **Hypotheses are tested before they cost anything.**
   - Retrodict replays a hypothesis against the log before acting; StateM checks transitions; Retrodict
     attaches a predicted outcome to every action.
   - The research-loop equivalent: *every experiment gets a written prediction before it runs*, and every
     claim is checked against logged results.
4. **Stuck detection and escalation.**
   - Retrodict escalates after 300 actions; AVO has a supervisor; Warp has a fallback model.
5. **Failures become procedure.**
   - StateM's runbook rules and Retrodict's playbook record what to do differently next time, not only
     what happened.
6. **The harness sets the cost; the model sets what is solvable.**
   - Retrodict used 6× fewer tokens than a same-model baseline; the Provider Adapter used 49% fewer tokens.

---

## 7. Implications for training a small drug-discovery model on one L4 GPU (24 GB) per run, about 24 h

**Compute budget.** An L4 gives roughly 30 TFLOPS BF16 dense in practice (lower sustained) and has 24 GB.
24 h is about the budget TRM needed for Sudoku on one L40S, which is roughly 2× an L4. So plan for models of
**5–50M params** and datasets of **10^4–10^6 examples**.

1. **Start with a TRM-style recursive refiner over a molecular encoder, and ablate the recursion
   honestly.**
   - Share one 2-layer block (an EGNN, GPS or pair-update block), recurse n = 4–6 × T = 2–3, and wrap it
     in deep supervision (N_sup of 4–8, not 16, to fit 24 h).
   - Copy TRM's recipe: AdamW β = (0.9, 0.95), lr 1e-4 with warm-up, **heavy weight decay (0.1–1.0)**,
     **EMA 0.999**, BCE halt/confidence head.
   - Run the 1-step vs full-recursion ablation on day one. [2512.11847](https://arxiv.org/abs/2512.11847)
     found most gains arrive at step 1.
2. **Use the recursion for structure-shaped outputs (pose refinement, affinity with pose recycling) rather
   than scalar ADMET.** That is where "refine an answer" has real meaning; it is AlphaFold's recycling.
   For scalar properties, a well-regularised GNN or tabular-ICL baseline (TabPFN-style,
   [2606.31126](https://arxiv.org/abs/2606.31126)) is the bar to beat.
3. **Train a verifier alongside the model and spend test-time compute through it.**
   - Use a halt/Q or confidence head plus PoseBusters checks; add a docking score for generators.
   - Then use PTRM-style noise + parallel trajectories + selection
     ([2605.19943](https://arxiv.org/abs/2605.19943)). That gave Sudoku 87 → 99% with no retraining.
     Best-of-N with a good verifier is how 1B beat 405B ([2502.06703](https://arxiv.org/abs/2502.06703)).
4. **Augmentation instead of data.**
   - SMILES enumeration, conformer sampling, and random rotations for non-equivariant models.
   - Test-time augmentation voting is worth about 10 pp on ARC. Expect a smaller but real effect here.
   - Split by scaffold. TRM's transductive tricks (per-task IDs, training on eval demos) are leakage in
     QSAR.
5. **MoE only at the task or scaffold level, and only aux-loss-free.**
   - If you multitask many assays, use 4–8 experts plus 1 shared expert, routed per molecule or per assay.
   - Balance with DeepSeek's bias update (u ≈ 1e-3; [2408.15664](https://arxiv.org/abs/2408.15664)) and a
     z-loss.
   - Skip token-level MoE and MoD. Molecules are too short for them to pay off.
6. **Use generative diffusion / flow only where it is the task (pocket-conditioned generation, docking),
   and as a reduced reproduction.**
   - DiffSBDD/TargetDiff-scale EGNNs (2–10M params) on CrossDocked fit on an L4. Expect a shortened
     schedule in 24 h.
   - Prefer flow matching or few-step samplers. Gate every output with PoseBusters.
7. **Use evolution in the outer loop, not on weights.**
   - Run a small GA or CMA-ES over a config genome (recursion depth, width, wd, augmentation strength,
     N_sup) with 1–2 h proxy runs.
   - After fine-tuning sibling models, do a **Sakana-style evolutionary merge**
     ([2403.13187](https://arxiv.org/abs/2403.13187)). It costs only forward passes on a validation split.
8. **Run the research loop like Retrodict and StateM.**
   - Keep the harness thin: shell + Python + files.
   - Keep an append-only `runs/` log of every config, metric and seed as ground truth.
   - Keep a curated `NOTEBOOK.md` / playbook that marks each claim *verified (run id)* or *assumed*.
   - Write the **expected metric down before each run** and diff it against the result afterwards.
   - Escalate when stuck: after K runs with no gain, the loop must change one assumption, not tune
     another knob.
   - Turn each failure into a written rule in the runbook.
