# 10 — Tiny recursive / hierarchical reasoning models and adaptive computation depth

Compiled 2026-10-05. Scope: exact mechanisms of TRM and HRM, the two classical adaptive-depth
methods (Graves ACT, PonderNet), the ARC Prize ablation of HRM, and an honest search for evidence
that any of it transfers to small tabular / molecular regression. Target of application: the
OpenADMET CYP450 task (4 pIC50 endpoints, 6,525 primary labels over 4,905 compounds, ~38k auxiliary
labels across 14 heads, ~2,500-dim fixed feature vector, macro-averaged **ST-RAE** — error is the
distance to the nearest credible-interval bound and **zero inside the interval**).

Verification key: **[verified]** = read in this session in the paper PDF, the arXiv-HTML algorithm
box, or the repo source. **[assumed]** = my inference or well-established background not re-checked
today. Where a number is quoted from another paper's table rather than its own experiment, it says so.

---

## 0. The verdict up front

1. **TRM and HRM are not two architectures; they are one architecture plus a training harness,
   and the harness is doing the work.** The ARC Prize team's controlled ablation found a plain
   transformer of the same size lands **within ~5pp** of HRM, and that going from 1 to 2 outer
   refinement loops is worth **~+13pp** [verified]. The hierarchy is not the mechanism.
2. **The one genuinely transferable idea is the outer loop with per-step supervision and a learned
   "am I done" head** — which on this task is a free intra-model ensemble with a learned selector,
   not "reasoning".
3. **There is no published evidence of recursive-reasoning or adaptive-depth models on molecular
   property regression.** arXiv returns **zero** results for recursive-reasoning / adaptive-depth /
   dynamic-depth / deep-equilibrium ∧ molecular-property / QSAR / ADMET, and zero for
   ponder/halting ∧ bioactivity/IC50 [verified, queries in §6.4].
4. **The one tabular TRM result that exists (Tab-TRM) is on 9 features and ~680k rows.** We have
   2,500 features and 6.5k rows. Its margin over the previous best compact model is **0.33%
   relative**, which maps to ~0.0014 ST-RAE here — two orders below this task's detectability.
5. **Pure weight-tied recursion on tabular data has already been measured as a null twice**: once by
   this programme (H2, gap 0.0017) and once in the literature (looped transformers, **−0.39% ±
   0.80% over 27 tabular datasets, winning 14/27**) [verified, §6.2].

So: run it, but run it as a *selector* experiment, and register the null in advance. §7 has the
minimal implementation, §8 the expected value, §9 the four ways it goes silently wrong.

---

## 1. TRM — Tiny Recursive Model (Jolicoeur-Martineau, Samsung SAIL Montréal, Oct 2025)

*Less is More: Recursive Reasoning with Tiny Networks*, arXiv
[2510.04871](https://arxiv.org/abs/2510.04871). Repo
[`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

### 1.1 Architecture and recursion structure

One network, two latents. `y` is the **current answer** (embedded solution) and `z` is a **latent
scratchpad**. The *same* network `net` is applied in two roles: n times to refine `z` given
`(x, y, z)`, then once to refine `y` given `(y, z)`.

Verbatim from the paper's algorithm boxes [verified]:

```python
def latent_recursion(x, y, z, n=6):
    for i in range(n):
        z = net(x, y, z)      # n latent reasoning steps
    y = net(y, z)             # 1 answer update
    return y, z

def deep_recursion(x, y, z, n=6, T=3):
    with torch.no_grad():                        # T-1 cycles carry NO gradient
        for j in range(T-1):
            y, z = latent_recursion(x, y, z, n)
    y, z = latent_recursion(x, y, z, n)          # last cycle IS differentiated, fully
    return (y.detach(), z.detach()), output_head(y), Q_head(y)
```

Three nesting levels, and it matters to keep them straight:

| Level | Symbol | Default | What it is |
|---|---|---|---|
| inner | `n` (repo: `L_cycles`) | 6 | latent `z` updates per answer update |
| middle | `T` (repo: `H_cycles`) | 3 | answer `y` updates per forward pass; only the **last** is differentiated |
| outer | `N_sup` (repo: `halt_max_steps`) | 16 | deep-supervision segments; each is a **separate optimiser step**, carry detached |

Effective depth = `2 layers × (n+1) × T = 2 × 7 × 3 = 42` against HRM's 24, at 5–7M params
vs 27M [verified].

**Confirmed against the repo source** (`models/recursive_reasoning/trm.py`), where the naming is
misleading: `z_H` is the *answer* y and `z_L` is the *latent* z, and `L_level` is the one shared
network [verified]:

```python
with torch.no_grad():
    for _H_step in range(self.config.H_cycles-1):
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)
for _L_step in range(self.config.L_cycles):            # 1 with grad
    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
z_H = self.L_level(z_H, z_L, **seq_info)
new_carry = InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
output   = self.lm_head(z_H)[:, self.puzzle_emb_len:]
q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
```

Note `x` enters as an **additive injection** into the z-update (`z_H + input_embeddings`), not a
concatenation [verified from source].

### 1.2 Training loop and loss

Verbatim from the paper [verified]:

```python
for x_input, y_true in train_dataloader:
    y, z = y_init, z_init
    for step in range(N_supervision):
        x = input_embedding(x_input)
        (y, z), y_hat, q_hat = deep_recursion(x, y, z)
        loss  = softmax_cross_entropy(y_hat, y_true)
        loss += binary_cross_entropy(q_hat, (y_hat == y_true))
        loss.backward()
        opt.step()
        opt.zero_grad()
        if q_hat > 0:
            break
```

Two things here are load-bearing and easy to miss:

- **`opt.step()` is inside the supervision loop.** One batch produces up to `N_sup` optimiser
  steps. A comparison at equal *epochs* therefore gives the recursive arm up to 16× the parameter
  updates. (See §9.5 — this is the confound H12 already caught in another guise.)
- **The halt target is `(y_hat == y_true)`, exact-match correctness, computed under no-grad.** The
  repo's loss head confirms the weighting [verified, `models/losses.py`]:

```python
lm_loss      = (self.loss_fn(outputs["logits"], labels, ...) / loss_divisor).sum()
q_halt_loss  = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"],
                                                  seq_is_correct.to(...), reduction="sum")
return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), ...
```

so **L = L_task + 0.5·BCE(q_halt, 1{ŷ = y}) + 0.5·BCE(q_continue, target)**, where
`seq_is_correct` is computed inside a `torch.no_grad()` block, and `q_continue_loss = 0` unless
`target_q_continue` is present.

### 1.3 Halting: TRM drops HRM's Q-learning

TRM keeps HRM's ACT *scaffolding* (`halt_max_steps`, `halt_exploration_prob`) but defaults to a
single-head version. From the config class [verified, source]:

```python
no_ACT_continue: bool = True  # No continue ACT loss, only use the sigmoid of the halt
                              # which makes much more sense
```

and in the loss file, next to the continue branch: `# Alexia: This fits Q-learning, but seems
totally unecessary` [verified]. Consequences:

- **One** forward pass per segment instead of HRM's two (HRM needs a second pass to bootstrap the
  continue target).
- Halting rule in training is `halted |= (q_halt_logits > 0)` when `no_ACT_continue`, else
  `q_halt > q_continue` [verified, source].
- `halt_exploration_prob = 0.1`: with that probability a per-sample `min_halt_steps` is drawn
  uniformly from `[2, halt_max_steps]` and halting is suppressed until then — the forced-exploration
  trick inherited from HRM [verified, source].
- The paper's own ablation says enabling full ACT **costs** accuracy: TRM 87.4 → **86.1 with ACT**,
  at 2 forward passes instead of 1 [verified].

### 1.4 Recipe (paper table + `config/arch/trm.yaml` + `config/cfg_pretrain.yaml`) [verified]

| Item | Value |
|---|---|
| Layers | `L_layers: 2`, `H_layers: 0` (ignored) |
| Hidden | `hidden_size: 512`, `num_heads: 8`, `expansion: 4`, RMSNorm, RoPE |
| Params | 5M (MLP-mixer variant), 7M (self-attention variant) |
| `H_cycles` (T) / `L_cycles` (n) | 3 / 6 |
| `halt_max_steps` (N_sup) | 16 |
| `halt_exploration_prob` | 0.1 |
| Optimiser | AdamW, β = (0.9, 0.95), `weight_decay: 0.1` (1.0 for Sudoku/Maze per paper) |
| LR | `1e-4`, `lr_warmup_steps: 2000`, `lr_min_ratio: 1.0` (i.e. **flat**, no decay) |
| Puzzle-embedding LR | `1e-2` |
| Batch | `global_batch_size: 768` |
| Epochs | `100000` (ARC); Sudoku/Maze 60K |
| EMA | **`ema: False` in the committed config**, `ema_rate: 0.999` when on |
| Loss | `stablemax_cross_entropy` (a numerically-tame softmax surrogate, `models/losses.py`) |
| Compute | Sudoku-Extreme < 36 h on 1× L40S; ARC ~3 days on 4× H100 |

The committed `ema: False` contradicts the paper's "no EMA: 87.4 → 79.9" ablation. Read it as: EMA
helps on Sudoku, and was not left on by default [verified, both sources].

### 1.5 Ablations (Sudoku-Extreme) — the part worth copying

Verbatim from the paper's ablation table [verified]:

| Variant | Acc (%) | Depth | Fwd passes | Params |
|---|---|---|---|---|
| HRM | 55.0 | 24 | 2 | 27M |
| **TRM (T=3, n=6)** | **87.4** | 42 | 1 | 5M |
| w/ ACT (Q-continue) | 86.1 | 42 | 2 | 5M |
| w/ separate f_H, f_L | 82.4 | 42 | 1 | 10M |
| **no EMA** | **79.9** | 42 | 1 | 5M |
| w/ 4 layers, n=3 | 79.5 | 48 | 1 | 10M |
| w/ self-attention | 74.7 | 42 | 1 | 7M |
| w/ T=2, n=2 | 73.7 | 12 | 1 | 5M |
| **w/ 1-step gradient** | **56.5** | 42 | 1 | 5M |

And the latent-feature count:

| Variant | # latents | Acc (%) |
|---|---|---|
| TRM (y, z) | 2 | 87.4 |
| TRM multi-scale z | 7 | 77.6 |
| TRM single z | 1 | 71.9 |

Recursion-depth sweep (k = n-equivalent, T = outer):

| k | T | HRM depth | HRM acc | TRM depth | TRM acc |
|---|---|---|---|---|---|
| 1 | 1 | 9 | 46.4 | 7 | 63.2 |
| 2 | 2 | 24 | 55.0 | 20 | 81.9 |
| 3 | 3 | 48 | 61.6 | 42 | **87.4** |
| 4 | 4 | 80 | 59.5 | 72 | 84.2 |

**Reading for us:** recursion depth is non-monotonic and peaks at ~42 effective layers; the single
largest ablation delta is **1-step gradient (−30.9pp)**; two latents beat one and beat seven;
2 layers beat 4. Also note **T=2,n=2 → 73.7**, a 13.7pp drop — the shallow settings we would use on
6.5k rows are the settings where TRM's own advantage mostly evaporates.

### 1.6 Headline results [verified, as reported in the TRM paper]

TRM vs HRM: Sudoku-Extreme **87.4 / 55.0**; Maze-Hard **85.3 / 74.5**; ARC-AGI-1 **44.6 / 40.3**;
ARC-AGI-2 **7.8 / 5.0**. (HRM's numbers here are as quoted by TRM, not independently verified; see
§5 for the verified semi-private numbers, which are lower.)

---

## 2. HRM — Hierarchical Reasoning Model (Sapient Intelligence, 2025)

*Hierarchical Reasoning Model*, Wang et al., arXiv [2506.21734](https://arxiv.org/abs/2506.21734).
27M params, two coupled recurrent transformer modules, trained from scratch on ~1,000 examples per
task with no pretraining [verified].

### 2.1 The two-level recursion, exactly

Input embedding `x̃ = f_I(x; θ_I)`. The low-level module L steps every timestep; the high-level
module H steps once per cycle of T timesteps [verified, verbatim]:

```
z_L^i = f_L( z_L^{i-1}, z_H^{i-1}, x̃ ; θ_L )

z_H^i = { f_H( z_H^{i-1}, z_L^i ; θ_H )   if i ≡ 0 (mod T)
        { z_H^{i-1}                        otherwise

ŷ = f_O( z_H^{NT} ; θ_O )
```

So a single forward pass is `N` cycles × `T` timesteps. The paper's design intent is **"hierarchical
convergence"**: L converges to a local fixed point within a cycle, H then resets L's context so that
the system does *not* converge globally and effective depth is preserved [verified].

### 2.2 The 1-step gradient approximation and its justification

HRM does **not** do BPTT. Pseudocode [verified, verbatim, Figure 4]:

```python
def hrm(z, x, N=2, T=2):
    x = input_embedding(x)
    zH, zL = z
    with torch.no_grad():
        for _i in range(N * T - 1):
            zL = L_net(zL, zH, x)
            if (_i + 1) % T == 0:
                zH = H_net(zH, zL)
    # 1-step grad
    zL = L_net(zL, zH, x)
    zH = H_net(zH, zL)
    return (zH, zL), output_head(zH)
```

Gradient path: **output head → final state of H → final state of L → input embedding**. O(1) memory
[verified].

The justification is DEQ + the Implicit Function Theorem. Assume L reaches a fixed point
`z_L^* = f_L(z_L^*, z_H^{k-1}, x̃; θ_L)`, write the H update as `z_H^k = F(z_H^{k-1}; x̃, θ)`, let
`J_F = ∂F/∂z_H`. Then IFT gives the exact fixed-point gradient [verified, Eq. 1]:

```
∂z_H* / ∂θ = ( I − J_F|_{z_H*} )^{-1} · ∂F/∂θ |_{z_H*}
```

Inverting `(I − J_F)` is expensive, so via the Neumann series
`(I − J_F)^{-1} = I + J_F + J_F² + …` the **1-step gradient takes only the first term**,
`(I − J_F)^{-1} ≈ I` [verified, verbatim], giving

```
∂z_H*/∂θ_H ≈ ∂f_H/∂θ_H ,   ∂z_H*/∂θ_L ≈ (∂f_H/∂z_L*)(∂z_L*/∂θ_L) ,
∂z_H*/∂θ_I ≈ (∂f_H/∂z_L*)(∂z_L*/∂θ_I)                                 (Eq. 2)
∂z_L*/∂θ_L ≈ ∂f_L/∂θ_L ,   ∂z_L*/∂θ_I ≈ ∂f_L/∂θ_I                     (Eq. 3)
```

**This is the single most important caveat for us.** The approximation is only defensible *at a
fixed point*. TRM measured what it costs when the premise is weak: **87.4 → 56.5** [verified, §1.5].
See §9.1.

### 2.3 Deep supervision

Given `(x, y)`, run M "segments", each one forward pass. At segment m [verified, verbatim]:

1. `(z^m, ŷ^m) ← HRM(z^{m-1}, x; θ)`
2. `L_m ← Loss(ŷ^m, y)`
3. `θ ← OptimizerStep(θ, ∇_θ L_m)`

with `z^m` **detached** before being used as the input state of segment m+1, so gradients from
segment m+1 do not flow into segment m — "effectively creating a 1-step approximation of the
gradient of the recursive deep supervision process" [verified, verbatim]. The paper frames it as a
regulariser and as a stability aid for deep equilibrium models.

### 2.4 ACT: the full Q-head mechanism

A Q-head reads H's final state and emits two values [verified, verbatim]:

```
Q̂^m = σ( θ_Q^T  z_H^{mNT} ) ,      Q̂^m = (Q̂^m_halt , Q̂^m_continue)
```

Halting rule: let `M_max` be a fixed cap and `M_min` a **random variable** — with probability ε it
is drawn uniformly from `{2, …, M_max}` (to encourage longer thinking), and with probability 1−ε it
is 1. Halt when either `m > M_max`, or `Q̂_halt > Q̂_continue` **and** `m ≥ M_min` [verified].

The MDP: state `z^m`, actions {halt, continue}. "halt" ends the episode with binary reward
`1{ŷ^m = y}`; "continue" gives reward 0 and transitions to `z^{m+1}`. Targets [verified, verbatim]:

```
Ĝ^m_halt      = 1{ŷ^m = y}

Ĝ^m_continue  = { Q̂^{m+1}_halt                                if m ≥ M_max
                { max( Q̂^{m+1}_halt , Q̂^{m+1}_continue )     otherwise
```

Total per-segment loss [verified, verbatim]:

```
L^m_ACT = Loss(ŷ^m, y) + BinaryCrossEntropy( Q̂^m , Ĝ^m )
```

Computing `Ĝ^m_continue` requires **a second forward pass** at segment m+1 — which is the cost TRM
removes. The paper reports ACT matches a fixed `M_max` step count at much lower mean compute, and
that raising `M_max` at inference only (no retraining) gives test-time scaling on Sudoku but
"minimal gains" on ARC [verified].

Stability note the paper flags: Q-learning without a replay buffer or target network is fragile;
they rely on bounded parameters + weight decay + post-norm, citing Gallici et al. [verified].

---

## 3. Adaptive Computation Time (Graves, 2016)

arXiv [1603.08983](https://arxiv.org/abs/1603.08983). All equations below are verbatim [verified].

Per input step `t`, the RNN performs `N(t)` intermediate updates over a *shared* state function
`S` and shared output weights:

```
s_t^n = { S(s_{t-1}, x_t^1)   if n = 1
        { S(s_t^{n-1}, x_t^n) otherwise                          (3)

y_t^n = W_y s_t^n + b_y                                           (4)
```

where `x_t^n = x_t + δ_{n,1}` — a **binary flag on the first intermediate step only**, so the
network can tell "new input" from "more pondering on the same input" [verified]. A sigmoidal
halting unit:

```
h_t^n = σ( W_h s_t^n + b_h )                                      (5)
```

Halting distribution, step count, and remainder:

```
p_t^n = { R(t)   if n = N(t)
        { h_t^n  otherwise                                        (6)

N(t)  = min{ n' :  Σ_{n=1}^{n'} h_t^n  ≥  1 − ε }                 (7)

R(t)  = 1 − Σ_{n=1}^{N(t)-1} h_t^n                                (8)
```

with **ε = 0.01** in the paper, whose purpose is to let computation halt after a single update if
`h_t^1 ≥ 1 − ε` [verified]. By construction `Σ_n p_t^n = 1`.

ACT then takes a **mean-field** (not sampled) combination — this is its defining choice:

```
s_t = Σ_{n=1}^{N(t)} p_t^n s_t^n ,      y_t = Σ_{n=1}^{N(t)} p_t^n y_t^n     (9)
```

Ponder cost and loss:

```
ρ_t  = N(t) + R(t)                                                (10)
P(x) = Σ_{t=1}^{T} ρ_t                                            (11)

L̂(x, y) = L(x, y) + τ · P(x)                                      (12)
```

`P(x)` is an upper bound on the non-differentiable `Σ_t N(t)`. The paper is explicit that "the
behaviour of the network is quite sensitive to the value of τ, and it is not obvious how to choose a
good value" [verified, verbatim].

**Gradient handling.** `ρ_t` is discontinuous where `N(t)` increments. "In practice we simply ignore
the discontinuities by treating `N(t)` as constant and minimising `R(t)` everywhere" [verified,
verbatim], which gives

```
∂P(x)/∂h_t^n = { 0   if n = N(t)
               { −1  otherwise                                    (14)
```

**The documented cheat (ACT footnote 1) — read this before implementing anything.** Graves notes
that the "natural" geometric halting distribution
`p_t^n = h_t^n ∏_{n'<n} (1 − h_t^{n'})` with expected-ponder-time cost `ρ_t = Σ_n n·p_t^n` was tried
and the networks **learn to cheat**: they set `h_t^1` just below the halting threshold, keep
`h_t^n = 0` until some `N(t)`, then set `h_t^{N(t)}` high enough to halt. Because `p_t^{N(t)} ≪
p_t^1`, the late step gets tiny weight in the mean-field update — **so the network inflates the
magnitude of the late state and output vectors** until they dominate the sum anyway, having paid a
low ponder penalty [verified, verbatim]. This is the canonical halting-collapse failure, and it is
documented by the author of the method, not by a critic.

---

## 4. PonderNet (Banino, Balaguer, Blundell, 2021)

arXiv [2107.05407](https://arxiv.org/abs/2107.05407). All equations verbatim [verified].

Step function `ŷ_n, h_{n+1}, λ_n = s(x, h_n)` with an initial state `h_0`; `s` can be any network
(MLP, LSTM, transformer). `λ_n` is the scalar probability of halting at step n. Define a Bernoulli
Markov process with states "continue" (Λ=0) and "halt" (Λ=1), `Λ_0 = 0`:

```
P(Λ_n = 1 | Λ_{n-1} = 0) = λ_n ,   ∀ 1 ≤ n ≤ N                    (1)
```

The unconditional halting distribution is a **generalised geometric distribution** — exactly the
form Graves rejected for ACT:

```
p_n = λ_n ∏_{j=1}^{n-1} (1 − λ_j)                                  (2)
```

Truncation at N: either renormalise `p_n`, or dump the remaining mass on the last step
`p_N = 1 − Σ_{n=1}^{N-1} p_n`. In training, `N` is set as the smallest `n` with
`Σ_{j≤n} p_j > 1 − ε`, with **ε = 0.05** [verified].

**The loss** [verified, verbatim, Eq. 3]:

```
L  =  Σ_{n=1}^{N} p_n · L(y, ŷ_n)   +   β · KL( p_n ‖ p_G(λ_p) )
      └──────── L_Rec ────────┘           └────── L_Reg ──────┘
```

- `L_Rec` is the **expectation of the task loss over the halting distribution**, not a mean-field
  average of the *predictions*. Every step's prediction is trained, weighted by its halting
  probability. This is the key structural difference from ACT, which averages the predictions.
- `L_Reg` is the KL to a **geometric prior** `p_G(λ_p)` truncated at N. It serves two purposes
  stated in the paper: it biases the network towards an expected `1/λ_p` steps, and it "provides an
  incentive to give a non-zero probability to all possible number of steps, thus promoting
  exploration" [verified, verbatim]. That second purpose is the anti-collapse term.

**Inference is sampled, not averaged**: at each step draw `Λ_n ~ B(λ_n)` and emit `y = y_n` at the
first "halt"; if `N` is reached, emit `y_N` [verified].

**Hyperparameters and sensitivity** [verified]: `β = 0.01` fixed in all experiments; `λ_p` swept
uniform(0, 1.0] and uniform(0, 0.5]; `ε = 0.05`. The paper's Figure 2 compares ACT's sensitivity to
τ against PonderNet's to λ_p and reports PonderNet is the more robust of the two; the only parity
failure was at `λ_p = 0.9` (prior mean ~1.1 steps), while `λ_p = 0.1` (prior mean 10 steps) worked
and then *settled to a more efficient* average [verified]. PonderNet also **extrapolates** to harder
instances than seen in training by using more steps, "whereas ACT remained at chance level"
[verified].

**Why this matters for us:** PonderNet is strictly the better-engineered of the two adaptive-depth
methods, with one fewer sharp edge (two hyperparameters but a documented-robust one, versus ACT's
single very sensitive τ), and an explicit exploration term. If we were to implement *learned*
adaptive depth rather than TRM's binary correctness head, this is the formulation to use.

---

## 5. ARC-AGI results and the critique — did the architecture matter?

### 5.1 The ARC Prize team's independent analysis

[arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis) [verified]. They
reproduced HRM on the semi-private sets and then ablated it. Exact findings:

| Question | Finding |
|---|---|
| Verified score, ARC-AGI-1 | **32%** (9h 16m, $148.50, $1.48/task) |
| Verified score, ARC-AGI-2 | **2%** (12h 35m, $201, $1.68/task) |
| **Does the hierarchy matter?** | "a regular transformer comes within **~5pp** of the HRM model without any hyperparameter optimization. The gap is smallest for just 1 outer loop, where both models are on par." Their conclusion, verbatim: **"the performance on ARC-AGI is not an effect of the HRM architecture."** |
| **Does the outer refinement loop matter?** | 1 → 2 loops: **+13pp**. 1 → 8 loops: performance roughly **doubled**. |
| Train-time vs inference-time refinement | Models *trained* with 16 refinement loops beat models trained with 1 loop by **>15pp even at single-loop inference**. Training with refinement matters more than refining at inference. |
| Transduction / puzzle embeddings | Training on the 400 evaluation tasks only: **31% pass@2** vs **41%** with the extra data. "large majority of the performance is driven by training on the tasks seen at evaluation time." |
| Augmentation count | 1,000 augs (original) ≈ 300 augs ("near max performance"); 30 augs "within 4% of max". |
| Learned halting (ACT) | "does improve performance" but "within a few percentage points" of fixed refinement. |

### 5.2 Answering the question as asked

**Did the architecture matter at all? No — not measurably.** The decomposition of HRM's ARC score
is, in descending order of contribution:

1. **Transduction — training on the eval tasks' own demonstration pairs with per-task puzzle-ID
   embeddings.** 31 of 41 points survive when *only* the eval tasks are trained on, so this is the
   substrate, not a bonus. A corroborating audit of TRM found that a **blank or random puzzle ID
   gives 0%** [verified via arXiv [2512.11847](https://arxiv.org/abs/2512.11847), recorded as
   verified in `research/01-reasoning-models.md`].
2. **The outer refinement loop with per-step supervision.** +13pp from one extra loop; ~2× from
   eight; and the effect is overwhelmingly a *training-time* effect (>15pp).
3. **Augmentation + voting.** Saturates by 300 augmentations; worth ~11pp pass@1 in the TRM audit.
4. **Learned halting.** A few pp. TRM's own ablation has full ACT *losing* 1.3pp (87.4 → 86.1).
5. **The two-level hierarchy.** ~0–5pp, and 0pp at a single outer loop. TRM then removed the second
   module entirely and *gained* 5pp (one shared net 87.4 vs two nets 82.4) [verified, §1.5].

So the honest causal statement: **the hierarchy is the one component with no demonstrated effect,
and the component TRM deleted to get better results.** The 2026-03 follow-up
[2603.22871](https://arxiv.org/abs/2603.22871) goes further and reconstructs the whole thing as a
5M-param contraction-mapping / neural-ODE system [verified, title/abstract only].

### 5.3 What this implies for porting to a new domain

The transferable parts are, in order: **(a)** an outer loop that re-predicts and is supervised at
every step; **(b)** heavy weight sharing as a regulariser in small-data regimes; **(c)** a
correctness head usable as a selector. The non-transferable parts are **(d)** the hierarchy,
**(e)** the transduction and per-task embeddings, and **(f)** 1000× symmetry augmentation. We have
no analogue of (e) at all, and no 1000×-strength analogue of (f) for molecules.

---

## 6. Does any of this transfer off grid-puzzle tasks?

I searched the arXiv API directly (WebSearch budget was exhausted; all queries below were run and
their result sets read).

### 6.1 Tab-TRM — the only published TRM on tabular regression

*Tab-TRM: Tiny Recursive Model for Insurance Pricing on Tabular Data*, Padayachy, Richman,
Wüthrich, arXiv [2601.07675](https://arxiv.org/abs/2601.07675), submitted 2026-01-12 [verified,
full PDF read].

**Architecture.** Tokenise the covariates (piecewise-linear embeddings for continuous, entity
embeddings for categorical), prepend two learned prefix tokens `a^(0)` (answer) and `z^(0)`
(reasoning), and run [verified, verbatim, Eqs. 3–4]:

```
z^(t,s+1) = z^(t,s) + f_z( Flatten[ a^(t), z^(t,s), e_1, …, e_L ] ) ,   s = 0..m-1
a^(t+1)   = a^(t)   + f_a( Concat( a^(t), z^(t,m) ) )
z^(t,0)   = z^(t-1,m)
```

both residual, then a linear head + `exp` to preserve the Poisson GLM log-link. **Three deliberate
departures from TRM** [verified, verbatim]: separate `f_z` and `f_a` rather than one shared net;
pure FNNs instead of attention (L ≤ 20 tokens, so flatten and feed an MLP); and **"we omit both deep
supervision and the ACT halting mechanism"** — because `T·m ≈ 16–64` total steps "permit full
gradient flow through the entire unrolled computation without memory or stability issues".

So Tab-TRM is *recursion only*: the one component ARC Prize found didn't matter, with the two
components that did (deep supervision, halting) removed.

**Results** [verified, Table 3, out-of-sample Poisson deviance ×10⁻²]:

| Model | Params | In-sample | Out-of-sample |
|---|---|---|---|
| Null (intercept) | 1 | 25.213 | 25.445 |
| Poisson GLM3 | 50 | 24.084 | 24.102 |
| Poisson GAM | (66.7) | 23.920 | 23.956 |
| Plain FNN | 792 | 23.728 (±0.026) | 23.819 (±0.017) |
| Ensemble plain FNN | 792 | 23.691 | 23.783 |
| CAFFT | 27,133 | 23.715 (±0.047) | 23.807 (±0.017) |
| Ensemble CAFFT | 27,133 | 23.630 | 23.726 |
| Credibility Transformer | 1,746 | 23.641 (±0.053) | 23.788 (±0.040) |
| Ensemble Credibility Transformer | 1,746 | 23.562 | 23.711 |
| Tree-like PIN | 4,147 | 23.593 (±0.046) | 23.740 (±0.025) |
| Ensemble Tree-like PIN | 4,147 | 23.522 | 23.667 |
| **Tab-TRM** | 14,820 | 23.570 (±0.027) | **23.666 (±0.027)** |
| **Ensemble Tab-TRM** | 14,820 | 23.496 | **23.589** |

Hyperparameters were Optuna-selected; best config `T = 6, m = 3` by a validation sweep over
`T ∈ {1,2,3,4,5,6,8,10} × m ∈ {0,1,2,3,4,5,6,8}` [verified]. A 10-run nagging ensemble moved test
deviance 23.630 → 23.589 (mean per-run 23.666) [verified].

**Four things about this result that bear directly on us:**

1. **The margin is 0.33% relative.** Ensemble Tab-TRM 23.589 vs ensemble Tree-like PIN 23.667 =
   **0.078 ×10⁻² on a base of 23.6**. Single-run sd is ±0.027, so ~2.9 sd — real, and tiny.
2. **The 10-run ensemble is worth as much as the architecture.** Single 23.666 → ensemble 23.589 is
   **0.077**, i.e. ensembling buys the entire margin over the previous best model a second time.
   [assumed inference from their own numbers.]
3. **There is no GBM in the benchmark table.** The paper's introduction frames the contribution as
   bridging GLMs and "Gradient Boosting Machines", but Table 3 contains no XGBoost / LightGBM /
   CatBoost row at all [verified — I read the whole table]. Since our current best is *tied* between
   an MLP ensemble and XGBoost at ~0.44, the comparison we care about is precisely the one Tab-TRM
   does not make.
4. **The scale is wrong for us by two axes at once.** French MTPL: **L = 9 feature tokens**,
   `d = 28`, flattened input **308-dim**, and the paper itself describes the portfolio as containing
   "hundreds of thousands of policies" (the standard `freMTPL2freq` benchmark is ~678k rows
   [assumed, well-known dataset size, not re-verified]). We have **~2,500 features and 6,525
   labels** — roughly 275× more features and 100× fewer rows. Recursion-as-regulariser is a
   plausible story at 9 features / 680k rows; at 2,500 features / 6.5k rows the binding constraint
   is a different one entirely.

**And the paper's own best ablation is a warning.** §6 "Linearized Tab-TRM" removes *all* nonlinear
activations from `f_z` and `f_a`, leaving affine updates [verified, verbatim Eqs. 30–31]. Result:
10-run ensemble **23.698**, mean 23.742, range 23.695–23.780 — i.e. **only 0.109 ×10⁻² worse than
the full model, and still better than a plain FNN**. Their explanation: at `d = 28` the piecewise-
linear numerical embeddings already do "a fine-grained binning of the continuous covariates", and
"the TRM merely needs to select and rearrange this fine-grained representation", so "non-linearity
in the TRM recursion is not important" [verified, verbatim]. They test this by re-running at `d = 4`
and the update linearity drops (R² 0.8 / 0.5) [verified].

Read plainly: **on the one tabular dataset where TRM-style recursion has been shown to win, most of
the win survives making the recursion linear, and the authors attribute the performance to the
input embeddings rather than to the recursion.** That is about as close to a null-on-the-mechanism
as a positive-result paper gets.

### 6.2 Looped / weight-tied recursion on tabular: a measured null

*Latent Chain-of-Thought Improves Structured-Data Transformers*, arXiv
[2605.11262](https://arxiv.org/abs/2605.11262), 2026-05 [verified, full PDF read]. 36 datasets:
9 LTSF time-series + a **27-dataset OpenML subset** used in TabPFN evaluations, on a TabICL
three-stage architecture, cross-entropy for classification and RMSE for regression.

Their headline table, average gain vs a **same-depth, parameter-matched** baseline [verified]:

| Domain | Variant | Wins | Average gain |
|---|---|---|---|
| Time series | Deeper (2× params) | — | −5.45% |
| Time series | Best CoT | 7/9 | +2.77% ± 1.05% (and +12.63% headline vs baseline) |
| Time series | Best Looped | 7/9 | — |
| **Tabular** | Deeper (2× params) | 14/27 | **+0.66% ± 0.63%** |
| **Tabular** | **Best CoT** | **23/27** | **+3.25% ± 0.88%** |
| **Tabular** | **Best Looped (weight-tied)** | **14/27** | **−0.39% ± 0.80%** |

**This is the single most decision-relevant number in this document.** Pure **weight-tied depth
recurrence over a shared block — which is what "recursion" means mechanically — is a null on
tabular data: −0.39% ± 0.80%, winning 14 of 27.** What *does* help (+3.25%, 23/27) is appending
*feedback tokens* derived from the prediction positions back into the input sequence — i.e. the
model re-reads its own previous answer, which is the `a`/`y` channel of TRM, not the weight sharing.

That replicates this programme's **H2** exactly (TRM-style recursive encoder 0.2510 vs plain MLP
0.2493, gap 0.0017, "confirmed null") from an independent direction and at 27-dataset scale.

Two further notes from that paper, both relevant:
- They cite a mechanistic result that in tabular foundation models "a single transformer block
  looped six times recovers the performance of a full six-layer model with only 20% of the
  parameters" — recursion buys **parameter efficiency at equal accuracy**, not accuracy [verified,
  as cited in 2605.11262 §2].
- Their limitations section says, verbatim: "The chain-of-thought depth R is fixed per model rather
  than chosen adaptively per input… **An adaptive-CoT mechanism similar to PonderNet would be a
  useful next step**" [verified]. As of 2026-05, *nobody had done adaptive depth on tabular data*.

### 6.3 The other near-miss: Recursive Inference Machines

*Recursive Inference Machines for Neural Reasoning*, arXiv
[2603.05234](https://arxiv.org/abs/2603.05234) [verified, PDF read]. The abstract's claim that RIMs
improve "classification of tabular data, outperforming TabPFNs" is real but is **not an adaptive-
depth result**:

- Setup: two tiny UCI **classification** sets, Cleveland Heart Disease and Ljubljana Breast Cancer,
  with **25% of all feature values replaced by random values** to simulate corruption [verified].
- Method: TabRIM treats the input as a noisy realisation of a latent clean state and runs a **Gibbs
  sampler** over feature conditionals via repeated TabPFN forward passes — 10 samples, 5 burn-in
  [verified].
- Results (Table 2, vs TabPFN only) [verified]: Cleveland AUC-ROC 0.85 → **0.87** (+0.02), AUC-PR
  0.83 → 0.84; Ljubljana AUC-ROC 0.63 → **0.74** (+0.11), AUC-PR 0.42 → 0.54.

That is a **denoising** result on synthetically corrupted features, measured against a single
baseline, on classification. It is not evidence for recursive depth on tabular regression. Its
genuinely interesting contribution for us is elsewhere: on Sudoku, replacing TRM's implicit identity
reweighting with a learned dynamic reweighter moves 87.16 → **89.34** (+2.18), while *static* scalar
reweighting is **worse** than TRM (80.26 at fixed α, 84.27 at learnable scalars) [verified, Table 4]
— i.e. the useful knob is a *per-step, input-dependent* combination weight, not depth.

### 6.4 Molecular property / bioactivity regression: no published evidence

Two arXiv API queries, both returning usable result sets [verified]:

- `(recursive reasoning | adaptive depth | dynamic depth | deep equilibrium) ∧ (molecular property |
  QSAR | ADMET)` → **`totalResults = 0`**.
- `(ponder | halting | recursive refinement) ∧ (drug discovery | bioactivity | IC50 | molecular
  property prediction)` → 3 hits, none relevant: a BACE1 multi-ligand docking paper, a SARS-CoV-2
  Mpro design paper, and DeepRetro (LLM retrosynthesis, "recursive refinement" in the agentic sense,
  not latent depth).
- `abs:"adaptive computation" ∧ abs:molecular` → 6 hits; the only genuinely adaptive-compute one is
  **BranchIP** ([2610.02013](https://arxiv.org/abs/2610.02013), 2026-10-01), learned adaptive tensor
  products to *accelerate* machine-learning interatomic potentials at matched accuracy — a FLOPs
  result, not an accuracy result, and on potentials rather than property regression.
- `all:"PonderNet"` → 5 hits total (PonderNet, PALBERT, NeuronSpark, RecurTrace, one cognitive-
  architecture paper). **None** on tabular, regression, chemistry, materials or small-data science.

**Honest answer: no published evidence.** The full set of non-puzzle TRM applications that exist
(from a 25-entry arXiv sweep of "tiny recursive model") is: code generation (2608.29376), LLM
logical reasoning (2608.08113), **engineering PDE/FEA surrogate modelling** (2607.27251), **orbital
mechanics — the J2-perturbed Lambert problem at 2.3M params** (2606.00895), image classification
(2603.19503), machine-translation quality estimation (2603.14593), tabular insurance (2601.07675),
and combinatorial optimisation (2609.34585) [verified, titles/abstracts]. The two *continuous
regression* entries (2607.27251, 2606.00895) are physics surrogates where the target is a smooth
function of low-dimensional inputs with effectively unlimited synthetic training data — the opposite
regime from 6.5k noisy assay labels over a 2,500-dim sparse fingerprint. I could not extract a
baseline comparison from 2607.27251's abstract page, so I am not claiming either direction for it.

### 6.5 What *is* established in our own domain

Iterative latent refinement is already standard in molecular ML, just not called "recursion":
message passing is recurrent refinement over a graph, and AlphaFold2 / Boltz **recycling** feeds the
predicted structure back 3+ times [assumed — well-established, carried over from
`research/01-reasoning-models.md`]. The important asymmetry: recycling refines a *structured,
high-dimensional output* where successive passes have something concrete to correct. A scalar pIC50
has no internal structure to refine, so the "improvement operator" prior has much less to grip on.
That is the mechanistic reason to expect this to behave differently here than on a 9×9 grid or a
3-D pose.

---

## 7. What to actually implement here

### 7.1 Which variant, and why that one

Not TRM. **Deep supervision with a band-membership halt head used as a step selector.**

The argument, in four steps:

1. Recursion/weight-sharing is a measured null on this kind of data — twice (H2 here; −0.39% over
   27 tabular datasets in §6.2). Do not spend the experiment on it.
2. The component that actually carried HRM is the **outer refinement loop with per-step
   supervision**, and it is a *training-time* effect (+13pp for one loop, >15pp train-vs-inference)
   (§5.1). That component has never been tested on this task — H2 varied the *encoder* only.
3. This task has a **natural binary correctness signal that most regression tasks lack**: ST-RAE is
   zero when the prediction lands inside the credible interval. That is precisely the shape of TRM's
   halt target `1{ŷ = y}`, which on a continuous target is normally undefinable. This is the one
   structural reason the method fits this task better than it fits tabular regression in general.
4. The only intervention that has ever cleared this task's noise floor is **blending** (+0.0283,
   95% CI [+0.0160, +0.0427]). Deep supervision produces `N_sup` predictions per molecule from one
   model, and the halt head is a learned selector over them. So the variant to test is the one whose
   mechanism is *the mechanism that already works here*, obtained for ~1× the training cost instead
   of 2 models.

Frame it honestly in the hypothesis register: **this is a cheap-ensemble-with-a-learned-selector
experiment wearing a reasoning-model costume.** That framing also tells you the right control
(§7.4, arm A4b).

### 7.2 Minimal pseudocode

Keep the existing encoder and the existing interval-hinge loss. Everything below is additive.

```python
# ---- shapes -------------------------------------------------------------
# x        : (B, 2500)  count-ECFP4 + RDKit2D + Avalon + ErG, standardised as now
# lo, hi   : (B, 4)     credible-interval bounds per endpoint (train/val only)
# mask     : (B, 4)     label present
# D = 256 latent width

class RefineNet(nn.Module):
    def __init__(self, D=256, n_endpoints=4):
        super().__init__()
        self.enc  = ExistingMLPEncoder(2500, D)          # unchanged, reuse weights/init
        self.y0   = nn.Parameter(torch.zeros(D))         # learned answer init
        self.z0   = nn.Parameter(torch.zeros(D))         # learned latent init
        self.f_z  = MLP(3*D, 2*D, D)                     # Δz from (h, y, z)
        self.f_y  = MLP(2*D, 2*D, D)                     # Δy from (y, z)
        self.ln_y, self.ln_z = nn.LayerNorm(D), nn.LayerNorm(D)
        self.head = nn.Linear(D, n_endpoints)            # pIC50 per endpoint
        self.halt = nn.Linear(D, n_endpoints)            # "inside the band?" logit per endpoint

    def cycle(self, h, y, z, n):                         # one answer update, n latent updates
        for _ in range(n):
            z = self.ln_z(z + self.f_z(torch.cat([h, y, z], -1)))
            y = self.ln_y(y + self.f_y(torch.cat([y, z], -1)))
        return y, z

    def segment(self, h, y, z, n=2, T=2):
        with torch.no_grad():                            # T-1 cycles: NO grad  (TRM)
            for _ in range(T - 1):
                y, z = self.cycle(h, y, z, n)
        y, z = self.cycle(h, y, z, n)                    # last cycle: FULL backprop (NOT 1-step)
        return y, z, self.head(y), self.halt(y)

# ---- training -----------------------------------------------------------
# interval hinge: 0 inside [lo, hi], Huber on the distance to the nearer bound outside.
def hinge(pred, lo, hi, mask, delta=1.0):
    d = torch.clamp(lo - pred, min=0) + torch.clamp(pred - hi, min=0)   # 0 inside the band
    h = torch.where(d <= delta, 0.5 * d**2 / delta, d - 0.5 * delta)    # Huber
    return (h * mask).sum() / mask.sum().clamp_min(1)

for x, lo, hi, mask in loader:
    h = model.enc(x)                                  # encode ONCE per batch
    y = model.y0.expand(x.size(0), -1)
    z = model.z0.expand(x.size(0), -1)
    for step in range(N_SUP):                         # deep supervision
        y, z, pred, q = model.segment(h, y, z, n=N, T=T)

        loss = hinge(pred, lo, hi, mask, DELTA)
        with torch.no_grad():                         # STOP-GRAD the halt target (TRM does this)
            inside = ((pred >= lo) & (pred <= hi)).float()
        loss = loss + LAMBDA_Q * (
            F.binary_cross_entropy_with_logits(q, inside, reduction="none") * mask
        ).sum() / mask.sum().clamp_min(1)

        loss.backward()                               # one optimiser step PER SEGMENT (TRM)
        opt.step(); opt.zero_grad()

        y, z = y.detach(), z.detach()                 # carry detached across segments
        h    = h.detach()                             # encoder gets grad from THIS segment only
```

Note `h = h.detach()` after the first segment: without it the encoder accumulates gradient from
every segment through a graph that `opt.step()` has already consumed, and PyTorch raises
(or, with `retain_graph=True` bolted on, you silently train the encoder `N_SUP` times harder than
the refiner). TRM sidesteps this by re-embedding `x` inside the loop; re-encoding 2,500-dim inputs
`N_SUP` times is wasteful here, so encode once and detach. **Decide explicitly whether the encoder
should see gradient from all segments (re-encode inside the loop, no detach) or only the first
(encode once, detach after) — and record the choice.** [assumed: these two differ materially; I have
not measured which is better.]

```python
# ---- inference ----------------------------------------------------------
@torch.no_grad()
def predict(model, x, N_SUP, n, T):
    h = model.enc(x)
    y = model.y0.expand(x.size(0), -1); z = model.z0.expand(x.size(0), -1)
    preds, qs = [], []
    for _ in range(N_SUP):
        y, z, p, q = model.segment(h, y, z, n=n, T=T)
        preds.append(p); qs.append(q)
    P = torch.stack(preds, 1)                 # (B, N_SUP, 4)
    Q = torch.stack(qs,    1)                 # (B, N_SUP, 4)

    sel  = P.gather(1, Q.argmax(1, keepdim=True)).squeeze(1)   # halt-head selector, per endpoint
    mean = P.mean(1)                                            # the control: plain step-average
    last = P[:, -1]                                             # the other control
    return sel, mean, last
```

Report **all three**. If `sel` does not beat `mean`, the halt head learned nothing and the whole
premise of the experiment has failed — report that, it is the informative outcome.

### 7.3 Starting hyperparameters

Do **not** copy TRM's numbers; they are tuned for 60k–100k epochs on 1,000 puzzles with 1000×
augmentation. Anchor on this programme's existing CYP recipe and change only what the method
requires.

| Knob | Start at | Why |
|---|---|---|
| `D` (latent width) | 256 | matches the existing encoder output; TRM's 512 is for 900-token grids |
| `N` (inner, latent updates) | **2** | TRM's `T=2,n=2` costs it 13.7pp on Sudoku — we are deliberately in the shallow regime, because depth is the part predicted null |
| `T` (middle, no-grad cycles) | **2** | one no-grad cycle + one differentiated cycle |
| `N_SUP` (deep supervision) | **4** | TRM's 16 × a short run is a different optimisation problem entirely (§9.5); 4 keeps the step-count confound to 4× and gives 4 ensemble members |
| `LAMBDA_Q` | **0.5** | exactly TRM's weighting (`lm_loss + 0.5 * q_halt_loss`) |
| Halt rule in training | **none — run all `N_SUP` steps** | early-breaking on `q>0` is the halting-collapse trap (§9.2); add it only after the fixed-depth version is measured |
| `halt_exploration_prob` | 0.0 initially | the exploration trick only matters once you actually break early |
| Optimiser | AdamW, β=(0.9, 0.95) | TRM's betas, cheap to adopt |
| LR | **1e-3**, warmup 200 steps, cosine | the existing CYP net's LR; TRM's 1e-4 flat is for 100k epochs |
| Weight decay | sweep **{0.01, 0.1, 1.0}** | TRM uses 1.0 on small-data tasks and 0.1 on ARC; this is the one TRM hyperparameter with a real small-data argument behind it |
| Batch | 256 | as now |
| Dropout | 0.1 in `f_z`, `f_y` | as now |
| EMA | **off at first**; if on, `ema_rate` from measured steps-to-best, **not** 0.999 | the repo ships `ema: False`; 0.999 has a ~1000-step time constant and our runs peak at epoch 20–65 (H12), so a copied 0.999 EMA may never converge inside the useful window |
| Epoch budget | the existing early-stopping rule, unchanged | H12: 800 extra epochs is pure noise; a longer budget is not the lever |
| Seeds | **4 paired** (identical split, data order, init seed per arm) | H12's design; the paired across-seed sd was ±0.0098 |
| Split | the existing 5-fold scaffold split | so the result is comparable to the 0.4346 / 0.4432 numbers |

### 7.4 The ablation ladder, cheapest-first

Run in this order, because the order is the ARC Prize ordering and it stops early on a null:

| Arm | What changes | Isolates | Prediction |
|---|---|---|---|
| **A0** | existing MLP ensemble | baseline | 0.4346 (known) |
| **A1** | `N_SUP=4`, `T=1`, `N=0` — same net re-applied once per segment, carry detached, per-step hinge | **the outer refinement loop alone** | the live arm |
| **A2** | `N_SUP=1`, `T=2`, `N=2` — recursion, no deep supervision | **weight-tied depth alone** (≈ H2) | **null**, ≤0.01 either way |
| **A3** | `N_SUP=4`, `T=2`, `N=2` | both | ≈ A1 |
| **A4a** | A3 scored by halt-head selection | **the learned selector** | the second live arm |
| **A4b** | A3 scored by plain step-average | **the selector's control** | if A4a ≈ A4b, the halt head is inert |
| **A5** | A0 but with `4 × N_SUP` optimiser steps at matched wall-clock | **the step-count confound** (§9.5) | must be run before claiming A1 > A0 |

Cost: H2 ran two encoder arms in ~0.13 L4-hours. Six arms × 4 paired seeds × 5 folds is **~3–4
L4-hours total** [assumed, scaled from H2]. This is cheap enough that the right move is to run the
whole ladder rather than argue about it.

**Register the readout rule before running** (T2/H11 discipline): adopt only if the **paired** mean
gain over A0 exceeds the pooled across-seed sd, with the 95% paired-bootstrap CI excluding zero, and
report A5 alongside. A1 beating A0 while A5 also beats A0 means we measured optimiser steps, not
refinement.

---

## 8. Honest expected value

**I expect a null, at 70–80% [assumed].** The reasoning, with the counter-argument stated fairly:

**Why a null is the base case.**

1. **The architecture has no demonstrated effect even on its home turf.** A same-size plain
   transformer is within ~5pp of HRM and on par at one outer loop; the ARC Prize team's own words
   are "the performance on ARC-AGI is not an effect of the HRM architecture" (§5.1, verified).
2. **Weight-tied recursion on tabular data is a measured null at 27-dataset scale** (−0.39% ± 0.80%,
   14/27), and this programme already measured it here (H2, 0.0017) (§6.2).
3. **The one tabular TRM win is 0.33% relative, and mostly survives linearising the recursion.** On
   this task 0.33% relative is ~**0.0014 ST-RAE**. This task's single-fold bootstrap sd is ≈**0.03**
   and H12's paired across-seed sd is ±**0.0098**. A Tab-TRM-sized effect is **7× below** the paired
   MDE and **20× below** the unpaired one. If the true effect is the size of the only published
   positive result, **this experiment cannot see it, and no amount of seeds within our budget
   fixes that.** That is the honest statistical statement and it should be in the register.
4. **The mechanisms that carried HRM are unavailable or already exploited.** The transduction and
   per-task embeddings have no analogue (the test set is 750 blinded compounds with only
   `Molecule_Name` and `SMILES`). 1000× symmetry augmentation has no molecular analogue of that
   strength. Deep supervision's regularising effect competes with what the existing recipe already
   does — early stopping at epoch 20–65, dropout, weight decay, and a 5-model ensemble.
5. **This programme's own prior is explicit and well-evidenced**: of every intervention tried, only
   **representation** changes (H1 count-ECFP4+RDKit2D, 18/21; H5 Avalon+ErG, 14/21) have moved the
   number. Capacity moved the *trajectory* and not the destination (H12). Auxiliary heads did
   nothing twice (H7a; 0.4346 vs 0.4359). Recursion did nothing (H2). An architecture experiment is
   a draw from a distribution this programme has already sampled five times, with five nulls.
6. **A scalar has nothing to refine.** The TRM prior is "the optimal answer is reachable by iterated
   improvement of a structured partial answer." Sudoku has 81 cells, ARC a 30×30 grid, AlphaFold
   a coordinate set. A pIC50 is one number; the "improvement operator" has no internal structure to
   operate on, so the inductive bias that makes TRM work is largely vacuous here (§6.5) [assumed,
   this is my inference and not something I found measured].

**The two reasons it is not a pure waste of GPU-hours.**

- **The halt head is a genuinely well-matched object.** ST-RAE's zero-band gives a *defined* binary
  correctness target for a regression task, which is exactly what TRM's halt head wants and what
  continuous targets normally cannot supply. And a learned per-step selector is the closest
  architectural relative of the only thing that has worked here (+0.0283 from blending). If
  A4a > A4b by ≥0.01, that is a real finding about *selection*, and it would generalise to the
  21-endpoint ADMET panel.
- **It closes a question cheaply and with a verdict, which is what this register is for.** Four
  GPU-hours buys a registered null on "do reasoning-model architectures help small molecular
  regression", with the decomposition (refinement loop vs depth vs selector) that the literature
  itself says is the right decomposition. The decomposition is the deliverable even when every arm
  nulls.

**The cheapest way to establish the null, concretely.** Run **A2 first, not A1** if the goal is
purely to close the question: A2 is one config change to the existing training script (no deep
supervision, no new loss, no halt head), it reproduces H2's design on the CYP data at 4 paired
seeds, and a null there plus H2's null plus §6.2's null is three independent nulls on the same
claim, which is enough to retire it. **A1 + A4a/A4b** are the arms worth paying for, and they should
be run as a *selector* experiment with that framing in the register — because if they win, the
conclusion is "a learned per-step selector over an intra-model ensemble helps", which is a claim
about ensembling, not about reasoning, and conflating the two is how this result would get
over-sold.

**What would change my mind before running:** a published adaptive-depth or recursive-refinement
result on a small (<20k rows), high-dimensional (>500 features) regression task with a GBM baseline
in the table. I searched for it (§6.4) and it does not exist as of 2026-10-05.

---

## 9. Pitfalls that make a naive implementation silently wrong

Ordered by how long each would cost before you noticed.

### 9.1 The 1-step gradient approximation is unjustified here, and it fails quietly

HRM's 1-step gradient is `(I − J_F)^{-1} ≈ I`, the first term of a Neumann series, and it is only
defensible **at a fixed point** of the latent recursion (§2.2). Nothing in a 4-output regression
head over a 2,500-dim fingerprint forces the latents to converge, so the premise is simply absent.
TRM measured the cost when the premise is weak: **87.4 → 56.5, a 30.9pp collapse** — the largest
single ablation in its table. **Always backprop fully through the last cycle.** The failure mode is
that training works, the loss decreases, and you get a worse model with no error and no warning.
If you want the memory saving, take it by reducing `T` (more no-grad cycles), not by truncating the
gradient of the last one.

### 9.2 Halting collapse, in both of its documented forms

- **Graves's own footnote 1** documents the cheat for geometric halting with an expected-time cost:
  the network sets the first halting unit just below threshold, zeroes the rest, then halts late and
  **inflates the magnitude of the late state/output vectors** so they dominate the mean-field sum
  while paying a low ponder penalty (§3, verified verbatim). Any mean-field combination over steps
  with a ponder penalty is exposed to this.
- **The base-rate collapse, which is the one that will bite *this* task.** TRM's halt target is
  `1{ŷ = y}`. Here that is `1{prediction inside the band}` — and that base rate is wildly uneven
  across endpoints: CYP3A4 scores **0.2550** while CYP2D6 scores **0.6013** against a 0.6116 floor
  (H11). On the endpoint where the model is usually inside the band, the BCE target is almost always
  1, the halt logit saturates positive, and `if q > 0: break` halts at step 1 **every time** — so
  you have silently trained and are silently scoring a 1-step model while the config says `N_SUP=4`.
  Mitigations, all of which are in the pseudocode above: (a) **a per-endpoint halt head**, not one
  shared head, since a shared head is driven by the easiest endpoint; (b) **no early break during
  training** until the fixed-depth version is measured; (c) log the **halting-step histogram per
  endpoint per epoch** and the **base rate of `inside`** — if either the histogram is a spike at
  step 1 or the base rate is >0.9, the head is inert by construction, not by failure to learn;
  (d) if you do add early-breaking, bring `halt_exploration_prob = 0.1` with it, which is exactly
  what that hyperparameter is for (§1.3).

### 9.3 The halt target must be stop-gradded

TRM computes `seq_is_correct` inside a `torch.no_grad()` block (§1.2, verified in source). If you
let gradient flow into the target, the model can reduce the BCE term by changing *whether it is
correct* rather than by predicting its correctness better — and on a hinge loss with a flat zero
region there is a particularly cheap version of this, because nudging a prediction just outside a
wide band costs almost nothing in hinge loss while giving the halt head an easier target to match.
The combined loss goes down; the metric gets worse.

### 9.4 Detach discipline across segments — three distinct ways to get it wrong

- **Forget to detach the carry `(y, z)`** → you are doing BPTT through `N_SUP × T × (N+1)` network
  applications. At `N_SUP=16` that is 100+ applications; memory blows up, and if it doesn't, you are
  no longer running the method described in either paper.
- **Detach in the wrong place** → the encoder receives no gradient and you silently train only the
  refiner and heads on a frozen random projection of the fingerprint. Guard: assert
  `enc.weight.grad is not None` and non-zero after the first backward, every run.
- **Reuse a consumed graph** → after `opt.step()` on segment 1, segment 2's backward touches the
  same encoder graph. Either re-encode inside the loop (TRM's choice) or `h = h.detach()` after the
  first segment (cheaper here) — but pick one deliberately and record it, because the two give
  materially different encoder gradients (§7.2).

### 9.5 Deep supervision multiplies optimiser steps — this is an unmatched-budget confound

In both TRM's and HRM's training loops, `opt.step()` is **inside** the supervision loop (§1.2, §2.3,
both verified verbatim). One batch therefore yields up to `N_SUP` parameter updates. **A comparison
against the baseline at equal epochs gives the deep-supervision arm `N_SUP`× the optimiser steps.**
This programme has already been bitten by exactly this class of error in another guise: H12 found
that capacity "moves the trajectory, not the destination" — the best epoch shifted by **+116.2 ±
20.2 (4/4 seeds, p=0.0014)** while the score gain was **+0.0019 ± 0.0098 (p=0.723)**. A4/A5 exists
precisely so that a step-count effect cannot be reported as an architecture effect. Match on
optimiser steps, report both, and remember that the epoch budget itself is not a lever here (800
extra epochs is noise).

### 9.6 Smaller edges worth a line each

- **Don't copy `ema_rate = 0.999`.** TRM's paper says EMA is worth 7.5pp on Sudoku, but the repo
  ships `ema: False`, and 0.999 has a ~1000-step time constant against runs that peak at epoch
  20–65. Either set the rate from measured steps-to-best or leave EMA off and say so.
- **Don't copy `lr = 1e-4` with `lr_min_ratio = 1.0`.** That is a flat LR for a 100,000-epoch run.
- **`T=2, n=2` is TRM's *worst* depth setting** (73.7 vs 87.4). We are choosing it deliberately for
  cost and small-data reasons, but it means "we tried TRM and it didn't work" would be an unfair
  summary of a null — the fair summary is "we tried the shallow regime that fits our data budget".
  Record that.
- **Two latents, not one and not seven** (87.4 / 71.9 / 77.6). Keep `(y, z)`.
- **One shared net beat two separate nets in TRM (87.4 vs 82.4), but Tab-TRM deliberately used two
  separate nets** because the z-update conditions on the features and the a-update does not
  (§6.1, verified). Our `f_z` takes `(h, y, z)` and `f_y` takes `(y, z)` — different input
  arities — so separate nets are the right default here, following Tab-TRM rather than TRM.
- **The halt head's target does not exist at inference.** The blinded test set has no interval
  bounds, so the halt head can be *trained* and *used* (it only reads `y`) but its selector quality
  can only be *validated* on folds that have bands. Don't accidentally build an inference path that
  needs `lo`/`hi`.
- **Interval bounds are leakage-adjacent.** The 2026-10-05 entry records a 7× score jump from
  feeding auxiliary columns as features, caught only because baselines ran first. `lo`/`hi` are
  label-derived. They belong in the loss and in the halt *target*, never in `x`, never in the
  encoder, never concatenated into `h`. Assert on the input dimension.

---

## 10. Sources

Primary, read in this session:

- **TRM** — *Less is More: Recursive Reasoning with Tiny Networks*, Jolicoeur-Martineau, arXiv
  [2510.04871](https://arxiv.org/abs/2510.04871) (algorithm boxes, ablation tables, recipe via the
  arXiv HTML render) [verified]
- **TRM source** — [`SamsungSAILMontreal/TinyRecursiveModels`](https://github.com/SamsungSAILMontreal/TinyRecursiveModels):
  `models/recursive_reasoning/trm.py`, `models/losses.py`, `config/arch/trm.yaml`,
  `config/cfg_pretrain.yaml` [verified]
- **HRM** — *Hierarchical Reasoning Model*, Wang et al., arXiv
  [2506.21734](https://arxiv.org/abs/2506.21734) (PDF, §§ on hierarchical convergence, 1-step
  gradient/IFT, deep supervision, ACT) [verified]
- **ARC Prize analysis of HRM** — [arcprize.org/blog/hrm-analysis](https://arcprize.org/blog/hrm-analysis)
  [verified]
- **ACT** — Graves, *Adaptive Computation Time for Recurrent Neural Networks*, arXiv
  [1603.08983](https://arxiv.org/abs/1603.08983) (PDF §2, Eqs. 3–14 and footnote 1) [verified]
- **PonderNet** — Banino, Balaguer, Blundell, *PonderNet: Learning to Ponder*, arXiv
  [2107.05407](https://arxiv.org/abs/2107.05407) (PDF §§2.1–2.5, Eqs. 1–3, appendix hyperparameters)
  [verified]
- **Tab-TRM** — Padayachy, Richman, Wüthrich, *Tab-TRM: Tiny Recursive Model for Insurance Pricing
  on Tabular Data*, arXiv [2601.07675](https://arxiv.org/abs/2601.07675) (full PDF: Eqs. 3–4,
  Tables 1–3, §6 linearised variant, §7 discussion) [verified]
- **Looped transformers on tabular** — *Latent Chain-of-Thought Improves Structured-Data
  Transformers*, arXiv [2605.11262](https://arxiv.org/abs/2605.11262) (full PDF: results table,
  baselines, limitations) [verified]
- **RIM** — *Recursive Inference Machines for Neural Reasoning*, arXiv
  [2603.05234](https://arxiv.org/abs/2603.05234) (full PDF: §5, Tables 2 and 4) [verified]

Secondary / titles-and-abstracts only [verified at that depth]:
[2512.11847](https://arxiv.org/abs/2512.11847) (TRM audit: blank puzzle-ID → 0%, accuracy saturates
after a few latent steps), [2605.19943](https://arxiv.org/abs/2605.19943) (PTRM),
[2603.22871](https://arxiv.org/abs/2603.22871) (HRM as a contraction mapping),
[2607.27251](https://arxiv.org/abs/2607.27251) (TRM for FEA/PDE surrogates),
[2606.00895](https://arxiv.org/abs/2606.00895) (TRM for the J2 Lambert problem),
[2610.02013](https://arxiv.org/abs/2610.02013) (BranchIP, adaptive compute for interatomic
potentials), [2404.02258](https://arxiv.org/abs/2404.02258) (Mixture-of-Depths),
[2507.10524](https://arxiv.org/abs/2507.10524) (Mixture-of-Recursions),
[1807.03819](https://arxiv.org/abs/1807.03819) (Universal Transformer).

Negative searches (arXiv API, run 2026-10-05) [verified]:
`(recursive reasoning | adaptive depth | dynamic depth | deep equilibrium) ∧ (molecular property |
QSAR | ADMET)` → **0 results**. `(ponder | halting | recursive refinement) ∧ (drug discovery |
bioactivity | IC50 | molecular property prediction)` → 3 results, none relevant.
`all:"PonderNet"` → 5 results, none tabular/chemical/scientific-regression.

Internal: `docs/drugrx-training/research/01-reasoning-models.md` §§1, 3 (the earlier survey this
brief deepens); `HYPOTHESES.md` H1, H2, H5, H7a, H11, H12; `PROGRESS.md` 2026-10-05 (CYP entry: the
ST-RAE metric, the leak, the bake-off, the blend result).
