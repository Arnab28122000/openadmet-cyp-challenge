# DeepSeek-style Mixture-of-Experts, and MoE at small scale

Research read 2026-10-05, for the CYP450 four-endpoint model (`talindrew-drugrx`). Every claim is
marked **[verified]** (read in the paper, the repo, or the code) or **[assumed]** (my inference, not
in a source). Equations are transcribed from the papers; where a fetch summarised rather than quoted,
the claim is marked as such.

---

## 0. Bottom line before the detail

1. **DeepSeek-style sparse MoE is the wrong family for this problem.** Its whole purpose is more
   *total* parameters at fixed *active* FLOPs, which pays off when you are compute-bound on a corpus
   of trillions of tokens. We are label-bound on 6,525 primary labels and we train on A100s where the
   dense model costs minutes. There is nothing for sparsity to buy.
2. **Multi-gate MoE (MMoE/PLE) is the right family to *consider*** — the user's instinct is correct —
   but its published effect sizes are tiny, it has hurt on public benchmarks, and the mechanism it
   fixes (bad local minima from task conflict, high across-seed variance) is **the same mechanism a
   16–32 member deep ensemble already fixes**. See §4.6 and §6.3; this is the strongest argument
   against, and it is not in the literature — it is specific to our setup.
3. **At matched parameters and a dense (non-sparse) gate, an MMoE layer is algebraically the dense
   layer plus one input-dependent scalar per hidden block.** That is a very small change in function
   class (§5.3). An effect of that size is below this programme's measurement resolution (~0.009
   macro ST-RAE band SD, per `scripts/cyp_curse.py`). **Predicted: null.**
4. The two *non-MoE* ideas this literature actually hands us are cheaper and more likely to move:
   **PLE's dynamic per-task loss weight** (eq. 10, §4.5) and **"Can Small Heads Help?"'s
   under-parameterised self-auxiliary heads** (§4.8). Both are ~20 lines inside the existing
   `vmap` loop.
5. Sparse top-k dispatch is **incompatible with `vmap`**; dense masked top-k is compatible but
   throws away all of MoE's efficiency, leaving only its pathologies (§6.5).

---

## 1. DeepSeekMoE (arXiv 2401.06066)

Source: <https://arxiv.org/abs/2401.06066>, HTML at <https://arxiv.org/html/2401.06066v1>.

### 1.1 The baseline it modifies (§2 of the paper) [verified]

A standard Transformer block:

```
u_t^l = SelfAtt(h_{1:T}^{l-1}) + h_t^{l-1}
h_t^l = FFN(u_t^l) + u_t^l
```

A conventional top-K MoE layer (GShard/Switch family) replaces the FFN with `N` experts and
activates `K`:

```
h_t^l = Σ_{i=1..N} ( g_{i,t} · FFN_i(u_t^l) ) + u_t^l                         (1)

g_{i,t} = s_{i,t}   if s_{i,t} ∈ TopK({s_{j,t} | 1 ≤ j ≤ N}, K)
        = 0         otherwise                                                 (2)

s_{i,t} = Softmax_i( (u_t^l)^T e_i^l )                                        (3)
```

`e_i^l` is expert `i`'s **centroid** (a learned `d_model`-dim vector); the router is a single linear
map with no bias. `g` is both the selection mask **and** the output weight.

### 1.2 Innovation 1 — fine-grained expert segmentation (§3.1) [verified]

Split every expert FFN into `m` smaller ones by **reducing the FFN intermediate hidden dimension to
1/m of its original size**, and activate `m·K` of the `m·N`:

```
h_t^l = Σ_{i=1..mN} ( g_{i,t} · FFN_i(u_t^l) ) + u_t^l                        (4)

g_{i,t} = s_{i,t}   if s_{i,t} ∈ TopK({s_{j,t} | 1 ≤ j ≤ mN}, mK)
        = 0         otherwise                                                 (5)

s_{i,t} = Softmax_i( (u_t^l)^T e_i^l )                                        (6)
```

The paper's own justification is **combinatorial**, quoted [verified]:

> "A typical top-2 routing strategy can yield (16 choose 2)=120 possible combinations. By contrast,
> if each expert is split into 4 smaller experts, the fine-grained routing strategy can yield
> (64 choose 8)=4,426,165,368 potential combinations."

and the cost claim, quoted [verified]:

> "while maintaining a consistent number of expert parameters and computational cost, we segment the
> experts with a finer grain"

**Accounting.** One expert of intermediate width `d_ff` has `≈ 2·d_model·d_ff` params. `m` experts of
width `d_ff/m` have `m · 2·d_model·d_ff/m = 2·d_model·d_ff` — identical. Activating `mK` of them
costs `mK · 2·d_model·d_ff/m = K · 2·d_model·d_ff` FLOPs — identical. So segmentation is
**exactly parameter- and FLOP-neutral**; it buys only routing resolution. [verified by the quoted
claim; the arithmetic is mine]

### 1.3 Innovation 2 — shared expert isolation (§3.2) [verified]

Reserve `K_s` experts as **always-on**, and reduce the number of *routed* activations by `K_s` to keep
cost constant:

```
h_t^l = Σ_{i=1..K_s} FFN_i(u_t^l)
      + Σ_{i=K_s+1..mN} ( g_{i,t} · FFN_i(u_t^l) )
      + u_t^l                                                                 (7)

g_{i,t} = s_{i,t}   if s_{i,t} ∈ TopK({s_{j,t} | K_s+1 ≤ j ≤ mN}, mK − K_s)
        = 0         otherwise                                                 (8)

s_{i,t} = Softmax_i( (u_t^l)^T e_i^l )                                        (9)
```

Quoted [verified]: "the number of activated experts among the other routed experts will be decreased
by K_s". Purpose quoted from the abstract [verified]: shared experts "aim at capturing common
knowledge and mitigating redundancy in routed experts".

### 1.4 The two balance losses (§3.3) [verified]

**Expert-level balance loss** — explicitly "to mitigate the risk of routing collapse" [verified
quote]:

```
L_ExpBal = α₁ · Σ_{i=1..N'} f_i · P_i

f_i = (N' / (K'·T)) · Σ_{t=1..T} 1( token t selects expert i )
P_i = (1/T) · Σ_{t=1..T} s_{i,t}

N' = mN − K_s      (number of routed experts)
K' = mK − K_s      (number of routed activations)
```

`f_i` is the *normalised fraction* of tokens routed to expert `i` (it equals 1 under perfect balance,
because the `N'/(K'T)` factor divides out the `K'T/N'` expected assignments). `P_i` is the mean
router probability. The product `f·P` is minimised when both are uniform; it is differentiable
through `P` only — `f` is a hard count and carries no gradient. [verified equations; the "gradient
flows only through P" reading is standard and is mine — **[assumed]**]

**Device-level balance loss** — for `D` device groups `E_1..E_D` partitioning the routed experts:

```
L_DevBal = α₂ · Σ_{i=1..D} f'_i · P'_i

f'_i = (1/|E_i|) · Σ_{j ∈ E_i} f_j
P'_i = Σ_{j ∈ E_i} P_j
```

Rationale, paraphrased from the paper [verified]: the device-level loss "encourage[s] balanced
computation across devices" **without** enforcing strict expert-level balance, because excessive
balance constraints compromise performance. `α₂ > α₁` in practice. **This is purely a distributed-
training concern and is irrelevant to us** (one GPU, one device).

### 1.5 Configurations [verified]

| | DeepSeekMoE 2B | 16B | 145B |
|---|---|---|---|
| layers / d_model / heads | 9 / 1280 / 10 | 28 / 2048 / 16 | 62 / 4096 / 32 |
| experts (shared + routed) | 1 + 63 | 2 + 64 | 4 + 128 |
| `m` / `K_s` / `mK` | 4 / 1 / 8 | 4 / 2 / 8 | 8 / 4 / 16 |
| total → activated params | 2.0B → 0.3B | 16.4B → 2.8B | 144.6B → 22.2B |

Expert parameters are held at 16× a standard FFN, activated at 2× a standard FFN. [verified]

### 1.6 What the ablations actually say

- GShard baseline (top-2 of 16): **Pile loss 1.867**; full DeepSeekMoE 2B: **1.808**. [verified]
- Shared-to-routed **ratio barely matters**, quoted [verified]: "different ratios of the shared
  experts and routed experts do not significantly impact the performance, and 1, 2, and 4 shared
  experts achieve a Pile loss of 1.808, 1.806, and 1.811, respectively."
- A clean incremental "+segmentation / +shared expert" row-by-row table with full benchmark numbers
  is **not** in the paper — the ablations are in Figure 3 and prose, so I could not verify
  per-component deltas. **[verified that the table does not exist in that form]**

**Reading for us:** a 3.2% relative loss improvement on a 100B-token LM run, from a mechanism whose
own key hyperparameter is flat across a 4× range. There is no claim anywhere in this paper about
small data or few examples.

---

## 2. DeepSeek-V3 (arXiv 2412.19437) + Loss-Free Balancing (arXiv 2408.15664)

Sources: <https://arxiv.org/abs/2412.19437>, <https://arxiv.org/html/2412.19437v2>;
<https://arxiv.org/abs/2408.15664>, <https://arxiv.org/html/2408.15664v1>.

### 2.1 Sigmoid gate, then renormalise (not softmax) [verified]

```
s_{i,t} = Sigmoid( u_t^T e_i )                                (affinity, per-expert, independent)

g'_{i,t} = s_{i,t}   if (s_{i,t} + b_i) ∈ TopK({s_{j,t} + b_j | 1 ≤ j ≤ N_r}, K_r)
         = 0         otherwise

g_{i,t} = g'_{i,t} / Σ_j g'_{j,t}                             (renormalise over the selected K_r)
```

Two changes from DeepSeekMoE v1: (a) **sigmoid instead of softmax** for the affinity, so affinities
are independent per expert and a bias added to one does not implicitly move the others; (b) the
gating values are **renormalised after top-k** so they sum to 1 over the selected experts. V3's
config: `N_s = 1` shared expert, `N_r = 256` routed, `K_r = 8` activated. [verified]

### 2.2 Auxiliary-loss-free load balancing — the exact update rule

This is the detail the task asked for, and it comes from the companion paper (2408.15664), which V3
cites for it.

**The bias enters only the selection, never the weight.** Quoted from V3 [verified]:

> "the bias term is only used for routing. The gating value, which will be multiplied with the FFN
> output, is still derived from the original affinity score s_{i,t}."

And from Loss-Free Balancing [verified]: the bias `b_i` "is only used to adjust the routing strategy
by influencing the top-K selection" and is **not** added to `g_{i,t}`.

**The update rule** [verified]:

```
e_i = c̄_i − c_i               # load violation error: expected count − actual count for expert i
b_i ← b_i + u · sign(e_i)      # u = update rate / "speed" parameter
```

So it is a **sign-only, fixed-step controller** — not proportional, not a gradient. An overloaded
expert has `c_i > c̄_i`, so `e_i < 0`, so `b_i` decreases by exactly `u`; an underloaded expert's
`b_i` increases by exactly `u`. V3 states the same thing in words: decrease `b_i` by `γ` if expert
`i` is overloaded, increase by `γ` if underloaded. [verified]

**Algorithm 1, per training batch** [verified]:

```
b_i ← 0 for all i
for each batch:
    route using biased scores  (s_i + b_i) for selection, s_i for weighting
    c_i ← number of tokens assigned to expert i in this batch
    c̄_i ← (total assignments) / N_r          # the uniform expectation
    e_i ← c̄_i − c_i
    b_i ← b_i + u · sign(e_i)
```

**Speed parameter values** [verified]:

- Loss-Free Balancing ablation: `u = 0.001` is optimal. `u = 0.0001` gives **poor balance in the
  early stage**; `u = 0.01` causes **late-stage fluctuation and performance degradation**.
- DeepSeek-V3: `γ = 0.001` for the first 14.3T tokens, then `γ = 0.0` (bias frozen) for the
  remainder.

**The metric they tune against** [verified]:

```
MaxVio = ( max_i Load_i − mean_i Load_i ) / mean_i Load_i
```

reported as `MaxVio_global` (over the validation set) and `MaxVio_batch` (per training batch).

**Scale of the evidence** [verified]: 1B params / 100B tokens and 3B params / 200B tokens, 9–11 MoE
layers, 64 routed experts, 6 activated. The auxiliary-loss baseline used `α = 0.001`.

**Expert Choice routing was excluded from the comparison because of "future token leakage"**
violating the causal constraint. [verified] — see §8.2; this is the canonical instance of "the router
sees something it must not".

### 2.3 The complementary sequence-wise balance loss [verified]

V3 keeps a *vestigial* auxiliary loss, computed **per sequence** rather than per batch:

```
L_Bal = α · Σ_{i=1..N_r} f_i · P_i                                           (17)

f_i = (N_r / (K_r·T)) · Σ_{t=1..T} 1( s_{i,t} ∈ TopK({s_{j,t} | 1≤j≤N_r}, K_r) )   (18)

s'_{i,t} = s_{i,t} / Σ_{j=1..N_r} s_{j,t}                                    (19)

P_i = (1/T) · Σ_{t=1..T} s'_{i,t}                                            (20)
```

`T` is the number of tokens **in one sequence**. Purpose, quoted [verified]: "sequence-wise balance
loss encourages the expert load on each sequence to be balanced." `α` is described in v2 as "an
extremely small value"; the v1 HTML gives **α = 0.0001**. [verified both; treat 1e-4 as the number]

Note eq. 18's indicator is on the **un-biased** `s_{i,t}` top-k, and eq. 19/20 renormalise the
sigmoid affinities before averaging. [verified from the fetched text; worth re-checking against the
PDF if implementing]

### 2.4 Node-limited routing, and no token dropping [verified]

Each token is "sent to at most M nodes", `M = 4`, chosen by the sum of the top-`K_r/M` affinities per
node. V3 "does not drop any tokens during training". Both are distributed-systems constraints —
**irrelevant to us**.

### 2.5 Why the bias/gate separation is the load-bearing idea

An auxiliary balance loss adds a gradient that is *not* the task gradient; it trades task loss for
balance. The bias controller changes **which** experts win the top-k without changing the **weight**
any winner receives, so the task gradient is untouched. That is the whole trick, and it is the one
part of the DeepSeek line that is conceptually portable to small scale: *if* you ever run a sparse
top-k router, prefer the bias controller over a balance loss. [the mechanism is **[verified]**; the
"portable to small scale" judgement is **[assumed]**]

---

## 3. Does MoE help at small scale and small data?

Searched hard. The honest summary: **there is no positive evidence for sparse MoE below ~10k
examples, and there is direct negative evidence.** The strongest four results:

### 3.1 The one clean dataset-size experiment: ST-MoE [verified]

"ST-MoE: Designing Stable and Transferable Sparse Expert Models" (arXiv 2202.08906,
<https://arxiv.org/html/2202.08906v2>) fine-tunes a sparse and a dense model of matched pretraining
quality on two SuperGLUE tasks of very different size:

| task | train examples | winner |
|---|---|---|
| Commitment Bank (CB) | **250** | **dense** beats sparse on validation |
| ReCoRD | **138,000** | sparse beats dense significantly |

The sparse model "converges faster to 100% train accuracy on both tasks but fails to generalize on
the smaller dataset". The paper states plainly: **"sparse models are prone to overfit."** [verified]

Its mitigations [verified]: update only the **non-MoE** parameters when fine-tuning ("works about as
well as updating all the parameters"); fine-tuning only the MoE parameters (≈80% of params) "results
in a drastic quality reduction"; sparse models want **smaller batches and higher learning rate** than
dense ones; expert-specific dropout above a global 0.1 helps modestly.

**This is the closest analogue in the literature to our situation, and it says dense wins.**
We have 6,525 primary labels — 26× CB, 0.05× ReCoRD. [the interpolation is **[assumed]**]

### 3.2 NLLB: at matched compute, MoE loses to dense, and low-resource tasks overfit [verified]

"No Language Left Behind" (arXiv 2207.04672, §6.2, read from the PDF) is the best *multi-task* MoE
evidence there is, because it is a mixture of thousands of tasks with wildly different label counts —
structurally our problem. Verbatim findings:

- "we see that a vanilla MoE model does not outperform the corresponding 1.3B dense model on the
  ablation benchmark."
- "for the MoE-64 model (computationally similar to the dense 1.3B model), we see neutral or worse
  performance compared to the dense 1.3B model."
- "As we increase computational cost per update, **the propensity for low or very low-resource pairs
  to overfit increases thus causing performance to deteriorate.**"
- "the model continues to face significant overfitting when trained for 100k updates. This is
  unsurprising, as **iterating over a small training set with large capacity causes overfitting**."
- Regularisation is what rescues it: adding overall dropout flips a **−0.1 chrF++** relative decline
  into **+0.9** for very-low-resource pairs out of English.

Their two purpose-built regularisers are worth knowing, because the second is a shared expert by
another name:

**Expert Output Masking (EOM)** — zero the expert outputs for a fraction `p_eom` of input tokens
*before* combining them, so the token takes only the residual path. Gains over vanilla-MoE-with-
dropout: +0.4 chrF++ into English, +0.6 non-English, and **larger on low/very-low resource** (+0.6,
+0.9 out of English). Sweep `(p_drop, p_eom) ∈ {0.1,0.2,0.3}²`. [verified]

**Conditional MoE Routing (CMR)** — a sigmoid gate interpolating a shared dense FFN against the MoE
block, with a *budget* penalty [verified]:

```
g(x_t) = sigmoid( W_CMR · x_t )
CMR(x_t) = (1 − g(x_t)) · FFN_shared(x_t) + g(x_t) · MoE(x_t)
L_CMR = (1/T) · Σ_t | g(x_t) − p |                                            (12)
```

with `p = 0.8`, plus `p_cmr` = fraction of gates force-zeroed per minibatch. At `p = 0`
the model is dense; at `p = 1` it is free to always use the MoE. CMR gave +0.8 / +0.7 chrF++, larger
again on low-resource (+0.7, +1.0), at **18–25% extra training cost**, and they shipped EOM instead
because CMR was not worth the overhead. [verified]

CMR is, functionally, DeepSeek's shared-expert isolation with an explicit budget — arrived at
independently, and motivated *explicitly as regularisation for the low-resource tasks*. **[the
equivalence reading is assumed]**

### 3.3 Tabular: the controlled comparison is a null [verified]

"(GG) MoE vs. MLP on Tabular Data" (arXiv 2502.03608, <https://arxiv.org/html/2502.03608v1>) is the
only head-to-head MoE-vs-MLP study on tabular data I found with a real benchmark suite: **38 datasets**
(28 from Grinsztajn et al. 2022, 10 from Gorishniy et al. 2022), sizes from **~2,000 to 723,412**
training rows, median 15–18 experts.

Its own conclusion, quoted [verified]:

> "the performance gap between GG MoE and E+MLP is small that we cannot confidently declare a
> significant difference."

Their ranking procedure "assigns the same rank to models where the difference between mean scores is
smaller than the standard deviation" — i.e. deliberately conservative. The real result is
**parameter efficiency**: MLP median 0.88M params vs MoE median 0.08M, GG MoE 0.11M, at
indistinguishable accuracy. [verified]

So: on tabular data, MoE is a *compression* result, not an accuracy result. We do not need
compression.

Other MoE-on-tabular hits (arXiv API search for `"mixture-of-experts" AND "tabular"`, 22 results)
are overwhelmingly single-application papers — RowNet (2606.04445), IME (2206.02107), MoDE
(2402.00893), Xiaomi-TabLDM (2609.03880), Tabby (2503.02152). **None reports a data-scale ablation**,
and IME's claim is interpretability-at-equal-accuracy. [verified from the search results; I read only
abstracts for these]

### 3.4 Scaling laws say the advantage *grows* with scale — i.e. it is smallest at ours [verified]

"Scaling Laws for Fine-Grained Mixture of Experts" (arXiv 2402.07871) introduces **granularity `G`**
(our `m`) as a scaling-law variable and finds the MoE/dense efficiency gap "widens as we scale up the
model size and training budget", and that sizing experts to mirror the FFN "is not optimal at almost
any computational budget". The paper's framing is about billions of params and hundreds of billions
of tokens. **A law whose stated property is "advantage widens with scale" is, read backwards, a law
that says the advantage is minimal at the small end.** [the forward claims are verified; the
backwards reading is **[assumed]**]

### 3.5 Known failure modes, with mechanism

| failure | mechanism | source |
|---|---|---|
| **Routing collapse** | the router's own gradient is self-reinforcing: experts that win get trained, get better, win more. DeepSeekMoE's expert-level balance loss exists explicitly "to mitigate the risk of routing collapse". | 2401.06066 [verified] |
| **Representation collapse** | X-MoE's Jacobian decomposition: token representations get "updated toward a linear combination of the expert embeddings". With `N` experts spanning at most an `N`-dim subspace and `N ≪ d`, "the mechanism renders the Transformer hidden vector collapsed to an N-dimensional subspace", and tokens routed to the same expert get "pushed closer". | arXiv 2204.09179, <https://arxiv.org/html/2204.09179v3> [verified] |
| **Router instability across runs** | X-MoE's inter-run consistency analysis: baseline SMoE "converged to different routing solutions across multiple runs". Their fix: project to `d_e = N/2` dims, L2-normalise onto a hypersphere, learnable temperature — `s_i = ((Wh)·e_i)/(‖Wh‖·‖e_i‖)`, `g(s_k) = exp(s_k/τ)/Σ exp(s_j/τ)`. | 2204.09179 [verified] |
| **Numerical instability from the router's exponentials** | ST-MoE: sparse models "contain more exponential functions due to the routers"; bfloat16 has "up to 65,536x worse roundoff errors than float32". Fix: router z-loss `L_z(x) = (1/B) Σ_i (log Σ_j e^{x_j^{(i)}})²`. Switch Transformer's fix: "cast the router input to float32 precision". | 2202.08906, 2101.03961 [verified] |
| **Overfitting on small downstream data** | §3.1, §3.2. Switch Transformer: more params than the dense baseline "can lead to more severe overfitting on these smaller downstream tasks"; fix is expert dropout — "a smaller dropout rate (0.1) at non-expert layers and a much larger dropout rate (0.4) at expert layers". | 2101.03961 [verified] |
| **Gates that never separate** (the quiet one) | PLE §5.3: in MMoE/ML-MMoE "there is no zero-weight for all experts", gates land on "much similar weights", and "it is hard for MMOE and ML-MMOE to converge to the structure of CGC and PLE without prior knowledge in practice, despite the existence of theoretical possibility." | PLE RecSys 2020 [verified] |

That last row is the pitfall most likely to bite *us*, because with a dense softmax gate the
degenerate solution is not "one expert wins" but **"all gates ≈ 1/E", which is exactly the dense
model** — and it produces a clean null that looks like a model-class answer when it is really "the
mechanism never engaged". See §8.1.

### 3.6 The <10k-examples question, answered directly

**Evidence for:** none that I could find for *sparse, SGD-trained, load-balanced* MoE. The only
genuinely small-data MoE literature is the **classical** one: Jacobs & Jordan's mixture of experts
and hierarchical ME were fit by EM on small datasets as *statistical* models, and "Fast Deep Mixtures
of Gaussian Process Experts" (arXiv 2006.13309, *Machine Learning* 2024) reports MoE-of-GPs
"competitive cost on low-dimensional and small data sets". [verified from the abstract] That is a
different algorithm family — closed-form/EM responsibilities, not a top-k router trained by SGD
against a balance loss — and it does not transfer.

**Evidence against:** ST-MoE at 250 examples (dense wins, §3.1); NLLB at matched compute (dense ties
or wins, and low-resource pairs deteriorate, §3.2); the tabular suite's explicit non-significance
(§3.3); MMoE *losing* to single-task on UCI census-income (§4.6).

**Verdict [assumed, but strongly supported]:** below ~10k examples per task, sparse MoE has no
demonstrated benefit and a documented overfitting liability. The sub-family that *is* defensible at
this scale is the **dense-gated multi-task** one (§4) — because there the "experts" are not a
capacity trick, they are a *parameter-sharing* trick.

---

## 4. Multi-task MoE: MMoE and PLE

This is the relevant branch, and it is a different idea from DeepSeek's despite the shared name:
**the gate is per-task, not per-token, and it is dense (softmax over all experts), not top-k.** No
load balancing, no capacity factor, no token dropping.

### 4.1 Shared-bottom baseline (MMoE eq. 1) [verified]

```
y_k = h_k( f(x) )                                                             (1)
```
`f` shared bottom, `h_k` task-`k` tower. **Our current `CypNet` is exactly this with
`h_k` = one row of a single `Linear(1024, 24)`** — i.e. the most extreme shared-bottom possible:
zero task-specific depth. [verified from `talindrew-drugrx/scripts/cyp_train.py:55-70`]

### 4.2 MMoE (Ma et al., KDD 2018) [verified, read from the PDF]

Original MoE:
```
y = Σ_{i=1..n} g(x)_i · f_i(x) ,      Σ_i g(x)_i = 1                          (5)
```

MMoE replaces the shared bottom `f` with an MoE layer and adds **one gate per task**:

```
y_k = h_k( f^k(x) )                                                           (6)

f^k(x) = Σ_{i=1..n} g^k(x)_i · f_i(x)                                         (7)

g^k(x) = softmax( W_{g^k} · x ) ,   W_{g^k} ∈ R^{n×d}                         (8)
```

`n` = number of experts, `d` = input feature dimension, experts are "identical multilayer perceptrons
with ReLU activations", gates are "simply linear transformations of the input with a softmax layer".
**All experts are used by every task, with task-specific weights.** [verified]

**Synthetic task-correlation experiment** [verified] — the paper's central evidence, and worth
reproducing exactly because it is the cleanest MTL-architecture experiment in the literature:

```
# generate two regression tasks with controlled correlation p
u1, u2 ∈ R^d  orthogonal unit vectors
w1 = c·u1 ,  w2 = c·( p·u1 + sqrt(1 − p²)·u2 )                                (2)
x ~ N(0, I_d)
y1 = w1ᵀx + Σ_{i=1..m} sin( α_i · w1ᵀx + β_i ) + ε1                           (3)
y2 = w2ᵀx + Σ_{i=1..m} sin( α_i · w2ᵀx + β_i ) + ε2                           (4)
ε ~ N(0, 0.01)
```
`cos(w1,w2) = p` is called "task correlation". `d = 100`; both MoE models use **8 experts, each a
single-layer net of hidden size 16**; towers are single-layer size 8; 200 runs per setting; LR grid
`[1e-4, 1e-3, 1e-2]`.

Results [verified, quoted]:
1. all models do better at higher correlation;
2. "in the extreme case where the two tasks are identical, there is almost no difference in
   performance between the MMoE model and the OMoE model; when the correlation between tasks
   decreases, however, there is an obvious degeneration of performance for the OMoE model while there
   is little influence on the MMoE model";
3. "Both MoE models are better than the Shared-Bottom model in all scenarios **in terms of average
   performance**."

**The trainability finding — the most important sentence in the paper for us** [verified, quoted]:

> "the performance variances of Shared-Bottom model are much larger than those of the MoE based
> model. This means that Shared-Bottom models in general have much more poor quality local minima
> than the MoE based models do. … **it's worth to observe that the lowest losses of all the three
> models are comparable.** This is not surprising as neural networks are theoretically universal
> approximator. With enough model capacity, there should exist a 'right' Shared-Bottom model that
> learns both tasks well. However, note that this is the distribution of 200 independent runs."

**MMoE's measured benefit is a reduction in across-seed variance, not an increase in attainable
quality.** Hold that thought for §6.3.

**Parameter matching, exactly as MMoE did it** [verified, quoted]:

> "the total number of model parameters in the shared experts and the towers is
> 100 × 16 × 8 + 16 × 8 × 2 = 13056. For the baseline Shared-Bottom model, we still set the tower
> network as a single-layer network with size=8. We set the single-layer shared bottom network with
> size 13056/(100 + 8 × 2) ≈ 113."

This is the right standard and §5 generalises it.

### 4.3 CGC — Customized Gate Control (PLE §4.1) [verified, read from the PDF]

CGC adds **task-specific experts alongside shared experts**, and a task's tower may only see its own
experts plus the shared ones:

```
w^k(x) = Softmax( W_{g^k} · x ) ,   W_{g^k} ∈ R^{(m_k + m_s) × d}             (3)

S^k(x) = [ E^T_{(k,1)}, …, E^T_{(k,m_k)}, E^T_{(s,1)}, …, E^T_{(s,m_s)} ]^T   (4)

g^k(x) = w^k(x) · S^k(x)                                                      (2)

y_k(x) = t_k( g^k(x) )                                                        (5)
```

`m_s` shared experts, `m_k` experts private to task `k`, `d` = input dim, `t_k` = task-`k` tower.
Quoted [verified]: "the parameters of shared experts are affected by all tasks while parameters of
task-specific experts are only affected by the corresponding specific task"; and "Compared with
MMOE, CGC **removes connections** between a task's tower network and task-specific experts of other
tasks".

### 4.4 PLE — the multi-level generalisation (PLE §4.2) [verified]

Stack `J` extraction networks. Each level has shared experts and per-task experts; the **shared**
experts also get their own gate that fuses *all* experts at that level; and the gate at level `j`
takes the **level-`j−1` fused output** as its selector instead of the raw input:

```
g^{k,j}(x) = w^{k,j}( g^{k,j−1}(x) ) · S^{k,j}(x)                             (6)
```

Quoted [verified]: "The calculation of weighting function, selected matrix, and gating network in PLE
are the same as that in CGC"; "parameters of different tasks in PLE are not fully separated in the
early layer as CGC but are separated progressively in upper layers"; the higher-level gates use the
lower-level fusion "as it may provide better information for selecting abstract knowledge".

### 4.5 PLE's loss engineering — the part we should copy [verified]

```
L(θ_1,…,θ_K, θ_s) = Σ_{k=1..K} ω_k · L_k(θ_k, θ_s)                            (8)

L_k(θ_k, θ_s) = ( Σ_i δ^i_k )^{-1} · Σ_i δ^i_k · loss_k( ŷ^i_k, y^i_k )       (9)

ω_k^{(t)} = ω_{k,0} × γ_k^t                                                   (10)
```

Eq. 9 is the **heterogeneous sample space** fix: `δ^i_k ∈ {0,1}` marks whether sample `i` is in task
`k`'s sample space, and each task's loss is normalised by its own count. **We already do exactly
this** — `scripts/cyp_curse.py` masks with `am`/`lm` and divides by `am.sum().clamp(min=1)`.
[verified against the code]

Eq. 10 is the **dynamic loss weight**: `t` is the epoch, `ω_{k,0}` and `γ_k` are hyperparameters, "it
is observed that each task may have different importance at different training phases". **We do not
do this** — we use a single static `--aux-weight 0.3` shared across all 20 auxiliary heads.
[verified against `scripts/cyp_curse.py:59`] This is the cheapest untested lever in this whole
document.

### 4.6 The effect sizes, stated honestly [verified]

Tencent industrial video dataset (46.9M users, **0.995 billion samples**), CTR/VCR task group:

| model | CTR AUC | VCR MSE | MTL gain CTR | MTL gain VCR |
|---|---|---|---|---|
| Single-task | 0.7379 | 0.1179 | — | — |
| Cross-Stitch | 0.7220 | 0.1158 | −0.0158 | +0.0021 |
| MMoE | 0.7382 | 0.1175 | **−0.0001** | +0.0004 |
| ML-MMoE | 0.7378 | 0.1169 | −0.0009 | — |
| CGC | 0.7398 | 0.1155 | +0.0020 | +0.0021 |
| PLE | **0.7406** | **0.1150** | **+0.0027** | — |

Public datasets:

| model | Census task1 AUC | Census task2 AUC | Ali-CCP CTR | Ali-CCP CVR |
|---|---|---|---|---|
| Single-task | 0.9445 | 0.9923 | 0.6088 | 0.6040 |
| MMoE | **0.9393** | 0.9928 | 0.6094 | **0.5738** |
| PLE | 0.9522 | 0.9945 | 0.6112 | 0.6097 |

Read those two rows carefully. **MMoE is *worse* than single-task on census task 1 (−0.0052 AUC) and
catastrophically worse on Ali-CCP CVR (−0.0302 AUC).** PLE's own best gains over single-task are
+0.0027 AUC on a billion samples and +0.0078 on census (~300k rows). On the VTR/VCR group, quoted
[verified]: "MMOE improves both tasks over single-task but the improvement is not significant, while
ML-MMOE improves VTR but hurts VCR."

PLE's setup for the public data [verified]: "a three-layer MLP network with RELU activation and
hidden layer size of [256, 128, 64] for each task in both MTL models and single-task model."

**AdaTT** (Meta, KDD 2023, arXiv 2304.04959) reproduces the same magnitudes: MMoE −0.481% and PLE
−0.448% test normalised entropy vs shared-bottom, AdaTT −0.508%; on five diverse tasks the spread
between MMoE and PLE is 0.03–0.07 percentage points. It also reports the task-relatedness effect
[verified]: "when tasks are more closely related, negative transfer is less severe, and both tasks
benefit from a higher level of shared knowledge" — i.e. **the more correlated your tasks, the less
MMoE/PLE can give you.** Its fairness protocol: "all models have 3 hidden layers activated by ReLU",
"all gating modules use a 1-layer MLP with softmax activation", with "a limit on the total number of
experts per level of fusion" to keep computation comparable.

### 4.7 Task-specific vs shared experts — what the evidence supports

- **Shared-only with per-task gates (MMoE)**: gates empirically fail to separate (§3.5 last row);
  gains ≈ 0 or negative on public data.
- **Explicit task-specific + shared (CGC)**: this is where the gains appear, and the mechanism is a
  *hard architectural prior*, not something learned. PLE's own explanation: MMoE *could* represent
  CGC but does not converge to it.
- **Progressive separation (PLE)**: adds another +0.0007 AUC over CGC on CTR. Marginal.

**So the ordering of prior probability for us is CGC > PLE > MMoE > sparse MoE.** [assumed, from the
numbers above]

### 4.8 The orthogonal result that may matter more: "Can Small Heads Help?" [verified]

Wang et al., WWW 2022 (Google), read from the PDF. Abstract, quoted: "larger models are not
necessarily better in terms of multi-task generalization"; "over-parameterizing the task-specific
sub-networks may give the model too many 'degrees of freedom' and impedes the generalizability of the
shared representation."

Synthetic evidence [verified]: as task-specific depth grows 0 → 5 layers the test Pareto frontier
improves; from **5 → 9 it "rapidly deteriorates"**, and the same happens when widening instead of
deepening.

The method — an **under-parameterised self-auxiliary head per task**, used in training and discarded
at inference:

```
f_t(x; θ_sh, θ_t)  = f_t( h(x; θ_sh); θ_t )                                   (2)
f_t^a(x; θ_sh, θ_t^a) = f_t( h(x; θ_sh); θ_t^a )     # θ_t^a ≪ θ_t            (3)

L̂(θ) = Σ_{t=1..T} w_t · ( L̂_t(θ_sh, θ_t) + γ · L̂_t(θ_sh, θ_t^a) )          (4)
```

Mechanism, quoted [verified]: "the shared representation … is 'forced' toward learning a
representation which suits both fully parameterized and under-parameterized task-specific
transformations", acting as implicit regularisation so "sharing happens in the shared layers as much
as possible". Cost: "minimal additional training cost and no additional serving cost."

**Why this is interesting for us:** our towers are already minimal (one linear row each), so the
paper's warning does not indict our current model — but its *method* is a drop-in extra loss term
that costs almost nothing and has never been tried here. And it predicts that the MMoE/PLE move
(adding per-task experts, i.e. per-task capacity) is the direction that **hurts** multi-task
generalisation at our scale. [the application to us is **[assumed]**]

---

## 5. What a fair MoE comparison requires

### 5.1 The four quantities, and which comparison each one answers

| hold fixed | what varying the architecture then measures | honest? |
|---|---|---|
| **Total parameters** | whether the *same* budget is better spent as a conditional model | **Yes — this is the honest comparison for us.** |
| **Active parameters / FLOPs per example** | whether sparsity buys capacity for free | Only honest if you *also* report the matched-total-params dense model. |
| **Training steps / epochs / token budget** | nothing on its own | Necessary, never sufficient. |
| **Tuning budget** (HP search trials, seeds) | — | Necessary. Sparse models want different LR/batch/dropout (ST-MoE [verified]); giving the MoE a sweep and the dense model your old defaults is the most common way to fake a win. |

### 5.2 The misleading comparison, named

> "Our MoE has 4× the parameters of the dense baseline at the same FLOPs per example, and it wins."

This is the standard MoE claim and it is **honest only when compute is the binding constraint**. On
6,525 labels and an A100, compute is not binding — we can simply train the 4×-parameter *dense*
model, which is the comparison that answers "is this extra capacity useful?" If the MoE beats the
FLOP-matched dense model but ties the **parameter-matched** dense model, the result is "more
parameters helped", not "routing helped". DeepSeek's papers are careful about this (they report
GShard×1.5 and Dense×16 alongside) [verified]; most application papers are not.

The symmetric trap: declaring MoE a failure because it tied a dense model with 4× its active FLOPs.
That is a win for MoE on efficiency and should be reported as such.

### 5.3 The algebraic fact that decides it for our architecture [assumed — my derivation]

Take one layer of our trunk, written as a 2-layer expert: `W2 · act(W1 x)` with `W1 ∈ R^{H×d_in}`,
`W2 ∈ R^{d_out×H}`. Build an `E`-expert MMoE layer with the same total parameters by giving each
expert hidden width `H/E`:

```
dense:  y = W2 · act(W1 x)              = Σ_{i=1..E} W2^{(i)} · act(W1^{(i)} x)
MMoE :  y = Σ_{i=1..E} g_i(x) · ( W2^{(i)} · act(W1^{(i)} x) )
```

where `W1^{(i)}, W2^{(i)}` are the `i`-th row-/column-block. **They are the same expression, with the
`i`-th block's contribution scaled by a learned input-dependent scalar `g_i(x)`.** The dense layer is
the special case `g ≡ 1`; the MMoE layer is the dense layer plus `E` scalars of input-dependent,
per-task gain modulation per layer.

Three consequences:

1. A matched-parameter, dense-gated MMoE is a **tiny** enlargement of the dense function class — and
   because `g` is a softmax it is constrained to `Σ g_i = 1`, so it is really *reweighting* a fixed
   budget, not adding any.
2. Matched total params and matched FLOPs are the **same constraint** here, because a dense gate
   computes every expert. There is no efficiency axis to trade on. (Gate params: with MMoE eq. 8
   gating on the raw input, `24 tasks × 2500 × E`; at `E=4` that is **240k params, 5% of our 4.69M
   model** — not negligible. Gating on the first hidden layer instead costs `24 × 1024 × 4 ≈ 98k`.)
3. The only route to a *capacity* gain is full-width experts, which is `E×` the parameters at `E×` the
   FLOPs under a dense gate — i.e. just a bigger model, which H4 already explored (and found wider
   was better, so this is not a crazy direction, but it is not an MoE result).

### 5.4 The protocol I would hold us to

- **Arms:** (a) current dense `CypNet`; (b) matched-total-params MMoE/CGC; (c) dense model with the
  MoE's *total* parameter count. Report all three. Arm (c) is what makes arm (b) interpretable.
- **Seeds:** ≥5 per arm, both arms re-run (the programme's T5 rule). MMoE's own claim is about the
  *distribution* over 200 runs, not one run — a 1-seed MMoE-vs-dense comparison cannot test MMoE's
  actual claim. [verified from §4.2]
- **Selection:** on the select split only; score once on the untouched holdout. `cyp_curse.py`'s
  four-candidate protocol (final / best-on-select / SWA / pred-avg) already does this.
- **Threshold:** mean macro ST-RAE improvement must exceed the pooled across-seed SD. The band SD is
  **0.009** [verified from `scripts/cyp_curse.py` docstring], so anything under ~0.01 is unreadable.
- **Per-endpoint verdicts: do not seek them.** ~240 scorable labels per endpoint on the fold gives an
  MDE of ~0.02 [verified from `HYPOTHESES.md` H12].
- **Diagnostics are mandatory, not optional** — §8.

---

## 6. What to actually implement here

### 6.1 The argument, explicitly: MMoE/PLE over DeepSeek-style sparse MoE — but neither, first

**For the user's hypothesis (multi-gate over our 24 heads is the relevant variant): yes, correct.**
Reasons: (i) sparse MoE's benefit is total-params-at-fixed-FLOPs, and we are not FLOP-bound; (ii)
sparse MoE's documented liability is overfitting below ~10k examples, which is exactly our regime
(§3.1–3.2); (iii) MMoE/PLE are dense-gated, so no load balancing, no capacity factor, no token
dropping, no `vmap` incompatibility; (iv) our task structure — 4 tightly-related primary endpoints at
6.5k labels plus 20 auxiliary heads at 47k labels, where the recovered Emax heads correlate with the
primary at only `|r| ≤ 0.56` [verified from `HYPOTHESES.md` H12] — is literally the "task groups of
differing relatedness" problem MMoE/PLE were built for.

**Against it, and I think decisively:** the gains in that literature are +0.003 AUC on a billion
samples (§4.6), MMoE has gone *negative* on two public datasets, and MMoE's mechanism of action is
across-seed variance reduction (§4.2) **which our 16–32 member ensemble already provides**. Plus
§5.3: at matched params it is a per-block gain modulation, which is a very small function-class
change.

**So: if anything, build CGC with two task groups, not MMoE with 24 gates.** 24 gates on 4+20 heads
means 24 softmaxes fit against, in the primary case, 6,525 labels; the primary group's four endpoints
are from the same dose-response fits on the same assay family and will not want different experts
from each other. Two groups (primary / auxiliary) is the only split the data clearly supports.

### 6.2 Minimal implementation — a 2-group CGC head block, parameter-matched

Keep the trunk, replace the single `Linear(1024, 24)` with a CGC block. Parameter-matched by
**dropping trunk layer 3** (−1.05M) and spending it on four width-256 experts (+1.05M).

```python
class CgcCypNet(nn.Module):
    """Trunk (2 layers, was 3) -> 2 shared + 2 group-specific experts -> 2 towers."""
    def __init__(self, in_dim, n_prim=4, n_aux=20, d=1024, d_e=256, dropout=0.15):
        super().__init__()
        self.body = nn.Sequential(                      # 2 layers now, not 3
            nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, d),      nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout),
        )
        mk = lambda: nn.Sequential(nn.Linear(d, d_e), nn.GELU())
        self.e_shared = nn.ModuleList([mk(), mk()])     # m_s = 2, touched by both groups
        self.e_prim   = nn.ModuleList([mk()])           # m_prim = 1, primary gradient only
        self.e_aux    = nn.ModuleList([mk()])           # m_aux  = 1, auxiliary gradient only
        self.g_prim = nn.Linear(d, 3)                   # PLE eq.3: W_g ∈ R^{(m_k+m_s) x d}
        self.g_aux  = nn.Linear(d, 3)
        self.t_prim = nn.Linear(d_e, n_prim)            # towers t_k, PLE eq.5
        self.t_aux  = nn.Linear(d_e, n_aux)

    @staticmethod
    def _fuse(h, gate, experts):
        # PLE eq.2/4: S^k(x) stacked, w^k(x) = softmax(W_g h), g^k = w^k . S^k
        S = torch.stack([e(h) for e in experts], dim=1)     # (B, m_k+m_s, d_e)
        w = torch.softmax(gate(h), dim=-1).unsqueeze(-1)    # (B, m_k+m_s, 1)
        return (w * S).sum(1)                               # (B, d_e)

    def forward(self, x):
        h = self.body(x)
        z_p = self._fuse(h, self.g_prim, list(self.e_shared) + list(self.e_prim))
        z_a = self._fuse(h, self.g_aux,  list(self.e_shared) + list(self.e_aux))
        # primary heads first, exactly as CypNet's contract requires
        return torch.cat([self.t_prim(z_p), self.t_aux(z_a)], dim=-1)
```

**The CGC property is enforced by the architecture, not a loss:** `e_prim` appears only in `z_p`, so
the auxiliary loss term cannot reach it, and vice versa. That is PLE's "removes connections" claim
[verified] made structural. No balance loss. No top-k. `vmap`-safe (§6.5).

Everything else in `cyp_curse.py` stays: the interval-hinge primary loss, the masked auxiliary MSE,
`aux_weight`, `stack_module_state`, the four-candidate selection.

### 6.3 The ensemble-redundancy argument, spelled out [assumed — this is my argument, not a citation]

MMoE's demonstrated benefit is that shared-bottom models "have much more poor quality local minima"
while "the lowest losses of all the three models are comparable" — a *distributional* claim over 200
independent runs [verified quote, §4.2].

We train **16–32 members under `vmap` with `randomness="different"`** [verified from
`scripts/cyp_curse.py:124-127`] and average their predictions. That is a deliberate, already-shipped
fix for exactly the failure MMoE fixes: the bad-local-minimum draws are averaged out. Worse for the
MoE case, each member will learn a *different* gate, so the ensemble mean is an average of 16
differently-partitioned piecewise functions — which is smoother and closer to dense than any single
member. **Deep ensembling attenuates the very specialisation the architecture is for.**

Two testable corollaries, if someone wants to probe this cheaply:
- At `--members 1`, MMoE/CGC should beat dense by more than it does at `--members 16`. If the gap
  shrinks with ensemble size, the mechanism is variance reduction and the ensemble already has it.
- Gate-weight agreement across the 16 members: if member gates are uncorrelated, the ensemble has
  averaged the routing away.

That single-member-vs-16-member contrast is, in my view, the most informative cheap experiment in
this whole document, and it costs one extra arm.

### 6.4 If someone insists on DeepSeek-style sparse routing anyway

Minimal faithful version with the V3 bias controller, dense-masked so it survives `vmap`:

```python
# params:  W1 (E, d, d_e), W2 (E, d_e, d_out), centroids (E, d)
# buffers: b  (E,)   <- NOT a parameter; updated by the controller, no grad

def moe_forward(h, W1, W2, cent, b, k):
    s = torch.sigmoid(h @ cent.T)                    # V3: sigmoid affinity, (B, E)
    idx = torch.topk(s + b, k, dim=-1).indices       # bias enters SELECTION only
    mask = torch.zeros_like(s).scatter_(-1, idx, 1.0)
    g = s * mask                                     # weight from the UNBIASED score
    g = g / g.sum(-1, keepdim=True).clamp(min=1e-9)  # V3: renormalise over the selected k
    hz = torch.einsum('bd,edh->beh', h, W1).gelu()
    y  = torch.einsum('beh,eho->beo', hz, W2)
    return (g.unsqueeze(-1) * y).sum(1), mask.sum(0) # also return per-expert counts

# controller, run once per optimiser step, OUTSIDE autograd (2408.15664 Algorithm 1)
with torch.no_grad():
    c_bar = counts.sum(-1, keepdim=True) / E
    b += u * torch.sign(c_bar - counts)              # u = 1e-3; sign-only, fixed step
```

Notes: `counts` must come back from the loss via `grad_and_value(..., has_aux=True)` — the signature
is `torch.func.grad_and_value(func, argnums=0, has_aux=False)` and `has_aux=True` makes it return
"tuple of gradients and tuple of the forward computation with output auxiliary objects" [verified
from the PyTorch docs]. `b` has shape `(K_members, E)` after stacking, and the controller update is a
plain elementwise op on that stacked tensor — no `vmap` needed for it.

**Skip the device-level loss entirely** (one GPU). If you add the sequence-wise loss, there is no
"sequence" here — the only sensible `T` is the minibatch, which makes it the ordinary Switch-style
batch balance loss, `α·N·Σ f_i P_i` with `α = 1e-2` [verified from 2101.03961], and at that point
prefer the bias controller and set `α = 0`.

### 6.5 MoE inside the `vmap`'d / `stack_module_state` ensemble loop

**Verdict: dense-masked MoE is fully compatible. Sparse *dispatch* MoE is not, and the reason is
structural, not a bug.** [mechanism verified from the PyTorch docs; the conclusion is mine]

What the docs say [verified, quoted]:
- `stack_module_state`: "All of the modules being stacked together must be the same (except for the
  values of their parameters/buffers)"; returns "two dictionaries that stack all of their parameters
  and buffers together, indexed by name", and the stacked params are new optimisable leaves.
- `vmap`: "We don't yet support `vmap` over data-dependent control flow"; "We do not (and will not)
  support vmap over a user-defined function that calls `.item()` on a Tensor"; `randomness` ∈
  {`"error"`, `"different"`, `"same"`}.

Therefore:

| MoE style | `vmap`-compatible? | why |
|---|---|---|
| Dense softmax gate (MMoE/CGC/PLE) | **Yes** | every expert runs on every row; all shapes static. |
| Dense **masked** top-k | **Yes** | `topk` → `scatter` → multiply are batchable with fixed shapes. You pay full dense FLOPs and get none of MoE's efficiency. |
| True sparse dispatch (gather each expert's tokens, run on a variable-size slice, scatter back) | **No** | the per-expert slice length is a function of tensor *values*, i.e. data-dependent shapes. |
| Capacity-factor dispatch with token dropping | **No** | needs `.item()`/dynamic shapes, and the drop decision is data-dependent control flow. |

Mechanical notes for the existing loop (all against `scripts/cyp_curse.py` [verified]):

1. **Experts must be fixed-shape tensors, not a `ModuleList` traversed by index.** A `ModuleList` is
   fine if you loop over *all* of it (static trip count) — that is what §6.2 does. For many experts,
   hold them as stacked parameters `(E, d_in, d_out)` and use one `einsum`; under `vmap` the member
   dim is just another leading axis, so it stays a single big GEMM.
2. **Batch statistics ARE available inside `loss_fn`.** `in_dims=(0, 0, None, None, …)` means each
   member sees the *whole* minibatch, so `f_i`, `P_i`, gate entropy and expert counts are all
   computable per member inside the vmapped function. Compute them there, return them via
   `has_aux=True`. Do **not** try to compute them outside — outside, you only have the stacked
   params, not the routing.
3. **The bias `b` must be a buffer, not a parameter,** or AdamW will apply weight decay to it and
   fight the controller. `stack_module_state` returns buffers separately; keep `b` there.
4. **`randomness="different"` already gives each member a different dropout mask and init.** With a
   router, it also gives each member a different *routing*. Expect per-member gate distributions to
   diverge; that is §6.3's point, and it is a measurement, not a bug.
5. **Memory.** Dense-computing `E` experts of width `d_e` for `B=4096` rows across `K=16` members
   costs `K·B·E·d_e·4` bytes per activation tensor. At `E=4, d_e=256`: 16·4096·4·256·4 B ≈ **67 MB**
   per layer — negligible. At full width `d_e=1024`: ≈ 268 MB — still fine, but the existing
   halve-and-retry OOM guard in `cyp_curse.py` will catch anything worse. Full-width experts also
   quadruple param memory: `16 × 4 × 4.69M × 4 B × 3` (params + 2 Adam states) ≈ **3.6 GB** on a
   40 GB A100 — comfortable.
6. **`base = members[0].to("meta")`** is unchanged; the meta module just needs the same structure.

### 6.6 The two things I would run before any MoE arm

Both are cheaper than an MoE arm, both come from this same literature, and neither has been tried:

**(a) PLE eq. 10, dynamic per-task loss weight** [verified mechanism]. We have one static
`aux_weight=0.3` for 20 heads of wildly different label counts. Minimal version:

```python
# per-epoch, per-head weight: start the aux heads high, decay them
w_aux = args.aux_weight * (args.gamma ** ep)          # gamma ~ 0.97
# or per-head, proportional to that head's label count on the batch
```

**(b) "Can Small Heads Help?" self-auxiliaries** [verified mechanism, eq. 2–5]. Add a second,
deliberately tiny head per task group, trained on the same labels, discarded at inference:

```python
self.head      = nn.Linear(d, n_heads)                       # unchanged
self.head_aux  = nn.Sequential(nn.Linear(d, 16), nn.GELU(),
                               nn.Linear(16, n_heads))        # under-parameterised
# loss: L = L(head) + gamma * L(head_aux),  gamma ~ 0.3 ; drop head_aux at predict time
```

Cost: ~`1024×16 + 16×24 ≈ 16.8k` params, one extra forward through a 16-wide bottleneck, zero
inference cost. If the "too many degrees of freedom" story is right for us, this moves the number; if
it does not, that is evidence our shared representation is already as good as the data allows — which
is the data-bound conclusion the programme keeps reaching.

---

## 7. Honest expected value

**This is very likely a null on 6,525 primary labels. I would predict null and register it as such.**

The case, stacked:

1. **Programme priors.** H2 (recursive encoder vs MLP) was a confirmed null at 0.0017. H4 (capacity
   and regularisation) was **refuted** — dropout hurt monotonically and the predicted
   over-parameterisation story "was backwards". [verified from `HYPOTHESES.md`] Model-class changes
   have not moved this task; features and metric exploitation have.
2. **Resolution.** The post-warmup validation band has mean 0.485 and **SD 0.009** over ~800 epochs
   [verified from `scripts/cyp_curse.py`]. Per-endpoint MDE is ~0.02 at ~240 scorable labels
   [verified from H12]. A per-block gain modulation (§5.3) will not produce a ≥0.01 macro effect.
3. **Literature effect sizes.** PLE over single-task: **+0.0027 AUC** on a billion samples; MMoE:
   **−0.0001**, and negative on two public datasets (§4.6). Tabular MoE vs MLP: explicitly
   non-significant (§3.3). Matched-compute MoE vs dense in a massively multi-task setting:
   "neutral or worse" (§3.2).
4. **Task correlation works against us.** AdaTT: "when tasks are more closely related, negative
   transfer is less severe, and both tasks benefit from a higher level of shared knowledge"
   [verified]; MMoE's synthetic result: at correlation 1.0 the multi-gate structure adds nothing over
   a single gate [verified]. Our four primary endpoints come from the same dose-response fits on the
   same assay family — high correlation, little conflict to resolve.
5. **The ensemble already bought the mechanism** (§6.3).
6. **Direction of risk is asymmetric.** The documented small-data failure mode is overfitting
   (ST-MoE, NLLB, Switch), so the realistic outcomes are "null" and "slightly worse", not "null" and
   "slightly better".

**Where a non-null could come from, if one does:** the **primary/auxiliary split**, not the
within-primary split. 6,525 primary labels against 47,360 auxiliary ones, correlated at only
`|r| ≤ 0.56`, is a genuine resource imbalance of the kind NLLB saw deteriorate, and CGC's hard
separation of a primary-only expert is exactly the instrument for it. That makes the §6.2 two-group
CGC the one arm worth the GPU hour — and it is a 2-gate architecture, not a 24-gate one.

**What I would not do:** build a 24-gate MMoE; build sparse top-k routing with a balance loss; or run
a single seed per arm and report the difference.

---

## 8. Pitfalls that make a naive implementation silently wrong

### 8.1 Expert collapse that *disguises itself as a dense model* — and produces a clean, wrong null

With a **dense softmax** gate the degenerate solution is not "one expert takes everything", it is
**`g_i(x) ≈ 1/E` for all `i` and all `x`** — at which point, by §5.3, the model *is* the dense model
and the experiment measured nothing. This is not hypothetical: PLE §5.3 found exactly it, quoted
[verified]:

> "there is no zero-weight for all experts in MMOE and ML-MMOE, which further shows that it is hard
> for MMOE and ML-MMOE to converge to the structure of CGC and PLE without prior knowledge in
> practice, despite the existence of theoretical possibility."

**Mandatory diagnostic, logged every eval:**

```python
w = softmax(gate(h))                       # (B, E)
ent  = -(w * w.clamp_min(1e-9).log()).sum(-1).mean()   # mean gate entropy; log(E) == uniform == dead
spread = w.mean(0).std()                   # across-expert spread of mean weight; ~0 == dead
cos_pa = F.cosine_similarity(w_prim.mean(0), w_aux.mean(0), dim=0)  # ~1 == groups route identically
```

If `ent ≈ log(E)` or `cos_pa ≈ 1`, **the arm did not test MoE and must not be reported as a
model-class result.** Record these numbers in `PROGRESS.md` alongside the score. With a **sparse
top-k** gate the opposite collapse applies — one expert wins everything — and `MaxVio`
(§2.2) is the metric for it.

### 8.2 The router seeing the label, or a proxy for it

Three distinct versions, in increasing subtlety:

1. **The textbook one.** Expert Choice routing was excluded from Loss-Free Balancing's comparison for
   "future token leakage" violating the causal constraint [verified] — because experts picking their
   tokens lets information flow across positions. Our analogue: never let the gate see `y`, the
   interval bounds `LO`/`HI`, or the mask `M`.
2. **The one we are actually exposed to.** `X = cat(xs[tr], xes)` — the training tensor is the
   **concatenation of the challenge molecules and the Veith external set**, and the label masks are
   *disjoint by construction*: `mb` has primary labels and no aux, `me` has aux and no primary
   [verified from `scripts/cyp_curse.py:87-106`]. Any input-conditioned gate will find
   "challenge-vs-Veith" as its highest-variance split long before it finds anything about CYP
   chemistry, because the two corpora differ in descriptor distribution. A gate that learns
   "which corpus" has learned a **near-perfect proxy for which heads are supervised** — it will route
   primary-bearing rows to the primary expert and learn nothing transferable, and at prediction time
   every row is a challenge row, so one expert goes unused. **Diagnose by scoring a logistic
   regression from the gate weights to the corpus indicator; if it separates, the gate is a corpus
   detector.** [the hazard is my analysis — **[assumed]**; the data layout is **[verified]**]
3. **Standardisation leakage into the gate.** `standardise_descriptors` fits stats on `x[tr]` and
   applies them to the external set too [verified]. That is correct for the trunk, but it means the
   external rows sit at systematically non-zero mean in the standardised space — which makes (2)
   *easier*, not harder.

### 8.3 Balance loss dominating, or silently doing nothing

Our primary loss is a masked interval hinge normalised by `lm.sum()`, so it lives at O(0.1–1)
[verified from the code]. A Switch-style balance loss is `α·N·Σ f_i P_i`, which at perfect balance
equals `α` and at full collapse approaches `α·N`. With `α = 1e-2` and `N = 4`, the term ranges over
0.01–0.04 — plausibly 10% of the total loss, which is enough to bend the primary fit for no benefit.
With `α = 1e-4` (V3's value) it cannot do anything at all on 4 experts. **Either value is defensible
and neither is safe without a log of both loss terms separately.** The DeepSeek line's own answer is
that you should not be doing this: use the bias controller, where the balance mechanism has **zero**
gradient footprint (§2.5). And at `E ≤ 4` with a dense gate, balance is not a problem that needs
solving at all.

### 8.4 Benchmarking against your own old defaults

ST-MoE [verified]: "Sparse and dense models have vastly different performance across different batch
sizes and learning rates", with sparse models preferring **smaller batch, higher LR**. Our defaults
are `lr 3e-4`, `batch 4096`, `dropout 0.15`, tuned for the dense model over several runs. Handing
those to an MoE arm and calling the result a null is not a test of MoE — and handing the MoE a sweep
the dense arm never got is not a test either. Sweep both or neither.

### 8.5 Mechanical traps specific to this codebase

- **Primary heads must stay first.** `loss_fn` slices `o[:, :n_prim]` and `o[:, n_prim:]`, and
  `predict_with` returns `[:, :n_prim]` [verified]. Any head block must preserve that order — §6.2's
  `torch.cat([t_prim(z_p), t_aux(z_a)])` does.
- **The bias buffer and AdamW.** `opt = AdamW(list(params.values()), weight_decay=0.01)` [verified] —
  a routing bias placed in `params` gets decayed toward zero every step, directly opposing the
  controller. Put it in `buffers`.
- **SWA over gate parameters is not obviously valid.** `cyp_curse.py` averages *weights* across the
  post-warmup band [verified]. Averaging two routers that have settled into different partitions
  gives a router that implements neither. The `pred-avg` candidate is safe; `swa` is not, for gated
  models. Report both and expect them to disagree, and treat a large SWA-vs-pred-avg gap as evidence
  the routing is wandering between basins rather than jittering in one.
- **`stack_module_state` requires identical modules** [verified quote] — so no per-member expert
  counts, and all members must be in the same train/eval mode.

---

## Sources

DeepSeek line
- [DeepSeekMoE: Towards Ultimate Expert Specialization (2401.06066)](https://arxiv.org/abs/2401.06066) · [HTML](https://arxiv.org/html/2401.06066v1)
- [DeepSeek-V3 Technical Report (2412.19437)](https://arxiv.org/abs/2412.19437) · [HTML v2](https://arxiv.org/html/2412.19437v2)
- [Auxiliary-Loss-Free Load Balancing Strategy for MoE (2408.15664)](https://arxiv.org/abs/2408.15664) · [HTML v1](https://arxiv.org/html/2408.15664v1)
- [Scaling Laws for Fine-Grained Mixture of Experts (2402.07871)](https://arxiv.org/abs/2402.07871)

Stability, collapse, small-data behaviour
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (2202.08906)](https://arxiv.org/html/2202.08906v2)
- [Switch Transformers (2101.03961)](https://arxiv.org/html/2101.03961v3)
- [On the Representation Collapse of Sparse Mixture of Experts (2204.09179)](https://arxiv.org/html/2204.09179v3)
- [No Language Left Behind (2207.04672)](https://arxiv.org/abs/2207.04672) — §6.2, read from the PDF
- [(GG) MoE vs. MLP on Tabular Data (2502.03608)](https://arxiv.org/html/2502.03608v1)
- [Interpretable Mixture of Experts (2206.02107)](https://arxiv.org/abs/2206.02107)
- [Fast Deep Mixtures of Gaussian Process Experts (2006.13309)](https://arxiv.org/abs/2006.13309)

Multi-task MoE
- MMoE — Ma et al., *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018, <https://doi.org/10.1145/3219819.3220007> (ACM blocks direct fetch; read from the PDF archived at <https://github.com/tangxyw/RecSysPapers> → `Multi-Task/[2018][Google][MMOE] …pdf`)
- PLE — Tang et al., *Progressive Layered Extraction (PLE)*, RecSys 2020 (best paper), <https://doi.org/10.1145/3383313.3412236> (same archive, `Multi-Task/[2020][Tencent][PLE] …pdf`)
- [AdaTT: Adaptive Task-to-Task Fusion Network (2304.04959)](https://arxiv.org/html/2304.04959v2)
- Wang et al., *Can Small Heads Help? Understanding and Improving Multi-Task Generalization*, WWW 2022, <https://doi.org/10.1145/3485447.3512021> (same archive)
- [DSelect-k: Differentiable Selection in the Mixture of Experts (2106.03760)](https://arxiv.org/abs/2106.03760)
- [Multi-Task Deep Recommender Systems: A Survey (2302.03525)](https://arxiv.org/html/2302.03525v2)

Tooling
- [torch.func UX limitations](https://docs.pytorch.org/docs/2.8/func.ux_limitations.html)
- [torch.func.stack_module_state](https://docs.pytorch.org/docs/2.8/generated/torch.func.stack_module_state.html)
- [torch.func.grad_and_value](https://docs.pytorch.org/docs/2.8/generated/torch.func.grad_and_value.html)

Code read for this brief (all paths absolute)
- `/Users/arnab/Documents/talindrew/talindrew-drugrx/scripts/cyp_curse.py`
- `/Users/arnab/Documents/talindrew/talindrew-drugrx/scripts/cyp_train.py` (`CypNet`, `build_loss`)
- `/Users/arnab/Documents/talindrew/docs/drugrx-training/HYPOTHESES.md` (H2, H4, H11, H12)
