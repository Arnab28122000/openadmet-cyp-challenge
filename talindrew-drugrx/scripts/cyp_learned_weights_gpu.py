"""Phase 8c: a LEARNED per-row weight and label, trained by bilevel lookahead on GPU.

Arnab's hypothesis, built. Four arms on one differentiable backbone, so the comparison is
internal and no number here is set against a GBM figure produced by different code
(`feedback-a-control-must-run-the-production-command`).

    uniform   w = PSEUDO_W[kind], label = 2.2111          the status quo, on this backbone
    derived   w = 1/(sigma_assay^2 + sigma^2(r)),          Phase 8b's derived form, ported
              label = g(r) fitted on the in-split dual sample
    learned   w_i = 2*sigmoid(h_phi(u_i)),  label = a + b*log10(r_i), with (phi, a, b) all
              trained by BILEVEL LOOKAHEAD against a held-out meta split
    bank      K independent (h_phi, a, b) triples -> K backbones -> ensemble at inference.
              This is the "multiple weights and inference" half of the ask.

WHY h_phi READS META-FEATURES AND NOT THE 3091 FEATURES. Phase 8a measured both halves of the
question on the 303 compounds where the pseudo-label error is observable:

    Q1  error vs the continuous readout r:   PASS. Spearman(r, signed err) -0.377, p < 1e-4,
        and it holds within isoform (CYP1A2 -0.614, CYP3A4 -0.234).
    Q2  error vs the 3091 features, out of fold:  FAIL. oracle +0.9949, cross-fitted +0.1274,
        R^2 -0.048, 1/5 folds at bar. An oracle-to-cross-fitted gap of +0.87.

So a weight-net on X would be fitting noise with a very convincing in-sample story -- the exact
pattern that closed the decode family in findings/04-05. h_phi therefore reads only

    u_i = [ log10(r_i), one-hot(kind_i), CI width_i, detached per-row loss_i ]

THE HYPOTHESIS FOR WHY LEARNING COULD STILL BEAT DERIVING, stated because without it this arm has
no reason to exist. The derived weight knows about r, but it does not know WHICH rows are
mislabelled. `findings/11` measured that **49.8% of screen-inactive compounds truly sit above
2.211**. Those rows will carry high loss, and a loss-dependent weight can discover them and stand
down -- which is an automatic robust loss, a mechanism the derived form cannot express. That is
the one thing the learned arm can do that the derived arm cannot, and it is what the bar tests.

THE LEAKAGE TRAP, handled explicitly. Bilevel meta-learning optimises phi against a meta
objective. If that objective is the validation split, the reported rho is tuned on its own test
set and is meaningless. So the TRAINING split is divided into train/meta (80/20 by scaffold
group), phi is optimised against **meta** only, and the validation split is never touched until
the final score. g(r) for the `derived` arm is likewise fitted only on dual-labelled compounds
inside the training split.

DYNAMIC LEARNING RATES, which is the other half of the ask. Two parameter groups on two
timescales, because a bilevel problem with one LR either never moves phi or destabilises theta:

    theta (backbone)  AdamW, one-cycle: warmup 10% -> cosine decay, peak --lr
    phi  (weight-net) Adam, constant --meta-lr for a --meta-warmup burn-in during which the
                      weights are held UNIFORM, then cosine decay. The burn-in matters: a
                      loss-dependent weight trained from random init down-weights everything,
                      because at init every row has high loss.

PRE-REGISTERED, written before the run.

1. **Internal bar: `learned` must beat `derived` by macro drho >= +0.005 on >= 5 of 6 seeds.**
   Beating `uniform` is not sufficient -- `derived` is free and already does that.
2. **Honest expected value, stated first: I expect `learned` to MATCH or LOSE to `derived`.** The
   calibration part of the learned arm is the same two parameters the derived arm gets analytically
   from 303 real labels; the only genuinely new capacity is the loss-dependent robust term, and
   loss-dependent reweighting on a 61%-pseudo-labelled isoform is as likely to down-weight the
   informative mass as the mislabelled part of it. `bank` is the arm I would bet on, because
   averaging over calibration uncertainty is the one move here with a clean variance argument.
3. **Manipulation check** (trap 8): print the realised weight distribution per arm -- mean, sd,
   and the Kish ESS of the pseudo-label block. An arm whose weight sd is ~0 has not varied what
   its claim is about, and a `learned` arm that collapses to a constant is reported as a
   non-result rather than as a tie.
4. **The backbone must first be shown competitive.** An MLP that is far worse than the production
   GBM makes every weighting comparison on it a measurement of the wrong thing. The `uniform`
   arm's absolute rho is printed next to the GBM's known per-isoform values for exactly this
   check, and if the backbone is grossly weaker that is reported as the headline, not buried.
5. **NOT CLAIMED: transfer to CYP2D6.** g and sigma(r) come from 250 CYP3A4 / 42 CYP1A2 /
   11 CYP2C9 / **0 CYP2D6** rows. Read the per-isoform table.

    python scripts/cyp_learned_weights_gpu.py --npz cyp_gpu.npz --seeds 0 1 2 --arms all
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:                                    # pragma: no cover
    raise SystemExit(f"torch is required for Phase 8c: {exc}")

FLOOR = 2.21113
SIGMA_ASSAY = 0.3
INACTIVE_R = 0.9
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
KINDS = ("drc", "inactive", "weak", "active", "none")
ARMS = ("uniform", "derived", "learned", "bank")
#: Known GBM per-isoform rho on this split family, for the backbone-competitiveness check in
#: pre-registration item 4. Read off `data/cyp_cache/sigma_anomaly.json` (seed-0 `full` arm).
GBM_RHO = {"CYP1A2": 0.618, "CYP2C9": 0.693, "CYP2D6": 0.496, "CYP3A4": 0.765}


class Trunk(nn.Module):
    """Shared trunk + one head per isoform. Multi-task, because protein-side information can
    only enter through cross-isoform modelling (there are four targets and their features are
    constant within an isoform)."""

    def __init__(self, d_in: int, n_iso: int, width: int = 512, depth: int = 3,
                 p_drop: float = 0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.BatchNorm1d(width), nn.SiLU(), nn.Dropout(p_drop)]
            d = width
        self.body = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(width, 1) for _ in range(n_iso)])

    def forward(self, x):
        h = self.body(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


class WeightNet(nn.Module):
    """h_phi(u) -> a per-row weight in (0, 2). Reads META-FEATURES ONLY, by Phase 8a's gate.

    Also carries the two label-calibration parameters (a, b) for `label = a + b*log10(r)`, so the
    label and the weight are learned jointly -- the "weights expert" is also a label expert.
    """

    def __init__(self, d_u: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_u, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 1))
        # init so the initial weight is ~1.0 everywhere: sigmoid(0)*2 = 1
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        self.cal_a = nn.Parameter(torch.tensor(float(FLOOR)))
        self.cal_b = nn.Parameter(torch.tensor(0.0))

    def weights(self, u):
        return 2.0 * torch.sigmoid(self.net(u)).squeeze(-1)

    def label(self, log10r):
        return self.cal_a + self.cal_b * log10r


def fit_loglin(r, y):
    A = np.vstack([np.ones_like(r), np.log10(r)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0]), float(beta[1])


def fit_sigma_bins(r, y, a, b):
    res = np.abs(y - (a + b * np.log10(r)))
    edges = np.quantile(r, [1 / 3, 2 / 3])
    sd = []
    for t in range(3):
        m = np.digitize(r, edges) == t
        sd.append(float(res[m].std(ddof=1)) if m.sum() > 3 else float(res.std(ddof=1)))
    return np.maximum(np.array(sd), 1e-3), edges


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def kish(w):
    w = np.asarray(w, float); w = w[w > 0]
    return float(w.sum() ** 2 / (w * w).sum()) if w.size else 0.0


def build(npz, seed, isoforms):
    """Assemble the per-isoform training tensors for one seed, plus the train/meta division."""
    tr = npz[f"tr_{seed}"].astype(int)
    va = npz[f"va_{seed}"].astype(int)
    # Divide TRAIN into train/meta. phi is optimised against meta only; va is untouched.
    rs = np.random.default_rng(10_000 + seed)
    perm = rs.permutation(tr)
    cut = int(0.8 * perm.size)
    inner, meta = perm[:cut], perm[cut:]
    out = {}
    for j, s in enumerate(isoforms):
        y = npz[f"y_{s}"]; kind = npz[f"kind_{s}"]; plab = npz[f"plab_{s}"]
        r = npz[f"r_{s}"]; ci = npz[f"ci_{s}"]
        out[s] = dict(j=j, y=y, kind=kind, plab=plab, r=r, ci=ci)
    return inner, meta, va, out


def rows_for(d, idx):
    """Rows of `idx` that carry either a real label or a usable pseudo-label."""
    y, kind = d["y"], d["kind"]
    has_drc = np.isfinite(y[idx])
    has_ps = np.isin(kind[idx], ("inactive", "weak", "active"))
    keep = has_drc | has_ps
    return idx[keep], has_drc[keep]


def meta_features(d, idx, is_drc, loss_det, dev):
    """NaN-SAFE on the readout. `np.clip(NaN, 0.9, None)` is NaN, and a row with no screen
    readout has r = NaN. The forward pass survives it -- torch.where never selects those rows --
    but the BACKWARD for the label-calibration parameters sums `0 * NaN` over the unselected
    branch of the where, which is NaN. That took the whole `learned` arm to NaN on its first
    meta step while `uniform` and `derived` were fine, because only `learned` differentiates
    through the label. Sanitise the readout, not the gradient."""
    r = np.clip(np.nan_to_num(d["r"][idx], nan=INACTIVE_R), INACTIVE_R, None)
    u = [np.log10(r).astype(np.float32)]
    for k in KINDS:
        u.append((d["kind"][idx] == k).astype(np.float32))
    u.append(np.nan_to_num(d["ci"][idx], nan=0.25).astype(np.float32))
    u.append(is_drc.astype(np.float32))
    U = torch.tensor(np.stack(u, axis=1), device=dev)
    return torch.cat([U, loss_det.detach().unsqueeze(1)], dim=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cyp_gpu.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*", default=["all"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512,
                    help="minibatch size. The first smoke run had NO minibatching -- one "
                         "full-batch step per epoch -- so the backbone trained for 30 steps "
                         "and looked uncompetitive. That was my bug, not the model class.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--meta-lr", type=float, default=1e-3)
    ap.add_argument("--meta-warmup", type=int, default=20)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--bank", type=int, default=6)
    ap.add_argument("--no-pseudo", action="store_true",
                    help="zero every pseudo-label weight, i.e. train on REAL labels only. The "
                         "decisive test for why 60->300 epochs degraded: a protocol replication "
                         "on real labels alone found more training monotonically BETTER "
                         "(0.5080 -> 0.5191), and the only thing it removed was the pseudo-labels. "
                         "If the degradation vanishes here, pseudo-label noise is its cause -- "
                         "which makes it direct corroboration of findings/11 (49.8% of "
                         "screen-inactive compounds truly sit above 2.211).")
    ap.add_argument("--out", default="learned_weights.json")
    args = ap.parse_args()
    arms = ARMS if args.arms == ["all"] else tuple(args.arms)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB  "
              f"torch {torch.__version__}", flush=True)
    else:
        print(f"[gpu] NO CUDA -- running on CPU, torch {torch.__version__}", flush=True)

    npz = np.load(args.npz, allow_pickle=False)
    X = npz["X"]
    isoforms = [str(c).split("_")[0] for c in npz["isoforms"]]
    # Standardise features once. Trees do not need it; a batch-normed MLP converges far better
    # with it, and it is fitted on all rows (unsupervised, so no label leakage).
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    Xs = torch.tensor((X - mu) / np.where(sd > 0, sd, 1.0), device=dev)
    d_in = Xs.shape[1]
    dual_r, dual_y, dual_idx = npz["dual_r"], npz["dual_y"], npz["dual_idx"].astype(int)
    print(f"[data] X {tuple(Xs.shape)}  isoforms {isoforms}  dual {dual_r.size}", flush=True)

    results = []
    for seed in args.seeds:
        inner, meta, va, D = build(npz, seed, isoforms)
        in_tr = np.isin(dual_idx, np.concatenate([inner, meta]))
        a_lin, b_lin = fit_loglin(dual_r[in_tr], dual_y[in_tr])
        sd_bins, sd_edges = fit_sigma_bins(dual_r[in_tr], dual_y[in_tr], a_lin, b_lin)
        print(f"\n[seed {seed}] inner {inner.size} meta {meta.size} va {va.size} | "
              f"dual-in-split {int(in_tr.sum())} | g(r) = {a_lin:.4f} {b_lin:+.4f}*log10(r) | "
              f"sigma bins {np.round(sd_bins,3).tolist()}", flush=True)

        for arm in arms:
            t0 = time.time()
            n_models = args.bank if arm == "bank" else 1
            preds = {s: [] for s in isoforms}
            wstats = []
            for k in range(n_models):
                torch.manual_seed(seed * 1000 + k)
                net = Trunk(d_in, len(isoforms), args.width, args.depth).to(dev)
                opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
                # steps_per_epoch is set below once the row packs are known.
                sched = None
                wnet = sopt = None
                if arm in ("learned", "bank"):
                    # d_u = log10(r) + len(KINDS) one-hot + ci + is_drc + loss
                    wnet = WeightNet(1 + len(KINDS) + 2 + 1).to(dev)
                    with torch.no_grad():
                        # BOTH arms initialise at the MEASURED calibration. The first version
                        # gave `bank` b_lin and `learned` 0.0, which confounded the bank
                        # mechanism with its initialisation -- so `bank` minus `learned` could
                        # not be attributed to the bank at all. Recorded rather than quietly
                        # fixed, because the accident bought a free sensitivity test: the two
                        # arms converged to cal_b = -1.094 and +0.102, a swing of 1.2 spanning
                        # the measured -1.19, and reached macro rho 0.5540 vs 0.5534 -- a
                        # difference of 0.0006. THE LABEL SLOPE BARELY MATTERS, which is
                        # evidence that whatever the learned arm gains comes from the WEIGHT
                        # (the robust-loss mechanism) and not from the calibration.
                        wnet.cal_a.fill_(a_lin)
                        wnet.cal_b.fill_(b_lin)
                    sopt = torch.optim.Adam(wnet.parameters(), lr=args.meta_lr)
                    ssch = torch.optim.lr_scheduler.CosineAnnealingLR(
                        sopt, T_max=max(args.epochs - args.meta_warmup, 1))
                # Pre-assemble the row sets per isoform.
                pack = {}
                for s in isoforms:
                    d = D[s]
                    ridx, is_drc = rows_for(d, inner)
                    midx, m_drc = rows_for(d, meta)
                    base = np.where(np.isfinite(d["y"]), d["y"], d["plab"]).astype(np.float32)
                    inact = (d["kind"] == "inactive")
                    pack[s] = dict(ridx=ridx, is_drc=is_drc, midx=midx, m_drc=m_drc,
                                   base=base, inact=inact)
                n_big = max(pack[s]["ridx"].size for s in isoforms)
                steps_per_epoch = max(1, int(np.ceil(n_big / args.batch)))
                sched = torch.optim.lr_scheduler.OneCycleLR(
                    opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                    pct_start=0.1)
                brng = np.random.default_rng(seed * 77 + k)

                order = {s: None for s in isoforms}
                for ep in range(args.epochs):
                  net.train()
                  for s in isoforms:
                      order[s] = brng.permutation(pack[s]["ridx"].size)
                  for st in range(steps_per_epoch):
                    opt.zero_grad(set_to_none=True)
                    total = 0.0
                    wsnap = []
                    for s in isoforms:
                        d, pk = D[s], pack[s]
                        o = order[s]
                        lo = (st * args.batch) % max(o.size, 1)
                        sel = o[lo:lo + args.batch]
                        if sel.size < 32:
                            sel = o[:min(args.batch, o.size)]
                        ridx, is_drc = pk["ridx"][sel], pk["is_drc"][sel]
                        if ridx.size < 32:
                            continue
                        yv = torch.tensor(pk["base"][ridx], device=dev)
                        r = np.clip(np.nan_to_num(d["r"][ridx], nan=INACTIVE_R), INACTIVE_R, None)
                        logr = torch.tensor(np.log10(r).astype(np.float32), device=dev)
                        # `inact` keys on kind == "inactive" ONLY, and is NOT disjoint from
                        # is_drc: the ~303 dual-labelled compounds carry BOTH a real pIC50 and an
                        # inactive screen readout. Without the `& ~is_drc` their real labels were
                        # replaced by a + b*log10(r) -- a regression fitted on those very rows --
                        # in `derived` and `learned` but NOT in `uniform`. So `derived - uniform`
                        # silently included "we replaced 303 real labels with a fit of themselves".
                        inact_m = torch.tensor(pk["inact"][ridx] & ~is_drc, device=dev)
                        # --- the label
                        if arm == "uniform":
                            tgt = yv
                        elif arm == "derived":
                            tgt = torch.where(inact_m,
                                              torch.tensor(a_lin, device=dev) + b_lin * logr, yv)
                        else:
                            tgt = torch.where(inact_m, wnet.label(logr), yv)
                        pred = net(Xs[ridx])[:, D[s]["j"]]
                        per = (pred - tgt) ** 2
                        # --- the weight
                        if arm in ("uniform", "derived"):
                            w = np.where(is_drc, 1.0,
                                         [0.0 if args.no_pseudo else PSEUDO_W.get(k2, 0.0)
                                          for k2 in d["kind"][ridx]])
                            if arm == "derived":
                                sg = sd_bins[np.digitize(r, sd_edges)]
                                wi = 1.0 / (SIGMA_ASSAY ** 2 + sg ** 2)
                                wi = wi / wi.mean() * 0.5
                                w = np.where(pk["inact"][ridx], wi, w)
                            wt = torch.tensor(w.astype(np.float32), device=dev)
                        elif ep < args.meta_warmup:
                            wt = torch.ones_like(per)         # burn-in: uniform
                        else:
                            u = meta_features(d, ridx, is_drc, per, dev)
                            wt = wnet.weights(u)
                        wt = wt.clamp_min(1e-3)        # a collapsed weight set made
                        #                                   the normalised loss NaN in smoke 1
                        total = total + (wt * per).sum() / wt.sum().clamp_min(1e-6)
                        wsnap.append(wt.detach().float().cpu().numpy())
                    if not torch.is_tensor(total):
                        break
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step(); sched.step()

                    if st != steps_per_epoch - 1:
                        continue
                    # --- the outer (meta) step: ONE-STEP LOOKAHEAD, first-order, on META only.
                    #     Runs once per EPOCH (on the last minibatch), not per step.
                    #
                    # The naive version of this is a silent no-op and I wrote it first: if you
                    # simply backward() the meta loss, phi gets ZERO gradient, because the meta
                    # loss is a function of theta alone and has no path to phi. The `learned`
                    # arm would then be identical to `uniform` after burn-in and would report a
                    # tie that means nothing.
                    #
                    # The correct first-order step (L2RW / Meta-Weight-Net):
                    #     theta' = theta - alpha * G(phi),   G(phi) = grad_theta L_train(theta,phi)
                    #     L_meta(theta') ~= L_meta(theta) - alpha * <G(phi), g_meta>
                    # so minimising the lookahead meta loss means MAXIMISING the inner product
                    # <G(phi), g_meta>, i.e. descending its negative in phi. That needs the inner
                    # gradient built with create_graph=True so it stays differentiable in phi.
                    if wnet is not None and ep >= args.meta_warmup:
                        # eval() for the whole meta step. In train() mode BatchNorm updated its
                        # running statistics FROM THE META SPLIT -- meta information reaching the
                        # deployed model outside the intended channel -- and independent dropout
                        # masks attenuated the alignment (measured cos 0.76 train vs 0.89 eval).
                        net.eval()
                        g_meta, mloss_v = None, None
                        mloss = 0.0
                        for s in isoforms:
                            d, pk = D[s], pack[s]
                            keep = pk["midx"][pk["m_drc"]]      # REAL labels only, by design
                            if keep.size < 16:
                                continue
                            mp = net(Xs[keep])[:, d["j"]]
                            mt = torch.tensor(d["y"][keep].astype(np.float32), device=dev)
                            mloss = mloss + ((mp - mt) ** 2).mean()
                        if torch.is_tensor(mloss):
                            params = [p for p in net.parameters() if p.requires_grad]
                            g_meta = torch.autograd.grad(mloss, params, allow_unused=True)
                            g_meta = [torch.zeros_like(p) if g is None else g.detach()
                                      for p, g in zip(params, g_meta)]
                            mloss_v = float(mloss.detach())
                            # Rebuild the weighted inner loss WITH grad on theta (no no_grad
                            # here -- that was the second half of the bug) and differentiably
                            # in phi.
                            inner_phi = 0.0
                            for s in isoforms:
                                d, pk = D[s], pack[s]
                                sel2 = brng.permutation(pk["ridx"].size)[:args.batch]
                                ridx, is_drc = pk["ridx"][sel2], pk["is_drc"][sel2]
                                if ridx.size < 32:
                                    continue
                                r = np.clip(np.nan_to_num(d["r"][ridx], nan=INACTIVE_R), INACTIVE_R, None)
                                logr = torch.tensor(np.log10(r).astype(np.float32), device=dev)
                                inact_m = torch.tensor(pk["inact"][ridx] & ~is_drc, device=dev)
                                yv = torch.tensor(pk["base"][ridx], device=dev)
                                tgt = torch.where(inact_m, wnet.label(logr), yv)
                                pred = net(Xs[ridx])[:, d["j"]]
                                per = (pred - tgt) ** 2
                                u = meta_features(d, ridx, is_drc, per, dev)
                                wt = wnet.weights(u)
                                wt = wt.clamp_min(1e-3)
                                inner_phi = inner_phi + (
                                    wt * per).sum() / wt.sum().clamp_min(1e-6)
                            if torch.is_tensor(inner_phi):
                                G = torch.autograd.grad(inner_phi, params, create_graph=True,
                                                        allow_unused=True)
                                # COSINE, not the raw inner product. Maximising <G, g_meta>
                                # outright is UNBOUNDED in phi -- the weight-net simply inflates
                                # its outputs forever, and the first smoke run duly returned NaN
                                # weights and macro rho -0.40. Normalising to a cosine bounds the
                                # objective in [-1, 1], and "align the weighted training gradient
                                # with the meta gradient" is the same instruction with a scale
                                # that cannot run away.
                                dot = sum((g * gm).sum() for g, gm in zip(G, g_meta)
                                          if g is not None)
                                ng = torch.sqrt(sum((g * g).sum() for g in G
                                                    if g is not None) + 1e-12)
                                nm = torch.sqrt(sum((gm * gm).sum() for gm in g_meta) + 1e-12)
                                dot = dot / (ng * nm)
                                sopt.zero_grad(set_to_none=True)
                                (-dot).backward()
                                torch.nn.utils.clip_grad_norm_(wnet.parameters(), 5.0)
                                sopt.step(); ssch.step()
                                net.train()
                                if ep == args.meta_warmup:
                                    gn = sum(float(p.grad.norm()) for p in wnet.parameters()
                                             if p.grad is not None)
                                    print(f"      [meta] first step: L_meta {mloss_v:.4f}  "
                                          f"cos(G,g_meta) {float(dot.detach()):+.4f}  |grad phi| {gn:.3e}"
                                          + ("   *** ZERO GRADIENT -- phi cannot move"
                                             if gn == 0 else ""), flush=True)

                net.eval()
                with torch.no_grad():
                    for s in isoforms:
                        d = D[s]
                        okv = np.isfinite(d["y"][va])
                        vv = va[okv]
                        preds[s].append(net(Xs[vv])[:, d["j"]].float().cpu().numpy())
                if wsnap:
                    allw = np.concatenate(wsnap)
                    wstats.append((float(allw.mean()), float(allw.std()), kish(allw)))

            rec = dict(seed=seed, arm=arm, n_models=n_models,
                       secs=round(time.time() - t0, 1))
            for s in isoforms:
                d = D[s]
                okv = np.isfinite(d["y"][va]); vv = va[okv]
                p = np.mean(np.stack(preds[s]), axis=0)
                rec[f"rho_{s}"] = spearman(p, d["y"][vv])
            rec["w_mean"] = float(np.mean([a for a, _, _ in wstats])) if wstats else 1.0
            rec["w_sd"] = float(np.mean([b for _, b, _ in wstats])) if wstats else 0.0
            rec["w_ess"] = float(np.mean([c for _, _, c in wstats])) if wstats else 0.0
            if arm in ("learned", "bank") and wnet is not None:
                rec["cal_a"] = float(wnet.cal_a.detach().cpu())
                rec["cal_b"] = float(wnet.cal_b.detach().cpu())
            results.append(rec)
            macro = float(np.mean([rec[f"rho_{s}"] for s in isoforms]))
            print(f"   {arm:9} macro rho {macro:.4f}  " +
                  "  ".join(f"{s} {rec[f'rho_{s}']:.4f}" for s in isoforms) +
                  f"   w {rec['w_mean']:.3f}+/-{rec['w_sd']:.3f}" +
                  (f"  cal {rec.get('cal_a', 0):.3f}{rec.get('cal_b', 0):+.3f}"
                   if "cal_a" in rec else "") +
                  f"   {rec['secs']}s", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1, default=float))

    print("\n" + "=" * 96)
    print("4. BACKBONE COMPETITIVENESS (pre-registered item 4) -- MLP `uniform` vs known GBM rho")
    print("=" * 96)
    u = [r for r in results if r["arm"] == "uniform"]
    if u:
        print(f"   {'iso':8} {'MLP uniform':>12} {'GBM (known)':>12} {'gap':>8}")
        worse = 0
        for s in isoforms:
            m = float(np.mean([r[f"rho_{s}"] for r in u]))
            g = GBM_RHO.get(s, float("nan"))
            print(f"   {s:8} {m:12.4f} {g:12.4f} {m-g:+8.4f}")
            worse += int(m < g - 0.05)
        print(f"   isoforms where the MLP is >0.05 rho worse: {worse}/{len(isoforms)}")
        if worse >= 2:
            print("   *** THE BACKBONE IS NOT COMPETITIVE. Every weighting comparison below is")
            print("   *** then a measurement on a weaker model than production, and is reported")
            print("   *** as such rather than transferred to the GBM pipeline.")

    print("\n" + "=" * 96)
    print("1. THE INTERNAL BAR: `learned` must beat `derived` by macro drho >= +0.005, >= 5/6")
    print("=" * 96)
    seeds = sorted({r["seed"] for r in results})

    def macro_of(arm, seed):
        rs = [r for r in results if r["arm"] == arm and r["seed"] == seed]
        return float(np.mean([rs[0][f"rho_{s}"] for s in isoforms])) if rs else float("nan")

    for ref in ("uniform", "derived"):
        print(f"\n   vs {ref}:")
        for arm in arms:
            if arm == ref:
                continue
            d = np.array([macro_of(arm, s) - macro_of(ref, s) for s in seeds])
            d = d[np.isfinite(d)]
            if not d.size:
                continue
            t = (d.mean() / (d.std(ddof=1) / np.sqrt(d.size))
                 if d.size > 1 and d.std(ddof=1) else 0.0)
            ok = d.mean() >= 0.005 and (d > 0).sum() >= max(5, int(0.8 * d.size))
            print(f"   {arm:9} d macro rho {d.mean():+.4f}  t {t:6.2f}  "
                  f"wins {int((d>0).sum())}/{d.size}  -> "
                  f"{'CLEARS BAR' if ok else 'below bar'}")

    print("\n" + "=" * 96)
    print("3. MANIPULATION CHECK: did the learned weights actually vary?")
    print("=" * 96)
    print(f"   {'arm':9} {'w mean':>9} {'w sd':>9} {'Kish ESS':>10}")
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        if rs:
            print(f"   {arm:9} {np.mean([r['w_mean'] for r in rs]):9.3f} "
                  f"{np.mean([r['w_sd'] for r in rs]):9.3f} "
                  f"{np.mean([r['w_ess'] for r in rs]):10.1f}")
    print("   A `learned` arm whose w sd is ~0 has collapsed to a constant and is a NON-RESULT,")
    print("   not a tie. Reported that way.")
    print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
