"""Train ONE net config in its own process and save its val/test predictions.

Process isolation is not a style choice here. Three separate attempts to free
GPU memory in-process failed — `del`, scoping each arm in a function, and
`empty_cache()` — because the training closures (`loss_fn`, the vmapped gradient
function, the prediction helper) capture the tensors and keep them alive. Peaks
climbed 18.8 -> 31.9 -> 26.2 -> 39.4 -> 33.7 GB until a 40 GB card ran out. When
the process exits the OS reclaims the device regardless, and one OOM then costs
one data point rather than the run.

Saves predictions rather than weights, because the assembler only needs the
numbers to blend.
"""
from __future__ import annotations

import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyp_train import CypNet
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.resume import EnsembleCheckpoint
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.cyp.metric import METRIC_ID, macro_st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="base")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--patience", type=int, default=150,
                    help="epochs without improvement before stopping; <=0 never stops")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--schedule", choices=["cosine", "sgdr", "plateau", "const"],
                    default="cosine", help="LR schedule; an arm, not a constant")
    ap.add_argument("--lr-patience", type=int, default=10,
                    help="evaluations without improvement before --schedule plateau halves the LR")
    ap.add_argument("--aux-weight", type=float, default=0.3)
    # The metric is a pure L1 hinge. This loss Huber-smooths it, and measurement
    # says 83-94% of molecules miss the interval by less than 1.0 -- so at the
    # default the model is trained almost entirely under 0.5*d^2 while being
    # scored on d. A squared hinge is mean-seeking; the L1 hinge is minimised at
    # the point where P(lo>p) == P(hi<p). 0 means pure hinge, matching the metric
    # exactly. Default kept at 1.0 so this flag changes nothing unless asked.
    ap.add_argument("--huber-delta", type=float, default=1.0)
    # The programme's founding design decision was that the loss must be
    # interval-aware ("the only part of our stack that needed a new objective").
    # That decision has never been ablated, and it has a cost nobody measured: a
    # hinge that is exactly zero inside the credible interval supplies **no
    # gradient** for any compound already inside it, and the median interval is
    # 0.27-0.53 wide. "point" trains Huber on the point estimate instead, which
    # keeps every compound informative; the ST-RAE offset is applied afterwards.
    ap.add_argument("--loss", choices=["interval", "point"], default="interval")
    # Screen-derived interval labels (drugrx/cyp/screen.py) for compounds the DRC
    # stage never measured: every DRC label is a *promoted* one, and on CYP2D6 the
    # model had never seen a single non-inhibitor. "none" reproduces the incumbent.
    ap.add_argument("--screen", choices=["none", "inactive", "all"], default="none")
    ap.add_argument("--screen-weight", type=float, default=1.0,
                    help="loss weight of a no-inhibition-at-50uM pseudo-row")
    ap.add_argument("--screen-weight-other", type=float, default=0.5,
                    help="loss weight of a weak/active pseudo-row")
    # V1 is DRC-only validation (every row promoted, so blind to over-prediction on
    # unselected compounds); V2 adds held-out screen inactives with the floor window.
    ap.add_argument("--select-on", choices=["v1", "v2"], default="v1")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--aux-prefix", nargs="*", default=["veith_"])
    ap.add_argument("--out-dir", default="/content/out/cyp_net")
    ap.add_argument("--workers", type=int, default=12)
    # Featurisation is deterministic and is the CPU bottleneck; pass a directory to
    # reuse content-keyed .npz caches (a changed SMILES list is a cache miss, not reuse).
    ap.add_argument("--feat-cache", default=None)
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    # Measured 2026-10-05: one checkpoint of a 16-member d1024/L6 ensemble is
    # **2.1 GB**, not the ~71 MB the older comments assume -- params, plus Adam's two
    # moments, plus a full duplicate of `best_state`. At `--ckpt-every 10` a 170-epoch
    # run therefore writes ~35 GB to protect a run that costs ~3 minutes to repeat.
    # 0 disables checkpointing, which is the right setting for short runs; resume is
    # for the multi-hour kind it was built for.
    ap.add_argument("--ckpt-every", type=int, default=10,
                    help="epochs between checkpoints; 0 disables (short runs)")
    ap.add_argument("--push-every", type=int, default=50)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame, test = data.frame, data.test
    n_prim = len(PRIMARY)
    x = featurise(frame["smiles"].tolist(), workers=args.workers, cache=args.feat_cache,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    xt = featurise(test["smiles"].tolist(), workers=args.workers, cache=args.feat_cache,
                   counts=True, desc_mode="rdkit2d", extra_fps=True)
    # "--aux-prefix none" means train with NO auxiliary heads at all. Without
    # this the ablation has no control arm: every comparison would be between two
    # auxiliary configurations, and H7a's question (do the aux heads help?) could
    # not be re-asked under the corrected metric.
    no_aux = list(args.aux_prefix) == ["none"]
    ext = pd.read_parquet(args.external)
    # Prefixes rather than a hard-coded "veith_": the ChEMBL acquisition adds a
    # second continuous source and both are auxiliary heads, so the trainer should
    # not need editing per source. --aux-prefix selects which sources are active,
    # which is what makes a with/without ablation possible at all.
    ext_cols = [] if no_aux else [c for c in ext.columns
                                  if any(c.startswith(p) for p in args.aux_prefix)]
    if no_aux:
        ext = ext.iloc[:0]            # no external rows enter the training matrix
    if not ext_cols and not no_aux:
        raise SystemExit(f"no auxiliary columns matching {args.aux_prefix} in "
                         f"{args.external}; columns are {list(ext.columns)[:12]}")
    print(f"[net] auxiliary heads: {len(ext_cols)} "
          f"{'(NONE -- control arm)' if no_aux else f'matching {args.aux_prefix}'}",
          flush=True)
    if no_aux:
        # featurise([]) is not a supported input; build the empty block with the
        # right width so every downstream concatenate keeps working unchanged.
        xe = np.zeros((0, x.shape[1]), np.float32)
    else:
        xe = featurise(ext["smiles"].tolist(), workers=args.workers, cache=args.feat_cache,
                       counts=True, desc_mode="rdkit2d", extra_fps=True)
    n_heads = n_prim + len(ext_cols)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=0)   # fixed split: configs are compared
    va = np.concatenate([va, rest]) if len(rest) else va      # on the same molecules

    # Per-label loss weights: 1 for a DRC label, the screen weights for pseudo-rows,
    # 0 for no label. Bounds are zero-filled where the weight is 0 so the loss value
    # is finite (the gradient through those entries was already exactly zero).
    labs = screen_labels(frame)
    floor = allr_floor(frame)
    kinds = {"none": (), "inactive": ("inactive",),
             "all": ("inactive", "weak", "active")}[args.screen]
    lw = (np.isfinite(lo) & np.isfinite(hi)).astype(np.float32)
    lo_t, hi_t = lo.copy(), hi.copy()
    for k, c in enumerate(PRIMARY):
        kd = labs[c]["kind"].to_numpy()
        sel = np.isin(kd, kinds) & (lw[:, k] == 0)
        lo_t[sel, k] = labs[c]["lo"].to_numpy(float)[sel]
        hi_t[sel, k] = labs[c]["hi"].to_numpy(float)[sel]
        lw[sel, k] = np.where(kd[sel] == "inactive", args.screen_weight,
                              args.screen_weight_other)
        print(f"[net] {c}: train labels {int((lw[tr, k] > 0).sum())} "
              f"(of which screen-derived {int(sel[tr].sum())})", flush=True)
    lo_t, hi_t = np.nan_to_num(lo_t, nan=0.0), np.nan_to_num(hi_t, nan=0.0)
    _, st = standardise_descriptors(x[tr])
    xs, _ = standardise_descriptors(x, stats=st)
    xts, _ = standardise_descriptors(xt, stats=st)
    xes, _ = standardise_descriptors(xe, stats=st)

    yb = np.zeros((len(tr), n_heads), np.float32); mb = np.zeros((len(tr), n_heads), bool)
    # Primary point estimates, needed by --loss point. Previously the primary
    # columns of Y/M stayed zero because the interval loss only ever read lo/hi,
    # so a point loss would have silently regressed everything toward 0.0.
    prim_y = frame[list(PRIMARY)].to_numpy(np.float32)[tr]
    yb[:, :n_prim] = np.nan_to_num(prim_y, nan=0.0)
    mb[:, :n_prim] = np.isfinite(prim_y)
    ev = ext[ext_cols].to_numpy(np.float32)
    ye = np.zeros((len(ext), n_heads), np.float32); me = np.zeros((len(ext), n_heads), bool)
    ye[:, n_prim:] = np.nan_to_num(ev, nan=0.0); me[:, n_prim:] = ~np.isnan(ev)
    pad = np.zeros((len(ext), n_prim), np.float32)
    lmb = lw[tr]

    dev = "cuda"
    X = torch.from_numpy(np.ascontiguousarray(np.concatenate([xs[tr], xes]), np.float32)).to(dev)
    Y = torch.from_numpy(np.ascontiguousarray(np.concatenate([yb, ye]))).to(dev)
    M = torch.from_numpy(np.ascontiguousarray(np.concatenate([mb, me]))).to(dev)
    LO = torch.from_numpy(np.ascontiguousarray(np.concatenate([lo_t[tr], pad]), np.float32)).to(dev)
    HI = torch.from_numpy(np.ascontiguousarray(np.concatenate([hi_t[tr], pad]), np.float32)).to(dev)
    LM = torch.from_numpy(np.ascontiguousarray(
        np.concatenate([lmb, np.zeros((len(ext), n_prim), np.float32)]), np.float32)).to(dev)
    XV = torch.from_numpy(np.ascontiguousarray(xs[va], np.float32)).to(dev)
    XT = torch.from_numpy(np.ascontiguousarray(xts, np.float32)).to(dev)

    members = []
    for k in range(args.members):
        torch.manual_seed(args.seed * 101 + k * 7919)
        members.append(CypNet(xs.shape[1], n_heads, args.d_model, args.n_layers,
                              args.dropout).to(dev))
    params, buffers = stack_module_state(members)
    base = members[0].to("meta")

    def loss_fn(p, bf, xx, yy, mm, ll, hh, lm):
        o = functional_call(base, (p, bf), (), {"x": xx})
        if args.loss == "point":
            # Huber to the point estimate; every labelled compound contributes.
            r = o[:, :n_prim] - yy[:, :n_prim]
            a = r.abs()
            q = torch.clamp(a, max=1.0)
            per = 0.5 * q.pow(2) + (a - q)
            pm = mm[:, :n_prim]
            prim = (per * pm).sum() / pm.sum().clamp(min=1)
        else:
            d = torch.clamp(ll - o[:, :n_prim], min=0) + torch.clamp(o[:, :n_prim] - hh, min=0)
            if args.huber_delta <= 0:
                per = d                                # pure L1 hinge == the metric
            else:
                q = torch.clamp(d, max=args.huber_delta)
                per = 0.5 * q.pow(2) / args.huber_delta + (d - q)
            prim = (per * lm).sum() / lm.sum().clamp(min=1)
        if n_heads == n_prim:
            return prim                      # control arm: no auxiliary term at all
        am = mm[:, n_prim:]
        aux = (((o[:, n_prim:] - yy[:, n_prim:]) ** 2) * am).sum() / am.sum().clamp(min=1)
        return prim + args.aux_weight * aux

    gf = vmap(grad_and_value(loss_fn), in_dims=(0, 0, None, None, None, None, None, None),
              randomness="different")
    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.01, fused=True)
    # The schedule is an ARM, not a constant. A cosine with T_max = epochs makes a long
    # run a different experiment from a short one -- the LR at epoch 200 of a 1,200-epoch
    # cosine is nowhere near the LR at epoch 200 of a 30,000-epoch cosine -- so comparing
    # budgets under one cosine confounds "trained longer" with "decayed more gently".
    # `sgdr` and `plateau` tie the decay cadence to progress instead of to the budget,
    # which is the only way the comparison means what it looks like it means.
    if args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    elif args.schedule == "sgdr":
        # Warm restarts on a lengthening period (T_mult=2): the interval between decays
        # itself grows as training goes on, so the cadence early on is not the cadence at
        # epoch 10,000. That is what "dynamic" has to mean once a budget spans two orders
        # of magnitude, and it is also the regime grokking has been reported in.
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=max(10, args.epochs // 64), T_mult=2)
    elif args.schedule == "plateau":
        # Drop the LR only when validation actually stalls -- no budget in the schedule.
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.lr * 1e-3)
    else:
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=0)

    # Full training state, mirrored to the Hub: a Pro+ runtime has died twice
    # mid-experiment, and without this a run killed in progress restarts cold.
    ckpt = EnsembleCheckpoint(
        args.tag, args.seed,
        {"d_model": args.d_model, "n_layers": args.n_layers, "dropout": args.dropout,
         "members": args.members, "seed": args.seed, "in_dim": int(xs.shape[1]),
         "n_heads": n_heads, "aux_weight": args.aux_weight, "lr": args.lr},
        local_dir=args.ckpt_dir, push_every=args.push_every)
    start_ep = 0
    truth = {c: (frame[c].to_numpy()[va], lo[va, k], hi[va, k]) for k, c in enumerate(PRIMARY)}
    # V2 = V1 plus held-out compounds showing no inhibition at 50 uM, scored against the
    # floor window. Rows with neither stay NaN and the scorer drops them.
    truth2 = {}
    for k, c in enumerate(PRIMARY):
        y1 = frame[c].to_numpy(float)[va]
        has = np.isfinite(y1)
        inact = labs[c]["kind"].to_numpy()[va] == "inactive"
        truth2[c] = (np.where(has, y1, np.where(inact, floor["y"], np.nan)),
                     np.where(has, lo[va, k], np.where(inact, floor["lo"], np.nan)),
                     np.where(has, hi[va, k], np.where(inact, floor["hi"], np.nan)))
    curve_v12: list[tuple[int, float, float]] = []

    def pred(z):
        base.train(False)
        f = vmap(lambda p, bf, q: functional_call(base, (p, bf), (), {"x": q}),
                 in_dims=(0, 0, None), randomness="different")
        with torch.no_grad():
            return torch.cat([f(params, buffers, z[s:s+8192])
                              for s in range(0, z.shape[0], 8192)], dim=1).mean(0).float().cpu().numpy()

    best, since, best_state, best_ep = float("inf"), 0, None, -1
    blob = None if args.no_resume else ckpt.load(dev)
    if blob is not None:
        for k, v in blob["params"].items():
            params[k].data.copy_(v.to(dev))
        for k, v in blob["buffers"].items():
            buffers[k].data.copy_(v.to(dev))
        opt.load_state_dict(blob["opt"])
        if blob.get("sched"):
            sched.load_state_dict(blob["sched"])
        best, best_ep, since = blob["best"], blob["best_epoch"], blob["since"]
        start_ep = blob["epoch"] + 1
        if blob.get("best_state"):
            best_state = ({k: v.to(dev) for k, v in blob["best_state"][0].items()},
                          {k: v.to(dev) for k, v in blob["best_state"][1].items()})
    # The whole curve, not just its minimum: the question "is it still improving
    # at epoch 100?" cannot be answered from a best-point summary, and the
    # best point alone hid that capacity moves the optimum without moving the
    # score.
    curve: list[tuple[int, float]] = [] if blob is None else [tuple(c) for c in blob.get("curve", [])]
    t0 = time.time()
    bs = args.batch_size
    for ep in range(start_ep, args.epochs):
        perm = torch.randperm(X.shape[0], device=dev)
        s = 0
        while s < X.shape[0]:
            i = perm[s:s+bs]
            try:
                g, _ = gf(params, buffers, X[i], Y[i], M[i], LO[i], HI[i], LM[i])
            except torch.OutOfMemoryError:
                if bs <= 256:
                    raise
                bs //= 2; torch.cuda.empty_cache()
                print(f"[net] OOM -> batch {bs}", flush=True); continue
            opt.zero_grad(set_to_none=True)
            for n_, gg in g.items():
                params[n_].grad = gg
            opt.step(); s += bs
        if args.schedule != "plateau":
            sched.step()                     # plateau steps on the metric, below
        if ep % 5 == 0:
            m = pred(XV)
            pm = {c: m[:, k] for k, c in enumerate(PRIMARY)}
            s1 = macro_st_rae(pm, truth)[0]
            s2 = macro_st_rae(pm, truth2)[0]
            sc = s1 if args.select_on == "v1" else s2
            curve.append((ep, float(sc)))
            curve_v12.append((ep, float(s1), float(s2)))
            if args.schedule == "plateau":
                sched.step(sc)
            if sc < best - 1e-5:
                best, since, best_ep = sc, 0, ep
                best_state = ({k: v.detach().clone() for k, v in params.items()},
                              {k: v.detach().clone() for k, v in buffers.items()})
            else:
                since += 5
            if ep % 50 == 0:
                print(f"[net] {args.tag} s{args.seed} ep {ep:4d} v1 {s1:.4f} v2 {s2:.4f} "
                      f"best({args.select_on}) {best:.4f}@{best_ep} "
                      f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
            # patience <= 0 disables early stopping. Required to answer "does it come
            # back at epoch 10,000?" at all: with patience 150 every run in this
            # programme ended near epoch 170, so the question was never asked, only
            # assumed. A long run must be allowed to get worse and then recover.
            if args.patience > 0 and since >= args.patience:
                ckpt.save(epoch=ep, params=params, buffers=buffers, opt=opt, sched=sched,
                          best=best, best_epoch=best_ep, since=since, curve=curve,
                          best_state=best_state, force_push=True)
                break
        if args.ckpt_every and ep % args.ckpt_every == 0:
            ckpt.save(epoch=ep, params=params, buffers=buffers, opt=opt, sched=sched,
                      best=best, best_epoch=best_ep, since=since, curve=curve,
                      best_state=best_state)
    if best_state:
        params, buffers = best_state

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    pv, pt = pred(XV)[:, :n_prim], pred(XT)[:, :n_prim]
    np.savez(out / f"{args.tag}_s{args.seed}.npz", val=pv, test=pt, va_index=va)
    pmv = {c: pv[:, k] for k, c in enumerate(PRIMARY)}
    v1_all, v1_per = macro_st_rae(pmv, truth)
    v2_all, v2_per = macro_st_rae(pmv, truth2)
    (out / f"{args.tag}_s{args.seed}.json").write_text(json.dumps(
        {"tag": args.tag, "seed": args.seed, "valid": best, "best_epoch": best_ep,
         "metric_id": METRIC_ID,
         "select_on": args.select_on, "screen": args.screen,
         "screen_weight": args.screen_weight,
         "screen_weight_other": args.screen_weight_other,
         "v1": v1_all, "v2": v2_all, "v1_per": v1_per, "v2_per": v2_per,
         "test_mean": {c: float(pt[:, k].mean()) for k, c in enumerate(PRIMARY)},
         "curve": curve, "curve_v12": curve_v12,
         "d_model": args.d_model, "n_layers": args.n_layers, "dropout": args.dropout,
         "members": args.members, "epochs_run": ep + 1,
         "schedule": args.schedule, "lr": args.lr, "patience": args.patience,
         "epochs_budget": args.epochs,
         # Recorded so an arm comparison can be *checked* rather than assumed
         # matched: an earlier experiment in this programme was invalidated
         # because OOM backoff silently changed the batch size in one arm only.
         "huber_delta": args.huber_delta, "loss": args.loss,
         "aux_prefix": list(args.aux_prefix), "n_aux_heads": len(ext_cols),
         "batch_size_final": bs,
         "batch_size_requested": args.batch_size,
         "minutes": (time.time()-t0)/60,
         "peak_gb": torch.cuda.max_memory_allocated()/1e9}, indent=2))
    print(f"[net] {args.tag} s{args.seed}: valid {best:.4f} @ep{best_ep} "
          f"({(time.time()-t0)/60:.1f} min, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB, "
          f"{ep+1} epochs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
