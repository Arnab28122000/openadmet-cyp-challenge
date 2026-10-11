"""Is "best epoch" a real optimum, or the luckiest draw from a noisy band?

The long runs settle this programme's selection procedure rather than its model.
After epoch ~50 the validation score oscillates in a band of mean 0.485 and sd
0.009 for another 800 epochs. We then report the *minimum* of ~160 such
evaluations -- 0.4441 for long-base s0, 4.4 sd below the band mean -- and the
epoch it happened at moved from 25 to 360 between two seeds whose final scores
were identical (0.4448 vs 0.4441). An optimum that relocates by 335 epochs
without changing the score is not an optimum.

That matters beyond tidiness, because the number we submitted to the challenge
was chosen exactly this way: minimum-over-epochs on the same molecules we then
quoted the score from. If the band is noise, the quoted score is the winner's
curse and the true generalisation is nearer 0.485 than 0.444 -- which would mean
most of our apparent gap to the leaderboard leader was never real.

So: train once on an 80/10/10 scaffold split, select on the middle split, and
score four candidates on a **holdout that selection never saw**:

* **best-on-select** -- what we have been doing and what we submitted.
* **final** -- the last epoch, no selection at all. The control.
* **swa** -- a running weight average over the post-warmup band. If the band is
  noise around a good solution, its centre should beat any single draw.
* **pred-avg** -- the band's *predictions* averaged. Costs no extra memory and,
  unlike SWA, is valid even when the band wanders between basins rather than
  jittering inside one.

The paired bootstrap over holdout molecules decides, so a 0.005 difference is
not read as a result.
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
from drugrx.cyp.metric import macro_st_rae, st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="curse")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--swa-start", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--out-dir", default="/content/out/cyp_curse")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    n_prim = len(PRIMARY)
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    ext = pd.read_parquet(args.external)
    ext_cols = [c for c in ext.columns if c.startswith("veith_")]
    xe = featurise(ext["smiles"].tolist(), workers=args.workers,
                   counts=True, desc_mode="rdkit2d", extra_fps=True)
    n_heads = n_prim + len(ext_cols)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    # 80/10/10. `rest` is the holdout and is touched exactly once, at the end.
    tr, sel, hold = scaffold_split(frame["smiles"].tolist(), frac_train=0.80,
                                   frac_valid=0.10, seed=0)
    if len(hold) == 0:
        raise SystemExit("no holdout split -- refusing to run a test with no holdout")
    print(f"[curse] {len(tr)} train / {len(sel)} select / {len(hold)} holdout", flush=True)

    _, st = standardise_descriptors(x[tr])
    xs, _ = standardise_descriptors(x, stats=st)
    xes, _ = standardise_descriptors(xe, stats=st)

    yb = np.zeros((len(tr), n_heads), np.float32); mb = np.zeros((len(tr), n_heads), bool)
    ev = ext[ext_cols].to_numpy(np.float32)
    ye = np.zeros((len(ext), n_heads), np.float32); me = np.zeros((len(ext), n_heads), bool)
    ye[:, n_prim:] = np.nan_to_num(ev, nan=0.0); me[:, n_prim:] = ~np.isnan(ev)
    pad = np.zeros((len(ext), n_prim), np.float32)
    lmb = np.isfinite(lo[tr]) & np.isfinite(hi[tr])

    dev = "cuda"
    cat = lambda a, b, t: torch.from_numpy(np.ascontiguousarray(np.concatenate([a, b]), t)).to(dev)
    X = cat(xs[tr], xes, np.float32)
    Y = cat(yb, ye, np.float32)
    M = cat(mb, me, bool)
    LO = cat(lo[tr], pad, np.float32)
    HI = cat(hi[tr], pad, np.float32)
    LM = cat(lmb, np.zeros((len(ext), n_prim), bool), bool)
    XS = torch.from_numpy(np.ascontiguousarray(xs[sel], np.float32)).to(dev)
    XH = torch.from_numpy(np.ascontiguousarray(xs[hold], np.float32)).to(dev)

    members = []
    for k in range(args.members):
        torch.manual_seed(args.seed * 101 + k * 7919)
        members.append(CypNet(xs.shape[1], n_heads, args.d_model, args.n_layers,
                              args.dropout).to(dev))
    params, buffers = stack_module_state(members)
    base = members[0].to("meta")

    def loss_fn(p, bf, xx, yy, mm, ll, hh, lm):
        o = functional_call(base, (p, bf), (), {"x": xx})
        d = torch.clamp(ll - o[:, :n_prim], min=0) + torch.clamp(o[:, :n_prim] - hh, min=0)
        q = torch.minimum(d, torch.ones_like(d))
        prim = ((0.5 * q.pow(2) + (d - q)) * lm).sum() / lm.sum().clamp(min=1)
        am = mm[:, n_prim:]
        aux = (((o[:, n_prim:] - yy[:, n_prim:]) ** 2) * am).sum() / am.sum().clamp(min=1)
        return prim + args.aux_weight * aux

    gf = vmap(grad_and_value(loss_fn), in_dims=(0, 0, None, None, None, None, None, None),
              randomness="different")
    opt = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=0.01, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def predict_with(p, bf, z):
        base.train(False)
        f = vmap(lambda pp, bb, q: functional_call(base, (pp, bb), (), {"x": q}),
                 in_dims=(0, 0, None), randomness="different")
        with torch.no_grad():
            return torch.cat([f(p, bf, z[s:s+8192])
                              for s in range(0, z.shape[0], 8192)],
                             dim=1).mean(0).float().cpu().numpy()[:, :n_prim]

    def score(pred, idx):
        truth = {c: (frame[c].to_numpy()[idx], lo[idx, k], hi[idx, k])
                 for k, c in enumerate(PRIMARY)}
        return macro_st_rae({c: pred[:, k] for k, c in enumerate(PRIMARY)}, truth)[0]

    best, best_ep, best_state = float("inf"), -1, None
    swa = None; swa_n = 0
    acc_sel = None; acc_hold = None; acc_n = 0
    curve = []
    t0 = time.time(); bs = args.batch_size

    for ep in range(args.epochs):
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
                print(f"[curse] OOM -> batch {bs}", flush=True); continue
            opt.zero_grad(set_to_none=True)
            for n_, gg in g.items():
                params[n_].grad = gg
            opt.step(); s += bs
        sched.step()

        if ep >= args.swa_start:
            # Running weight average: the band's centre, not one of its draws.
            if swa is None:
                swa = {k: v.detach().clone().float() for k, v in params.items()}; swa_n = 1
            else:
                swa_n += 1
                for k, v in params.items():
                    swa[k] += (v.detach().float() - swa[k]) / swa_n

        if ep % 5 == 0:
            ps = predict_with(params, buffers, XS)
            sc = score(ps, sel)
            curve.append((ep, float(sc)))
            if sc < best - 1e-5:
                best, best_ep = sc, ep
                best_state = ({k: v.detach().clone() for k, v in params.items()},
                              {k: v.detach().clone() for k, v in buffers.items()})
            if ep >= args.swa_start:
                ph = predict_with(params, buffers, XH)
                acc_n += 1
                acc_sel = ps if acc_sel is None else acc_sel + (ps - acc_sel) / acc_n
                acc_hold = ph if acc_hold is None else acc_hold + (ph - acc_hold) / acc_n
            if ep % 50 == 0:
                print(f"[curse] ep {ep:4d} select {sc:.4f} best {best:.4f}@{best_ep}", flush=True)

    # ── the four candidates, scored on a holdout selection never saw ──
    cand = {}
    cand["final"] = predict_with(params, buffers, XH)
    if best_state:
        cand["best-on-select"] = predict_with(best_state[0], best_state[1], XH)
    if swa is not None:
        sw = {k: v.to(next(iter(params.values())).dtype) for k, v in swa.items()}
        cand["swa"] = predict_with(sw, buffers, XH)
    if acc_hold is not None:
        cand["pred-avg"] = acc_hold

    print(f"\n[curse] holdout ({len(hold)} molecules), never used for selection:", flush=True)
    hold_scores = {k: float(score(v, hold)) for k, v in cand.items()}
    sel_best = float(best)
    for k, v in sorted(hold_scores.items(), key=lambda kv: kv[1]):
        print(f"    {k:16s} {v:.4f}", flush=True)
    print(f"\n[curse] selection-set best was {sel_best:.4f}; the same weights score "
          f"{hold_scores.get('best-on-select', float('nan')):.4f} on holdout "
          f"-> winner's curse {hold_scores.get('best-on-select', float('nan')) - sel_best:+.4f}",
          flush=True)

    # ── paired bootstrap over holdout molecules ──────────────────────
    rng = np.random.default_rng(0)
    ref = "best-on-select" if "best-on-select" in cand else "final"
    y_all = {c: frame[c].to_numpy()[hold] for c in PRIMARY}
    deltas = {k: [] for k in cand if k != ref}
    # The resample is drawn per endpoint ONCE per iteration and reused across
    # arms: the comparison is paired, so every arm must be scored on the same
    # molecules. Drawing inside the arm loop would compare arms on different
    # resamples and inflate the interval until nothing could ever be called.
    lo_h, hi_h = lo[hold], hi[hold]
    for _ in range(400):
        resample = []
        for i, c in enumerate(PRIMARY):
            y = y_all[c]; keep = np.isfinite(y)
            n = int(keep.sum())
            if n < 10:
                continue
            resample.append((i, keep, rng.integers(0, n, n)))
        draw = {}
        for k in cand:
            draw[k] = float(np.mean([
                st_rae(cand[k][keep][idx, i], y_all[PRIMARY[i]][keep][idx],
                       lo_h[keep][idx, i], hi_h[keep][idx, i])
                for i, keep, idx in resample]))
        for k in deltas:
            deltas[k].append(draw[ref] - draw[k])

    verdict = {}
    for k, v in deltas.items():
        v = np.array(v)
        q = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        # Three outcomes, not two. The first version printed "inside noise"
        # whenever the arm failed to *beat* the reference, which labelled an
        # interval lying entirely below zero -- a decisive loss -- as if nothing
        # had been measured. An interval that excludes zero is a result whichever
        # side it falls on.
        if q[0] > 0:
            tag, better = "BETTER", True
        elif q[1] < 0:
            tag, better = "WORSE (CI excludes zero)", False
        else:
            tag, better = "inside noise", False
        verdict[k] = {"gain_over_ref": float(v.mean()), "ci": q,
                      "beats_ref": better, "verdict": tag}
        print(f"[curse] {k} vs {ref}: {v.mean():+.4f} [{q[0]:+.4f}, {q[1]:+.4f}] "
              f"-> {tag}", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_s{args.seed}.json").write_text(json.dumps(
        {"tag": args.tag, "seed": args.seed, "select_best": sel_best,
         "select_best_epoch": best_ep, "holdout": hold_scores,
         "winners_curse": (hold_scores["best-on-select"] - sel_best
                           if "best-on-select" in hold_scores else None),
         "vs_best_on_select": verdict, "curve": curve,
         "n_train": len(tr), "n_select": len(sel), "n_holdout": len(hold),
         "swa_start": args.swa_start, "epochs": args.epochs,
         "minutes": (time.time()-t0)/60}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
