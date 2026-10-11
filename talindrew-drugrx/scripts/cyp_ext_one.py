"""Train ONE (seed, arm) of the external-data experiment and write one score.

This exists because the in-process version leaked: GPU memory climbed
18.8 -> 31.9 -> 26.2 -> 39.4 -> 33.7 GB across arms and the third seed OOMed on a
40 GB card. `del` could not free the tensors because the training closures
(`loss_fn`, the vmapped gradient function, `predict`) still captured them, and
scoping each arm in a function did not help either.

**One process per run is the fix that actually works**: when the process exits
the OS reclaims the device memory regardless of what Python was still holding.
It also makes each run independent, so a single OOM costs one data point instead
of the rest of the experiment.
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
from drugrx.cyp.metric import macro_st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=["with", "without"], required=True)
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--out-dir", default="/content/out/cyp/ext_runs")
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    n_prim = len(PRIMARY)
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    ext = pd.read_parquet(args.external)
    ext_cols = [c for c in ext.columns if c.startswith("veith_")]
    use_ext = args.arm == "with"
    n_heads = n_prim + len(ext_cols)      # same head count in BOTH arms, so the
                                          # model shape is not the variable under test
    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=600 + args.seed)
    va = np.concatenate([va, rest]) if len(rest) else va
    _, st = standardise_descriptors(x[tr])
    xs, _ = standardise_descriptors(x, stats=st)

    xb, yb = xs[tr], np.zeros((len(tr), n_heads), np.float32)
    mb = np.zeros((len(tr), n_heads), bool)
    lob, hib = lo[tr], hi[tr]
    lmb = np.isfinite(lob) & np.isfinite(hib)

    if use_ext:
        xe = featurise(ext["smiles"].tolist(), workers=args.workers,
                       counts=True, desc_mode="rdkit2d", extra_fps=True)
        xes, _ = standardise_descriptors(xe, stats=st)
        ev = ext[ext_cols].to_numpy(np.float32)
        ye = np.zeros((len(ext), n_heads), np.float32)
        me = np.zeros((len(ext), n_heads), bool)
        ye[:, n_prim:] = np.nan_to_num(ev, nan=0.0)
        me[:, n_prim:] = ~np.isnan(ev)
        pad = np.zeros((len(ext), n_prim), np.float32)
        xb = np.concatenate([xb, xes]); yb = np.concatenate([yb, ye])
        mb = np.concatenate([mb, me]); lob = np.concatenate([lob, pad])
        hib = np.concatenate([hib, pad])
        lmb = np.concatenate([lmb, np.zeros((len(ext), n_prim), bool)])

    dev = args.device
    X = torch.from_numpy(np.ascontiguousarray(xb, np.float32)).to(dev)
    Y = torch.from_numpy(np.ascontiguousarray(yb)).to(dev)
    M = torch.from_numpy(np.ascontiguousarray(mb)).to(dev)
    LO = torch.from_numpy(np.ascontiguousarray(lob, np.float32)).to(dev)
    HI = torch.from_numpy(np.ascontiguousarray(hib, np.float32)).to(dev)
    LM = torch.from_numpy(np.ascontiguousarray(lmb)).to(dev)
    XV = torch.from_numpy(np.ascontiguousarray(xs[va], np.float32)).to(dev)

    members = []
    for k in range(args.members):
        torch.manual_seed(args.seed * 101 + k * 7919)
        members.append(CypNet(xs.shape[1], n_heads, 1024, 3, 0.15).to(dev))
    params, buffers = stack_module_state(members)
    base = members[0].to("meta")

    def loss_fn(p, b, xx, yy, mm, ll, hh, lm):
        out = functional_call(base, (p, b), (), {"x": xx})
        d = torch.clamp(ll - out[:, :n_prim], min=0) + torch.clamp(out[:, :n_prim] - hh, min=0)
        q = torch.minimum(d, torch.ones_like(d))
        prim = ((0.5 * q.pow(2) + (d - q)) * lm).sum() / lm.sum().clamp(min=1)
        am = mm[:, n_prim:]
        aux = (((out[:, n_prim:] - yy[:, n_prim:]) ** 2) * am).sum() / am.sum().clamp(min=1)
        return prim + args.aux_weight * aux

    gf = vmap(grad_and_value(loss_fn),
              in_dims=(0, 0, None, None, None, None, None, None), randomness="different")
    opt = torch.optim.AdamW(list(params.values()), lr=3e-4, weight_decay=0.01, fused=True)
    truth = {c: (frame[c].to_numpy()[va], lo[va, k], hi[va, k]) for k, c in enumerate(PRIMARY)}

    def score() -> float:
        base.train(False)
        f = vmap(lambda p, b, z: functional_call(base, (p, b), (), {"x": z}),
                 in_dims=(0, 0, None), randomness="different")
        with torch.no_grad():
            m = torch.cat([f(params, buffers, XV[s:s+8192])
                           for s in range(0, XV.shape[0], 8192)], dim=1).mean(0).float().cpu().numpy()
        return macro_st_rae({c: m[:, k] for k, c in enumerate(PRIMARY)}, truth)[0]

    best, since, t0 = float("inf"), 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(X.shape[0], device=dev)
        for s in range(0, X.shape[0], 4096):
            i = perm[s:s+4096]
            g, _ = gf(params, buffers, X[i], Y[i], M[i], LO[i], HI[i], LM[i])
            opt.zero_grad(set_to_none=True)
            for n_, gg in g.items():
                params[n_].grad = gg
            opt.step()
        if ep % 5 == 0:
            sc = score()
            if sc < best:
                best, since = sc, 0
            else:
                since += 5
            if since >= args.patience:
                break

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.arm}_{args.seed}.json").write_text(json.dumps(
        {"arm": args.arm, "seed": args.seed, "macro_st_rae": best,
         "n_train_rows": int(X.shape[0]), "minutes": (time.time()-t0)/60,
         "peak_gb": torch.cuda.max_memory_allocated()/1e9}, indent=2))
    print(f"[one] {args.arm} seed {args.seed}: {best:.4f} "
          f"({(time.time()-t0)/60:.1f} min, peak {torch.cuda.max_memory_allocated()/1e9:.1f} GB, "
          f"{X.shape[0]} rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
