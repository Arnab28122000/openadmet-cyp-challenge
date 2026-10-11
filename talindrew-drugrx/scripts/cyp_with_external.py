"""Do 49,965 external PubChem CYP labels help the challenge model?

The last lever that is *data* rather than features. Everything else tested on
this task has landed inside noise: three feature arms (gaps 0.002–0.031 against
seed sd 0.035–0.080), protonation descriptors (+0.0071 against a predicted 0.02),
auxiliary heads from the challenge's own files (0.4346 vs 0.4359), and model
class (xgb 0.4432 vs net 0.4346). Data is the one thing that has ever moved this
programme.

TDC mirrors Veith's PubChem qHTS panels for all four CYPs — the same enzymes, far
more molecules. They are **binary** inhibitor calls against the challenge's
pIC50, so they cannot be merged into the target and enter only as auxiliary
**classification** heads, which the blinded test format permits (heads are
predicted in training and not required at inference).

Audited before use: 1 collision with the blinded test set on each of
CYP1A2/2C9/3A4, zero on CYP2D6, all dropped.

**Caveat recorded by a peer session:** the scikit-fingerprints mirrors have moved
since this repo's baselines were committed (`ppbr` 1,614 rows against 1,797
recorded, a 10% loss; `bbb` 2,030 against 1,975). The Veith panel counts here are
as of 2026-10-05 and are not comparable to any pre-2026-10-04 figure.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.func import functional_call, grad_and_value, stack_module_state, vmap

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from cyp_train import CypNet
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import macro_st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--out", default="/content/out/cyp/with_external.json")
    ap.add_argument("--members", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    ext = pd.read_parquet(args.external)
    ext_cols = [c for c in ext.columns if c.startswith("veith_")]
    print(f"[ext] {len(ext)} external molecules, {len(ext_cols)} panels, "
          f"{int(ext[ext_cols].notna().sum().sum())} labels", flush=True)

    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    xe = featurise(ext["smiles"].tolist(), workers=args.workers,
                   counts=True, desc_mode="rdkit2d", extra_fps=True)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    n_prim = len(PRIMARY)

    results = {"with": [], "without": []}
    for seed in range(args.seeds):
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=600 + seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        _, st = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=st)
        xes, _ = standardise_descriptors(xe, stats=st)

        def run_arm(arm: str) -> float:
            n_heads = n_prim + (len(ext_cols) if arm == "with" else 0)
            # Challenge rows: primary targets, external heads unlabelled.
            yb = np.zeros((len(tr), n_heads), np.float32)
            mb = np.zeros((len(tr), n_heads), bool)
            lob = lo[tr]; hib = hi[tr]; lmb = np.isfinite(lob) & np.isfinite(hib)
            xb = xs[tr]

            if arm == "with":
                ye = np.zeros((len(ext), n_heads), np.float32)
                me = np.zeros((len(ext), n_heads), bool)
                ev = ext[ext_cols].to_numpy(np.float32)
                ye[:, n_prim:] = np.nan_to_num(ev, nan=0.0)
                me[:, n_prim:] = ~np.isnan(ev)
                xb = np.concatenate([xb, xes])
                yb = np.concatenate([yb, ye]); mb = np.concatenate([mb, me])
                pad = np.zeros((len(ext), n_prim), np.float32)
                lob = np.concatenate([lob, pad]); hib = np.concatenate([hib, pad])
                lmb = np.concatenate([lmb, np.zeros((len(ext), n_prim), bool)])

            dev = args.device
            X = torch.from_numpy(np.ascontiguousarray(xb, np.float32)).to(dev)
            Y = torch.from_numpy(np.ascontiguousarray(yb)).to(dev)
            M = torch.from_numpy(np.ascontiguousarray(mb)).to(dev)
            LO = torch.from_numpy(np.ascontiguousarray(lob, np.float32)).to(dev)
            HI = torch.from_numpy(np.ascontiguousarray(hib, np.float32)).to(dev)
            LM = torch.from_numpy(np.ascontiguousarray(lmb)).to(dev)
            XV = torch.from_numpy(np.ascontiguousarray(xs, np.float32)).to(dev)

            members = []
            for k in range(args.members):
                torch.manual_seed(seed * 101 + k * 7919)
                members.append(CypNet(xs.shape[1], n_heads, 1024, 3, 0.15).to(dev))
            params, buffers = stack_module_state(members)
            base = members[0].to("meta")

            def loss_fn(p, b, xx, yy, mm, ll, hh, lm):
                out = functional_call(base, (p, b), (), {"x": xx})
                d = torch.clamp(ll - out[:, :n_prim], min=0) + torch.clamp(out[:, :n_prim] - hh, min=0)
                q = torch.minimum(d, torch.ones_like(d))
                prim = ((0.5 * q.pow(2) + (d - q)) * lm).sum() / lm.sum().clamp(min=1)
                aux = (((out[:, n_prim:] - yy[:, n_prim:]) ** 2) * mm[:, n_prim:]).sum() \
                    / mm[:, n_prim:].sum().clamp(min=1) if n_heads > n_prim else out.sum() * 0
                return prim + 0.3 * aux

            gf = vmap(grad_and_value(loss_fn),
                      in_dims=(0, 0, None, None, None, None, None, None), randomness="different")
            opt = torch.optim.AdamW(list(params.values()), lr=3e-4, weight_decay=0.01, fused=True)

            def predict(xx):
                base.train(False)
                f = vmap(lambda p, b, z: functional_call(base, (p, b), (), {"x": z}),
                         in_dims=(0, 0, None), randomness="different")
                with torch.no_grad():
                    return torch.cat([f(params, buffers, xx[s:s+8192])
                                      for s in range(0, xx.shape[0], 8192)], dim=1)

            best, best_p, since = float("inf"), None, 0
            t0 = time.time()
            for ep in range(args.epochs):
                perm = torch.randperm(X.shape[0], device=dev)
                for s in range(0, X.shape[0], 4096):
                    i = perm[s:s+4096]
                    g, _l = gf(params, buffers, X[i], Y[i], M[i], LO[i], HI[i], LM[i])
                    opt.zero_grad(set_to_none=True)
                    for n_, gg in g.items():
                        params[n_].grad = gg
                    opt.step()
                if ep % 5 == 0:
                    m = predict(XV[va]).mean(0).float().cpu().numpy()
                    sc, _ = macro_st_rae({c: m[:, k] for k, c in enumerate(PRIMARY)},
                                         {c: (frame[c].to_numpy()[va], lo[va, k], hi[va, k])
                                          for k, c in enumerate(PRIMARY)})
                    if sc < best:
                        best, since = sc, 0
                    else:
                        since += 5
                    if since >= args.patience:
                        break
            return best


        for arm in ("without", "with"):
            t_arm = time.time()
            score = run_arm(arm)
            results[arm].append(score)
            print(f"  seed {seed} {arm:8s} MA-ST-RAE {score:.4f}  "
                  f"({(time.time()-t_arm)/60:.1f} min, peak "
                  f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB)", flush=True)
            # run_arm's locals are gone now that it has returned; this reclaims
            # the allocator's cache so the next arm starts from a clean card.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    a, b = np.array(results["without"]), np.array(results["with"])
    print(f"\nwithout external: {a.mean():.4f} ± {a.std():.4f}")
    print(f"with external:    {b.mean():.4f} ± {b.std():.4f}")
    print(f"gain {a.mean()-b.mean():+.4f} ({int((b<a).sum())}/{len(a)} seeds better)")
    Path(args.out).write_text(json.dumps(
        {"without": a.tolist(), "with": b.tolist(),
         "gain": float(a.mean()-b.mean()), "seeds_better": int((b<a).sum())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
