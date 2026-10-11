"""The interval hinge was worth -0.064 of ST-RAE. How much of that is RANKING?

`cyp_hinge_head.py` established that CheMeleon's representation adds nothing and that the
entire gain of the fine-tuned net was its **loss function** -- the competition metric's own
interval hinge, worth about -0.064 macro against the GBM's point loss on identical features.
That number is ST-RAE, and ST-RAE is the wrong yardstick for comparing models here: a tuned
affine placement step sits downstream, and it is monotone, so any ST-RAE gain at fixed Spearman
is one the placement step reaches by itself.

That critique has already overturned one conclusion today. Scored on ST-RAE, the screen fix
looked six times more valuable on CYP2D6 than on CYP1A2; scored on Spearman it is *less*
valuable, because most of CYP2D6's ST-RAE gain was placement all along -- v5 captured it with a
pure spread rescale and no model change. So the hinge's -0.064 deserves the same question.

Three arms, same features, same splits, same held-out rows, reported on **both** metrics:

    lgbm_l2     LightGBM, squared loss on point labels            architecture A, point loss
    mlp_l2      MLP, squared loss on the same point labels        architecture B, point loss
    mlp_hinge   MLP, the metric's interval hinge                  architecture B, interval loss

`mlp_l2` is the control that `cyp_hinge_head.py` lacked: it had no arm pairing the production
head with a *point* loss, so "the hinge did it" and "the net did it" were still confounded.
Here `mlp_hinge - mlp_l2` isolates the loss and `mlp_l2 - lgbm_l2` isolates the architecture.

Reading it:

* hinge beats l2 on **Spearman** -> the interval loss genuinely learns a better ordering, and
  the -0.064 is mostly real.
* hinge beats l2 on ST-RAE but **not** on Spearman -> the hinge is a placement device in
  disguise, the affine step already captures it, and the net's advantage over the GBM is
  smaller than recorded.

    python scripts/cyp_hinge_rho.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
CLAMP = 6.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/hinge_rho.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    import torch
    from scipy.stats import spearmanr
    from torch import nn

    from cyp_chemeleon_ft import train_labels, v2_targets
    from drugrx.features import featurise, standardise_descriptors

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)
    truth = v2_targets(frame, labs, floor)
    LO, HI, W = train_labels(frame, labs, floor)
    xraw = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                     counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[hinge-rho] features {xraw.shape} on {dev}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        _, stt = standardise_descriptors(xraw[tr])
        X, _ = standardise_descriptors(xraw, stats=stt)
        X = np.clip(X, -CLAMP, CLAMP).astype(np.float32)
        Xt = torch.tensor(X, device=dev)
        LOt = torch.tensor(np.nan_to_num(LO, nan=0.0), dtype=torch.float32, device=dev)
        HIt = torch.tensor(np.nan_to_num(HI, nan=0.0), dtype=torch.float32, device=dev)
        Wt = torch.tensor(W, dtype=torch.float32, device=dev)
        # Point targets for the squared-loss arms: the interval's midpoint, which is the
        # published value on a dose-response row and the bin centre on a screen row.
        MIDt = torch.tensor(np.nan_to_num((LO + HI) / 2.0, nan=0.0),
                            dtype=torch.float32, device=dev)
        print(f"\n[hinge-rho] seed {seed}: {len(tr)} train / {len(va)} val", flush=True)

        def score(p: np.ndarray) -> dict:
            """Per-isoform V2 ST-RAE and Spearman on the held-out rows."""
            out = {}
            for k, c in enumerate(PRIMARY):
                y, lo_, hi_ = truth[c]
                m = np.isfinite(y[va])
                if m.sum() <= 20:
                    continue
                yy = frame[c].to_numpy(float)
                real = np.isfinite(yy[va])
                out[c[:6]] = {
                    "v2": float(st_rae(p[m, k], y[va][m], lo_[va][m], hi_[va][m])),
                    "rho": float(spearmanr(p[real, k], yy[va][real]).statistic)}
            return out

        def train_mlp(loss_kind: str) -> np.ndarray:
            torch.manual_seed(seed)
            net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(512, len(PRIMARY))).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            best, best_p = float("inf"), None
            for ep in range(args.epochs):
                net.train()
                perm = np.random.default_rng(ep).permutation(len(tr))
                for s in range(0, len(tr), args.batch_size):
                    idx = tr[perm[s:s + args.batch_size]]
                    o = net(Xt[idx])
                    lo, hi, w = LOt[idx], HIt[idx], Wt[idx]
                    if loss_kind == "hinge":
                        d = torch.clamp(lo - o, min=0) + torch.clamp(o - hi, min=0)
                    else:
                        d = (o - MIDt[idx]) ** 2
                    loss = (d * w).sum() / w.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sch.step()
                if ep % 5 == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        p = net(Xt[va]).cpu().numpy()
                    sc = score(p)
                    v2 = float(np.mean([d["v2"] for d in sc.values()]))
                    if v2 < best - 1e-5:
                        best, best_p = v2, p
            return best_p

        row = {"seed": seed}
        for arm in ("lgbm_l2", "mlp_l2", "mlp_hinge"):
            t0 = time.time()
            if arm == "lgbm_l2":
                p = np.zeros((len(va), len(PRIMARY)))
                for k, c in enumerate(PRIMARY):
                    yy = frame[c].to_numpy(float)
                    kind = labs[c]["kind"].to_numpy()
                    use = np.zeros(len(frame), bool); use[tr] = True
                    drc = use & np.isfinite(yy)
                    ps = use & np.isin(kind, ("inactive", "weak", "active"))
                    yv = np.where(np.isfinite(yy), yy, labs[c]["y"].to_numpy(float))
                    w = np.where(drc, 1.0, [PSEUDO_W.get(z, 0.0) for z in kind])
                    idx = np.flatnonzero(drc | ps)
                    m = lgb.LGBMRegressor(
                        n_estimators=800, learning_rate=0.03, num_leaves=31,
                        min_child_samples=20, subsample=0.8, subsample_freq=1,
                        colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                        n_jobs=10, verbose=-1)
                    m.fit(xraw[idx], yv[idx], sample_weight=w[idx])
                    p[:, k] = m.predict(xraw[va])
            else:
                p = train_mlp("hinge" if arm == "mlp_hinge" else "l2")
            sc = score(p)
            row[arm] = sc
            print(f"   {arm:10} " + "  ".join(
                f"{i} rho {d['rho']:.4f} V2 {d['v2']:.4f}" for i, d in sc.items())
                + f"   [{time.time()-t0:.0f}s]", flush=True)
        rows.append(row)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    isos = [c[:6] for c in PRIMARY]
    print("\n=== paired contrasts over seeds ===")
    for a, b, what in (("mlp_hinge", "mlp_l2", "THE LOSS (hinge - squared, same net)"),
                       ("mlp_l2", "lgbm_l2", "the architecture (mlp - lgbm, same loss)"),
                       ("mlp_hinge", "lgbm_l2", "both together, vs the incumbent GBM")):
        print(f"\n--- {what}")
        print(f"{'iso':8} {'d Spearman':>22} {'t':>6} | {'d V2 ST-RAE':>22} {'t':>6}")
        macro = {"rho": [], "v2": []}
        for iso in isos:
            line = f"{iso:8}"
            for key in ("rho", "v2"):
                d = np.array([r[a][iso][key] - r[b][iso][key] for r in rows if iso in r[a]])
                if len(d) < 2:
                    line += f"{'n/a':>31}"; continue
                se = d.std(ddof=1) / np.sqrt(len(d))
                macro[key].append(d.mean())
                line += f" {d.mean():+13.4f} +-{se:.4f} {d.mean()/se if se else 0:6.2f} |"
            print(line.rstrip("|"))
        print(f"{'MACRO':8} {np.mean(macro['rho']):+13.4f}{'':16}| "
              f"{np.mean(macro['v2']):+13.4f}")
    print("\nIf the hinge's advantage is in V2 but not in Spearman, it is a placement device")
    print("and the affine step already reaches it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
