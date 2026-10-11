"""Do the release's own auxiliary heads buy ranking? 47,360 labels say they might.

Three things are now measured and they corner this question. A plain LightGBM reproduces the
board's Spearman on CYP2D6 and CYP3A4 but falls **0.204 and 0.128 short** on CYP1A2 and
CYP2C9, so the deployed ensemble is doing something large on those two. It is not the interval
hinge -- on matched labels that costs ρ (`cyp_hinge_rho.py`). And it is not CheMeleon, whose
representation scored identically to ECFP under the same head.

The remaining candidate the production net has and a single-task GBM does not is **multitask
supervision from the release's own auxiliary columns** -- 20 heads carrying 47,360 labels
against the primary task's 6,525:

    CYP*_pIC50_TDI_condition            4 heads,  7,778 labels, r = 0.90-0.99 to the target
    CYP*_EmaxVsPosCtrl_TDI_condition    4 heads,  7,778
    CYP*_EmaxVsPosCtrl_direct_inhibition 4 heads, 7,525, the SAME dose-response fit
    CYP*_is_TDI                         4 heads,  7,779
    log2fc_CYP*                         4 heads, 17,500, the 50 uM screen readout

`cyp_aux_ablation.py` ablated the *external* acquisitions (ChEMBL, Veith) and found a macro
null; it never ablated the release's own heads, and it reported one scalar per run with no
per-endpoint breakdown, so even its macro null cannot be read per isoform now.

Two arms, identical in everything else, reported on Spearman **and** V2 ST-RAE per isoform:

    noaux   4 primary outputs
    aux     4 primary + 20 auxiliary outputs, each masked where its label is missing

Auxiliary heads are standardised per column on the training fold and contribute a masked
squared loss at weight AUX_W; the primary heads are scored exactly as before, so a change can
only come from the shared trunk.

    python scripts/cyp_aux_heads_rho.py --seeds 0 1 2 3 4
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

from drugrx.cyp.data import AUXILIARY, PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
CLAMP = 6.0
AUX_W = 0.3          # auxiliary heads are a regulariser, not the task


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--aux-w", type=float, default=AUX_W)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/aux_heads_rho.json"))
    args = ap.parse_args()

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
    MID = (LO + HI) / 2.0

    aux_cols = [c for c in AUXILIARY if c in frame.columns]
    A = frame[aux_cols].to_numpy(float)
    AM = np.isfinite(A).astype(np.float32)
    print(f"[aux] {len(aux_cols)} auxiliary heads, {int(AM.sum())} labels "
          f"against {int(np.isfinite(frame[PRIMARY].to_numpy(float)).sum())} primary",
          flush=True)

    xraw = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                     counts=True, desc_mode="rdkit2d", extra_fps=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        _, stt = standardise_descriptors(xraw[tr])
        X, _ = standardise_descriptors(xraw, stats=stt)
        X = np.clip(X, -CLAMP, CLAMP).astype(np.float32)

        # Auxiliary targets standardised per column on the training fold only.
        At = A.copy()
        for j in range(At.shape[1]):
            col = At[tr, j]
            col = col[np.isfinite(col)]
            mu, sd = (col.mean(), col.std()) if len(col) > 10 else (0.0, 1.0)
            At[:, j] = (At[:, j] - mu) / (sd if sd > 1e-8 else 1.0)
        At = np.nan_to_num(At, nan=0.0)

        Xt = torch.tensor(X, device=dev)
        MIDt = torch.tensor(np.nan_to_num(MID, nan=0.0), dtype=torch.float32, device=dev)
        Wt = torch.tensor(W, dtype=torch.float32, device=dev)
        AT = torch.tensor(At, dtype=torch.float32, device=dev)
        AMt = torch.tensor(AM, dtype=torch.float32, device=dev)
        print(f"\n[aux] seed {seed}: {len(tr)} train / {len(va)} val", flush=True)

        def score(p: np.ndarray) -> dict:
            out = {}
            for k, c in enumerate(PRIMARY):
                y, lo_, hi_ = truth[c]
                m = np.isfinite(y[va])
                yy = frame[c].to_numpy(float)
                real = np.isfinite(yy[va])
                if m.sum() <= 20 or real.sum() <= 20:
                    continue
                out[c[:6]] = {
                    "v2": float(st_rae(p[m, k], y[va][m], lo_[va][m], hi_[va][m])),
                    "rho": float(spearmanr(p[real, k], yy[va][real]).statistic)}
            return out

        def train(use_aux: bool) -> np.ndarray:
            torch.manual_seed(seed)
            n_out = len(PRIMARY) + (len(aux_cols) if use_aux else 0)
            net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(),
                                nn.Dropout(0.1), nn.Linear(512, n_out)).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            best, best_p = float("inf"), None
            np_ = len(PRIMARY)
            for ep in range(args.epochs):
                net.train()
                perm = np.random.default_rng(ep).permutation(len(tr))
                for s in range(0, len(tr), args.batch_size):
                    idx = tr[perm[s:s + args.batch_size]]
                    o = net(Xt[idx])
                    w = Wt[idx]
                    d = (o[:, :np_] - MIDt[idx]) ** 2
                    loss = (d * w).sum() / w.sum().clamp(min=1)
                    if use_aux:
                        am = AMt[idx]
                        da = (o[:, np_:] - AT[idx]) ** 2
                        loss = loss + args.aux_w * (da * am).sum() / am.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                sch.step()
                if ep % 5 == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        p = net(Xt[va]).cpu().numpy()[:, :np_]
                    sc = score(p)
                    v2 = float(np.mean([d["v2"] for d in sc.values()]))
                    if v2 < best - 1e-5:
                        best, best_p = v2, p
            return best_p

        row = {"seed": seed}
        for arm, ua in (("noaux", False), ("aux", True)):
            t0 = time.time()
            row[arm] = score(train(ua))
            print(f"   {arm:6} " + "  ".join(
                f"{i} rho {d['rho']:.4f} V2 {d['v2']:.4f}" for i, d in row[arm].items())
                + f"   [{time.time()-t0:.0f}s]", flush=True)
        rows.append(row)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print(f"\n=== aux - noaux, paired over {len(rows)} seeds "
          f"(negative d rho = the auxiliary heads COST ranking) ===")
    print(f"{'iso':8} {'d Spearman':>22} {'t':>6} | {'d V2 ST-RAE':>22} {'t':>6}")
    macro = {"rho": [], "v2": []}
    for c in PRIMARY:
        iso = c[:6]
        line = f"{iso:8}"
        for key in ("rho", "v2"):
            d = np.array([r["aux"][iso][key] - r["noaux"][iso][key]
                          for r in rows if iso in r["aux"] and iso in r["noaux"]])
            if len(d) < 2:
                line += f"{'n/a':>31}"; continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            macro[key].append(d.mean())
            line += f" {d.mean():+13.4f} +-{se:.4f} {d.mean()/se if se else 0:6.2f} |"
        print(line.rstrip("|"))
    if macro["rho"]:
        print(f"{'MACRO':8} {np.mean(macro['rho']):+13.4f}{'':16}| "
              f"{np.mean(macro['v2']):+13.4f}")
    print("\nContext: a plain LightGBM falls 0.204 and 0.128 of rho short of the board on")
    print("CYP1A2 and CYP2C9. If the auxiliary heads do not explain that, the next candidate")
    print("is the ensembling itself, and nothing else obvious remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
