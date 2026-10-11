"""Is CheMeleon's representation worth anything, once the loss matches the metric?

`cyp_chemeleon_ft.py` fine-tuned CheMeleon end-to-end under the competition metric's own
interval hinge and scored **ft_head 0.5179 +-0.0130** macro V2, beating every frozen arm
(`xgb_ecfp` 0.5819, `xgb_chem` 0.6433, `icl_chem` 0.6616). I attributed that to the loss
rather than the representation, because embeddings read back out of the fine-tuned encoder
scored `xgb_ft` 0.5862 / `icl_ft` 0.5886 -- no better than frozen ECFP.

**That attribution was an inference, not a measurement.** No arm existed that paired our own
ECFP+descriptor block with the same hinge-trained head, so "the hinge did it" and "CheMeleon
did it" were not separated. This script separates them. Three arms, identical in every
respect except the feature block and whether it is learned:

    ecfp_head    fixed ECFP + rdkit2d descriptors  -> hinge head
    chem_head    FROZEN CheMeleon embedding        -> hinge head
    (ft_head)    fine-tuned CheMeleon              -> hinge head   [already measured]

Everything else is pinned to the fine-tuning script by import: the same `v2_targets` and
`train_labels`, the same scaffold split per repeat with `te = valid + rest`, the same head
(512-unit ReLU with dropout 0.1), the same AdamW at 3e-4 / weight decay 0.01, the same cosine
schedule, the same batch size, the same best-state-on-V2 selection and the same scorer.

Reading the result:

* `ecfp_head` ~= `ft_head`  -> the gain was the **loss**, and CheMeleon adds nothing. Close the
  foundation-model line and spend the budget on ranking instead.
* `ecfp_head` >> `ft_head`  -> CheMeleon's representation is doing real work after all, and the
  frozen-vs-fine-tuned cell resolves in favour of fine-tuning.
* `chem_head` sits between them and says how much of any gain is representation versus
  adaptation.

    python scripts/cyp_hinge_head.py --repeats 3 --out /content/out/chemeleon/hinge.json
"""
from __future__ import annotations

import argparse
import json
import statistics as sst
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_chemeleon_ft import train_labels, v2_targets
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels

CLAMP = 6.0


def scale_clamp(x: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu, sd = x[tr].mean(0), x[tr].std(0)
    sd[sd < 1e-8] = 1.0
    return np.clip((x - mu) / sd, -CLAMP, CLAMP).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cache", default="/content/data/chemeleon_cyp.npz")
    ap.add_argument("--arms", nargs="*", default=["ecfp_head", "chem_head"])
    ap.add_argument("--out", default="/content/out/chemeleon/hinge.json")
    args = ap.parse_args()

    import torch
    from torch import nn

    from drugrx.data.splits import scaffold_split
    from drugrx.features import featurise, standardise_descriptors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load(workers=args.workers)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    truth = v2_targets(frame, labs, floor)
    LO, HI, W = train_labels(frame, labs, floor)

    blocks = {}
    if "ecfp_head" in args.arms:
        xe = featurise(smiles, workers=args.workers, counts=True,
                       desc_mode="rdkit2d", extra_fps=True)
        blocks["ecfp_head"] = xe
        print(f"[hinge] ECFP block {xe.shape}", flush=True)
    if "chem_head" in args.arms:
        z = np.load(args.cache, allow_pickle=True)
        assert list(z["smiles"]) == smiles, "CheMeleon cache is for a different molecule list"
        blocks["chem_head"] = z["x"]
        print(f"[hinge] frozen CheMeleon block {z['x'].shape}", flush=True)

    LOt = torch.tensor(np.nan_to_num(LO, nan=0.0), dtype=torch.float32, device=dev)
    HIt = torch.tensor(np.nan_to_num(HI, nan=0.0), dtype=torch.float32, device=dev)
    Wt = torch.tensor(W, dtype=torch.float32, device=dev)

    rows = []
    for rep in range(args.repeats):
        tr, te, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=rep)
        te = np.concatenate([te, rest]) if len(rest) else te
        print(f"\n[hinge] rep {rep}: {len(tr)} train / {len(te)} test", flush=True)
        row = {"rep": rep, "n_train": int(len(tr)), "n_test": int(len(te))}

        for arm in args.arms:
            t0 = time.time()
            if arm == "ecfp_head":
                _, stt = standardise_descriptors(blocks[arm][tr])
                X, _ = standardise_descriptors(blocks[arm], stats=stt)
                X = X.astype(np.float32)
            else:
                X = scale_clamp(blocks[arm], tr)
            Xt = torch.tensor(X, dtype=torch.float32, device=dev)

            torch.manual_seed(rep)
            head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(512, len(PRIMARY))).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

            best, best_state, best_ep = float("inf"), None, -1
            for ep in range(args.epochs):
                head.train()
                perm = np.random.default_rng(ep).permutation(len(tr))
                tot, nb = 0.0, 0
                for s in range(0, len(tr), args.batch_size):
                    idx = tr[perm[s:s + args.batch_size]]
                    o = head(Xt[idx])
                    lo, hi, w = LOt[idx], HIt[idx], Wt[idx]
                    d = torch.clamp(lo - o, min=0) + torch.clamp(o - hi, min=0)
                    loss = (d * w).sum() / w.sum().clamp(min=1)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    tot += float(loss.detach())
                    nb += 1
                sched.step()
                if ep % 5 == 0 or ep == args.epochs - 1:
                    head.eval()
                    with torch.no_grad():
                        p = head(Xt[te]).cpu().numpy()
                    sc = []
                    for k, c in enumerate(PRIMARY):
                        y, lo_, hi_ = truth[c]
                        m = np.isfinite(y[te])
                        if m.sum() > 20:
                            sc.append(st_rae(p[m, k], y[te][m], lo_[te][m], hi_[te][m]))
                    v2 = float(np.mean(sc)) if sc else float("nan")
                    if v2 < best - 1e-5:
                        best, best_ep = v2, ep
                        best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                    if ep % 15 == 0 or ep == args.epochs - 1:
                        print(f"[hinge] {arm} rep {rep} ep {ep:3d} train {tot/max(nb,1):.4f} "
                              f"V2 {v2:.4f} best {best:.4f}@{best_ep}", flush=True)
            row[arm] = best
            row[arm + "_ep"] = best_ep
            row[arm + "_s"] = round(time.time() - t0, 1)
            print(f"[hinge] {arm} rep {rep}: best V2 {best:.4f} @ep {best_ep} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))

    print("\n[hinge] macro ST-RAE on V2, same splits/head/loss/scorer as cyp_chemeleon_ft.py:")
    for arm in args.arms:
        v = [r[arm] for r in rows if np.isfinite(r.get(arm, np.nan))]
        if v:
            sd = f" +-{sst.stdev(v):.4f}" if len(v) > 1 else ""
            print(f"    {arm:12} {sst.mean(v):.4f}{sd}   (n={len(v)})")
    print(f"    {'ft_head':12} 0.5179 +-0.0130   (n=2, measured earlier)")
    print(f"    {'xgb_ecfp':12} 0.5819            (frozen baseline)")
    print(f"\n[hinge] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
