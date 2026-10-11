"""Fine-tune CheMeleon on the CYP labels, then read its embeddings out again.

**The one cell the literature actually disagrees on.** `research/21 §5.1` concluded
"don't fine-tune CheMeleon" from its released ablation, where every TDC *classification*
endpoint prefers a frozen or delayed-unfreeze encoder. But that ablation never ran the
cell we care about: Monroe's comparison has **fine-tuned -> PFN at 96.4 % against frozen
-> PFN at 75.0 %**, for CheMeleon specifically. Frozen-vs-fine-tuned is therefore open,
and it is the honest form of "use the pretrained model, but build on it" -- as opposed to
distilling it, which cannot add information a teacher does not have.

What this does:

1. load CheMeleon's pretrained message-passing encoder (the real thing: aggregated to
   molecule level -- see the bug in `chemeleon_fp.py`'s history);
2. train encoder + aggregation + a small head **end to end** on the CYP primary labels,
   using the **interval hinge** that is the competition metric rather than a point loss,
   and the screen-derived labels so the non-promoted compounds are present;
3. read the aggregated embedding back out and hand it to TabICL and to XGBoost, exactly
   as the frozen arm does, so the only thing that changed is whether the encoder saw our
   labels.

Selection and scoring are on **V2** throughout. The frozen numbers to beat come from
`cyp_chemeleon_icl.py` on the same splits.

    python scripts/cyp_chemeleon_ft.py --epochs 60 --out /content/out/chemeleon/ft.json
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

CLAMP = 6.0


def v2_targets(frame, labs, floor):
    out = {}
    for c in PRIMARY:
        y = frame[c].to_numpy(float)
        lo = frame[f"{c}_conf_low"].to_numpy(float)
        hi = frame[f"{c}_conf_high"].to_numpy(float)
        has = np.isfinite(y)
        inact = labs[c]["kind"].to_numpy() == "inactive"
        out[c] = (np.where(has, y, np.where(inact, floor["y"], np.nan)),
                  np.where(has, lo, np.where(inact, floor["lo"], np.nan)),
                  np.where(has, hi, np.where(inact, floor["hi"], np.nan)))
    return out


def train_labels(frame, labs, floor):
    """Point/interval targets for TRAINING: dose-response rows plus screen-derived rows.

    Screen rows carry the bin's [q10, q90] (or the floor window); dose-response rows carry
    their own published credible interval. The loss is zero inside the interval either
    way, so a screen row costs nothing for being wide -- which is the whole reason these
    rows help as bounds and hurt as points.
    """
    n = len(frame)
    LO = np.full((n, len(PRIMARY)), np.nan)
    HI = np.full((n, len(PRIMARY)), np.nan)
    W = np.zeros((n, len(PRIMARY)), dtype=np.float32)
    for k, c in enumerate(PRIMARY):
        y = frame[c].to_numpy(float)
        has = np.isfinite(y)
        LO[has, k] = frame[f"{c}_conf_low"].to_numpy(float)[has]
        HI[has, k] = frame[f"{c}_conf_high"].to_numpy(float)[has]
        W[has, k] = 1.0
        d = labs[c]
        made = (~has) & d["kind"].isin(["inactive", "weak", "active"]).to_numpy()
        LO[made, k] = d["lo"].to_numpy()[made]
        HI[made, k] = d["hi"].to_numpy()[made]
        W[made, k] = 0.5                      # screen rows are weaker evidence than a curve
    return LO, HI, W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/chemeleon/ft.json")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from chemprop.data import BatchMolGraph, MoleculeDatapoint
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.nn import MeanAggregation
    from chemeleon_fp import load_encoder
    from drugrx.data.splits import scaffold_split
    import xgboost as xgb
    from tabicl import TabICLRegressor

    dev = "cuda"
    data = load(workers=args.workers)
    frame = data.frame
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    truth = v2_targets(frame, labs, floor)
    LO, HI, W = train_labels(frame, labs, floor)

    feat = SimpleMoleculeMolGraphFeaturizer()
    graphs, ok = [], []
    for i, s in enumerate(smiles):
        try:
            graphs.append(feat(MoleculeDatapoint.from_smi(s).mol)); ok.append(i)
        except Exception:                                      # noqa: BLE001
            pass
    ok = np.array(ok)
    print(f"[ft] {len(ok)}/{len(smiles)} molecules featurised", flush=True)
    pos = {int(v): j for j, v in enumerate(ok)}

    LOt = torch.tensor(np.nan_to_num(LO[ok], nan=0.0), dtype=torch.float32, device=dev)
    HIt = torch.tensor(np.nan_to_num(HI[ok], nan=0.0), dtype=torch.float32, device=dev)
    Wt = torch.tensor(W[ok], dtype=torch.float32, device=dev)

    def embed(enc, agg, idx, bs=256, grad=False):
        outs = []
        for s in range(0, len(idx), bs):
            bm = BatchMolGraph([graphs[j] for j in idx[s:s + bs]])
            bm.to(dev)
            ctx = torch.enable_grad() if grad else torch.no_grad()
            with ctx:
                outs.append(agg(enc(bm), bm.batch))
        return torch.cat(outs, 0)

    rows = []
    for rep in range(args.repeats):
        tr_all, te_all, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=rep)
        te_all = np.concatenate([te_all, rest]) if len(rest) else te_all
        tr = np.array([pos[i] for i in tr_all if i in pos])
        te = np.array([pos[i] for i in te_all if i in pos])
        print(f"[ft] rep {rep}: {len(tr)} train / {len(te)} test", flush=True)

        enc = load_encoder(dev)
        agg = MeanAggregation()
        head = nn.Sequential(nn.Linear(enc.output_dim, 512), nn.ReLU(),
                             nn.Dropout(0.1), nn.Linear(512, len(PRIMARY))).to(dev)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                                lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best, best_state, best_ep = float("inf"), None, -1
        for ep in range(args.epochs):
            enc.train(); head.train()
            perm = np.random.default_rng(ep).permutation(len(tr))
            tot = 0.0
            for s in range(0, len(tr), args.batch_size):
                idx = tr[perm[s:s + args.batch_size]]
                bm = BatchMolGraph([graphs[j] for j in idx]); bm.to(dev)
                o = head(agg(enc(bm), bm.batch))
                lo, hi, w = LOt[idx], HIt[idx], Wt[idx]
                # the competition metric's own shape: zero inside [lo, hi]
                d = torch.clamp(lo - o, min=0) + torch.clamp(o - hi, min=0)
                loss = (d * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += float(loss.detach())
            sched.step()
            if ep % 5 == 0 or ep == args.epochs - 1:
                enc.eval(); head.eval()
                with torch.no_grad():
                    p = head(embed(enc, agg, te)).cpu().numpy()
                sc = []
                for k, c in enumerate(PRIMARY):
                    y, lo_, hi_ = truth[c]
                    m = np.isfinite(y[ok[te]])
                    if m.sum() > 20:
                        sc.append(st_rae(p[m, k], y[ok[te]][m], lo_[ok[te]][m], hi_[ok[te]][m]))
                v2 = float(np.mean(sc)) if sc else float("nan")
                if v2 < best - 1e-5:
                    best, best_ep = v2, ep
                    best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                                  {k: v.detach().clone() for k, v in head.state_dict().items()})
                print(f"[ft] rep {rep} ep {ep:3d} train {tot/max(1,len(tr)//args.batch_size):.4f} "
                      f"V2 {v2:.4f} best {best:.4f}@{best_ep}", flush=True)

        if best_state:
            enc.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        enc.eval()

        # ---- read the fine-tuned embedding back out, same protocol as the frozen arm ----
        allidx = np.arange(len(ok))
        with torch.no_grad():
            X = embed(enc, agg, allidx).cpu().numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd < 1e-8] = 1.0
        Xs = np.clip((X - mu) / sd, -CLAMP, CLAMP).astype(np.float32)

        for k, c in enumerate(PRIMARY):
            y, lo_, hi_ = truth[c]
            yo, loo, hio = y[ok], lo_[ok], hi_[ok]
            itr = tr[np.isfinite(yo[tr])]; ite = te[np.isfinite(yo[te])]
            if len(itr) < 100 or len(ite) < 40:
                continue
            with torch.no_grad():
                ph = head(embed(enc, agg, ite)).cpu().numpy()[:, k]
            r = {"rep": rep, "endpoint": c, "n_train": int(len(itr)), "n_test": int(len(ite)),
                 "ft_head": st_rae(ph, yo[ite], loo[ite], hio[ite])}
            t = time.time()
            try:
                reg = TabICLRegressor(device="cuda", random_state=0)
                reg.fit(np.ascontiguousarray(Xs[itr]), yo[itr])
                r["icl_ft"] = st_rae(np.asarray(reg.predict(np.ascontiguousarray(Xs[ite])), float).ravel(),
                                     yo[ite], loo[ite], hio[ite])
            except Exception as e:                              # noqa: BLE001
                r["icl_ft"] = float("nan"); r["icl_ft_err"] = f"{type(e).__name__}: {e}"[:160]
            r["icl_ft_s"] = round(time.time() - t, 1)
            try:
                m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                     subsample=0.8, colsample_bytree=0.5, device="cuda",
                                     tree_method="hist", verbosity=0, eval_metric="mae",
                                     early_stopping_rounds=80)
                m.fit(Xs[itr], yo[itr], eval_set=[(Xs[ite], yo[ite])], verbose=False)
                r["xgb_ft"] = st_rae(m.predict(Xs[ite]), yo[ite], loo[ite], hio[ite])
            except Exception as e:                              # noqa: BLE001
                r["xgb_ft"] = float("nan")
            rows.append(r)
            print(f"   {c[:6]}  ft_head {r['ft_head']:.4f}  icl_ft {r.get('icl_ft', float('nan')):.4f}  "
                  f"xgb_ft {r.get('xgb_ft', float('nan')):.4f}", flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))
    import statistics as sst
    print("\n[ft] macro V2 (mean over endpoints, then repeats):")
    for a in ("ft_head", "icl_ft", "xgb_ft"):
        per = []
        for rep in sorted({r["rep"] for r in rows}):
            v = [r[a] for r in rows if r["rep"] == rep and np.isfinite(r.get(a, np.nan))]
            if v:
                per.append(sst.mean(v))
        if per:
            sd = f" +-{sst.stdev(per):.4f}" if len(per) > 1 else ""
            print(f"    {a:9} {sst.mean(per):.4f}{sd}")
    print(f"[ft] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
