r"""A learned molecular representation (D-MPNN) against the fixed-fingerprint incumbent.

Runs on Colab. Pre-registered in `PROGRESS.md` (2026-10-06): model *class* on fixed features is a
settled null here (H6 neural-vs-boosting 12/21 p=0.180; every nested per-endpoint arm gap inside
its seed spread), so this tests the thing that has never been tested -- a **learned**
representation. The programme's own verdict on the endpoint that matters is that *"93% of CYP2D6's
label variance is real signal, and this endpoint needs a different representation, not one more
feature block."*

## Three arms, one script, one split per seed

* `fp-mlp`    -- MLP on the cached fixed feature block (ECFP counts + RDKit2D + Avalon + ErG,
                 3,091 columns). **The incumbent.**
* `dmpnn`     -- directed message passing over the molecular graph only. The clean representation
                 test, and the weak config: D-MPNN without molecule-level features usually loses
                 to fingerprints on small data, so a loss here says little.
* `dmpnn+fp`  -- the graph embedding concatenated with the same 3,091 features. **The primary
                 arm**, because it asks the question that matters: does a learned representation
                 add anything *on top of* what we already have?

All three share the split, the seed, the loss, the schedule and the evaluation, so no effect is
measured at a config the comparison arm did not also run. That is the `config_id` discipline, and
it exists because an ensembling gain of -0.0168 was once read against a curve measured at
16 members/400 epochs instead of 32/300.

## Scoring: a three-way split, because both obvious choices are wrong

Two traps, and the first version of this script walked into the second.

* **Minimum over epochs on the split you report** is optimistically biased -- this programme
  measured two seeds at 0.4441/0.4448 whose best epochs were 360 and 25, so "best epoch" is
  largely an extreme-value artifact.
* **Mean of the last N evaluations** is worse here. The programme's own curves *bottom at epoch
  190-365 and then rise monotonically*, with every run 10-17% off its own best by epoch 700. A
  tail average therefore measures the overfitting region, not performance: at 250 epochs it
  scored `fp-mlp` at **1.0386**, worse than predicting the mean, against 0.7847 for the same arm
  at epoch 10.

So: `scaffold_split` three ways. Train on `tr`, **early-stop on `fit`**, and report the stopped
model's score on `rep`, which nothing selected on. That is the tr/fit/rep pattern
`cyp_recalibrate.py` already uses, and it is the only arrangement here that is neither
selection-biased nor dominated by overfitting.

Per-endpoint reporting throughout. A macro-only improvement does not count: a macro mean is what
hid CYP2D6 at ST-RAE 1.4577 behind three healthy endpoints for three weeks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG_ID = "dmpnn-v1"

ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 34, 1]
HYB = None  # filled after rdkit import


def atom_features(a) -> list[float]:
    from rdkit.Chem import rdchem
    z = a.GetAtomicNum()
    f = [1.0 if z == e else 0.0 for e in ELEMENTS] + [1.0 if z not in ELEMENTS else 0.0]
    deg = a.GetTotalDegree()
    f += [1.0 if deg == i else 0.0 for i in range(6)]
    ch = a.GetFormalCharge()
    f += [1.0 if ch == i else 0.0 for i in (-2, -1, 0, 1, 2)] + \
         [1.0 if ch not in (-2, -1, 0, 1, 2) else 0.0]
    f += [1.0 if int(a.GetChiralTag()) == i else 0.0 for i in range(4)]
    nh = a.GetTotalNumHs()
    f += [1.0 if nh == i else 0.0 for i in range(5)]
    h = a.GetHybridization()
    f += [1.0 if h == x else 0.0 for x in HYB] + [1.0 if h not in HYB else 0.0]
    f += [1.0 if a.GetIsAromatic() else 0.0, 1.0 if a.IsInRing() else 0.0,
          a.GetMass() * 0.01]
    return f


def bond_features(b) -> list[float]:
    from rdkit.Chem import rdchem
    bt = b.GetBondType()
    f = [1.0 if bt == x else 0.0 for x in (rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE,
                                           rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC)]
    f += [1.0 if b.GetIsConjugated() else 0.0, 1.0 if b.IsInRing() else 0.0]
    f += [1.0 if int(b.GetStereo()) == i else 0.0 for i in range(6)]
    return f


def build_graphs(smiles: list[str]):
    """One graph per molecule, as flat arrays. Directed bonds: edge 2k and 2k+1 are reverses."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None or m.GetNumAtoms() == 0:
            out.append(None)
            continue
        xa = np.array([atom_features(a) for a in m.GetAtoms()], dtype=np.float32)
        src, dst, eb = [], [], []
        for b in m.GetBonds():
            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            bf = bond_features(b)
            src += [u, v]; dst += [v, u]; eb += [bf, bf]
        out.append((xa,
                    np.array(src, dtype=np.int64),
                    np.array(dst, dtype=np.int64),
                    np.array(eb, dtype=np.float32).reshape(len(src), -1)))
    return out


def collate(graphs, idx, device):
    """Batch graphs by concatenation with index offsets. Returns tensors + atom->mol map."""
    import torch
    xs, srcs, dsts, ebs, mol_of_atom = [], [], [], [], []
    ao = 0
    for k, i in enumerate(idx):
        g = graphs[i]
        xa, src, dst, eb = g
        xs.append(xa)
        srcs.append(src + ao); dsts.append(dst + ao)
        if len(eb):
            ebs.append(eb)
        mol_of_atom.append(np.full(len(xa), k, dtype=np.int64))
        ao += len(xa)
    t = lambda a, d: torch.as_tensor(np.concatenate(a) if a else np.zeros((0,), dtype=d),
                                     device=device)
    x = torch.as_tensor(np.concatenate(xs), device=device)
    src = t(srcs, np.int64); dst = t(dsts, np.int64)
    eb = torch.as_tensor(np.concatenate(ebs) if ebs else
                         np.zeros((0, ebs[0].shape[1] if ebs else 12), dtype=np.float32),
                         device=device)
    m2a = torch.as_tensor(np.concatenate(mol_of_atom), device=device)
    return x, src, dst, eb, m2a, len(idx)


class DMPNN:
    pass


def make_model(kind: str, n_atom: int, n_bond: int, n_desc: int, hidden: int, depth: int,
               head_hidden: int, head_layers: int, dropout: float, device):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            self.depth = depth
            emb = 0
            if kind in ("dmpnn", "dmpnn+fp"):
                self.W_i = nn.Linear(n_atom + n_bond, hidden, bias=False)
                self.W_h = nn.Linear(hidden, hidden, bias=False)
                self.W_o = nn.Linear(n_atom + hidden, hidden)
                emb += hidden
            if kind in ("fp-mlp", "dmpnn+fp"):
                emb += n_desc
            blocks, d = [], emb
            for _ in range(head_layers):
                blocks += [nn.Linear(d, head_hidden), nn.ReLU(), nn.Dropout(dropout)]
                d = head_hidden
            self.head = nn.Sequential(*blocks, nn.Linear(d, 4))
            self.act = nn.ReLU()

        def graph_embed(self, x, src, dst, eb, m2a, nmol):
            h0 = self.act(self.W_i(torch.cat([x[src], eb], dim=1)))
            h = h0
            for _ in range(self.depth - 1):
                a_msg = torch.zeros(x.shape[0], h.shape[1], device=x.device, dtype=h.dtype)
                a_msg.index_add_(0, dst, h)
                # message on edge u->v excludes the reverse edge v->u
                rev = torch.arange(h.shape[0], device=x.device) ^ 1
                m = a_msg[src] - h[rev]
                h = self.act(h0 + self.W_h(m))
            a_msg = torch.zeros(x.shape[0], h.shape[1], device=x.device, dtype=h.dtype)
            a_msg.index_add_(0, dst, h)
            hv = self.act(self.W_o(torch.cat([x, a_msg], dim=1)))
            out = torch.zeros(nmol, hv.shape[1], device=x.device, dtype=hv.dtype)
            cnt = torch.zeros(nmol, 1, device=x.device, dtype=hv.dtype)
            out.index_add_(0, m2a, hv)
            cnt.index_add_(0, m2a, torch.ones_like(cnt[m2a]))
            return out / cnt.clamp(min=1.0)

        def forward(self, batch, desc):
            parts = []
            if self.kind in ("dmpnn", "dmpnn+fp"):
                parts.append(self.graph_embed(*batch))
            if self.kind in ("fp-mlp", "dmpnn+fp"):
                parts.append(desc)
            return self.head(torch.cat(parts, dim=1) if len(parts) > 1 else parts[0])

    return Net().to(device)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["fp-mlp", "dmpnn", "dmpnn+fp"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--patience", type=int, default=50,
                    help="epochs without improvement ON THE INNER SET before stopping")
    ap.add_argument("--hidden", type=int, default=300,
                    help="GRAPH message hidden size (chemprop's default is 300)")
    ap.add_argument("--head-hidden", type=int, default=64,
                    help="prediction-head width. The incumbent is NARROW (v2 'L3/32', "
                         "v3deep 'L6/16'), which is what stops a 3,091-feature input "
                         "overfitting 4,300 rows in five epochs.")
    ap.add_argument("--head-layers", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--rep-from", default="/content/baseline_pool.npz",
                    help="npz carrying the INCUMBENT's own va_index and its pooled cached "
                         "predictions. Scoring on those exact rows makes the baseline the "
                         "real incumbent rather than a reimplementation of it.")
    ap.add_argument("--featcache", default="/content/featcache")
    ap.add_argument("--out", default="/content/out/dmpnn")
    args = ap.parse_args()

    import torch
    from rdkit.Chem import rdchem
    global HYB
    HYB = [rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2,
           rdchem.HybridizationType.SP3, rdchem.HybridizationType.SP3D,
           rdchem.HybridizationType.SP3D2]

    from drugrx.cyp.data import PRIMARY, load
    from drugrx.cyp.metric import METRIC_ID, interval_loss, st_rae
    from drugrx.data.splits import scaffold_split
    from drugrx.features import featurise, standardise_descriptors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[dmpnn] device {dev} · metric {METRIC_ID} · config {CONFIG_ID}", flush=True)

    data = load(workers=args.workers)
    fr = data.frame
    smi = fr["smiles"].astype(str).tolist()
    t0 = time.time()
    desc = featurise(smi, workers=args.workers, counts=True, desc_mode="rdkit2d",
                     extra_fps=True, cache=args.featcache)
    print(f"[dmpnn] features {desc.shape} in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    graphs = build_graphs(smi)
    bad = [i for i, g in enumerate(graphs) if g is None]
    print(f"[dmpnn] graphs built in {time.time()-t0:.1f}s · {len(bad)} unparsed", flush=True)
    n_atom = graphs[[i for i, g in enumerate(graphs) if g is not None][0]][0].shape[1]
    n_bond = 12
    for g in graphs:
        if g is not None and len(g[3]):
            n_bond = g[3].shape[1]; break
    print(f"[dmpnn] atom feats {n_atom} · bond feats {n_bond}", flush=True)

    Y = fr[list(PRIMARY)].to_numpy(float)
    LO = fr[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    HI = fr[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    MASK = np.isfinite(Y) & np.isfinite(LO) & np.isfinite(HI)

    def macro_on(p, idx):
        per = {}
        for j, c in enumerate(PRIMARY):
            k = MASK[idx, j]
            if k.sum() > 20:
                per[c] = st_rae(p[k, j], Y[idx][k, j], LO[idx][k, j], HI[idx][k, j])
        return float(np.mean(list(per.values()))), per

    # The incumbent's own split and its own pooled predictions, so the baseline is the real
    # thing. A reimplementation of it scored 0.84 against the incumbent's 0.67 on an equivalent
    # split -- that gap is this harness being a weaker learner, not a finding, and comparing a
    # new arm against it would have been a straw man.
    pool = np.load(args.rep_from)
    rep_fixed = pool["va_index"]
    base_val = pool["val"]
    m_base, per_base = None, None

    results = {}
    pooled_pred: dict[str, list] = {}
    for seed in range(args.seeds):
        # `rep` is the incumbent's validation rows, fixed across seeds. The early-stopping set
        # is carved out of the remainder by scaffold, so it never touches `rep`.
        va = np.array([i for i in rep_fixed if graphs[i] is not None])
        pool_rest = np.setdiff1d(np.arange(len(smi)), rep_fixed)
        sub = [smi[i] for i in pool_rest]
        s_tr, s_fit, s_rest = scaffold_split(sub, frac_train=0.80, frac_valid=0.15, seed=seed)
        tr = np.array([i for i in np.concatenate([pool_rest[s_tr], pool_rest[s_rest]])
                       if graphs[i] is not None])
        fit = np.array([i for i in pool_rest[s_fit] if graphs[i] is not None])
        if m_base is None:
            m_base, per_base = macro_on(base_val[np.isin(rep_fixed, va)], va)
            print(f"[dmpnn] INCUMBENT on these exact rows ({len(va)}): macro {m_base:.4f}  "
                  + " ".join(f"{c[3:9]} {per_base[c]:.4f}" for c in PRIMARY)
                  + f"   [{int(pool['n_seeds'])} pooled seeds]", flush=True)
            results["INCUMBENT-cached"] = [dict(seed=-1, rep=m_base, per=per_base,
                                                fit=float('nan'), stop_epoch=0,
                                                epochs_run=0, secs=0.0, params=0)]
        _, stt = standardise_descriptors(desc[tr])
        ds, _ = standardise_descriptors(desc, stats=stt)
        ds = np.nan_to_num(ds, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dt = torch.as_tensor(ds, device=dev)
        yt = torch.as_tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
        lot = torch.as_tensor(np.nan_to_num(LO), dtype=torch.float32, device=dev)
        hit = torch.as_tensor(np.nan_to_num(HI), dtype=torch.float32, device=dev)
        mt = torch.as_tensor(MASK.astype(np.float32), device=dev)

        for arm in args.arms:
            torch.manual_seed(seed); np.random.seed(seed)
            net = make_model(arm, n_atom, n_bond, desc.shape[1], args.hidden, args.depth,
                             args.head_hidden, args.head_layers, args.dropout, dev)
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
            # Cosine rather than OneCycle: OneCycleLR's warmup phase collapses to a single
            # step at small `total_steps` and then divides by zero computing the phase
            # fraction, which is exactly what a 10-epoch smoke test hits. Cosine has no
            # phase boundary, and the programme's own curves (validation bottoms at epoch
            # 190-365 then rises monotonically) want a decaying rate anyway.
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
            curve, t0 = [], time.time()
            va_batch = collate(graphs, va, dev) if arm != "fp-mlp" else None
            fit_batch = collate(graphs, fit, dev) if arm != "fp-mlp" else None
            best_fit, best_rep, best_per, best_ep, since = math.inf, None, None, 0, 0
            best_pred = None
            for ep in range(args.epochs):
                net.train()
                perm = np.random.permutation(len(tr))
                for b in range(0, len(perm), args.batch):
                    idx = tr[perm[b:b + args.batch]]
                    batch = collate(graphs, idx, dev) if arm != "fp-mlp" else None
                    it = torch.as_tensor(idx, device=dev)
                    pred = net(batch, dt[it])
                    loss = interval_loss(pred, lot[it], hit[it], mt[it])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                sched.step()
                if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        pf = net(fit_batch, dt[torch.as_tensor(fit, device=dev)]
                                 ).float().cpu().numpy()
                        pv = net(va_batch, dt[torch.as_tensor(va, device=dev)]
                                 ).float().cpu().numpy()
                    m_fit, _ = macro_on(pf, fit)
                    m_rep, per_rep = macro_on(pv, va)
                    curve.append((ep + 1, m_fit, m_rep))
                    # Early stopping is decided on `fit`; `rep` is only recorded.
                    if m_fit < best_fit - 1e-6:
                        best_fit, best_rep, best_per, best_ep, since = \
                            m_fit, m_rep, per_rep, ep + 1, 0
                        best_pred = pv.copy()
                    else:
                        since += args.eval_every
                        if since >= args.patience:
                            break
            pooled_pred.setdefault(arm, []).append(best_pred)
            results.setdefault(arm, []).append(
                dict(seed=seed, rep=best_rep, per=best_per, fit=best_fit,
                     stop_epoch=best_ep, epochs_run=curve[-1][0] if curve else 0,
                     rep_min=float(min(c[2] for c in curve)) if curve else None,
                     rep_final=float(curve[-1][2]) if curve else None,
                     secs=time.time() - t0,
                     params=sum(q.numel() for q in net.parameters())))
            print(f"[dmpnn] seed {seed} {arm:<10s} REP {best_rep:.4f} (stopped @{best_ep}/"
                  f"{curve[-1][0] if curve else 0}, fit {best_fit:.4f})  "
                  + " ".join(f"{c[3:9]} {best_per.get(c, float('nan')):.4f}" for c in PRIMARY)
                  + f"  {time.time()-t0:.0f}s", flush=True)
            (out / "results.json").write_text(json.dumps(
                dict(metric_id=METRIC_ID, config_id=CONFIG_ID, args=vars(args),
                     results=results), indent=1, default=float))

    print(f"\n{'arm':<11s} {'macro(REP)':>12s} {'seed sd':>9s} {'stop ep':>8s} " +
          " ".join(f"{c[3:9]:>9s}" for c in PRIMARY))
    for arm in args.arms:
        if arm not in results:
            continue
        v = np.array([r["rep"] for r in results[arm]])
        ep = np.mean([r["stop_epoch"] for r in results[arm]])
        print(f"{arm:<11s} {v.mean():12.4f} {v.std(ddof=1):9.4f} {ep:8.0f} " +
              " ".join(f"{np.mean([r['per'].get(c, np.nan) for r in results[arm]]):9.4f}"
                       for c in PRIMARY))

    # POOLED scores — the only like-for-like comparison with the incumbent, which is itself
    # an 81-seed pooled ensemble. A single net cannot be compared with a pooled one, and the
    # first version of this table did exactly that.
    print(f"\n{'arm':<18s} {'POOLED macro':>13s} {'vs incumbent':>13s} " +
          " ".join(f"{c[3:9]:>9s}" for c in PRIMARY))
    print(f"{'INCUMBENT (81)':<18s} {m_base:13.4f} {0.0:+13.4f} " +
          " ".join(f"{per_base[c]:9.4f}" for c in PRIMARY))
    pooled_summary = {"INCUMBENT": dict(macro=m_base, per=per_base)}
    for arm, preds in pooled_pred.items():
        ok = [q for q in preds if q is not None]
        if not ok:
            continue
        mp, pp = macro_on(np.mean(ok, axis=0), va)
        pooled_summary[arm] = dict(macro=mp, per=pp, n=len(ok))
        print(f"{arm + f' ({len(ok)})':<18s} {mp:13.4f} {mp - m_base:+13.4f} " +
              " ".join(f"{pp.get(c, float('nan')):9.4f}" for c in PRIMARY))
    (out / "results.json").write_text(json.dumps(
        dict(metric_id=METRIC_ID, config_id=CONFIG_ID, args=vars(args),
             results=results, pooled=pooled_summary), indent=1, default=float))

    base = "fp-mlp"
    if base in results:
        print(f"\npaired against {base} (the incumbent) — per endpoint, which is the only level"
              f" that counts")
        for arm in args.arms:
            if arm == base or arm not in results:
                continue
            print(f"  {arm} vs {base}:")
            for c in list(PRIMARY) + ["MACRO"]:
                if c == "MACRO":
                    d = np.array([r["rep"] for r in results[arm]]) - \
                        np.array([r["rep"] for r in results[base]])
                else:
                    d = np.array([r["per"].get(c, np.nan) for r in results[arm]]) - \
                        np.array([r["per"].get(c, np.nan) for r in results[base]])
                d = d[np.isfinite(d)]
                sd = d.std(ddof=1) if len(d) > 1 else float("nan")
                t = d.mean() / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("nan")
                flag = "DECIDED" if abs(t) > 2.5 else ("band" if sd and abs(d.mean()) < sd
                                                       else "")
                print(f"    {c[:34]:<34s} {d.mean():+.4f} ± {sd:.4f}  t={t:+.2f}  "
                      f"{int((d < 0).sum())}/{len(d)} better  {flag}")
        print("\n  Decision rule was |t| > 2.5 on CYP2D6. Seed sd is reported first because the")
        print("  pre-registration expected the effect to sit UNDER the band — in which case the")
        print("  result is 'unresolvable at this seed count', not a null.")
    print(f"\n[dmpnn] wrote {out/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
