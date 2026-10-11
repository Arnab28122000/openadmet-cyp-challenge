r"""Pretrained chemical-language-model embeddings as a representation. Runs on Colab.

Companion to `cyp_dmpnn.py`. Both test a **representation**, which is the axis the evidence points
at -- model class on fixed features is a settled null here (H6 neural-vs-boosting 12/21 p=0.180;
every nested per-endpoint arm gap inside its seed spread) and the programme's verdict on the
endpoint that matters is that CYP2D6 *"needs a different representation, not one more feature
block"*.

They differ in one way that matters: a D-MPNN learns its representation **from scratch on 6,145
molecules**, which is the obvious reason it might fail. A pretrained encoder brings a
representation learned from millions of structures, so it tests the same axis without the
data-scarcity handicap.

## Contamination, checked rather than assumed

The programme's rule, set when Mol-JEPA was demoted: an encoder pretrained on a corpus that
**includes bioactivity labels** would carry exactly the contamination H5/H6 were designed to
exclude, and any gain from it would be uninterpretable. So only **structure-only** pretraining is
used here:

* `ibm/MoLFormer-XL-both-10pct` -- masked language modelling over ZINC + PubChem **SMILES
  strings**. No activity labels in the objective.
* `seyonec/ChemBERTa-zinc-base-v1` -- masked LM over ZINC SMILES. Same.
* `DeepChem/ChemBERTa-77M-MTR` -- multi-task *regression*, but on **computed physicochemical
  descriptors** (RDKit-derived), not measured bioactivity. Included and labelled, because the
  distinction matters: it is a cheaper version of information we already compute ourselves in the
  RDKit2D block, so a gain from it would be surprising and a null is uninformative.

None of the three was trained on CYP inhibition measurements. That is the claim; it rests on the
model cards and is recorded as such rather than independently verified.

## Arms

`fp-mlp` (the incumbent, the same 3,091-column block), `emb` (embeddings alone), `emb+fp`
(concatenated -- the primary arm, since the question is whether a pretrained representation adds
anything *on top of* what we have). Identical split, seed, loss, schedule and evaluation to
`cyp_dmpnn.py`, so the two experiments are directly comparable.

Headline score is the **mean of the last 10 evaluations**, not the minimum over epochs: this
programme measured that two seeds scored 0.4441/0.4448 with best epochs 360 and 25, so a
min-over-epochs on the selecting split is largely an extreme-value artifact.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONFIG_ID = "pretrained-embed-v1"
MODELS = {
    "molformer": ("ibm/MoLFormer-XL-both-10pct", True),
    "chemberta-zinc": ("seyonec/ChemBERTa-zinc-base-v1", False),
    "chemberta-mtr": ("DeepChem/ChemBERTa-77M-MTR", False),
}


def embed(smiles: list[str], repo: str, trust: bool, batch: int, device, cache: Path):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if cache.exists():
        z = np.load(cache)
        if z["n"] == len(smiles):
            print(f"[emb] cache hit {cache} {z['x'].shape}", flush=True)
            return z["x"]
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=trust)
    mod = AutoModel.from_pretrained(repo, trust_remote_code=trust).to(device).eval()
    outs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), batch):
            chunk = smiles[i:i + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            h = mod(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            outs.append(((h * m).sum(1) / m.sum(1).clamp(min=1e-6)).float().cpu().numpy())
    x = np.concatenate(outs).astype(np.float32)
    np.savez_compressed(cache, x=x, n=len(smiles))
    print(f"[emb] {repo} -> {x.shape} in {time.time()-t0:.1f}s, cached {cache}", flush=True)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="molformer", choices=tuple(MODELS))
    ap.add_argument("--arms", nargs="+", default=["fp-mlp", "emb", "emb+fp"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=300)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--embed-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--featcache", default="/content/featcache")
    ap.add_argument("--out", default="/content/out/embed")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    from drugrx.cyp.data import PRIMARY, load
    from drugrx.cyp.metric import METRIC_ID, interval_loss, st_rae
    from drugrx.data.splits import scaffold_split
    from drugrx.features import featurise, standardise_descriptors

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[emb] device {dev} · metric {METRIC_ID} · config {CONFIG_ID} · "
          f"model {args.model}", flush=True)

    fr = load(workers=args.workers).frame
    smi = fr["smiles"].astype(str).tolist()
    desc = featurise(smi, workers=args.workers, counts=True, desc_mode="rdkit2d",
                     extra_fps=True, cache=args.featcache)
    repo, trust = MODELS[args.model]
    emb = embed(smi, repo, trust, args.embed_batch, dev, out / f"emb_{args.model}.npz")
    print(f"[emb] features {desc.shape} · embeddings {emb.shape}", flush=True)

    Y = fr[list(PRIMARY)].to_numpy(float)
    LO = fr[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    HI = fr[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    MASK = np.isfinite(Y) & np.isfinite(LO) & np.isfinite(HI)

    def head(d):
        blocks, w = [], d
        for _ in range(args.layers):
            blocks += [nn.Linear(w, args.hidden), nn.ReLU(), nn.Dropout(args.dropout)]
            w = args.hidden
        return nn.Sequential(*blocks, nn.Linear(w, 4)).to(dev)

    results = {}
    for seed in range(args.seeds):
        tr, va, rest = scaffold_split(smi, frac_train=0.80, frac_valid=0.15, seed=seed)
        tr = np.concatenate([tr, rest])
        blocks = {}
        for name, mat in (("fp", desc), ("emb", emb)):
            _, stt = standardise_descriptors(mat[tr])
            z, _ = standardise_descriptors(mat, stats=stt)
            blocks[name] = torch.as_tensor(
                np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
                device=dev)
        yt = torch.as_tensor(np.nan_to_num(Y), dtype=torch.float32, device=dev)
        lot = torch.as_tensor(np.nan_to_num(LO), dtype=torch.float32, device=dev)
        hit = torch.as_tensor(np.nan_to_num(HI), dtype=torch.float32, device=dev)
        mt = torch.as_tensor(MASK.astype(np.float32), device=dev)

        for arm in args.arms:
            use = {"fp-mlp": ["fp"], "emb": ["emb"], "emb+fp": ["emb", "fp"]}[arm]
            X = torch.cat([blocks[u] for u in use], dim=1) if len(use) > 1 else blocks[use[0]]
            torch.manual_seed(seed); np.random.seed(seed)
            net = head(X.shape[1])
            opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
            curve, t0 = [], time.time()
            for ep in range(args.epochs):
                net.train()
                perm = np.random.permutation(len(tr))
                for b in range(0, len(perm), args.batch):
                    idx = torch.as_tensor(tr[perm[b:b + args.batch]], device=dev)
                    loss = interval_loss(net(X[idx]), lot[idx], hit[idx], mt[idx])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                sched.step()
                if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
                    net.eval()
                    with torch.no_grad():
                        p = net(X[torch.as_tensor(va, device=dev)]).float().cpu().numpy()
                    per = {c: st_rae(p[MASK[va, j], j], Y[va][MASK[va, j], j],
                                     LO[va][MASK[va, j], j], HI[va][MASK[va, j], j])
                           for j, c in enumerate(PRIMARY) if MASK[va, j].sum() > 20}
                    curve.append((ep + 1, float(np.mean(list(per.values()))), per))
            tail = curve[-10:]
            per_sm = {c: float(np.mean([t[2].get(c, np.nan) for t in tail])) for c in PRIMARY}
            best = min(curve, key=lambda c: c[1])
            results.setdefault(arm, []).append(
                dict(seed=seed, smooth=float(np.mean([c[1] for c in tail])), per=per_sm,
                     best=best[1], best_epoch=best[0], dim=int(X.shape[1]),
                     secs=time.time() - t0))
            r = results[arm][-1]
            print(f"[emb] seed {seed} {arm:<8s} dim {X.shape[1]:5d} smooth {r['smooth']:.4f} "
                  f"best {r['best']:.4f}@{r['best_epoch']}  "
                  + " ".join(f"{c[3:9]} {per_sm[c]:.4f}" for c in PRIMARY)
                  + f"  {r['secs']:.0f}s", flush=True)
            (out / f"results_{args.model}.json").write_text(json.dumps(
                dict(metric_id=METRIC_ID, config_id=CONFIG_ID, model=repo,
                     args=vars(args), results=results), indent=1, default=float))

    print(f"\n{'arm':<9s} {'macro(smooth)':>14s} {'sd':>8s} " +
          " ".join(f"{c[3:9]:>9s}" for c in PRIMARY))
    for arm in args.arms:
        if arm not in results:
            continue
        v = np.array([r["smooth"] for r in results[arm]])
        print(f"{arm:<9s} {v.mean():14.4f} {v.std(ddof=1):8.4f} " +
              " ".join(f"{np.mean([r['per'][c] for r in results[arm]]):9.4f}" for c in PRIMARY))

    if "fp-mlp" in results:
        print(f"\npaired against fp-mlp — per endpoint, the only level that counts")
        for arm in args.arms:
            if arm == "fp-mlp" or arm not in results:
                continue
            print(f"  {arm} vs fp-mlp:")
            for c in list(PRIMARY) + ["MACRO"]:
                key = (lambda r: r["smooth"]) if c == "MACRO" else (lambda r: r["per"][c])
                d = np.array([key(r) for r in results[arm]]) - \
                    np.array([key(r) for r in results["fp-mlp"]])
                sd = d.std(ddof=1) if len(d) > 1 else float("nan")
                t = d.mean() / (sd / math.sqrt(len(d))) if sd and sd > 0 else float("nan")
                print(f"    {c[:34]:<34s} {d.mean():+.4f} ± {sd:.4f}  t={t:+.2f}  "
                      f"{int((d < 0).sum())}/{len(d)} better  "
                      f"{'DECIDED' if abs(t) > 2.5 else ''}")
    print(f"\n[emb] wrote {out}/results_{args.model}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
