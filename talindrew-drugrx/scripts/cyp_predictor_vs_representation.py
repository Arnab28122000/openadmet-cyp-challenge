"""Which lever is it -- the REPRESENTATION or the PREDICTOR? We only ever moved one.

THE REALISATION THIS SCRIPT EXISTS FOR. This programme has recorded, repeatedly, that "every
representation lever measured here is 0.0000" -- CheMeleon frozen, CheMeleon finetuned, JEPA-style
pretraining. Every one of those was measured **with LightGBM nailed down as the predictor.** We
changed the representation and held the head fixed, and concluded the representation does not
matter.

The literature's win is the other factorisation. Monroe (arXiv 2608.18982, open weights) reports
that "PFN-based downstream predictors also substantially improve two leading existing models,
MiniMol and CheMeleon, yielding new state-of-the-art variants we call MiniMol_PFN and
CheMeleon_PFN". And an independent benchmark by Pat Walters -- who is also an OpenADMET programme
author, so an insider on this data family -- pits **Monroe embedding + TabPFN against Morgan
fingerprints + LightGBM** on ExpansionRx (5,326 train / 2,282 test, 9 endpoints, 5x5
cluster-grouped CV, Tukey HSD with explicit ties) and reports it winning **5 of 9**, plus **6 of 6**
on Biogen ADME (3,521 molecules). Our n is 6,145. That is the closest published protocol to our
setting, and the configuration it tests is the one cell of the grid we have never occupied.

Corroborating the same factorisation from the other direction, Guan et al. (CSIRO, arXiv
2606.31126) find that for "small-molecule classification with ECFP/RDKit descriptors, no single
predictor-representation pairing dominates", and conclude tabular foundation models "can be strong
biomolecular predictors, **but only when coupled with expressive representations**". That is also
the explanation for our own `cyp_tabfm.py` result: TabICL on RAW descriptors came out a tie at
matched features (CYP3A4 -0.0014). Right experiment, wrong input.

So: a representation x predictor grid, which is the only design that can attribute the effect.

    rep         what
    desc        our 3091 Morgan-count + RDKit-2D features (production)
    chem        CheMeleon's frozen 2048-d encoder output (MIT licence, clean commercially)
    both        concatenation -- the form molecular LeJEPA found positive, 0.803 -> 0.830

    predictor   what
    lgbm        the production 8-member bootstrap LightGBM ensemble
    tabicl      TabICLv2 regressor (permissive licence; TabPFN is excluded, see below)

THE CONTRASTS, and which are new:

    chem_lgbm  - desc_lgbm    representation effect, predictor fixed. ALREADY MEASURED at ~0.000.
                              This arm is here to REPLICATE that, as the control.
    tabicl     - lgbm         **predictor effect, representation fixed. NEVER MEASURED.**
    best cell  - desc_lgbm    what the grid is actually worth.

DIMENSION HANDLING, which is not cosmetic. TabICL is pre-trained on 2-100 columns and documented to
degrade well beyond 100, and our own sweep confirmed a real cost to a 100-feature budget
(lgbm_topk - lgbm_all = -0.024 macro). So each representation is reduced for the TabICL arms, by
the method appropriate to its structure:

  * `desc` is SPARSE and axis-meaningful (a fingerprint bit is "this substructure is present"), so
    PCA would destroy exactly the structure a tree exploits -> top-k by LightGBM split gain.
  * `chem` is a DENSE learned embedding with no privileged axes -> PCA, which is also what
    molecular LeJEPA used (PCA-32 of the embedding concatenated with a fingerprint).

Selection and PCA are both fitted on TRAINING ROWS ONLY and refitted per seed, so they sit inside
the cross-validation rather than before it.

LICENSING. CheMeleon is MIT (weights Zenodo 10.5281/zenodo.15426600, pretraining data CC-BY-4.0,
PubChem public domain) and TabICL is permissively licensed, so every arm here is usable
commercially. **TabPFN is deliberately absent**: its weights require a one-time licence acceptance
and its terms prohibit production deployment and commercial benchmarking, so a win from it could
not be submitted or shipped. If the predictor effect is real, TabICL is the one we can act on.

PRE-REGISTERED, written before the run.

1. **The replication control can void the whole thing.** `chem_lgbm - desc_lgbm` must come out near
   zero, consistent with the programme's prior measurement. If CheMeleon suddenly looks good under
   LightGBM, something differs from the earlier run (pooling, scaling, weights) and the grid is
   measuring that difference rather than the predictor.
2. **The bar: the best cell must beat `desc_lgbm` on CYP3A4 by >= +0.015 Spearman on >= 4 of 5
   seeds.** CYP3A4 is primary -- board rank 137, the most real labels (2335) and almost no
   pseudo-label contamination, so it is both our worst cell relative to the field and the cleanest
   supervised problem of the four.
3. **Honest expectation, stated first.** I expect `chem_lgbm ~= desc_lgbm` (replicating 0.000), and
   the predictor contrast to be POSITIVE but modest -- +0.005 to +0.02 -- with `chem_pca + tabicl`
   the best cell. I do NOT expect the +0.03 end of the research estimate, because (a) Yao &
   Duran-Frigola's pre-registered parity test put a 2048-d foundation embedding statistically
   indistinguishable from a compact descriptor portfolio (0.764 vs 0.762, CI [-0.020, +0.030]) at
   low n, and (b) on CYP specifically, tuned XGBoost beat deep learning 90.4% to 88.5% (Wu et al.
   2019) and multitask DL across five isoforms lost to single-task XGBoost.
4. **Report per isoform always.** A macro gain carried by CYP2D6 (60.5% pseudo-label mass, zero
   dual-labelled compounds) is a different result from one on CYP3A4.
5. **Structural-overlap caveat, recorded in advance.** Walters found ~53% of Biogen test molecules
   present as *structures* in Monroe's PM6 pretraining corpus. CheMeleon is pretrained on PubChem
   descriptors, so our compounds are almost certainly in its corpus too. That is not label leakage,
   but it means a frozen-embedding gain is partly "this molecule was seen unlabelled" and should
   not be read as pure generalisation.

    python cyp_predictor_vs_representation.py --npz cyp_gpu.npz --smiles smiles.json \
        --seeds 0 1 2 3 4 --pca 32 100 --topk 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
N_MEMBERS = 8
GBM_KW = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
              n_jobs=10, verbose=-1)
PRIMARY_ENDPOINT = "CYP3A4"
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"


def gbm(seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=seed, **GBM_KW)


def selftest():
    import lightgbm as lgb
    want = dict(GBM_KW); want["random_state"] = 7
    got = gbm(7).get_params()
    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    assert not bad, f"learner drifted from production: {bad}"
    assert N_MEMBERS == 8
    print(f"[selftest] learner matches production ({lgb.__version__}); ensemble 8")


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def chemeleon_embed(smiles, cache_path, device="cuda", batch_size=256):
    """Frozen CheMeleon encoder output, mean-pooled over atoms. Cached to disk.

    Mean pooling is CheMeleon's own recipe and the only pooling that keeps the vector comparable
    across molecules of different size -- a sum would make the embedding scale with heavy-atom
    count, which is most of what the descriptor block already encodes.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        z = np.load(cache_path)
        print(f"[chem] loaded cached embedding {z['E'].shape} from {cache_path}", flush=True)
        return z["E"]
    import torch
    import urllib.request
    from chemprop.data import BatchMolGraph, MoleculeDatapoint
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    from chemprop.nn import BondMessagePassing, MeanAggregation

    wt = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not wt.exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        print(f"[chem] downloading weights -> {wt}", flush=True)
        urllib.request.urlretrieve(CHEMELEON_URL, wt)
    blob = torch.load(wt, weights_only=True, map_location=device)
    # Build from the checkpoint's own hyper_parameters: constructing from guesses would load the
    # weights into a differently shaped network and silently produce nonsense.
    enc = BondMessagePassing(**blob["hyper_parameters"])
    enc.load_state_dict(blob["state_dict"])
    enc.eval().to(device)
    agg = MeanAggregation()
    fz = SimpleMoleculeMolGraphFeaturizer()
    E = np.zeros((len(smiles), enc.output_dim), dtype=np.float32)
    bad = 0
    with torch.no_grad():
        for st in range(0, len(smiles), batch_size):
            chunk = smiles[st:st + batch_size]
            graphs, keep = [], []
            for i, smi in enumerate(chunk):
                try:
                    graphs.append(fz(MoleculeDatapoint.from_smi(smi).mol)); keep.append(i)
                except Exception:
                    bad += 1          # leave the row as zeros rather than dropping it
            if not graphs:
                continue
            bmg = BatchMolGraph(graphs).to(device)
            H = enc(bmg)
            V = agg(H, bmg.batch).float().cpu().numpy()
            for j, i in enumerate(keep):
                E[st + i] = V[j]
    print(f"[chem] embedded {len(smiles)} molecules, dim {enc.output_dim}, {bad} failures",
          flush=True)
    np.savez_compressed(cache_path, E=E)
    return E


def reduce_sparse(X, y, w, idx, k, seed):
    """top-k by LightGBM split gain, fitted on TRAINING rows only."""
    m = gbm(seed).fit(X[idx], y[idx], sample_weight=w[idx])
    imp = np.asarray(m.booster_.feature_importance(importance_type="gain"), float)
    return np.argsort(-imp)[:k]


def reduce_dense(E, idx, k, seed):
    """PCA fitted on TRAINING rows only. Returns a transform closure."""
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(k, E.shape[1], idx.size - 1), random_state=seed).fit(E[idx])
    return p.transform


def fit_lgbm(X, y, w, idx, Xva, seed):
    rng = np.random.default_rng(1000 + seed)
    preds = []
    for mi in range(N_MEMBERS):
        bs = rng.choice(idx, size=idx.size, replace=True)
        preds.append(gbm(seed * 100 + mi).fit(X[bs], y[bs],
                                              sample_weight=w[bs]).predict(Xva))
    return np.vstack(preds).mean(axis=0)


def fit_tabicl(X, y, idx, Xva, seed):
    from tabicl import TabICLRegressor
    m = TabICLRegressor(random_state=seed)
    m.fit(X[idx], y[idx])
    return np.asarray(m.predict(Xva), float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cyp_gpu.npz")
    ap.add_argument("--smiles", default="smiles.json")
    ap.add_argument("--chem-cache", default="chemeleon_emb.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--pca", type=int, nargs="*", default=[32, 100])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default="pred_vs_rep.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return 0

    z = np.load(args.npz, allow_pickle=False)
    X = np.asarray(z["X"], float)
    isoforms = [str(c).split("_")[0] for c in z["isoforms"]]
    smiles = json.load(open(args.smiles))
    assert len(smiles) == X.shape[0], f"{len(smiles)} smiles vs {X.shape[0]} rows"
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    E = chemeleon_embed(smiles, args.chem_cache, device=dev)
    try:
        import tabicl  # noqa: F401
        has_icl = True
    except Exception as exc:
        has_icl = False
        print(f"[grid] tabicl unavailable: {type(exc).__name__}")
    print(f"[grid] desc {X.shape}  chem {E.shape}  tabicl {has_icl}  "
          f"primary {PRIMARY_ENDPOINT}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va = z[f"tr_{seed}"].astype(int), z[f"va_{seed}"].astype(int)
        print(f"\n[grid] seed {seed}", flush=True)
        for s in isoforms:
            y = z[f"y_{s}"].astype(float)
            kind = z[f"kind_{s}"]
            plab = z[f"plab_{s}"].astype(float)
            use = np.zeros(len(y), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, plab)
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if idx.size < 50 or okv.sum() < 60:
                continue
            Both = np.hstack([X, E])
            plan = [("desc", "lgbm", X, None), ("chem", "lgbm", E, None),
                    ("both", "lgbm", Both, None)]
            if has_icl:
                cols = reduce_sparse(X, yy, w, idx, args.topk, seed)
                plan.append((f"desc_top{args.topk}", "tabicl", X[:, cols], None))
                for k in args.pca:
                    tf = reduce_dense(E, idx, k, seed)
                    plan.append((f"chem_pca{k}", "tabicl", tf(E), None))
                    # the concatenation form, reduced consistently on each block
                    plan.append((f"both_pca{k}", "tabicl",
                                 np.hstack([X[:, cols], tf(E)]), None))
            for rep, pred, M, _ in plan:
                t0 = time.time()
                try:
                    if pred == "lgbm":
                        p = fit_lgbm(M, yy, w, idx, M[vv], seed)
                    else:
                        p = fit_tabicl(M, yy, idx, M[vv], seed)
                    rho, err = spearman(p, yv), None
                except Exception as exc:
                    rho, err = float("nan"), f"{type(exc).__name__}: {exc}"[:140]
                arm = f"{rep}+{pred}"
                rows.append(dict(seed=seed, iso=s, arm=arm, rep=rep, pred=pred, rho=rho,
                                 d=int(M.shape[1]), secs=round(time.time() - t0, 1),
                                 error=err))
                print(f"   {s} {arm:20} d={M.shape[1]:5d}: rho {rho:.4f}  "
                      f"{rows[-1]['secs']}s" + (f"  ERROR {err}" if err else ""), flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1
    seeds = sorted({r["seed"] for r in rows})
    arms = sorted({r["arm"] for r in rows})

    def cell(iso, arm, seed):
        v = [r["rho"] for r in rows if r["iso"] == iso and r["arm"] == arm and r["seed"] == seed]
        return v[0] if v else np.nan

    print("\n" + "=" * 100)
    print("1. THE REPLICATION CONTROL (item 1): chem+lgbm - desc+lgbm must be near zero")
    print("=" * 100)
    for iso in isoforms:
        d = np.array([cell(iso, "chem+lgbm", s) - cell(iso, "desc+lgbm", s) for s in seeds])
        d = d[np.isfinite(d)]
        if d.size:
            print(f"   {iso:8} {d.mean():+.4f}  ({int((d>0).sum())}/{d.size} positive)")
    print("   Near zero replicates the programme's prior measurement. A large positive here means")
    print("   something differs from the earlier CheMeleon run and the grid measures THAT.")

    print("\n" + "=" * 100)
    print("2. THE PREDICTOR EFFECT, representation fixed -- the contrast never measured here")
    print("=" * 100)
    for rep_pair in [("desc", f"desc_top{args.topk}")] + \
                    [("chem", f"chem_pca{k}") for k in args.pca]:
        a, b = f"{rep_pair[0]}+lgbm", f"{rep_pair[1]}+tabicl"
        if b not in arms:
            continue
        print(f"\n   {b}  minus  {a}")
        for iso in isoforms:
            d = np.array([cell(iso, b, s) - cell(iso, a, s) for s in seeds])
            d = d[np.isfinite(d)]
            if d.size:
                print(f"   {iso:8} {d.mean():+.4f}  ({int((d>0).sum())}/{d.size})")

    print("\n" + "=" * 100)
    print(f"3. THE BAR (item 2): best cell must beat desc+lgbm on {PRIMARY_ENDPOINT} by >= +0.015")
    print("=" * 100)
    print(f"   {'arm':22} {'CYP3A4 d':>10} {'wins':>8} {'macro d':>9}")
    best = (None, -9.9)
    for arm in arms:
        if arm == "desc+lgbm":
            continue
        per = {}
        for iso in isoforms:
            d = np.array([cell(iso, arm, s) - cell(iso, "desc+lgbm", s) for s in seeds])
            d = d[np.isfinite(d)]
            if d.size:
                per[iso] = (d.mean(), int((d > 0).sum()), d.size)
        if not per:
            continue
        p3 = per.get(PRIMARY_ENDPOINT, (float("nan"), 0, 0))
        mac = float(np.mean([v[0] for v in per.values()]))
        print(f"   {arm:22} {p3[0]:+10.4f} {p3[1]:4d}/{p3[2]:<3d} {mac:+9.4f}")
        if np.isfinite(p3[0]) and p3[0] > best[1]:
            best = (arm, p3[0], p3[1], p3[2])
    if best[0]:
        ok = best[1] >= 0.015 and best[2] >= max(4, int(0.8 * best[3]))
        print(f"\n   best on {PRIMARY_ENDPOINT}: {best[0]} at {best[1]:+.4f} "
              f"on {best[2]}/{best[3]} seeds -> "
              f"{'CLEARS BAR' if ok else 'below bar'}")
    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n   {len(errs)} cell(s) errored; first: {errs[0]['arm']} -> {errs[0]['error']}")
    print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
