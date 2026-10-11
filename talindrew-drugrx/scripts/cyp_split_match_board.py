"""Validate the validation: which held-out subset predicts the board's ranking?

`cyp_split_realism.py` measured that the blinded test set is **easier than either split we
use** -- median nearest-neighbour ECFP4 Tanimoto to training 0.581, against 0.439 for our
scaffold split and 0.455 for a random one, with 96.9% of test compounds having a neighbour at
>= 0.4 against 68% for scaffold. An analog expansion of training hits gives every test compound
a close relative by construction, so our validation is dominated by a hard tail the board never
asks about.

That is a measurement about chemistry, not about score. This script turns it into one about
score, and it is falsifiable in one direction: the board reports Spearman per isoform, so a
validation surface is **right** to the extent that the rho it reports matches the rho the board
reports for the same model.

Three surfaces, same model, same predictions, scored on nested subsets of the same held-out
compounds:

    all          every held-out compound (what we have always used)
    near         only those with NN Tanimoto >= NEAR, the test's own regime
    matched      importance-weighted to the test's full NN-similarity histogram

If `near`/`matched` land closer to the board's rho than `all` does, model selection should move
to that surface. If they do not, the chemistry finding stands but has no consequence for
selection, and this line closes cheaply -- which is worth as much.

Scored with Spearman and not ST-RAE deliberately: rho is invariant to the affine placement step
that sits between the model and the board, so it is the one quantity that compares the model to
its own leaderboard row without the placement confound.

    python scripts/cyp_split_match_board.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.matched import matched_spearman
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
NEAR = 0.50          # the test's 10th percentile: below it, the board has essentially nothing
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}

# Our live board row (v4 submission, read 2026-10-09). Placement is monotone, so these
# rho values belong to the MODEL and are directly comparable to a validation rho.
BOARD_RHO = {"CYP1A2": 0.7597, "CYP2C9": 0.8084, "CYP2D6": 0.5004, "CYP3A4": 0.7685}


def short(col: str) -> str:
    """PRIMARY carries full column names; the board reports per isoform."""
    return col.split("_")[0]


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
            for s in smiles]


def nn_sim(query, ref):
    from rdkit import DataStructs
    out = np.zeros(len(query))
    ref = [f for f in ref if f is not None]
    for i, f in enumerate(query):
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else 0.0
    return out


def hist_weights(v: np.ndarray, target: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Importance weights making `v`'s similarity histogram match `target`'s."""
    ht, _ = np.histogram(target, bins=edges)
    hv, _ = np.histogram(v, bins=edges)
    ht = ht / max(ht.sum(), 1)
    hv = hv / max(hv.sum(), 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(edges) - 2)
    w = np.where(hv[idx] > 0, ht[idx] / np.maximum(hv[idx], 1e-12), 0.0)
    return w


def wspearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Spearman: Pearson on ranks, weighted. Ties averaged."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    mx, my = (rx * w).sum() / sw, (ry * w).sum() / sw
    cx, cy = rx - mx, ry - my
    den = np.sqrt((w * cx * cx).sum() * (w * cy * cy).sum())
    return float((w * cx * cy).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/split_match_board.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[match] features {X.shape}", flush=True)

    all_fp = fps(smiles)
    test_fp = fps(data.test["SMILES"].tolist())
    edges = np.linspace(0.0, 1.0, 21)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        ref = [all_fp[i] for i in tr]
        sim_va = nn_sim([all_fp[i] for i in va], ref)
        sim_te = nn_sim(test_fp, ref)
        w_match = hist_weights(sim_va, sim_te, edges)
        print(f"\n[match] seed {seed}: {len(tr)} train / {len(va)} val; "
              f"val median NN {np.median(sim_va):.3f}, test median NN {np.median(sim_te):.3f}, "
              f"val fraction >= {NEAR}: {np.mean(sim_va >= NEAR):.3f}", flush=True)

        for c in PRIMARY:
            # Label assembly is copied from scripts/cyp_screen_lgbm.py's `drc+screen` arm
            # verbatim, including the hyperparameters, so that the only thing varying
            # between the two scripts is which held-out compounds get scored.
            lab = labs[c]
            y = frame[c].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool)
            use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            if len(idx) < 50:
                continue
            t0 = time.time()
            m = lgb.LGBMRegressor(
                n_estimators=args.n_estimators, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                n_jobs=10, verbose=-1)
            m.fit(X[idx], yy[idx], sample_weight=w[idx])
            p = m.predict(X[va])

            # score only on held-out compounds that carry a real DRC label
            keep = np.isfinite(y[va])
            yv, pv, sv, wv = y[va][keep], p[keep], sim_va[keep], w_match[keep]
            if keep.sum() < 30:
                continue
            near = sv >= NEAR
            r_all = float(spearmanr(pv, yv).statistic)
            r_near = float(spearmanr(pv[near], yv[near]).statistic) if near.sum() > 30 else float("nan")
            r_match = wspearman(pv, yv, wv)
            # The PRODUCTION instrument, so this comparison is about the estimator actually in
            # use and not a proxy for it: `drugrx.cyp.matched` uses 5 coarse NN bands and
            # weight-proportional resampling, where `wspearman` above uses 20 bins and a
            # weighted rank correlation. Same idea, different estimator.
            r_prod, prod_sd, prod_ess = matched_spearman(pv, yv, sv)
            b = BOARD_RHO[short(c)]
            rows.append(dict(seed=seed, iso=c, n=int(keep.sum()), n_near=int(near.sum()),
                             rho_all=r_all, rho_near=r_near, rho_match=r_match,
                             rho_prodmatch=float(r_prod), prodmatch_sd=float(prod_sd),
                             prodmatch_ess=int(prod_ess), board=b,
                             secs=round(time.time() - t0, 1)))
            print(f"   {short(c)}: n={keep.sum():4d} (near {near.sum():4d})  "
                  f"rho all {r_all:.4f}  near {r_near:.4f}  matched {r_match:.4f}  "
                  f"prod {r_prod:.4f} (ess {prod_ess})  "
                  f"board {b:.4f}   [{time.time()-t0:.0f}s]", flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== mean |validation rho - board rho| over seeds, lower is a better surface ===")
    print(f"{'iso':8} {'board':>7} {'all':>8} {'near':>8} {'matched':>8}   "
          f"{'|err| all':>10} {'|err| near':>11} {'|err| match':>12}")
    errs = {k: [] for k in ("all", "near", "match")}
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if not rs:
            continue
        b = BOARD_RHO[short(c)]
        got = {k: float(np.nanmean([r[f"rho_{k}"] for r in rs])) for k in ("all", "near", "match")}
        for k in errs:
            errs[k].append(abs(got[k] - b))
        print(f"{short(c):8} {b:7.4f} {got['all']:8.4f} {got['near']:8.4f} {got['match']:8.4f}   "
              f"{abs(got['all']-b):10.4f} {abs(got['near']-b):11.4f} {abs(got['match']-b):12.4f}")
    print(f"\n{'MACRO':8} {'':7} {'':8} {'':8} {'':8}   "
          f"{np.mean(errs['all']):10.4f} {np.mean(errs['near']):11.4f} "
          f"{np.mean(errs['match']):12.4f}")

    # Power check, before any verdict. The `near` and `matched` subsets are small, so their
    # rho is noisy by construction, and a ranking of surfaces that is smaller than its own
    # standard error is not a result. Report the comparison only when it is resolvable.
    print("\n--- can this experiment actually tell the surfaces apart? ---")
    se = {}
    for k in ("all", "near", "match"):
        per = [float(np.std([r[f"rho_{k}"] for r in rows if r["iso"] == c], ddof=1))
               for c in PRIMARY if any(r["iso"] == c for r in rows)]
        se[k] = float(np.sqrt(sum((s / np.sqrt(len(args.seeds))) ** 2 for s in per)) / len(per))
        print(f"  macro standard error of '{k}' over {len(args.seeds)} seeds: {se[k]:.4f}")
    spread = max(np.mean(errs[k]) for k in errs) - min(np.mean(errs[k]) for k in errs)
    worst_se = max(se.values())
    print(f"  surfaces are separated by {spread:.4f}; largest standard error {worst_se:.4f}")
    if spread < 2 * worst_se:
        print("  -> UNDERPOWERED. The ranking of surfaces is inside its own noise; do not act")
        print(f"     on it. The 'near' subsets hold only "
              f"{min(r['n_near'] for r in rows)}-{max(r['n_near'] for r in rows)} compounds, "
              f"where rho alone carries se ~{1/np.sqrt(min(r['n_near'] for r in rows)-3):.2f}.")
        print("     Add seeds, or pool predictions across seeds before computing rho.")
    else:
        best = min(errs, key=lambda k: float(np.mean(errs[k])))
        print(f"  -> resolvable: closest surface to the board is '{best}'")

    # What IS resolvable per isoform: does this plain LightGBM reach the board's rho at all?
    print("\n--- per isoform, is the gap to the board bigger than the seed noise? ---")
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if len(rs) < 2:
            continue
        v = [r["rho_all"] for r in rs]
        mu, sd = float(np.mean(v)), float(np.std(v, ddof=1))
        b = BOARD_RHO[short(c)]
        z = abs(mu - b) / sd if sd > 0 else float("inf")
        verdict = "MATCHES the board" if z < 2 else f"falls {b-mu:+.3f} short ({z:.1f} sd)"
        print(f"  {short(c):8} lgbm {mu:.4f} +-{sd:.4f}   board {b:.4f}   {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
