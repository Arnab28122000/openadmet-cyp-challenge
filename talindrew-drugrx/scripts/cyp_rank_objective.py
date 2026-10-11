"""Does training a RANKING objective beat regression on rho?

Every lever measured so far changed the *features* or the *hyperparameters* under a squared-
error objective: screen labels, ensembling, TDI-condition labels, GBM tuning, auxiliary heads,
CheMeleon, the interval hinge, kNN blending. The objective itself has never been a ranking one,
and rho is the quantity that is both (a) the hard ceiling on R^2 via Murphy's decomposition and
(b) completely untouched by placement, which is where all the recent gains came from.

**The argument that this data suits a ranking loss specifically.** A large fraction of these
labels are *censored at the assay floor* -- the compound was inactive, so the truth is "below
~4.5", not a number. Squared error spends real capacity pulling predictions towards an
arbitrary point inside a censored interval, and `screen_labels` exists precisely to stop it
doing that badly. A ranking objective does not care where a censored label sits, only that it
sits below the actives, which is exactly the information the assay actually provides. So this
is not "another loss to try": censoring is the specific reason to expect a difference.

**The argument against, stated up front.** LightGBM's lambdarank needs discrete relevance
grades, so continuous pIC50 has to be binned, and binning throws away the resolution that
regression keeps among the actives. If ranking wins it must win *despite* that, and if it loses
by a little the honest reading is "binning cost more than censoring gained", not "ranking does
not work here".

Arms, identical in features, splits and seeds:

    l2            the incumbent regression objective (control)
    lambdarank    binned relevance, one group, NDCG-optimised
    rank_xendcg   cross-entropy variant, usually steadier than lambdarank on one group

Guards, all three from AGENTS.md: paired seeds, Delta-rho reported beside Delta-ST-RAE, and the
NN >= 0.5 restricted contrast -- the regime the blinded test set actually occupies, and the one
that reversed the kNN blend's apparent +0.0186 into -0.0616.

    python scripts/cyp_rank_objective.py --iso CYP3A4 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
CYPS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
#: Board rho - validation rho, per isoform. A validation gain is worth this much on the board.
LIFT = {"CYP1A2": 0.175, "CYP2C9": 0.135, "CYP2D6": 0.016, "CYP3A4": -0.0052}
N_BINS = 16          # overridden by --bins


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
            for s in smiles]


def nn_sim(query, ref):
    from rdkit import DataStructs
    ref = [f for f in ref if f is not None]
    out = np.zeros(len(query))
    for i, f in enumerate(query):
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else 0.0
    return out


def grade(y: np.ndarray, floor: float, n_bins: int = N_BINS) -> np.ndarray:
    """Relevance grades for a ranking objective.

    Everything at or below the assay floor collapses to grade 0 -- that is the point, and it is
    what the assay actually asserts. The actives are then split into equal-count bins so no
    grade is empty, which lambdarank needs.
    """
    g = np.zeros(len(y), dtype=int)
    act = y > floor
    if act.sum() < n_bins:
        return g
    q = np.quantile(y[act], np.linspace(0, 1, n_bins)[1:-1])
    g[act] = np.searchsorted(q, y[act], side="right") + 1
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso", default="CYP3A4", choices=CYPS)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bins", type=int, default=N_BINS,
                    help="relevance grades; raising this tests whether BINNING is the cost")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    iso = args.iso
    col = f"{iso}_pIC50_direct_inhibition"
    out_path = Path(args.out or
                    ROOT / f"data/cyp_cache/rank_objective_{iso}_b{args.bins}.json")

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)

    y = frame[col].to_numpy(float)
    lo = frame[f"{col}_conf_low"].to_numpy(float)
    hi = frame[f"{col}_conf_high"].to_numpy(float)
    # allr_floor returns one pooled floor window {y, lo, hi, n}, not a per-isoform map.
    fl = float(floor["y"])

    common = dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                  colsample_bytree=0.3, min_child_samples=20, verbose=-1, n_jobs=8)
    arms = {
        "l2": dict(objective="regression", **common),
        "lambdarank": dict(objective="lambdarank",
                           label_gain=list(range(args.bins + 1)), **common),
        # rank_xendcg uses label_gain too; LightGBM's default has only 31 entries, so a
        # higher --bins fails there as well unless it is passed explicitly.
        "rank_xendcg": dict(objective="rank_xendcg",
                            label_gain=list(range(args.bins + 1)), **common),
    }

    res = {k: {"rho": [], "rho_nn": [], "st": []} for k in arms}
    for seed in args.seeds:
        tr, va, _ = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        itr = np.array([i for i in tr if np.isfinite(y[i])])
        iva = np.array([i for i in va if np.isfinite(y[i])])
        if len(iva) < 30:
            continue
        nn = nn_sim(fps([smiles[i] for i in iva]), fps([smiles[i] for i in itr]))
        sel = nn >= 0.5
        g = grade(y[itr], fl, args.bins)

        print(f"\n[rank] {iso} seed {seed}: {len(itr)} train, {len(iva)} valid, "
              f"{int(sel.sum())} at NN>=0.5, grades {g.min()}..{g.max()}", flush=True)

        for name, params in arms.items():
            m = lgb.LGBMRanker(**params) if "rank" in params["objective"] \
                else lgb.LGBMRegressor(**params)
            if "rank" in params["objective"]:
                m.fit(X[itr], g, group=[len(itr)])
            else:
                m.fit(X[itr], y[itr])
            p = m.predict(X[iva])
            r = float(spearmanr(p, y[iva]).statistic)
            rn = float(spearmanr(p[sel], y[iva][sel]).statistic) if sel.sum() > 10 else np.nan
            # A ranking model's output is not on the pIC50 scale, so ST-RAE is only meaningful
            # after the same affine placement the real pipeline applies. Rank-map it onto the
            # regression arm's own distribution so the comparison is like-for-like.
            if "rank" in params["objective"]:
                ref = np.sort(res["l2"].get("_last_pred", p))
                p_scaled = ref[np.argsort(np.argsort(p)).clip(0, len(ref) - 1)]
            else:
                res["l2"]["_last_pred"] = p
                p_scaled = p
            s = float(st_rae(p_scaled, y[iva], lo[iva], hi[iva]))
            res[name]["rho"].append(r)
            res[name]["rho_nn"].append(rn)
            res[name]["st"].append(s)
            print(f"   {name:12} rho {r:.4f}  rho@NN {rn:.4f}  ST-RAE {s:.4f}", flush=True)

    base = np.array(res["l2"]["rho"])
    print(f"\n=== paired against `l2`, {len(base)} seeds, {iso} ===")
    print(f"{'arm':14}{'d rho':>10}{'t':>8}{'d rho @NN':>12}{'t':>8}{'d ST-RAE':>11}")
    summary = {}
    for name in arms:
        if name == "l2":
            continue
        d = np.array(res[name]["rho"]) - base
        dn = np.array(res[name]["rho_nn"]) - np.array(res["l2"]["rho_nn"])
        ds = np.array(res[name]["st"]) - np.array(res["l2"]["st"])
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 and d.std() else np.nan
        tn = dn.mean() / (dn.std(ddof=1) / np.sqrt(len(dn))) if len(dn) > 1 and dn.std() else np.nan
        print(f"{name:14}{d.mean():+10.4f}{t:8.2f}{np.nanmean(dn):+12.4f}{tn:8.2f}{ds.mean():+11.4f}")
        summary[name] = {"d_rho": float(d.mean()), "t": float(t),
                         "d_rho_nn": float(np.nanmean(dn)), "t_nn": float(tn),
                         "d_st_rae": float(ds.mean())}

    print(f"\nl2 rho {base.mean():.4f}; a validation gain here is worth "
          f"{LIFT[iso]:+.4f} more on the board for {iso}.")
    print("Bar: below +0.01 is noise at the scale that has already evaporated twice today.")
    print("Anything clearing it must be confirmed on seeds never looked at.")
    res.pop("_last_pred", None)
    for v in res.values():
        v.pop("_last_pred", None)
    out_path.write_text(json.dumps({"iso": iso, "seeds": args.seeds,
                                    "raw": res, "summary": summary}, indent=1, default=float))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
