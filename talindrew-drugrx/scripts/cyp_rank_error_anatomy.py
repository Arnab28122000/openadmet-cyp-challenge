"""Where does the CYP3A4 model actually lose its ranking?

The field's best CYP3A4 rho is 0.8741 and ours is 0.7685 -- a gap of +0.106, and on this board
that is 155 places. Every *lever* has been measured and found empty, so the useful question is
no longer "what else could I add" but "which compounds are we getting wrong", because a loss
concentrated in an identifiable subgroup points somewhere and a loss spread evenly does not.

The diagnostic is rank error: for each validation compound, |rank(prediction) - rank(truth)|
normalised to [0,1]. Then slice it by things that could plausibly explain a systematic failure:

* **potency decile** -- are we mis-ordering the actives, or failing to separate inactives?
  Spearman is dominated by whichever block is larger, and these have different fixes.
* **nearest-neighbour similarity to the training set** -- the extrapolation regime. The blinded
  test set sits at high similarity, so an error concentrated at low NN matters less than it
  looks, and one at high NN matters more.
* **censoring status** -- censored labels have no true rank, only a bound. If the loss is
  concentrated there, the ceiling is the labels, not the model.
* **scaffold frequency** -- singletons versus compounds from a well-represented series.

A rank error is computed against the *observed* label, so for censored compounds it is partly
an artefact of pinning them to the floor value; that slice is reported separately rather than
pooled, and the `n` for each slice is printed so a suggestive mean over 12 compounds is visible
as such.

    python scripts/cyp_rank_error_anatomy.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import load
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
ISO = "CYP3A4"
COL = f"{ISO}_pIC50_direct_inhibition"
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}


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


def scaffolds(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(s))
        except Exception:
            out.append("")
    return out


def report(name: str, groups: dict[str, np.ndarray], err: np.ndarray) -> list[dict]:
    rows = []
    print(f"\n  by {name}:")
    print(f"    {'slice':22} {'n':>5} {'mean rank err':>14} {'vs overall':>11}")
    base = err.mean()
    for k, mask in groups.items():
        if mask.sum() == 0:
            continue
        m = float(err[mask].mean())
        rows.append({"slice": k, "n": int(mask.sum()), "mean_rank_err": m})
        flag = "  <-- worst" if m > base * 1.25 and mask.sum() >= 20 else ""
        print(f"    {k:22} {int(mask.sum()):5d} {m:14.4f} {m - base:+11.4f}{flag}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/rank_error_anatomy.json"))
    args = ap.parse_args()

    import lightgbm as lgb
    from scipy.stats import rankdata, spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    n = len(frame)
    labs, floor = screen_labels(frame), allr_floor(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    y = frame[COL].to_numpy(float)
    kind = labs[COL]["kind"].to_numpy()
    yy = np.where(np.isfinite(y), y, labs[COL]["y"].to_numpy(float))
    allfp = fps(smiles)
    scaf = scaffolds(smiles)
    scaf_n = Counter(scaf)

    out = {"iso": ISO, "seeds": args.seeds, "per_seed": []}
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        use = np.zeros(n, bool); use[tr] = True
        drc = use & np.isfinite(y)
        ps = use & np.isin(kind, ("inactive", "weak", "active"))
        w0 = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
        idx = np.flatnonzero(drc | ps)

        m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                              min_child_samples=20, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                              n_jobs=10, verbose=-1)
        m.fit(X[idx], yy[idx], sample_weight=w0[idx])

        v1 = np.isfinite(y[va])
        vi = va[v1]
        p = m.predict(X[vi])
        t = y[vi]
        rho = float(spearmanr(p, t).statistic)
        k = len(vi)
        err = np.abs(rankdata(p) - rankdata(t)) / k      # 0 = perfect, 1 = reversed

        sim = nn_sim([allfp[i] for i in vi], [allfp[i] for i in tr])
        print(f"\n[anatomy] seed {seed}: {k} scored validation compounds, rho {rho:.4f}, "
              f"mean rank error {err.mean():.4f}")

        dec = np.digitize(t, np.quantile(t, [0.2, 0.4, 0.6, 0.8]))
        rows = {}
        rows["potency quintile"] = report("potency quintile (0 = least potent)",
            {f"Q{d}": dec == d for d in range(5)}, err)
        rows["NN similarity"] = report("nearest-neighbour similarity to train",
            {"<0.3": sim < 0.3, "0.3-0.5": (sim >= 0.3) & (sim < 0.5),
             "0.5-0.7": (sim >= 0.5) & (sim < 0.7), ">=0.7": sim >= 0.7}, err)
        at_floor = t <= floor["y"] + 1e-9
        rows["at assay floor"] = report("label at the assay floor",
            {"at floor": at_floor, "above floor": ~at_floor}, err)
        sc = np.array([scaf_n[scaf[i]] for i in vi])
        rows["scaffold frequency"] = report("Murcko scaffold frequency in the full set",
            {"singleton": sc <= 1, "2-5": (sc > 1) & (sc <= 5),
             "6-20": (sc > 5) & (sc <= 20), ">20": sc > 20}, err)

        out["per_seed"].append({"seed": seed, "rho": rho, "n": k,
                                "mean_rank_err": float(err.mean()), "slices": rows})

    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {args.out}")
    print("\nA loss concentrated in one slice points somewhere. A flat profile says the model is")
    print("uniformly behind, which would mean the gap is capacity or data, not a fixable subgroup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
