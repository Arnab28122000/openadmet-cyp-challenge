"""In-context tabular models on the features we already have.

**Not TabPFN.** Its weights are gated behind a one-time licence acceptance tied to
a Hugging Face account, and accepting a licence is the owner's decision rather
than an agent's. Two ungated in-context regressors are used instead -- TabICL and
TabDPT -- which also makes a null attributable to the *approach* rather than to
one implementation.

The pretraining literature says our 6,525 labels sit in the dead zone where a
learned molecular representation does not beat count-ECFP (25 models x 25
datasets, five of them CYP Veith panels; count-ECFP mean rank 7.52/25). The same
literature points somewhere else: a **tabular foundation model**, which does not
learn a representation at all but does Bayesian inference in-context over a prior
of synthetic tabular problems. TabICLv2 beat Mol-JEPA on paired wins in
Mol-JEPA's *own* table, and our 4,905 compounds sit inside the <=10,000-sample
envelope these models are designed for.

**The honest obstacle is the feature budget.** These models are designed for
roughly <=500 features and we have ~3,100, so they cannot be given the same input
as the incumbent. Pretending otherwise would be the comparison error this programme
keeps catching, so the reduction is part of the arm and three different
reductions are run rather than one flattering choice:

* **_pca** -- 500 principal components, fitted on the training fold only.
* **_top** -- the 500 features with the largest absolute Spearman correlation
  to the endpoint, again ranked on the training fold only. (Ranking on all rows
  is a textbook selection leak and would quietly inflate this arm.)
* **_desc** -- the RDKit2D descriptor block alone, which is small enough to pass
  whole and is the chemically interpretable subset.

Every arm is scored against xgb on the **full** feature set on the same scaffold
splits, paired across repeats. A reduced-input model beating a full-input
gradient-boosting baseline would be a real finding; losing to it tells us the
reduction cost more than the prior bought, which is also worth knowing and is
the more likely outcome.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load          # noqa: E402
from drugrx.cyp.metric import st_rae               # noqa: E402
from drugrx.data.splits import scaffold_split      # noqa: E402
from drugrx.features import descriptor_span, featurise, standardise_descriptors  # noqa: E402


def spearman_rank_select(xtr, ytr, k):
    """Top-k by |Spearman|, ranked on the training fold only."""
    from scipy.stats import rankdata
    yr = rankdata(ytr)
    xr = np.apply_along_axis(rankdata, 0, xtr)
    xr -= xr.mean(0); yr = yr - yr.mean()
    num = xr.T @ yr
    den = np.sqrt((xr ** 2).sum(0) * (yr ** 2).sum()) + 1e-12
    r = np.abs(num / den)
    return np.argsort(-r)[:k]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-features", type=int, default=500)
    ap.add_argument("--arms", nargs="*", default=["xgb", "icl_pca", "icl_top", "icl_desc", "dpt_pca", "dpt_top", "dpt_desc"])
    ap.add_argument("--out", default="/content/out/cyp_incontext/incontext.json")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    lo_all = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi_all = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    desc = descriptor_span(x.shape[1])          # returns a slice; layout-aware
    print(f"[icl] features {x.shape[1]}, descriptor block {desc.start}:{desc.stop} "
          f"({desc.stop - desc.start} columns)", flush=True)

    import xgboost as xgb
    from sklearn.decomposition import PCA
    from tabicl import TabICLRegressor
    from tabdpt import TabDPTRegressor

    rows = []
    for rep in range(args.repeats):
        # scaffold_split requires the fractions to leave a remainder, so the
        # test fold is valid+rest rather than a single 0.15 request.
        tr, te, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                      frac_valid=0.10, seed=rep)
        te = np.concatenate([te, rest]) if len(rest) else te
        _, stt = standardise_descriptors(x[tr])
        xs, _ = standardise_descriptors(x, stats=stt)
        print(f"[icl] rep {rep}: {len(tr)} train / {len(te)} test", flush=True)

        # PCA fitted on the training fold only.
        pca = PCA(n_components=min(args.n_features, len(tr) - 1), random_state=0).fit(xs[tr])
        xp = pca.transform(xs)

        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            itr = tr[np.isfinite(y[tr])]
            ite = te[np.isfinite(y[te])]
            if len(itr) < 100 or len(ite) < 40:
                continue
            lor, hir, yr = lo_all[ite, i], hi_all[ite, i], y[ite]
            r = {"rep": rep, "endpoint": c, "n_test": int(len(ite)), "n_train": int(len(itr))}

            if "xgb" in args.arms:
                t = time.time()
                m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                     subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                     device="cuda", tree_method="hist", verbosity=0,
                                     eval_metric="mae", early_stopping_rounds=80)
                m.fit(xs[itr], y[itr], eval_set=[(xs[ite], y[ite])], verbose=False)
                r["xgb"] = st_rae(m.predict(xs[ite]), yr, lor, hir)
                r["xgb_s"] = round(time.time() - t, 1)

            def run_icl(name, maker, feats):
                """One in-context arm. Failures are recorded per arm rather than
                raised, so a model that cannot run does not cost the whole sweep."""
                t = time.time()
                try:
                    reg = maker()
                    reg.fit(np.ascontiguousarray(feats[itr], dtype=np.float32), y[itr])
                    p = reg.predict(np.ascontiguousarray(feats[ite], dtype=np.float32))
                    r[name] = st_rae(np.asarray(p, float).ravel(), yr, lor, hir)
                except Exception as e:
                    r[name] = float("nan"); r[name + "_err"] = f"{type(e).__name__}: {e}"[:200]
                r[name + "_s"] = round(time.time() - t, 1)

            MODELS = {"icl": lambda: TabICLRegressor(device="cuda", random_state=0),
                      "dpt": lambda: TabDPTRegressor(device="cuda")}
            sel = None
            for mk, maker in MODELS.items():
                if f"{mk}_pca" in args.arms:
                    run_icl(f"{mk}_pca", maker, xp)
                if f"{mk}_top" in args.arms:
                    if sel is None:
                        sel = spearman_rank_select(xs[itr], y[itr], args.n_features)
                    run_icl(f"{mk}_top", maker, xs[:, sel])
                if f"{mk}_desc" in args.arms:
                    run_icl(f"{mk}_desc", maker, xs[:, desc])

            rows.append(r)
            print("    " + f"{c:30s} " + "  ".join(
                f"{k}:{r[k]:.4f}" for k in args.arms
                if k in r and np.isfinite(r.get(k, float("nan")))), flush=True)
            for k in args.arms:
                if r.get(k + "_err"):
                    print(f"        {k} failed: {r[k+'_err']}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out, orient="records", indent=2)

    macro = df.groupby("rep").mean(numeric_only=True)
    print("\n[icl] macro ST-RAE per arm (lower is better)", flush=True)
    for k in args.arms:
        if k in macro and np.isfinite(macro[k]).all():
            print(f"    {k:8s} {macro[k].mean():.4f} (per-repeat sd {macro[k].std(ddof=1):.4f})")
    print("\n[icl] paired vs xgb on full features (positive = in-context better)", flush=True)
    for k in [a for a in args.arms if a != "xgb"]:
        if k not in macro or not np.isfinite(macro[k]).all():
            print(f"    {k:8s} unavailable"); continue
        d = macro["xgb"] - macro[k]
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se and se > 0 else float("nan")
        wins = int((d > 0).sum())
        # Direction matters: "REAL" alone labelled a consistent *loss* the same
        # way it labelled a win.
        if abs(t) > 2.5 and wins == len(d):
            tag = "BETTER"
        elif abs(t) > 2.5 and wins == 0:
            tag = "WORSE (consistent)"
        else:
            tag = "inside noise"
        print(f"    {k:8s} {d.mean():+.4f} +/- {se:.4f} (n={len(d)}, wins "
              f"{wins}/{len(d)}, t={t:+.2f}) -> {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
