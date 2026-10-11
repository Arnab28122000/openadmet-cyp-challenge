"""CheMeleon embeddings into an in-context regressor, at native width.

**Why this arm exists: the earlier in-context null was not a null about in-context
learning.** `cyp_incontext.py` swapped the *model* while holding the featuriser at
count-ECFP and then PCA'd it to 500 components. Both of those are separately documented
losing choices, so the arm could not have shown a gain even if one were there:

* count-ECFP vs CheMeleon under the *same* tabular foundation model and splits is a
  **22.4 % vs 86.2 %** win rate over 58 tasks -- the featuriser, not the head, is what
  moves these numbers;
* the published protocol says **"no dimensionality reduction"** (standard-scale and
  clamp instead), and our own `*_pca` arms came in at -0.0547/-0.1098, which was us
  independently rediscovering that result and then reading it as a verdict on ICL;
* the <=500-feature budget belongs to TabPFN **v2**; TabICL/TabDPT take ~2,000, and
  TabPFN subsamples features per estimator anyway, so 2048 needs no PCA at all.

And the thing that makes this cheap: `scripts/chemeleon_fp.py` has been in the tree for
days and **nothing in `drugrx/cyp/` has ever called it**. This is one featuriser swap
away from an experiment that already ran, against a known in-tree baseline
(`icl_top` = 0.4597).

The comparison is deliberately narrow: the SAME model, the SAME splits, the SAME
scorer, changing only the feature block. Anything else and we would again be unable to
say what moved.

    python scripts/cyp_chemeleon_icl.py --repeats 3 --out /content/out/chemeleon/icl.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels

CLAMP = 6.0            # the published protocol: standard-scale, then clamp to [-6, 6]


def scale_clamp(x: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """Standard-scale on the training fold only, then clamp. No PCA -- that is the point."""
    mu = x[tr].mean(0)
    sd = x[tr].std(0)
    sd[sd < 1e-8] = 1.0
    return np.clip((x - mu) / sd, -CLAMP, CLAMP).astype(np.float32)


def v2_targets(frame, labs, floor):
    """V1 (dose-response rows) plus held-out screen inactives against the floor window.

    The surface that reproduces the board per isoform; selecting on V1 is what hid the
    selection bias for two days.
    """
    out = {}
    for i, c in enumerate(PRIMARY):
        y = frame[c].to_numpy(float)
        lo = frame[f"{c}_conf_low"].to_numpy(float)
        hi = frame[f"{c}_conf_high"].to_numpy(float)
        has = np.isfinite(y)
        inact = labs[c]["kind"].to_numpy() == "inactive"
        out[c] = (np.where(has, y, np.where(inact, floor["y"], np.nan)),
                  np.where(has, lo, np.where(inact, floor["lo"], np.nan)),
                  np.where(has, hi, np.where(inact, floor["hi"], np.nan)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/content/out/chemeleon/icl.json")
    ap.add_argument("--cache", default="/content/data/chemeleon_cyp.npz")
    ap.add_argument("--arms", nargs="*",
                    default=["xgb_ecfp", "xgb_chem", "icl_ecfp", "icl_chem", "icl_both"])
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    from drugrx.features import featurise, standardise_descriptors
    from drugrx.data.splits import scaffold_split

    data = load(workers=args.workers)
    frame = data.frame
    smiles = frame["smiles"].tolist()
    labs, floor = screen_labels(frame), allr_floor(frame)
    truth = v2_targets(frame, labs, floor)

    # ---- the two feature blocks ----
    xe = featurise(smiles, workers=args.workers, counts=True,
                   desc_mode="rdkit2d", extra_fps=True)
    print(f"[chem] ECFP/descriptor block: {xe.shape}", flush=True)

    cache = Path(args.cache)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["smiles"]) == smiles:
            xc = z["x"]
            print(f"[chem] CheMeleon cache hit: {xc.shape}", flush=True)
        else:
            cache.unlink(); xc = None
    else:
        xc = None
    if xc is None:
        from chemeleon_fp import featurise as chem_featurise
        t = time.time()
        xc = chem_featurise(smiles, device="cuda", batch_size=256)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, x=xc, smiles=np.array(smiles, dtype=object))
        print(f"[chem] CheMeleon {xc.shape} in {time.time()-t:.0f}s -> {cache}", flush=True)

    import xgboost as xgb
    from tabicl import TabICLRegressor

    rows = []
    for rep in range(args.repeats):
        tr, te, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=rep)
        te = np.concatenate([te, rest]) if len(rest) else te
        _, stt = standardise_descriptors(xe[tr])
        xes, _ = standardise_descriptors(xe, stats=stt)
        xcs = scale_clamp(xc, tr)                       # native 2048, no reduction
        xbs = np.concatenate([xcs, xes], axis=1).astype(np.float32)
        print(f"[chem] rep {rep}: {len(tr)} train / {len(te)} test | "
              f"ecfp {xes.shape[1]} chem {xcs.shape[1]} both {xbs.shape[1]}", flush=True)

        for i, c in enumerate(PRIMARY):
            y, lo, hi = truth[c]
            itr = tr[np.isfinite(y[tr])]
            ite = te[np.isfinite(y[te])]
            if len(itr) < 100 or len(ite) < 40:
                continue
            r = {"rep": rep, "endpoint": c, "n_train": int(len(itr)), "n_test": int(len(ite))}
            yr, lor, hir = y[ite], lo[ite], hi[ite]

            def run(name, fn):
                if name not in args.arms:
                    return
                t = time.time()
                try:
                    p = np.asarray(fn(), float).ravel()
                    r[name] = st_rae(p, yr, lor, hir)
                except Exception as e:
                    r[name] = float("nan")
                    r[name + "_err"] = f"{type(e).__name__}: {e}"[:200]
                r[name + "_s"] = round(time.time() - t, 1)

            def gbm(feats):
                m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                     subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                     device="cuda", tree_method="hist", verbosity=0,
                                     eval_metric="mae", early_stopping_rounds=80)
                m.fit(feats[itr], y[itr], eval_set=[(feats[ite], y[ite])], verbose=False)
                return m.predict(feats[ite])

            def icl(feats):
                reg = TabICLRegressor(device="cuda", random_state=0)
                reg.fit(np.ascontiguousarray(feats[itr], dtype=np.float32), y[itr])
                return reg.predict(np.ascontiguousarray(feats[ite], dtype=np.float32))

            run("xgb_ecfp", lambda: gbm(xes))
            run("xgb_chem", lambda: gbm(xcs))
            run("icl_ecfp", lambda: icl(xes))
            run("icl_chem", lambda: icl(xcs))
            run("icl_both", lambda: icl(xbs))
            rows.append(r)
            keys = [k for k in args.arms if k in r]
            print("   " + c[:6] + "  " + "  ".join(f"{k} {r[k]:.4f}" for k in keys), flush=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))

    print("\n[chem] macro ST-RAE on V2 (mean over endpoints, then over repeats):")
    import statistics as sst
    for a in args.arms:
        per = []
        for rep in sorted({r["rep"] for r in rows}):
            v = [r[a] for r in rows if r["rep"] == rep and isinstance(r.get(a), float)
                 and np.isfinite(r.get(a, np.nan))]
            if v:
                per.append(sst.mean(v))
        if per:
            sd = f" +-{sst.stdev(per):.4f}" if len(per) > 1 else ""
            print(f"    {a:10} {sst.mean(per):.4f}{sd}   (n_reps {len(per)})")
    print(f"\n[chem] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
