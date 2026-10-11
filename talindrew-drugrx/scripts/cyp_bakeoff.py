"""Which model actually wins on ST-RAE? Baselines first, fail fast, then tune.

This programme's accumulated evidence says a tuned gradient-boosting model on
count-ECFP4 + RDKit2D + Avalon + ErG is at least as good as a neural net on
molecular property tasks (H1 18/21, H5 14/21, H6 showed the two indistinguishable
at p=0.180). **So the neural net does not get to be the default here either.**

Everything is scored with the challenge's own metric on the *same* scaffold
split, so the comparison is between models rather than between evaluation
choices. Arms, cheapest first:

* **median** — predict each endpoint's training median. The floor. If a model
  cannot beat this it has learned nothing, and on a metric where predictions
  inside a credible interval score zero, a constant is less silly than it sounds:
  CYP3A4 has 40% of labels below pIC50 4 with wide intervals.
* **interval-median** — predict the midpoint of each endpoint's typical interval.
  Exploits the metric directly without any chemistry.
* **xgb** — per-endpoint XGBoost on the GPU. No auxiliary heads, no multi-task.
* **xgb-aux** — XGBoost with the auxiliary columns as *features* rather than as
  heads, which is how a tree model can use them at all.
* **nn / nn-noaux** — the multi-task net with and without auxiliary heads, which
  is the H7a question asked again in a setting where the primary task is far
  thinner (6,525 primary labels against 38,137 auxiliary).

Reported with a permutation-style uncertainty: each arm is bootstrapped over the
validation molecules so a difference of 0.01 is not read as a result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.metric import macro_st_rae, st_rae  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402
from drugrx.features import featurise, standardise_descriptors  # noqa: E402


def bootstrap_strae(pred, y, lo, hi, draws=400, seed=0):
    rng = np.random.default_rng(seed)
    keep = np.isfinite(pred) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    p, yy, l, h = pred[keep], y[keep], lo[keep], hi[keep]
    if len(p) < 10:
        return float("nan"), float("nan")
    vals = []
    for _ in range(draws):
        i = rng.integers(0, len(p), len(p))
        vals.append(st_rae(p[i], yy[i], l[i], h[i]))
    return float(np.mean(vals)), float(np.std(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/content/out/cyp/bakeoff.json")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame
    x = featurise(frame["smiles"].tolist(), workers=args.workers,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)

    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=args.seed)
    va = np.concatenate([va, rest]) if len(rest) else va
    _, stats = standardise_descriptors(x[tr])
    x, _ = standardise_descriptors(x, stats=stats)
    print(f"[bake] {len(tr)} train / {len(va)} valid molecules", flush=True)

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(dtype=float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(dtype=float)
    truth = {c: (frame[c].to_numpy()[va], lo[va, i], hi[va, i]) for i, c in enumerate(PRIMARY)}

    results = {}

    def record(name, preds, seconds):
        macro, per = macro_st_rae(preds, truth)
        boot = {}
        for i, c in enumerate(PRIMARY):
            m, s = bootstrap_strae(np.asarray(preds[c]), *truth[c])
            boot[c] = {"mean": m, "sd": s}
        results[name] = {"macro": macro, "per_endpoint": per, "bootstrap": boot,
                         "seconds": round(seconds, 1)}
        sds = np.mean([b["sd"] for b in boot.values()])
        print(f"  {name:12s} MA-ST-RAE {macro:.4f}  (per-endpoint bootstrap sd ~{sds:.4f})"
              f"  {seconds:.1f}s", flush=True)

    # ── floors ───────────────────────────────────────────────────────
    t0 = time.time()
    record("median", {c: np.full(len(va), np.nanmedian(frame[c].to_numpy()[tr]))
                      for c in PRIMARY}, time.time() - t0)

    t0 = time.time()
    mid = {}
    for i, c in enumerate(PRIMARY):
        m = np.nanmedian((lo[tr, i] + hi[tr, i]) / 2.0)
        mid[c] = np.full(len(va), m)
    record("interval-mid", mid, time.time() - t0)

    # ── gradient boosting, per endpoint ──────────────────────────────
    import xgboost as xgb

    def run_xgb(extra_features: bool, tag: str):
        t = time.time()
        feats = x
        if extra_features:
            aux = frame[data.auxiliary].to_numpy(dtype=np.float32)
            aux = np.nan_to_num(aux, nan=0.0)
            flags = (~np.isnan(frame[data.auxiliary].to_numpy(dtype=np.float32))).astype(np.float32)
            feats = np.concatenate([x, aux, flags], axis=1)
        preds = {}
        for c in PRIMARY:
            y = frame[c].to_numpy(dtype=float)
            itr = tr[np.isfinite(y[tr])]
            if len(itr) < 50:
                preds[c] = np.full(len(va), np.nanmedian(y[tr]))
                continue
            m = xgb.XGBRegressor(n_estimators=1200, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0,
                                 device="cuda", tree_method="hist", verbosity=0,
                                 eval_metric="mae", early_stopping_rounds=80)
            ival = va[np.isfinite(y[va])]
            m.fit(feats[itr], y[itr], eval_set=[(feats[ival], y[ival])], verbose=False)
            preds[c] = m.predict(feats[va])
        record(tag, preds, time.time() - t)

    run_xgb(False, "xgb")
    # **"xgb-aux" is deleted, not kept as an upper bound.** Feeding the auxiliary
    # columns as *features* scored 0.0617 against 0.4432 — a 7x jump that is
    # leakage twice over: TDI-condition pIC50 correlates r=0.90-0.99 with the
    # direct-inhibition target (median |diff| 0.06-0.33), so it is the target
    # measured under a slightly different condition; and the blinded test set
    # contains only Molecule_Name and SMILES, so those features do not exist at
    # inference. An arm that cannot be submitted is not a baseline, and leaving
    # it in a results table invites someone to quote it.

    best = min(results.items(), key=lambda kv: kv[1]["macro"])
    print(f"\n[bake] best so far: {best[0]} at {best[1]['macro']:.4f}", flush=True)
    print("[bake] a difference smaller than ~2x the bootstrap sd is not a result.", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
