"""The ceiling on ANY per-compound decode: price the lever before building the method.

`cyp_decide_vs_affine.py` showed the balance point losing to the affine map, and diagnosed the
reason as calibration rather than theory. The named next step was a *mixture* posterior — a
per-compound inactivity probability through `decide.mixture_draws` — on the structural argument
that the ST-RAE optimum is **non-monotone** (a likely-inactive compound belongs at 3.52, the top
of the free window, which is ABOVE a moderately-active one at 3.0) and an affine map cannot
express that.

Before building it, bound it. ST-RAE is separable per compound, so for any family of decoders
that are **functions of our prediction `p`**, the best possible member is computable directly:
bin `p`, and give each bin the output minimising that bin's own soft-threshold error — which is
exactly `decide.balance_point` on the bin's intervals. No model, no posterior, no classifier.

That is the **ceiling over every monotone and non-monotone map of `p`**, affine included. If it
sits on top of the affine map, then no decode of `p` can help and the mixture build is dead
before it starts. Only a decode using information *beyond* `p` (a genuine inactivity probability
that `p` does not already carry) could then do better, and the gap measured here is exactly the
budget available to it.

Two numbers, because one of them is a fantasy:

* **oracle** — bin outputs chosen on the very compounds being scored. An upper bound, and
  unattainable: with ~250 compounds in 12 bins it fits ~12 labels per output.
* **cross-fitted** — bin outputs chosen on one half, scored on the other, both ways. What a real
  decoder could actually deliver at this bin resolution.

A large oracle gap with a cross-fitted gap near zero means the ceiling is all overfitting, and
the honest conclusion is the same as a low ceiling: do not build it.

    python scripts/cyp_decode_ceiling.py --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_decide_vs_affine import PSEUDO_W, affine, best_affine, gbm, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.decide import balance_point  # noqa: E402
from drugrx.cyp.metric import st_rae  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402

NBINS = 12
#: the oracle gain grows with resolution while the cross-fitted gain decays, so a single
#: resolution proves nothing; sweep it and ask whether ANY resolution attains a real gain.
BIN_SWEEP = (3, 5, 8, 12, 20, 32)


def binned_map(p_fit, lo_fit, hi_fit, p_apply, nbins=NBINS):
    """Per-bin balance point, fitted on (p_fit, lo_fit, hi_fit), applied to p_apply.

    Bin edges come from the FIT predictions' quantiles. This is the unconstrained optimum over
    functions of p at this resolution -- it is free to be non-monotone.
    """
    edges = np.quantile(p_fit, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    which_fit = np.clip(np.digitize(p_fit, edges[1:-1]), 0, nbins - 1)
    which_app = np.clip(np.digitize(p_apply, edges[1:-1]), 0, nbins - 1)
    out = np.empty(len(p_apply), float)
    for b in range(nbins):
        m = which_fit == b
        tgt = which_app == b
        if not tgt.any():
            continue
        out[tgt] = balance_point(lo_fit[m], hi_fit[m]) if m.sum() >= 3 else np.median(p_apply[tgt])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/decode_ceiling.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    labs = screen_labels(frame)
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    print(f"[ceiling] features {X.shape}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        rng = np.random.default_rng(seed)
        print(f"\n[ceiling] seed {seed}", flush=True)
        for c in PRIMARY:
            lab = labs[c]
            y = frame[c].to_numpy(float)
            lo = frame[f"{c}_conf_low"].to_numpy(float)
            hi = frame[f"{c}_conf_high"].to_numpy(float)
            kind = lab["kind"].to_numpy()
            use = np.zeros(len(frame), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, lab["y"].to_numpy(float))
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            okv = np.isfinite(y[va]) & np.isfinite(lo[va]) & np.isfinite(hi[va])
            if len(idx) < 50 or okv.sum() < 60:
                continue
            t0 = time.time()
            p = gbm(seed).fit(X[idx], yy[idx], sample_weight=w[idx]).predict(X[va])[okv]
            yv, lv, hv = y[va][okv], lo[va][okv], hi[va][okv]

            co, so = best_affine(p, yv, lv, hv)
            p_aff = affine(p, co, so)

            perm = rng.permutation(len(p))
            h1, h2 = perm[: len(p) // 2], perm[len(p) // 2:]
            rec = dict(seed=seed, iso=c, n=int(okv.sum()),
                       strae_affine_oracle=float(st_rae(p_aff, yv, lv, hv)),
                       secs=round(time.time() - t0, 1))
            for nb in BIN_SWEEP:
                p_orc = binned_map(p, lv, hv, p, nbins=nb)
                p_cf = np.empty(len(p), float)
                for fit_i, app_i in ((h1, h2), (h2, h1)):
                    p_cf[app_i] = binned_map(p[fit_i], lv[fit_i], hv[fit_i], p[app_i], nbins=nb)
                rec[f"strae_bin_oracle_{nb}"] = float(st_rae(p_orc, yv, lv, hv))
                rec[f"strae_bin_crossfit_{nb}"] = float(st_rae(p_cf, yv, lv, hv))
            rec["strae_bin_oracle"] = rec[f"strae_bin_oracle_{NBINS}"]
            rec["strae_bin_crossfit"] = rec[f"strae_bin_crossfit_{NBINS}"]
            rows.append(rec)
            print(f"   {short(c)}: n={rec['n']:4d}  affine_oracle {rec['strae_affine_oracle']:.4f}"
                  f"  bin_ORACLE {rec['strae_bin_oracle']:.4f}"
                  f"  bin_crossfit {rec['strae_bin_crossfit']:.4f}  [{time.time()-t0:.0f}s]",
                  flush=True)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    keys = ("strae_affine_oracle", "strae_bin_oracle", "strae_bin_crossfit")
    print("\n=== ST-RAE: affine oracle vs the ceiling over ALL maps of p (lower is better) ===")
    print(f"{'iso':8} {'affine_oracle':>14} {'bin ORACLE':>12} {'bin crossfit':>13} "
          f"{'oracle gain':>12} {'real gain':>11}")
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if not rs:
            continue
        a, o, x = (float(np.mean([r[k] for r in rs])) for k in keys)
        print(f"{short(c):8} {a:14.4f} {o:12.4f} {x:13.4f} {a-o:+12.4f} {a-x:+11.4f}")
    a, o, x = (float(np.mean([np.mean([r[k] for r in rows if r["iso"] == c])
                              for c in PRIMARY if any(r["iso"] == c for r in rows)]))
               for k in keys)
    print(f"{'MACRO':8} {a:14.4f} {o:12.4f} {x:13.4f} {a-o:+12.4f} {a-x:+11.4f}")
    print("\n=== resolution sweep: does ANY bin count attain a real gain? ===")
    print(f"  {'bins':>5} {'bin ORACLE':>12} {'bin crossfit':>13} {'oracle gain':>12} "
          f"{'REAL gain':>11}")
    for nb in BIN_SWEEP:
        mo = float(np.mean([np.mean([r[f"strae_bin_oracle_{nb}"] for r in rows if r["iso"] == c])
                            for c in PRIMARY if any(r["iso"] == c for r in rows)]))
        mx = float(np.mean([np.mean([r[f"strae_bin_crossfit_{nb}"] for r in rows if r["iso"] == c])
                            for c in PRIMARY if any(r["iso"] == c for r in rows)]))
        flag = "  <- positive!" if a - mx > 0 else ""
        print(f"  {nb:5d} {mo:12.4f} {mx:13.4f} {a-mo:+12.4f} {a-mx:+11.4f}{flag}")
    print("  The oracle gain rises with resolution and the real gain falls: more freedom to be")
    print("  non-monotone is bought entirely with variance. If no row is positive, the affine")
    print("  family is not the binding constraint and no decode of p is worth building.")

    print("\n  'oracle gain' is the ceiling over every map of p, fitted on the scored compounds")
    print("  -- unattainable. 'real gain' is what survives cross-fitting, and it is the budget")
    print("  any decode of p could actually claim. A mixture posterior is a function of p and q,")
    print("  so it can only exceed the ceiling by however much q carries that p does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
