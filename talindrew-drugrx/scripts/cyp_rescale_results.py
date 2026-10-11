r"""Re-weight finished results to the official metric by arithmetic, not retraining.

ST-RAE for endpoint `e` is `N_e / D_e`, and `D_e` depends only on the labels and
intervals of the evaluation subset -- never on the model. On a given split every
arm therefore shares one `D_e`, so correcting the denominator rescales each
endpoint's score by a single constant `1/c_e` and does nothing else. Two
consequences decide what actually needs a GPU:

* **Within-endpoint conclusions survive exactly.** "A beats B on CYP2D6" has both
  numerators divided by the same constant, so the sign cannot move -- only the
  magnitude.
* **Macro conclusions change by re-weighting, and a sign can flip only where the
  per-endpoint effects had mixed signs.** An effect positive on all four isoforms
  survives in sign and needs only a new magnitude. A macro *null* produced by
  cancellation across endpoints is the dangerous case: re-weighting at 0.54-0.67
  can un-cancel it.

So any experiment that stored **per-endpoint** scores is recomputable here in
seconds. This script does that for `cyp_decode.py` and `cyp_incontext.py`, which
both stored one row per (repeat, endpoint).

**What this cannot fix, and must be retrained instead:** `cyp_net_one.py` scores
with `macro_st_rae` once per epoch and early-stops on it, storing only the macro
curve. The argmin over epochs can differ when the weights differ, so the depth and
long-run results are genuinely downstream of the correction and are re-run rather
than rescaled.

`c_e` is recovered from the data itself (official denominator / old denominator per
endpoint per split) rather than assumed, because it is split-dependent.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load                  # noqa: E402
from drugrx.cyp.metric import soft_threshold_error         # noqa: E402
from drugrx.data.splits import scaffold_split              # noqa: E402


def correction_factors(frame, repeats, frac_train, frac_valid, merge_rest):
    """c_e per (repeat, endpoint): official_denominator / old_denominator.

    old  = sum|y - mean(y)|
    new  = sum( soft_threshold_error(mean(y), lo, hi) )
    official_score = old_score / c_e    (so multiply stored scores by 1/c_e)
    """
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)
    rows = []
    for rep in range(repeats):
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=frac_train,
                                      frac_valid=frac_valid, seed=rep)
        ev = np.concatenate([va, rest]) if (merge_rest and len(rest)) else (
            rest if merge_rest is None else va)
        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            idx = ev[np.isfinite(y[ev])]
            if len(idx) < 10:
                continue
            yy, l, h = y[idx], lo[idx, i], hi[idx, i]
            old = np.abs(yy - yy.mean()).sum()
            new = soft_threshold_error(np.full(yy.shape, yy.mean()), l, h).sum()
            rows.append({"rep": rep, "endpoint": c, "c_e": float(new / old) if old else np.nan,
                         "n": int(len(idx))})
    return pd.DataFrame(rows)


def rescale(df, cf, arms, label):
    m = df.merge(cf[["rep", "endpoint", "c_e"]], on=["rep", "endpoint"], how="left")
    missing = m["c_e"].isna().sum()
    if missing:
        print(f"[rescale] {label}: {missing} rows without a correction factor -- "
              f"split geometry does not match; refusing to report these", flush=True)
        m = m.dropna(subset=["c_e"])
    print(f"\n[rescale] {label}: correction factor c_e per endpoint "
          f"(official = old / c_e)", flush=True)
    for c, g in cf.groupby("endpoint"):
        print(f"    {c[:26]:26s} c_e {g['c_e'].mean():.3f}  "
              f"-> official is {1/g['c_e'].mean():.2f}x the old number")

    for a in arms:
        if a not in m:
            continue
        m[a + "__official"] = m[a] / m["c_e"]

    print(f"\n[rescale] {label}: macro, old vs official", flush=True)
    macro_old = {a: m.groupby("rep")[a].mean() for a in arms if a in m}
    macro_new = {a: m.groupby("rep")[a + "__official"].mean() for a in arms if a in m}
    for a in macro_old:
        print(f"    {a:14s} old {macro_old[a].mean():.4f}  official {macro_new[a].mean():.4f}")

    ref = arms[0]
    print(f"\n[rescale] {label}: paired vs {ref}, OLD then OFFICIAL "
          f"(positive = better than {ref})", flush=True)
    flips = []
    for a in arms[1:]:
        if a not in macro_new:
            continue
        out = {}
        for tag, mm in (("old", macro_old), ("official", macro_new)):
            d = mm[ref] - mm[a]
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            t = d.mean() / se if se and se > 0 else float("nan")
            w = int((d > 0).sum())
            v = ("BETTER" if abs(t) > 2.5 and w == len(d) else
                 "WORSE (consistent)" if abs(t) > 2.5 and w == 0 else "inside noise")
            out[tag] = (d.mean(), se, w, len(d), t, v)
        o, n = out["old"], out["official"]
        flag = "  <-- VERDICT CHANGED" if o[5] != n[5] else ""
        print(f"    {a:14s} old {o[0]:+.4f}+/-{o[1]:.4f} {o[5]:18s} | "
              f"official {n[0]:+.4f}+/-{n[1]:.4f} {n[5]:18s}{flag}")
        if flag:
            flips.append(a)

    # Where could a sign flip even come from? Only mixed per-endpoint signs.
    print(f"\n[rescale] {label}: per-endpoint sign agreement vs {ref} "
          f"(a macro null from cancellation is the fragile case)", flush=True)
    for a in arms[1:]:
        if a not in m:
            continue
        signs = []
        for c, g in m.groupby("endpoint"):
            d = (g[ref] - g[a]).mean()
            signs.append("+" if d > 0 else "-")
        mixed = len(set(signs)) > 1
        print(f"    {a:14s} {' '.join(signs)}  "
              f"{'MIXED -- reweighting can flip this' if mixed else 'uniform -- sign is safe'}")
    return flips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", default="/content/out/cyp_decode/decode.json")
    ap.add_argument("--incontext", default="/content/out/cyp_incontext/incontext.json")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    data = load(workers=args.workers)
    frame = data.frame

    if Path(args.decode).exists():
        df = pd.DataFrame(json.load(open(args.decode)))
        arms = ["offset", "raw"] + [c for c in df.columns if c.startswith("map")]
        arms = ["offset"] + [a for a in arms if a != "offset"]
        # decode used 0.70/0.15 with the third split as the report set
        cf = correction_factors(frame, df["rep"].nunique(), 0.70, 0.15, merge_rest=None)
        rescale(df, cf, arms, "cyp_decode (xgb, report split)")

    if Path(args.incontext).exists():
        df = pd.DataFrame(json.load(open(args.incontext)))
        arms = ["xgb"] + [c for c in ("icl_pca", "icl_top", "icl_desc",
                                      "dpt_pca", "dpt_top", "dpt_desc") if c in df]
        # incontext used 0.85/0.10 with valid+rest as the test set
        cf = correction_factors(frame, df["rep"].nunique(), 0.85, 0.10, merge_rest=True)
        rescale(df, cf, arms, "cyp_incontext (test split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
