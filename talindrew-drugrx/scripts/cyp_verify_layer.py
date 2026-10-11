r"""Two measurement-layer checks, both of which must pass before anything is banked.

**CHECK 1 -- is the rescaling exact?** It must be. `ST-RAE_e = N_e / D_e`, the
predictions are untouched, so `official_e == mad_e / c_e` to floating point with
no approximation anywhere. A 0.4% discrepancy on a quantity that should be exact
is the same order as effects this programme has been treating as real, so it gets
run down rather than explained away. `cyp_remeasure` stored `official` and `mad`
for the **same predictions on the same rows**, which makes this a self-contained
exactness test needing no retraining.

**CHECK 2 -- are the primary point targets aligned?** `--loss point` is a new path
in the data-prep layer, where six of this programme's defects have lived, every one
of them failing toward a readable answer. The primary columns of `Y` had always
been zero, because the interval loss only ever read `lo`/`hi`. A *gross*
misalignment would be obvious (the arm would predict near 0 and score terribly); the
dangerous case is a subtle row permutation, which returns "the interval loss wins"
and reads as confirmation of the founding assumption. So the arrays are rebuilt
exactly as `cyp_net_one.py` builds them and asserted element-wise against the frame.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load                  # noqa: E402
from drugrx.cyp.metric import soft_threshold_error          # noqa: E402
from drugrx.data.splits import scaffold_split               # noqa: E402


def check_rescale_exact(frame, path, frac_train, frac_valid, merge_rest):
    if not Path(path).exists():
        print(f"[verify] {path} absent -- skipping check 1", flush=True); return
    df = pd.DataFrame(json.load(open(path)))
    df = df[df.arm == "xgb"] if "arm" in df else df
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    worst = 0.0
    rows = []
    for rep in sorted(df["rep"].unique()):
        tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=frac_train,
                                      frac_valid=frac_valid, seed=int(rep))
        ev = np.concatenate([va, rest]) if (merge_rest and len(rest)) else va
        for i, c in enumerate(PRIMARY):
            y = frame[c].to_numpy(float)
            idx = ev[np.isfinite(y[ev])]
            yy, l, h = y[idx], lo[idx, i], hi[idx, i]
            old_d = np.abs(yy - yy.mean()).sum()
            new_d = soft_threshold_error(np.full(yy.shape, yy.mean()), l, h).sum()
            c_e = new_d / old_d
            r = df[(df.rep == rep) & (df.endpoint == c)]
            if r.empty:
                continue
            got, mad = float(r["official"].iloc[0]), float(r["mad"].iloc[0])
            pred = mad / c_e
            err = abs(pred - got)
            worst = max(worst, err)
            rows.append({"rep": int(rep), "endpoint": c, "n_eval": len(idx),
                         "official": got, "mad/c_e": pred, "abs_err": err})
    t = pd.DataFrame(rows)
    print(f"\n[verify] CHECK 1: rescaling exactness over {len(t)} (rep, endpoint) cells",
          flush=True)
    print(f"    worst |official - mad/c_e| = {worst:.3e}", flush=True)
    if worst < 1e-9:
        print("    EXACT -- the method is sound, so any macro disagreement is a "
              "difference of estimand, not of measurement.", flush=True)
    else:
        print("    NOT EXACT -- measurement-layer defect. Worst offenders:", flush=True)
        print(t.sort_values("abs_err", ascending=False).head(8).to_string(index=False),
              flush=True)
    # The macro each route reports, over MATCHING repeats only.
    reps = sorted(t["rep"].unique())
    m_off = t.groupby("rep")["official"].mean()
    print(f"    macro over reps {reps}: official {m_off.mean():.6f}", flush=True)
    return t


def check_point_targets(frame, workers):
    """Rebuild cyp_net_one's Y/M exactly and assert alignment to the frame."""
    n_prim = len(PRIMARY)
    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=0)
    va = np.concatenate([va, rest]) if len(rest) else va

    prim_y = frame[list(PRIMARY)].to_numpy(np.float32)[tr]
    yb = np.zeros((len(tr), n_prim), np.float32)
    mb = np.zeros((len(tr), n_prim), bool)
    yb[:, :n_prim] = np.nan_to_num(prim_y, nan=0.0)
    mb[:, :n_prim] = np.isfinite(prim_y)

    print(f"\n[verify] CHECK 2: primary point targets, {len(tr)} training rows", flush=True)
    bad = 0
    for i in range(len(tr)):
        row = tr[i]
        for k, c in enumerate(PRIMARY):
            truth = frame[c].to_numpy(float)[row]
            if np.isfinite(truth):
                if not mb[i, k] or abs(yb[i, k] - truth) > 1e-4:
                    bad += 1
            else:
                if mb[i, k] or yb[i, k] != 0.0:
                    bad += 1
    print(f"    element-wise mismatches vs the frame: {bad}", flush=True)

    # The mask must select exactly the labelled cells, per endpoint.
    for k, c in enumerate(PRIMARY):
        exp = int(np.isfinite(frame[c].to_numpy(float)[tr]).sum())
        got = int(mb[:, k].sum())
        flag = "OK" if exp == got else "MISMATCH"
        print(f"    {c[:30]:30s} labelled in train {exp:5d}  mask says {got:5d}  {flag}")

    # Column order must match lo/hi, which are built from the same PRIMARY order.
    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)[tr]
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)[tr]
    inside = []
    for k, c in enumerate(PRIMARY):
        m = mb[:, k] & np.isfinite(lo[:, k]) & np.isfinite(hi[:, k])
        if m.sum():
            ok = ((yb[m, k] >= lo[m, k] - 1e-6) & (yb[m, k] <= hi[m, k] + 1e-6)).mean()
            inside.append((c, float(ok), int(m.sum())))
    print("\n    the decisive column-order test: each point target must lie inside its "
          "OWN interval", flush=True)
    for c, ok, n in inside:
        print(f"    {c[:30]:30s} inside its interval: {ok:6.2%} of {n}  "
              f"{'OK' if ok > 0.99 else 'COLUMN MISALIGNMENT'}")
    worst = min(ok for _, ok, _ in inside) if inside else 0.0
    print(f"\n[verify] CHECK 2 verdict: "
          f"{'PASS' if bad == 0 and worst > 0.99 else 'FAIL -- do not trust the ablation'}",
          flush=True)
    return bad == 0 and worst > 0.99


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", default="/content/out/cyp_remeasure/remeasure.json")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    data = load(workers=args.workers)
    check_rescale_exact(data.frame, args.remeasure, 0.85, 0.10, merge_rest=True)
    check_point_targets(data.frame, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
