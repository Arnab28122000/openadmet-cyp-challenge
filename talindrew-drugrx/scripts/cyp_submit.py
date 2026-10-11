"""Assemble a regression submission: place the predictions, check them, write the file.

Two steps, kept separate so each can be judged on its own:

1. **Placement** (`drugrx/cyp/placement.py`) — the affine map onto the blinded population's
   location and scale. Monotone, so ρ and τ must come back unchanged; that is the end-to-end
   check this programme has used three times.
2. **Schema validation** — exactly 750 rows, the four columns spelled as the organisers spell
   them, every value a finite float. A failed upload costs a 12-hour slot.

Before writing anything it runs the **nested placement check**: on the V2 validation fold, place
using `r` estimated on an inner split and that fold's own moments, and report whether ST-RAE
improves. That tests the *mechanism* (does affine placement with an estimated `r` help?) without
pretending validation can tell us the blinded set's moments — it cannot, which is why those come
from the board probe and the cross-check described in `placement.py`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, macro_st_rae, st_rae
from drugrx.cyp.placement import (BLIND_MOMENTS, ST_RAE_DISCOUNT, describe, place)
from drugrx.cyp.screen import allr_floor, screen_labels

ROOT = Path(__file__).resolve().parents[1]


def v2_truth(frame, va, labs, floor):
    out = {}
    for c in PRIMARY:
        y = frame[c].to_numpy(float)[va]
        has = np.isfinite(y)
        inact = labs[c]["kind"].to_numpy()[va] == "inactive"
        lo = frame[f"{c}_conf_low"].to_numpy(float)[va]
        hi = frame[f"{c}_conf_high"].to_numpy(float)[va]
        out[c] = (np.where(has, y, np.where(inact, floor["y"], np.nan)),
                  np.where(has, lo, np.where(inact, floor["lo"], np.nan)),
                  np.where(has, hi, np.where(inact, floor["hi"], np.nan)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with submission.csv + nets/*.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--discount", type=float, default=ST_RAE_DISCOUNT)
    ap.add_argument("--no-place", action="store_true", help="write the raw predictions")
    # Explicit per-isoform (centre, sd) targets, as JSON. Preferred over the r-rule: the
    # ST-RAE optimum is not the squared-error one, and the spread it wants depends on a
    # correlation we can only bound. scripts/cyp_place_sim.py and the minimax scan over
    # (inactive fraction, r) produce these.
    ap.add_argument("--targets", default=None,
                    help='JSON like {"CYP1A2": {"centre": 4.66, "sd": 0.95}, ...}')
    args = ap.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "submission_placed.csv"

    data = load(workers=8)
    frame, test = data.frame.reset_index(drop=True), data.test.reset_index(drop=True)
    labs, floor = screen_labels(frame), allr_floor(frame)

    raw = pd.read_csv(run / "submission.csv")
    if len(raw) != len(test):
        raise SystemExit(f"{run}: {len(raw)} rows, expected {len(test)}")

    # r per isoform, measured on the V2 validation fold of the same run's nets.
    nets = sorted((run / "nets").glob("*.npz"))
    if not nets:
        raise SystemExit(f"no net predictions under {run/'nets'}")
    zs = [np.load(p) for p in nets]
    va = zs[0]["va_index"]
    if any(not np.array_equal(z["va_index"], va) for z in zs):
        raise SystemExit("net seeds disagree on the validation split")
    pv = np.mean([z["val"] for z in zs], axis=0)
    truth = v2_truth(frame, va, labs, floor)
    r, rho = {}, {}
    for k, c in enumerate(PRIMARY):
        y = truth[c][0]
        m = np.isfinite(y)
        r[c] = float(np.corrcoef(pv[m, k], y[m])[0, 1])
        rho[c] = float(spearmanr(pv[m, k], y[m]).statistic)

    # ---- nested placement check on V2: does the mechanism help when r is estimated? ----
    rng = np.random.default_rng(0)
    half = rng.permutation(len(va))
    a, b = half[: len(half) // 2], half[len(half) // 2:]
    before, after = {}, {}
    for k, c in enumerate(PRIMARY):
        y, lo, hi = truth[c]
        ma, mb = np.isfinite(y[a]), np.isfinite(y[b])
        r_in = float(np.corrcoef(pv[a][ma, k], y[a][ma])[0, 1])
        mom = (float(np.mean(y[a][ma])), float(np.std(y[a][ma])))
        pb = pv[b][:, k]
        placed = place(pb, r_in, *mom, discount=args.discount)
        before[c] = st_rae(pb[mb], y[b][mb], lo[b][mb], hi[b][mb])
        after[c] = st_rae(placed[mb], y[b][mb], lo[b][mb], hi[b][mb])
    print("[submit] nested placement check on V2 (inner-fitted r and moments):")
    for c in PRIMARY:
        print(f"    {c[:6]}: {before[c]:.4f} -> {after[c]:.4f}  ({after[c]-before[c]:+.4f})")
    print(f"    macro : {np.mean(list(before.values())):.4f} -> "
          f"{np.mean(list(after.values())):.4f} "
          f"({np.mean(list(after.values()))-np.mean(list(before.values())):+.4f})")

    # ---- the submission ----
    preds = {c: raw[c].to_numpy(float) for c in PRIMARY}
    if args.no_place:
        final = preds
        rows = []
    elif args.targets:
        tg = json.loads(Path(args.targets).read_text())
        moments = {c: (tg[c[:6]]["centre"], tg[c[:6]]["sd"]) for c in PRIMARY}
        # discount/r are folded into the target sd already, so place() is asked for exactly it.
        final = {c: place(preds[c], 1.0, *moments[c], discount=1.0) for c in PRIMARY}
        rows = describe(preds, final, {c: 1.0 for c in PRIMARY}, moments)
    else:
        final = {c: place(preds[c], r[c], *BLIND_MOMENTS[c[:6]], discount=args.discount)
                 for c in PRIMARY}
        rows = describe(preds, final, r, {c: BLIND_MOMENTS[c[:6]] for c in PRIMARY})

    # Reporting and the monotonicity check run for EVERY placed branch. They lived inside the
    # `else` above, so adding `--targets` silently skipped them -- and the thing skipped was
    # the one free end-to-end check this programme has (a rescale is monotone, so rho and tau
    # must come back bit-identical from the board).
    if rows:
        print("\n[submit] placement:")
        for d in rows:
            print(f"    {d['endpoint'][:6]}: mean {d['mean_before']:.2f} -> "
                  f"{d['mean_after']:.2f} (target {d['blind_mean']}) | sd {d['sd_before']:.2f} -> "
                  f"{d['sd_after']:.2f} (target {d['blind_sd']}) | floored {d['n_floored']}")
        for c in PRIMARY:
            s = spearmanr(preds[c], final[c]).statistic
            if not np.isclose(s, 1.0):
                raise SystemExit(f"{c}: placement changed the ranking (rho={s})")
        print("    monotonicity check: rho(raw, placed) == 1.0 on all four")

    sub = pd.DataFrame({"SMILES": test["SMILES"], "Molecule_Name": test["Molecule_Name"]})
    for c in PRIMARY:
        sub[c] = final[c]
    bad = {c: int((~np.isfinite(sub[c])).sum()) for c in PRIMARY}
    if any(bad.values()):
        raise SystemExit(f"non-finite predictions: {bad}")
    if len(sub) != 750:
        raise SystemExit(f"{len(sub)} rows, the organisers require exactly 750")
    sub.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps(
        {"run": str(run), "metric_id": METRIC_ID, "discount": args.discount,
         "placed": not args.no_place, "r_v2": r, "rho_v2": rho, "placement": rows,
         "nested_check": {"before": before, "after": after},
         "pred_summary": {c: {"mean": float(sub[c].mean()), "sd": float(sub[c].std()),
                              "min": float(sub[c].min()), "max": float(sub[c].max())}
                          for c in PRIMARY}}, indent=1, default=float))
    print(f"\n[submit] wrote {out} ({len(sub)} rows)")
    for c in PRIMARY:
        print(f"    {c[:6]}: mean {sub[c].mean():.2f} sd {sub[c].std():.2f} "
              f"range {sub[c].min():.2f}-{sub[c].max():.2f} | V2 rho {rho[c]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
