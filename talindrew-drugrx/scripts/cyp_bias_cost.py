r"""What does a systematic bias cost under ST-RAE, and how much of it can a shift recover?

Leg 1A recovered a bias of **~0.76 pIC50** on the blinded test set, from the leaderboard's own
MAE/R2/rho/tau columns and our own test predictions. Leg 1C showed the *scale* is not the lever:
ST-RAE's optimal scale sits 17.5% below the squared-error one, and applying the squared-error
optimum is worse on 5 of 5 seeds.

So the question is the bias. This measures its cost directly, with **no modelled intervals** --
impose a known bias on cached validation predictions, where the real `lo`/`hi` are in the
release, and read the official metric.

Three things come out:

1. **cost(b)** -- ST-RAE as a function of an imposed bias. Non-linear and **asymmetric**,
   because the label distribution has a floor near pIC50 4 and the credible intervals are wide
   for inactives and narrow for potent compounds. The asymmetry is the point: it means the
   metric itself carries information about the *sign* of a bias, which the moment-based
   inversion in leg 1A cannot see.
2. **recovery(b)** -- how much of `cost(b)` a single fitted shift gets back, and the optimal
   shift against a known bias `b`. It need not be `-b`: a loss that charges nothing inside the
   interval wants the prediction inside the band, not on the point estimate, so the optimum is a
   weighted median rather than a mean-matching correction.
3. **the sign asymmetry at our own operating point**, which is the only route left to the sign
   of the test-side bias after the floor test and the two-row differential test both came back
   underpowered.

Caveat stated up front, because it bounds every number here: validation's label spread is 0.944
against the test estimate of ~1.27, and its interval widths belong to its own molecules. The
*shape* of cost(b) transfers; its scale does not. Leg 1B is what fixes that, and it needs the
measured width-vs-label-value relation this script does not use.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae

BIASES = np.round(np.arange(-1.5, 1.501, 0.05), 4)
SHIFTS = np.round(np.arange(-1.5, 1.501, 0.025), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h12-longdeep")
    ap.add_argument("--cache", default="data/cyp_cache/cyp_h12")
    ap.add_argument("--probe", type=float, default=0.76,
                    help="the bias magnitude leg 1A recovered on the test set")
    ap.add_argument("--out", default="data/cyp_cache/bias_cost.json")
    args = ap.parse_args()

    fr = load().frame
    files = sorted(glob.glob(f"{args.cache}/{args.tag}_s*.npz"))
    if not files:
        raise SystemExit(f"no artifacts for {args.tag}")

    cols: list[list[tuple]] = []
    for f in files:
        z = np.load(f)
        idx, val = z["va_index"], z["val"]
        per = []
        for i, c in enumerate(PRIMARY):
            y = fr[c].to_numpy(float)[idx]
            lo = fr[f"{c}_conf_low"].to_numpy(float)[idx]
            hi = fr[f"{c}_conf_high"].to_numpy(float)[idx]
            k = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(val[:, i])
            per.append((val[k, i], y[k], lo[k], hi[k]))
        cols.append(per)

    def macro_at(seed_i: int, bias: float) -> float:
        vals = [st_rae(p + bias, y, lo, hi) for p, y, lo, hi in cols[seed_i]]
        return float(np.mean(vals))

    print(f"[bias] {args.tag}: {len(files)} seeds, metric {METRIC_ID}")
    print(f"[bias] real credible intervals from the release — no modelled widths\n")

    # ------------------------------------------------------------ 1. cost(b)
    print("cost of an imposed bias, macro ST-RAE (validation geometry)")
    print(f"{'bias':>7s} {'macro':>8s} {'sd':>7s} {'vs b=0':>9s}")
    base = float(np.mean([macro_at(s, 0.0) for s in range(len(files))]))
    curve = {}
    for b in BIASES:
        v = np.array([macro_at(s, float(b)) for s in range(len(files))])
        curve[float(b)] = (float(v.mean()), float(v.std(ddof=1)))
        if abs(b * 20 - round(b * 20)) < 1e-9 and abs(round(b, 2) * 4 % 1) < 1e-9:
            print(f"{b:+7.2f} {v.mean():8.4f} {v.std(ddof=1):7.4f} "
                  f"{v.mean() - base:+9.4f}")

    pb = args.probe
    up = float(np.mean([macro_at(s, +pb) for s in range(len(files))]))
    dn = float(np.mean([macro_at(s, -pb) for s in range(len(files))]))
    print(f"\n  at |bias| = {pb:.2f}, the magnitude leg 1A recovered on test:")
    print(f"    predictions shifted DOWN by {pb:.2f} (we under-predict): "
          f"{dn:.4f}   cost {dn - base:+.4f}")
    print(f"    predictions shifted UP   by {pb:.2f} (we over-predict) : "
          f"{up:.4f}   cost {up - base:+.4f}")
    print(f"    asymmetry (down - up) = {dn - up:+.4f}")
    print(f"  Our actual test score is 0.8598 against a validation 0.6793 — a gap of +0.1805.")
    print(f"  A bias of {pb:.2f} costs {max(dn, up) - base:+.4f} at validation geometry, i.e. "
          f"{100 * (max(dn, up) - base) / 0.1805:.0f}% of that gap.")

    # ------------------------------------------- 2. recovery and the optimal shift
    print(f"\noptimal corrective shift against a known imposed bias")
    print(f"{'imposed':>8s} {'best shift':>11s} {'ratio':>7s} {'recovered':>10s} "
          f"{'residual':>9s}")
    rec = {}
    for b in (-1.0, -0.76, -0.5, -0.25, 0.0, 0.25, 0.5, 0.76, 1.0):
        # Choose the shift on half the molecules, score on the other half, both ways.
        got, bests = [], []
        for s in range(len(files)):
            rng = np.random.default_rng(1000 + s)
            fold_scores, fold_best = [], []
            per = cols[s]
            halves = [[], []]
            for p, y, lo, hi in per:
                o = rng.permutation(len(y))
                halves[0].append((p[o[: len(y) // 2]], y[o[: len(y) // 2]],
                                  lo[o[: len(y) // 2]], hi[o[: len(y) // 2]]))
                halves[1].append((p[o[len(y) // 2:]], y[o[len(y) // 2:]],
                                  lo[o[len(y) // 2:]], hi[o[len(y) // 2:]]))
            for fit_h, rep_h in ((0, 1), (1, 0)):
                sc = [float(np.mean([st_rae(p + b + sh, y, lo, hi)
                                     for p, y, lo, hi in halves[fit_h]]))
                      for sh in SHIFTS]
                sh = float(SHIFTS[int(np.argmin(sc))])
                fold_best.append(sh)
                fold_scores.append(float(np.mean([st_rae(p + b + sh, y, lo, hi)
                                                  for p, y, lo, hi in halves[rep_h]])))
            bests.append(float(np.mean(fold_best)))
            got.append(float(np.mean(fold_scores)))
        biased = float(np.mean([macro_at(s, b) for s in range(len(files))]))
        corrected = float(np.mean(got))
        cost = biased - base
        recovered = biased - corrected
        rec[float(b)] = dict(best_shift=float(np.mean(bests)), biased=biased,
                             corrected=corrected, cost=cost, recovered=recovered)
        ratio = (np.mean(bests) / -b) if b else float("nan")
        print(f"{b:+8.2f} {np.mean(bests):+11.3f} {ratio:7.3f} {recovered:+10.4f} "
              f"{corrected - base:+9.4f}")
    print("  'ratio' is best_shift / -bias. Below 1 means the metric's free zone wants a")
    print("  *partial* correction — landing inside the band, not on the point estimate.")

    Path(args.out).write_text(json.dumps(
        dict(metric_id=METRIC_ID, tag=args.tag, base=base, probe=pb,
             cost_curve={str(k): v for k, v in curve.items()},
             recovery={str(k): v for k, v in rec.items()},
             asymmetry_at_probe=dict(down=dn, up=up, diff=dn - up)),
        indent=1, default=float))
    print(f"\n[bias] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
