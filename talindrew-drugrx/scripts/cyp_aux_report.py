r"""Report the auxiliary ablation from pulled artifacts, on the laptop.

`cyp_aux_ablation.py` prints this summary itself, but only on the VM and only if
it survives to the end of all sixteen runs. Three Colab sessions have now died
mid-experiment in this programme, and on 2026-10-05 one of them took a completed
arm with it. The per-seed `.npz` files are pulled to the laptop as they land, and
they carry `va_index`, so the whole analysis is reproducible here without the
session, without re-running the scaffold split, and without a GPU.

Two things it reports, and the second is the one that matters:

* **macro ST-RAE** over the four primary heads, where 1.0 is "as good as
  predicting the mean";
* **the most-active bin separately.** The tail analysis established that the
  most-active quintile carries 45.8% of the ST-RAE numerator on 20% of the
  molecules, and ChEMBL was acquired against a pre-registered tail criterion. An
  arm that improves the macro by moving the inactive bulk has not done what the
  acquisition was for, and an arm that improves the tail while the macro stays
  flat is still the right direction.

Paired across seeds: every arm trains the same seeds on the same split, so the
within-seed difference is the statistic, not the difference of means.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, soft_threshold_error, st_rae

ARMS = ("none", "veith", "chembl", "both")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/cyp_cache/cyp_aux")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    d = Path(args.dir)
    runs: dict[str, dict[int, dict]] = {}
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        if "aux_prefix" not in j:
            continue
        npz = f.with_suffix(".npz")
        if not npz.exists():
            print(f"[rep] {f.name}: no .npz beside it; skipping")
            continue
        arm = j["tag"].replace("aux-", "")
        runs.setdefault(arm, {})[int(j["seed"])] = {"json": j, "npz": npz}

    ids = {r["json"].get("metric_id") for a in runs.values() for r in a.values()}
    if len(ids) > 1:
        print(f"[rep] REFUSING to compare: mixed metric ids {ids}")
        return 1
    if ids and ids != {METRIC_ID}:
        print(f"[rep] REFUSING: artifacts scored under {ids}, this build is {METRIC_ID}")
        return 1
    bss = {r["json"].get("batch_size_final") for a in runs.values() for r in a.values()}
    if len(bss) > 1:
        print(f"[rep] REFUSING to compare: batch sizes diverged {bss}")
        return 1

    data = load()
    frame = data.frame
    lo = {c: frame[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: frame[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}
    y = {c: frame[c].to_numpy(float) for c in PRIMARY}

    print(f"metric {METRIC_ID}; batch size {bss.pop() if bss else '?'}; "
          f"1.0 == predicting the mean\n")
    print(f"{'arm':8s} {'seeds':>5s} {'macro':>8s} {'sd':>8s}   per-seed")
    macro: dict[str, dict[int, float]] = {}
    for arm in ARMS:
        if arm not in runs:
            continue
        per = {}
        for seed, r in sorted(runs[arm].items()):
            z = np.load(r["npz"])
            idx, pred = z["va_index"], z["val"]
            scores = [st_rae(pred[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                      for i, c in enumerate(PRIMARY)]
            per[seed] = float(np.nanmean(scores))
        macro[arm] = per
        v = list(per.values())
        sd = f"{st.stdev(v):8.4f}" if len(v) > 1 else "       -"
        print(f"{arm:8s} {len(v):5d} {st.mean(v):8.4f} {sd}   "
              + " ".join(f"{x:.4f}" for x in v))

    base = "none"
    if base in macro:
        print(f"\npaired vs '{base}' (negative == better than the control)")
        for arm in ARMS:
            if arm == base or arm not in macro:
                continue
            shared = sorted(set(macro[arm]) & set(macro[base]))
            if len(shared) < 2:
                print(f"  {arm:8s} too few shared seeds")
                continue
            diffs = [macro[arm][s] - macro[base][s] for s in shared]
            m, sd = st.mean(diffs), st.stdev(diffs)
            se = sd / len(diffs) ** 0.5
            t = m / se if se else float("nan")
            wins = sum(1 for x in diffs if x < 0)
            verdict = "DECIDED" if abs(t) > 2.5 else "inside noise"
            print(f"  {arm:8s} {m:+.4f} +/- {se:.4f} (n={len(diffs)}, "
                  f"better on {wins}/{len(diffs)}, t={t:+.2f}) -> {verdict}")

    # ── the tail read ────────────────────────────────────────────────
    # Per-molecule MEANS, not sums. The ST-RAE numerator is a sum, so a bin total
    # confounds error with bin population: a 5-unit reduction over 400 molecules
    # and over 40 are different claims. Quantile bins come out near-equal here
    # (~212 pooled, no label ties), but printing the mean and the count removes
    # the ambiguity instead of relying on that holding.
    def binned(arm, head=None):
        """(mean error per bin, count per bin) for one arm, pooled or per head."""
        tot = np.zeros(args.bins)
        cnt = np.zeros(args.bins)
        for seed, r in sorted(runs[arm].items()):
            z = np.load(r["npz"])
            idx, pred = z["va_index"], z["val"]
            for i, c in enumerate(PRIMARY):
                if head is not None and c != head:
                    continue
                yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
                keep = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) \
                    & np.isfinite(pred[:, i])
                if keep.sum() < args.bins:
                    continue
                err = soft_threshold_error(pred[keep, i], ll[keep], hh[keep])
                # Bin edges from the *labels*, so bins mean the same thing in
                # every arm; quantiles of the prediction would move per arm and
                # make the columns incomparable.
                edges = np.quantile(yy[keep], np.linspace(0, 1, args.bins + 1))
                which = np.clip(np.searchsorted(edges, yy[keep], side="right") - 1,
                                0, args.bins - 1)
                for b in range(args.bins):
                    m = which == b
                    tot[b] += err[m].sum()
                    cnt[b] += int(m.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            return tot / np.where(cnt > 0, cnt, np.nan), cnt

    print(f"\nMEAN soft-threshold error per molecule by true-label bin "
          f"({args.bins} bins, most-active last)")
    present = [a for a in ARMS if a in runs]
    _, cnt0 = binned(present[0])
    print(f"{'arm':8s} " + " ".join(f"{'bin'+str(b+1):>9s}" for b in range(args.bins)))
    print(f"{'(n/seed)':8s} " + " ".join(
        f"{cnt0[b] / max(len(runs[present[0]]), 1):9.0f}" for b in range(args.bins)))
    means = {}
    for arm in present:
        mu, _ = binned(arm)
        means[arm] = mu
        print(f"{arm:8s} " + " ".join(f"{x:9.4f}" for x in mu))
    if base in means:
        print(f"\ndelta vs '{base}' (negative == better)")
        for arm in present:
            if arm == base:
                continue
            print(f"{arm:8s} " + " ".join(
                f"{means[arm][b] - means[base][b]:+9.4f}" for b in range(args.bins)))

    # ── per isoform, most-active bin only ────────────────────────────
    # Pre-registered prediction: if the tail effect is tail-specific learning it
    # should be largest on CYP2D6 and CYP1A2 (widest label spreads, hardest tails)
    # and near-absent on CYP3A4. If it is uniform, that points to a global shift.
    print(f"\nMost-active bin only, per isoform (mean error; delta vs '{base}')")
    print(f"{'head':10s} " + " ".join(f"{a:>18s}" for a in present))
    for c in PRIMARY:
        cells = []
        bmu, _ = binned(base, head=c)
        for arm in present:
            mu, _ = binned(arm, head=c)
            d = mu[-1] - bmu[-1]
            cells.append(f"{mu[-1]:8.4f}" + ("        —" if arm == base
                                             else f" ({d:+.4f})"))
        print(f"{c.split('_')[0]:10s} " + " ".join(f"{x:>18s}" for x in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
