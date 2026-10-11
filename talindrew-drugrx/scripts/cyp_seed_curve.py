r"""Macro as a function of how many seeds are averaged. Free, from saved predictions.

`cyp_final.py` averages `--seeds` net runs and defaults to **3**, but the ensembling
gain measured in this programme was for **2**-model averaging (-0.005 to -0.009).
The gain depends on the count, so the submitted count must be the validated one or
the quoted number does not describe what ships.

Averaging happens before the blend weight and the offsets are fitted
(`cyp_final.py:173`), which is the correct order: the offset is the largest
confirmed effect here (+0.0211) and fitting it per-seed then averaging
offset-applied predictions would average k different offsets instead of fitting one
to the thing being submitted. Verified, not assumed.

All subsets of each size are averaged, so each point is a mean over C(4,k)
combinations rather than one arbitrary pick.
"""
from __future__ import annotations

import itertools, json, statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae


def main() -> int:
    runs = {}
    for f in sorted(Path("data/cyp_cache/cyp_aux").glob("*.json")):
        j = json.loads(f.read_text())
        npz = f.with_suffix(".npz")
        if npz.exists():
            runs.setdefault(j["tag"].replace("aux-", ""), {})[int(j["seed"])] = npz
    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    def macro(pred, idx):
        return float(np.nanmean([st_rae(pred[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                 for i, c in enumerate(PRIMARY)]))

    print(f"{'arm':8s} {'k':>2s} {'combos':>7s} {'macro':>8s} {'sd':>8s} "
          f"{'vs k=1':>9s} {'marginal':>9s}")
    for arm in ("none", "veith", "chembl", "both"):
        if arm not in runs:
            continue
        seeds = sorted(runs[arm])
        preds = {}
        for s in seeds:
            z = np.load(runs[arm][s])
            preds[s] = (z["va_index"], z["val"])
        idx = preds[seeds[0]][0]
        assert all(np.array_equal(preds[s][0], idx) for s in seeds), "splits differ"
        prev = None
        base = None
        for k in range(1, len(seeds) + 1):
            vals = []
            for combo in itertools.combinations(seeds, k):
                avg = np.mean([preds[s][1] for s in combo], axis=0)
                vals.append(macro(avg, idx))
            m = st.mean(vals)
            sd = f"{st.stdev(vals):8.4f}" if len(vals) > 1 else "       -"
            if k == 1:
                base = m
            marg = f"{m - prev:+9.4f}" if prev is not None else "        -"
            print(f"{arm:8s} {k:2d} {len(vals):7d} {m:8.4f} {sd} "
                  f"{m - base:+9.4f} {marg}")
            prev = m
    print("\n'vs k=1' is the cumulative ensembling gain; 'marginal' is what the k-th")
    print("seed added. A marginal gain that has collapsed means more seeds buy nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
