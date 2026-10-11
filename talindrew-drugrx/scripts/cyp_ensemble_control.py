r"""Is mean(none, both) a fact about auxiliary heads, or just about averaging?

`cyp_route_ceiling.py` found that averaging the `none` and `both` arms beats the
`none` control by 0.0094 +/- 0.0017 on the macro (t=-5.69, 4/4 seeds) -- clearing
the pre-registered bar, while the `both` arm on its own does not.

The obvious reading is "the arms are complementary, so combining them helps". The
boring reading is "averaging two decorrelated models of any kind helps", which
would make the result a fact about ensembling and say nothing about the auxiliary
heads or the ChEMBL acquisition. Averaging two models from the SAME arm,
different seeds, separates them, and it needs no GPU: all the predictions exist.

If `mean(none_i, none_j)` gains as much as `mean(none_i, both_i)`, the finding is
ensembling and the aux heads are irrelevant to it.
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
    seeds = sorted(set(runs["none"]) & set(runs["both"]))
    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    def load_pred(p):
        z = np.load(p)
        return z["va_index"], z["val"]

    def macro(pred, idx):
        return float(np.nanmean([st_rae(pred[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                 for i, c in enumerate(PRIMARY)]))

    single = {a: {} for a in ("none", "both")}
    for a in single:
        for s in seeds:
            idx, p = load_pred(runs[a][s])
            single[a][s] = (idx, p, macro(p, idx))

    print(f"single-model macro, {len(seeds)} seeds")
    for a in single:
        v = [single[a][s][2] for s in seeds]
        print(f"  {a:6s} mean {st.mean(v):.4f}   " + " ".join(f"{x:.4f}" for x in v))

    print("\ntwo-model averages (lower is better)")
    rows = {}
    # same-arm, different seeds: the ensembling control
    for a in ("none", "both"):
        vals = []
        for i, j in itertools.combinations(seeds, 2):
            idx, pi, _ = single[a][i]
            _, pj, _ = single[a][j]
            vals.append(macro(0.5 * (pi + pj), idx))
        rows[f"mean({a},{a}) diff seeds"] = vals
    # cross-arm, same seed: the original finding
    vals = []
    for s in seeds:
        idx, pn, _ = single["none"][s]
        _, pb, _ = single["both"][s]
        vals.append(macro(0.5 * (pn + pb), idx))
    rows["mean(none,both) same seed"] = vals
    # cross-arm, different seeds: separates arm-complementarity from seed decorrelation
    vals = []
    for i, j in itertools.permutations(seeds, 2):
        idx, pn, _ = single["none"][i]
        _, pb, _ = single["both"][j]
        vals.append(macro(0.5 * (pn + pb), idx))
    rows["mean(none,both) diff seeds"] = vals

    base = st.mean([single["none"][s][2] for s in seeds])
    print(f"  {'combination':30s} {'n':>3s} {'macro':>8s} {'vs none mean':>13s}")
    for k, v in rows.items():
        print(f"  {k:30s} {len(v):3d} {st.mean(v):8.4f} {st.mean(v) - base:+13.4f}")
    print(f"\n  reference: single none = {base:.4f}")
    print("\nIf same-arm averaging gains as much as cross-arm, the effect is ensembling")
    print("and says nothing about the auxiliary heads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
