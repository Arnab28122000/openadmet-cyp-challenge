r"""Per-isoform tail effects **with their paired standard errors**.

`cyp_aux_report.py` prints per-isoform most-active-bin deltas, and a table of
deltas without errors beside them invites exactly the reading this programme keeps
having to retract: four seeds produce differences of the same size as the effect,
so an ordering across four isoforms and three arms is twelve numbers of which
several will look large by chance.

The pre-registered prediction was that the tail effect should be largest on CYP2D6
and CYP1A2 (widest label spreads) and near-absent on CYP3A4. Testing it needs the
within-seed paired difference per isoform, not the difference of pooled means --
every arm trains the same seeds on the same split, so pairing removes the
seed-to-seed component that dominates here.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error

ARMS = ("none", "veith", "chembl", "both")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/cyp_cache/cyp_aux")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    runs: dict[str, dict[int, Path]] = {}
    for f in sorted(Path(args.dir).glob("*.json")):
        j = json.loads(f.read_text())
        if "aux_prefix" not in f.read_text() and "aux_prefix" not in j:
            continue
        npz = f.with_suffix(".npz")
        if npz.exists():
            runs.setdefault(j["tag"].replace("aux-", ""), {})[int(j["seed"])] = npz

    data = load()
    fr = data.frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    def tail_mean(npz: Path, head: str) -> float:
        z = np.load(npz)
        idx, pred = z["va_index"], z["val"]
        i = PRIMARY.index(head)
        yy, ll, hh = y[head][idx], lo[head][idx], hi[head][idx]
        keep = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) & np.isfinite(pred[:, i])
        yy, ll, hh, pp = yy[keep], ll[keep], hh[keep], pred[keep, i]
        edges = np.quantile(yy, np.linspace(0, 1, args.bins + 1))
        which = np.clip(np.searchsorted(edges, yy, side="right") - 1, 0, args.bins - 1)
        m = which == args.bins - 1
        return float(soft_threshold_error(pp, ll, hh)[m].mean())

    seeds = sorted(set.intersection(*(set(runs[a]) for a in ARMS if a in runs)))
    print(f"paired on {len(seeds)} shared seeds: {seeds}")
    print("most-active-bin mean error, paired difference vs 'none'\n")
    print(f"{'head':8s} {'none':>8s} " + "".join(
        f"{a:>26s}" for a in ARMS if a != "none" and a in runs))

    for head in PRIMARY:
        base = {s: tail_mean(runs["none"][s], head) for s in seeds}
        cells = []
        for arm in ARMS:
            if arm == "none" or arm not in runs:
                continue
            d = [tail_mean(runs[arm][s], head) - base[s] for s in seeds]
            m = st.mean(d)
            se = st.stdev(d) / len(d) ** 0.5 if len(d) > 1 else float("nan")
            t = m / se if se else float("nan")
            wins = sum(1 for x in d if x < 0)
            flag = "*" if abs(t) > 2.5 else " "
            cells.append(f"{m:+.4f}±{se:.4f} t{t:+5.2f} {wins}/{len(d)}{flag}")
        print(f"{head.split('_')[0]:8s} {st.mean(base.values()):8.4f} "
              + "".join(f"{c:>26s}" for c in cells))

    print("\n* = clears the pre-registered |t| > 2.5 bar. Pairing is within seed; "
          "'wins' counts seeds where the arm beat the control on that isoform's tail.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
