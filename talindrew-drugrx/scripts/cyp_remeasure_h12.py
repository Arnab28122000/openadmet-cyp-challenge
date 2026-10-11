r"""Re-measure the depth and width hypotheses under the corrected metric.

H12 ("capacity moves the trajectory, not the destination") concluded that depth
(p=0.723) and width (+0.0000) buy nothing. Both were measured under the **superseded
ST-RAE denominator**, so neither conclusion had been verified against the objective the
challenge actually scores. Re-measuring them needs no GPU: the per-seed `.npz` files
hold validation predictions, and predictions are metric-independent.

**The stored `valid` fields in those files cannot be used.** They are a mixture:
`long-base_s0` reads 0.7134 (official scale) while `s1` and `s2` read 0.4448 and 0.4381
(old scale), and every file has `metric_id: None`. This is the contamination that
`METRIC_ID` was introduced to prevent -- the log records eight depth runs being pooled
from five old-metric and three official-metric JSONs, with a paired test averaged across
both. Re-scoring from predictions removes it rather than working around it.

Paired within seed, which is what the design supports: all three configurations share
one validation split (verified by hash against the other 36 runs in this programme).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae

CONFIGS = {"long-base": "d1024 L3", "long-deep": "d1024 L6", "long-wide": "d2048 L4"}


def main() -> int:
    d = Path("data/cyp_cache/cyp_long")
    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    scores, stored, ref = {}, {}, None
    for cfg in CONFIGS:
        scores[cfg], stored[cfg] = {}, {}
        for s in range(5):
            f, npz = d / f"{cfg}_s{s}.json", d / f"{cfg}_s{s}.npz"
            if not npz.exists():
                continue
            z = np.load(npz)
            if ref is None:
                ref = z["va_index"]
            assert np.array_equal(z["va_index"], ref), f"{cfg}_s{s}: different split"
            idx, p = z["va_index"], z["val"]
            scores[cfg][s] = float(np.nanmean(
                [st_rae(p[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                 for i, c in enumerate(PRIMARY)]))
            stored[cfg][s] = json.loads(f.read_text()).get("valid")

    print(f"re-scored under {METRIC_ID}; 1.0 == predicting the mean\n")
    print(f"{'config':12s} {'n':>2s} {'RE-SCORED':>10s} {'sd':>8s}   "
          f"{'stored (UNUSABLE, mixed metrics)':>34s}")
    for cfg, desc in CONFIGS.items():
        v = list(scores[cfg].values())
        if not v:
            continue
        sd = f"{st.stdev(v):8.4f}" if len(v) > 1 else "       -"
        old = " ".join(f"{stored[cfg][s]:.3f}" for s in sorted(stored[cfg]))
        print(f"{cfg:12s} {len(v):2d} {st.mean(v):10.4f} {sd}   {old:>34s}")

    print(f"\n{'comparison':26s} {'paired diff':>12s} {'se':>8s} {'t':>7s} "
          f"{'wins':>6s} verdict")
    for a, b, label in (("long-deep", "long-base", "DEPTH  L6 vs L3"),
                        ("long-wide", "long-base", "WIDTH  d2048L4 vs d1024L3")):
        shared = sorted(set(scores[a]) & set(scores[b]))
        if len(shared) < 2:
            print(f"{label:26s} too few shared seeds")
            continue
        diffs = [scores[a][s] - scores[b][s] for s in shared]
        m = st.mean(diffs)
        se = st.stdev(diffs) / len(diffs) ** 0.5
        t = m / se if se else float("nan")
        wins = sum(1 for x in diffs if x < 0)
        verdict = "DECIDED" if abs(t) > 2.5 else "inside noise"
        print(f"{label:26s} {m:+12.4f} {se:8.4f} {t:+7.2f} "
              f"{wins:3d}/{len(diffs)}  {verdict}")
        print(f"{'':26s} per-seed: " + " ".join(f"{x:+.4f}" for x in diffs))
        need = (2.5 * st.stdev(diffs) / abs(m)) ** 2 if m else float("inf")
        print(f"{'':26s} seeds/arm to resolve at this effect: {need:.0f}")

    print("\nNegative favours the larger model. 'wins' counts seeds where it was better.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
