r"""The same-arm null for the oracle ceiling. The control I should have run first.

`cyp_route_ceiling.py` reported an oracle gap of -0.1077 +/- 0.0061 (t=-17.60)
between the `none` and `both` arms and I called it "the largest unexploited signal
this programme has measured". That reading was wrong, and the error is arithmetic
rather than statistical.

Oracle selection takes `min(e1, e2)` per molecule. For two error draws from the
same distribution the minimum has a strictly smaller expectation than either draw
-- for independent normal errors the reduction is around 40%, falling as the errors
correlate. **So a positive oracle gap is guaranteed even with zero
complementarity**, and t=-17.60 tests a null that was never in doubt. The question
is whether the cross-arm gap exceeds what two *equivalent* models produce from seed
noise alone.

That null needs no training: take two seeds of the SAME arm and compute the same
statistic. If same-arm lands near the cross-arm value, there is no complementarity
to route on and a selector is structurally impossible rather than merely hard.

Note the pairs are not equally correlated, and the direction matters for reading
the result: same-arm/different-seed models differ in initialisation, while
cross-arm/same-seed models share it and differ only in auxiliary heads. Less
correlated errors give a *larger* min-gap, so if anything this comparison is
tilted in favour of the same-arm control being bigger -- which would make the
cross-arm "headroom" worse than nothing.
"""
from __future__ import annotations

import itertools, json, statistics as st
from pathlib import Path

import numpy as np

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error, st_rae


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

    def get(p):
        z = np.load(p)
        return z["va_index"], z["val"]

    def macro(pred, idx):
        return float(np.nanmean([st_rae(pred[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                 for i, c in enumerate(PRIMARY)]))

    def oracle(pa, pb, idx):
        out = np.empty_like(pa)
        for i, c in enumerate(PRIMARY):
            ea = soft_threshold_error(pa[:, i], lo[c][idx], hi[c][idx])
            eb = soft_threshold_error(pb[:, i], lo[c][idx], hi[c][idx])
            out[:, i] = np.where(eb < ea, pb[:, i], pa[:, i])
        return macro(out, idx)

    def corr(pa, pb, idx):
        """Mean per-head correlation of the two models' errors."""
        cs = []
        for i, c in enumerate(PRIMARY):
            ea = soft_threshold_error(pa[:, i], lo[c][idx], hi[c][idx])
            eb = soft_threshold_error(pb[:, i], lo[c][idx], hi[c][idx])
            k = np.isfinite(ea) & np.isfinite(eb)
            if ea[k].std() > 0 and eb[k].std() > 0:
                cs.append(float(np.corrcoef(ea[k], eb[k])[0, 1]))
        return float(np.mean(cs)) if cs else float("nan")

    combos = {
        "none_i vs none_j  (SAME arm)": [(("none", i), ("none", j))
                                         for i, j in itertools.combinations(seeds, 2)],
        "both_i vs both_j  (SAME arm)": [(("both", i), ("both", j))
                                         for i, j in itertools.combinations(seeds, 2)],
        "none_i vs both_i  (cross, same seed)": [(("none", s), ("both", s)) for s in seeds],
        "none_i vs both_j  (cross, diff seed)": [(("none", i), ("both", j))
                                                for i, j in itertools.permutations(seeds, 2)],
    }

    singles = {}
    for a in ("none", "both"):
        for s in seeds:
            idx, p = get(runs[a][s])
            singles[(a, s)] = (idx, p, macro(p, idx))
    base = st.mean([singles[("none", s)][2] for s in seeds])

    print(f"single none macro = {base:.4f}   (reference)\n")
    print(f"{'pairing':38s} {'n':>3s} {'oracle':>8s} {'gap vs better':>14s} "
          f"{'err corr':>9s}")
    for name, pairs in combos.items():
        gaps, orcs, cors = [], [], []
        for (a1, s1), (a2, s2) in pairs:
            idx, p1, m1 = singles[(a1, s1)]
            _, p2, m2 = singles[(a2, s2)]
            o = oracle(p1, p2, idx)
            orcs.append(o)
            # vs the BETTER of the two members, which is the honest comparison:
            # an oracle that cannot beat its own best member buys nothing.
            gaps.append(o - min(m1, m2))
            cors.append(corr(p1, p2, idx))
        print(f"{name:38s} {len(pairs):3d} {st.mean(orcs):8.4f} "
              f"{st.mean(gaps):+14.4f} {st.mean(cors):9.3f}")

    def gaps_of(pairs):
        out = []
        for (a1, s1), (a2, s2) in pairs:
            idx, p1, m1 = singles[(a1, s1)]
            _, p2, m2 = singles[(a2, s2)]
            out.append(oracle(p1, p2, idx) - min(m1, m2))
        return out

    # The control must be matched on ERROR CORRELATION, not merely be same-arm.
    # The gap is a function of how decorrelated the two models are: none/none
    # errors correlate at 0.971 against cross-arm's 0.935, so none/none
    # understates the null and makes the cross-arm gap look like headroom.
    # both/both sits at 0.933 -- the like-for-like comparison.
    cross = gaps_of(combos["none_i vs both_i  (cross, same seed)"])
    ctrl_loose = gaps_of(combos["none_i vs none_j  (SAME arm)"])
    ctrl_matched = gaps_of(combos["both_i vs both_j  (SAME arm)"])

    print(f"\ncross-arm oracle gap                      {st.mean(cross):+.4f}")
    print(f"same-arm control, none/none (corr 0.971)  {st.mean(ctrl_loose):+.4f}"
          "   <- NOT the right control")
    print(f"same-arm control, both/both (corr 0.933)  {st.mean(ctrl_matched):+.4f}"
          "   <- matched on error correlation")
    resid = st.mean(cross) - st.mean(ctrl_matched)
    print(f"\nRESIDUAL attributable to the arms differing: {resid:+.4f}")
    print("A residual at or below zero means the cross-arm oracle gap is entirely")
    print("explained by error decorrelation, there is no arm-specific competence to")
    print("select on, and a selector is structurally impossible rather than merely hard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
