r"""Per-bin paired differences vs the control, pooled over the four heads.

`cyp_aux_report.py` prints per-bin means per arm, and those means suggested a
clean story: every auxiliary arm improved the most-active bin by 0.012-0.033 and
worsened the least-active one by a similar amount -- a trade, with the macro
cancelling. This script is what refuted it.

Two things only the paired per-seed view shows:

* **The spread is 3-5x the mean.** veith's bin-5 deltas across four seeds are
  -0.0879, -0.0003, +0.0175, -0.0220: mean -0.0232 with a standard error of
  0.0230. Nothing in any bin, in any arm, clears |t| > 2.5.
* **Cross-arm agreement is not corroboration.** The three arms train the *same*
  seeds, so a seed that happens to produce a good tail produces one in every arm
  -- seed 0 is -0.0879 / -0.0444 / -0.1103. "All three arms agree" is therefore
  close to one observation, not three, and reading it as replication is how the
  trade story survived as long as it did.

Bins are quantiles of the labels, computed per head, so a bin means the same thing
in every arm; quantiles of the prediction would move per arm and make the columns
incomparable.
"""
import json, statistics as st, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import soft_threshold_error

runs = {}
for f in sorted(Path("data/cyp_cache/cyp_aux").glob("*.json")):
    j = json.loads(f.read_text())
    npz = f.with_suffix(".npz")
    if npz.exists():
        runs.setdefault(j["tag"].replace("aux-", ""), {})[int(j["seed"])] = npz

def main():
    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    def bins_pooled(npz, nb=5):
        """Pooled mean error per bin for one seed: errors from all four heads together."""
        z = np.load(npz); idx, pred = z["va_index"], z["val"]
        acc = [[] for _ in range(nb)]
        for i, c in enumerate(PRIMARY):
            yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
            k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh) & np.isfinite(pred[:, i])
            yy, ll, hh, pp = yy[k], ll[k], hh[k], pred[k, i]
            e = soft_threshold_error(pp, ll, hh)
            edges = np.quantile(yy, np.linspace(0, 1, nb + 1))
            w = np.clip(np.searchsorted(edges, yy, side="right") - 1, 0, nb - 1)
            for b in range(nb):
                acc[b].append(e[w == b])
        return np.array([np.concatenate(a).mean() for a in acc])

    seeds = sorted(set.intersection(*(set(v) for v in runs.values())))
    base = {s: bins_pooled(runs["none"][s]) for s in seeds}
    print(f"pooled over 4 heads, paired on {len(seeds)} seeds; negative == better\n")
    print(f"{'arm':8s} " + " ".join(f"{'bin'+str(b+1):>22s}" for b in range(5)))
    for arm in ("veith", "chembl", "both"):
        cells = []
        for b in range(5):
            d = [bins_pooled(runs[arm][s])[b] - base[s][b] for s in seeds]
            m = st.mean(d); se = st.stdev(d)/len(d)**0.5; t = m/se
            cells.append(f"{m:+.4f}±{se:.4f} t{t:+5.2f}{'*' if abs(t)>2.5 else ''}")
        print(f"{arm:8s} " + " ".join(f"{c:>22s}" for c in cells))
    print("\nsign test across the three arms, most-active bin:")
    for arm in ("veith","chembl","both"):
        d = [bins_pooled(runs[arm][s])[4] - base[s][4] for s in seeds]
        print(f"  {arm:8s} per-seed deltas: " + " ".join(f"{x:+.4f}" for x in d)
              + f"   better on {sum(1 for x in d if x<0)}/{len(d)}")


if __name__ == "__main__":
    main()
