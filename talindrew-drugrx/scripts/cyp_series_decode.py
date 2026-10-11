r"""Series-aware decoding: keep the model's ranking, replace the series level.

CYP2D6 fails in a specific, diagnosable way. Its Spearman rho went 0.391 (validation) -> 0.330
(test), essentially intact, while its R2 went 0.144 -> **-0.951**. **A ranking that transfers
alongside a level that collapses is a calibration failure**, and a per-*series* level is a much
better target for it than the single per-endpoint offset this programme has been fitting:
`cyp_recover_series.py` finds the per-series local training-label level varies with sd **0.500**
on CYP2D6 against our entire test prediction spread of 0.398.

So the arm that matters here is `recentre`:

    p'_i = p_i + (local_level(g) - mean_{j in g} p_j)        for i in series g

which preserves the within-series ordering **exactly** and changes nothing else. `local_level(g)`
is the mean label of the k nearest labelled **training** compounds to series g -- never a
validation or test label.

Arms, cheapest first:

* `raw` -- the incumbent.
* `local` -- the local level alone, a constant per series. The floor: what pure
  nearest-neighbour-at-series-level achieves with no model at all.
* `recentre` -- as above. Ranking kept, level replaced.
* `blend` -- `w*p + (1-w)*local`, `w` cross-fitted. Shrinks toward the local level instead of
  toward a global mean.
* `recentre_shrunk` -- recentre, then apply the scale `cyp_scale_at_test_r.py` measured. The two
  corrections are orthogonal in principle (one is level, one is spread) and this checks whether
  they compose.

## The range objection, stated up front because it bounds the whole experiment

The evaluation set's series have within-cluster nearest-neighbour similarity **0.649**.
Validation is a scaffold split of a diversity library: its *most clustered quartile* reaches only
**0.483**, and an analog-expansion re-split of our own data reaches 0.446. **The target regime
lies outside the range this experiment can test**, which is the same hard bound that blocked every
earlier attempt to emulate the evaluation set.

Therefore this script does not report one number. It reports the effect **as a function of series
tightness**, so the trend can be read and the extrapolation to 0.649 is visible as an
extrapolation rather than hidden inside an average. A positive effect that *grows* with tightness
is evidence; a null at 0.45 is **untested at 0.65**, not refuted there.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
WEIGHTS = np.round(np.arange(0.0, 1.001, 0.05), 3)
SCALES = np.round(np.arange(0.30, 1.501, 0.05), 3)


def fp_of(smiles):
    out, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out.append(GEN.GetFingerprint(m)); ok.append(i)
    return out, np.array(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h12-longdeep")
    ap.add_argument("--cache", default="data/cyp_cache/cyp_h12")
    ap.add_argument("--k", type=int, default=5, help="nearest labelled training compounds")
    ap.add_argument("--per-cluster", type=int, default=10, help="target series size")
    ap.add_argument("--out", default="data/cyp_cache/series_decode.json")
    args = ap.parse_args()

    fr = load().frame
    files = sorted(glob.glob(f"{args.cache}/{args.tag}_s*.npz"))
    val = np.mean([np.load(f)["val"] for f in files], axis=0)
    va = np.load(files[0])["va_index"]
    tr = np.setdiff1d(np.arange(len(fr)), va)          # training rows only, for local levels
    print(f"[series-dec] {args.tag}, {len(files)} seeds · {len(va)} validation / {len(tr)} "
          f"training rows · metric {METRIC_ID}")

    smi = fr["smiles"].astype(str).to_numpy()
    va_fp, va_ok = fp_of(smi[va].tolist())
    tr_fp, tr_ok = fp_of(smi[tr].tolist())

    # Cluster validation among itself, at the same ~10-per-series granularity as the test set.
    n = len(va_fp)
    sim = np.ones((n, n))
    for i in range(n):
        v = DataStructs.BulkTanimotoSimilarity(va_fp[i], va_fp[i + 1:])
        if v:
            sim[i, i + 1:] = v; sim[i + 1:, i] = v
    d = 1.0 - sim
    np.fill_diagonal(d, 0.0)
    k_clust = max(2, round(n / args.per_cluster))
    lab = hierarchy.fcluster(hierarchy.linkage(squareform(d, checks=False), method="average"),
                             t=k_clust, criterion="maxclust")
    groups = {int(g): np.where(lab == g)[0] for g in sorted(set(lab))}

    # Series tightness, and the local level per series per endpoint.
    tight, level = {}, {c: {} for c in PRIMARY}
    for g, m in groups.items():
        if len(m) > 1:
            sub = sim[np.ix_(m, m)].copy(); np.fill_diagonal(sub, 0.0)
            tight[g] = float(sub.max(axis=1).mean())
        else:
            tight[g] = float("nan")
        mean_to_tr = np.mean(
            [np.array(DataStructs.BulkTanimotoSimilarity(va_fp[i], tr_fp)) for i in m], axis=0)
        order = np.argsort(mean_to_tr)[::-1]
        for c in PRIMARY:
            y = fr[c].to_numpy(float)
            picked = [int(tr[tr_ok[j]]) for j in order
                      if np.isfinite(y[int(tr[tr_ok[j]])])][: args.k]
            if picked:
                level[c][g] = float(np.mean([y[p] for p in picked]))

    tv = np.array([tight[g] for g in groups if np.isfinite(tight[g])])
    print(f"[series-dec] {len(groups)} series, median size "
          f"{np.median([len(m) for m in groups.values()]):.1f}, tightness median "
          f"{np.median(tv):.3f} (max {tv.max():.3f})")
    print(f"             TEST series tightness is 0.649 — "
          f"{'INSIDE' if tv.max() >= 0.649 else 'OUTSIDE'} this experiment's range\n")

    report = {"metric_id": METRIC_ID, "tag": args.tag, "n_series": len(groups),
              "tightness_median": float(np.median(tv)), "tightness_max": float(tv.max()),
              "endpoints": {}}

    rows = []
    for ci, c in enumerate(PRIMARY):
        y_all = fr[c].to_numpy(float)
        lo_all = fr[f"{c}_conf_low"].to_numpy(float)
        hi_all = fr[f"{c}_conf_high"].to_numpy(float)
        # rows of the validation block that are labelled on this endpoint
        keep = np.array([i for i in range(n)
                         if np.isfinite(y_all[va[va_ok[i]]])
                         and np.isfinite(lo_all[va[va_ok[i]]])
                         and np.isfinite(hi_all[va[va_ok[i]]])
                         and np.isfinite(val[va_ok[i], ci])
                         and lab[i] in level[c]])
        if len(keep) < 80:
            print(f"{c}: only {len(keep)} usable rows, skipped")
            continue
        p = val[va_ok[keep], ci]
        y = y_all[va[va_ok[keep]]]
        lo = lo_all[va[va_ok[keep]]]
        hi = hi_all[va[va_ok[keep]]]
        gl = np.array([lab[i] for i in keep])
        loc = np.array([level[c][g] for g in gl])
        tg = np.array([tight[g] for g in gl])

        # recentre: keep within-series order, move the series mean onto the local level
        rec = p.copy()
        for g in np.unique(gl):
            m = gl == g
            rec[m] = p[m] + (level[c][int(g)] - p[m].mean())

        # cross-fitted blend weight and scale
        rng = np.random.default_rng(11)
        o = rng.permutation(len(keep)); halves = [o[: len(o) // 2], o[len(o) // 2:]]
        bl_sc, rs_sc, w_pick, s_pick = [], [], [], []
        for f_i, r_i in ((0, 1), (1, 0)):
            f, r = halves[f_i], halves[r_i]
            w = WEIGHTS[int(np.argmin([st_rae(wv * p[f] + (1 - wv) * loc[f], y[f], lo[f], hi[f])
                                       for wv in WEIGHTS]))]
            s = SCALES[int(np.argmin([st_rae(rec[f].mean() + sv * (rec[f] - rec[f].mean()),
                                             y[f], lo[f], hi[f]) for sv in SCALES]))]
            w_pick.append(float(w)); s_pick.append(float(s))
            bl_sc.append(st_rae(w * p[r] + (1 - w) * loc[r], y[r], lo[r], hi[r]))
            rs_sc.append(st_rae(rec[r].mean() + s * (rec[r] - rec[r].mean()),
                                y[r], lo[r], hi[r]))

        arms = {"raw": st_rae(p, y, lo, hi),
                "local": st_rae(loc, y, lo, hi),
                "recentre": st_rae(rec, y, lo, hi),
                "blend": float(np.mean(bl_sc)),
                "recentre_shrunk": float(np.mean(rs_sc))}
        report["endpoints"][c] = dict(n=int(len(keep)), arms=arms,
                                      w=float(np.mean(w_pick)), scale=float(np.mean(s_pick)),
                                      corr_local_y=float(np.corrcoef(loc, y)[0, 1]))
        rows.append((c, arms, len(keep)))
        print(f"{c}  (n={len(keep)})   blend w={np.mean(w_pick):.2f}  "
              f"scale={np.mean(s_pick):.2f}  corr(local,y)={np.corrcoef(loc, y)[0, 1]:+.3f}")
        for a, v in arms.items():
            print(f"    {a:<17s} {v:.4f}   {v - arms['raw']:+.4f}")

        # tightness stratification — the only honest way to read this
        qs = np.nanquantile(tg, [0.25, 0.5, 0.75])
        print(f"    by series tightness (the target regime is 0.649):")
        for lab_q, m in (("Q1 loosest", tg <= qs[0]), ("Q2", (tg > qs[0]) & (tg <= qs[1])),
                         ("Q3", (tg > qs[1]) & (tg <= qs[2])), ("Q4 tightest", tg > qs[2])):
            if m.sum() < 25:
                continue
            print(f"      {lab_q:<12s} n={m.sum():4d} tight={np.nanmean(tg[m]):.3f}  "
                  f"raw {st_rae(p[m], y[m], lo[m], hi[m]):.4f}  "
                  f"recentre {st_rae(rec[m], y[m], lo[m], hi[m]):+.4f} "
                  f"({st_rae(rec[m], y[m], lo[m], hi[m]) - st_rae(p[m], y[m], lo[m], hi[m]):+.4f})")

    if rows:
        print(f"\n{'arm':<18s} {'macro':>8s} {'vs raw':>9s}")
        for a in ("raw", "local", "recentre", "blend", "recentre_shrunk"):
            v = float(np.mean([r[1][a] for r in rows]))
            b = float(np.mean([r[1]["raw"] for r in rows]))
            print(f"{a:<18s} {v:8.4f} {v - b:+9.4f}")
        report["macro"] = {a: float(np.mean([r[1][a] for r in rows]))
                           for a in ("raw", "local", "recentre", "blend", "recentre_shrunk")}

    print("\n  Read the tightness strata, not the macro. The test set's series sit at 0.649 and")
    print("  this experiment cannot reach there; a null at 0.45 is UNTESTED at 0.65.")
    Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    print(f"\n[series-dec] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
