r"""Recover the 75 analog series the blinded test set was built from.

The organisers state the construction verbatim (challenge space `config.py`):

    "Analog Expansion Test Set: Hit expansion was performed on the top 25 hits from CYP1A2,
     CYP2C9, and CYP3A4 by purchasing the top 10 chemisimilars (ECFP4 Tanimoto similarity) per
     hit compound from the Enamine US catalog."

75 hits x 10 analogs = 750, which is exactly the test set size. The hits themselves are
**training** compounds whose labels we hold; the analogs were purchased and are not in training.

So the generative structure of the evaluation set is knowable, not merely estimable:

* each test compound should have, as its nearest training neighbour, **the hit it was bought to
  resemble**;
* the 750 should fall into **~75 groups of ~10**;
* the recovered anchors should be the **top 25** of three endpoints under whatever "hit" meant.

Every earlier use of this structure in the programme was a *similarity statistic* -- "84.3% of
test compounds sit within Tanimoto 0.4 of a training active". This asks for the grouping itself.
If it recovers, three things follow that no similarity number gives:

1. **Every test compound gets a named anchor with a known label.** The pre-registered
   pairwise/delta model was built around guessing an anchor by nearest-neighbour search; here the
   anchors are handed over by the construction.
2. **The 750 partition into 75 series**, so a prediction can be made series-aware -- shrunk
   toward its own series' anchor rather than toward a global mean. CYP2D6's failure is a *level*
   failure (rho intact at 0.330, R2 collapsed to -0.951), and a per-series level is a far better
   target than a per-endpoint one.
3. **Which endpoint each series was selected for.** CYP2D6 was *not* one of the three, so for
   CYP2D6 every series is an arbitrary set of analogs -- which is the mechanism behind its
   collapse, and this would make it explicit per compound rather than inferred from a mean.

## What is a prediction here, and what would falsify it

The group-size distribution is the test. If the construction is as stated and ECFP4/2048 binary
Tanimoto is what they used, the recovered anchor count should be near **75** and group sizes near
**10**. A long tail of singletons, or 400 distinct anchors, falsifies either the construction or
our reading of the fingerprint.

"Top 25 hits" is ambiguous and is therefore **tested rather than assumed**: ranked by
direct-inhibition pIC50, by TDI-condition pIC50, by Emax, or by the single-concentration primary
screen (`log2fc`) -- the last being the most likely reading of "hits", since the primary screen is
what a hit comes from. The ranking that best explains the recovered anchor set is reported, not
chosen in advance. A prior attempt noted in review recovered only 18/25 for CYP3A4, 21/25 for
CYP2C9 and **6/25** for CYP1A2 under direct pIC50, so direct pIC50 is probably not it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import AUX_EMAX_DIRECT, AUX_LOG2FC, AUX_TDI, PRIMARY, load

RDLogger.DisableLog("rdApp.*")
# radius 2 / 2048 bits binary == ECFP4, the organisers' stated similarity, and the convention
# already used by cyp_analog_split.py and cyp_pair_space_feasibility.py.
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# The three endpoints the expansion was built from. CYP2D6 is deliberately absent.
SELECTED = ("CYP1A2", "CYP2C9", "CYP3A4")


def fps(smiles: list[str]):
    out, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None:
            continue
        out.append(GEN.GetFingerprint(m))
        ok.append(i)
    return out, np.array(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cyp_cache/series.json")
    args = ap.parse_args()

    d = load()
    fr, test = d.frame, d.test
    tr_fp, tr_ok = fps(fr["smiles"].astype(str).tolist())
    te_fp, te_ok = fps(test["smiles"].astype(str).tolist())
    print(f"[series] {len(tr_fp)} training / {len(te_fp)} test fingerprints "
          f"(ECFP4, radius 2, 2048 bits, binary)")

    # Nearest training neighbour of every test compound.
    anchors, sims, runner_up = [], [], []
    for f in te_fp:
        v = np.array(DataStructs.BulkTanimotoSimilarity(f, tr_fp))
        order = np.argsort(v)[::-1]
        anchors.append(int(tr_ok[order[0]]))
        sims.append(float(v[order[0]]))
        runner_up.append(float(v[order[1]]))
    anchors, sims, runner_up = np.array(anchors), np.array(sims), np.array(runner_up)

    counts = Counter(anchors.tolist())
    sizes = Counter(counts.values())
    print(f"\n[series] distinct nearest-training-neighbours: **{len(counts)}**  "
          f"(the construction predicts ~75)")
    print(f"[series] NN similarity: median {np.median(sims):.3f}  "
          f"mean {sims.mean():.3f}  min {sims.min():.3f}  max {sims.max():.3f}")
    print(f"\ngroup-size distribution (construction predicts ~10 each):")
    for sz in sorted(sizes):
        print(f"   size {sz:3d}: {sizes[sz]:4d} anchor(s)   "
              f"{'#' * min(sizes[sz], 60)}")
    print(f"   compounds in groups of exactly 10: "
          f"{sum(n for a, n in counts.items() if n == 10)}")
    print(f"   compounds in groups of 8-12:       "
          f"{sum(n for a, n in counts.items() if 8 <= n <= 12)}")

    # Is the anchor set the top-25 of the three selected endpoints, under some ranking?
    print(f"\nwhich ranking explains the recovered anchors? "
          f"(top-25 per endpoint, {len(SELECTED)} endpoints = 75 expected)")
    anchor_set = set(counts)
    rankings = {
        "direct pIC50": {c: f"{c}_pIC50_direct_inhibition" for c in SELECTED},
        "TDI pIC50": {c: f"{c}_pIC50_TDI_condition" for c in SELECTED},
        "Emax (direct)": {c: f"{c}_EmaxVsPosCtrl_direct_inhibition" for c in SELECTED},
        "primary screen log2fc": {c: f"log2fc_{c}" for c in SELECTED},
    }
    best_name, best_hits, detail = None, -1, {}
    for name, cols in rankings.items():
        per, tot = {}, 0
        for cyp, col in cols.items():
            if col not in fr.columns:
                per[cyp] = "col missing"
                continue
            v = fr[col].to_numpy(float)
            # log2fc is a depletion/fold-change readout: more negative == more inhibition for
            # some conventions, so take the tail that actually contains the potent compounds by
            # checking which end correlates with direct pIC50.
            ref = fr[f"{cyp}_pIC50_direct_inhibition"].to_numpy(float)
            k = np.isfinite(v) & np.isfinite(ref)
            sign = 1.0
            if k.sum() > 30 and np.corrcoef(v[k], ref[k])[0, 1] < 0:
                sign = -1.0
            vv = sign * v
            idx = np.argsort(np.where(np.isfinite(vv), vv, -np.inf))[::-1][:25]
            hit = len(set(int(i) for i in idx) & anchor_set)
            per[cyp] = f"{hit}/25" + ("" if sign > 0 else " (sign flipped)")
            tot += hit
        detail[name] = per
        print(f"   {name:<24s} " + "  ".join(f"{c} {per[c]}" for c in SELECTED)
              + f"   total {tot}/75")
        if tot > best_hits:
            best_name, best_hits = name, tot

    print(f"\n   best explanation: {best_name} at {best_hits}/75")
    if best_hits < 50:
        print("   NOT EXPLAINED — fewer than half the anchors are top-25 under any ranking we")
        print("   hold. Either 'hits' were ranked on a readout not in the release, or the")
        print("   nearest-neighbour recovery is not finding the true anchors. Treat the")
        print("   grouping as a hypothesis, not as the construction.")

    # How concentrated is each recovered group, and does the grouping look generative?
    big = {a: n for a, n in counts.items() if n >= 5}
    print(f"\nanchors with >=5 analogs: {len(big)}, covering "
          f"{sum(big.values())} of {len(te_fp)} test compounds "
          f"({100*sum(big.values())/len(te_fp):.1f}%)")
    print(f"margin of the nearest anchor over the runner-up (median): "
          f"{np.median(sims - runner_up):+.4f}")
    print("  A small margin means the assignment is ambiguous and the grouping is soft.")

    # Anchor labels — the thing a series-aware decoder would use.
    print(f"\nlabel coverage of the recovered anchors (what a series-aware decoder could use):")
    for c in PRIMARY:
        v = fr[c].to_numpy(float)
        have = sum(1 for a in counts if np.isfinite(v[a]))
        covered = sum(n for a, n in counts.items() if np.isfinite(v[a]))
        print(f"   {c:<34s} {have:3d}/{len(counts)} anchors labelled, covering "
              f"{covered:3d}/{len(te_fp)} test compounds "
              f"({100*covered/len(te_fp):5.1f}%)   anchor mean "
              f"{np.nanmean([v[a] for a in counts if np.isfinite(v[a])]):.3f}")

    # ------------------------------------------------------------------ pass 2
    # Assigning each test compound to its own nearest *training* neighbour splits true series:
    # when a hit has close analogs already in training, some of its ten purchased analogs are
    # nearer to one of those than to the hit itself. The ten analogs of one hit are, however,
    # mutually similar by construction -- so cluster the test set **among itself** and give each
    # cluster one consensus anchor. The spike at exactly 10 in pass 1 is what says this is worth
    # doing; a flat distribution would have said the construction is not visible at all.
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    n = len(te_fp)
    sim = np.ones((n, n), float)
    for i in range(n):
        v = DataStructs.BulkTanimotoSimilarity(te_fp[i], te_fp[i + 1:])
        if v:
            sim[i, i + 1:] = v
            sim[i + 1:, i] = v
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    Z = hierarchy.linkage(squareform(dist, checks=False), method="average")

    print(f"\n{'-' * 70}\nPASS 2 — cluster the test set among itself, then name one anchor per "
          f"cluster")
    print(f"{'k':>5s} {'size=10':>8s} {'8-12':>7s} {'singletons':>11s} {'median':>7s} "
          f"{'mean innerNN':>13s}")
    cl_report = {}
    for k in (60, 70, 75, 80, 90, 100):
        lab = hierarchy.fcluster(Z, t=k, criterion="maxclust")
        cc = Counter(lab.tolist())
        inner = []
        for g in set(lab):
            m = np.where(lab == g)[0]
            if len(m) > 1:
                sub = sim[np.ix_(m, m)].copy()
                np.fill_diagonal(sub, 0.0)
                inner.append(float(sub.max(axis=1).mean()))
        cl_report[k] = dict(
            exact10=int(sum(v for v in cc.values() if v == 10)),
            near10=int(sum(v for v in cc.values() if 8 <= v <= 12)),
            singletons=int(sum(1 for v in cc.values() if v == 1)),
            median=float(np.median(list(cc.values()))),
            inner=float(np.mean(inner)) if inner else float("nan"))
        r = cl_report[k]
        print(f"{k:5d} {r['exact10']:8d} {r['near10']:7d} {r['singletons']:11d} "
              f"{r['median']:7.1f} {r['inner']:13.3f}")
    print("  'size=10' counts COMPOUNDS in clusters of exactly ten. The construction is 75x10,")
    print("  so k=75 is the pre-registered cut; the others are shown to see whether the signal")
    print("  is a property of the data or of the chosen k.")

    lab = hierarchy.fcluster(Z, t=75, criterion="maxclust")
    cons, cons_sim = {}, {}
    for g in sorted(set(lab)):
        m = np.where(lab == g)[0]
        mean_to_train = np.mean(
            [np.array(DataStructs.BulkTanimotoSimilarity(te_fp[i], tr_fp)) for i in m], axis=0)
        j = int(np.argmax(mean_to_train))
        cons[int(g)] = int(tr_ok[j])
        cons_sim[int(g)] = float(mean_to_train[j])
    print(f"\n  k=75: {len(set(cons.values()))} distinct consensus anchors for 75 clusters "
          f"(mean similarity to cluster {np.mean(list(cons_sim.values())):.3f})")

    cons_set = set(cons.values())
    print(f"  are the consensus anchors the top-25 of the three selected endpoints?")
    for name, cols in rankings.items():
        per, tot = {}, 0
        for cyp, col in cols.items():
            if col not in fr.columns:
                per[cyp] = "n/a"; continue
            v = fr[col].to_numpy(float)
            ref = fr[f"{cyp}_pIC50_direct_inhibition"].to_numpy(float)
            kk = np.isfinite(v) & np.isfinite(ref)
            sign = -1.0 if (kk.sum() > 30 and np.corrcoef(v[kk], ref[kk])[0, 1] < 0) else 1.0
            vv = sign * v
            idx = np.argsort(np.where(np.isfinite(vv), vv, -np.inf))[::-1][:25]
            hit = len(set(int(i) for i in idx) & cons_set)
            per[cyp] = hit; tot += hit
        print(f"     {name:<24s} " + "  ".join(f"{c} {per[c]}/25" for c in SELECTED)
              + f"   total {tot}/75")

    # ------------------------------------------------------------------ pass 3
    # Run the construction FORWARDS. For each candidate hit, take its ten nearest test
    # compounds; if it really is a hit, those ten should be one cluster. This is the decisive
    # test, because it uses the stated procedure in the stated direction instead of inverting it.
    print(f"\n{'-' * 70}\nPASS 3 — run the construction forwards from each candidate hit")
    print(f"{'ranking':<22s} {'endpoint':<8s} {'mean purity':>12s} {'10/10':>7s} "
          f"{'>=8/10':>7s} {'mean sim':>9s}")
    fwd = {}
    for name, cols in (("direct pIC50", rankings["direct pIC50"]),
                       ("TDI pIC50", rankings["TDI pIC50"])):
        for cyp, col in cols.items():
            if col not in fr.columns:
                continue
            v = fr[col].to_numpy(float)
            idx = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1][:25]
            pur, sims10, perfect, good = [], [], 0, 0
            for i in idx:
                row = np.where(tr_ok == i)[0]
                if not len(row):
                    continue
                s = np.array(DataStructs.BulkTanimotoSimilarity(tr_fp[int(row[0])], te_fp))
                top = np.argsort(s)[::-1][:10]
                labs = lab[top]
                p = Counter(labs.tolist()).most_common(1)[0][1] / 10.0
                pur.append(p); sims10.append(float(s[top].mean()))
                perfect += p == 1.0
                good += p >= 0.8
            if pur:
                fwd[f"{name}/{cyp}"] = dict(purity=float(np.mean(pur)), perfect=perfect,
                                            good=good, sim=float(np.mean(sims10)))
                print(f"{name:<22s} {cyp:<8s} {np.mean(pur):12.3f} {perfect:5d}/25 "
                      f"{good:5d}/25 {np.mean(sims10):9.3f}")
    print("  purity = largest share of a hit's ten nearest test compounds falling in one")
    print("  cluster. A true hit should score near 1.0; a non-hit scatters across clusters.")

    # ------------------------------------------------------------------ pass 4
    # The actionable output, which does NOT depend on naming the official hits: a local
    # training-label level per cluster per endpoint. CYP2D6's failure is a level failure
    # (rho 0.330 intact, R2 -0.951), and a per-series level is a far better target than a
    # per-endpoint one. k nearest LABELLED training compounds to each cluster.
    print(f"\n{'-' * 70}\nPASS 4 — local training-label level per cluster (the usable product)")
    K = 5
    local = {}
    print(f"{'endpoint':<34s} {'clusters w/ level':>18s} {'test covered':>13s} "
          f"{'level sd':>9s} {'spread':>16s}")
    for ci, c in enumerate(PRIMARY):
        v = fr[c].to_numpy(float)
        lv, cov = {}, 0
        for g in sorted(set(lab)):
            m = np.where(lab == g)[0]
            mean_to_train = np.mean(
                [np.array(DataStructs.BulkTanimotoSimilarity(te_fp[i], tr_fp)) for i in m],
                axis=0)
            order = np.argsort(mean_to_train)[::-1]
            picked = [int(tr_ok[j]) for j in order if np.isfinite(v[int(tr_ok[j])])][:K]
            if picked:
                lv[int(g)] = float(np.mean([v[p] for p in picked]))
                cov += len(m)
        local[c] = lv
        vals = np.array(list(lv.values()))
        print(f"{c:<34s} {len(lv):10d}/{len(set(lab))} {cov:13d} {vals.std():9.3f} "
              f"{vals.min():7.2f}-{vals.max():.2f}")
    print(f"  'level sd' is the spread of the per-cluster local level. If it is large relative")
    print(f"  to our prediction spread, a series-aware level carries information a single")
    print(f"  per-endpoint offset cannot. Test it on validation before believing it.")

    out = dict(
        n_anchors=len(counts), nn_median=float(np.median(sims)),
        clusters=cl_report, forward=fwd,
        local_level={c: {str(k): v for k, v in d.items()} for c, d in local.items()},
        cluster_of_test={str(i): int(lab[i]) for i in range(n)},
        consensus={str(g): dict(anchor_row=a, sim=cons_sim[g],
                                members=[int(x) for x in np.where(lab == g)[0]])
                   for g, a in cons.items()},
        group_sizes={str(k): v for k, v in sorted(sizes.items())},
        best_ranking=best_name, best_hits=best_hits, ranking_detail=detail,
        assignment=[{"test_row": int(i), "molecule": str(test["Molecule_Name"].iloc[int(i)]),
                     "anchor_row": int(a), "sim": float(s), "margin": float(s - r)}
                    for i, (a, s, r) in enumerate(zip(anchors, sims, runner_up))],
    )
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\n[series] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
