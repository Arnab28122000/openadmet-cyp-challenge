r"""A held-out split that replicates the evaluation set's SELECTION, not its geometry.

The programme's standing bound is that no re-split of our own data can emulate the evaluation
set: an analog-expansion re-split reaches within-set similarity 0.446 against the test set's
0.665, because the organisers *purchased* analogs we do not hold at that density. That bound is
real and it is about **geometry**.

But the mechanism behind the per-endpoint results is not geometry. `cyp_recover_series.py`
confirmed the construction (75 series of 10, median cluster size exactly 10.0, forward purity
0.90 on CYP2C9 and 0.82 on CYP3A4 from the top-25 by direct pIC50) and the pattern it produces is:

| endpoint | anchors the expansion? | val -> test |
|---|---|---|
| CYP2C9 | yes, purity 0.90 | 0.5523 -> 0.5237 (**better**) |
| CYP3A4 | yes, purity 0.82 | 0.4758 -> 0.5735 |
| CYP1A2 | nominally; does not recover (sim 0.350) | 0.7582 -> 0.8842 |
| CYP2D6 | **no, absent from the selection** | 0.8938 -> **1.4577** |

The defect is **"compounds selected for endpoint X, scored on endpoint Y"**. That is a *selection*
effect, and selection can be replicated with the data we hold even though the similarity cannot.
Geometry sets how *hard* the set is; selection sets which endpoint it is hard *for*.

## Construction — the organisers' procedure, run on our own library

For a chosen set of **selector** endpoints:

1. take the top 25 compounds per selector endpoint by direct-inhibition pIC50 (the ranking
   confirmed for CYP2C9 and CYP3A4);
2. for each, take its 10 nearest ECFP4 neighbours **within our training frame**, excluding the
   hits themselves and anything already taken;
3. those analogs are the held-out set. **The hits stay in training**, exactly as in the real
   construction.

Then score *every* endpoint on that held-out set — including the ones that did not select it.
CYP2D6 is never a selector, which is the point.

## The control, which the comparison needs

The leaderboard numbers come from our neural ensemble; this scores a CPU surrogate
(`HistGradientBoostingRegressor` on the standard feature block), because retraining the ensemble
per split is not local work and the programme has already measured neural versus gradient
boosting as a **tie** (H6, 12/21, p = 0.180). A surrogate is therefore a legitimate stand-in for
measuring a *selection* effect, but it is not comparable to the board in absolute terms.

So the same surrogate is scored on a plain **scaffold split** as a control, and the claim is
about the **profile across endpoints** — whether the selection split reproduces the contrast
(CYP2D6 bad while CYP2C9/CYP3A4 stay good) that a scaffold split does not. Absolute levels are
reported and are not the criterion.

## Acceptance test, pre-registered in `DISCONNECTED-BENCH.md`

- **A4, the one that matters**: CYP2D6 is the worst endpoint, and its R² deficit against CYP3A4
  is at least half the board's. Board: −0.9509 against +0.5518, a contrast of **1.503**, so the
  split must show **>= 0.75**.
- A3: the per-endpoint profile is closer to the board's than the scaffold control is, on MAE, R²
  and rho.
- A2: geometry reported, not required to reach 0.665.
- A1 (disconnection from external sources) is not applicable here — this split is built from the
  challenge's own training data, so it is "disconnected" in the sense that matters for *model
  selection* (held out from fitting) but it is not an independent assay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, st_rae
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Our submission's published per-endpoint columns — the profile to be reproduced.
BOARD = {
    "CYP1A2_pIC50_direct_inhibition": dict(strae=0.8842, mae=1.0987, r2=0.0995, rho=0.6964),
    "CYP2C9_pIC50_direct_inhibition": dict(strae=0.5237, mae=0.5385, r2=0.5206, rho=0.7753),
    "CYP2D6_pIC50_direct_inhibition": dict(strae=1.4577, mae=1.7446, r2=-0.9509, rho=0.3302),
    "CYP3A4_pIC50_direct_inhibition": dict(strae=0.5735, mae=0.6181, r2=0.5518, rho=0.7612),
}
SELECTORS = {"confirmed": ("CYP2C9", "CYP3A4"),
             "stated": ("CYP1A2", "CYP2C9", "CYP3A4")}


def fps(smiles):
    out, ok = [], []
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is not None:
            out.append(GEN.GetFingerprint(m)); ok.append(i)
    return out, np.array(ok)


def selection_split(fr, selectors, n_hits=25, n_analogs=10):
    """The organisers' procedure, applied to our own library."""
    allfp, ok = fps(fr["smiles"].astype(str).tolist())
    pos = {int(r): i for i, r in enumerate(ok)}
    hits = []
    for cyp in selectors:
        v = fr[f"{cyp}_pIC50_direct_inhibition"].to_numpy(float)
        order = np.argsort(np.where(np.isfinite(v), v, -np.inf))[::-1]
        hits += [int(i) for i in order[:n_hits] if int(i) in pos]
    hits = list(dict.fromkeys(hits))
    hitset = set(hits)
    held, series = [], {}
    taken = set(hitset)
    for h in hits:
        s = np.array(DataStructs.BulkTanimotoSimilarity(allfp[pos[h]], allfp))
        order = np.argsort(s)[::-1]
        picked = []
        for j in order:
            row = int(ok[j])
            if row in taken:
                continue
            picked.append(row); taken.add(row)
            if len(picked) == n_analogs:
                break
        series[h] = picked
        held += picked
    return np.array(sorted(set(held))), hits, series


def geometry(fr, held, train_rows):
    allfp, ok = fps(fr["smiles"].astype(str).tolist())
    pos = {int(r): i for i, r in enumerate(ok)}
    hf = [allfp[pos[int(r)]] for r in held if int(r) in pos]
    tf = [allfp[pos[int(r)]] for r in train_rows if int(r) in pos]
    within = []
    for i, f in enumerate(hf):
        o = hf[:i] + hf[i + 1:]
        if o:
            within.append(max(DataStructs.BulkTanimotoSimilarity(f, o)))
    to_tr = [max(DataStructs.BulkTanimotoSimilarity(f, tf)) for f in hf]
    return float(np.median(within)), float(np.median(to_tr))


def score(fr, x, held, train_rows, label):
    """Fit a CPU surrogate per endpoint on `train_rows`, score on `held`."""
    _, stt = standardise_descriptors(x[train_rows])
    xs, _ = standardise_descriptors(x, stats=stt)
    out = {}
    for c in PRIMARY:
        y = fr[c].to_numpy(float)
        lo = fr[f"{c}_conf_low"].to_numpy(float)
        hi = fr[f"{c}_conf_high"].to_numpy(float)
        itr = np.array([r for r in train_rows if np.isfinite(y[r])])
        ite = np.array([r for r in held
                        if np.isfinite(y[r]) and np.isfinite(lo[r]) and np.isfinite(hi[r])])
        if len(itr) < 200 or len(ite) < 30:
            out[c] = dict(n=int(len(ite)), note="too few rows")
            continue
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_depth=None, min_samples_leaf=20,
                                          l2_regularization=1.0, random_state=0)
        m.fit(xs[itr], y[itr])
        p = m.predict(xs[ite])
        yt = y[ite]
        out[c] = dict(n=int(len(ite)),
                      strae=float(st_rae(p, yt, lo[ite], hi[ite])),
                      mae=float(np.mean(np.abs(p - yt))),
                      r2=float(1 - np.sum((yt - p) ** 2) / np.sum((yt - yt.mean()) ** 2)),
                      rho=float(stats.spearmanr(p, yt).statistic),
                      sd_p=float(p.std()), sd_y=float(yt.std()),
                      bias=float(p.mean() - yt.mean()))
        r = out[c]
        print(f"  {c[:34]:<34s} n={r['n']:4d}  ST-RAE {r['strae']:.4f}  MAE {r['mae']:.4f}  "
              f"R2 {r['r2']:+.4f}  rho {r['rho']:.4f}  bias {r['bias']:+.3f}")
    return out


def profile_distance(got, key):
    vals, ref = [], []
    for c in PRIMARY:
        if key in got.get(c, {}):
            vals.append(got[c][key]); ref.append(BOARD[c][key])
    if not vals:
        return float("nan")
    # Centre both, so a uniform level difference (surrogate vs ensemble) does not count
    v = np.array(vals) - np.mean(vals)
    r = np.array(ref) - np.mean(ref)
    return float(np.mean(np.abs(v - r)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selectors", choices=tuple(SELECTORS), default="confirmed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/cyp_cache/selection_split.json")
    args = ap.parse_args()

    fr = load(workers=args.workers).frame
    sel = SELECTORS[args.selectors]
    print(f"[sel] selector endpoints: {sel}  (CYP2D6 is deliberately never a selector)")
    x = featurise(fr["smiles"].astype(str).tolist(), workers=args.workers, counts=True,
                  desc_mode="rdkit2d", extra_fps=True, cache="data/featcache")
    print(f"[sel] features {x.shape}")

    held, hits, series = selection_split(fr, sel)
    train_rows = np.setdiff1d(np.arange(len(fr)), held)
    print(f"[sel] {len(hits)} hits (kept in TRAINING), {len(held)} analogs held out, "
          f"{len(train_rows)} training rows")
    w, t = geometry(fr, held, train_rows)
    print(f"[sel] geometry: within-set NN {w:.3f}  NN->train {t:.3f}   "
          f"(test set: 0.665 / 0.588; scaffold validation: 0.354 / 0.455)")

    print(f"\nSELECTION SPLIT — every endpoint scored on compounds chosen for {sel}")
    got = score(fr, x, held, train_rows, "selection")

    # Sized to match the selection split (470 held out of 6145, so ~7.6%) so the control
    # differs only in HOW the held-out set was chosen, not in how much training data remains.
    # `scaffold_split` requires frac_train + frac_valid < 1 and returns a third remainder.
    frac_v = len(held) / len(fr)
    tr, va, rest = scaffold_split(fr["smiles"].tolist(), frac_train=1.0 - frac_v - 0.01,
                                  frac_valid=frac_v, seed=0)
    tr = np.concatenate([tr, rest])
    print(f"\nSCAFFOLD CONTROL — same surrogate, same sizes, conventional split "
          f"({len(va)} held out, {len(tr)} training)")
    cw, ct = geometry(fr, va, tr)
    print(f"[ctl] geometry: within-set NN {cw:.3f}  NN->train {ct:.3f}")
    ctl = score(fr, x, va, tr, "scaffold")

    print(f"\n{'-' * 78}\nACCEPTANCE TEST against the board's per-endpoint profile")
    print(f"{'key':>6s} {'selection split':>16s} {'scaffold control':>17s} {'better?':>9s}")
    passes = {}
    for key in ("mae", "r2", "rho"):
        a, b = profile_distance(got, key), profile_distance(ctl, key)
        passes[key] = a < b
        print(f"{key:>6s} {a:16.4f} {b:17.4f} {'YES' if a < b else 'no':>9s}")
    print("  (profile distance = mean |centred difference| from the board's per-endpoint vector;")
    print("   centred so a uniform surrogate-vs-ensemble level offset does not count)")

    def contrast(d):
        try:
            return d["CYP3A4_pIC50_direct_inhibition"]["r2"] - d["CYP2D6_pIC50_direct_inhibition"]["r2"]
        except KeyError:
            return float("nan")

    board_c = BOARD["CYP3A4_pIC50_direct_inhibition"]["r2"] - BOARD["CYP2D6_pIC50_direct_inhibition"]["r2"]
    cs, cc = contrast(got), contrast(ctl)
    print(f"\nA4 — the CYP3A4-minus-CYP2D6 R2 contrast (the actual failure):")
    print(f"   board            {board_c:+.4f}")
    print(f"   selection split  {cs:+.4f}   {'PASS' if cs >= 0.75 else 'FAIL'} "
          f"(threshold 0.75 = half the board's)")
    print(f"   scaffold control {cc:+.4f}")
    worst = min(got, key=lambda c: -got[c].get("strae", -np.inf)) if got else None
    worst = max((c for c in got if "strae" in got[c]), key=lambda c: got[c]["strae"], default=None)
    print(f"   worst endpoint on the selection split: "
          f"{worst[:34] if worst else 'n/a'}  "
          f"{'(CYP2D6 as required)' if worst and 'CYP2D6' in worst else '(NOT CYP2D6)'}")

    verdict = (cs >= 0.75) and (worst is not None and "CYP2D6" in worst)
    print(f"\nVERDICT: {'A4 PASSES — this split reproduces the real failure' if verdict else 'A4 FAILS — the split does not reproduce the failure'}")
    if not verdict:
        print("  Then it is not a safer surface to tune on, and saying so is the result.")

    Path(args.out).write_text(json.dumps(
        dict(metric_id=METRIC_ID, selectors=list(sel), n_held=int(len(held)),
             n_hits=len(hits), geometry=dict(within=w, to_train=t),
             selection=got, scaffold=ctl,
             profile=dict({k: dict(selection=profile_distance(got, k),
                                   scaffold=profile_distance(ctl, k)) for k in
                           ("mae", "r2", "rho")}),
             contrast=dict(board=board_c, selection=cs, scaffold=cc),
             verdict=bool(verdict)), indent=1, default=float))
    print(f"\n[sel] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
