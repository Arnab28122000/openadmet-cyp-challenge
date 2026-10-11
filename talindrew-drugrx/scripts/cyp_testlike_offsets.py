r"""Re-fit the per-endpoint offsets on the TEST-LIKE part of validation, and compare.

`cyp_shift_audit.py` established that the blinded test set is analog-expanded around
training actives: 84.3% of test compounds sit within Tanimoto 0.4 of a training active
against 15.1% of our validation compounds, and the test set's internal nearest-neighbour
similarity is 0.665 against validation's 0.354.

Our per-endpoint offsets -- the programme's largest confirmed effect, and one of the
quantities the leaderboard result suggests we over-fitted -- are chosen on the full
validation set. If validation and test differ in composition, those offsets are centred on
the wrong distribution.

This re-fits them on the subset of validation that **looks like the test set** (high
similarity to training actives) and reports how far they move. No new labels, no GPU: it uses
the stored validation predictions. A large move is a quantified mis-centring; a small one
means composition is not the mechanism and the search should go elsewhere.
"""
from __future__ import annotations

import glob
import json
import statistics as st

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import st_rae

RDLogger.DisableLog("rdApp.*")
GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
GRID = np.arange(-0.40, 0.4001, 0.02)


def best_offset(p, lo, hi):
    """Minimise the metric's numerator; the denominator never depends on p."""
    err = [float((np.clip(lo - (p + d), 0, None) + np.clip((p + d) - hi, 0, None)).sum())
           for d in GRID]
    return float(GRID[int(np.argmin(err))])


def main() -> int:
    d = load()
    fr = d.frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}
    smi = fr["smiles"].to_numpy()

    runs = sorted(glob.glob("data/cyp_cache/cyp_h12/h12-longdeep_s*.npz"))
    if not runs:
        print("no longdeep predictions found"); return 1
    idx = np.load(runs[0])["va_index"]

    def fp(s):
        m = Chem.MolFromSmiles(s)
        return GEN.GetFingerprint(m) if m is not None else None

    act = np.zeros(len(fr), bool)
    for c in PRIMARY:
        act |= np.nan_to_num(y[c], nan=-9) >= 5.5
    train_mask = np.ones(len(fr), bool); train_mask[idx] = False
    active_fp = [f for f in (fp(s) for s in smi[act & train_mask]) if f is not None]

    sims = []
    for s in smi[idx]:
        q = fp(s)
        sims.append(max(DataStructs.BulkTanimotoSimilarity(q, active_fp)) if q else np.nan)
    sims = np.array(sims, float)

    # The test set's profile: 84.3% above 0.4. Take validation's most test-like quartile.
    thresh = np.nanquantile(sims, 0.75)
    testlike = sims >= thresh
    print(f"validation n={len(idx)}; test-like subset = top quartile by similarity to "
          f"training actives (Tanimoto >= {thresh:.3f}), n={int(testlike.sum())}")
    print(f"  mean similarity: all {np.nanmean(sims):.3f}  test-like "
          f"{np.nanmean(sims[testlike]):.3f}  (test set itself: 0.536)\n")

    print(f"{'endpoint':10s} {'offset(all)':>12s} {'offset(test-like)':>18s} {'shift':>8s} "
          f"{'mean y(all)':>12s} {'mean y(TL)':>11s}")
    moves = []
    for i, c in enumerate(PRIMARY):
        pa = np.mean([np.load(r)["val"][:, i] for r in runs], axis=0)
        yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
        k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh)
        o_all = best_offset(pa[k], ll[k], hh[k])
        kt = k & testlike
        o_tl = best_offset(pa[kt], ll[kt], hh[kt]) if kt.sum() > 20 else float("nan")
        moves.append(o_tl - o_all)
        print(f"{c.split('_')[0]:10s} {o_all:+12.2f} {o_tl:+18.2f} {o_tl-o_all:+8.2f} "
              f"{np.nanmean(yy[k]):12.3f} {np.nanmean(yy[kt]):11.3f}")

    print(f"\nmean |shift| in the fitted offset: {np.nanmean(np.abs(moves)):.3f} pIC50")
    # What does using the test-like offsets cost or gain, scored on the test-like subset?
    print(f"\n{'scored on the test-like subset':34s} {'macro':>8s}")
    for label, use_tl in (("offsets fitted on ALL validation", False),
                          ("offsets fitted on the test-like part", True)):
        sc = []
        for i, c in enumerate(PRIMARY):
            pa = np.mean([np.load(r)["val"][:, i] for r in runs], axis=0)
            yy, ll, hh = y[c][idx], lo[c][idx], hi[c][idx]
            k = np.isfinite(yy) & np.isfinite(ll) & np.isfinite(hh)
            kt = k & testlike
            o = best_offset(pa[kt], ll[kt], hh[kt]) if use_tl else best_offset(pa[k], ll[k], hh[k])
            sc.append(st_rae(pa[kt] + o, yy[kt], ll[kt], hh[kt]))
        print(f"{label:34s} {np.nanmean(sc):8.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
