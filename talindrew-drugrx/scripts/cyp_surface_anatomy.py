"""Why do the `near` / `matched` surfaces move rho, if rank error is flat in similarity?

`cyp_rank_error_anatomy.py` measured that normalised rank error is **flat across the
similarity band** -- 0.152/0.141/0.150 at NN 0.3-0.5 against 0.138/0.163/0.144 at 0.5-0.7, on
CYP3A4 over three seeds. If per-compound rank error does not depend on NN similarity, then
importance-weighting validation compounds by similarity cannot make the subset *easier* in the
sense that matters, and should only shrink the effective sample size.

Yet `cyp_split_match_board.py` shows the `near` surface moving rho **consistently and in
opposite directions by isoform**: on all three of seeds 0-2, CYP1A2's rho goes UP against
`all` (0.605->0.777, 0.495->0.487, 0.549->0.537) while CYP2C9's goes DOWN (0.675->0.544,
0.655->0.597, 0.644->0.624). A pure variance effect would not pick a direction per isoform.

The remaining candidate is the **label distribution**, not the error. Spearman is a function of
the joint ranking, so a subset whose true labels are compressed into a narrower range scores a
lower rho at identical per-compound rank error -- and conversely. This script measures that
directly: for the same splits, the same NEAR cut and the same histogram weights, it reports the
spread of the TRUE labels on each surface, and the Kish effective sample size the matched
weighting actually leaves.

It fits no model, so it is fast and cannot be confounded by model noise.

Surface helpers are **imported** from `cyp_split_match_board` rather than copied, so the
subsets here are provably the same subsets scored there.

    python scripts/cyp_surface_anatomy.py --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_split_match_board import NEAR, fps, hist_weights, nn_sim, short  # noqa: E402

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.data.splits import scaffold_split  # noqa: E402


def kish_ess(w: np.ndarray) -> float:
    """Effective sample size of a weighted mean: (sum w)^2 / sum w^2."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.square(w).sum()) if np.any(w) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/surface_anatomy.json"))
    args = ap.parse_args()

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    all_fp = fps(smiles)
    test_fp = fps(data.test["SMILES"].tolist())
    edges = np.linspace(0.0, 1.0, 21)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        ref = [all_fp[i] for i in tr]
        sim_va = nn_sim([all_fp[i] for i in va], ref)
        sim_te = nn_sim(test_fp, ref)
        w_match = hist_weights(sim_va, sim_te, edges)

        for c in PRIMARY:
            y = frame[c].to_numpy(float)
            keep = np.isfinite(y[va])
            if keep.sum() < 30:
                continue
            yv, sv, wv = y[va][keep], sim_va[keep], w_match[keep]
            near = sv >= NEAR
            if near.sum() <= 30:
                continue
            # weighted sd of the true labels under the matched weights
            mu_w = float(np.average(yv, weights=wv))
            sd_w = float(np.sqrt(np.average((yv - mu_w) ** 2, weights=wv)))
            rows.append(dict(
                seed=seed, iso=c,
                n_all=int(keep.sum()), n_near=int(near.sum()),
                ess_match=round(kish_ess(wv), 1),
                sd_all=round(float(yv.std(ddof=1)), 4),
                sd_near=round(float(yv[near].std(ddof=1)), 4),
                sd_match=round(sd_w, 4),
                iqr_all=round(float(np.subtract(*np.percentile(yv, [75, 25]))), 4),
                iqr_near=round(float(np.subtract(*np.percentile(yv[near], [75, 25]))), 4),
                frac_floor_all=round(float(np.mean(yv <= 4.0)), 4),
                frac_floor_near=round(float(np.mean(yv[near] <= 4.0)), 4),
            ))

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print("\n=== true-label spread on each surface (no model fitted) ===")
    print(f"{'iso':9} {'n_all':>6} {'n_near':>7} {'ESS_m':>7}   "
          f"{'sd all':>7} {'sd near':>8} {'sd match':>9}   {'d sd near-all':>14}   "
          f"{'floor all':>10} {'floor near':>11}")
    for c in PRIMARY:
        rs = [r for r in rows if r["iso"] == c]
        if not rs:
            continue
        g = lambda k: float(np.mean([r[k] for r in rs]))  # noqa: E731
        print(f"{short(c):9} {g('n_all'):6.0f} {g('n_near'):7.0f} {g('ess_match'):7.0f}   "
              f"{g('sd_all'):7.3f} {g('sd_near'):8.3f} {g('sd_match'):9.3f}   "
              f"{g('sd_near') - g('sd_all'):+14.3f}   "
              f"{g('frac_floor_all'):10.3f} {g('frac_floor_near'):11.3f}")

    print("\nESS as a fraction of n (matched weighting): "
          + ", ".join(f"{short(c)} {np.mean([r['ess_match'] for r in rows if r['iso'] == c]) / np.mean([r['n_all'] for r in rows if r['iso'] == c]):.2f}"
                      for c in PRIMARY if any(r["iso"] == c for r in rows)))
    print("\nPer seed, d sd (near - all), to show the direction is not a seed accident:")
    for c in PRIMARY:
        rs = sorted((r for r in rows if r["iso"] == c), key=lambda r: r["seed"])
        if not rs:
            continue
        print(f"  {short(c):9} " + " ".join(f"s{r['seed']}:{r['sd_near'] - r['sd_all']:+.3f}" for r in rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
