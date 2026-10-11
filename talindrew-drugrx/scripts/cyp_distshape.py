"""Phase 1 of the LeWAM transfer: point the new instruments at our own data.

`drugrx/distest.py` implements the two measurement objects LeWAM uses -- the Epps-Pulley
statistic against N(0,1) on random projections (its SIGReg loss) and Roy-Vetterli effective rank
(its anti-collapse readout). All 24 validation gates in `tests/test_distest.py` pass, including
two that caught real defects. This script is the first use on real data.

Why bother. `findings/07` spent GPU on an arm that did not vary the quantity its claim was about:
the Tobit-mean imputation moved the screen-inactive point mass's LOCATION (2.211 -> ~2.10) and
left its CONCENTRATION intact (imputed sd 0.030-0.084, 3-5% of s_y). Nothing in this lab measured
concentration, so nothing caught it until afterwards. That is the gap being closed here.

A trap the tests made explicit and this script must respect: the statistic STANDARDISES, so it
measures distributional SHAPE, not spread. Two Gaussians with sd 0.03 and sd 0.70 are
indistinguishable to it. Concentration is only meaningful relative to the labels an imputation
JOINS, so every number below is computed on the COMBINED training target vector -- the thing the
model actually fits -- never on a subset of imputed values on its own.

PRE-REGISTERED, written before the run:

1. **The ordering claim.** The screen-inactive point-mass share is 61 / 29 / 15 / 2 % for
   CYP2D6 / CYP1A2 / CYP2C9 / CYP3A4, known independently and never seen by the statistic. The
   Epps-Pulley `distance` on the combined training target must order the four isoforms the same
   way. PASS = exact rank agreement (Spearman 1.0 on four points).
   **If it fails, the instrument does not measure what I claim and Phase 2 does not start.**
2. **The contrast.** Dropping the pseudo-labels and keeping only real dose-response labels must
   reduce `distance` on every isoform, and by far the most on CYP2D6.
3. **Effective rank is reported under four normalisations and is NOT called a finding if the
   answer flips qualitatively between them.** For a fixed, sparse, hand-engineered matrix a low
   effective rank is evidence of REDUNDANCY, not of the representational collapse the SSL
   literature uses it for, and the Cauchy test in `tests/test_distest.py` shows it reads heavy
   tails as collapse -- which sparse count fingerprints have in abundance.

    python scripts/cyp_distshape.py                 # labels only, seconds
    python scripts/cyp_distshape.py --features      # adds the effective-rank half
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.cyp.data import PRIMARY, load  # noqa: E402
from drugrx.cyp.screen import screen_labels  # noqa: E402
from drugrx.distest import (  # noqa: E402
    POINT_MASS_DISTANCE,
    effective_rank,
    ep_null,
    epps_pulley,
)

FLOOR = 2.211
#: Known independently of the statistic, from findings/06. The ordering claim is tested against
#: these and nothing else.
KNOWN_SHARE = {"CYP2D6": 0.61, "CYP1A2": 0.29, "CYP2C9": 0.15, "CYP3A4": 0.02}


def short(c: str) -> str:
    return c.split("_")[0] if "_" in c else c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true",
                    help="also compute effective rank of the feature matrix (slower)")
    ap.add_argument("--null-reps", type=int, default=400)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/distshape.json"))
    args = ap.parse_args()

    t0 = time.time()
    frame = load(workers=8).frame.reset_index(drop=True)
    labs = screen_labels(frame)
    print(f"[shape] {len(frame)} compounds loaded in {time.time()-t0:.1f}s")
    print(f"[shape] point-mass ceiling of `distance` = {POINT_MASS_DISTANCE:.6f} "
          "(a pure point mass); a true Gaussian -> 0\n")

    rows = {}
    print("=" * 100)
    print("1. THE ORDERING CLAIM -- does the statistic recover the known point-mass share?")
    print("=" * 100)
    hdr = (f"   {'iso':8} {'n_train':>8} {'mass%':>6} {'known%':>7} {'distance':>10} "
           f"{'drc-only':>10} {'delta':>9} {'stat':>11} {'null q99':>9}")
    print(hdr)
    for c in PRIMARY:
        lab = labs[c]
        y = frame[c].to_numpy(float)
        kind = lab["kind"].to_numpy()
        drc = np.isfinite(y)
        ps = np.isin(kind, ("inactive", "weak", "active"))
        # Exactly the vector the model fits: real labels where they exist, pseudo-labels else.
        combined = np.where(drc, y, lab["y"].to_numpy(float))[drc | ps]
        drc_only = y[drc]
        inact = ((kind == "inactive") & ps & ~drc)
        n_lab = int((drc | ps).sum())
        share = float(inact.sum()) / max(n_lab, 1)

        d_all = epps_pulley(combined)
        d_drc = epps_pulley(drc_only)
        nul = ep_null(n_lab, reps=args.null_reps, seed=0)
        rows[c] = dict(n_train=n_lab, mass_share=share, known_share=KNOWN_SHARE[short(c)],
                       distance=d_all["distance"], stat=d_all["stat"],
                       drc_distance=d_drc["distance"], drc_n=int(drc.sum()),
                       null_q99=nul["q99"], null_mean=nul["mean"],
                       s_y=float(np.std(drc_only, ddof=1)))
        print(f"   {short(c):8} {n_lab:8d} {share*100:5.1f}% {KNOWN_SHARE[short(c)]*100:6.0f}% "
              f"{d_all['distance']:10.5f} {d_drc['distance']:10.5f} "
              f"{d_all['distance']-d_drc['distance']:+9.5f} {d_all['stat']:11.1f} "
              f"{nul['q99']:9.2f}")

    order = sorted(PRIMARY, key=lambda c: -rows[c]["distance"])
    by_share = sorted(PRIMARY, key=lambda c: -rows[c]["mass_share"])
    rho = float(spearmanr([rows[c]["mass_share"] for c in PRIMARY],
                          [rows[c]["distance"] for c in PRIMARY]).statistic)
    print(f"\n   by distance:    {' > '.join(short(c) for c in order)}")
    print(f"   by mass share:  {' > '.join(short(c) for c in by_share)}")
    print(f"   Spearman(share, distance) over the four isoforms = {rho:+.4f}")
    ok_order = order == by_share
    print(f"   CLAIM 1: {'PASS' if ok_order else 'FAIL'} -- exact rank agreement "
          f"{'is' if ok_order else 'is NOT'} achieved")
    ok_drc = all(rows[c]["distance"] > rows[c]["drc_distance"] for c in PRIMARY)
    worst = max(PRIMARY, key=lambda c: rows[c]["distance"] - rows[c]["drc_distance"])
    print(f"   CLAIM 2: {'PASS' if ok_drc else 'FAIL'} -- pseudo-labels raise `distance` on "
          f"{sum(rows[c]['distance'] > rows[c]['drc_distance'] for c in PRIMARY)}/4; "
          f"largest rise on {short(worst)}"
          f"{' (as predicted)' if short(worst) == 'CYP2D6' else ' (NOT CYP2D6 -- predicted wrong)'}")

    print("\n" + "=" * 100)
    print("2. THE CONCENTRATION AXIS -- `distance` as a point mass is diluted toward a draw.")
    print("   This is the dose-response curve findings/07 needed as a manipulation check and")
    print("   did not have. Read it as: how far does an imputation have to spread before the")
    print("   combined label distribution stops looking like it has an atom in it?")
    print("=" * 100)
    axis = {}
    rng = np.random.default_rng(0)
    for c in PRIMARY:
        lab = labs[c]
        y = frame[c].to_numpy(float)
        kind = lab["kind"].to_numpy()
        drc = np.isfinite(y)
        ps = np.isin(kind, ("inactive", "weak", "active"))
        inact = (kind == "inactive") & ps & ~drc
        keep = drc | ps
        base = np.where(drc, y, lab["y"].to_numpy(float))
        s_y = float(np.std(y[drc], ddof=1))
        if inact.sum() < 20:
            continue
        curve = []
        n_i = int(inact.sum())
        for frac in (0.0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
            v = base.copy()
            # A half-normal below the floor is the right SHAPE for left-censoring, but its mean
            # is -sigma*sqrt(2/pi) ~ -0.8*sigma, so using it raw moves LOCATION and spread
            # together -- the precise confound this axis exists to avoid, and the bug the first
            # run of this script actually had (it put opposite signs on CYP2D6 and CYP2C9).
            # Re-centre on FLOOR so only the spread varies.
            draw = -np.abs(rng.normal(0.0, max(frac * s_y, 1e-12), n_i))
            if frac > 0:
                draw = draw - draw.mean()
            v[inact] = FLOOR + draw
            curve.append((frac, epps_pulley(v[keep])["distance"],
                          float(np.std(v[inact], ddof=1)) / s_y,
                          float(np.mean(v[inact]))))
        axis[c] = curve
        line = "  ".join(f"{f:.2f}:{d:.4f}" for f, d, _, _ in curve)
        mean_drift = max(abs(m - FLOOR) for _, _, _, m in curve)
        print(f"   {short(c):8} s_y={s_y:.3f}  spread(xs_y) -> distance   {line}"
              f"   [mean drift {mean_drift:.1e}]")
    print("\n   frac is the imputation sd as a MULTIPLE of s_y, with the imputed MEAN pinned at")
    print("   2.211 so only concentration varies. findings/07's Tobit-mean arm sat at frac")
    print("   0.03-0.09 -- the far left of this axis, which is why it changed nothing about")
    print("   concentration. A draw-based arm lands near frac 1.0.")

    print("\n" + "=" * 100)
    print("2b. WHAT THE CHEAP SCALAR SAYS -- the control on the imported apparatus.")
    print("    The question Phase 2 needs answered is 'did the arm change the imputation's")
    print("    concentration relative to s_y'. That is sd(imputed)/s_y, one line of numpy.")
    print("    If the scalar separates the arms as well as the statistic does, the statistic is")
    print("    more apparatus than the question needs, and that is worth knowing before Phase 3")
    print("    builds anything on it.")
    print("=" * 100)
    print(f"    {'iso':8} {'arm':14} {'sd(imp)/s_y':>12} {'distance':>10}")
    for c in PRIMARY:
        if c not in axis:
            continue
        for frac, d, ratio, _m in axis[c]:
            if frac in (0.0, 0.05, 1.0):
                arm = {0.0: "point mass", 0.05: "tobit-mean~", 1.0: "draw~"}[frac]
                print(f"    {short(c):8} {arm:14} {ratio:12.4f} {d:10.5f}")
    print("\n    The scalar is monotone in the manipulation by construction and identical across")
    print("    isoforms; `distance` is not monotone and differs in SIGN between isoforms. For")
    print("    THIS job the scalar is the better instrument.")

    print("\n" + "=" * 100)
    print("2c. POST-HOC, AND LABELLED AS SUCH. Why claim 1 failed, and what does order.")
    print("=" * 100)
    print("    Claim 1 compared `distance` LEVELS across isoforms. That cannot work: the")
    print("    real-label distributions differ 6.7x in baseline non-Gaussianity on their own")
    print("    (drc-only column, 0.0025 to 0.0165), which swamps the 30x mass-share range. The")
    print("    level is total non-Gaussianity; a point mass is only one contributor to it.")
    print("    A WITHIN-isoform paired contrast cancels that baseline:")
    pair = {}
    for c in PRIMARY:
        if c in axis:
            pair[c] = axis[c][0][1] - axis[c][-1][1]
    po = sorted(pair, key=lambda c: -pair[c])
    print(f"\n    {'iso':8} {'mass%':>6} {'drop in distance, mass -> draw':>31} {'relative':>9}")
    for c in po:
        print(f"    {short(c):8} {rows[c]['mass_share']*100:5.1f}% {pair[c]:31.5f} "
              f"{pair[c]/axis[c][0][1]*100:8.1f}%")
    ok_pair = po == by_share
    print(f"\n    paired order:  {' > '.join(short(c) for c in po)}")
    print(f"    by mass share: {' > '.join(short(c) for c in by_share)}")
    print(f"    {'exact agreement' if ok_pair else 'NO agreement'}.")
    print("    THIS STATISTIC WAS CHOSEN AFTER SEEING THE DATA. Four points admit 24 orderings,")
    print("    so one exact match after trying two statistics is worth about p ~ 1/12 -- weak.")
    print("    It is a hypothesis for a future test, NOT a rescue of claim 1, which failed.")

    out = dict(rows=rows, axis={k: [list(p) for p in v] for k, v in axis.items()},
               point_mass_ceiling=POINT_MASS_DISTANCE, claim1_pass=bool(ok_order),
               claim2_pass=bool(ok_drc), spearman_share_distance=rho,
               paired_drop={k: float(v) for k, v in pair.items()},
               paired_order_matches_share=bool(ok_pair))

    if args.features:
        from drugrx.features import featurise
        X = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8,
                      counts=True, desc_mode="rdkit2d", extra_fps=True)
        print("\n" + "=" * 100)
        print(f"3. EFFECTIVE RANK of the feature matrix {X.shape}, four normalisations.")
        print("   Reported, not interpreted: a low value here is REDUNDANCY, and the Cauchy")
        print("   gate in the tests shows effective rank reads heavy tails as collapse.")
        print("=" * 100)
        fr = {}
        print(f"   {'normalisation':18} {'erank':>9} {'stable rank':>12} {'erank/d':>9}")
        for name in ("raw", "centred", "standardised", "row-L2"):
            A = np.asarray(X, float)
            if name == "centred":
                A = A - A.mean(axis=0, keepdims=True)
            elif name == "standardised":
                sd = A.std(axis=0, keepdims=True)
                A = (A - A.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0)
            elif name == "row-L2":
                nrm = np.linalg.norm(A, axis=1, keepdims=True)
                A = A / np.where(nrm > 0, nrm, 1.0)
            r = effective_rank(A, centre=False)
            fr[name] = r
            print(f"   {name:18} {r['erank']:9.2f} {r['stable_rank']:12.2f} "
                  f"{r['erank']/X.shape[1]:9.4f}")
        vals = [fr[k]["erank"] for k in fr]
        spread = max(vals) / max(min(vals), 1e-9)
        print(f"\n   spread across normalisations = {spread:.1f}x")
        print("   " + ("QUALITATIVELY STABLE -- reportable" if spread < 3 else
                       "FLIPS ACROSS NORMALISATIONS -- recorded as UNINFORMATIVE, per the "
                       "pre-registration"))
        out["effective_rank"] = fr
        out["erank_spread"] = spread
        out["feature_shape"] = list(X.shape)

    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\n   wrote {args.out}  ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
