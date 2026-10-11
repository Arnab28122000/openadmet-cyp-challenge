"""Phase 8a: is a PER-ROW pseudo-label weight learnable at all? The gate before any GPU.

THE HYPOTHESIS (Arnab's, 2026-10-11). Stop using one scalar weight for every pseudo-labelled row.
Learn a weight MATRIX -- one weight per (compound, isoform) cell -- train a bank of such
weightings, and combine them at inference. Build the weighting into training with dynamic
learning rates.

WHY THIS IS A STRICTLY LARGER FAMILY THAN WHAT `findings/11` CLOSED, which is the reason it gets
measured rather than priced away. The scalar arm's measured ceiling (+0.0037 macro drho) bounds
SCALAR weightings only: 1 free parameter against 24,580 cells (6145 x 4). And there is a specific
signal the pipeline currently DISCARDS:

    screen_labels thresholds the continuous readout  r = 2^log2fc  at INACTIVE_R = 0.9.

A compound at r = 0.99 (no inhibition whatsoever) and one at r = 0.90 (barely over the line) get
the SAME pseudo-label 2.211 and the SAME weight 1.0. The within-bin position is thrown away. If
the pseudo-label error depends on r, a per-row weight is derivable, not merely learnable.

AND IT IS TESTABLE WITHOUT GPU, because the 303 dual-labelled compounds -- those carrying both a
dose-response curve and a 50 uM readout showing no inhibition -- are a LABELLED SAMPLE OF THE
PSEUDO-LABEL ERROR ITSELF. For each we know the assigned 2.211 and the true pIC50.

So this script answers, on real labels, the two questions that decide whether a learned weighting
can work at all:

  Q1. Does the pseudo-label error depend on the continuous readout r?
      -> if yes, a DERIVED per-row weight 1/(sigma_assay^2 + sigma^2(r)) exists for free, and it
         is the honest baseline any learned weighting must beat.
  Q2. Is the pseudo-label error predictable FROM THE FEATURES, out of fold?
      -> this is the quantity a learned weight-net must regress. If its cross-fitted signal is
         zero, no amount of GPU produces a weight function that generalises, and the bank-of-
         weightings idea collapses to the derived form.

PRE-REGISTERED, written before the run (`AGENTS.md` traps 7, 8, 10).

1. **Q1 PASS bar**: |Spearman(r, |error|)| >= 0.15 with p < 0.05, OR the sd of the error in the
   top-r tercile differs from the bottom-r tercile by >= 25% relative. Either establishes an
   r-dependence worth weighting on.
2. **Q2 PASS bar**: out-of-fold Spearman(predicted |error|, actual |error|) >= 0.20 on >= 4 of 5
   folds. **This is the gate on the LEARNED arm.** Below it, a weight-net is fitting noise, and
   the GPU budget goes to the derived form instead. Reported as oracle AND cross-fitted together
   (`feedback-compute-the-ceiling-before-building`) -- the in-fold number will be large and
   meaningless, and the gap is the whole story.
3. **The nuisance, named in advance** (trap 8): the 303 compounds are **250 CYP3A4, 42 CYP1A2,
   11 CYP2C9, 0 CYP2D6**. Any r-dependence measured here is dominated by CYP3A4 and must be
   transferred across a selection boundary to reach CYP2D6, which carries 60.5% of the mass and
   has **zero** compounds in this sample. So a PASS here is permission to build, not evidence the
   lever works on 2D6; and a per-isoform breakdown is printed so the transfer is visible.
4. **Expected outcome, stated first and honestly**: I expect Q1 to PASS weakly and Q2 to FAIL.
   The error being regressed is the residual of a *measured* assay against a *constant*, and
   `findings/11` showed its distribution is right-skewed with a hard floor -- most of its variance
   looks like assay noise, which is by definition unpredictable from structure. If Q2 fails, the
   honest version of the hypothesis is a derived r-dependent weight, not a learned one, and that
   is a cheaper and better answer than the one asked for.

    python scripts/cyp_weight_predictability.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cyp_decide_vs_affine import gbm, short  # noqa: E402

from drugrx.cyp.data import load  # noqa: E402
from drugrx.cyp.screen import CYPS, INACTIVE_R, _col  # noqa: E402

FLOOR = 2.21113


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/weight_predictability.json"))
    args = ap.parse_args()

    from drugrx.features import featurise

    frame = load(workers=8).frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    X = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    X = np.asarray(X, float)

    # Build the dual-labelled sample: one row per (compound, isoform) cell that has BOTH a
    # dose-response pIC50 and a readout showing no inhibition. These are the only rows in the
    # release where the pseudo-label error is observable.
    rows_i, rows_iso, rows_r, rows_y = [], [], [], []
    for c in CYPS:
        col = _col(c)
        r = 2.0 ** frame[f"log2fc_{c}"].to_numpy(float)
        y = frame[col].to_numpy(float)
        m = np.isfinite(y) & np.isfinite(r) & (r >= INACTIVE_R)
        for i in np.flatnonzero(m):
            rows_i.append(int(i)); rows_iso.append(c)
            rows_r.append(float(r[i])); rows_y.append(float(y[i]))
    idx = np.array(rows_i); iso = np.array(rows_iso)
    rr = np.array(rows_r); yy = np.array(rows_y)
    err = yy - FLOOR
    aerr = np.abs(err)
    n = idx.size

    print("=" * 96)
    print("PHASE 8a -- is a per-row pseudo-label weight learnable? The gate before any GPU.")
    print("=" * 96)
    print(f"\nDual-labelled cells: n = {n}")
    for c in CYPS:
        k = int((iso == c).sum())
        print(f"   {short(c):8} {k:4d}" + ("   <-- ZERO: its floor law is unidentified"
                                           if k == 0 else ""))
    print(f"\n   pseudo-label constant      {FLOOR:.4f}")
    print(f"   true y*: mean {yy.mean():.4f}  sd {yy.std(ddof=1):.4f}  "
          f"min {yy.min():.3f}  max {yy.max():.3f}")
    print(f"   signed error (y* - c): mean {err.mean():+.4f}  sd {err.std(ddof=1):.4f}")
    print(f"   |error|: mean {aerr.mean():.4f}  median {np.median(aerr):.4f}  "
          f"max {aerr.max():.3f}")
    print(f"   readout r: min {rr.min():.4f}  median {np.median(rr):.4f}  max {rr.max():.4f}")

    print("\n" + "=" * 96)
    print("Q1. Does the pseudo-label error depend on the CONTINUOUS readout r?")
    print("    (the pipeline thresholds r at 0.9 and discards the within-bin position)")
    print("=" * 96)
    s_a = spearmanr(rr, aerr)
    s_s = spearmanr(rr, err)
    print(f"   Spearman(r, |error|)      {s_a.statistic:+.4f}   p = {s_a.pvalue:.4f}")
    print(f"   Spearman(r, signed error) {s_s.statistic:+.4f}   p = {s_s.pvalue:.4f}")
    q = np.quantile(rr, [1 / 3, 2 / 3])
    terc = np.digitize(rr, q)
    print(f"\n   {'tercile of r':16} {'n':>5} {'r range':>18} {'mean y*':>9} "
          f"{'sd y*':>8} {'mean|err|':>10}")
    sds = []
    for t in (0, 1, 2):
        m = terc == t
        sds.append(float(yy[m].std(ddof=1)))
        print(f"   {['bottom', 'middle', 'top'][t]:16} {int(m.sum()):5d} "
              f"[{rr[m].min():.4f},{rr[m].max():.4f}] {yy[m].mean():9.4f} "
              f"{yy[m].std(ddof=1):8.4f} {aerr[m].mean():10.4f}")
    rel = abs(sds[2] - sds[0]) / max(min(sds[2], sds[0]), 1e-9)
    q1_corr = abs(s_a.statistic) >= 0.15 and s_a.pvalue < 0.05
    q1_terc = rel >= 0.25
    print(f"\n   top-vs-bottom tercile sd: {sds[2]:.4f} vs {sds[0]:.4f} -> "
          f"{rel*100:.1f}% relative difference")
    print(f"   Q1: {'PASS' if (q1_corr or q1_terc) else 'FAIL'} "
          f"(correlation bar {'met' if q1_corr else 'not met'}, "
          f"tercile bar {'met' if q1_terc else 'not met'})")

    print("\n" + "=" * 96)
    print("Q2. Is the error predictable FROM THE FEATURES, out of fold? THE GATE ON A LEARNED")
    print("    WEIGHT-NET. Oracle (in-fold) and cross-fitted are reported together; the in-fold")
    print("    number will be large and meaningless and the GAP is the whole story.")
    print("=" * 96)
    Xs = X[idx]
    # Group by compound so the same molecule cannot appear on both sides of a fold -- a compound
    # can contribute up to 4 cells, and ignoring that would leak.
    groups = idx
    uniq = np.unique(groups)
    splitter = (GroupKFold(n_splits=args.folds) if uniq.size >= args.folds * 2
                else KFold(n_splits=args.folds, shuffle=True, random_state=0))
    oof = np.full(n, np.nan)
    infold = []
    for k, (tr, te) in enumerate(splitter.split(Xs, aerr, groups=groups)):
        m = gbm(k).fit(Xs[tr], aerr[tr])
        oof[te] = m.predict(Xs[te])
        infold.append(float(spearmanr(m.predict(Xs[tr]), aerr[tr]).statistic))
    ok = np.isfinite(oof)
    per_fold = []
    for k, (tr, te) in enumerate(splitter.split(Xs, aerr, groups=groups)):
        v = spearmanr(oof[te], aerr[te]).statistic
        per_fold.append(float(v) if np.isfinite(v) else 0.0)
    pooled = float(spearmanr(oof[ok], aerr[ok]).statistic)
    ss_res = float(((aerr[ok] - oof[ok]) ** 2).sum())
    ss_tot = float(((aerr[ok] - aerr[ok].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    print(f"   ORACLE (in-fold)  Spearman: {np.mean(infold):+.4f}   <- fitted on its own rows")
    print(f"   CROSS-FITTED      Spearman: {pooled:+.4f}  (pooled)   R^2 = {r2:+.4f}")
    print(f"   per fold: " + "  ".join(f"{v:+.3f}" for v in per_fold))
    wins = int(sum(v >= 0.20 for v in per_fold))
    q2 = wins >= 4
    print(f"   folds at or above the +0.20 bar: {wins}/{len(per_fold)}")
    print(f"   Q2: {'PASS -- a learned weight-net is licensed' if q2 else 'FAIL'}")
    print(f"   oracle-to-cross-fitted gap: {np.mean(infold) - pooled:+.4f}")

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    if q1_corr or q1_terc:
        print("   Q1 PASSED: the error depends on the continuous readout, so a DERIVED per-row")
        print("   weight 1/(sigma_assay^2 + sigma^2(r)) exists and costs nothing. This is the")
        print("   honest baseline, and it is a real improvement on one scalar.")
    else:
        print("   Q1 FAILED: the error does not depend on r, so thresholding at 0.9 discards")
        print("   nothing, and the per-row story loses its derivable component.")
    if q2:
        print("   Q2 PASSED: the error carries out-of-fold predictable structure, so a learned")
        print("   weight-net has something to learn. GPU arms are licensed. It must still BEAT")
        print("   the derived weight, not merely beat the scalar.")
    else:
        print("   Q2 FAILED: the error is NOT predictable from structure out of fold. A learned")
        print("   weight-net would be fitting noise -- and note the oracle number above is large,")
        print("   which is exactly how this family produces a convincing in-sample lie")
        print("   (findings/04-05 closed the decode family on this same pattern).")
        print("   -> the GPU budget goes to the DERIVED weight, and the bank-of-weightings idea")
        print("      is restricted to derived weightings, which is still a real experiment.")
    print("\n   NUISANCE, named in advance: this sample is 250 CYP3A4 / 42 CYP1A2 / 11 CYP2C9 /")
    print("   0 CYP2D6. Anything measured here must cross a selection boundary to reach CYP2D6,")
    print("   which carries 60.5% of the mass. A PASS is permission to build, not proof.")

    Path(args.out).write_text(json.dumps(dict(
        n=int(n), per_iso={short(c): int((iso == c).sum()) for c in CYPS},
        rho_r_abserr=float(s_a.statistic), p_r_abserr=float(s_a.pvalue),
        rho_r_signed=float(s_s.statistic), tercile_sd=sds, tercile_rel=float(rel),
        oracle=float(np.mean(infold)), crossfit=pooled, crossfit_r2=float(r2),
        per_fold=per_fold, q1=bool(q1_corr or q1_terc), q2=bool(q2),
        err_mean=float(err.mean()), err_sd=float(err.std(ddof=1)),
    ), indent=1, default=float))
    print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
