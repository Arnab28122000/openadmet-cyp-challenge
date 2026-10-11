"""TDI classification track: CYP2D6_is_TDI and CYP3A4_is_TDI, scored by MCC.

The organisers' label is a deterministic rule on two arms (FAQ, read 2026-10-08):
positive if direct pIC50 > 4 and the TDI-arm shift exceeds 0.301 log; *inferred
positive* if direct <= 4 but the TDI arm exceeds 4.301; negative if direct > 4 with no
shift; *assigned negative* if both arms are below 4. The released `is_TDI` columns match
that rule on **every** compound where both arms exist (2D6 1,475/1,475; 3A4 2,175/2,175).

**The same selection problem as the regression track.** Released `is_TDI` labels exist
only where a DRC was run, i.e. on promoted compounds: CYP2D6 has 1,497 of them at a 21.6%
positive rate, and the 2,882 compounds the 50 µM screen found inactive carry none. Those
are, by the rule, assigned negatives (no signal in either arm), so they enter here as
negatives — the test set is scored on all 750 compounds, most of them CYP2D6-inactive.

Threshold. MCC is threshold-dependent and the test base rate is not the training one, so
the cut cannot be left at 0.5 — the MCC-optimal threshold here is 0.08–0.11, and 0.5 costs
0.13–0.15 MCC. Three rules are compared **nested** (chosen on a training-fold split, scored
on the outer validation fold), because choosing a threshold on the fold you then report is
the same selection error this programme has paid for twice:

* ``half`` — 0.5, the naive cut;
* ``oof`` — the MCC-argmax on out-of-fold predictions over the training fold;
* ``plugin`` — expected-MCC argmax on the *test* predictions after cross-fitted isotonic
  calibration. Treating calibrated probabilities as the truth's expectation, expected
  TP/FP/FN/TN are sums of p and 1-p on either side of t. It is the only rule that adapts
  to the blinded set's own base rate, and it needs calibration to mean anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drugrx.cyp.data import load
from drugrx.cyp.screen import screen_labels
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise

ROOT = Path(__file__).resolve().parents[1]
TDI = ("CYP2D6", "CYP3A4")


def mcc_from(tp, fp, fn, tn) -> float:
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def plugin_threshold(p: np.ndarray) -> tuple[float, float]:
    """Threshold maximising the MCC of the *expected* confusion matrix under p."""
    best_t, best = 0.5, -1.0
    for t in np.unique(np.round(p, 4)):
        on = p >= t
        tp, fp = p[on].sum(), (1 - p[on]).sum()
        fn, tn = p[~on].sum(), (1 - p[~on]).sum()
        m = mcc_from(tp, fp, fn, tn)
        if m > best:
            best, best_t = m, float(t)
    return best_t, best


def labels_for(frame: pd.DataFrame, labs: dict, c: str, assigned_neg: bool):
    y = frame[f"{c}_is_TDI"].to_numpy(float)
    kind = labs[f"{c}_pIC50_direct_inhibition"]["kind"].to_numpy()
    add = np.isnan(y) & (kind == "inactive") if assigned_neg else np.zeros(len(y), bool)
    y2 = y.copy()
    y2[add] = 0.0
    return y2, add


def oof_predict(x, y, idx, seed, n_est, folds=4):
    """Cross-fitted probabilities on `idx`, by scaffold-free random folds within it."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx))
    out = np.zeros(len(idx))
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        m = fit(x[idx[tr]], y[idx[tr]].astype(int), None, seed, n_est)
        out[te] = m.predict_proba(x[idx[te]])[:, 1]
    return out


def calibrator(p_oof, y_oof):
    """Isotonic map from raw score to probability, fitted out of fold."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_oof, y_oof)
    return iso


def fit(x, y, w, seed, n_est):
    m = lgb.LGBMClassifier(n_estimators=n_est, learning_rate=0.03, num_leaves=31,
                           min_child_samples=20, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                           n_jobs=10, verbose=-1)
    m.fit(x, y, sample_weight=w)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--final", action="store_true", help="fit on everything and write a submission")
    ap.add_argument("--threshold", choices=["oof", "plugin"], default="oof")
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/tdi"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    data = load(workers=8)
    frame, test = data.frame.reset_index(drop=True), data.test.reset_index(drop=True)
    labs = screen_labels(frame)
    opts = dict(counts=True, desc_mode="rdkit2d", extra_fps=True)
    x = featurise(frame["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)
    xt = featurise(test["smiles"].tolist(), cache=ROOT / "data/featcache", workers=8, **opts)

    res = []
    for c in TDI:
        for arm, aneg in (("released", False), ("released+assigned-neg", True)):
            y, add = labels_for(frame, labs, c, aneg)
            print(f"[tdi] {c} {arm}: n={int(np.isfinite(y).sum())} pos rate "
                  f"{np.nanmean(y):.3f} (assigned negatives added {int(add.sum())})", flush=True)
            for seed in args.seeds:
                tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                              frac_valid=0.10, seed=seed)
                va = np.concatenate([va, rest]) if len(rest) else va
                trl = tr[np.isfinite(y[tr])]
                m = fit(x[trl], y[trl].astype(int), None, seed, args.n_estimators)
                pv = m.predict_proba(x[va])[:, 1]
                # Thresholds chosen on the TRAINING fold only, then applied to validation.
                po = oof_predict(x, y, trl, seed, args.n_estimators)
                yo = y[trl].astype(int)
                t_oof = max(np.unique(np.round(po, 3)),
                            key=lambda t: matthews_corrcoef(yo, po >= t))
                iso = calibrator(po, yo)
                t_plug = plugin_threshold(iso.predict(pv))[0]
                # Score on V1 (released labels only) and V2 (+ held-out assigned negatives),
                # whatever the training arm -- the same two lenses as the regression track.
                y1, _ = labels_for(frame, labs, c, False)
                y2, _ = labels_for(frame, labs, c, True)
                for vname, yy in (("v1", y1[va]), ("v2", y2[va])):
                    k = np.isfinite(yy)
                    t_or = max(np.unique(np.round(pv[k], 3)),
                               key=lambda t: matthews_corrcoef(yy[k], pv[k] >= t))
                    res.append({"endpoint": c, "arm": arm, "seed": seed, "val": vname,
                                "auc": roc_auc_score(yy[k], pv[k]),
                                "mcc_half": matthews_corrcoef(yy[k], pv[k] >= 0.5),
                                "mcc_oof": matthews_corrcoef(yy[k], pv[k] >= t_oof),
                                "mcc_plugin": matthews_corrcoef(yy[k], iso.predict(pv[k]) >= t_plug),
                                "mcc_oracle": matthews_corrcoef(yy[k], pv[k] >= t_or),
                                "t_oof": float(t_oof), "t_plugin": float(t_plug),
                                "t_oracle": float(t_or),
                                "pos_rate": float(np.mean(yy[k]))})
    df = pd.DataFrame(res)
    pd.set_option("display.width", 220)
    print(df.groupby(["endpoint", "arm", "val"])[
        ["auc", "mcc_half", "mcc_oof", "mcc_plugin", "mcc_oracle",
         "t_oof", "t_plugin", "t_oracle", "pos_rate"]].mean().round(3).to_string())
    df.to_json(out / "cv.json", orient="records", indent=1)

    if args.final:
        sub = test[["SMILES", "Molecule_Name"]].copy()
        report = {}
        for c in TDI:
            y, _ = labels_for(frame, labs, c, True)
            idx = np.flatnonzero(np.isfinite(y))
            ps, pos, ts = [], [], []
            for seed in args.seeds:
                ps.append(fit(x[idx], y[idx].astype(int), None, seed, args.n_estimators)
                          .predict_proba(xt)[:, 1])
                po = oof_predict(x, y, idx, seed, args.n_estimators)
                pos.append(po)
                yo = y[idx].astype(int)
                ts.append(max(np.unique(np.round(po, 3)),
                              key=lambda t: matthews_corrcoef(yo, po >= t)))
            p, po = np.mean(ps, axis=0), np.mean(pos, axis=0)
            yo = y[idx].astype(int)
            t_oof = float(np.median(ts))
            iso = calibrator(po, yo)
            t_plug, em = plugin_threshold(iso.predict(p))
            t, scale = (t_oof, p) if args.threshold == "oof" else (t_plug, iso.predict(p))
            sub[f"{c}_is_TDI"] = scale >= t
            report[c] = {"rule": args.threshold, "threshold": float(t),
                         "t_oof": t_oof, "t_plugin": float(t_plug), "expected_mcc": em,
                         "pred_pos_rate": float((scale >= t).mean()),
                         "train_pos_rate": float(yo.mean()), "mean_p": float(p.mean())}
            np.save(out / f"test_p_{c}.npy", p)
            print(f"[tdi] {c}: rule={args.threshold} t={t:.3f} (oof {t_oof:.3f}, plugin "
                  f"{t_plug:.3f}); predicted positive rate {(scale >= t).mean():.3f}, "
                  f"train {yo.mean():.3f}")
        sub.to_csv(out / "submission_tdi.csv", index=False)
        (out / "final.json").write_text(json.dumps(report, indent=1))
        print(f"[tdi] wrote {out / 'submission_tdi.csv'} ({len(sub)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
