"""Does the production early-stopping selection actually cost us anything?

THE DEFECT, verified. `scripts/cyp_final.py:114` builds XGBoost's early-stopping set as

    b = va[np.isfinite(y[va]) | (ps[va] & (kd[va] == "inactive"))]

with `eval_metric="mae"`. **MAE against a constant is minimised by predicting that constant**, and
the share of `b` that IS the single pseudo-label 2.2111, measured on the real scaffold split:

    CYP2D6  61.8% / 64.0% / 65.1%      CYP2C9  33.7% / 33.3% / 32.2%
    CYP1A2  45.6% / 46.5% / 46.7%      CYP3A4   2.8% /  2.4% /  5.4%

So early stopping rewards collapse onto 2.2111 precisely where the floor mass is largest. **That
ordering is exactly `findings/06`'s anomaly ordering**, and findings/06 attributed "the ensemble
members spuriously agree" to the point mass in TRAINING without anyone checking that it is also in
the MODEL-SELECTION signal.

BUT "it looks wrong" is not a result. A defect can be real and inert -- XGBoost may stop on the
real-label component anyway, or the extra rounds may cost nothing. So this measures it rather than
assuming, and it is a production change gated on the measurement.

ARMS, identical but for the early-stopping set:

    es_mixed   b = real-label rows PLUS pseudo-inactive rows   (the status quo)
    es_real    b = real-label rows only                        (the proposed fix)
    es_none    no early stopping, fixed n_estimators           (the control that shows how much
                                                                early stopping is doing at all)

Training data, weights, features, splits and every XGBoost hyperparameter are held identical.

PRE-REGISTERED, written before the run. The mechanism makes three predictions, and they fail
independently -- which is the point of stating all three.

1. **THE ROUND COUNT.** If the pseudo rows are steering selection, the chosen `best_iteration`
   must differ materially between `es_mixed` and `es_real` on CYP2D6 (65% constant) and barely on
   CYP3A4 (2.4%). **If the round counts are near-identical everywhere, the defect is INERT** and
   the rest of this script is measuring noise -- report that and stop.
2. **THE ORDERING.** d-rho (es_real - es_mixed) must be ordered by the constant share:
   **CYP2D6 > CYP1A2 > CYP2C9 > CYP3A4**, with CYP3A4 ~0. This is the falsifiable one: the
   ordering is predicted by a mechanism and the four shares were measured independently of any
   rho. A different ordering means the mechanism is wrong even if the fix helps.
3. **THE findings/06 LINK.** r(sigma_hat, p) on CYP2D6 is +0.324 under the status quo and is the
   ONLY reliably positive cell in the whole 4x3 grid of findings/06. If early-stopping collapse is
   a cause, **it must fall under `es_real`**. If it does not move, the defect is not a cause of
   that anomaly and `findings/06`'s training-side explanation stands alone.

4. **HONEST EXPECTED VALUE, stated first.** I expect the round counts to differ (prediction 1 to
   pass) and the rho effect to be SMALL -- +0.000 to +0.005 macro -- because XGBoost's MAE on a
   mixed set still falls when the real-label rows are fit better, so the constant rows dilute the
   signal rather than inverting it. I expect prediction 3 to be the interesting one: a measurable
   fall in r(sigma_hat,p) with little rho change would mean the defect distorts the UNCERTAINTY
   without costing accuracy, which is a correctness fix rather than a lever.

5. **This is a production-path change, so it is gated.** It is proposed for `cyp_final.py` only if
   prediction 2 holds AND macro d-rho >= 0 on >= 4 of 6 seeds. A correctness argument alone does
   not earn a change to the submission builder (`feedback-score-interventions-on-what-survives-postprocessing`).

    python cyp_earlystop_defect.py --npz cyp_gpu.npz --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

#: Verbatim from scripts/cyp_final.py. Asserted by --selftest.
XGB_KW = dict(n_estimators=1500, learning_rate=0.03, max_depth=7, subsample=0.8,
              colsample_bytree=0.5, tree_method="hist", verbosity=0,
              eval_metric="mae", early_stopping_rounds=90)
PSEUDO_W_FINAL = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
N_MEMBERS = 8
ARMS = ("es_mixed", "es_real", "es_none")


def selftest():
    import xgboost as xgb
    m = xgb.XGBRegressor(**XGB_KW)
    p = m.get_params()
    bad = {k: (v, p.get(k)) for k, v in XGB_KW.items() if p.get(k) != v}
    assert not bad, f"xgb config drifted from cyp_final.py: {bad}"
    print(f"[selftest] xgb config matches cyp_final.py ({xgb.__version__}); "
          f"{len(XGB_KW)} params")


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cyp_gpu.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--members", type=int, default=N_MEMBERS)
    ap.add_argument("--out", default="earlystop.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return 0
    import xgboost as xgb

    z = np.load(args.npz, allow_pickle=False)
    X = np.asarray(z["X"], np.float32)
    isoforms = [str(c).split("_")[0] for c in z["isoforms"]]
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(f"[es] X {X.shape}  device {dev}  members {args.members}", flush=True)

    rows = []
    for seed in args.seeds:
        tr, va = z[f"tr_{seed}"].astype(int), z[f"va_{seed}"].astype(int)
        print(f"\n[es] seed {seed}", flush=True)
        for s in isoforms:
            y = z[f"y_{s}"].astype(float)
            kd = z[f"kind_{s}"]
            plab = z[f"plab_{s}"].astype(float)
            ps = np.isin(kd, ("inactive", "weak", "active")) & ~np.isfinite(y)
            yy = np.where(np.isfinite(y), y, plab)
            w = np.where(np.isfinite(y), 1.0,
                         np.where(kd == "inactive", 1.0, 0.5))
            a_rows = tr[np.isfinite(y[tr]) | ps[tr]]
            b_mixed = va[np.isfinite(y[va]) | (ps[va] & (kd[va] == "inactive"))]
            b_real = va[np.isfinite(y[va])]
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if a_rows.size < 50 or okv.sum() < 60:
                continue
            const_share = float((~np.isfinite(y[b_mixed])
                                 & (kd[b_mixed] == "inactive")).mean())
            for arm in ARMS:
                t0 = time.time()
                preds, iters = [], []
                rng = np.random.default_rng(1000 + seed)
                for mi in range(args.members):
                    bs = rng.choice(a_rows, size=a_rows.size, replace=True)
                    kw = dict(XGB_KW); kw["random_state"] = seed * 100 + mi
                    kw["device"] = dev
                    if arm == "es_none":
                        kw.pop("early_stopping_rounds")
                        kw["n_estimators"] = 1500
                        m = xgb.XGBRegressor(**kw)
                        m.fit(X[bs], yy[bs], sample_weight=w[bs], verbose=False)
                        iters.append(1500)
                    else:
                        b = b_mixed if arm == "es_mixed" else b_real
                        m = xgb.XGBRegressor(**kw)
                        m.fit(X[bs], yy[bs], sample_weight=w[bs],
                              eval_set=[(X[b], yy[b])], verbose=False)
                        iters.append(int(getattr(m, "best_iteration", 1500) or 1500))
                    preds.append(m.predict(X[vv]))
                P = np.vstack(preds)
                p, sd = P.mean(axis=0), P.std(axis=0, ddof=1)
                rec = dict(seed=seed, iso=s, arm=arm, rho=spearman(p, yv),
                           r_sig_p=spearman(sd, p),
                           r_sig_err=spearman(sd, np.abs(yv - p)),
                           best_iter=float(np.mean(iters)),
                           const_share=const_share, n_eval=int(
                               b_mixed.size if arm == "es_mixed" else b_real.size),
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {s} {arm:9}: rho {rec['rho']:.4f}  r(sig,p) {rec['r_sig_p']:+.3f}"
                      f"  rounds {rec['best_iter']:6.0f}  n_eval {rec['n_eval']:4d}"
                      f"  const {const_share*100:4.1f}%  {rec['secs']}s", flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1
    seeds = sorted({r["seed"] for r in rows})

    def cell(iso, arm, seed, key="rho"):
        v = [r[key] for r in rows if r["iso"] == iso and r["arm"] == arm and r["seed"] == seed]
        return v[0] if v else np.nan

    print("\n" + "=" * 96)
    print("1. THE ROUND COUNT (item 1). If these are near-identical the defect is INERT.")
    print("=" * 96)
    print(f"   {'iso':8} {'const%':>7} {'rounds mixed':>13} {'rounds real':>12} {'delta':>8}")
    inert = True
    for iso in isoforms:
        cm = np.nanmean([cell(iso, "es_mixed", s, "best_iter") for s in seeds])
        cr = np.nanmean([cell(iso, "es_real", s, "best_iter") for s in seeds])
        cs = np.nanmean([cell(iso, "es_mixed", s, "const_share") for s in seeds]) * 100
        print(f"   {iso:8} {cs:6.1f}% {cm:13.0f} {cr:12.0f} {cr-cm:+8.0f}")
        if abs(cr - cm) > 0.1 * max(cm, 1):
            inert = False
    print(f"   -> {'INERT: selection barely moves; the rest is noise' if inert else
          'the selection DOES move -- continue'}")

    print("\n" + "=" * 96)
    print("2. THE ORDERING (item 2). d rho must follow the constant share:")
    print("   CYP2D6 > CYP1A2 > CYP2C9 > CYP3A4, with CYP3A4 ~0")
    print("=" * 96)
    dr, parts = {}, []
    print(f"   {'iso':8} {'const%':>7} {'d rho':>9} {'t':>7} {'wins':>8}")
    for iso in isoforms:
        d = np.array([cell(iso, "es_real", s) - cell(iso, "es_mixed", s) for s in seeds])
        d = d[np.isfinite(d)]
        if not d.size:
            continue
        t = d.mean()/(d.std(ddof=1)/np.sqrt(d.size)) if d.size > 1 and d.std(ddof=1) else 0.0
        cs = np.nanmean([cell(iso, "es_mixed", s, "const_share") for s in seeds]) * 100
        dr[iso] = d.mean(); parts.append(d)
        print(f"   {iso:8} {cs:6.1f}% {d.mean():+9.4f} {t:7.2f} {int((d>0).sum()):4d}/{d.size:<3d}")
    macro = np.array([np.mean([x[i] for x in parts]) for i in range(len(parts[0]))])
    print(f"   {'MACRO':8} {'':>7} {macro.mean():+9.4f}")
    want = ["CYP2D6", "CYP1A2", "CYP2C9", "CYP3A4"]
    got = sorted(dr, key=lambda k: -dr[k])
    print(f"\n   predicted order: {' > '.join(want)}")
    print(f"   observed order:  {' > '.join(got)}")
    ok_order = got == want
    print(f"   -> {'ORDERING HOLDS -- the mechanism is supported' if ok_order else
          'ORDERING FAILS -- the mechanism is wrong even if the fix helps'}")

    print("\n" + "=" * 96)
    print("3. THE findings/06 LINK (item 3). r(sigma,p) on CYP2D6 must FALL under es_real.")
    print("=" * 96)
    print(f"   {'iso':8} {'mixed':>9} {'real':>9} {'delta':>9} {'none':>9}")
    for iso in isoforms:
        a = np.nanmean([cell(iso, "es_mixed", s, "r_sig_p") for s in seeds])
        b = np.nanmean([cell(iso, "es_real", s, "r_sig_p") for s in seeds])
        c = np.nanmean([cell(iso, "es_none", s, "r_sig_p") for s in seeds])
        print(f"   {iso:8} {a:+9.3f} {b:+9.3f} {b-a:+9.3f} {c:+9.3f}")
    print("   CYP2D6 is the only reliably positive cell in findings/06's 4x3 grid (+0.324).")
    print("   A fall here means early-stopping collapse is a SECOND cause of that anomaly;")
    print("   no movement means findings/06's training-side explanation stands alone.")

    print("\n" + "=" * 96)
    print("4. THE GATE (item 5): propose the change to cyp_final.py only if the ordering holds")
    print("   AND macro d rho >= 0 on >= 4 of 6 seeds")
    print("=" * 96)
    wins = int((macro > 0).sum())
    gate = ok_order and wins >= 4
    print(f"   ordering {'holds' if ok_order else 'fails'}; macro d rho {macro.mean():+.4f} "
          f"on {wins}/{macro.size} seeds")
    print(f"   -> {'PROPOSE the change' if gate else 'DO NOT change production on this evidence'}")
    print("   A correctness argument alone does not earn a change to the submission builder.")
    Path(str(args.out).replace(".json", "_summary.json")).write_text(json.dumps(
        dict(order_ok=bool(ok_order), macro=float(macro.mean()), wins=wins,
             inert=bool(inert), d_rho={k: float(v) for k, v in dr.items()}),
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
