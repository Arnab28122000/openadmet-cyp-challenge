"""Shardable, self-contained runner for the Phase 8b arms. One seed list per machine.

WHY SELF-CONTAINED IS SAFE HERE, which is the only reason this exists rather than uploading the
repo. `feedback-a-control-must-run-the-production-command` forbids a reimplementation that could
drift from production. It cannot drift here, because the production learner is seven lines of
fully explicit LightGBM parameters with no hidden state:

    lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                      min_child_samples=20, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                      n_jobs=10, verbose=-1)

copied verbatim from `scripts/cyp_decide_vs_affine.py:gbm`, and every input (features, labels,
kinds, readouts, credible intervals, scaffold splits, the dual-labelled sample) is carried in the
npz written by `scratchpad/export_gpu.py` from the production loaders. `--selftest` asserts the
learner's parameters against the literal above so a future edit to one and not the other fails
loudly rather than silently measuring a different model.

ARMS -- identical to `scripts/cyp_rowwise_labels.py`, see that file for the pre-registration.
Local runs and shard runs must be poolable, so nothing here may differ but the seed list.

    python cyp_rowwise_shard.py --npz cyp_gpu.npz --seeds 3 4 5 --out shard_345.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FLOOR = 2.21113
SIGMA_ASSAY = 0.3
INACTIVE_R = 0.9
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
N_MEMBERS = 8
ARMS = ("base", "meanfloor", "rowlabel", "rowiso", "rowboth", "bank")
#: Phase 7's scalar-GLS arms, so one self-contained script serves every VM. Identical to
#: scripts/cyp_gls_weights.py; see that file's docstring for the pre-registration.
GLS_ARMS = ("base", "meanfloor", "gls", "gls_tight", "rowwise", "both")
#: Verbatim from scripts/cyp_decide_vs_affine.py:gbm. Asserted by --selftest.
GBM_KW = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
              n_jobs=10, verbose=-1)


def gbm(seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=seed, **GBM_KW)


def fit_loglin(r, y):
    A = np.vstack([np.ones_like(r), np.log10(r)]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0]), float(beta[1])


def apply_loglin(a, b, q):
    return a + b * np.log10(np.clip(np.nan_to_num(q, nan=INACTIVE_R), INACTIVE_R, None))


def fit_iso(r, y):
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(r, y)
    return lambda q: ir.predict(np.clip(np.nan_to_num(q, nan=INACTIVE_R), INACTIVE_R, None))


def fit_sigma_bins(r, y, a, b):
    res = np.abs(y - apply_loglin(a, b, r))
    edges = np.quantile(r, [1 / 3, 2 / 3])
    sd = []
    for t in range(3):
        m = np.digitize(r, edges) == t
        sd.append(float(res[m].std(ddof=1)) if m.sum() > 3 else float(res.std(ddof=1)))
    return np.maximum(np.array(sd), 1e-3), edges


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


#: Every constant that changes the measurement, with its production source. The first version of
#: this file checked ONLY the learner's parameters and hardcoded N_MEMBERS = 9 against a
#: production value of 8 -- so --selftest passed while the shard measured a nine-member ensemble.
#: It showed up as CYP1A2 base rho 0.6217 on the VM against 0.6181 locally. A selftest that
#: covers the learner but not the protocol is not a parity check.
PARITY = {
    "N_MEMBERS": (N_MEMBERS, "scripts/cyp_borda_decode.py:N_MEMBERS", 8),
    "FLOOR": (FLOOR, "drugrx/cyp/screen.py:allr_floor()['y']", 2.21113),
    "INACTIVE_R": (INACTIVE_R, "drugrx/cyp/screen.py:INACTIVE_R", 0.9),
    "PSEUDO_W_inactive": (PSEUDO_W["inactive"],
                          "scripts/cyp_decide_vs_affine.py:PSEUDO_W", 1.0),
    "PSEUDO_W_weak": (PSEUDO_W["weak"], "scripts/cyp_decide_vs_affine.py:PSEUDO_W", 0.5),
    "PSEUDO_W_active": (PSEUDO_W["active"], "scripts/cyp_decide_vs_affine.py:PSEUDO_W", 0.5),
}


def selftest():
    import lightgbm as lgb
    want = dict(GBM_KW); want["random_state"] = 7
    got = gbm(7).get_params()
    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    assert not bad, f"learner drifted from production: {bad}"
    drift = {k: (v, exp) for k, (v, _src, exp) in PARITY.items() if v != exp}
    assert not drift, f"PROTOCOL drifted from production: {drift}"
    print(f"[selftest] learner matches production ({lgb.__version__}); "
          f"{len(want)} parameters + {len(PARITY)} protocol constants checked")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cyp_gpu.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default="shard.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--family", choices=("rowwise", "gls"), default="rowwise",
                    help="rowwise = Phase 8b per-row labels; gls = Phase 7 scalar weights")
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return 0

    z = np.load(args.npz, allow_pickle=False)
    X = np.asarray(z["X"], float)
    isoforms = [str(c).split("_")[0] for c in z["isoforms"]]
    d_idx, d_r, d_y = z["dual_idx"].astype(int), z["dual_r"], z["dual_y"]
    print(f"[shard] X {X.shape}  isoforms {isoforms}  dual {d_r.size}  seeds {args.seeds}  "
          f"family {args.family}", flush=True)

    # Phase 7's derived scalar weights. The GLS ratio is w_floor over the MEAN DR WEIGHT,
    # mean(1/sig_i^2) -- NOT a ratio of sd summaries, which is inflated by the wide-CI tail.
    sig_floor = float(d_y.std(ddof=1))
    ci_all = []
    for s in isoforms:
        c = z[f"ci_{s}"].astype(float)
        yy_ = z[f"y_{s}"].astype(float)
        ci_all.append(c[np.isfinite(c) & np.isfinite(yy_)])
    ci_all = np.concatenate(ci_all)
    w_tight = (1.0 / sig_floor ** 2) / float(np.mean(1.0 / ci_all ** 2))
    w_loose = (1.0 / sig_floor ** 2) / float(np.mean(1.0 / (SIGMA_ASSAY ** 2 + ci_all ** 2)))
    ci_med = float(np.median(ci_all))
    print(f"[shard] sig_floor {sig_floor:.4f}  DR sd median {ci_med:.4f}  "
          f"derived w_floor/w_drc: tight {w_tight:.4f}  loose {w_loose:.4f}  "
          f"(PSEUDO_W inactive is {PSEUDO_W['inactive']})", flush=True)
    arms = GLS_ARMS if args.family == "gls" else ARMS

    rows = []
    for seed in args.seeds:
        tr, va = z[f"tr_{seed}"].astype(int), z[f"va_{seed}"].astype(int)
        in_tr = np.isin(d_idx, tr)
        if in_tr.sum() < 30:
            print(f"[shard] seed {seed}: only {int(in_tr.sum())} dual rows in split, skip")
            continue
        gr, gy = d_r[in_tr], d_y[in_tr]
        a_lin, b_lin = fit_loglin(gr, gy)
        g_iso = fit_iso(gr, gy)
        sd_bins, sd_edges = fit_sigma_bins(gr, gy, a_lin, b_lin)
        mean_floor = float(gy.mean())
        print(f"\n[shard] seed {seed}  dual-in-split {int(in_tr.sum())}  "
              f"g(r) = {a_lin:.4f} {b_lin:+.4f}*log10(r)  mean_floor {mean_floor:.4f}",
              flush=True)

        for s in isoforms:
            y = z[f"y_{s}"].astype(float)
            kind = z[f"kind_{s}"]
            plab = z[f"plab_{s}"].astype(float)
            r_row = z[f"r_{s}"].astype(float)
            use = np.zeros(len(y), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            base_y = np.where(np.isfinite(y), y, plab)
            idx = np.flatnonzero(drc | ps)
            inact = (kind == "inactive") & ps
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if idx.size < 50 or okv.sum() < 60 or inact.sum() < 20:
                continue
            rng = np.random.default_rng(1000 + seed)
            boots = [rng.choice(idx, size=idx.size, replace=True) for _ in range(N_MEMBERS)]
            brng = np.random.default_rng(7000 + seed)
            bank = []
            for _ in range(N_MEMBERS):
                bs = brng.choice(gr.size, size=gr.size, replace=True)
                bank.append(fit_loglin(gr[bs], gy[bs]))

            for arm in arms:
                t0 = time.time()
                w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
                shifts = np.zeros(0)
                preds = []
                for mi, bs in enumerate(boots):
                    yy = base_y.copy()
                    if arm in ("meanfloor", "both"):
                        yy[inact] = mean_floor
                    elif arm in ("rowlabel", "rowboth"):
                        yy[inact] = apply_loglin(a_lin, b_lin, r_row[inact])
                    elif arm == "rowiso":
                        yy[inact] = g_iso(r_row[inact])
                    elif arm == "bank":
                        yy[inact] = apply_loglin(*bank[mi], r_row[inact])
                    ww = w
                    if arm in ("gls", "both"):
                        ww = w.copy(); ww[inact] = w_loose
                    elif arm == "gls_tight":
                        ww = w.copy(); ww[inact] = w_tight
                    elif arm == "rowwise":
                        ww = w.copy(); ww[inact] = w_loose
                        ci_row = np.nan_to_num(z[f"ci_{s}"].astype(float), nan=ci_med)
                        pr = 1.0 / (SIGMA_ASSAY ** 2 + ci_row ** 2)
                        pr = pr / pr[drc].mean()
                        ww = np.where(drc, pr, ww)
                    if arm == "rowboth":
                        sg = sd_bins[np.digitize(
                            np.clip(np.nan_to_num(r_row[inact], nan=INACTIVE_R),
                                    INACTIVE_R, None), sd_edges)]
                        wi = 1.0 / (SIGMA_ASSAY ** 2 + sg ** 2)
                        ww = w.copy()
                        ww[inact] = wi / wi.mean()
                    if mi == 0:
                        shifts = yy[inact] - FLOOR
                    preds.append(gbm(seed * 100 + mi).fit(
                        X[bs], yy[bs], sample_weight=ww[bs]).predict(X[vv]))
                p = np.vstack(preds).mean(axis=0)
                wk = ww[idx]; wk = wk[wk > 0]
                rec = dict(seed=seed, iso=s, arm=arm, n=int(okv.sum()), family=args.family,
                           rho=spearman(p, yv), shift_mean=float(shifts.mean()),
                           shift_sd=float(shifts.std(ddof=1)),
                           ess=float(wk.sum() ** 2 / (wk * wk).sum()) if wk.size else 0.0,
                           w_inact=float(ww[inact].mean()),
                           secs=round(time.time() - t0, 1))
                rows.append(rec)
                print(f"   {s} {arm:10}: rho {rec['rho']:.4f}  shift "
                      f"{rec['shift_mean']:+.4f} +/- {rec['shift_sd']:.4f}  "
                      f"w_in {rec['w_inact']:.3f}  ESS {rec['ess']:.0f}  "
                      f"{rec['secs']}s", flush=True)
        Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    print(f"\n[shard] wrote {args.out}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
