"""Tabular foundation models against the production GBM. Self-contained, shardable, VM-ready.

WHY THIS, AND WHY NOW. Model class is the only lever this programme has left: placement is
measured-exhausted (v9 took the vertex on all four axes, produced the best row we have had, and
the rank still fell 54 -> 59), and every representation lever measured here is 0.0000.

The literature points at exactly one untried family. Mol-JEPA (arXiv 2608.22642, Boehringer +
Brown) reports that on its OWN FDR-corrected paired Wilcoxon test, **TabICLv2 + AlvaDesc beat it
47% to 38%** across 23 ADME/potency datasets -- i.e. an off-the-shelf tabular foundation model on
descriptors beat a purpose-built molecular JEPA. Our data shape (n = 6145, d = 3091 tabular
descriptors, regression) is precisely that regime, and we have never tried it.

WHERE TO AIM, from disaggregating the board rather than reading its mean:

    cell      ST-RAE   board rank   real DRC labels   pseudo-label mass
    CYP2D6    0.5493       25            1493              60.5%
    CYP1A2    0.5369       57            1412              29.5%
    CYP2C9    0.4555       92            1285              14.9%
    CYP3A4    0.5513      137            2335               1.9%

**CYP3A4 is our worst cell relative to the field and it has the MOST real labels and almost no
pseudo-label contamination.** So it is the cleanest supervised problem of the four and the one we
are worst at -- which points away from the pseudo-label work entirely and at model class.
CYP3A4 is therefore the PRIMARY endpoint here; the others are reported but secondary.

And the transfer prior is endpoint-dependent, not a blanket discount. The programme's
"validation gains transfer at 17%" finding was MAJORLY CORRECTED: the gap was 74% one endpoint,
and board Spearman actually ROSE on CYP1A2 (+0.128) and CYP2C9 (+0.101). So a validation gain on
CYP3A4/CYP1A2/CYP2C9 is worth taking at close to face value; a CYP2D6 gain is not.

ARMS. All on identical splits and seeds, Spearman on held-out REAL dose-response labels only.

    lgbm_all     production: 8-member bootstrap LightGBM ensemble on all 3091 features
    lgbm_topk    the SAME learner on the top-k features. **This is the control that makes the
                 rest interpretable**: if it matches lgbm_all, our feature set is massively
                 redundant and any foundation model's feature limit costs nothing. If it loses
                 badly, then every k-limited arm below is handicapped and must be read that way.
    tabicl_topk  TabICLv2 regressor. Permissively licensed. Pre-trained on 2-100 columns and
                 documented to "degrade when going much beyond 100 features", hence topk.
    tabpfn_all   TabPFN-3.5 on all 3091 features (its limit is 20,000), **opt-in only**.
    tabpfn_topk  TabPFN-3.5 at matched k, so TabICL-vs-TabPFN is not confounded by feature count.

LICENSING, WHICH GATES WHAT THIS CAN BE USED FOR -- flagged rather than discovered later.
TabPFN-3.5's licence permits "internal testing and experimentation" but prohibits "production
deployment, commercial benchmarking, vendor evaluation, or product integration without a
commercial agreement". Talindrew is a commercial product and this challenge is a public
leaderboard, so the TabPFN arms are **INTERNAL MEASUREMENT ONLY**: they are gated behind
--allow-tabpfn, they are never written into a submission path, and a positive result from them is
information about the method family, not a shippable component. TabICL is permissively licensed
and carries no such restriction. If a tabular foundation model wins, TabICL is the one we could
actually use.

FEATURE SELECTION, and the leakage trap in it. top-k is chosen by LightGBM split gain fitted on
the TRAINING rows of that seed ONLY, refitted per seed. Selecting features on all rows -- or on
the pooled label vector -- leaks the validation labels into the feature set and inflates every
arm that uses it. Measured per seed so the selection is part of the cross-validation, not prior
to it.

PRE-REGISTERED, written before the run.

1. **The control first, and it can void the comparison.** If `lgbm_topk` loses more than 0.02
   macro rho to `lgbm_all`, then k features are not enough to carry this problem, and the
   TabICL arm is being handicapped rather than tested. In that case the headline comparison
   becomes `tabpfn_all` vs `lgbm_all` (same feature count) and the topk arms are reported as
   a feature-budget measurement only.
2. **The bar: a tabular foundation model must beat `lgbm_all` on CYP3A4 by >= +0.015 Spearman on
   >= 4 of 5 seeds** to be worth pursuing. That is ~3x the noise band of this harness and large
   enough that an endpoint-level board move is plausible. A smaller gain is recorded and not
   pursued (`feedback-a-magnitude-bar-is-not-a-significance-bar`).
3. **Honest expected value, stated first.** I expect the tabular foundation models to be
   COMPETITIVE but not clearly better, for two reasons. (a) The sceptical literature is
   consistent that descriptor + GBM is very hard to beat at n in the thousands -- Jiang et al.
   (J Cheminform 2021, 660 cites) found descriptor models beat graph models on average, and
   Yao & Duran-Frigola's pre-registered parity test put a compact descriptor portfolio at 0.762
   AUC against CheMeleon's 0.764. (b) TabICL is documented to degrade well beyond 100 features
   and our signal is spread across 3091 sparse columns, which is the opposite of the low-dimensional
   dense tabular data these models were pre-trained on. The result I consider most likely is
   `lgbm_all >= tabpfn_all > tabicl_topk ~= lgbm_topk`, and the most VALUABLE outcome is the
   control: learning how much of our 3091 features is load-bearing.
4. **Report per isoform always.** A macro gain driven by CYP2D6 (60.5% pseudo-label mass, zero
   dual-labelled compounds, and the endpoint that caused the 17% aggregate error) is a different
   result from one on CYP3A4. *Before explaining an aggregate, disaggregate it.*

    python cyp_tabfm.py --npz cyp_gpu.npz --seeds 0 1 2 3 4 --topk 100
    python cyp_tabfm.py --npz cyp_gpu.npz --seeds 0 1 2 3 4 --topk 100 --allow-tabpfn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
N_MEMBERS = 8
#: Verbatim from scripts/cyp_decide_vs_affine.py:gbm, asserted by --selftest.
GBM_KW = dict(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
              n_jobs=10, verbose=-1)
PRIMARY_ENDPOINT = "CYP3A4"


def gbm(seed):
    import lightgbm as lgb
    return lgb.LGBMRegressor(random_state=seed, **GBM_KW)


def selftest():
    import lightgbm as lgb
    want = dict(GBM_KW); want["random_state"] = 7
    got = gbm(7).get_params()
    bad = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
    assert not bad, f"learner drifted from production: {bad}"
    assert N_MEMBERS == 8, "ensemble size must match scripts/cyp_borda_decode.py:N_MEMBERS"
    print(f"[selftest] learner matches production ({lgb.__version__}); "
          f"{len(want)} params + ensemble size 8")


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def topk_features(X, y, w, idx, k, seed):
    """Top-k columns by LightGBM split gain, fitted on TRAINING rows only.

    Selecting on all rows would leak the validation labels into the feature set and inflate every
    arm that consumes it. Refitted per seed so selection sits inside the cross-validation.
    """
    m = gbm(seed).fit(X[idx], y[idx], sample_weight=w[idx])
    imp = np.asarray(m.booster_.feature_importance(importance_type="gain"), float)
    return np.argsort(-imp)[:k]


def fit_ensemble(X, y, w, idx, Xva, seed):
    """The production arm: N_MEMBERS bootstrap LightGBM fits, averaged."""
    rng = np.random.default_rng(1000 + seed)
    preds = []
    for mi in range(N_MEMBERS):
        bs = rng.choice(idx, size=idx.size, replace=True)
        preds.append(gbm(seed * 100 + mi).fit(
            X[bs], y[bs], sample_weight=w[bs]).predict(Xva))
    return np.vstack(preds).mean(axis=0)


def fit_tabicl(X, y, idx, Xva, seed):
    from tabicl import TabICLRegressor
    m = TabICLRegressor(random_state=seed)
    m.fit(X[idx], y[idx])
    return np.asarray(m.predict(Xva), float)


def fit_tabpfn(X, y, idx, Xva, seed, all_features):
    from tabpfn import TabPFNRegressor
    kw = dict(random_state=seed)
    if all_features:
        kw["ignore_pretraining_limits"] = True
    m = TabPFNRegressor(**kw)
    m.fit(X[idx], y[idx])
    return np.asarray(m.predict(Xva), float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="cyp_gpu.npz")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--ks", type=int, nargs="*", default=None,
                    help="sweep the feature budget for the k-limited arms. TabICL's 100-column "
                         "limit is documented DEGRADATION, not a hard cap, and the pre-registered "
                         "control showed 100 features lose 0.024 macro rho to the full set -- so "
                         "the budget is the binding constraint and must be swept, not assumed.")
    ap.add_argument("--allow-tabpfn", action="store_true",
                    help="enable the TabPFN arms. INTERNAL MEASUREMENT ONLY -- its licence "
                         "prohibits production deployment and commercial benchmarking.")
    ap.add_argument("--out", default="tabfm.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest()
    if args.selftest:
        return 0

    z = np.load(args.npz, allow_pickle=False)
    X = np.asarray(z["X"], float)
    isoforms = [str(c).split("_")[0] for c in z["isoforms"]]
    have = {}
    for name, mod in (("tabicl", "tabicl"), ("tabpfn", "tabpfn")):
        try:
            __import__(mod)
            have[name] = True
        except Exception as exc:
            have[name] = False
            print(f"[tabfm] {name} unavailable: {type(exc).__name__}")
    arms = ["lgbm_all", "lgbm_topk"]
    if have.get("tabicl"):
        arms.append("tabicl_topk")
    if have.get("tabpfn") and args.allow_tabpfn:
        arms += ["tabpfn_topk", "tabpfn_all"]
    print(f"[tabfm] X {X.shape}  arms {arms}  topk {args.topk}  "
          f"primary endpoint {PRIMARY_ENDPOINT}", flush=True)
    if not args.allow_tabpfn:
        print("[tabfm] TabPFN arms DISABLED (licence: internal testing only; pass "
              "--allow-tabpfn to enable for internal measurement)")

    rows = []
    for seed in args.seeds:
        tr, va = z[f"tr_{seed}"].astype(int), z[f"va_{seed}"].astype(int)
        print(f"\n[tabfm] seed {seed}", flush=True)
        for s in isoforms:
            y = z[f"y_{s}"].astype(float)
            kind = z[f"kind_{s}"]
            plab = z[f"plab_{s}"].astype(float)
            use = np.zeros(len(y), bool); use[tr] = True
            drc = use & np.isfinite(y)
            ps = use & np.isin(kind, ("inactive", "weak", "active"))
            yy = np.where(np.isfinite(y), y, plab)
            w = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
            idx = np.flatnonzero(drc | ps)
            okv = np.isfinite(y[va]); vv = va[okv]; yv = y[va][okv]
            if idx.size < 50 or okv.sum() < 60:
                continue
            ks = args.ks or [args.topk]
            colmap = {k: topk_features(X, yy, w, idx, k, seed) for k in ks}
            plan = []
            for arm in arms:
                if arm == "lgbm_all":
                    plan.append((arm, None))
                elif arm == "tabpfn_all":
                    plan.append((arm, None))
                else:
                    plan += [(arm, k) for k in ks]
            for arm, kk in plan:
                cols = colmap[kk] if kk is not None else None
                t0 = time.time()
                try:
                    if arm == "lgbm_all":
                        p = fit_ensemble(X, yy, w, idx, X[vv], seed)
                    elif arm == "lgbm_topk":
                        p = fit_ensemble(X[:, cols], yy, w, idx, X[vv][:, cols], seed)
                    elif arm == "tabicl_topk":
                        p = fit_tabicl(X[:, cols], yy, idx, X[vv][:, cols], seed)
                    elif arm == "tabpfn_topk":
                        p = fit_tabpfn(X[:, cols], yy, idx, X[vv][:, cols], seed, False)
                    elif arm == "tabpfn_all":
                        p = fit_tabpfn(X, yy, idx, X[vv], seed, True)
                    else:
                        continue
                    rho = spearman(p, yv)
                    err = None
                except Exception as exc:
                    rho, err = float("nan"), f"{type(exc).__name__}: {exc}"[:160]
                rec = dict(seed=seed, iso=s, arm=arm, rho=rho, n=int(okv.sum()),
                           topk=kk, secs=round(time.time() - t0, 1), error=err)
                rows.append(rec)
                print(f"   {s} {arm:12} k={str(kk):>5}: rho {rho:.4f}  {rec['secs']}s"
                      + (f"  ERROR {err}" if err else ""), flush=True)
            Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))
    if not rows:
        print("no rows"); return 1
    seeds = sorted({r["seed"] for r in rows})

    def cell(iso, arm, seed, k="any"):
        v = [r["rho"] for r in rows if r["iso"] == iso and r["arm"] == arm
             and r["seed"] == seed and (k == "any" or r["topk"] == k)]
        return v[0] if v else np.nan

    ks_seen = sorted({r["topk"] for r in rows if r["topk"] is not None})

    print("\n" + "=" * 92)
    print("1. THE CONTROL (pre-registered item 1): does top-k carry the problem?")
    print("=" * 92)
    d_all = []
    print(f"   feature budgets swept: {ks_seen}")
    print(f"\n   {'iso':8} " + "".join(f"{'k='+str(k):>12}" for k in ks_seen)
          + f"{'lgbm_all':>12}")
    for iso in isoforms:
        row = [np.nanmean([cell(iso, "lgbm_topk", s, k) for s in seeds]) for k in ks_seen]
        base = np.nanmean([cell(iso, "lgbm_all", s) for s in seeds])
        print(f"   {iso:8} " + "".join(f"{v:12.4f}" for v in row) + f"{base:12.4f}")
    print(f"\n   {'iso':8} " + "".join(f"{'tabicl k='+str(k):>14}" for k in ks_seen))
    for iso in isoforms:
        row = [np.nanmean([cell(iso, "tabicl_topk", s, k) for s in seeds]) for k in ks_seen]
        print(f"   {iso:8} " + "".join(f"{v:14.4f}" for v in row))
    print("\n   MATCHED-FEATURE contrast (tabicl_topk - lgbm_topk at the same k):")
    print(f"   {'iso':8} " + "".join(f"{'k='+str(k):>12}" for k in ks_seen))
    for iso in isoforms:
        row = []
        for k in ks_seen:
            d = np.array([cell(iso, "tabicl_topk", s, k) - cell(iso, "lgbm_topk", s, k)
                          for s in seeds])
            d = d[np.isfinite(d)]
            row.append(d.mean() if d.size else np.nan)
        print(f"   {iso:8} " + "".join(f"{v:+12.4f}" for v in row))
    for iso in isoforms:
        d = np.array([cell(iso, "lgbm_topk", s, ks_seen[-1]) - cell(iso, "lgbm_all", s)
                      for s in seeds])
        d = d[np.isfinite(d)]
        if d.size:
            d_all.append(d.mean())
            print(f"   {iso:8} lgbm_topk - lgbm_all = {d.mean():+.4f}")
    macro_ctrl = float(np.mean(d_all)) if d_all else float("nan")
    print(f"   {'MACRO':8} {macro_ctrl:+.4f}")
    if macro_ctrl < -0.02:
        print(f"   *** top-k LOSES {-macro_ctrl:.4f} macro rho: {args.topk} features do NOT")
        print("   *** carry this problem, so every k-limited arm below is HANDICAPPED, not")
        print("   *** tested. The headline becomes tabpfn_all vs lgbm_all at matched features.")
    else:
        print(f"   top-k is within 0.02 of all-features: our 3091 columns are largely")
        print(f"   redundant and a {args.topk}-feature budget costs little. The k-limited")
        print("   arms are a fair test.")

    print("\n" + "=" * 92)
    print(f"2. THE BAR: beat lgbm_all on {PRIMARY_ENDPOINT} by >= +0.015 rho on >= 4 of 5 seeds")
    print("=" * 92)
    summary = {}
    for arm in arms:
        if arm == "lgbm_all":
            continue
        print(f"\n   arm = {arm}")
        print(f"   {'iso':8} {'d rho':>9} {'wins':>8} {'mean rho':>9}")
        per = {}
        for iso in isoforms:
            d = np.array([cell(iso, arm, s) - cell(iso, "lgbm_all", s) for s in seeds])
            d = d[np.isfinite(d)]
            if not d.size:
                continue
            raw = np.array([cell(iso, arm, s) for s in seeds])
            per[iso] = (float(d.mean()), int((d > 0).sum()), d.size)
            print(f"   {iso:8} {d.mean():+9.4f} {int((d>0).sum()):4d}/{d.size:<3d} "
                  f"{np.nanmean(raw):9.4f}")
        if PRIMARY_ENDPOINT in per:
            m, wins, n = per[PRIMARY_ENDPOINT]
            ok = m >= 0.015 and wins >= max(4, int(0.8 * n))
            print(f"   -> {PRIMARY_ENDPOINT}: {m:+.4f} on {wins}/{n} seeds  "
                  f"{'CLEARS BAR -- pursue' if ok else 'below bar -- record, do not pursue'}")
        mac = float(np.mean([v[0] for v in per.values()])) if per else float("nan")
        print(f"   macro (secondary, disaggregate before reading): {mac:+.4f}")
        summary[arm] = dict(per={k: v for k, v in per.items()}, macro=mac)

    errs = [r for r in rows if r.get("error")]
    if errs:
        print(f"\n   {len(errs)} cell(s) errored. First: {errs[0]['arm']} / "
              f"{errs[0]['iso']} -> {errs[0]['error']}")
    Path(str(args.out).replace(".json", "_summary.json")).write_text(
        json.dumps(dict(summary=summary, control_macro=macro_ctrl, arms=arms,
                        topk=args.topk, primary=PRIMARY_ENDPOINT), indent=1, default=float))
    print(f"\n   wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
