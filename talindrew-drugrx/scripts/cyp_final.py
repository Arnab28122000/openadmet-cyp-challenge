"""Final submission model: external heads + blend + offsets, in one pass.

Assembles everything this track has evidence for, and nothing it does not:

* **External PubChem Veith heads** — +0.0029 +/- 0.0032 paired, 5/5 seeds better
  (sign test p=0.0625, paired t p=0.11). Small and never harmful, so included;
  it is 0.64% of current error and will not change a ranking on its own.
* **Blend of the net with gradient boosting** — the one intervention that
  survived nested selection (-0.0095 +/- 0.0044 over 5 folds, all negative,
  selection optimism -0.0013). The two are statistically tied individually, which
  is exactly when averaging decorrelated residuals pays.
* **Per-endpoint offsets** — tiny (+0.02 to +0.08 pIC50, worth ~0.003) and kept
  only because they are fitted on validation and cost nothing.

Deliberately NOT included, each tested and found inside noise on this task:
per-endpoint feature-arm selection (every arm gap inside its seed spread, and on
CYP2C9 the inner choice *lost* on the outer fold), protonation descriptors
(+0.0071 against a predicted 0.02, then -0.0080 when selected), and the
challenge's own auxiliary heads (0.4346 vs 0.4359).

The blend weight and the offsets are chosen on validation only; the blinded test
set is never consulted.
"""
from __future__ import annotations

import argparse, json, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, "/content"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from drugrx.cyp.data import PRIMARY, load
from drugrx.cyp.metric import METRIC_ID, macro_st_rae, st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split
from drugrx.features import featurise, standardise_descriptors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default="/content/out/cyp/external.parquet")
    ap.add_argument("--out", default="/content/out/cyp_final")
    # The architecture was hardcoded in the subprocess call below, so the submission
    # could not be rebuilt at a different depth without editing the file. Depth was
    # measured on 2026-10-06 as worth -0.0234 (t=-4.92, 5/5) **at a 3000-epoch
    # schedule** and a null at 400, so depth and schedule have to be settable together
    # or the configuration with evidence behind it is unreachable.
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--members", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3, help="net seeds, averaged")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    # Screen-derived interval labels for the compounds the DRC stage never measured, and
    # the validation surface that can see them. Defaults are round 3's: train on the
    # unselected population, select on V2. "--screen none --select-on v1" is the incumbent.
    ap.add_argument("--screen", choices=["none", "inactive", "all"], default="all")
    ap.add_argument("--select-on", choices=["v1", "v2"], default="v2")
    ap.add_argument("--feat-cache", default=None)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    data = load(workers=args.workers)
    frame, test = data.frame, data.test

    x = featurise(frame["smiles"].tolist(), workers=args.workers, cache=args.feat_cache,
                  counts=True, desc_mode="rdkit2d", extra_fps=True)
    xt = featurise(test["smiles"].tolist(), workers=args.workers, cache=args.feat_cache,
                   counts=True, desc_mode="rdkit2d", extra_fps=True)
    ext = pd.read_parquet(args.external)
    ext_cols = [c for c in ext.columns if c.startswith("veith_")]

    lo = frame[[f"{c}_conf_low" for c in PRIMARY]].to_numpy(float)
    hi = frame[[f"{c}_conf_high" for c in PRIMARY]].to_numpy(float)

    tr, va, rest = scaffold_split(frame["smiles"].tolist(), frac_train=0.85,
                                  frac_valid=0.10, seed=0)
    va = np.concatenate([va, rest]) if len(rest) else va
    _, st = standardise_descriptors(x[tr])
    xs, _ = standardise_descriptors(x, stats=st)
    xts, _ = standardise_descriptors(xt, stats=st)
    truth_v1 = {c: (frame[c].to_numpy()[va], lo[va, k], hi[va, k]) for k, c in enumerate(PRIMARY)}
    # V2 = V1 + held-out compounds the 50 uM screen found inactive, against the floor window.
    labs = screen_labels(frame)
    floor = allr_floor(frame)
    truth_v2 = {}
    for k, c in enumerate(PRIMARY):
        y1 = frame[c].to_numpy(float)[va]
        has = np.isfinite(y1)
        inact = labs[c]["kind"].to_numpy()[va] == "inactive"
        truth_v2[c] = (np.where(has, y1, np.where(inact, floor["y"], np.nan)),
                       np.where(has, lo[va, k], np.where(inact, floor["lo"], np.nan)),
                       np.where(has, hi[va, k], np.where(inact, floor["hi"], np.nan)))
    truth = truth_v1 if args.select_on == "v1" else truth_v2
    kinds = {"none": (), "inactive": ("inactive",),
             "all": ("inactive", "weak", "active")}[args.screen]
    print(f"[final] {len(tr)} train / {len(va)} valid / {len(test)} test; "
          f"screen={args.screen}, selecting on {args.select_on}", flush=True)

    # ── gradient boosting ────────────────────────────────────────────
    import xgboost as xgb
    val_gbm, test_gbm = {}, {}
    for c in PRIMARY:
        y = frame[c].to_numpy(float)
        kd = labs[c]["kind"].to_numpy()
        ps = np.isin(kd, kinds) & ~np.isfinite(y)
        # The GBM is a point learner, so a pseudo-row enters at the bin's median with a
        # reduced weight; the net is the arm that uses the interval directly.
        yy = np.where(np.isfinite(y), y, labs[c]["y"].to_numpy(float))
        w = np.where(np.isfinite(y), 1.0, np.where(kd == "inactive", 1.0, 0.5))
        a = tr[np.isfinite(y[tr]) | ps[tr]]
        b = va[np.isfinite(y[va]) | (ps[va] & (kd[va] == "inactive"))]
        m = xgb.XGBRegressor(n_estimators=1500, learning_rate=0.03, max_depth=7,
                             subsample=0.8, colsample_bytree=0.5, device="cuda",
                             tree_method="hist", verbosity=0, eval_metric="mae",
                             early_stopping_rounds=90)
        m.fit(xs[a], yy[a], sample_weight=w[a], eval_set=[(xs[b], yy[b])], verbose=False)
        val_gbm[c], test_gbm[c] = m.predict(xs[va]), m.predict(xts)
    print(f"[final] gbm valid MA-ST-RAE {macro_st_rae(val_gbm, truth)[0]:.4f}", flush=True)

    # ---- the net: ONE SUBPROCESS PER SEED --------------------------------
    # This loop used to train every seed in-process and free the GPU with
    # `del ...; empty_cache()`. That is precisely the remedy `cyp_net_one.py`'s
    # docstring records as having been tried three times and failed: the training
    # closures (`pred`, `loss_fn`, the vmapped gradient function) capture the
    # tensors and keep them alive, so peaks climb until a 40 GB card runs out. It
    # did exactly that here at 32 members -- seed 0 scored 0.6954 and seed 1 died
    # with 38.84 GB still allocated. The `del` did not even name `pred`.
    #
    # When the process exits the OS reclaims the device regardless, so each seed
    # now runs in its own process and one OOM costs one seed rather than the whole
    # submission. `cyp_net_one.py` already saves exactly what is needed here.
    import subprocess
    net_dir = out / "nets"
    net_dir.mkdir(parents=True, exist_ok=True)
    val_acc, test_acc = [], []
    for seed in range(args.seeds):
        tag = "final-net"
        done = net_dir / f"{tag}_s{seed}.json"
        if not done.exists():
            r = subprocess.run(
                [sys.executable, "/content/scripts/cyp_net_one.py",
                 "--tag", tag, "--seed", str(seed),
                 "--d-model", str(args.d_model), "--n-layers", str(args.n_layers),
                 "--dropout", "0.15",
                 "--members", str(args.members), "--epochs", str(args.epochs),
                 "--patience", str(args.patience),
                 "--external", args.external, "--aux-prefix", "veith_",
                 "--screen", args.screen, "--select-on", args.select_on,
                 "--ckpt-every", "0",
                 *(["--feat-cache", args.feat_cache] if args.feat_cache else []),
                 "--out-dir", str(net_dir)],
                cwd="/content", env={**os.environ, "PYTHONPATH": "/content"},
                capture_output=True, text=True)
            for ln in r.stdout.splitlines():
                if ln.startswith("[net]"):
                    print(ln, flush=True)
            if r.returncode != 0:
                print(f"[final] seed {seed} FAILED rc={r.returncode}: "
                      f"{(r.stderr or '')[-400:]}", flush=True)
                continue
        z = np.load(net_dir / f"{tag}_s{seed}.npz")
        # The subprocess ran its own scaffold split. It must be the same one, or
        # the averaged predictions are indexed against the wrong rows, silently.
        if not np.array_equal(z["va_index"], va):
            raise SystemExit(f"seed {seed}: validation split differs from the "
                             "assembler's; refusing to average misaligned rows")
        val_acc.append(z["val"])
        test_acc.append(z["test"])
        print(f"[final] net seed {seed}: "
              f"{json.loads(done.read_text())['valid']:.4f}", flush=True)

    if not val_acc:
        raise SystemExit("[final] no seed produced predictions")
    if len(val_acc) != args.seeds:
        print(f"[final] WARNING averaging {len(val_acc)} of {args.seeds} seeds; "
              "the ensembling gain was validated at the requested count", flush=True)

    val_net = {c: np.mean([v[:, k] for v in val_acc], axis=0) for k, c in enumerate(PRIMARY)}
    test_net = {c: np.mean([v[:, k] for v in test_acc], axis=0) for k, c in enumerate(PRIMARY)}
    print(f"[final] net valid MA-ST-RAE {macro_st_rae(val_net, truth)[0]:.4f}", flush=True)

    # ── blend weight on validation ───────────────────────────────────
    best_w, best_s = 1.0, None
    for w in np.arange(0.0, 1.01, 0.05):
        s, _ = macro_st_rae({c: w * val_gbm[c] + (1 - w) * val_net[c] for c in PRIMARY}, truth)
        if best_s is None or s < best_s:
            best_s, best_w = s, float(w)
    bv = {c: best_w * val_gbm[c] + (1 - best_w) * val_net[c] for c in PRIMARY}
    bt = {c: best_w * test_gbm[c] + (1 - best_w) * test_net[c] for c in PRIMARY}
    print(f"[final] blend w(gbm)={best_w:.2f} -> {best_s:.4f}", flush=True)

    # ── per-endpoint offsets on validation ───────────────────────────
    offsets = {}
    for k, c in enumerate(PRIMARY):
        y, l, h = truth[c]
        grid = np.arange(-1.0, 1.01, 0.02)
        scores = [st_rae(np.asarray(bv[c]) + d, y, l, h) for d in grid]
        offsets[c] = float(grid[int(np.nanargmin(scores))])
    final_val = {c: bv[c] + offsets[c] for c in PRIMARY}
    macro, per = macro_st_rae(final_val, truth)
    macro_v1, per_v1 = macro_st_rae(final_val, truth_v1)
    macro_v2, per_v2 = macro_st_rae(final_val, truth_v2)
    print(f"[final] FINAL valid MA-ST-RAE {macro:.4f} ({args.select_on}); "
          f"V1 {macro_v1:.4f}  V2 {macro_v2:.4f}", flush=True)
    for k in PRIMARY:
        print(f"    {k.split('_')[0]}: v1 {per_v1[k]:.4f}  v2 {per_v2[k]:.4f}", flush=True)

    sub = pd.DataFrame({"SMILES": test["SMILES"], "Molecule_Name": test["Molecule_Name"]})
    for c in PRIMARY:
        sub[c] = np.asarray(bt[c]) + offsets[c]
    sub.to_csv(out / "submission.csv", index=False)
    (out / "report.json").write_text(json.dumps(
        {"valid_macro": macro, "per_endpoint": per, "blend_w_gbm": best_w,
         "select_on": args.select_on, "screen": args.screen,
         "valid_macro_v1": macro_v1, "valid_macro_v2": macro_v2,
         "per_endpoint_v1": per_v1, "per_endpoint_v2": per_v2,
         # Which metric scored this. The blend weight and the offsets are both
         # CHOSEN on validation, so a submission built under the old denominator
         # optimised a mis-weighted objective -- the correction is not uniform per
         # endpoint (0.54-0.67), so those choices had to be re-fitted, not rescaled.
         "metric_id": METRIC_ID,
         "offsets": offsets, "net_seeds": args.seeds, "members": args.members,
         "d_model": args.d_model, "n_layers": args.n_layers,
         "epochs": args.epochs, "patience": args.patience,
         "external_heads": ext_cols,
         "pred_sd": {c: float(sub[c].std()) for c in PRIMARY}}, indent=2, default=float))
    print("\n[final] submission spread:", flush=True)
    for c in PRIMARY:
        print(f"    {c.split('_')[0]}: sd {sub[c].std():.3f} "
              f"range {sub[c].min():.2f}-{sub[c].max():.2f}", flush=True)
    print(f"[final] wrote {out/'submission.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
