"""Stage 3: do the docking features buy CYP3A4 any RANKING?

The bar was set in `docs/drugrx-training/DOCKING-PLAN.md` before any of this was built, and it
is deliberately high because this programme has produced three convincing false positives in a
day:

* **fail** below Δρ +0.01 -- the scale at which a hyperparameter sweep already reached t = 2.96
  and then evaporated on fresh seeds;
* **interesting** at +0.02 to +0.05;
* **+0.07 to +0.11** is what actually reaching the field's best on CYP3A4 requires.

CYP3A4 was chosen for stage 1 partly because its validation→board lift is **-0.005**: alone
among the four isoforms, a Δρ measured here transfers to the leaderboard roughly one-for-one.

Four arms, identical in every other respect:

    base        the incumbent 2D stack (ECFP4 counts + rdkit2d + Avalon + ErG, 3,091 dims)
    base+3d     plus the ligand-only 3D conformer descriptors
    base+dock   plus the docking features
    base+both   plus both

`base+3d` is the **control that decides what any gain means**. The 3D descriptors come free
with the conformers and are ligand-only, so they are identical across isoforms and cannot
carry isoform-specific information. If `base+3d` matches `base+dock`, then the gain is 3D
shape and the isoform-specificity argument that justified this whole build is wrong.

All three guards from `AGENTS.md` apply: paired seeds, Δρ reported beside ΔST-RAE, the NN ≥ 0.5
restricted contrast (the regime the blinded test set actually occupies), and -- if anything
clears the bar -- a confirmation on seeds never looked at.

    python scripts/cyp_dock_eval.py --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drugrx.cyp.data import load
from drugrx.cyp.matched import matched_spearman
from drugrx.cyp.metric import st_rae
from drugrx.cyp.screen import allr_floor, screen_labels
from drugrx.data.splits import scaffold_split

ROOT = Path(__file__).resolve().parents[1]
PSEUDO_W = {"inactive": 1.0, "weak": 0.5, "active": 0.5}
ISO = "CYP3A4"
COL = f"{ISO}_pIC50_direct_inhibition"
LIFT = -0.0052          # board rho - validation rho for CYP3A4
FIELD_BEST = 0.8741     # best CYP3A4 rho on the live board
DOCK_KEYS = ["best", "mean_top3", "spread", "n_poses",
             "fe_dist_best", "fe_dist_min", "n_heavy", "lig_eff"]


def load_dock(path: Path, n: int) -> tuple[np.ndarray, list[str]]:
    """Docking features aligned to frame order; missing rows stay NaN, never dropped."""
    X = np.full((n, len(DOCK_KEYS)), np.nan, dtype=np.float32)
    seen = 0
    for line in path.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        i = r.get("i")
        if i is None or i >= n or "err" in r:
            continue
        X[i] = [r.get(k, np.nan) for k in DOCK_KEYS]
        seen += 1
    print(f"[dock-eval] {seen}/{n} ligands have docking features "
          f"({100*seen/n:.1f}%)", flush=True)
    return X, list(DOCK_KEYS)


def fps(smiles):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
            for s in smiles]


def nn_sim(query, ref):
    from rdkit import DataStructs
    ref = [f for f in ref if f is not None]
    out = np.zeros(len(query))
    for i, f in enumerate(query):
        out[i] = max(DataStructs.BulkTanimotoSimilarity(f, ref)) if f is not None else 0.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    ap.add_argument("--confirm", action="store_true",
                    help="re-run on seeds 100-109, which no decision has ever looked at. "
                         "This exists because the GBM-tuning sweep reached t=2.96 on its "
                         "exploration seeds and then measured +0.0004 on fresh ones; the "
                         "guard only works if it is automatic rather than a printed "
                         "reminder at the bottom of the output.")
    ap.add_argument("--dock", default=str(ROOT / "data/cyp_cache/dock/dock_cyp3a4.jsonl"))
    ap.add_argument("--desc3d", default=str(ROOT / "data/cyp_cache/dock/desc3d.npz"))
    ap.add_argument("--out", default=str(ROOT / "data/cyp_cache/dock_eval.json"))
    args = ap.parse_args()
    if args.confirm:
        args.seeds = list(range(100, 110))
        args.out = args.out.replace(".json", "_confirm.json")
        print("[dock-eval] CONFIRMATION RUN on seeds 100-109 (never used for any decision)")

    import lightgbm as lgb
    from scipy.stats import spearmanr

    from drugrx.features import featurise

    data = load(workers=8)
    frame = data.frame.reset_index(drop=True)
    smiles = frame["smiles"].tolist()
    n = len(frame)
    labs, floor = screen_labels(frame), allr_floor(frame)
    base = featurise(smiles, cache=ROOT / "data/featcache", workers=8,
                     counts=True, desc_mode="rdkit2d", extra_fps=True)

    dock, dock_names = load_dock(Path(args.dock), n)
    z = np.load(args.desc3d, allow_pickle=True)
    d3 = np.asarray(z["x"], dtype=np.float32)[:n]
    print(f"[dock-eval] base {base.shape}  3d {d3.shape}  dock {dock.shape}", flush=True)

    blocks = {"base": base,
              "base+3d": np.hstack([base, d3]),
              "base+dock": np.hstack([base, dock]),
              "base+both": np.hstack([base, d3, dock])}

    y = frame[COL].to_numpy(float)
    lo = frame[f"{COL}_conf_low"].to_numpy(float)
    hi = frame[f"{COL}_conf_high"].to_numpy(float)
    kind = labs[COL]["kind"].to_numpy()
    yy = np.where(np.isfinite(y), y, labs[COL]["y"].to_numpy(float))
    allfp = fps(smiles)

    rows = []
    for seed in args.seeds:
        tr, va, rest = scaffold_split(smiles, frac_train=0.85, frac_valid=0.10, seed=seed)
        va = np.concatenate([va, rest]) if len(rest) else va
        use = np.zeros(n, bool); use[tr] = True
        drc = use & np.isfinite(y)
        ps = use & np.isin(kind, ("inactive", "weak", "active"))
        w0 = np.where(drc, 1.0, [PSEUDO_W.get(k, 0.0) for k in kind])
        idx = np.flatnonzero(drc | ps)
        v1 = np.isfinite(y[va])
        inact = kind[va] == "inactive"
        yv = np.where(v1, y[va], floor["y"])
        lv = np.where(v1, lo[va], floor["lo"])
        hv = np.where(v1, hi[va], floor["hi"])
        k2 = v1 | inact
        sim = nn_sim([allfp[i] for i in va], [allfp[i] for i in tr])
        near = v1 & (sim >= 0.50)
        print(f"\n[dock-eval] seed {seed}: {len(idx)} train rows, {int(v1.sum())} scored, "
              f"{int(near.sum())} at NN>=0.5", flush=True)

        row = {"seed": seed, "n": int(v1.sum()), "n_near": int(near.sum())}
        for arm, X in blocks.items():
            t0 = time.time()
            m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                                  min_child_samples=20, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.3, reg_lambda=1.0, random_state=seed,
                                  n_jobs=10, verbose=-1)
            m.fit(X[idx], yy[idx], sample_weight=w0[idx])
            p = m.predict(X[va])
            if arm == "base+dock":
                # Importance on the appended block only. Near-zero means LightGBM never
                # split on the docking features at all -- "no signal". Substantial
                # importance with no rho gain means the signal is real but already carried
                # by the 2D stack -- "redundant". Those are different conclusions and the
                # delta alone cannot tell them apart.
                imp = m.feature_importances_
                blk = imp[base.shape[1]:]
                row["dock_importance"] = {
                    "block_share": float(blk.sum() / max(imp.sum(), 1)),
                    "by_feature": {k: int(v) for k, v in zip(dock_names, blk)}}
            # Test-regime rho: our validation is 65% in NN 0.3-0.5, a band that is 4.8% of
            # the blinded test set, while 85% of the test sits in 0.5-0.7. Importance
            # weighting re-weights to the measured test histogram instead of discarding
            # three quarters of the compounds the way a hard NN>=0.5 cut does.
            mr, msd, mess = matched_spearman(p[v1], y[va][v1], sim[v1])
            row[arm] = {
                "rho": float(spearmanr(p[v1], y[va][v1]).statistic),
                "rho_matched": mr, "rho_matched_sd": msd, "matched_ess": mess,
                "v2": float(st_rae(p[k2], yv[k2], lv[k2], hv[k2])),
                "rho_near": (float(spearmanr(p[near], y[va][near]).statistic)
                             if near.sum() > 30 else float("nan")),
                "secs": round(time.time() - t0, 1)}
            print(f"   {arm:10} rho {row[arm]['rho']:.4f}  matched {row[arm]['rho_matched']:.4f}"
                  f"  V2 {row[arm]['v2']:.4f}  rho@NN {row[arm]['rho_near']:.4f}  "
                  f"[{row[arm]['secs']:.0f}s]", flush=True)
        rows.append(row)

    Path(args.out).write_text(json.dumps(rows, indent=1, default=float))

    print(f"\n=== paired against `base`, {len(rows)} seeds, CYP3A4 ===")
    print(f"{'arm':10} {'d rho':>21} {'t':>6} | {'d rho MATCHED':>21} {'t':>6} | "
          f"{'d rho @ NN>=0.5':>21} {'t':>6} | {'d V2':>12}")
    print("(`matched` re-weights validation to the test set's NN histogram. NOTE 2026-10-09 "
          "late:\n it is NOT 'the one to believe' -- on 12 fresh seeds it is no closer to the "
          "board than\n plain rho and costs 2.4x the per-seed sd. The docking verdict below is "
          "unaffected,\n because that arm failed on both metrics (+0.0024 raw, +0.0068 matched, "
          "bar +0.01).\n Kept unchanged so the recorded verdict stays reproducible. See "
          "drugrx/cyp/matched.py.)")
    verdicts = {}
    for arm in blocks:
        if arm == "base":
            continue
        line = f"{arm:10}"
        for key in ("rho", "rho_matched", "rho_near"):
            d = np.array([r[arm][key] - r["base"][key] for r in rows
                          if np.isfinite(r[arm][key]) and np.isfinite(r["base"][key])])
            if len(d) < 2:
                line += f"{'n/a':>30}"; continue
            se = d.std(ddof=1) / np.sqrt(len(d))
            if key == "rho_matched":
                verdicts[arm] = (d.mean(), se)
            line += f" {d.mean():+12.4f} +-{se:.4f} {d.mean()/se if se else 0:6.2f} |"
        dv = np.array([r[arm]["v2"] - r["base"]["v2"] for r in rows])
        print(line + f" {dv.mean():+12.4f}")

    print(f"\nbase rho {np.mean([r['base']['rho'] for r in rows]):.4f}   "
          f"field best {FIELD_BEST:.4f}   gap {FIELD_BEST - np.mean([r['base']['rho'] for r in rows]) + LIFT:+.4f}")
    ess = [r["base"].get("matched_ess", 0) for r in rows]
    print(f"\nmatched-rho effective sample size: {int(np.mean(ess))} of "
          f"{int(np.mean([r['n'] for r in rows]))} scored compounds (Kish). The matched "
          "estimate\nconcentrates weight on NN 0.5-0.7, so it has less power than the "
          "nominal n suggests.")
    print("\nVERDICT on MATCHED d rho, against the pre-registered bar:")
    for arm, (m, se) in verdicts.items():
        v = ("FAIL (noise scale)" if m < 0.01 else
             "INTERESTING" if m < 0.07 else "MATERIAL")
        print(f"   {arm:10} d rho {m:+.4f} +-{se:.4f}  -> {v}")
    if verdicts.get("base+dock") and verdicts.get("base+3d"):
        dd, _ = verdicts["base+dock"]; d3_, _ = verdicts["base+3d"]
        print(f"\nControl: base+3d {d3_:+.4f} vs base+dock {dd:+.4f}. If these are equal the")
        print("gain is 3D shape, NOT isoform-specific binding, and the build's premise fails.")
    shares = [r["dock_importance"]["block_share"] for r in rows if "dock_importance" in r]
    if shares:
        agg = {k: sum(r["dock_importance"]["by_feature"][k] for r in rows)
               for k in DOCK_KEYS}
        tot = max(sum(agg.values()), 1)
        print(f"\nDocking block took {100*np.mean(shares):.2f}% of total split importance.")
        print("  " + "  ".join(f"{k} {100*v/tot:.0f}%" for k, v in
                               sorted(agg.items(), key=lambda kv: -kv[1])))
        print("  Near-zero share with no gain = no signal. Substantial share with no gain =")
        print("  real but redundant against the 2D stack. Different conclusions.")

    if args.confirm:
        print("\nThis WAS the confirmation run. Its numbers are the ones to believe.")
    else:
        print("\nAnything clearing the bar must now be confirmed:  "
              "python scripts/cyp_dock_eval.py --confirm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
