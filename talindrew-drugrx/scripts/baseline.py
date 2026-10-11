"""The baseline every deep model in this programme has to beat.

ECFP4 fingerprints + LightGBM, on TDC's official scaffold split. This is the
stubborn, unglamorous baseline that a large share of published molecular deep
learning quietly loses to, which is exactly why it is run first and committed
before anything is trained on a GPU (see docs/drugrx-training/AGENTS.md).

Runs on a laptop CPU in minutes. No GPU, no Colab, no compute units.

    python scripts/baseline.py                 # every ADMET endpoint
    python scripts/baseline.py --tasks caco2 herg
    python scripts/baseline.py --seeds 5       # our scaffold splits, for an error bar

Writes docs/drugrx-training/runs/baseline/results.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drugrx.data.splits import scaffold_overlap, scaffold_split  # noqa: E402
from drugrx.data.tdc import ADMET_TASKS, TASK_BY_KEY, AdmetTask, load_task  # noqa: E402

OUT_DIR = ROOT.parent / "docs" / "drugrx-training" / "runs" / "baseline"


#: The two baseline arms. `binary-only` is what was committed first and is kept
#: so the comparison is reproducible; `count-rdkit2d` is the configuration the
#: literature reports topping or tying half of TDC's ADMET tasks, and is the
#: real bar a deep model has to clear.
ARMS: dict[str, dict] = {
    "binary-only": {"counts": False, "desc_mode": "none"},
    "count-rdkit2d": {"counts": True, "desc_mode": "rdkit2d"},
    # MapLight's recipe: ours plus Avalon (a different substructure
    # enumeration) and ErG (pharmacophore topology).
    "maplight": {"counts": True, "desc_mode": "rdkit2d", "extra_fps": True},
}


def featurise_arm(smiles: list[str], arm: str, cache: Path | None = None,
                  workers: int | None = None) -> np.ndarray:
    """Featurise with one arm's settings, via the shared featuriser.

    Both arms go through the same code path so a difference between them is a
    difference in features, not in two separately-written featurisers.
    """
    from drugrx.features import featurise as shared

    opts = dict(ARMS[arm])
    if opts.get("extra_fps"):
        # `workers` belongs here too. Leaving it out made the one arm with the
        # *most* work to do — Avalon and ErG on top of everything the other arm
        # computes — the only arm that ran single-process, which is both slow and
        # a difference between the arms that is not a difference in features.
        return shared(smiles, cache=cache, workers=workers, **opts)
    if opts.pop("desc_mode") == "none":
        # Fingerprint only: ask for the small block and slice it off, rather
        # than maintaining a second featuriser that could drift.
        x = shared(smiles, cache=cache, workers=workers, desc_mode="small", **opts)
        return x[:, :2048]
    return shared(smiles, cache=cache, workers=workers, desc_mode="rdkit2d", **opts)


def score(task: AdmetTask, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    if task.metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if task.metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if task.metric == "pr_auc":
        return float(average_precision_score(y_true, y_pred))
    if task.metric == "spearman":
        return float(spearmanr(y_true, y_pred).statistic)
    raise ValueError(f"unknown metric {task.metric}")


def fit_predict_gpu(task: AdmetTask, xtr, ytr, xva, yva, xte) -> np.ndarray:
    """XGBoost on the GPU.

    LightGBM here was CPU-bound on a 2-vCPU box and took hours; XGBoost's
    `device="cuda"` hist builder puts the same model on the GPU and turns the
    sweep into minutes. The hyper-parameters mirror the LightGBM arm as closely
    as the two libraries allow, so a difference between arms stays a difference
    in *features* rather than in library defaults.
    """
    import xgboost as xgb

    common = dict(
        n_estimators=2000,
        learning_rate=0.05,
        max_leaves=31,
        grow_policy="lossguide",
        min_child_weight=10,
        colsample_bytree=0.6,
        subsample=0.8,
        reg_lambda=1.0,
        device="cuda",
        tree_method="hist",
        early_stopping_rounds=100,
        verbosity=0,
    )
    if task.task == "binary":
        model = xgb.XGBClassifier(**common, eval_metric="auc")
        model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
        return model.predict_proba(xte)[:, 1]
    model = xgb.XGBRegressor(**common, eval_metric="mae")
    model.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)
    return model.predict(xte)


def fit_predict(task: AdmetTask, xtr, ytr, xva, yva, xte) -> np.ndarray:
    import lightgbm as lgb

    common = dict(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,
        colsample_bytree=0.6,
        subsample=0.8,
        subsample_freq=1,
        reg_lambda=1.0,
        verbose=-1,
        n_jobs=-1,
    )
    if task.task == "binary":
        model = lgb.LGBMClassifier(**common)
    else:
        model = lgb.LGBMRegressor(**common)

    model.fit(
        xtr, ytr,
        # LightGBM 4.7 deprecates eval_set in favour of eval_X/eval_y, but
        # eval_y rejects the list form the deprecation message implies. The
        # warning is noise; the replacement does not work. Revisit on 5.x.
        eval_set=[(xva, yva)],
        eval_metric="auc" if task.task == "binary" else "l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    if task.task == "binary":
        return model.predict_proba(xte)[:, 1]
    return model.predict(xte)


def run_task(task: AdmetTask, extra_seeds: int, arm: str, cache: Path | None,
             workers: int | None = None) -> dict:
    print(f"\n── {task.key} ({task.task}, {task.metric}) [{arm}] ──────────")
    t0 = time.perf_counter()
    frame, splits = load_task(task)
    smiles = frame["smiles"].tolist()
    y = frame["y"].to_numpy(dtype=float)
    x = featurise_arm(smiles, arm, cache, workers)

    result: dict = {
        "task": task.key,
        "arm": arm,
        "repo": task.repo,
        "type": task.task,
        "metric": task.metric,
        "n_rows": int(len(frame)),
        "meaning": task.meaning,
    }

    # ── the official split: the leaderboard-comparable number ─────────
    if splits and all(k in splits for k in ("train", "valid", "test")):
        tr = np.array(splits["train"])
        va = np.array(splits["valid"])
        te = np.array(splits["test"])
        leak = scaffold_overlap([smiles[i] for i in tr], [smiles[i] for i in te])
        pred = fit_predict(task, x[tr], y[tr], x[va], y[va], x[te])
        official = score(task, y[te], pred)
        # Per-molecule predictions, keyed by SMILES, so the NN and the GBM can
        # be compared on exactly the same molecules and the difference
        # bootstrapped. A point estimate on a 95-molecule test fold is not a
        # ranking, and without these we cannot say so quantitatively.
        result["test_predictions"] = {
            "smiles": [smiles[i] for i in te],
            "y_true": [float(v) for v in y[te]],
            "y_pred": [float(v) for v in pred],
        }
        result["official_split"] = {
            "value": official,
            "n_train": int(len(tr)), "n_valid": int(len(va)), "n_test": int(len(te)),
            "test_scaffold_overlap": round(leak, 4),
        }
        print(f"  official split: {task.metric}={official:.4f}  "
              f"(train {len(tr)}, test {len(te)}, scaffold overlap {leak:.3f})")
        if leak > 0.01:
            print(f"  [warn] {leak:.1%} of test scaffolds also occur in train")
    else:
        print("  no official split in the mirror")

    # ── our scaffold splits: the error bar ────────────────────────────
    if extra_seeds > 0:
        vals = []
        for seed in range(extra_seeds):
            tr, va, te = scaffold_split(smiles, seed=seed)
            if len(te) == 0 or len(va) == 0:
                print(f"  seed {seed}: empty split, skipped")
                continue
            # A single-class test fold makes AUC undefined, which is a property
            # of the split, not a result. Skip it rather than report NaN.
            if task.task == "binary" and len(np.unique(y[te])) < 2:
                print(f"  seed {seed}: test fold is single-class, skipped")
                continue
            pred = fit_predict(task, x[tr], y[tr], x[va], y[va], x[te])
            vals.append(score(task, y[te], pred))
        if vals:
            result["our_scaffold_splits"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_seeds": len(vals),
                "values": [float(v) for v in vals],
            }
            print(f"  our splits:     {task.metric}={np.mean(vals):.4f} ± {np.std(vals):.4f} "
                  f"over {len(vals)} seeds")

    result["seconds"] = round(time.perf_counter() - t0, 1)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="task keys; default is every ADMET endpoint")
    ap.add_argument("--seeds", type=int, default=3,
                    help="extra scaffold-split seeds for an error bar (0 to skip)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--arm", default="count-rdkit2d", choices=list(ARMS),
                    help="feature set; run both to compare")
    ap.add_argument("--cache", default="data/featcache")
    ap.add_argument("--gpu", action="store_true",
                    help="use XGBoost on the GPU instead of CPU LightGBM")
    ap.add_argument("--workers", type=int, default=None,
                    help="featurisation processes; default is every core")
    args = ap.parse_args()

    global fit_predict
    if args.gpu:
        fit_predict = fit_predict_gpu
        print("[baseline] GPU arm: XGBoost device=cuda")

    tasks = (
        [TASK_BY_KEY[k] for k in args.tasks] if args.tasks else list(ADMET_TASKS)
    )

    results = []
    for task in tasks:
        try:
            results.append(run_task(task, args.seeds, args.arm, Path(args.cache), args.workers))
        except Exception as exc:  # noqa: BLE001 — one bad endpoint must not stop the sweep
            print(f"  FAILED: {exc}")
            results.append({"task": task.key, "error": str(exc)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": args.arm,
        "model": f"ECFP4(2048,r=2) [{args.arm}] + LightGBM",
        "note": "official_split is TDC's own scaffold split and is leaderboard-comparable; "
                "our_scaffold_splits is this repo's splitter and is an error-bar estimate only.",
        "results": results,
    }
    suffix = "" if args.arm == "count-rdkit2d" else f"-{args.arm}"
    (out / f"results{suffix}.json").write_text(json.dumps(payload, indent=2))

    ok = [r for r in results if "error" not in r and "official_split" in r]
    if ok:
        table = pd.DataFrame([
            {
                "task": r["task"],
                "n": r["n_rows"],
                "metric": r["metric"],
                "official": round(r["official_split"]["value"], 4),
                "ours_mean": round(r.get("our_scaffold_splits", {}).get("mean", float("nan")), 4),
                "ours_std": round(r.get("our_scaffold_splits", {}).get("std", float("nan")), 4),
            }
            for r in ok
        ])
        print("\n" + table.to_string(index=False))
        (out / f"results{suffix}.md").write_text(
            "# Baseline — ECFP4 + LightGBM\n\n"
            f"Generated {payload['generated']}.\n\n{payload['note']}\n\n"
            + table.to_markdown(index=False) + "\n"
        )

    failed = [r for r in results if "error" in r]
    print(f"\n{len(ok)} ok, {len(failed)} failed → {out / f'results{suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
