"""Choose the gradient-boosting arm per endpoint, then rebuild the routing table.

**Why this runs before anything is served.** `runs/ROUTING.md` compares the neural
ensemble against the `count-rdkit2d` LightGBM, but H5 found the `maplight` arm
(plus Avalon + ErG) better on 14 of 21 endpoints. So the committed routing table
does not describe a comparison against the gradient-boosting model we would
actually choose to serve, and on the two endpoints it awards to the ensemble the
better arm moves things in opposite directions: `maplight` closes ~46% of the
ensemble's `pgp` margin, and is 0.133 *worse* on `half_life`.

Three design decisions, each fixed before the run because each one is a way to
get a readable-looking wrong answer:

**1. The arm is selected on validation, never on test.** Picking the better of two
arms per endpoint by its test score is test-set selection; it would inflate
gradient boosting by construction, the same way a fixed Tanimoto cut inflated the
Avalon leakage row (T6). So selection reads the official **valid** fold, and the
model whose valid score is read was early-stopped on an **inner** split of train,
so the selector's fold is untouched by the fit.

**2. The same arm refit is not the same number, so arms are compared as
distributions.** `baseline.py` subsamples rows and columns and sets no
`random_state`: two runs of `count-rdkit2d` on the same split disagree by up to
0.094, and on 9 of 21 endpoints that same-arm spread is at least as large as the
maplight-vs-count effect. A single fit per arm would measure the seed. So each arm
is fit `--seeds` times and the arms are compared mean-to-mean with the spread
reported (T5 applied to a place it had not been applied yet).

**3. The ensemble-vs-GBM interval must carry the GBM's fit variance.**
`ROUTING.md`'s paired bootstrap resamples *molecules* while holding one GBM fit
fixed, so its intervals are too narrow — and that bites exactly where the table is
most confident, `pgp`'s lower bound of +0.0017 against a same-arm spread of
±0.0083. Here the bootstrap draws a GBM fit **and** a molecule resample on each
draw, so the interval covers both sources.

    python scripts/route_select.py --seeds 5 --out runs/routing

Writes `routing.json` — the machine-readable table a served model dispatches on —
and `arms.json`, which carries **every fit's score**, so the arm selection can be
audited rather than trusted. `scripts/render_routing.py` turns the pair into
`runs/ROUTING.md` for a human; that runs locally, because this script runs on a
disposable VM and `docs/` is not its to write.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.data.splits import scaffold_split  # noqa: E402
from drugrx.data.standardise import standardise  # noqa: E402
from drugrx.data.tdc import ADMET_TASKS, LOWER_IS_BETTER, AdmetTask, load_task  # noqa: E402

#: The two arms in contention. `binary-only` is not here: H1 settled it at 18/21
#: and nothing since has argued for revisiting it.
ARMS = ("count-rdkit2d", "maplight")


def metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "spearman":
        return float(spearmanr(y_true, y_pred).statistic)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if metric == "pr_auc":
        return float(average_precision_score(y_true, y_pred))
    return float(roc_auc_score(y_true, y_pred))


def better(metric: str, a: float, b: float) -> bool:
    """Is `a` better than `b` under this metric?"""
    return a < b if metric in LOWER_IS_BETTER else a > b


def fit_predict(task: AdmetTask, xtr, ytr, xva, yva, xte, seed: int) -> np.ndarray:
    """`baseline.py`'s LightGBM, with the seed made explicit.

    Identical hyper-parameters to the committed baseline — the point of this
    script is to compare *features*, so the learner must not move. The only
    addition is `random_state`, because a comparison between two stochastic fits
    needs to know which draw it is looking at.
    """
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
        random_state=seed,
    )
    model = lgb.LGBMClassifier(**common) if task.task == "binary" else lgb.LGBMRegressor(**common)
    model.fit(
        xtr, ytr,
        eval_set=[(xva, yva)],
        eval_metric="auc" if task.task == "binary" else "l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    if task.task == "binary":
        return model.predict_proba(xte)[:, 1]
    return model.predict(xte)


def inner_split(n: int, seed: int, frac: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """A held-out slice of *train*, for early stopping during arm selection.

    Random rather than scaffold-based on purpose: this fold exists only to stop
    the booster, and taking it by scaffold would make the fit systematically
    worse in a way that differs between arms for reasons unrelated to features.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    cut = max(1, int(round(n * frac)))
    return order[cut:], order[:cut]


def select_arm(task: AdmetTask, feats: dict[str, np.ndarray], y: np.ndarray,
               tr: np.ndarray, va: np.ndarray, seeds: int) -> dict:
    """Score both arms on the official valid fold and return the winner.

    The fit never sees `va`: early stopping uses an inner slice of `tr`. That is
    what makes the selection clean rather than merely unlikely to matter.
    """
    scores: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for arm in ARMS:
        x = feats[arm]
        for seed in range(seeds):
            inner_tr, inner_va = inner_split(len(tr), seed)
            pred = fit_predict(
                task,
                x[tr[inner_tr]], y[tr[inner_tr]],
                x[tr[inner_va]], y[tr[inner_va]],
                x[va], seed,
            )
            value = metric_value(task.metric, y[va], pred)
            if not np.isnan(value):
                scores[arm].append(value)

    means = {arm: (float(np.mean(v)) if v else float("nan")) for arm, v in scores.items()}
    stds = {arm: (float(np.std(v)) if len(v) > 1 else 0.0) for arm, v in scores.items()}

    ranked = [a for a in ARMS if not np.isnan(means[a])]
    if not ranked:
        return {"chosen": ARMS[0], "reason": "no valid score on either arm",
                "valid_mean": means, "valid_std": stds, "valid_scores": scores}
    chosen = ranked[0]
    for arm in ranked[1:]:
        if better(task.metric, means[arm], means[chosen]):
            chosen = arm

    # Say so when the margin is inside the spread. The arm still has to be
    # chosen — something must be served — but a reader should know the choice
    # was not forced by the data.
    others = [a for a in ranked if a != chosen]
    margin = (min(abs(means[chosen] - means[a]) for a in others) if others else float("inf"))
    pooled = max(stds[chosen], max((stds[a] for a in others), default=0.0))
    return {
        "chosen": chosen,
        "valid_mean": means,
        "valid_std": stds,
        "valid_scores": scores,
        "margin": margin,
        "within_noise": bool(margin < pooled),
    }


def paired_bootstrap_over_fits(
    metric: str,
    y_true: np.ndarray,
    pred_nn: np.ndarray,
    gbm_fits: list[np.ndarray],
    draws: int = 2000,
    seed: int = 0,
) -> dict:
    """CI on (ensemble − GBM), resampling **molecules and GBM fits** together.

    `ROUTING.md` v1 resampled molecules only, holding one GBM fit fixed, so its
    intervals charge none of the booster's own seed variance to the comparison.
    Drawing a fit per bootstrap replicate is what makes the interval cover the
    question actually being asked — "is the ensemble better than *this arm*",
    not "better than *this particular fit of* this arm".
    """
    rng = np.random.default_rng(seed)
    size = len(y_true)
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0

    diffs = []
    for _ in range(draws):
        idx = rng.integers(0, size, size)
        gbm = gbm_fits[int(rng.integers(0, len(gbm_fits)))]
        yt = y_true[idx]
        va = metric_value(metric, yt, pred_nn[idx])
        vb = metric_value(metric, yt, gbm[idx])
        if np.isnan(va) or np.isnan(vb):
            continue
        diffs.append(sign * (va - vb))

    if not diffs:
        return {"n_test": int(size), "undecided": True, "reason": "no valid bootstrap draws"}

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = sign * (
        metric_value(metric, y_true, pred_nn)
        - float(np.mean([metric_value(metric, y_true, g) for g in gbm_fits]))
    )
    return {
        "n_test": int(size),
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "undecided": bool(lo <= 0.0 <= hi),
        "p_nn_better": float((diffs > 0).mean()),
        "n_gbm_fits": len(gbm_fits),
    }


def key_of(smiles: str) -> str:
    """Standardised identity. A SMILES join matched 21 of lipophilicity's 420."""
    result = standardise(smiles)
    return result.inchikey if result.ok else f"__raw__{smiles}"


def ensemble_test_predictions(artifact: Path, data_dir: Path, device: str,
                              workers: int) -> dict[str, dict]:
    """Per-molecule test predictions from the **shipped** ensemble checkpoint.

    Not from `runs/h6/ensemble.json`: that was a 16-member ensemble without the
    ExpansionRx heads, and it is not the artifact anyone would call. Routing has
    to describe the model being served.
    """
    import torch

    from drugrx.data.tdc import ADMET_TASKS as TASKS
    from drugrx.fast_train import predict_ensemble
    from train_fast import PlainMLP

    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    from train_admet import load_data

    data, x = load_data(data_dir)
    frame = data.frame

    if x.shape[1] != blob["in_dim"]:
        raise SystemExit(
            f"feature width {x.shape[1]} but the checkpoint wants {blob['in_dim']}. "
            "Rebuild the cache with the same featuriser the artifact was trained on "
            "(prepare_data.py --extra-fps for 3091)."
        )

    from drugrx.features import standardise_descriptors
    stats = (np.array(blob["feature_stats"]["mean"], dtype=np.float32),
             np.array(blob["feature_stats"]["std"], dtype=np.float32))
    x, _ = standardise_descriptors(x, stats=stats)

    base = PlainMLP(blob["in_dim"], blob["n_tasks"], **{
        "d_model": blob["architecture"]["d_model"],
        "n_layers": blob["architecture"]["n_layers"],
        "dropout": blob["architecture"]["dropout"],
    }).to("meta")
    params = {k: v.to(device) for k, v in blob["params"].items()}
    buffers = {k: v.to(device) for k, v in blob["buffers"].items()}

    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
    with torch.no_grad():
        preds = predict_ensemble(base, params, buffers, xt)
    mean = preds.mean(0).float().cpu().numpy()
    std = preds.std(0).float().cpu().numpy()

    out: dict[str, dict] = {}
    for i, task in enumerate(TASKS):
        col = f"{task.key}__fold"
        if col not in frame.columns:
            continue
        rows = np.flatnonzero((frame[col] == "test").to_numpy())
        y = frame[task.key].to_numpy()[rows]
        keep = ~np.isnan(y)
        rows, y = rows[keep], y[keep]
        p, s = mean[rows, i], std[rows, i]
        if task.task == "regression":
            mu, sd = data.norm[task.key]
            y, p, s = y * sd + mu, p * sd + mu, s * sd
        else:
            p = 1.0 / (1.0 + np.exp(-p))
        out[task.key] = {
            "smiles": frame["smiles"].to_numpy()[rows].tolist(),
            "y_true": [float(v) for v in y],
            "y_pred": [float(v) for v in p],
            "y_std": [float(v) for v in s],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5,
                    help="fits per arm; the arm comparison is mean-to-mean over these")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--cache", default="data/featcache")
    ap.add_argument("--data", default="data")
    ap.add_argument("--artifact", default="",
                    help="ensemble.pt; empty skips the ensemble half and writes arms.json only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--tasks", nargs="*", default=[])
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        ROOT.parent / "docs" / "drugrx-training" / "runs" / "routing")
    out.mkdir(parents=True, exist_ok=True)

    from baseline import featurise_arm

    tasks = ([t for t in ADMET_TASKS if t.key in args.tasks] if args.tasks
             else list(ADMET_TASKS))

    arms_report: list[dict] = []
    gbm_test: dict[str, dict] = {}

    for task in tasks:
        t0 = time.perf_counter()
        print(f"\n── {task.key} ({task.task}, {task.metric}) ──────────", flush=True)
        frame, splits = load_task(task)
        smiles = frame["smiles"].tolist()
        y = frame["y"].to_numpy(dtype=float)
        if not (splits and all(k in splits for k in ("train", "valid", "test"))):
            print("  no official split in the mirror; skipped")
            continue
        tr, va, te = (np.array(splits[k]) for k in ("train", "valid", "test"))

        feats = {arm: featurise_arm(smiles, arm, Path(args.cache), args.workers)
                 for arm in ARMS}

        sel = select_arm(task, feats, y, tr, va, args.seeds)
        chosen = sel["chosen"]
        print(f"  valid: " + "  ".join(
            f"{a}={sel['valid_mean'][a]:.4f}±{sel['valid_std'][a]:.4f}" for a in ARMS)
            + f"  → {chosen}" + ("  (inside the spread)" if sel.get("within_noise") else ""),
            flush=True)

        # Test predictions with the committed protocol — fit on train, early-stop
        # on the official valid fold — so the numbers stay comparable with
        # runs/baseline/results.json, and once per seed so the bootstrap can draw
        # a fit.
        fits, values = [], []
        for seed in range(args.seeds):
            pred = fit_predict(task, feats[chosen][tr], y[tr],
                               feats[chosen][va], y[va], feats[chosen][te], seed)
            fits.append(pred)
            values.append(metric_value(task.metric, y[te], pred))
        print(f"  test ({chosen}): {np.mean(values):.4f} ± {np.std(values):.4f} "
              f"over {args.seeds} fits   [{time.perf_counter() - t0:.0f}s]", flush=True)

        gbm_test[task.key] = {
            "arm": chosen,
            "smiles": [smiles[i] for i in te],
            "y_true": [float(v) for v in y[te]],
            "fits": [[float(v) for v in p] for p in fits],
        }
        arms_report.append({
            "task": task.key, "metric": task.metric, "chosen": chosen,
            "valid_mean": sel["valid_mean"], "valid_std": sel["valid_std"],
            "valid_scores": sel["valid_scores"],
            "within_noise": sel.get("within_noise"),
            "test_mean": float(np.mean(values)), "test_std": float(np.std(values)),
            "n_test": int(len(te)),
        })

    (out / "arms.json").write_text(json.dumps(
        {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "seeds": args.seeds,
         "arms": list(ARMS), "selection": "official valid fold, inner-split early stopping",
         "results": arms_report}, indent=2))
    print(f"\nwrote {out / 'arms.json'}", flush=True)

    if not args.artifact:
        print("no --artifact: skipping the ensemble comparison")
        return 0

    nn = ensemble_test_predictions(Path(args.artifact), Path(args.data),
                                  args.device, args.workers or 8)

    rows = []
    for task in tasks:
        key = task.key
        if key not in nn or key not in gbm_test:
            continue
        g = gbm_test[key]
        gbm_map = {key_of(s): i for i, s in enumerate(g["smiles"])}
        nn_keys = [key_of(s) for s in nn[key]["smiles"]]
        shared = [(i, gbm_map[k]) for i, k in enumerate(nn_keys) if k in gbm_map]
        if len(shared) < 20:
            rows.append({"task": key, "n_shared": len(shared), "undecided": True,
                         "reason": "fewer than 20 shared molecules", "arm": g["arm"]})
            continue

        ni = np.array([i for i, _ in shared])
        gi = np.array([j for _, j in shared])
        y_true = np.array(nn[key]["y_true"])[ni]
        p_nn = np.array(nn[key]["y_pred"])[ni]
        gbm_fits = [np.array(f)[gi] for f in g["fits"]]

        result = paired_bootstrap_over_fits(task.metric, y_true, p_nn, gbm_fits,
                                            draws=args.draws)
        result.update({"task": key, "metric": task.metric, "arm": g["arm"],
                       "n_shared": len(shared)})
        rows.append(result)

    for r in rows:
        r["serves"] = ("ensemble" if (not r.get("undecided") and r.get("diff", 0) > 0)
                       else "gradient boosting")
        r["verdict"] = ("undecided" if r.get("undecided")
                        else ("ensemble" if r["diff"] > 0 else "gbm"))

    ens = [r["task"] for r in rows if r["serves"] == "ensemble"]
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seeds": args.seeds,
        "rule": "an endpoint routes to the ensemble only when its 95% CI excludes zero "
                "in the ensemble's favour; the interval resamples molecules AND GBM fits",
        "ensemble_endpoints": ens,
        "rows": rows,
    }
    (out / "routing.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {out / 'routing.json'}")
    print(f"\nensemble serves {len(ens)}: {', '.join(ens) or '(none)'}")
    for r in sorted(rows, key=lambda r: r.get("diff", 0), reverse=True):
        if "diff" not in r:
            print(f"  {r['task']:16s} {'—':>9s}  undecided ({r.get('reason','')})")
            continue
        print(f"  {r['task']:16s} {r['diff']:+9.4f} "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  {r['arm']:14s} {r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
