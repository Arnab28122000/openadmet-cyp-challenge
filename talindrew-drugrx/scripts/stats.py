"""Is the difference between two models real, or is it the size of the test fold?

Eight of our 21 endpoints have test folds of 95–175 molecules. On a fold that
size a gap of 0.01 in ROC-AUC is noise, and "the net wins 8, LightGBM wins 13"
is a tally of coin flips unless each flip is shown to be weighted. This script
answers that and nothing else.

Two tests, because they answer different questions:

* **Per endpoint** — a paired bootstrap over the test molecules. Both models
  predicted the same molecules, so resampling *molecules* (not predictions)
  keeps the pairing and gives a confidence interval on the *difference*. An
  interval straddling zero means the endpoint is undecided, which is the honest
  verdict for most of them.
* **Across endpoints** — an exact sign test on the per-endpoint winners. It asks
  whether winning 13 of 21 is more than chance would give, and the answer at
  this count is no.

    python scripts/stats.py --nn runs/h3b/eval.json --gbm runs/baseline/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drugrx.data.tdc import LOWER_IS_BETTER, TASK_BY_KEY


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


def paired_bootstrap(
    metric: str,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n: int = 2000,
    seed: int = 0,
) -> dict:
    """CI on (a − b), oriented so positive always means **a is better**."""
    rng = np.random.default_rng(seed)
    size = len(y_true)
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0

    diffs = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        yt = y_true[idx]
        # A resample can be single-class; that bootstrap draw is undefined for
        # AUC rather than zero, so it is skipped, not counted as a tie.
        va = metric_value(metric, yt, pred_a[idx])
        vb = metric_value(metric, yt, pred_b[idx])
        if np.isnan(va) or np.isnan(vb):
            continue
        diffs.append(sign * (va - vb))

    if not diffs:
        return {"n_test": int(size), "undecided": True, "reason": "no valid bootstrap draws"}

    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = sign * (
        metric_value(metric, y_true, pred_a) - metric_value(metric, y_true, pred_b)
    )
    return {
        "n_test": int(size),
        "diff": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        # The interval containing zero is the whole point: it means this
        # endpoint does not choose between the two models.
        "undecided": bool(lo <= 0.0 <= hi),
        "p_a_better": float((diffs > 0).mean()),
    }


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial p for `wins` out of `wins+losses` at p=0.5."""
    from math import comb

    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", required=True, help="evaluate.py output with test_predictions")
    ap.add_argument("--gbm", required=True, help="baseline results.json with test_predictions")
    ap.add_argument("--out", default="")
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()

    nn_blob = json.loads(Path(args.nn).read_text())
    gbm_blob = json.loads(Path(args.gbm).read_text())

    nn_preds = nn_blob.get("test_predictions", {})
    gbm_preds = {
        r["task"]: r["test_predictions"]
        for r in gbm_blob.get("results", [])
        if "test_predictions" in r
    }

    if not nn_preds or not gbm_preds:
        print("Both files must carry per-molecule test_predictions. "
              "Re-run baseline.py and evaluate.py after the prediction-saving patch.")
        return 1

    rows = []
    for key, nn in nn_preds.items():
        gbm = gbm_preds.get(key)
        if gbm is None:
            continue
        task = TASK_BY_KEY[key]

        # **Join on standardised identity, not on the SMILES string.** The two
        # pipelines write the same molecule differently — the ensemble sees the
        # standardised form, the baseline the raw TDC form — so a string join
        # silently drops most rows. It matched 21 of lipophilicity's 420 before
        # this was fixed, which would have made that row noise wearing the
        # appearance of a measurement.
        from drugrx.data.standardise import standardise

        def key_of(s: str) -> str:
            r = standardise(s)
            return r.inchikey if r.ok else f"__raw__{s}"

        gbm_map = {key_of(s): (yt, yp)
                   for s, yt, yp in zip(gbm["smiles"], gbm["y_true"], gbm["y_pred"])}
        nn_keys = [key_of(s) for s in nn["smiles"]]
        shared = [k for k in nn_keys if k in gbm_map]
        if len(shared) < 20:
            rows.append({"task": key, "n_shared": len(shared), "undecided": True,
                         "reason": "fewer than 20 shared molecules"})
            continue

        nn_map = {k: (yt, yp) for k, yt, yp in zip(nn_keys, nn["y_true"], nn["y_pred"])}
        y_true = np.array([nn_map[k][0] for k in shared])
        p_nn = np.array([nn_map[k][1] for k in shared])
        p_gbm = np.array([gbm_map[k][1] for k in shared])

        result = paired_bootstrap(task.metric, y_true, p_nn, p_gbm, n=args.draws)
        result.update({"task": key, "metric": task.metric, "n_shared": len(shared)})
        rows.append(result)

    decided = [r for r in rows if not r.get("undecided")]
    nn_wins = sum(1 for r in decided if r.get("diff", 0) > 0)
    gbm_wins = len(decided) - nn_wins
    undecided = len(rows) - len(decided)

    print(f"{'endpoint':16s} {'n':>5s} {'diff':>9s} {'95% CI':>22s}  verdict")
    for r in sorted(rows, key=lambda r: r.get("diff", 0), reverse=True):
        if "diff" not in r:
            print(f"{r['task']:16s} {r.get('n_shared', 0):5d} {'—':>9s} {'—':>22s}  "
                  f"undecided ({r.get('reason', '')})")
            continue
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        verdict = "undecided" if r["undecided"] else ("net" if r["diff"] > 0 else "lightgbm")
        print(f"{r['task']:16s} {r['n_shared']:5d} {r['diff']:+9.4f} {ci:>22s}  {verdict}")

    p = sign_test(nn_wins, gbm_wins)
    print(f"\ndecided endpoints: net {nn_wins}, lightgbm {gbm_wins}, undecided {undecided}")
    print(f"sign test over decided endpoints: p = {p:.3f}"
          f"  → {'a real difference' if p < 0.05 else 'NOT distinguishable from chance'}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"rows": rows, "nn_wins": nn_wins, "gbm_wins": gbm_wins,
             "undecided": undecided, "sign_test_p": p}, indent=2, default=float))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
