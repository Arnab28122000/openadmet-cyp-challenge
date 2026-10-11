"""Train a deep ensemble on the GPU, with the whole dataset resident in VRAM.

The point is not speed for its own sake: an ensemble is the cheapest honest
uncertainty available, and the playground ranks candidates, so a confident wrong
score is worse than an honest wide one. Training the members simultaneously
under `vmap` makes the ensemble nearly free — the card was idle at batch 128.

    python scripts/make_driver.py --detach train_fast.py --members 16 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/content")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drugrx.data.tdc import ADMET_TASKS, LOWER_IS_BETTER  # noqa: E402
from drugrx.fast_train import GPUData, fit_ensemble, predict_ensemble  # noqa: E402
from drugrx.features import standardise_descriptors  # noqa: E402
from train_admet import load_data  # noqa: E402


class PlainMLP(nn.Module):
    """One tensor in, one tensor out — the shape `vmap` needs.

    The production model returns a dict of per-task heads; that is friendlier to
    read but cannot be `functional_call`ed cleanly, so the ensemble path uses a
    single wide output head and slices it per task.
    """

    def __init__(self, in_dim: int, n_tasks: int, d_model: int = 512,
                 n_layers: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)]
            d = d_model
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d_model, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


def score_fold(data, frame, preds_mean: np.ndarray, fold: str) -> dict[str, float]:
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    out: dict[str, float] = {}
    for i, task in enumerate(ADMET_TASKS):
        col = f"{task.key}__fold"
        if col not in frame.columns:
            continue
        rows = np.flatnonzero((frame[col] == fold).to_numpy())
        if len(rows) < 10:
            continue
        y = frame[task.key].to_numpy()[rows]
        p = preds_mean[rows, i]
        keep = ~np.isnan(y)
        y, p = y[keep], p[keep]
        if len(y) < 10:
            continue
        if task.task == "regression":
            mean, std = data.norm[task.key]
            y, p = y * std + mean, p * std + mean
            out[task.key] = (float(mean_absolute_error(y, p)) if task.metric == "mae"
                             else float(spearmanr(y, p).statistic))
        else:
            if len(np.unique(y)) < 2:
                continue
            prob = 1.0 / (1.0 + np.exp(-p))
            out[task.key] = (float(average_precision_score(y, prob))
                             if task.metric == "pr_auc" else float(roc_auc_score(y, prob)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--out", default="/content/out/fast_ensemble.json")
    ap.add_argument("--members", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=0, help="0 = auto-size to the card")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    train_rows = (data.frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    keys = [t.key for t in ADMET_TASKS]
    y_all = data.frame[keys].to_numpy(dtype=np.float32)
    mask_all = ~np.isnan(y_all)
    y_all = np.nan_to_num(y_all, nan=0.0)

    tr = np.flatnonzero(train_rows)
    va = np.flatnonzero((data.frame["split"] == "valid").to_numpy())

    gpu_train = GPUData.from_numpy(x[tr], y_all[tr], mask_all[tr], args.device)
    gpu_valid = GPUData.from_numpy(x[va], y_all[va], mask_all[va], args.device)
    x_all = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(args.device)

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[fast] device {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB), "
          f"train {gpu_train.n} rows, {len(keys)} tasks", flush=True)

    def make_model(_: int) -> nn.Module:
        return PlainMLP(x.shape[1], len(keys), args.d_model, args.n_layers, args.dropout)

    # Validation loss on the held-out fold, computed on the GPU each epoch.
    # At 0.26s an epoch this costs nothing and is the difference between an
    # ensemble that beats the single model and one that overfits past it.
    base_meta = make_model(0).to("meta")

    def score_fn(params, buffers) -> float:
        preds = predict_ensemble(base_meta, params, buffers, gpu_valid.x)
        mean = preds.mean(0)
        diff = (mean - gpu_valid.y) ** 2
        return float((diff * gpu_valid.mask).sum() / gpu_valid.mask.sum().clamp(min=1.0))

    t0 = time.time()
    params, buffers, history = fit_ensemble(
        make_model, gpu_train, gpu_valid,
        n_members=args.members, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size or None, device=args.device,
        patience=args.patience, score_fn=score_fn,
    )
    minutes = (time.time() - t0) / 60

    base = make_model(0).to("meta")
    per_member = predict_ensemble(base, params, buffers, x_all)  # (K, N, T)
    mean = per_member.mean(0).float().cpu().numpy()
    spread = per_member.std(0).float().cpu().numpy()

    valid_scores = score_fold(data, data.frame, mean, "valid")
    test_scores = score_fold(data, data.frame, mean, "test")

    peak = torch.cuda.max_memory_allocated() / 1e9
    report = {
        "members": args.members,
        "epochs": args.epochs,
        "minutes": round(minutes, 2),
        "peak_gb": round(peak, 2),
        "card_gb": round(total_gb, 1),
        "utilisation": round(peak / total_gb, 3),
        "final_loss": history[-1] if history else None,
        "valid": valid_scores,
        "test": test_scores,
        # The spread across members is the uncertainty we actually wanted.
        "mean_spread": float(np.nanmean(spread)),
    }

    # Per-molecule test predictions, so the ensemble can be compared against the
    # gradient-boosting baseline with a paired bootstrap rather than by eyeing
    # two point estimates on a 95-molecule fold.
    preds: dict = {}
    for i, task in enumerate(ADMET_TASKS):
        col = f"{task.key}__fold"
        if col not in data.frame.columns:
            continue
        rows = np.flatnonzero((data.frame[col] == "test").to_numpy())
        if len(rows) == 0:
            continue
        y = data.frame[task.key].to_numpy()[rows]
        p = mean[rows, i]
        sd = spread[rows, i]
        keep = ~np.isnan(y)
        rows, y, p, sd = rows[keep], y[keep], p[keep], sd[keep]
        if task.task == "regression":
            mu, sg = data.norm[task.key]
            y, p, sd = y * sg + mu, p * sg + mu, sd * sg
        else:
            p = 1.0 / (1.0 + np.exp(-p))
        preds[task.key] = {
            "smiles": [data.frame["smiles"].iloc[j] for j in rows],
            "y_true": [float(v) for v in y],
            "y_pred": [float(v) for v in p],
            "y_std": [float(v) for v in sd],
        }
    report["test_predictions"] = preds
    print(json.dumps({k: v for k, v in report.items() if k != "test"}, indent=2), flush=True)
    print("TEST:", json.dumps(test_scores, indent=2), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float))
    print(f"[fast] wrote {args.out} — peak {peak:.1f} GB of {total_gb:.1f} GB "
          f"({peak/total_gb:.0%}), {minutes:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
