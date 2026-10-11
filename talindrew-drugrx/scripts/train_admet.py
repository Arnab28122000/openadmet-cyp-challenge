"""Train the multi-task ADMET model.

On the VM:
    colab exec -s drugrx -f scripts/train_admet.py

Locally, to test the harness without spending compute units:
    python scripts/train_admet.py --device cpu --epochs 3 --data data

The baseline it has to beat is read from the committed baseline results, so the
"worse than ECFP4+LightGBM at halfway" kill rule has a real number behind it
rather than a guess.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from drugrx.config import RunConfig
from drugrx.data.multitask import MultiTaskData
from drugrx.data.tdc import ADMET_TASKS
from drugrx.models.multitask import MultiTaskAdmet, build_encoder, count_params
from drugrx.train import train

BASELINE = Path(__file__).resolve().parents[2] / "docs" / "drugrx-training" / "runs" / "baseline" / "results.json"


def load_data(data_dir: Path) -> tuple[MultiTaskData, np.ndarray]:
    """Load the cached table and features, and apply the stored normalisation.

    **The parquet holds raw targets on purpose** — it is the artifact a human
    reads, and a z-score is unreadable. The normalisation is stored alongside in
    meta.json and applied here. Skipping this step trains on raw targets, which
    does not crash: it just produces a loss two orders of magnitude too large
    and a model dominated by whichever endpoint has the widest units.
    """
    frame = pd.read_parquet(data_dir / "admet_multitask.parquet")
    meta = json.loads((data_dir / "meta.json").read_text())
    with np.load(data_dir / "features.npz") as handle:
        x = handle["x"]
    norm = {k: tuple(v) for k, v in meta["norm"].items()}

    for task in ADMET_TASKS:
        mean, std = norm[task.key]
        if task.task == "regression" and std != 1.0:
            frame[task.key] = (frame[task.key] - mean) / std

    return MultiTaskData(frame=frame, tasks=ADMET_TASKS, norm=norm), x


def baseline_for(monitor: str) -> float | None:
    """The committed ECFP4+LightGBM number for the monitored endpoint."""
    if not BASELINE.exists():
        return None
    blob = json.loads(BASELINE.read_text())
    key = monitor.split("/", 1)[-1]
    for row in blob.get("results", []):
        if row.get("task") == key and "official_split" in row:
            return float(row["official_split"]["value"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--arch", default="fingerprint",
                    choices=["fingerprint", "recursive", "chemberta"])
    ap.add_argument("--name", default="admet-mt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6, help="recursive arch only")
    ap.add_argument("--monitor", default="mean_rank_proxy")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max-hours", type=float, default=23.0)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hub-repo", default=os.environ.get("DRUGRX_HUB_REPO", ""),
                    help="private HF repo to mirror checkpoints to each epoch; "
                         "without it a dead VM loses the run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data)
    data, x = load_data(data_dir)

    # Descriptors are standardised with training-split statistics only.
    from drugrx.features import standardise_descriptors

    train_rows = (data.frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    config = RunConfig(
        name=f"{args.name}-{args.arch}",
        seed=args.seed,
        dataset="tdc-admet-21",
        split="official-union",
        tasks=tuple(t.key for t in ADMET_TASKS),
        arch=args.arch,
        d_model=args.d_model,
        n_layers=args.n_layers,
        arch_kwargs={"depth": args.depth} if args.arch == "recursive" else {},
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        monitor=args.monitor,
        monitor_mode="min",  # mean_rank_proxy is lower-is-better by construction
        patience=args.patience,
        max_hours=args.max_hours,
        precision=args.precision,
        baseline=baseline_for(args.monitor),
        hub_repo=args.hub_repo,
    )

    kwargs: dict = {"d_model": args.d_model, "dropout": config.dropout}
    if args.arch == "fingerprint":
        kwargs["n_layers"] = args.n_layers
    elif args.arch == "recursive":
        kwargs["depth"] = args.depth
    encoder = build_encoder(args.arch, in_dim=x.shape[1], **kwargs)
    model = MultiTaskAdmet(ADMET_TASKS, encoder, dropout=config.dropout)

    total, trainable = count_params(model)
    print(f"[run] {config.run_id}: {total/1e6:.2f}M params ({trainable/1e6:.2f}M trainable), "
          f"baseline={config.baseline}")

    state = train(config, model, data, x, run_dir=args.run_dir, device=args.device)
    print(f"[run] finished at epoch {state.epoch}, best {config.monitor}={state.best_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
