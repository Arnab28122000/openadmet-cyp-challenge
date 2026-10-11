"""One-variable-at-a-time sweep over capacity and regularisation.

Every run so far peaks around epoch 9 and then overfits for twenty more epochs
before the patience rule kills it: 4.9M parameters against 27k molecules with
dropout 0.1 and weight decay 0.01, none of it tuned. That is the cheapest
remaining lever, and it has to be pulled before any exotic technique gets credit
for a gain that better regularisation would have given for free.

Deliberately **one variable at a time** from a fixed base, not a grid: a grid of
this size costs more GPU-hours than the information is worth, and the point is
to find which knob matters, not to squeeze the last 0.001.

Runs sequentially in one process so a single detached launch covers the lot.
Each config is its own `run_id` (the config hash), so each checkpoints and
resumes independently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/content")

from drugrx.config import RunConfig
from drugrx.data.tdc import ADMET_TASKS
from drugrx.features import standardise_descriptors
from drugrx.models.multitask import MultiTaskAdmet, build_encoder, count_params
from drugrx.train import evaluate, train

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_admet import baseline_for, load_data  # noqa: E402

#: name → overrides on the base config. The base itself is first, so the sweep
#: re-measures it in the same session rather than comparing against a number
#: from a different run of the code.
VARIANTS: dict[str, dict] = {
    "base":        {},
    "drop0.3":     {"dropout": 0.3},
    "drop0.5":     {"dropout": 0.5},
    "wd0.1":       {"weight_decay": 0.1},
    "small256":    {"d_model": 256},
    "small256d03": {"d_model": 256, "dropout": 0.3},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--run-dir", default="/content/runs")
    ap.add_argument("--baseline", default="/content/baseline.json")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="/content/runs/sweep_reg.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--hub-repo", default=os.environ.get("DRUGRX_HUB_REPO", ""))
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    train_rows = (data.frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    results = []
    names = args.only or list(VARIANTS)
    for name in names:
        overrides = VARIANTS[name]
        torch.manual_seed(0)
        np.random.seed(0)

        config = RunConfig(
            name=f"sweep-{name}",
            seed=0,
            dataset="tdc-admet-21",
            split="official-union",
            tasks=tuple(t.key for t in ADMET_TASKS),
            arch="fingerprint",
            n_layers=3,
            epochs=args.epochs,
            patience=args.patience,
            monitor="mean_rank_proxy",
            monitor_mode="min",
            baseline=baseline_for("mean_rank_proxy"),
            hub_repo=args.hub_repo,
            **overrides,
        )

        encoder = build_encoder(
            "fingerprint", in_dim=x.shape[1],
            d_model=config.d_model, n_layers=config.n_layers, dropout=config.dropout,
        )
        model = MultiTaskAdmet(ADMET_TASKS, encoder, dropout=config.dropout)
        total, _ = count_params(model)

        print(f"\n{'=' * 70}\n[sweep] {name}: {overrides or 'base'} — "
              f"{total / 1e6:.2f}M params, run_id {config.run_id}\n{'=' * 70}", flush=True)

        t0 = time.time()
        state = train(config, model, data, x, run_dir=args.run_dir, device=args.device)

        # **Score the best checkpoint, not the model left in memory.** `train`
        # returns after the patience rule fires, so the live weights are from
        # the last (over-fitted) epoch, typically 20 epochs past the best one.
        # Evaluating those compares a different model than `best_valid`
        # describes, and makes every variant look worse than it is — which is
        # exactly the sort of mistake that would have had us "discover" that
        # regularisation does not help.
        best_path = Path(args.run_dir) / "ckpt" / config.run_id / "best.pt"
        if best_path.exists():
            payload = torch.load(best_path, map_location=args.device, weights_only=False)
            shadow = (payload.get("ema") or {}).get("shadow")
            if shadow:
                live = model.state_dict()
                model.load_state_dict(
                    {k: (shadow[k].to(v.dtype) if k in shadow else v) for k, v in live.items()},
                    strict=False,
                )
            else:
                model.load_state_dict(payload["model"])
        else:
            print(f"[sweep] WARNING: no best.pt for {name}; scoring live weights", flush=True)

        test = evaluate(model, data, x, "test", args.device)

        row = {
            "name": name,
            "overrides": overrides,
            "run_id": config.run_id,
            "params_m": round(total / 1e6, 2),
            "best_valid": state.best_metric,
            "best_epoch": state.best_epoch,
            "epochs_run": state.epoch,
            "test_mean_rank_proxy": test.get("mean_rank_proxy"),
            "minutes": round((time.time() - t0) / 60, 1),
            "test": {k: v for k, v in test.items() if k != "mean_rank_proxy"},
        }
        results.append(row)
        print(f"[sweep] {name}: valid {row['best_valid']:.4f} @ep{row['best_epoch']} | "
              f"test {row['test_mean_rank_proxy']:.4f} | {row['minutes']} min", flush=True)

        Path(args.out).write_text(json.dumps(results, indent=2, default=float))

    print(f"\n{'=' * 70}\nSWEEP SUMMARY (lower is better)\n{'=' * 70}")
    for r in sorted(results, key=lambda r: r["test_mean_rank_proxy"]):
        print(f"  {r['name']:14s} {r['params_m']:5.2f}M  valid {r['best_valid']:.4f} "
              f"@ep{r['best_epoch']:<3d}  test {r['test_mean_rank_proxy']:.4f}  {r['minutes']:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
