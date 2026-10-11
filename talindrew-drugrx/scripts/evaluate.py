"""Score a trained checkpoint on TDC's official **test** folds, against the baseline.

The training loop watches the *valid* fold, because that is what model selection
is allowed to see. This is the separate, once-per-model step that touches test —
and it compares against the committed ECFP4+LightGBM numbers, because a neural
net that does not beat them has not earned its GPU hours.

    python scripts/make_driver.py evaluate.py --run <run-id> --ckpt best
    colab exec -s drugrx-l4 -f /tmp/drugrx_driver.py
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
from drugrx.data.tdc import ADMET_TASKS, LOWER_IS_BETTER
from drugrx.features import standardise_descriptors
from drugrx.models.multitask import MultiTaskAdmet, build_encoder
from drugrx.train import evaluate


def load_data(data_dir: Path) -> tuple[MultiTaskData, np.ndarray]:
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


def per_molecule_predictions(model, data, x, fold: str, device: str,
                             batch_size: int = 512) -> dict:
    """Per-endpoint `{smiles, y_true, y_pred}` on one fold, on the real scale.

    Needed so the net and the GBM can be compared on identical molecules and
    the difference bootstrapped. Regression targets are de-standardised, because
    a z-score cannot be compared against the baseline's raw-scale numbers.
    """
    import numpy as np
    import torch

    model.eval()
    frame = data.frame
    chunks: dict[str, list] = {}
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            xb = torch.from_numpy(x[start:start + batch_size]).to(device)
            for key, value in model(x=xb).items():
                chunks.setdefault(key, []).append(value.float().cpu().numpy())
    allp = {k: np.concatenate(v) for k, v in chunks.items()}

    out: dict = {}
    for task in ADMET_TASKS:
        col = f"{task.key}__fold"
        if col not in frame.columns:
            continue
        rows = np.flatnonzero((frame[col] == fold).to_numpy())
        if len(rows) == 0:
            continue
        y_true = frame[task.key].to_numpy()[rows]
        y_pred = allp[task.key][rows]
        keep = ~np.isnan(y_true)
        rows, y_true, y_pred = rows[keep], y_true[keep], y_pred[keep]
        if task.task == "regression":
            mean, std = data.norm[task.key]
            y_true, y_pred = y_true * std + mean, y_pred * std + mean
        else:
            y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        out[task.key] = {
            "smiles": [frame["smiles"].iloc[i] for i in rows],
            "y_true": [float(v) for v in y_true],
            "y_pred": [float(v) for v in y_pred],
        }
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/content/data")
    ap.add_argument("--ckpt-dir", default="/content/runs/ckpt")
    ap.add_argument("--run", required=True, help="run id (checkpoint directory name)")
    ap.add_argument("--ckpt", default="best", choices=["best", "latest"])
    ap.add_argument("--baseline", default="/content/baseline.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data, x = load_data(Path(args.data))
    train_rows = (data.frame["split"] == "train").to_numpy()
    _, stats = standardise_descriptors(x[train_rows])
    x, _ = standardise_descriptors(x, stats=stats)

    path = Path(args.ckpt_dir) / args.run / f"{args.ckpt}.pt"
    payload = torch.load(path, map_location=args.device, weights_only=False)
    config = RunConfig.from_json(payload["config"])

    kwargs: dict = {"d_model": config.d_model, "dropout": config.dropout}
    if config.arch == "fingerprint":
        kwargs["n_layers"] = config.n_layers
    elif config.arch == "recursive":
        kwargs["depth"] = config.arch_kwargs.get("depth", 6)
    encoder = build_encoder(config.arch, in_dim=x.shape[1], **kwargs)
    model = MultiTaskAdmet(ADMET_TASKS, encoder, dropout=config.dropout).to(args.device)

    # Prefer the EMA weights: those are what training selected on and what would
    # ship. Falling back silently to the live weights would compare a different
    # model than the one the valid-fold numbers describe.
    if payload.get("ema") and payload["ema"].get("shadow"):
        shadow = payload["ema"]["shadow"]
        state = model.state_dict()
        merged = {k: (shadow[k].to(v.dtype) if k in shadow else v) for k, v in state.items()}
        model.load_state_dict(merged, strict=False)
        which = "EMA"
    else:
        model.load_state_dict(payload["model"])
        which = "live"
    print(f"[eval] {args.run} ({config.arch}, {which} weights) from {path}")

    scores = evaluate(model, data, x, "test", args.device)
    preds = per_molecule_predictions(model, data, x, "test", args.device)

    base: dict[str, float] = {}
    if os.path.exists(args.baseline):
        blob = json.loads(Path(args.baseline).read_text())
        for row in blob.get("results", []):
            if "official_split" in row:
                base[row["task"]] = float(row["official_split"]["value"])

    rows = []
    for task in ADMET_TASKS:
        if task.key not in scores:
            continue
        nn = scores[task.key]
        gbm = base.get(task.key)
        lower = task.metric in LOWER_IS_BETTER
        win = None if gbm is None else ((nn < gbm) if lower else (nn > gbm))
        rows.append({
            "task": task.key, "metric": task.metric,
            "nn": round(nn, 4),
            "lightgbm": None if gbm is None else round(gbm, 4),
            "nn_wins": "" if win is None else ("Y" if win else "."),
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    wins = (table["nn_wins"] == "Y").sum()
    compared = (table["nn_wins"] != "").sum()
    print(f"\nneural net beats LightGBM on {wins}/{compared} endpoints")
    print(f"mean_rank_proxy (test): {scores.get('mean_rank_proxy'):.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"run": args.run, "arch": config.arch, "weights": which,
             "wins": int(wins), "compared": int(compared),
             "scores": scores, "rows": rows,
             "test_predictions": preds}, indent=2, default=float))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
