"""The training loop.

One loop, used by every experiment, so that a comparison between two techniques
is a comparison between two configs rather than two programs. It owns the things
that are easy to get subtly wrong and expensive to get wrong late:

* **resume** — the budget is 24h across an unknown number of sessions, so the
  loop starts by asking the checkpointer where it is, and the elapsed-time
  budget carries across sessions rather than resetting;
* **EMA** — evaluated and shipped, while the live weights are only trained;
* **kill rules** — applied automatically, including "still worse than the
  ECFP4+LightGBM baseline at halfway", which is the outcome the literature says
  to expect;
* **per-endpoint scoring on TDC's own folds**, so a number here means what the
  leaderboard means by it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from drugrx.checkpoint import Checkpointer, TrainState
from drugrx.config import RunConfig
from drugrx.data.multitask import MultiTaskData
from drugrx.data.tdc import LOWER_IS_BETTER
from drugrx.metrics import Heartbeat, KillRules, MetricLog, is_improvement
from drugrx.schedule import EMA
from drugrx.schedule import build as build_schedule


class FeatureDataset(Dataset):
    """Precomputed features, labels and the labelled-mask."""

    def __init__(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> None:
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
        self.mask = torch.from_numpy(np.ascontiguousarray(mask))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[i], "y": self.y[i], "mask": self.mask[i]}


def evaluate(
    model: torch.nn.Module,
    data: MultiTaskData,
    frame_x: np.ndarray,
    fold: str,
    device: str,
    batch_size: int = 512,
) -> dict[str, float]:
    """Score every endpoint on **its own** TDC fold.

    Returns `{endpoint: value}` plus `"mean_rank_proxy"`, a crude single number
    for the kill rules: the mean over endpoints of each metric mapped so that
    lower is always better. It is a monitoring aid, not a result.
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

    model.eval()
    frame = data.frame
    out: dict[str, float] = {}

    with torch.no_grad():
        preds_all: dict[str, np.ndarray] = {}
        for start in range(0, len(frame), batch_size):
            xb = torch.from_numpy(frame_x[start:start + batch_size]).to(device)
            preds = model(x=xb)
            for key, value in preds.items():
                preds_all.setdefault(key, []).append(value.float().cpu().numpy())
        preds_all = {k: np.concatenate(v) for k, v in preds_all.items()}

    normalised: list[float] = []
    for task in data.tasks:
        fold_col = f"{task.key}__fold"
        if fold_col not in frame.columns:
            continue
        rows = np.flatnonzero((frame[fold_col] == fold).to_numpy())
        if len(rows) < 10:
            continue
        y_true = frame[task.key].to_numpy()[rows]
        y_pred = preds_all[task.key][rows]
        keep = ~np.isnan(y_true)
        y_true, y_pred = y_true[keep], y_pred[keep]
        if len(y_true) < 10:
            continue

        if task.task == "regression":
            # Back to the real scale: a z-scored MAE is not the leaderboard's.
            mean, std = data.norm[task.key]
            y_true_r, y_pred_r = y_true * std + mean, y_pred * std + mean
            if task.metric == "mae":
                value = float(mean_absolute_error(y_true_r, y_pred_r))
                normalised.append(min(1.0, value / (abs(mean) + std + 1e-9)))
            else:  # spearman
                value = float(spearmanr(y_true_r, y_pred_r).statistic)
                normalised.append(1.0 - max(0.0, value))
        else:
            if len(np.unique(y_true)) < 2:
                continue
            prob = 1.0 / (1.0 + np.exp(-y_pred))
            if task.metric == "pr_auc":
                value = float(average_precision_score(y_true, prob))
            else:
                value = float(roc_auc_score(y_true, prob))
            normalised.append(1.0 - value)

        out[task.key] = value

    out["mean_rank_proxy"] = float(np.mean(normalised)) if normalised else float("nan")
    model.train()
    return out


def train(
    config: RunConfig,
    model: torch.nn.Module,
    data: MultiTaskData,
    features: np.ndarray,
    run_dir: str | Path = "/content/runs",
    device: str = "cuda",
    on_epoch: Optional[Callable[[int, dict], None]] = None,
) -> TrainState:
    run_dir = Path(run_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log = MetricLog(run_dir / "metrics.jsonl")
    beat = Heartbeat(run_dir / "heartbeat.json", every=config.heartbeat_every)
    rules = KillRules(config)

    model = model.to(device)

    # Fused AdamW is meaningfully faster on Ada and costs nothing to ask for.
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay, fused=True
        )
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

    train_idx = np.flatnonzero((data.frame["split"] == "train").to_numpy())
    x_train = features[train_idx]
    y_train = data.frame[data.task_keys].to_numpy(dtype=np.float32)[train_idx]
    mask_train = ~np.isnan(y_train)
    y_train = np.nan_to_num(y_train, nan=0.0)

    loader = DataLoader(
        FeatureDataset(x_train, y_train, mask_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        # The VM has four vCPU and the features are already tensors; workers
        # would cost more in process overhead than they save.
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    steps_per_epoch = max(1, len(loader) // max(1, config.grad_accum))
    scheduler = build_schedule(
        config.schedule, optimizer, config.epochs * steps_per_epoch,
        config.warmup_steps, config.decay_frac,
    )
    ema = EMA(model, config.ema_decay) if config.ema_decay > 0 else None

    ckpt = Checkpointer(config, local_dir=run_dir.parent / "ckpt")
    state = ckpt.resume(model, optimizer, scheduler, ema=ema, map_location=device)

    use_bf16 = config.precision == "bf16" and device == "cuda" and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    # bf16 needs no loss scaling; fp16 does. Getting this backwards produces
    # silent NaNs an hour in.
    scaler = torch.amp.GradScaler(device, enabled=(config.precision == "fp16"))

    session_start = time.time()
    print(f"[train] {config.run_id}: {len(train_idx)} train molecules, "
          f"{steps_per_epoch} steps/epoch, precision={'bf16' if use_bf16 else config.precision}")

    for epoch in range(state.epoch, config.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(loader):
            xb = batch["x"].to(device, non_blocking=True)
            yb = batch["y"].to(device, non_blocking=True)
            mb = batch["mask"].to(device, non_blocking=True)

            with torch.autocast(device_type=device, dtype=autocast_dtype,
                                enabled=(device == "cuda" and config.precision != "fp32")):
                preds = model(x=xb)
                loss, _parts = model.loss(preds, yb, mb)

            scaler.scale(loss / config.grad_accum).backward()

            if (i + 1) % config.grad_accum == 0:
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                state.global_step += 1

            epoch_loss += float(loss.detach())
            n_batches += 1

            if state.global_step % config.log_every == 0:
                log.log(kind="step", epoch=epoch, step=state.global_step,
                        loss=epoch_loss / max(1, n_batches), lr=scheduler.get_last_lr()[0])
            beat.beat(epoch=epoch, step=state.global_step, phase="train")

        train_loss = epoch_loss / max(1, n_batches)
        diverged = not np.isfinite(train_loss)

        # Evaluate the EMA weights — those are what gets shipped.
        if ema is not None:
            ema.apply_to(model)
        scores = evaluate(model, data, features, "valid", device)
        if ema is not None:
            ema.restore(model)

        monitor_key = config.monitor.split("/", 1)[-1]
        current = scores.get(monitor_key, scores.get("mean_rank_proxy"))

        state.epoch = epoch + 1
        state.elapsed_seconds += time.time() - session_start
        session_start = time.time()

        improved = current is not None and is_improvement(
            current, state.best_metric, config.monitor_mode
        )
        if improved:
            state.best_metric, state.best_epoch, state.epochs_since_best = current, epoch, 0
        else:
            state.epochs_since_best += 1

        log.log(kind="epoch", epoch=epoch, train_loss=train_loss,
                elapsed_hours=state.elapsed_seconds / 3600, **scores)
        print(f"[train] epoch {epoch}: loss={train_loss:.4f} "
              f"{config.monitor}={current if current is None else round(current, 4)} "
              f"{'*' if improved else ''} ({state.elapsed_seconds / 3600:.2f}h)")

        if (epoch + 1) % config.checkpoint_every == 0 or improved:
            ckpt.save(model, optimizer, state, scheduler, scaler, ema, is_best=improved)

        if on_epoch is not None:
            on_epoch(epoch, scores)

        verdict = rules.check(
            epoch=epoch, epochs_since_best=state.epochs_since_best, current=current,
            elapsed_hours=state.elapsed_seconds / 3600, diverged=diverged,
        )
        if verdict.stop:
            print(f"[train] STOPPING: {verdict.reason}")
            log.log(kind="stopped", epoch=epoch, reason=verdict.reason)
            beat.beat(force=True, phase="stopped", reason=verdict.reason)
            break

    beat.beat(force=True, phase="done", epoch=state.epoch)
    log.close()
    return state
