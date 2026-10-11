"""The challenge's Soft-Threshold Relative Absolute Error, and a loss that matches it.

**The target is an interval, not a point.** The organisers score the distance
between a prediction and the *credible-interval bound* of the fitted dose-response
curve, and **a prediction anywhere inside the interval scores zero**. Low-activity
compounds (pIC50 < 4) fall below the lowest tested dose and carry wide intervals,
so this exists to avoid penalising models for noise the assay itself cannot
resolve.

Two consequences, and both change what we should do:

* **Optimising MAE against the point estimate is the wrong objective.** It spends
  capacity chasing the centre of intervals that are, for scoring purposes, free.
  CYP3A4 has 40% of its labels below pIC50 4 and a median interval width of 0.38,
  so that is a large fraction of the primary endpoint.
* **The loss should be interval-aware** — zero inside, hinge outside. That is what
  `interval_loss` below does, and it is the only part of our stack that needed a
  new objective rather than a new feature.

`st_rae` reimplements the published description. The organisers' exact
normalisation is not public, so the denominator here is the mean absolute
deviation of the truth — the standard RAE denominator. **Our absolute values are
therefore not guaranteed to match the leaderboard's**; they are valid for ranking
our own candidates against each other, which is what we use them for.
"""

from __future__ import annotations

import numpy as np

# **Every run artifact must record which metric scored it.** The denominator
# changed on 2026-10-05, and `cyp_long.py` caches on file existence alone, so
# eight depth runs were silently pooled from five old-metric and three
# official-metric JSONs -- and the paired test averaged across both. Deleting the
# output directory was supposed to prevent that and did not. Bump this string
# whenever the metric changes; aggregators refuse to compare runs whose ids differ,
# which turns a silent mixture into a loud one.
METRIC_ID = "official-strae-v1"


def soft_threshold_error(
    y_pred: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> np.ndarray:
    """Distance to the nearest credible-interval bound; zero inside the interval."""
    below = np.clip(lo - y_pred, 0.0, None)
    above = np.clip(y_pred - hi, 0.0, None)
    return below + above


def st_rae(
    y_pred: np.ndarray, y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """Soft-threshold relative absolute error, **as the organisers compute it**.

    Mirrors `rae_soft_threshold_absolute_error` in OpenADMET's
    blind-challenge-template: the naive baseline (a constant predictor at
    `mean(y_true)`) goes through the *same* soft-thresholding as the model, so
    1.0 means "as good as predicting the mean".
    """
    keep = np.isfinite(y_pred) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y_true)
    if keep.sum() == 0:
        return float("nan")
    p, y, l, h = y_pred[keep], y_true[keep], lo[keep], hi[keep]
    err = soft_threshold_error(p, l, h).sum()
    # The mean is taken over the scored subset, exactly as the official scorer
    # does after its merge-and-dropna.
    base = soft_threshold_error(np.full(y.shape, y.mean()), l, h).sum()
    return float(err / base) if base > 0 else float("nan")


def st_rae_mad(
    y_pred: np.ndarray, y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> float:
    """The superseded denominator: `sum|y - mean(y)|`.

    Kept so that numbers recorded in PROGRESS.md before 2026-10-05 can be
    reproduced. Do not use it for any new decision -- it reads roughly half the
    official score on this dataset.
    """
    keep = np.isfinite(y_pred) & np.isfinite(lo) & np.isfinite(hi) & np.isfinite(y_true)
    if keep.sum() == 0:
        return float("nan")
    err = soft_threshold_error(y_pred[keep], lo[keep], hi[keep]).sum()
    denom = np.abs(y_true[keep] - y_true[keep].mean()).sum()
    return float(err / denom) if denom > 0 else float("nan")


def macro_st_rae(preds: dict, truth: dict) -> tuple[float, dict]:
    """MA-ST-RAE across endpoints. `truth[e]` is `(y, lo, hi)`."""
    per = {}
    for e, (y, lo, hi) in truth.items():
        if e in preds:
            per[e] = st_rae(np.asarray(preds[e]), np.asarray(y), np.asarray(lo), np.asarray(hi))
    vals = [v for v in per.values() if np.isfinite(v)]
    return (float(np.mean(vals)) if vals else float("nan")), per


def interval_loss(pred, lo, hi, mask, huber_delta: float = 1.0):
    """Torch loss mirroring the metric: zero inside the interval, Huber outside.

    Huber rather than squared error outside the band because a handful of very
    wrong predictions should not dominate the gradient — the metric is an
    *absolute* error, so an L2 loss would optimise something the leaderboard does
    not measure.
    """
    import torch

    below = torch.clamp(lo - pred, min=0.0)
    above = torch.clamp(pred - hi, min=0.0)
    dist = below + above
    quad = torch.minimum(dist, torch.full_like(dist, huber_delta))
    loss = 0.5 * quad.pow(2) + huber_delta * (dist - quad)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom
