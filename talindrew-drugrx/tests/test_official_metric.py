"""Pin our ST-RAE to the organisers' implementation.

The denominator was wrong for this programme's whole first phase and read roughly
half the official score, so the fix gets a test that compares against the
organisers' own code path rather than against our reading of it.

Reference: OpenADMET/blind-challenge-template,
`backend/custom_scoring_functions.py::rae_soft_threshold_absolute_error`.
"""
from __future__ import annotations

import numpy as np
import pytest

from drugrx.cyp.metric import soft_threshold_error, st_rae, st_rae_mad


def official(y_true, y_pred, lo, hi):
    """Transcription of the organisers' function, kept deliberately verbatim."""
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    above_upper = np.clip(y_pred - hi, a_min=0, a_max=None)
    below_lower = np.clip(lo - y_pred, a_min=0, a_max=None)
    soft_abs_error = above_upper + below_lower
    mean_true = np.mean(y_true)
    baseline_above_upper = np.clip(mean_true - hi, a_min=0, a_max=None)
    baseline_below_lower = np.clip(lo - mean_true, a_min=0, a_max=None)
    soft_baseline_error = baseline_above_upper + baseline_below_lower
    return np.sum(soft_abs_error) / np.sum(soft_baseline_error)


def _case(rng, n=200):
    y = rng.normal(5.0, 1.2, n)
    w = np.abs(rng.normal(0.4, 0.3, n))          # asymmetric, variable widths
    lo = y - w * rng.uniform(0.2, 0.8, n)
    hi = y + w * rng.uniform(0.2, 0.8, n)
    p = y + rng.normal(0, 0.8, n)
    return y, p, lo, hi


@pytest.mark.parametrize("seed", range(40))
def test_matches_official_implementation(seed):
    rng = np.random.default_rng(seed)
    y, p, lo, hi = _case(rng)
    assert st_rae(p, y, lo, hi) == pytest.approx(official(y, p, lo, hi), rel=1e-12)


def test_mean_predictor_scores_exactly_one():
    """The definition that was missing: 1.0 == as good as predicting the mean.

    This is the property the old denominator destroyed, and the reason our
    internal numbers looked twice as good as the leaderboard's.
    """
    rng = np.random.default_rng(0)
    y, _, lo, hi = _case(rng)
    assert st_rae(np.full(y.shape, y.mean()), y, lo, hi) == pytest.approx(1.0)


def test_old_denominator_is_always_optimistic():
    """The superseded metric can only ever read lower (better). This is a theorem.

    Because the true value lies inside its own interval, the mean predictor's
    *soft* error is pointwise no greater than its plain error:
    `dist(mean, [lo, hi]) <= |mean - y|` whenever `lo <= y <= hi`. Summing gives
    `official_denominator <= sum|y - mean|`, hence `st_rae >= st_rae_mad`.

    The first version of this test also asserted the gap exceeded 20%. That is a
    property of the *real* interval geometry (where many labels cluster near the
    mean with wide intervals), not of these synthetic draws, so asserting it here
    was testing the generator rather than the metric.
    """
    for seed in range(40):
        rng = np.random.default_rng(seed)
        y, p, lo, hi = _case(rng)
        assert st_rae_mad(p, y, lo, hi) <= st_rae(p, y, lo, hi) + 1e-12, seed


def test_mean_soft_error_never_exceeds_plain_error():
    """The pointwise lemma the inequality above rests on."""
    for seed in range(40):
        rng = np.random.default_rng(seed)
        y, _, lo, hi = _case(rng)
        assert (lo <= y).all() and (y <= hi).all()      # the premise
        m = np.full(y.shape, y.mean())
        assert (soft_threshold_error(m, lo, hi) <= np.abs(m - y) + 1e-12).all(), seed


def test_prediction_inside_interval_is_free():
    # The baseline must NOT also be free, or the ratio is 0/0 -- see the next test.
    y = np.array([4.0, 7.0]); lo = np.array([3.9, 6.9]); hi = np.array([4.1, 7.1])
    assert soft_threshold_error(np.array([4.0, 7.0]), lo, hi).sum() == 0.0
    assert st_rae(np.array([4.0, 7.0]), y, lo, hi) == pytest.approx(0.0)


def test_undefined_when_the_mean_predictor_is_itself_perfect():
    """A real 0/0 in the official metric, not a quirk of ours.

    If `mean(y)` falls inside every interval the denominator is zero, and the
    organisers' expression is `0/0`. We return NaN rather than inventing a value;
    `macro_st_rae` already drops non-finite endpoints. Documented because a test
    asserting 0.0 here would be asserting something the official scorer does not
    do -- which is how the first version of this file failed.
    """
    y = np.array([5.0, 6.0]); lo = np.array([4.5, 5.5]); hi = np.array([5.5, 6.5])
    assert not np.isfinite(st_rae(np.array([5.2, 6.1]), y, lo, hi))


def test_nan_predictions_are_dropped_not_propagated():
    rng = np.random.default_rng(1)
    y, p, lo, hi = _case(rng)
    p2 = p.copy(); p2[:10] = np.nan
    v = st_rae(p2, y, lo, hi)
    assert np.isfinite(v)
