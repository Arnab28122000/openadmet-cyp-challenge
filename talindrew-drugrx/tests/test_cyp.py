"""Known-answer fixtures for the CYP challenge data layer and metric.

**The defect these exist to catch was a column filter, not a computation.** The
loader selected its auxiliary heads with
`endswith("_TDI_condition") and "Emax" in c`, which matched 4 of the 8 Emax
columns and silently dropped 6 usable heads carrying 9,223 labels — while
producing a head count (14) that matched the docstring, so nothing read as wrong.
The dropped set included `CYP*_EmaxVsPosCtrl_direct_inhibition`, whose per-enzyme
label counts are *exactly* the primary pIC50 counts because it is the same
dose-response fit: the most on-target auxiliary signal in the release.

That is the same shape as every other defect this programme has found — a
**selection or interpretation** step failing toward a plausible answer. So the
head list is now declared in the library and these tests import the same
declaration the loader uses. A test that re-derived the list would guard nothing.

Everything here is offline and deterministic: the defect lived in an expression
over column names, which is exactly what can be checked without the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drugrx.cyp.data import (
    AUX_EMAX,
    AUX_EMAX_DIRECT,
    AUX_IS_TDI,
    AUX_LOG2FC,
    AUX_TDI,
    AUXILIARY,
    CYPS,
    PRIMARY,
    _require,
)
from drugrx.cyp.metric import interval_loss, soft_threshold_error, st_rae

# ── the head declaration ─────────────────────────────────────────────


def test_four_cyps_and_four_primary_heads():
    assert len(CYPS) == 4
    assert len(PRIMARY) == 4
    assert PRIMARY == [f"{c}_pIC50_direct_inhibition" for c in CYPS]


def test_twenty_auxiliary_heads_no_duplicates():
    """20, not 14. The loose filter produced 14 and looked correct."""
    assert len(AUXILIARY) == 20
    assert len(set(AUXILIARY)) == 20, "duplicate auxiliary head would double-weight it"


def test_emax_direct_inhibition_heads_are_present():
    """The regression test for the actual bug.

    These four were dropped by `endswith("_TDI_condition")`. They carry 6,525
    labels — one per primary label — and are the Emax values measured under the
    primary assay condition.
    """
    for c in CYPS:
        col = f"{c}_EmaxVsPosCtrl_direct_inhibition"
        assert col in AUX_EMAX_DIRECT
        assert col in AUXILIARY, f"{col} dropped from the auxiliary heads again"


def test_all_four_is_tdi_flags_are_present():
    """CYP1A2 and CYP2C9 `is_TDI` exist only in the Emax file and were missed."""
    assert AUX_IS_TDI == [f"{c}_is_TDI" for c in CYPS]
    for c in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
        assert f"{c}_is_TDI" in AUXILIARY


def test_auxiliary_excludes_interval_and_std_variants():
    """Guard against over-correcting the fix.

    The original filter was trying to exclude `_conf_low`/`_conf_high`/`_std`.
    Declaring columns explicitly must not quietly readmit them as heads — a
    `_std` column as a regression target is a different quantity entirely.
    """
    for col in AUXILIARY:
        assert not col.endswith(("_conf_low", "_conf_high", "_std")), col


def test_primary_and_auxiliary_are_disjoint():
    """The direct-inhibition pIC50 appears in two files with identical counts.

    Re-adding it as an auxiliary head would train the same measurement twice and
    silently double-weight those compounds.
    """
    assert set(PRIMARY).isdisjoint(set(AUXILIARY))


def test_auxiliary_groups_partition_the_head_list():
    """The declared groups must account for every head, with none left over."""
    grouped = [*AUX_TDI, *AUX_EMAX, *AUX_EMAX_DIRECT, *AUX_IS_TDI, *AUX_LOG2FC]
    assert grouped == AUXILIARY
    assert all(len(g) == 4 for g in (AUX_TDI, AUX_EMAX, AUX_EMAX_DIRECT,
                                     AUX_IS_TDI, AUX_LOG2FC))


# ── _require: the loud-failure guarantee ─────────────────────────────


def test_require_raises_on_a_missing_column():
    """An upstream rename must break the run, not shrink it."""
    frame = pd.DataFrame({"SMILES": ["C"], "present": [1.0]})
    with pytest.raises(KeyError, match="absent"):
        _require(frame, ["present", "renamed_upstream"], "fixture")


def test_require_passes_when_every_column_is_present():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
    _require(frame, ["a", "b"], "fixture")  # must not raise


def test_require_names_the_missing_columns():
    """The message has to say which column went, or it cannot be acted on."""
    frame = pd.DataFrame({"a": [1.0]})
    with pytest.raises(KeyError) as excinfo:
        _require(frame, ["a", "gone_1", "gone_2"], "fixture")
    assert "gone_1" in str(excinfo.value) and "gone_2" in str(excinfo.value)


# ── the metric: zero inside the interval ─────────────────────────────


def test_prediction_inside_the_interval_scores_zero():
    """The whole point of the soft threshold. If this breaks, we optimise noise."""
    lo = np.array([4.0, 5.0, 6.0])
    hi = np.array([4.5, 5.5, 6.5])
    pred = np.array([4.2, 5.5, 6.0])      # inside, on the upper bound, on the lower
    assert np.allclose(soft_threshold_error(pred, lo, hi), 0.0)


def test_prediction_outside_scores_distance_to_the_nearer_bound():
    lo = np.array([4.0, 4.0])
    hi = np.array([5.0, 5.0])
    pred = np.array([3.4, 5.25])          # 0.6 below lo, 0.25 above hi
    assert np.allclose(soft_threshold_error(pred, lo, hi), [0.6, 0.25])


def test_st_rae_is_zero_when_everything_lands_inside():
    y = np.array([4.0, 6.0, 8.0])
    lo, hi = y - 0.2, y + 0.2
    assert st_rae(y.copy(), y, lo, hi) == 0.0


def test_st_rae_ignores_rows_it_cannot_score():
    """A NaN truth or bound must drop the row, not poison the whole endpoint."""
    y = np.array([4.0, np.nan, 8.0])
    lo = np.array([3.9, 1.0, 7.9])
    hi = np.array([4.1, 2.0, 8.1])
    pred = np.array([4.0, 99.0, 8.0])
    assert st_rae(pred, y, lo, hi) == 0.0


def test_st_rae_is_nan_when_no_row_is_scorable():
    """An endpoint with nothing to score must not report a confident 0.0."""
    nan3 = np.full(3, np.nan)
    assert np.isnan(st_rae(nan3, nan3, nan3, nan3))


def test_st_rae_grows_with_error():
    y = np.array([4.0, 6.0, 8.0])
    lo, hi = y - 0.1, y + 0.1
    near = st_rae(y + 0.5, y, lo, hi)
    far = st_rae(y + 2.0, y, lo, hi)
    assert 0.0 < near < far


# ── the loss mirrors the metric ──────────────────────────────────────


def test_interval_loss_is_zero_inside_the_band():
    torch = pytest.importorskip("torch")
    lo = torch.tensor([[4.0, 5.0]])
    hi = torch.tensor([[4.5, 5.5]])
    pred = torch.tensor([[4.2, 5.3]])
    mask = torch.ones_like(pred)
    assert float(interval_loss(pred, lo, hi, mask)) == pytest.approx(0.0)


def test_interval_loss_penalises_outside_and_respects_the_mask():
    torch = pytest.importorskip("torch")
    lo = torch.tensor([[4.0, 5.0]])
    hi = torch.tensor([[4.5, 5.5]])
    pred = torch.tensor([[1.0, 5.3]])          # first is far below lo
    unmasked = float(interval_loss(pred, lo, hi, torch.ones_like(pred)))
    masked = float(interval_loss(pred, lo, hi, torch.tensor([[0.0, 1.0]])))
    assert unmasked > 0.0
    assert masked == pytest.approx(0.0), "masking out the bad row must silence it"


def test_interval_loss_is_huber_not_squared_far_from_the_band():
    """A handful of very wrong predictions must not dominate the gradient.

    The metric is an absolute error, so beyond `huber_delta` the loss has to grow
    linearly. Squared growth would optimise something the leaderboard does not
    measure.
    """
    torch = pytest.importorskip("torch")
    lo = torch.tensor([[4.0]])
    hi = torch.tensor([[4.0]])
    mask = torch.ones_like(lo)
    at_10 = float(interval_loss(torch.tensor([[14.0]]), lo, hi, mask))
    at_20 = float(interval_loss(torch.tensor([[24.0]]), lo, hi, mask))
    # Linear region: doubling the distance must roughly double the loss, not 4x it.
    assert 1.8 < at_20 / at_10 < 2.2
