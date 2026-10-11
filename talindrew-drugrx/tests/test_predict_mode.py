"""Prediction must be deterministic, and this is the test that says so.

The defect these guard against shipped: `predict_ensemble`'s docstring claimed
"at prediction time the modules are in eval mode", but nothing called `.eval()`
and a freshly built module has `training=True`. So every number this programme
produced from a neural model — early-stopping validation scores, H3/H3b/H6 test
scores, H7a's arms, the shipped artifact's own card — was computed with dropout
active at 0.1.

It failed toward a readable answer, like the other five: the predictions were
*plausible*, merely noisier than they should be, which penalised the neural arm
against gradient boosting and put dropout noise inside the member spread that
Phase 4 serves as calibrated uncertainty.

T2 applies: a fixture nobody has seen fail is not known to work. Reintroducing
the defect — deleting the `base.train(mc_dropout)` line, or defaulting
`mc_dropout` to `True` — must fail `test_two_predictions_are_identical`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")
nn = torch.nn

from drugrx.fast_train import predict_ensemble  # noqa: E402


class Dropper(nn.Module):
    """Dropout heavy enough that train/eval cannot be confused for noise."""

    def __init__(self, in_dim: int = 8, n_tasks: int = 3) -> None:
        super().__init__()
        self.drop = nn.Dropout(0.5)
        self.head = nn.Linear(in_dim, n_tasks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.drop(x))


def _stacked(members: int = 4, in_dim: int = 8, n_tasks: int = 3):
    from torch.func import stack_module_state

    models = []
    for k in range(members):
        torch.manual_seed(k)
        models.append(Dropper(in_dim, n_tasks))
    params, buffers = stack_module_state(models)
    base = models[0].to("meta")
    return base, params, buffers


def test_two_predictions_are_identical() -> None:
    """The whole point: a prediction is a function of the weights, not a sample.

    This is the mutation target. With `base.train(mc_dropout)` removed, the two
    calls draw different dropout masks and the assertion fails.
    """
    base, params, buffers = _stacked()
    torch.manual_seed(0)
    x = torch.randn(32, 8)

    first = predict_ensemble(base, params, buffers, x)
    second = predict_ensemble(base, params, buffers, x)

    assert torch.equal(first, second), "prediction is not deterministic — dropout is on"


def test_mc_dropout_is_available_but_must_be_asked_for() -> None:
    """The old behaviour is still reachable, and only deliberately.

    MC-dropout is a legitimate uncertainty estimator; it was simply never what
    any caller here intended. Keeping it behind a flag means the fix does not
    quietly remove a capability.
    """
    base, params, buffers = _stacked()
    torch.manual_seed(0)
    x = torch.randn(32, 8)

    first = predict_ensemble(base, params, buffers, x, mc_dropout=True)
    second = predict_ensemble(base, params, buffers, x, mc_dropout=True)

    assert not torch.equal(first, second), "mc_dropout=True should resample the mask"


def test_the_spread_is_member_disagreement_not_dropout_noise() -> None:
    """The spread Phase 4 serves as uncertainty must come from the ensemble.

    With dropout on, a *single* member predicted twice disagrees with itself, so
    the across-member standard deviation mixes two unrelated things. One member
    repeated must have zero spread; four different members must not.
    """
    base, params, buffers = _stacked(members=4)
    torch.manual_seed(0)
    x = torch.randn(64, 8)

    # One member, duplicated: any spread here is noise, not disagreement.
    one = {k: v[:1].repeat(4, *([1] * (v.dim() - 1))) for k, v in params.items()}
    one_buffers = {k: v[:1].repeat(4, *([1] * (v.dim() - 1))) for k, v in buffers.items()}
    same = predict_ensemble(base, one, one_buffers, x)
    assert float(same.std(0).max()) == pytest.approx(0.0, abs=1e-6)

    # Four genuinely different initialisations must disagree.
    many = predict_ensemble(base, params, buffers, x)
    assert float(many.std(0).mean()) > 1e-3
