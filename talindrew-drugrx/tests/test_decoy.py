"""The decoy-target control, pinned on synthetic rankings.

This is the control that asks whether a screen is about its target at all, so the
statistic it reports has to be right before it is pointed at anything real. What these
fix:

* **The null band is reported, not implied.** At the 20-to-30-compound sets this
  pipeline runs, Spearman's sampling band under independence is ±0.4 or wider. A
  correlation of 0.35 at n = 20 is *not* "some target specificity"; it is noise, and a
  control that let someone read it as a result would be worse than no control.
* **The uncomfortable verdict has a name.** "Ranking is target-independent: this is a
  compound-quality prior" has to be reachable, or the control can only ever confirm.
* **Ranks, not scores.** Two runs' scores are not on a common scale; comparing them
  would measure the scale.
* **Composition is preserved exactly** by the shuffle arm, so anything depending only
  on amino-acid content is unchanged and the fold is the only thing destroyed.

T2 applies: these have been seen to fail. Comparing scores instead of ranks, or
dropping the null band, each breaks a named test below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from drugrx.decoy import (  # noqa: E402
    LYSOZYME,
    compare_rankings,
    ranking_from_candidates,
    shuffled_target,
    unrelated_target,
)

TARGET = (
    "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
    "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
)


# ── the decoys themselves ────────────────────────────────────────────────

def test_the_shuffle_preserves_composition_exactly() -> None:
    """Anything that depends only on amino-acid content must be unchanged.

    If the shuffle altered composition, a difference between the arms could be a
    composition effect rather than a structure effect, and the control would be
    measuring the wrong thing.
    """
    from collections import Counter

    decoy = shuffled_target(TARGET, seed=0)
    assert len(decoy) == len(TARGET)
    assert Counter(decoy) == Counter(TARGET)


def test_the_shuffle_actually_moves_the_sequence() -> None:
    decoy = shuffled_target(TARGET, seed=0)
    assert decoy != TARGET
    differing = sum(1 for a, b in zip(decoy, TARGET) if a != b)
    assert differing > len(TARGET) * 0.5, "barely shuffled"


def test_the_shuffle_is_reproducible() -> None:
    """A control that cannot be reproduced is an anecdote."""
    assert shuffled_target(TARGET, seed=7) == shuffled_target(TARGET, seed=7)
    assert shuffled_target(TARGET, seed=7) != shuffled_target(TARGET, seed=8)


def test_the_unrelated_decoy_is_a_real_protein_not_a_scramble() -> None:
    """The sharper arm: does *this* structure matter, not just any structure.

    A scramble alone could only ever show that unfoldable inputs behave oddly.
    """
    decoy = unrelated_target(TARGET)
    assert decoy == LYSOZYME
    assert len(decoy) == 129
    assert set(decoy) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert decoy != TARGET


# ── the comparison ───────────────────────────────────────────────────────

def test_an_identical_ranking_reads_as_target_independent() -> None:
    """The decisive case, and the one the control exists to be able to report."""
    order = [f"c{i}" for i in range(30)]
    result = compare_rankings(order, order)
    assert result.spearman == pytest.approx(1.0)
    assert result.distinguishable
    assert "compound-quality prior" in result.verdict


def test_a_reversed_ranking_is_target_specific() -> None:
    order = [f"c{i}" for i in range(30)]
    result = compare_rankings(order, list(reversed(order)))
    assert result.spearman == pytest.approx(-1.0)
    assert result.verdict == "ranking is target-specific"


def test_a_small_correlation_is_reported_as_indistinguishable() -> None:
    """The mutation target, and the whole reason the band is computed.

    At n = 20 the band is ±0.45. A rho of about 0.3 is noise; a control that called it
    "some target specificity" would manufacture a result from a 20-compound screen —
    the same error as reading a 15/21 win tally.
    """
    import numpy as np

    order = [f"c{i}" for i in range(20)]
    rng = np.random.default_rng(3)
    shuffled = list(order)
    rng.shuffle(shuffled)

    result = compare_rankings(order, shuffled)
    assert result.null_band == pytest.approx(1.96 / (19 ** 0.5), rel=1e-6)
    assert result.null_band > 0.4, "the band at this n is wide, and that is the point"
    if abs(result.spearman) < result.null_band:
        assert not result.distinguishable
        assert "indistinguishable from chance" in result.verdict


def test_the_band_narrows_as_the_screen_grows() -> None:
    """More compounds is the only thing that makes this control sharper."""
    small = compare_rankings([f"c{i}" for i in range(20)], [f"c{i}" for i in range(20)])
    large = compare_rankings([f"c{i}" for i in range(500)], [f"c{i}" for i in range(500)])
    assert large.null_band < small.null_band
    assert large.null_band < 0.1


def test_only_shared_compounds_are_compared() -> None:
    """A comparison across different molecules is not a comparison."""
    real = ["a", "b", "c", "d", "e", "f"]
    decoy = ["c", "b", "a", "z", "y"]
    result = compare_rankings(real, decoy)
    assert result.shared == 3
    assert result.n == 3


def test_too_few_shared_compounds_refuses_a_verdict() -> None:
    result = compare_rankings(["a", "b", "c"], ["a", "b", "q"])
    assert "too few shared compounds" in result.verdict


def test_top_k_overlap_is_measured_on_the_shortlist() -> None:
    """The shortlist is what anyone acts on, so it gets its own number.

    A rank correlation can be moderate while the top 10 is identical, which is the
    case that matters: nobody orders compounds 40 through 60.
    """
    real = [f"c{i}" for i in range(30)]
    decoy = [f"c{i}" for i in range(10)] + [f"c{i}" for i in reversed(range(10, 30))]
    result = compare_rankings(real, decoy, top_k=10)
    assert result.top_k == 10
    assert result.top_k_overlap == pytest.approx(1.0)


# ── reading a FoldRx ranking ─────────────────────────────────────────────

def test_a_candidate_list_is_sorted_rather_than_trusted() -> None:
    """The engine sorts already; assuming it would silently compare two arbitrary
    orders if that ever changed."""
    candidates = [
        {"id": "b", "composite_score": 0.4},
        {"id": "a", "composite_score": 0.9},
        {"id": "c", "composite_score": 0.1},
    ]
    assert ranking_from_candidates(candidates) == ["a", "b", "c"]


def test_a_candidate_without_an_id_is_dropped_not_guessed() -> None:
    candidates = [{"id": "a", "composite_score": 0.9}, {"composite_score": 1.0}]
    assert ranking_from_candidates(candidates) == ["a"]


def test_a_missing_score_sorts_last_rather_than_crashing() -> None:
    candidates = [{"id": "a"}, {"id": "b", "composite_score": 0.5}]
    assert ranking_from_candidates(candidates) == ["b", "a"]


def test_the_comparison_is_on_ranks_not_scores() -> None:
    """Two runs' scores are not on a common scale.

    Here the decoy's scores are the real run's divided by a thousand. The *order* is
    identical, so the correlation must be 1.0; anything that compared scores would
    see two unrelated magnitudes.
    """
    real = ranking_from_candidates(
        [{"id": "a", "composite_score": 0.9}, {"id": "b", "composite_score": 0.5}])
    decoy = ranking_from_candidates(
        [{"id": "a", "composite_score": 0.0009}, {"id": "b", "composite_score": 0.0005}])
    assert compare_rankings(real, decoy).spearman == pytest.approx(1.0)
