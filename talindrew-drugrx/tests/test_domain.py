"""Known-answer fixtures for the applicability domain (T1).

The domain check is an *interpretation* layer, which this programme's own history
says is the part whose failures look like results: five of the six defects found so
far lived in code that prepared or labelled rather than computed, and every one of
them failed toward a readable answer. So the obligations are the ones PLAN.md
already records.

What is pinned here:

* a molecule **in** the reference set is in domain (similarity 1.0);
* a molecule structurally unlike it is **out**, and the threshold is what decides
  that rather than a hand-picked constant;
* the threshold is **calibrated against a background**, so a dense reference set
  does not make everything look in-domain (T6 — the Avalon lesson, which cost a
  whole audit row);
* alignment is preserved for unparseable input, because the caller zips these rows
  onto compounds and a silent drop misaligns everything after it;
* `max_tanimoto` agrees with RDKit's own BulkTanimotoSimilarity, so the packed-bit
  arithmetic is not quietly its own metric.

T2 applies: these must be seen to fail. Replacing the calibrated threshold with a
fixed 0.3 fails `test_the_threshold_is_calibrated_not_fixed`; dropping unparseable
rows instead of placeholding them fails `test_unparseable_input_keeps_alignment`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit")

from drugrx.domain import (  # noqa: E402
    N_BITS,
    DomainReference,
    build_reference,
    fingerprints,
    max_tanimoto,
    membership,
)

#: A small congeneric series — straight-chain alcohols. Internally similar, so it
#: makes a believable "training set" for a narrow endpoint.
ALCOHOLS = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO"]

#: Stands in for a validation fold: the same chemistry as the training set, held
#: out. This is what the threshold is calibrated against, so the fixtures exercise
#: the real contract — a bar set by "molecules like the ones we measured on", not
#: by a percentile of random chemistry.
CALIBRATION = ["CCCCCCCCO", "CCCCCCCCCO", "CCCCCCCCCCO", "CC(C)CCO", "CCC(C)CO",
               "CCCCC(C)O", "OCCCCCCCCCCO", "CC(O)CCCC", "CCCCCCCCCCCO",
               "CCCCCCCCCCCCO", "CC(C)(C)CCO", "CCCC(O)CCC", "CCCCCCO",
               "OCCCCCCCO", "CCCCCCCCCCCCCO", "CC(C)CO", "CCC(O)CC",
               "CCCCCCCCCCCCCCO", "OCCCCCCCCCO", "CCCCCCCCCCCCCCCO",
               "CCCCCCCCCCCCCCCCO", "CC(C)CCCO", "CCCCCCCCCCCCCCCCCO"]

#: Chemistry from somewhere else entirely: aromatics, heterocycles, a sugar.
ELSEWHERE = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCNCC1",
    "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "CC(=O)Nc1ccc(O)cc1", "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "c1ccc(cc1)S(=O)(=O)N", "C1=CC2=C(C=C1)N=CC=C2", "FC(F)(F)c1ccccc1",
]


def test_fingerprints_are_packed_and_report_what_survived() -> None:
    packed, kept = fingerprints(["CCO", "not-a-molecule", "c1ccccc1"])
    assert packed.shape == (2, N_BITS // 8)
    assert packed.dtype == np.uint8
    assert kept.tolist() == [0, 2], "kept must index back into the input"


def test_max_tanimoto_agrees_with_rdkit() -> None:
    """The packed-bit arithmetic must be Tanimoto, not something close to it.

    Without this, a transposed popcount or an off-by-one union would give a
    plausible monotone score that ranks correctly and calibrates wrongly.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    query, reference = ELSEWHERE[:4], ALCOHOLS

    ours = max_tanimoto(fingerprints(query)[0], fingerprints(reference)[0])
    ref_fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in reference]
    theirs = [
        max(DataStructs.BulkTanimotoSimilarity(
            gen.GetFingerprint(Chem.MolFromSmiles(s)), ref_fps))
        for s in query
    ]
    assert ours.tolist() == pytest.approx(theirs, abs=1e-6)


def test_a_molecule_in_the_reference_set_is_in_domain() -> None:
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    rows = membership(ref, ["CCCO"])
    assert rows[0]["nn_similarity"] == pytest.approx(1.0)
    assert rows[0]["in_domain"] is True


def test_unlike_chemistry_is_out_of_domain() -> None:
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    rows = membership(ref, ["Cn1cnc2c1c(=O)[nH]c(=O)n2C"])  # caffeine
    assert rows[0]["in_domain"] is False
    assert rows[0]["nn_similarity"] < ref.threshold


def test_the_threshold_follows_the_calibration_set_not_the_background() -> None:
    """The point of the whole module, and the mutation target.

    The bar is a quantile of the **validation** fold's own nearest-neighbour
    distribution, so it must move when that distribution moves and stay put when
    the random background changes. A threshold driven by the background is the
    design this replaced: calibrated, defensible, and still flagging nine test
    molecules in ten (see the module docstring).
    """
    near = build_reference("near", ALCOHOLS, ELSEWHERE, CALIBRATION)
    # A calibration set of unrelated chemistry sits much further from the
    # alcohol training set, so the bar it implies is lower.
    far = build_reference("far", ALCOHOLS, ELSEWHERE, ELSEWHERE * 3)

    assert far.threshold < near.threshold, (
        "a calibration set further from training must give a lower bar; "
        f"near={near.threshold:.3f} far={far.threshold:.3f}"
    )
    assert far.calibration_median < near.calibration_median

    # And the background must *not* move it: same calibration, different
    # background, same threshold.
    other_background = build_reference("near", ALCOHOLS, ELSEWHERE[:5], CALIBRATION)
    assert other_background.threshold == pytest.approx(near.threshold)
    # The background genuinely changed — so the equality above is independence,
    # not two identical inputs producing the same answer.
    assert other_background.background_n != near.background_n
    assert other_background.background_mean != pytest.approx(near.background_mean)


def test_a_reference_records_both_populations_it_was_built_from() -> None:
    """A threshold with no provenance is a fixed cut wearing a calibration's name.

    Both populations are recorded: the calibration set that *set* the bar, and the
    random background that makes the resulting similarity number readable. The gap
    between `background_median` and `threshold` is the honest measure of how much
    proximity the training set actually buys.
    """
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION, quantile=10.0)
    assert ref.quantile == 10.0
    assert ref.percentile == 10.0, "the back-compatible alias must agree"
    assert ref.calibration_n == len(CALIBRATION)
    assert ref.background_n == len(ELSEWHERE)
    assert ref.n_reference == len(ALCOHOLS)
    assert 0.0 <= ref.threshold <= 1.0
    assert 0.0 <= ref.calibration_median <= 1.0


def test_a_lower_quantile_is_an_easier_bar() -> None:
    """Quantile semantics, which inverted when the calibration source changed.

    The bar is now the `quantile`-th percentile of the *calibration* fold, so a
    **lower** quantile is a **lower** threshold — q=1 admits all but the single
    least typical percent, q=50 admits only the closer half. Under the old
    background-percentile design the relationship ran the other way, which is
    exactly the kind of silent inversion `abs(shift)` already cost this project
    once.
    """
    lax = build_reference("a", ALCOHOLS, ELSEWHERE, CALIBRATION, quantile=1.0)
    strict = build_reference("a", ALCOHOLS, ELSEWHERE, CALIBRATION, quantile=50.0)
    assert lax.threshold <= strict.threshold


def test_too_few_calibration_molecules_is_refused() -> None:
    """A quantile over a handful of molecules is one compound setting the bar."""
    with pytest.raises(ValueError, match="calibration molecules"):
        build_reference("a", ALCOHOLS, ELSEWHERE, CALIBRATION[:5])


def test_unparseable_input_keeps_alignment() -> None:
    """The second mutation target: dropping a bad row misaligns every later one."""
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    rows = membership(ref, ["not-a-molecule", "CCCO", "", "c1ccccc1"])

    assert len(rows) == 4
    assert rows[0] == {"in_domain": False, "nn_similarity": None,
                       "measured_percentile": None, "reason": "unparseable"}
    assert rows[2]["reason"] == "unparseable"
    assert rows[1]["nn_similarity"] == pytest.approx(1.0), "alignment slipped"
    assert rows[3]["nn_similarity"] is not None


def test_the_fixed_cut_is_still_reported_for_comparability() -> None:
    """H9's 18% and H10's 11.2% are fixed-0.3 figures; they must stay comparable."""
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    rows = membership(ref, ["CCCO", "Cn1cnc2c1c(=O)[nH]c(=O)n2C"])
    assert rows[0]["above_0_3"] is True
    assert rows[1]["above_0_3"] is False


def test_a_reference_round_trips_through_npz() -> None:
    """The artifact carries this; a lossy round trip would change every verdict."""
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    blob = ref.to_npz()
    back = DomainReference.from_npz("alcohols", blob["packed"], blob["meta"],
                                    blob["calibration_sims"])

    assert back.threshold == pytest.approx(ref.threshold)
    assert back.percentile == ref.percentile
    assert back.n_reference == ref.n_reference
    assert membership(back, ["CCCO"])[0] == membership(ref, ["CCCO"])[0]


def test_chunking_does_not_change_the_answer() -> None:
    """A chunk boundary is where an off-by-one hides, and it would look like noise."""
    query = fingerprints(ELSEWHERE)[0]
    reference = fingerprints(ALCOHOLS)[0]
    whole = max_tanimoto(query, reference, chunk=1024)
    split = max_tanimoto(query, reference, chunk=3)
    assert whole.tolist() == pytest.approx(split.tolist(), abs=1e-7)


def test_an_empty_query_is_empty_not_an_error() -> None:
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    assert membership(ref, []) == []


def test_a_reference_with_nothing_parseable_refuses_to_build() -> None:
    with pytest.raises(ValueError, match="no parseable training molecules"):
        build_reference("broken", ["not-a-molecule"], ELSEWHERE, CALIBRATION)


# ── the background must not contain the reference set ────────────────────

def test_the_background_excludes_the_reference_sets_own_molecules() -> None:
    """Without this, a threshold measures dataset overlap rather than chemistry.

    TDC's datasets share molecules heavily, so an endpoint's pooled background is
    full of its own training set — each member scoring 1.0 against itself, which
    drags the background up and the bar with it.
    """
    from drugrx.domain import background_excluding

    key_of = {s: f"K{i}" for i, s in enumerate(["a", "b", "c", "d"])}
    assert background_excluding(["a", "b", "c", "d"], ["b", "d"], key_of) == ["a", "c"]


def test_the_exclusion_matches_salt_forms_not_strings() -> None:
    """A string match leaves every salt form of a training molecule in its background.

    Two spellings of one compound share a standardised identity, so excluding one
    must exclude the other. This is the join that matched 21 of lipophilicity's
    420 molecules when it was done on SMILES.
    """
    from drugrx.domain import background_excluding

    # "acid" and "sodium salt" are different strings, one identity.
    key_of = {"acid": "SAME", "sodium-salt": "SAME", "other": "DIFFERENT"}
    kept = background_excluding(["acid", "sodium-salt", "other"], ["acid"], key_of)
    assert kept == ["other"], "the salt form survived a standardised exclusion"


def test_a_molecule_with_no_known_identity_is_kept() -> None:
    """It cannot be shown to be in the reference set, and dropping it would shrink
    the background for the wrong reason."""
    from drugrx.domain import background_excluding

    key_of = {"known": "K1"}
    assert background_excluding(["known", "unstandardisable"], ["known"], key_of) == [
        "unstandardisable"
    ]


def test_excluding_the_reference_cleans_the_reported_background() -> None:
    """What the exclusion is still for, now that it no longer sets the bar.

    When the threshold came from a background percentile this was load-bearing: a
    background full of the training set's own molecules, each scoring 1.0 against
    itself, drove the bar up. The threshold now comes from the validation fold, so
    the exclusion no longer moves it — but the background is still *reported*, as
    the context that makes a similarity readable, and a contaminated one overstates
    how close ordinary chemistry is to this training set.
    """
    from drugrx.domain import background_excluding

    key_of = {s: s for s in ALCOHOLS + ELSEWHERE}
    contaminated = build_reference("x", ALCOHOLS, ELSEWHERE + ALCOHOLS, CALIBRATION)
    clean = build_reference(
        "x", ALCOHOLS, background_excluding(ELSEWHERE + ALCOHOLS, ALCOHOLS, key_of),
        CALIBRATION)

    assert contaminated.background_mean > clean.background_mean
    # And the threshold is now independent of it, which is the whole point of
    # moving the calibration onto the validation fold.
    assert contaminated.threshold == pytest.approx(clean.threshold)


def test_the_percentile_is_the_graded_signal() -> None:
    """The binary is a floor; this is what a caller should rank on.

    Moving the anchor from the background to the validation fold did not produce a
    clean separation, it moved the failure — the two populations genuinely overlap
    (measured shift only +0.019 to +0.180). So a query's position *within* the
    distribution of molecules the scores were measured on carries the information
    that a cut cannot.
    """
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)

    # A training molecule is at the top of the measured distribution.
    inside = membership(ref, ["CCCO"])[0]
    assert inside["measured_percentile"] == pytest.approx(100.0, abs=5.0)

    # Unrelated chemistry sits at the bottom of it.
    outside = membership(ref, ["Cn1cnc2c1c(=O)[nH]c(=O)n2C"])[0]
    assert outside["measured_percentile"] < inside["measured_percentile"]
    assert 0.0 <= outside["measured_percentile"] <= 100.0


def test_an_unparseable_row_has_no_percentile_rather_than_a_default() -> None:
    """Zero would read as "less similar than everything", which is a claim."""
    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    row = membership(ref, ["not-a-molecule"])[0]
    assert row["measured_percentile"] is None
    assert row["reason"] == "unparseable"


def test_an_older_reference_without_the_distribution_degrades_to_none() -> None:
    """A bundle built before the percentile existed must still load and serve."""
    import numpy as np

    ref = build_reference("alcohols", ALCOHOLS, ELSEWHERE, CALIBRATION)
    blob = ref.to_npz()
    old = DomainReference.from_npz("alcohols", blob["packed"], blob["meta"], None)
    assert len(old.calibration_sims) == 0
    row = membership(old, ["CCCO"])[0]
    assert row["measured_percentile"] is None
    # The binary still works, because the threshold is in the meta block.
    assert row["in_domain"] is True
