"""Known-answer fixtures for the audit layer.

**The verification code is the least-verified part of this stack, and its
failures look like results.** Two bugs in one session make the case:

* the statistics joined predictions on the SMILES string, so molecules written
  differently by two pipelines silently failed to match — it matched 21 of
  lipophilicity's 420 and reported a clean comparison;
* the similarity audit labelled its verdict with `abs(shift)`, so a *negative*
  shift (further away than background) printed as "real proximity".

Both were caught by eye, neither by a test, and **both failed toward
reassurance** — a clean "no leakage" and a confident "real proximity". Code that
checks the work needs its own checks more than the code that does the work,
because nobody audits the auditor.

Every test here is a known-answer case: a deliberately spiked input **must** come
back hot, a shuffled control **must** come back cold. If a change makes the audit
quietly permissive, these fail.
"""

from __future__ import annotations

import numpy as np

from drugrx.data.splits import (
    murcko_scaffold,
    nearest_neighbour_similarity,
    proximity_verdict as verdict,
)
from drugrx.data.standardise import audit_overlap, standardise
from drugrx.features import descriptor_span, featurise_one, standardise_descriptors

# A small, chemically varied set with no shared scaffolds between halves.
TEST_SET = [
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",     # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",       # ibuprofen
    "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)Cl",
    "CC(C)NCC(O)COc1cccc2ccccc12",      # propranolol
]
OTHER = [
    "OC(=O)CCc1ccccc1",
    "NC(=O)c1ccncc1",
    "CCOC(=O)c1ccccc1N",
    "CN(C)CCOC(c1ccccc1)c1ccccc1",
    "OCC1OC(O)C(O)C(O)C1O",
]


def keys(smiles: list[str]) -> set[str]:
    return {standardise(s).inchikey for s in smiles if standardise(s).ok}


# ── identity overlap ─────────────────────────────────────────────────

def test_spiked_pull_is_detected_as_hot():
    """A pull containing test molecules must report them. The join bug did not."""
    test_keys = keys(TEST_SET)
    spiked = keys(OTHER) | test_keys          # every test molecule present
    result = audit_overlap(spiked, test_keys, "spiked")
    assert result["n_overlap"] == len(test_keys)
    assert result["frac_test_leaked"] == 1.0


def test_clean_pull_is_detected_as_cold():
    result = audit_overlap(keys(OTHER), keys(TEST_SET), "clean")
    assert result["n_overlap"] == 0
    assert result["frac_test_leaked"] == 0.0


def test_spike_survives_rewriting_the_smiles():
    """The specific failure: same molecule, different spelling, must still match.

    A salt form and a non-canonical SMILES of aspirin are the same compound. A
    string join misses both and reports zero leakage.
    """
    disguised = ["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]", "O=C(O)c1ccccc1OC(C)=O"]
    result = audit_overlap(keys(disguised), keys(["CC(=O)Oc1ccccc1C(=O)O"]), "disguised")
    assert result["n_overlap"] == 1, "salt/alternate spelling must not evade the audit"


# ── similarity shift, with the sign preserved ────────────────────────

def shift(query: list[str], reference: list[str], background: list[str]) -> float:
    """Reproduces the audit's statistic: proximity to reference minus to background."""
    to_ref = nearest_neighbour_similarity(reference, query)
    to_bg = nearest_neighbour_similarity(background, query)
    return float(to_ref.mean() - to_bg.mean())


# `verdict` is imported from the library above, deliberately. An earlier draft
# re-implemented it here, which pinned a copy and would have passed happily while
# the shipped script kept its inverted label.


def test_identical_set_reads_as_proximity_not_distance():
    value = shift(TEST_SET, TEST_SET, OTHER)
    assert value > 0
    assert verdict(value) == "real proximity"


def test_negative_shift_is_not_labelled_proximity():
    """The inverted-label bug, pinned.

    A query closer to the background than to the reference has a negative shift
    and must never be reported as proximity — `abs(shift) < 0.02` did exactly
    that for every negative value outside the band.
    """
    assert verdict(-0.114) == "further than background"
    assert verdict(-0.05) == "further than background"
    assert verdict(0.0) == "baseline density"
    assert verdict(0.10) == "real proximity"


def test_shuffled_control_is_cold():
    """Unrelated chemistry must not read as proximate to the test set."""
    value = shift(OTHER, TEST_SET, OTHER)
    assert value <= 0.02, f"unrelated set reported as proximate (shift {value:+.3f})"


# ── scaffold handling ────────────────────────────────────────────────

def test_acyclic_molecules_give_empty_scaffold():
    """They must be segregated, not collapsed into a shared bucket.

    Treated as a scaffold, every acyclic molecule matches every other and
    manufactures a collision spike in any pull containing aliphatic drugs.
    """
    assert murcko_scaffold("CCCC(CCC)C(=O)O") == ""
    assert murcko_scaffold("CCO") == ""
    assert murcko_scaffold("CC(=O)Oc1ccccc1C(=O)O") == "c1ccccc1"


def test_scaffold_recovery_counts_only_real_scaffolds():
    acyclic_and_rings = ["CCO", "CCCC(CCC)C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"]
    scaffolds = {s for s in (murcko_scaffold(x) for x in acyclic_and_rings) if s}
    assert scaffolds == {"c1ccccc1"}, "empty scaffolds must not enter the count"


def test_nn_similarity_of_a_set_with_itself_is_one():
    sims = nearest_neighbour_similarity(TEST_SET, TEST_SET)
    assert np.allclose(sims, 1.0), "a molecule must be maximally similar to itself"


# ── feature-block standardisation ────────────────────────────────────

def test_descriptor_span_tracks_the_layout():
    """The third defect: "everything after the fingerprint" stopped being the
    descriptor block the moment Avalon and ErG were appended."""
    assert descriptor_span(2264) == slice(2048, 2264)
    assert descriptor_span(3091) == slice(2048, 2264)


def test_unrecognised_width_is_refused_not_guessed():
    import pytest

    with pytest.raises(ValueError):
        descriptor_span(2500)


def test_standardisation_leaves_every_fingerprint_block_alone():
    x = np.stack([
        featurise_one("CC(=O)Oc1ccccc1C(=O)O", extra_fps=True),
        featurise_one("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", extra_fps=True),
        featurise_one("CC(C)Cc1ccc(cc1)C(C)C(=O)O", extra_fps=True),
    ])
    out, (mean, std) = standardise_descriptors(x)
    assert len(mean) == 216, "stats must cover the descriptor block only"
    assert np.allclose(out[:, :2048], x[:, :2048]), "Morgan block must be untouched"
    assert np.allclose(out[:, 2264:], x[:, 2264:]), "Avalon/ErG blocks must be untouched"
    assert not np.allclose(out[:, 2048:2264], x[:, 2048:2264]), "descriptors must be scaled"
