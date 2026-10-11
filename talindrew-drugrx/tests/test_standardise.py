"""Standardisation must merge the same compound and never merge different ones.

The L/D case is the one that matters most and is the reason this file exists:
RDKit's tautomer canonicaliser removes sp3 stereo at tautomeric centres by
default, which silently merged L- and D-alanine — and with them every enantiomer
pair whose stereocentre neighbours a tautomeric group. That corrupts labels and
manufactures leakage in one step, and it does it quietly.

    python -m pytest tests/ -q
"""

from __future__ import annotations

from drugrx.data.standardise import audit_overlap, standardise

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_same_molecule_different_spelling_merges():
    assert standardise(ASPIRIN).inchikey == standardise("O=C(O)c1ccccc1OC(C)=O").inchikey


def test_salt_merges_with_free_acid():
    salt = "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"
    assert standardise(ASPIRIN).inchikey == standardise(salt).inchikey


def test_zwitterion_merges_with_neutral_form():
    assert (
        standardise("C[C@@H](N)C(=O)O").inchikey
        == standardise("C[C@@H]([NH3+])C(=O)[O-]").inchikey
    )


def test_enantiomers_stay_distinct():
    """The regression this module exists for. Do not relax it."""
    left = standardise("C[C@@H](N)C(=O)O")
    right = standardise("C[C@H](N)C(=O)O")
    assert left.ok and right.ok
    assert left.inchikey != right.inchikey, "enantiomers must not share an InChIKey"
    # ...but they do share the skeleton block, which is why that block is only
    # ever an audit key, never a merge key.
    assert left.skeleton == right.skeleton


def test_failures_are_reported_not_raised():
    bad = standardise("not_a_molecule")
    assert not bad.ok
    assert bad.inchikey == ""
    assert bad.note


def test_empty_input():
    assert not standardise("").ok


def test_audit_overlap_counts():
    result = audit_overlap({"A", "B", "C"}, {"C", "D"}, "demo")
    assert result["n_overlap"] == 1
    assert result["frac_test_leaked"] == 0.5


def test_audit_overlap_empty_test_set_does_not_divide_by_zero():
    assert audit_overlap({"A"}, set())["frac_test_leaked"] == 0.0
