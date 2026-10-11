"""The router's contract, pinned on a bundle built in the test.

A synthetic bundle rather than the real one, so these run on a laptop in under a
second and fail for one reason at a time. What they pin is the part of Phase 4 that
is a *decision* rather than a computation, and therefore the part a later change
will quietly undo:

* an endpoint the routing table leaves **undecided** must go to gradient boosting,
  never to the GPU model by omission;
* gradient boosting gets **no interval**, because it has no uncertainty to report
  and manufacturing one is the "estimate that reads as a measurement" the FoldRx
  registry forbids;
* every value carries `in_domain` and the similarity that decided it;
* rows stay aligned to the input, including for molecules RDKit cannot parse;
* an endpoint routed to the ensemble with no ensemble in the bundle is an **error**,
  not a silent downgrade to the booster — the output would be identical and wrong.

The reproduction check that the real bundle must pass lives in `ship_router.py`
itself, where the booster is reloaded from the file it was just written to.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("rdkit")
lgb = pytest.importorskip("lightgbm")

from drugrx.domain import build_reference  # noqa: E402
from drugrx.serve import Router  # noqa: E402

ALCOHOLS = ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO", "CCCCCCCCO"]
#: Stands in for a validation fold — the population the threshold is calibrated
#: against. Same chemistry as the training set, held out.
CALIBRATION = ["CCCCCCCCO", "CCCCCCCCCO", "CCCCCCCCCCO", "CC(C)CCO", "CCC(C)CO",
               "CCCCC(C)O", "OCCCCCCCCCCO", "CC(O)CCCC", "CCCCCCCCCCCO",
               "CCCCCCCCCCCCO", "CC(C)(C)CCO", "CCCC(O)CCC", "CCCCCCO",
               "OCCCCCCCO", "CCCCCCCCCCCCCO", "CC(C)CO", "CCC(O)CC",
               "CCCCCCCCCCCCCCO", "OCCCCCCCCCO", "CCCCCCCCCCCCCCCO",
               "CCCCCCCCCCCCCCCCO", "CC(C)CCCO", "CCCCCCCCCCCCCCCCCO"]

ELSEWHERE = [
    "c1ccccc1", "c1ccncc1", "c1ccc2ccccc2c1", "C1CCNCC1",
    "CC(=O)Nc1ccc(O)cc1", "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "c1ccc(cc1)S(=O)(=O)N", "FC(F)(F)c1ccccc1", "CCOC(=O)c1ccccc1",
]


def _bundle(tmp_path: Path, routing: dict[str, str] | None = None,
            with_ensemble: bool = False) -> Path:
    """A two-endpoint bundle in `ship_router.py`'s layout."""
    from drugrx.features import featurise

    root = tmp_path / "router"
    (root / "gbm").mkdir(parents=True)
    (root / "domain").mkdir(parents=True)

    train = ALCOHOLS + ELSEWHERE
    x = featurise(train, counts=True, desc_mode="rdkit2d", extra_fps=False, workers=1)

    endpoints = {}
    for key, name, kind in (("caco2", "Caco2_Wang", "regression"),
                            ("herg", "hERG", "binary")):
        if kind == "regression":
            y = np.arange(len(train), dtype=float)
            model = lgb.LGBMRegressor(n_estimators=20, num_leaves=4, verbose=-1,
                                      min_child_samples=2, random_state=0)
        else:
            y = np.array([i % 2 for i in range(len(train))], dtype=float)
            model = lgb.LGBMClassifier(n_estimators=20, num_leaves=4, verbose=-1,
                                       min_child_samples=2, random_state=0)
        model.fit(x, y)
        model.booster_.save_model(str(root / "gbm" / f"{key}.txt"))

        reference = build_reference(key, ALCOHOLS, ELSEWHERE, CALIBRATION)
        np.savez_compressed(root / "domain" / f"{key}.npz", **reference.to_npz())
        endpoints[key] = {
            "arm": "count-rdkit2d", "name": name, "task": kind,
            "metric": "mae" if kind == "regression" else "roc_auc",
            "meaning": "a test fixture", "test_score": 0.0,
            "n_train": len(train), "n_test": 0,
            "domain_threshold": reference.threshold,
            "domain_quantile": reference.quantile,
            "domain_background_median": reference.background_median,
            "domain_reference_n": reference.n_reference,
        }

    (root / "gbm" / "meta.json").write_text(json.dumps({"endpoints": endpoints}))
    if routing is not None:
        (root / "routing.json").write_text(json.dumps(
            {"rows": [{"task": k, "serves": v} for k, v in routing.items()]}))
    if with_ensemble:
        (root / "ensemble.pt").write_bytes(b"not a real checkpoint")
    return root


def test_an_undecided_endpoint_goes_to_gradient_boosting(tmp_path: Path) -> None:
    """The decision ROUTING.md makes, and the one a later edit would undo.

    Twelve of twenty-one endpoints are statistically undecided. "Undecided" must
    resolve to the cheaper model whose failure modes are understood — not to the
    GPU model because nobody wrote a rule.
    """
    router = Router.load(_bundle(tmp_path, routing={
        "caco2": "gradient boosting", "herg": "gradient boosting"}))
    assert router.served_by("caco2") == "gbm"
    # Silent routing table: still gradient boosting.
    silent = Router.load(_bundle(tmp_path / "b", routing=None))
    assert silent.served_by("caco2") == "gbm"
    assert silent.served_by("herg") == "gbm"


def test_gradient_boosting_reports_no_interval(tmp_path: Path) -> None:
    router = Router.load(_bundle(tmp_path, routing={"caco2": "gradient boosting"}))
    rows = router.predict(["CCCO"], workers=1)
    meta = rows[0]["endpoint_meta"]["Caco2_Wang"]
    assert meta["served_by"] == "gbm"
    assert meta["interval"] is None, "a booster has no uncertainty to report"
    assert "interval_basis" not in meta


def test_every_value_carries_its_applicability_domain(tmp_path: Path) -> None:
    router = Router.load(_bundle(tmp_path, routing=None))
    rows = router.predict(["CCCO", "Cn1cnc2c1c(=O)[nH]c(=O)n2C"], workers=1)

    inside = rows[0]["endpoint_meta"]["Caco2_Wang"]
    outside = rows[1]["endpoint_meta"]["Caco2_Wang"]
    assert inside["in_domain"] is True
    assert inside["nn_similarity"] == pytest.approx(1.0)
    assert outside["in_domain"] is False
    assert outside["nn_similarity"] < inside["nn_similarity"]
    # The threshold is reported, so a reader can see the verdict was calibrated.
    assert outside["domain_threshold"] is not None


def test_an_out_of_domain_value_is_still_returned(tmp_path: Path) -> None:
    """The owner's decision: flag it, do not withhold it.

    Withholding would empty a screen of novel chemistry, which is arguably true
    and certainly unusable; a flagged number stays auditable.
    """
    router = Router.load(_bundle(tmp_path, routing=None))
    rows = router.predict(["Cn1cnc2c1c(=O)[nH]c(=O)n2C"], workers=1)
    assert rows[0]["endpoints"]["Caco2_Wang"] is not None
    assert rows[0]["endpoint_meta"]["Caco2_Wang"]["in_domain"] is False


def test_rows_stay_aligned_when_a_molecule_cannot_be_parsed(tmp_path: Path) -> None:
    router = Router.load(_bundle(tmp_path, routing=None))
    rows = router.predict(["CCCO", "not-a-molecule", "c1ccccc1"], workers=1)
    assert len(rows) == 3
    assert rows[0]["endpoint_meta"]["Caco2_Wang"]["nn_similarity"] == pytest.approx(1.0)
    assert rows[1]["endpoint_meta"]["Caco2_Wang"]["in_domain"] is False
    assert rows[2]["endpoint_meta"]["Caco2_Wang"]["nn_similarity"] is not None


def test_endpoints_use_the_names_admet_ai_uses(tmp_path: Path) -> None:
    """One vocabulary, so switching model does not change the dictionary."""
    router = Router.load(_bundle(tmp_path, routing=None))
    rows = router.predict(["CCCO"], workers=1)
    assert set(rows[0]["endpoints"]) == {"Caco2_Wang", "hERG"}
    assert "caco2" not in rows[0]["endpoints"]


def test_routing_to_a_missing_ensemble_is_an_error(tmp_path: Path) -> None:
    """A silent downgrade to the booster would be indistinguishable in the output."""
    with pytest.raises(ValueError, match="no ensemble.pt"):
        Router.load(_bundle(tmp_path, routing={"caco2": "ensemble"}))


def test_a_classifier_endpoint_comes_back_as_a_probability(tmp_path: Path) -> None:
    router = Router.load(_bundle(tmp_path, routing=None))
    rows = router.predict(["CCCO"], workers=1)
    value = rows[0]["endpoints"]["hERG"]
    assert 0.0 <= value <= 1.0
    assert rows[0]["endpoint_meta"]["hERG"]["task"] == "binary"


def test_an_empty_request_is_empty(tmp_path: Path) -> None:
    router = Router.load(_bundle(tmp_path, routing=None))
    assert router.predict([], workers=1) == []


def test_an_unknown_arm_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A booster fed the wrong feature layout returns numbers, not errors."""
    from drugrx.serve import _arm_options

    assert _arm_options("maplight")["extra_fps"] is True
    assert _arm_options("count-rdkit2d")["extra_fps"] is False
    with pytest.raises(ValueError, match="unknown arm"):
        _arm_options("binary-only")


def test_the_served_architecture_matches_the_trained_one() -> None:
    """`serve._plain_mlp` deliberately duplicates `train_fast.PlainMLP`, so pin it.

    The duplication is on purpose: the serving container ships the package and not
    the `scripts/` directory, and a serve path that imports from `scripts/` breaks
    the first time the image is slimmed. But a duplicated architecture is a
    duplicate that drifts, and the failure is not a shape error at load time —
    `torch.load` is happy, and `functional_call` fails somewhere inside a `vmap`
    with a message about a parameter nobody has heard of.

    Both halves of the contract are checked: the parameter **names**, which is what
    `functional_call` matches on, and the **shapes**, which is what the stacked
    ensemble's weights have to fit.
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train_fast import PlainMLP  # the trainer's definition

    from drugrx.serve import EnsembleBundle, _plain_mlp

    arch = {"d_model": 512, "n_layers": 3, "dropout": 0.1}
    bundle = EnsembleBundle(
        params={}, buffers={}, in_dim=2264, n_tasks=27, tdc_tasks=[], erx_heads=[],
        feature_stats=(np.zeros(1, np.float32), np.ones(1, np.float32)),
        label_norm={}, architecture=arch, extra_fps=False,
    )

    served = _plain_mlp(bundle).state_dict()
    trained = PlainMLP(2264, 27, arch["d_model"], arch["n_layers"], arch["dropout"]).state_dict()

    assert sorted(served) == sorted(trained), (
        "parameter names differ between the serving and training architectures; "
        f"serve-only={sorted(set(served) - set(trained))} "
        f"train-only={sorted(set(trained) - set(served))}"
    )
    assert {k: tuple(v.shape) for k, v in served.items()} == {
        k: tuple(v.shape) for k, v in trained.items()
    }, "parameter shapes differ between the serving and training architectures"


def test_expansionrx_keys_are_stable_slugs_that_cannot_be_a_tdc_name() -> None:
    """The namespace, pinned where it is computed.

    ExpansionRx's own columns carry spaces and a `>`; those make poor response
    keys. More to the point, two of them name properties TDC also measures —
    `MLM CLint` against `Clearance_Microsome_AZ`, `Caco-2 Permeability Papp A>B`
    against `Caco2_Wang`. Different laboratories, different protocols, panels
    that share **zero** molecules, so no offset between them is estimable. If a
    slug could ever collide with a TDC name, a value on one scale would be served
    under the other's label — the worst failure available here, because it is a
    wrong number wearing a right name.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from ship_router import TDC_NAMES

    from drugrx.serve import ERX_SLUGS, erx_key

    keys = {erx_key(head) for head in ERX_SLUGS}
    assert len(keys) == len(ERX_SLUGS), "two heads produced the same key"
    assert keys.isdisjoint(set(TDC_NAMES.values())), "an ExpansionRx key collides with a TDC name"
    assert all(k.startswith("ExpansionRx_") for k in keys)
    # Stable and selector-safe: no spaces, no punctuation beyond the underscore.
    assert all(c.isalnum() or c == "_" for k in keys for c in k)

    # A column we have never seen must still produce a usable key rather than
    # crashing or silently dropping the head.
    assert erx_key("Caco-2 Permeability Papp A>B") == "ExpansionRx_Caco2_Papp_AB"
    assert erx_key("Brand New Assay (x)").startswith("ExpansionRx_Brand_New_Assay")
