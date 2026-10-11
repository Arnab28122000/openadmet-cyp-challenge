"""The shard runner must measure the same model and protocol as production.

`scripts/cyp_rowwise_shard.py` is self-contained so it can run on a VM without the repo, which
`feedback-a-control-must-run-the-production-command` only permits because every constant it
copies is asserted here against its production source. The first version copied the learner
faithfully and hardcoded N_MEMBERS = 9 against a production 8, so its own --selftest passed while
it measured a nine-member ensemble -- visible only as CYP1A2 base rho 0.6217 on the VM against
0.6181 locally. This test is the guard that would have caught it at the source.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_shard_learner_matches_production():
    import cyp_rowwise_shard as shard
    from cyp_decide_vs_affine import gbm as prod_gbm
    want = prod_gbm(7).get_params()
    got = shard.gbm(7).get_params()
    diff = {k: (want[k], got.get(k)) for k in want if got.get(k) != want[k]}
    assert not diff, f"shard learner differs from production: {diff}"


def test_shard_protocol_constants_match_production():
    import cyp_rowwise_shard as shard
    from cyp_borda_decode import N_MEMBERS
    from cyp_decide_vs_affine import PSEUDO_W

    from drugrx.cyp.screen import INACTIVE_R
    assert shard.N_MEMBERS == N_MEMBERS, (
        f"ensemble size differs: shard {shard.N_MEMBERS} vs production {N_MEMBERS}")
    assert shard.PSEUDO_W == PSEUDO_W, (
        f"pseudo-label weights differ: {shard.PSEUDO_W} vs {PSEUDO_W}")
    assert shard.INACTIVE_R == pytest.approx(INACTIVE_R)


def test_shard_floor_matches_the_release():
    """FLOOR is a measured quantity, not a convention -- it must equal what allr_floor returns
    on the real frame. Skipped if the release is not present, so the suite still runs cold."""
    import cyp_rowwise_shard as shard
    try:
        from drugrx.cyp.data import load
        from drugrx.cyp.screen import allr_floor
        frame = load(workers=2).frame
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"release not available: {exc}")
    assert shard.FLOOR == pytest.approx(float(allr_floor(frame)["y"]), abs=1e-5)


def test_shard_selftest_passes():
    import cyp_rowwise_shard as shard
    shard.selftest()
