"""`ADMET_TASKS` is a fixed-size contract, not a convenient list of endpoints.

On 2026-10-05 the Veith mirror-drift guard was found disabled for all four CYP
panels, because `cyp_external.py` fabricated its tasks with `n_rows=0`. The
obvious fix was to add the missing CYP1A2 panel to `ADMET_TASKS` and look tasks
up there. That fix would have been a worse bug than the one it repaired.

`ADMET_TASKS` sets:

* the multi-task model's output head count, via `MultiTaskAdmet(ADMET_TASKS, ...)`
  — so a 22nd entry invalidates the head layout of **every existing checkpoint**,
  including the published `arnab2812/drugrx-admet*` artifacts;
* the columns of the cached `admet_multitask.parquet` and `meta.json`;
* the shipped router's endpoint list (`ship_router.py`, `route_select.py`);
* and the denominator of every macro this programme has reported — every "18/21",
  "14/21", "8 of 21" in `PROGRESS.md` counts over this tuple.

None of those would have failed loudly. The model would simply have trained 22
heads, the macros would have been over 22 endpoints, and the numbers would have
stayed plausible while meaning something different from every number beside them
— the silent-renumbering defect this programme keeps finding, introduced by a fix
for another one.

So the size is pinned here. A deliberate change to the benchmark group must
update this test, which is the point: it forces the decision to be explicit and
the migration of the checkpoints and the reported macros to be considered.
"""
from __future__ import annotations

import pytest

from drugrx.data.tdc import ADMET_TASKS


def test_benchmark_group_has_exactly_21_endpoints() -> None:
    assert len(ADMET_TASKS) == 21, (
        f"ADMET_TASKS has {len(ADMET_TASKS)} entries, not 21. This changes the "
        "model's head count, invalidates every checkpoint, and silently moves "
        "the denominator of every '/21' macro in PROGRESS.md. If the change is "
        "intended, update this test and say in PROGRESS.md what happens to the "
        "existing checkpoints and reported numbers."
    )


def test_endpoint_keys_are_unique() -> None:
    keys = [t.key for t in ADMET_TASKS]
    assert len(set(keys)) == len(keys), "duplicate endpoint key in ADMET_TASKS"


def test_cyp1a2_is_not_in_the_benchmark_group() -> None:
    """It is not a TDC ADMET Benchmark Group endpoint, and it is load-bearing elsewhere.

    The CYP challenge's external acquisition needs all four Veith panels, and it
    gets them from `cyp_external.PANEL_ROWS` instead. If someone later moves
    CYP1A2 in here, the acquisition's pinned counts and this group's size both
    need revisiting rather than one of them silently winning.
    """
    assert not any(t.key.startswith("cyp1a2") for t in ADMET_TASKS)


def test_external_panels_carry_live_row_counts() -> None:
    """The acquisition must hand `load_task` a real expected count.

    `n_rows=0` made the mirror-drift check vacuous on all four panels while
    printing "expected 0", which reads as noise rather than as a disabled guard.
    """
    pytest.importorskip("pandas")
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "cyp_external.py"
    spec = importlib.util.spec_from_file_location("cyp_external_under_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert set(mod.PANEL_ROWS) == {"CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"}
    assert all(n > 0 for n in mod.PANEL_ROWS.values()), "a vacuous guard is not a guard"
    assert set(mod.PANELS) == set(mod.PANEL_ROWS)

    # The three panels that *are* in the benchmark group must agree with it, or
    # one of the two tables has drifted and the acquisition and the baseline are
    # no longer reading the same mirror.
    pinned = {t.repo: t.n_rows for t in ADMET_TASKS}
    for cyp, repo in mod.PANELS.items():
        if repo in pinned:
            assert pinned[repo] == mod.PANEL_ROWS[cyp], (
                f"{repo}: ADMET_TASKS says {pinned[repo]}, "
                f"PANEL_ROWS says {mod.PANEL_ROWS[cyp]}"
            )
