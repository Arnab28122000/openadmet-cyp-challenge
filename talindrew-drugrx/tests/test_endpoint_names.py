"""Every endpoint we serve must have a row in the panel that renders it.

The router emits ADMET-AI's TDC-canonical key names (`Caco2_Wang`, not `caco2`)
so that one vocabulary serves both ADMET models and the existing `AdmetPanel`
renders ours with no frontend change. That only works while the two lists agree,
and **they disagree silently**: the panel filters rows with
`endpoints[row.key] != null`, so an endpoint whose name it does not know simply
does not appear. No error, no empty row — the number is computed, returned, and
dropped on the floor.

Checked statically across the two deployables, like
`talindrew-backend/tests/foldrx/test_gateway_routes.py` and for the same reason:
the frontend cannot be imported here, and the keys are plain string literals in a
const array, so parsing them is enough. The test skips with a reason if the
sibling checkout is absent, because a guard that quietly does nothing is worse
than no guard.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

PANEL = (ROOT.parent / "Talindrew-frontend" / "src" / "features" / "foldrx"
         / "components" / "stagePanels.tsx")


def panel_keys() -> set[str]:
    """The `key:` strings inside `ADMET_GROUPS`, and nothing else in the file."""
    source = PANEL.read_text()
    match = re.search(r"const ADMET_GROUPS[^=]*=\s*\[(.*?)\n\];", source, re.S)
    if match is None:
        pytest.fail(f"could not find ADMET_GROUPS in {PANEL}")
    return set(re.findall(r'key:\s*"([^"]+)"', match.group(1)))


def tdc_names() -> dict[str, str]:
    from ship_router import TDC_NAMES

    return TDC_NAMES


@pytest.mark.skipif(not PANEL.exists(), reason=f"{PANEL} not checked out here")
def test_every_endpoint_we_serve_has_a_row_in_the_panel() -> None:
    missing = {key: name for key, name in tdc_names().items()
               if name not in panel_keys()}
    assert not missing, (
        "these endpoints would be computed, returned and never shown — add a row "
        f"to ADMET_GROUPS in {PANEL.name}: {missing}"
    )


def test_the_display_names_are_unique() -> None:
    """Two endpoints sharing a display name would overwrite each other's value.

    The output is a dict keyed by display name, so a collision is a lost
    endpoint rather than a duplicated row.
    """
    names = list(tdc_names().values())
    assert len(names) == len(set(names)), "duplicate display names in TDC_NAMES"


def test_every_admet_task_has_a_display_name() -> None:
    """A missing entry would raise a KeyError inside `ship_router` mid-sweep."""
    from drugrx.data.tdc import ADMET_TASKS

    missing = [t.key for t in ADMET_TASKS if t.key not in tdc_names()]
    assert not missing, f"no display name for {missing}"


def test_microsomal_clearance_is_not_the_hepatocyte_row() -> None:
    """The one collision worth naming: two different liver-clearance assays.

    TDC has both `clearance_microsome_az` and `clearance_hepatocyte_az`, we train
    on the microsome one, and the panel used to carry only the hepatocyte row.
    Mapping ours onto that row would have shown a microsome prediction under a
    hepatocyte label — a wrong number wearing the right name, which is the worst
    of the failure modes available here.
    """
    assert tdc_names()["cl_micro"] == "Clearance_Microsome_AZ"
    assert "Clearance_Microsome_AZ" in panel_keys()
    assert "Clearance_Hepatocyte_AZ" in panel_keys(), (
        "the hepatocyte row must stay: ADMET-AI predicts it and we do not"
    )


def test_the_expansionrx_prefix_cannot_collide_with_a_tdc_name() -> None:
    """The namespace separation, pinned where it is defined rather than assumed.

    `ExpansionRx_CLint` and `Clearance_Microsome_AZ` are different labs'
    microsomal clearance whose panels share zero molecules, so no offset between
    them is estimable. If a prefixed head could ever equal a TDC name, a value on
    one scale would be served under the other's label.
    """
    from ship_router import ERX_PREFIX

    assert ERX_PREFIX == "ExpansionRx_"
    assert not any(name.startswith(ERX_PREFIX) for name in tdc_names().values())
