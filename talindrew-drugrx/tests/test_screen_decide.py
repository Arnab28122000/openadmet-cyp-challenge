"""Pin the round-3 modules: screen-derived labels, the decision rule, placement.

Each test here exists because of something that went wrong or could have:

* `balance_point` is an exact minimiser, so it is checked against a brute-force grid rather
  than against itself. The tie-break at a flat segment is tested *separately*, because the
  risk value cannot distinguish the two ends — which is exactly why the first version
  returned the wrong one and no risk-based test caught it.
* `screen_labels` must never invent a label where there is no readout, and must never
  overwrite a real dose-response label with a pseudo one. Both are silent failures.
* `place` is monotone by construction; a regression that broke that would be invisible in the
  score but would destroy the one end-to-end check this programme has (ρ and τ unchanged).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from drugrx.cyp.decide import balance_point, decide, mixture_draws
from drugrx.cyp.placement import BLIND_MOMENTS, place
from drugrx.cyp.screen import (CYPS, FLOOR_SPLIT_R, INACTIVE_R, R_EDGES, allr_floor, calibrate,
                               screen_labels, summary)


def risk(p, lo, hi, w):
    return float((w * (np.clip(lo - p, 0, None) + np.clip(p - hi, 0, None))).sum())


@pytest.mark.parametrize("seed", range(25))
def test_balance_point_is_the_exact_minimiser(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 30))
    y = rng.normal(4.5, 1.2, n)
    lo = y - rng.uniform(0.05, 1.5, n)
    hi = y + rng.uniform(0.05, 1.5, n)
    w = rng.uniform(0.1, 2.0, n)
    p = balance_point(lo, hi, w)
    grid = np.linspace(lo.min() - 1, hi.max() + 1, 3001)
    assert risk(p, lo, hi, w) <= min(risk(q, lo, hi, w) for q in grid) + 1e-9


def test_flat_segment_resolves_to_the_right_hand_end():
    """Every draw inactive: the risk is flat across the free window, and we want its top.

    Both ends score zero against these draws, so this cannot be asserted through the risk.
    It is asserted directly, because the realistic error is an "inactive" compound being
    somewhat active, and the top of the window is 2.4 log units closer to that.
    """
    lo = np.full(8, 1.07)
    hi = np.full(8, 3.52)
    assert balance_point(lo, hi) == pytest.approx(3.52)


def test_mixture_prefers_the_window_top_over_the_weighted_mean():
    bounds = lambda q: (np.where(q < 3.6, 1.07, q - 0.25),      # noqa: E731
                        np.where(q < 3.6, 3.52, q + 0.25))
    draws, w = mixture_draws(np.array([5.4, 5.5, 5.6]), 0.8, 2.21)
    weighted_mean = float((draws * w).sum() / w.sum())
    chosen = decide(draws, bounds, w)
    assert chosen == pytest.approx(3.52)
    assert chosen > weighted_mean + 0.5          # the mean is the wrong operation here


def test_balance_point_handles_degenerate_input():
    assert np.isnan(balance_point(np.array([]), np.array([])))
    assert np.isnan(balance_point(np.array([np.nan]), np.array([np.nan])))


# --------------------------------------------------------------------------- screen labels
def _frame(n=400, seed=0):
    """A miniature release: readouts for everything, curves for a selected subset."""
    rng = np.random.default_rng(seed)
    out = {"smiles": [f"C{i}" for i in range(n)], "key": [f"K{i}" for i in range(n)]}
    for c in CYPS:
        lf = rng.normal(-0.8, 0.9, n)
        r = 2.0 ** lf
        # Potency falls smoothly with the fraction of signal left, as it does on the release
        # (median DRC pIC50 5.4-6.0 at r <= 0.25 down to 2.0-2.9 at r >= 0.9).
        y = 6.0 - 4.0 * np.clip(r, 0, 1.2) + rng.normal(0, 0.15, n)
        # Promotion: strong readouts get a curve, which is the defect being modelled, plus a
        # minority of weak ones -- true of three isoforms on the release, and the reason the
        # floor window can be calibrated at all.
        promoted = (r < 0.6) | (rng.random(n) < 0.25)
        col = f"{c}_pIC50_direct_inhibition"
        out[f"log2fc_{c}"] = lf
        out[col] = np.where(promoted, y, np.nan)
        out[f"{col}_conf_low"] = np.where(promoted, y - 0.3, np.nan)
        out[f"{col}_conf_high"] = np.where(promoted, y + 0.3, np.nan)
    return pd.DataFrame(out)


def test_screen_labels_never_overwrite_a_measurement():
    f = _frame()
    labs = screen_labels(f)
    for c in CYPS:
        col = f"{c}_pIC50_direct_inhibition"
        has = f[col].notna().to_numpy()
        kind = labs[col]["kind"].to_numpy()
        assert (kind[has] == "drc").all(), "a measured compound was given a pseudo-label"
        # And a pseudo-label is only ever produced where a readout exists.
        assert labs[col].loc[~f[f"log2fc_{c}"].notna() & ~has, "kind"].eq("none").all()


def test_every_no_inhibition_compound_gets_the_floor_window():
    f = _frame()
    labs = screen_labels(f)
    floor = allr_floor(f)
    for c in CYPS:
        col = f"{c}_pIC50_direct_inhibition"
        r = 2.0 ** f[f"log2fc_{c}"]
        want = f[col].isna() & (r >= INACTIVE_R)
        got = labs[col]["kind"] == "inactive"
        assert (want == got).all()
        sel = labs[col][got]
        assert np.allclose(sel["lo"], floor["lo"]) and np.allclose(sel["hi"], floor["hi"])
        assert floor["lo"] < floor["hi"]


def test_pseudo_intervals_are_ordered_and_finite():
    f = _frame()
    labs = screen_labels(f)
    for c in CYPS:
        d = labs[f"{c}_pIC50_direct_inhibition"]
        made = d["kind"].isin(["inactive", "weak", "active"])
        assert np.isfinite(d.loc[made, ["lo", "hi", "y"]].to_numpy()).all()
        assert (d.loc[made, "lo"] <= d.loc[made, "hi"]).all()


def test_calibration_is_monotone_in_the_readout():
    """Weaker inhibition at 50 uM must not map to a more potent label."""
    f = _frame(n=1200)
    tab = calibrate(f)["pooled"].sort_index()
    meds = tab["y_med"].dropna().to_numpy()
    assert (np.diff(meds) <= 1e-9).all(), f"non-monotone bin medians: {meds}"
    assert summary(screen_labels(f)).loc["drc"].gt(0).all()


def test_bin_edges_cover_the_readout_range():
    assert R_EDGES[0] == 0.0 and R_EDGES[-1] == np.inf
    assert INACTIVE_R in R_EDGES


# --------------------------------------------------------------------------- placement
def test_place_is_monotone_and_hits_its_target_moments():
    rng = np.random.default_rng(3)
    p = rng.normal(4.7, 0.6, 750)
    mean_b, sd_b = BLIND_MOMENTS["CYP2D6"]
    out = place(p, r=0.5, mean_blind=mean_b, sd_blind=sd_b, discount=0.9, floor=-99)
    assert spearmanr(p, out).statistic == pytest.approx(1.0)
    assert out.mean() == pytest.approx(mean_b, abs=1e-6)
    assert out.std() == pytest.approx(0.9 * 0.5 * sd_b, rel=1e-6)


def test_place_floors_and_survives_a_constant_vector():
    out = place(np.full(10, 4.0), r=0.5, mean_blind=3.1, sd_blind=1.6)
    assert np.allclose(out, 3.1)                      # sd 0 -> the centre, not a divide by zero
    low = place(np.linspace(-5, 0, 50), r=0.5, mean_blind=1.2, sd_blind=1.0)
    assert (low >= 1.0 - 1e-9).all()


def test_floor_is_issued_in_two_branches_split_at_the_existing_edge():
    """The floor label is a two-branch rule keyed on the compound's own readout.

    Selected out-of-fold over a fitted slope and over the pooled constant: split-at-1.0 medians
    is the only rule that wins BOTH OOF MSE (-4.40%) and OOF MAE (-1.44%), which matters because
    the submission is 0.30 x squared error + 0.70 x an L1-like interval hinge. A constant mean
    wins L2 by 12.87% and LOSES L1 by 8.51%, so it is not defensible under both.

    `y` must stay the POOLED median: tests/test_shard_parity.py pins a shard constant against it
    and every recorded shift is measured from it.
    """
    f = _frame()
    floor = allr_floor(f)
    assert floor["split_at"] == FLOOR_SPLIT_R == 1.0
    assert FLOOR_SPLIT_R in R_EDGES, "the split must reuse an existing bin edge, not add one"
    assert {"y", "lo", "hi", "n", "y_lo", "y_hi", "n_lo", "n_hi"} <= set(floor)
    assert floor["n_lo"] + floor["n_hi"] == floor["n"]
    labs = screen_labels(f)
    for c in CYPS:
        col = f"{c}_pIC50_direct_inhibition"
        r = (2.0 ** f[f"log2fc_{c}"]).to_numpy(float)
        d = labs[col]
        inact = (d["kind"] == "inactive").to_numpy()
        if not inact.any():
            continue
        want = np.where(r[inact] < FLOOR_SPLIT_R, floor["y_lo"], floor["y_hi"])
        assert np.allclose(d.loc[inact, "y"].to_numpy(float), want), (
            f"{c}: the branch must follow the compound's own readout")


def test_a_thin_floor_branch_degrades_to_the_pooled_median():
    """If a branch has too few compounds to estimate, fall back to the pooled constant rather
    than to a noisy half-sample -- so the behaviour degrades to the previous release."""
    f = _frame()
    g = f.copy()
    # push every floor compound above the split so the low branch is empty
    for c in CYPS:
        m = g[f"{c}_pIC50_direct_inhibition"].notna() & (2.0 ** g[f"log2fc_{c}"] >= INACTIVE_R)
        g.loc[m, f"log2fc_{c}"] = 2.0
    fl = allr_floor(g)
    assert fl["y_lo"] == fl["y_hi"] == pytest.approx(fl["y"]), (
        "an unestimable branch must fall back to the pooled median")
