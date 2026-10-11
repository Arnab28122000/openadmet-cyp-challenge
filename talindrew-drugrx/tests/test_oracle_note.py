"""Pins the numbers published in the oracle-ceiling note to the stored runs.

`docs/drugrx-training/notes/oracle-ceilings-measure-decorrelation.md` makes
quantitative claims that someone else is invited to act on. Seven defects have now
lived in this programme's interpretation layer, so a published figure that exists
only in a write-up is a figure with no provenance. Every number asserted here is
re-derived from the stored per-run prediction files, not copied from a log.

Opt-in because it reads the challenge labels from the Hub and needs the 36 cached
prediction files:

    TD_ORACLE_NOTE=1 .venv/bin/pytest tests/test_oracle_note.py

If the artifacts move or a metric changes, this fails and the note is wrong — which
is the point.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("TD_ORACLE_NOTE"),
    reason="set TD_ORACLE_NOTE=1 (reads the Hub and the cached prediction files)",
)

CACHE = Path(__file__).resolve().parent.parent / "data" / "cyp_cache"


@pytest.fixture(scope="module")
def pairs():
    from drugrx.cyp.data import PRIMARY, load
    from drugrx.cyp.metric import soft_threshold_error, st_rae

    models = {}
    for sub in ("cyp_loss", "cyp_aux"):
        d = CACHE / sub
        if not d.exists():
            pytest.skip(f"{d} absent; run the experiments or restore the cache")
        for f in sorted(d.glob("*.json")):
            npz = f.with_suffix(".npz")
            if npz.exists():
                tag = json.loads(f.read_text()).get("tag", f.stem)
                models[f.stem] = (npz, tag.replace("aux-", "").replace("loss-", ""))

    fr = load().frame
    y = {c: fr[c].to_numpy(float) for c in PRIMARY}
    lo = {c: fr[f"{c}_conf_low"].to_numpy(float) for c in PRIMARY}
    hi = {c: fr[f"{c}_conf_high"].to_numpy(float) for c in PRIMARY}

    cache, idx = {}, None
    for name, (npz, arm) in models.items():
        z = np.load(npz)
        if idx is None:
            idx = z["va_index"]
        assert np.array_equal(z["va_index"], idx), f"{name}: different split"
        p = z["val"]
        errs = {c: soft_threshold_error(p[:, i], lo[c][idx], hi[c][idx])
                for i, c in enumerate(PRIMARY)}
        macro = float(np.nanmean([st_rae(p[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                 for i, c in enumerate(PRIMARY)]))
        cache[name] = {"p": p, "errs": errs, "macro": macro, "arm": arm}

    rows = []
    for n1, n2 in itertools.combinations(sorted(cache), 2):
        a, b = cache[n1], cache[n2]
        rhos, out = [], np.empty_like(a["p"])
        for i, c in enumerate(PRIMARY):
            ea, eb = a["errs"][c], b["errs"][c]
            k = np.isfinite(ea) & np.isfinite(eb)
            if ea[k].std() > 0 and eb[k].std() > 0:
                rhos.append(float(np.corrcoef(ea[k], eb[k])[0, 1]))
            out[:, i] = np.where(eb < ea, b["p"][:, i], a["p"][:, i])
        orac = float(np.nanmean([st_rae(out[:, i], y[c][idx], lo[c][idx], hi[c][idx])
                                for i, c in enumerate(PRIMARY)]))
        rows.append((float(np.mean(rhos)),
                     orac - min(a["macro"], b["macro"]),
                     a["arm"] == b["arm"]))
    return len(cache), np.array([r[0] for r in rows]), \
        np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def test_thirty_six_models_and_630_pairs(pairs):
    n_models, rho, gap, _ = pairs
    assert n_models == 36, f"note says 36 models, found {n_models}"
    assert len(rho) == 630 == n_models * (n_models - 1) // 2


def test_every_gap_is_negative(pairs):
    """A positive oracle gap is arithmetic: the minimum cannot beat its own members."""
    _, _, gap, _ = pairs
    assert (gap <= 0).all(), "an oracle scored worse than its better member"


def test_theoretical_form_beats_linear_and_has_zero_intercept(pairs):
    _, rho, gap, _ = pairs
    ss_tot = ((gap - gap.mean()) ** 2).sum()

    def fit(x):
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, gap, rcond=None)
        r = gap - A @ coef
        return coef, 1 - (r ** 2).sum() / ss_tot, r

    (ci, si), r2_lin, _ = fit(rho)
    (cs, ss), r2_sqrt, _ = fit(np.sqrt(1 - rho))

    assert r2_sqrt > r2_lin, "note claims the sqrt(1-rho) form fits better"
    assert r2_sqrt == pytest.approx(0.9431, abs=0.002), r2_sqrt
    assert r2_lin == pytest.approx(0.9316, abs=0.002), r2_lin
    assert ss == pytest.approx(-0.4456, abs=0.002), ss
    # The intercept must be ~0: at rho=1 the errors are identical and the gap is 0.
    assert abs(cs) < 0.02, f"intercept {cs:+.4f} is not ~0; the derivation requires it"


def test_residual_is_signed_against_complementarity(pairs):
    """The only part that could be evidence, and it points the other way."""
    _, rho, gap, same = pairs
    A = np.vstack([np.ones_like(rho), np.sqrt(1 - rho)]).T
    coef, *_ = np.linalg.lstsq(A, gap, rcond=None)
    resid = gap - A @ coef
    rs, rc = resid[same], resid[~same]
    assert len(rs) == 114 and len(rc) == 516
    diff = rs.mean() - rc.mean()
    se = (rs.var(ddof=1) / len(rs) + rc.var(ddof=1) / len(rc)) ** 0.5
    assert diff == pytest.approx(-0.00395, abs=0.0005), diff
    # The NAIVE t. Pinned only to show it is the optimistic one: each of the 36 runs
    # appears in 35 pairs, so these 630 pairs are not 630 observations. The note
    # quotes the cluster-bootstrap se of 0.00148 (t = -2.66) instead; see
    # scripts/cyp_oracle_bootstrap.py.
    assert diff / se == pytest.approx(-4.89, abs=0.4), diff / se
    assert se < 0.00148, "naive se should be smaller than the clustered one"
    # A MORE NEGATIVE residual means the pair beat the fitted curve, i.e. a larger
    # oracle improvement in magnitude. Same-config pairs sitting below cross-config
    # ones is therefore the OPPOSITE of complementarity: if configuration conferred
    # region-specific competence, cross-config pairs would beat the curve by more.
    assert rs.mean() < 0 < rc.mean(), (rs.mean(), rc.mean())
    assert diff < 0, "the note's central claim is that the residual is signed " \
                     "against complementarity; this run says otherwise"
    assert abs(diff) / abs(gap.mean()) < 0.05, "note says the residual is ~3% of the effect"


def test_intercept_is_small_but_not_claimed_to_be_zero(pairs):
    """The theory requires zero at rho=1; we measure +0.0095, CI excluding zero.

    The note says "small where the derivation requires zero" rather than "~0",
    because a run-level bootstrap puts the 95% CI at [+0.0042, +0.0149]. Mild
    misspecification is expected: the Gaussian derivation assumes symmetric unbounded
    errors and this metric's are non-negative with a point mass at zero.
    """
    _, rho, gap, _ = pairs
    A = np.vstack([np.ones_like(rho), np.sqrt(1 - rho)]).T
    coef, *_ = np.linalg.lstsq(A, gap, rcond=None)
    assert coef[0] == pytest.approx(0.0095, abs=0.002), coef[0]
    # Small relative to what is being fitted, which is the claim actually made.
    assert abs(coef[0]) / abs(gap).mean() < 0.15
