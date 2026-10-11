"""Validation gates for the LeJEPA/LeWAM instruments. All must pass before any real data.

These are not smoke tests. `findings/07` was wasted because an arm was run before anyone checked
that it varied the quantity its claim was about, and the instrument that would have caught it is
the one under test here. So the gates are the ones whose answers are known independently of the
implementation:

1. the two implementations are derived from one integral by different routes and must agree;
2. a pure point mass must hit the closed-form ceiling 1 - sqrt(2) + 1/sqrt(3), which both routes
   predict analytically;
3. the statistic must be MONOTONE in how much point mass is present, because that monotonicity
   is the entire basis for using it as a concentration measure;
4. the null must be calibrated, or "far from normal" has no units.
"""

from __future__ import annotations

import numpy as np
import pytest

from drugrx.distest import (  # noqa: I001
    POINT_MASS_DISTANCE,
    effective_rank,
    epps_pulley,
    epps_pulley_knots,
    epps_pulley_lejepa,
    ep_null,
    rankme,
    sigreg,
)


def test_point_mass_distance_constant_is_right():
    """1 - sqrt(2) + 1/sqrt(3), derived twice: as T/n of the closed form and as the quadrature
    of (1 - e^{-t^2/2})^2 against the standard normal weight."""
    assert POINT_MASS_DISTANCE == pytest.approx(0.1631367, abs=1e-6)


#: Measured absolute accuracy floor of the 17-knot form on `distance`, from the ladder below.
KNOT17_FLOOR = 2e-5


@pytest.mark.parametrize("n", [64, 512, 2048])
def test_exact_and_knots_agree_to_an_ABSOLUTE_floor_not_a_relative_one(n):
    """The quadrature error is absolute, ~1e-7..8e-6, and essentially independent of n. But
    `distance` under the null shrinks as 1/n (4.5e-4 at n=64, 1.2e-5 at n=2048), so the
    RELATIVE error grows with n: measured up to 1.3e-2. Neither tolerance alone is correct and
    picking the wrong one is how this gate would have been made meaningless."""
    rng = np.random.default_rng(7)
    for _ in range(5):
        x = rng.standard_normal(n)
        a = epps_pulley(x)["distance"]
        b = epps_pulley_knots(x, knots=17)["distance"]
        assert abs(a - b) < KNOT17_FLOOR, f"n={n}: exact {a:.3e} vs 17-knot {b:.3e}"


def test_the_knot_floor_is_comparable_to_the_null_signal_at_our_n():
    """Why that floor matters, stated as a measurement rather than a worry. At n = 6145 -- our
    actual compound count -- a truly-Gaussian sample's `distance` is of the same order as the
    17-knot error itself. So the 17-knot form can separate "collapsed" from "fine", which is all
    a REGULARISER needs, but it cannot resolve degrees of near-Gaussianity. Any number quoted in
    a finding uses the exact form."""
    rng = np.random.default_rng(71)
    d = np.mean([epps_pulley(rng.standard_normal(6145))["distance"] for _ in range(8)])
    assert d < 30 * KNOT17_FLOOR, (
        f"null distance {d:.2e} at n=6145 vs knot floor {KNOT17_FLOOR:.0e} -- the regimes are "
        "comparable, which is the point")


def test_knot_quadrature_degrades_on_heavy_tails_and_the_exact_form_is_the_reference():
    """The approximation's LIMIT, asserted so it cannot regress unnoticed.

    The integrand carries cos(t*y_j), so its oscillation frequency is set by max|y|. On a sample
    standardised to max|y| ~ 21, the measured relative error of the 17-knot form is ~2e-3 and is
    NOT monotone in the node count (33 knots came out worse than 17; only 257 reached machine
    precision). That is ordinary Gauss-Hermite behaviour on an oscillatory integrand, but it
    means LeJEPA's own 17-knot config is a training signal, not a measurement instrument."""
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.standard_exponential(400) ** 2,
                        np.full(300, -1.0),
                        rng.standard_cauchy(50)])
    y = (x - x.mean()) / x.std(ddof=1)
    assert abs(y).max() > 10, "sample is not heavy-tailed enough to be the check it claims"
    a = epps_pulley(x)["distance"]
    assert a > 0.02, "sample is not non-normal enough to be a real check"
    err17 = abs(epps_pulley_knots(x, knots=17)["distance"] - a) / a
    err257 = abs(epps_pulley_knots(x, knots=257)["distance"] - a) / a
    assert err17 > 1e-4, f"17-knot error {err17:.2e} -- if this is now tiny, re-read the docstring"
    assert err17 < 1e-2, f"17-knot error {err17:.2e} is worse than documented"
    assert err257 < 1e-9, f"many knots should converge to the exact form, got {err257:.2e}"


@pytest.mark.parametrize("value", [4.2, 2.211, 0.0, -17.5, 1e6])
def test_point_mass_hits_the_analytic_ceiling_both_ways(value):
    """REGRESSION. A literal point mass does NOT have sd 0 in floating point --
    `np.full(1000, 4.2).std(ddof=1)` is 1.78e-15 -- so a `sd <= 0` guard never fires, the
    standardisation divides by 1.78e-15, and amplified ULP noise is reported as distance 0.476
    instead of the ceiling 0.163. A 3x error in the one case this instrument exists for.
    2.211 is in the list because it is the actual screen-inactive pseudo-label."""
    x = np.full(1000, value)
    # The exact form is exact here: y is all zeros and T/n collapses to 1 - sqrt(2) + 1/sqrt(3).
    r = epps_pulley(x)
    assert r["degenerate"] is True, f"exact form missed a point mass at {value}"
    assert r["distance"] == pytest.approx(POINT_MASS_DISTANCE, abs=1e-12)
    # The 17-knot form carries its quadrature error (6.6e-6 here); 33 knots converges.
    k = epps_pulley_knots(x)
    assert k["degenerate"] is True, f"knot form missed a point mass at {value}"
    assert k["distance"] == pytest.approx(POINT_MASS_DISTANCE, abs=KNOT17_FLOOR)
    assert epps_pulley_knots(x, knots=33)["distance"] == pytest.approx(
        POINT_MASS_DISTANCE, abs=1e-9)


def test_a_genuinely_tiny_but_real_spread_is_not_called_degenerate():
    """The other side of that guard: the imputations in findings/07 had sd 0.030-0.084 around a
    mean of 2.10, which is small but entirely real. Calling that a point mass would destroy the
    very comparison Phase 2 is built on."""
    rng = np.random.default_rng(41)
    x = 2.10 + 0.030 * rng.standard_normal(2648)
    r = epps_pulley(x)
    assert r["degenerate"] is False
    assert r["distance"] < 0.01, "a narrow Gaussian is still Gaussian in SHAPE"


def test_distance_is_monotone_in_the_point_mass_fraction():
    """GATE 3, the load-bearing one. If this is not monotone the statistic cannot be used to
    order our four isoforms by their mass share, and Phase 2 does not start."""
    rng = np.random.default_rng(3)
    n = 20000
    fracs = [0.0, 0.02, 0.05, 0.10, 0.15, 0.29, 0.40, 0.61, 0.80]
    d = []
    for f in fracs:
        k = int(round(f * n))
        x = np.concatenate([np.full(k, 2.211), rng.standard_normal(n - k) * 0.9 + 5.0])
        d.append(epps_pulley_knots(x)["distance"])
    assert all(b > a for a, b in zip(d, d[1:])), f"not monotone: {list(zip(fracs, d))}"
    assert d[0] < 0.001, f"pure normal should be ~0, got {d[0]}"
    assert d[-1] > 0.05, f"80% point mass should be large, got {d[-1]}"


def test_null_level_matches_the_CLOSED_FORM_not_itself():
    """REWRITTEN. The previous version compared `ep_null` to a loop computing the same statistic,
    so any monotone corruption of the statistic cancelled exactly -- replacing it with
    `7.3*distance + 0.9` still passed. A calibration gate that compares a function to itself
    tests nothing.

    Anchored to an EXTERNAL value instead. Under standardisation by the SAMPLE mean and sd, the
    BHEP limit process has covariance e^{-(s-t)^2/2} - e^{-(s^2+t^2)/2}(1 + st + s^2 t^2/2), so

        E[n * distance] -> 1 - sqrt(3)/2 = 0.13397460

    and under TRUE-moment standardisation it is 1 - 1/sqrt(3) = 0.42264973. The ratio of the two
    is the exact 3.154701 shrink that standardising costs."""
    import numpy as _np
    sample_form = 1.0 - _np.sqrt(3.0) / 2.0
    true_form = 1.0 - 1.0 / _np.sqrt(3.0)
    assert true_form / sample_form == pytest.approx(3.154701, abs=1e-5)
    rng = np.random.default_rng(99)
    n, reps = 1024, 3000
    v = np.array([epps_pulley_knots(rng.standard_normal(n))["distance"] for _ in range(reps)]) * n
    se = v.std(ddof=1) / np.sqrt(reps)
    assert abs(v.mean() - sample_form) < 4 * se, (
        f"sample-standardised null level {v.mean():.5f} vs closed form {sample_form:.5f} "
        f"(4*se = {4*se:.5f})")
    # and the true-moment null must hit the OTHER constant
    u = np.array([epps_pulley_knots(rng.standard_normal(n), standardise=False)["distance"]
                  for _ in range(reps)]) * n
    seu = u.std(ddof=1) / np.sqrt(reps)
    assert abs(u.mean() - true_form) < 4 * seu, (
        f"true-moment null level {u.mean():.5f} vs closed form {true_form:.5f}")


def test_null_statistic_is_order_one_in_n_but_distance_falls():
    """The scaling claim in the module docstring, asserted: `stat` is O(1) under normality while
    `distance` -> 0. Getting these the wrong way round is how you compare unequal n and get a
    wrong answer."""
    rng = np.random.default_rng(5)
    s, d = {}, {}
    for n in (256, 4096):
        v = np.array([epps_pulley_knots(rng.standard_normal(n))["stat"] for _ in range(60)])
        s[n], d[n] = float(v.mean()), float((v / n).mean())
    assert 0.3 < s[4096] / s[256] < 3.0, f"stat should be O(1) in n: {s}"
    assert d[4096] < 0.4 * d[256], f"distance should fall ~1/n: {d}"


@pytest.mark.parametrize("seed", [13, 101, 202, 303, 404])
def test_sigreg_detects_a_collapsed_cloud_ONLY_without_standardisation(seed):
    """REWRITTEN. The previous version of this test asserted something mathematically FALSE and
    passed on a coin flip (8 of 20 data seeds).

    `randn(n,4) @ randn(4,dim)` has rows A'g with g ~ N(0, I_4), so every row is EXACTLY Gaussian
    with a singular covariance, and therefore **every 1-D projection of it is exactly Gaussian**.
    A standardised shape statistic must be blind to it, and the old assertion
    `b["mean"] > a["mean"]` was asking for a difference that cannot exist. It happened to hold on
    seed 13.

    What is actually true, and is what SIGReg relies on: the deviation lives in the SCALE along
    null-space directions, so the UNSTANDARDISED statistic sees it by a wide margin while the
    standardised one cannot. That is the correct gate, and it also pins the default."""
    rng = np.random.default_rng(seed)
    n, dim = 1024, 64
    iso = rng.standard_normal((n, dim))
    low = rng.standard_normal((n, 4)) @ rng.standard_normal((4, dim))
    # LeJEPA's statistic: no standardisation. Must separate them decisively.
    a = sigreg(iso, n_proj=128, seed=0, standardise=False)
    b = sigreg(low, n_proj=128, seed=0, standardise=False)
    assert b["mean"] > 10 * a["mean"], (
        f"unstandardised SIGReg must flag the collapse: iso {a['mean']:.6f} vs low {b['mean']:.6f}")
    # Shape-only: must be effectively blind, because every projection really is Gaussian.
    a2 = sigreg(iso, n_proj=128, seed=0, standardise=True)
    b2 = sigreg(low, n_proj=128, seed=0, standardise=True)
    assert abs(b2["mean"] - a2["mean"]) < 0.002, (
        f"standardised version should be blind to a Gaussian-with-singular-covariance cloud, "
        f"got iso {a2['mean']:.6f} vs low {b2['mean']:.6f}")


@pytest.mark.parametrize("seed", [77, 78, 79])
def test_standardising_makes_sigreg_scale_blind(seed):
    """The property that makes `standardise=True` the wrong default for anything called SIGReg.

    Note the SHAPE of the dependence, which I first got wrong: the unstandardised statistic
    responds to DISTANCE FROM UNIT SCALE, not monotonically to scale -- a cloud whose projections
    happen to have variance near 1 scores low, and mis-scaling in either direction scores high.
    So the assertion is on the SPREAD across scales, not on a max/min ratio."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((1024, 4)) @ rng.standard_normal((4, 64))
    scales = (0.03, 1.0, 30.0)
    shape = [sigreg(base * s, n_proj=64, seed=0, standardise=True)["mean"] for s in scales]
    raw = [sigreg(base * s, n_proj=64, seed=0, standardise=False)["mean"] for s in scales]
    assert max(shape) - min(shape) < 1e-9, (
        f"shape-only must be EXACTLY scale-invariant, got {shape}")
    assert max(raw) - min(raw) > 0.05, (
        f"unstandardised must respond to scale, got {raw}")
    assert max(raw) > 20 * max(shape), (
        f"unstandardised must reach a far larger value than shape-only: "
        f"raw max {max(raw):.4f} vs shape max {max(shape):.6f}")


def test_effective_rank_cannot_tell_heavy_tails_from_collapse_but_sigreg_can():
    """I predicted a full-rank Cauchy cloud would have HIGH effective rank. It does not: 17.2
    against 63.5 for a Gaussian cloud of identical shape, because a few extreme rows dominate
    the leading singular directions. So effective rank reads heavy-tailedness as collapse.

    That is the substantive reason not to use effective rank alone as the diagnostic on a sparse
    count-fingerprint matrix, where extreme rows are the norm -- and the reason SIGReg is a
    different instrument rather than a dressed-up rank: it rejects this cloud outright."""
    rng = np.random.default_rng(17)
    heavy = rng.standard_cauchy((1024, 64))
    iso = np.random.default_rng(18).standard_normal((1024, 64))
    e_heavy = effective_rank(heavy)["erank"]
    e_iso = effective_rank(iso)["erank"]
    assert e_heavy < 0.5 * e_iso, f"heavy {e_heavy} vs iso {e_iso}"
    assert sigreg(heavy, n_proj=64, seed=0)["mean"] > 0.05, "SIGReg must reject it"
    assert np.linalg.matrix_rank(heavy) == 64, "it really is full rank"


def test_the_statistic_measures_shape_not_spread():
    """A trap worth asserting, because Phase 2's manipulation check depends on getting it right.
    Standardisation removes location AND scale, so two Gaussians with sd 0.03 and sd 0.7 are
    INDISTINGUISHABLE to this statistic. It cannot therefore be applied to a set of imputed
    values on its own to ask "are these concentrated?" -- concentration is only meaningful
    relative to the spread of the labels they join, so the statistic must be computed on the
    COMBINED label vector."""
    rng = np.random.default_rng(43)
    tight = 2.1 + 0.03 * rng.standard_normal(4000)
    wide = 2.1 + 0.70 * rng.standard_normal(4000)
    a = epps_pulley(tight)["distance"]
    b = epps_pulley(wide)["distance"]
    assert a == pytest.approx(b, abs=5e-4), "shape-invariance claim is wrong"
    # ... whereas in the combined vector, the tight component IS a near point mass:
    rest = rng.standard_normal(4000) * 0.9 + 5.0
    ca = epps_pulley(np.concatenate([tight, rest]))["distance"]
    cb = epps_pulley(np.concatenate([wide, rest]))["distance"]
    assert ca > 1.3 * cb, f"combined: tight {ca:.4f} should far exceed wide {cb:.4f}"


@pytest.mark.parametrize("k", [1, 3, 10])
def test_effective_rank_recovers_a_known_rank(k):
    """erank of a matrix with k EQUAL singular values is exactly k."""
    rng = np.random.default_rng(23)
    q, _ = np.linalg.qr(rng.standard_normal((128, 128)))
    A = q[:, :k] @ q[:, :k].T * 3.0
    assert effective_rank(A, centre=False)["erank"] == pytest.approx(k, rel=1e-6)


def test_effective_rank_ordering_and_the_ambient_dimension_caveat():
    """A Gaussian matrix is full rank but its erank is well BELOW d, because p is built from the
    singular values and not their squares. Asserted so nobody later reads erank against d."""
    rng = np.random.default_rng(29)
    n, dim = 2000, 100
    iso = rng.standard_normal((n, dim))
    e_iso = effective_rank(iso)["erank"]
    e_low = effective_rank(rng.standard_normal((n, 5)) @ rng.standard_normal((5, dim)))["erank"]
    assert e_low < 6 < e_iso < dim, f"low {e_low}, iso {e_iso}, d {dim}"
    assert rankme(iso) == pytest.approx(e_iso, rel=1e-3)


def test_effective_rank_is_scale_sensitive_across_columns():
    """The normalisation caveat, asserted rather than asserted-in-prose: rescaling ONE column of
    a fixed feature matrix changes its effective rank, so an erank reported without saying how
    the matrix was normalised means nothing."""
    rng = np.random.default_rng(31)
    A = rng.standard_normal((500, 20))
    B = A.copy()
    B[:, 0] *= 200.0
    assert effective_rank(B)["erank"] < 0.5 * effective_rank(A)["erank"]


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        epps_pulley(np.array([1.0]))
    with pytest.raises(ValueError):
        sigreg(np.full((10, 3), np.nan))      # non-finite is the only raising condition now


@pytest.mark.parametrize("value", [0.0, 4.2])
def test_sigreg_on_a_rank_zero_input_reports_rather_than_raises(value):
    """REWRITTEN. The old version asserted `sigreg(np.zeros((10,3)))` RAISES, and it passed only
    because 0.0 is exactly representable: with the old `std(ddof=1) <= 0` filter, a matrix of
    4.2s left ULP noise, 23 of 32 projections survived the filter, and the call returned a number.
    Same collapsed input, two different behaviours, decided by float representability -- and the
    raise contradicted the module's own stated intent that a point mass "is not an error to
    raise: it is the limiting case we most want a number for".

    Now neither raises, and the two paths behave as their definitions require -- which is NOT the
    same answer, and that is correct rather than a defect:

      * standardise=True removes location and scale, so a constant projection maps to all-zeros
        and the statistic is the point-mass CEILING, with every projection flagged degenerate.
      * standardise=False (LeJEPA's form) feeds the raw constant c to the test, and
        |e^{itc} - e^{-t^2/2}|^2 is a large c-DEPENDENT value, not the ceiling. Nothing is
        standardised, so nothing is flagged degenerate.

    I asserted the ceiling for both at first, which was the old default leaking into the
    expectation."""
    Z = np.full((512, 16), value)
    s_shape = sigreg(Z, n_proj=32, standardise=True)
    assert s_shape["n_proj"] == 32, "every projection counted, not silently dropped"
    assert s_shape["n_degenerate"] == 32, "every projection of a constant matrix IS a point mass"
    assert s_shape["mean"] == pytest.approx(POINT_MASS_DISTANCE, abs=KNOT17_FLOOR)
    s_raw = sigreg(Z, n_proj=32, standardise=False)
    assert s_raw["n_degenerate"] == 0, "no standardisation means nothing is flagged"
    assert s_raw["mean"] > 0.1, f"a rank-zero cloud must score high unstandardised, got {s_raw}"


# --- LeJEPA's own quadrature, added after reading its reference implementation -------------
# Two research briefs gave contradictory accounts of this (one had the weight right and the grid
# wrong, the other had both wrong), so it is pinned here against the source: trapezoid over the
# half-line [0, 3] with interior weights doubled, window exp(-t^2/2), times N.

def test_lejepa_quadrature_matches_the_exact_integral_to_a_RELATIVE_tolerance():
    """Unlike the Gauss-Hermite variant above, LeJEPA's trapezoid-on-[0,3] has a RELATIVE error
    (~2-3%, from truncating |t| > 3) rather than an absolute floor -- so its absolute error
    SHRINKS with n. findings/08 originally asserted the opposite, having measured the wrong
    quadrature; this test pins the correct behaviour so the claim cannot drift back."""
    for n in (256, 2048):
        rng = np.random.default_rng(99)
        pairs = [(epps_pulley(x, beta=1.0)["distance"], epps_pulley_lejepa(x)["distance"])
                 for x in (rng.standard_normal(n) for _ in range(12))]
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        rel = float(np.mean(np.abs(a - b) / a))
        assert rel < 0.08, f"n={n}: LeJEPA quadrature off by {rel:.1%} relative"
        assert float(np.mean(b - a)) < 0, "truncation should bias it LOW"


def test_lejepa_quadrature_on_a_point_mass():
    k = epps_pulley_lejepa(np.full(500, 2.211))
    assert k["degenerate"] is True
    # [-3,3] truncation loses a little of the point-mass integral; it must still be close.
    assert k["distance"] == pytest.approx(POINT_MASS_DISTANCE, rel=0.05)
    assert k["stat_lejepa"] == pytest.approx(
        k["distance"] * 500 * np.sqrt(2 * np.pi), rel=1e-12)


@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
def test_generalised_closed_form_matches_its_own_quadrature(beta):
    """The closed form is now parameterised by the Gaussian smoothing beta. Check it against a
    brute-force numerical integral of the defining BHEP integral at each beta, so a wrong
    algebraic generalisation cannot pass."""
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.standard_normal(300), rng.standard_exponential(100)])
    y = (x - x.mean()) / x.std(ddof=1)
    t = np.linspace(-40.0, 40.0, 200001)
    phi_re = np.cos(t[:, None] * y[None, :]).mean(axis=1)
    phi_im = np.sin(t[:, None] * y[None, :]).mean(axis=1)
    integrand = ((phi_re - np.exp(-0.5 * t * t)) ** 2 + phi_im ** 2)
    w = np.exp(-0.5 * (t / beta) ** 2) / (beta * np.sqrt(2 * np.pi))
    numeric = float(np.trapezoid(integrand * w, t))
    closed = epps_pulley(x, beta=beta)["distance"]
    assert closed == pytest.approx(numeric, rel=1e-6), f"beta={beta}"


def test_rankme_reproduces_the_published_epsilon_convention():
    """The published RankMe/LiDAR formula puts eps OUTSIDE the normalisation, so sum(p) = 1+Q*eps.
    Reproduced deliberately -- renormalising would give a different number from every published
    RankMe. Asserted so a well-meaning 'fix' is caught."""
    rng = np.random.default_rng(3)
    A = rng.standard_normal((500, 40))
    s = np.linalg.svd(A - A.mean(0, keepdims=True), compute_uv=False)
    p = s / s.sum() + 1e-7
    expected = float(np.exp(-(p * np.log(p)).sum()))
    assert rankme(A) == pytest.approx(expected, rel=1e-12)
    # and it must differ measurably from the renormalised variant at large Q
    B = rng.standard_normal((200, 3000))
    sB = np.linalg.svd(B - B.mean(0, keepdims=True), compute_uv=False)
    pB = sB / sB.sum() + 1e-7
    renorm = pB / pB.sum()
    assert abs(np.exp(-(pB * np.log(pB)).sum())
               - np.exp(-(renorm * np.log(renorm)).sum())) > 0
