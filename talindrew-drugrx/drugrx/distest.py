"""Distributional-shape instruments, taken from LeJEPA/LeWAM and used here as MEASUREMENT.

Two objects, both borrowed from `LeWAM: End-to-End World and Action Modeling with JEPAs`
(Fu, Siebert, Halicki, Balestriero; https://le-wam.github.io/), which regularises its latent
states with

    L_SIGReg = (1/M) sum_m T( Z u^(m) )

for M random unit directions u^(m) and T the univariate Epps-Pulley statistic against N(0,1)
-- 1024 projections and 17 knots in their config -- and reports effective rank (Roy & Vetterli
2007) as its anti-collapse readout.

We are not training an encoder here, so SIGReg cannot act as a regulariser on our pipeline:
gradient-boosted trees are invariant to monotone per-feature transforms, so reshaping the
marginal geometry of a fixed feature matrix cannot change a single prediction. What IS useful is
the statistic itself, as an instrument. `findings/07` failed because an arm was run that did not
vary the quantity its claim was about -- the arm changed a point mass's LOCATION and left its
CONCENTRATION intact -- and nothing in the lab measured concentration. T is exactly that
measurement.

THE STATISTIC, DERIVED RATHER THAN COPIED. Epps-Pulley is the BHEP integral

    T = n * integral | phi_n(t) - e^{-t^2/2} |^2 w(t) dt ,    phi_n(t) = (1/n) sum_j e^{i t y_j}

with Gaussian weight w. Expanding the modulus and integrating term by term with
`integral e^{iat} w(t) dt = e^{-a^2 b^2/2}` for w = N(0, b^2) gives, at b = 1,

    T = 1 + n/sqrt(3) + (2/n) sum_{j<k} exp(-(y_j-y_k)^2/2) - sqrt(2) sum_j exp(-y_j^2/4)

on standardised y. That closed form is implemented in `epps_pulley`. The weight at b = 1 is
exactly the N(0,1) density, so the SAME integral is a Gauss-Hermite quadrature in 17 nodes --
which is what "17 knots" means, and what `epps_pulley_knots` does. The two are derived from one
integral by different routes, so they are a real cross-check on each other, and
`tests/test_distest.py` asserts they agree.

SCALING, which matters for how the number is read:

* `stat` (= T) is the TEST statistic. O(1) under normality, O(n) away from it, so it is what a
  critical value applies to.
* `distance` (= T/n) is the EFFECT SIZE -- a squared-characteristic-function distance, which
  tends to 0 under normality and to a positive constant otherwise. It is comparable across
  sample sizes, so it is the one to use when comparing isoforms with different n.

Report both. Comparing `stat` across unequal n is the obvious way to get a wrong answer here.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "epps_pulley",
    "epps_pulley_knots",
    "sigreg",
    "effective_rank",
    "rankme",
    "ep_null",
]

#: Pure point mass, standardised to zeros: T/n -> 1 - sqrt(2) + 1/sqrt(3). The ceiling of
#: `distance`, and the thing a concentrated imputation sits near. Asserted in the tests.
POINT_MASS_DISTANCE = 1.0 - np.sqrt(2.0) + 1.0 / np.sqrt(3.0)


#: How many ULPs of spread still count as a point mass. NOT a cosmetic epsilon: a literal point
#: mass does not have sd 0 in floating point. `np.full(1000, 4.2).std(ddof=1)` is **1.78e-15**,
#: because 4.2 is not representable and `x - x.mean()` leaves rounding noise. Testing `sd <= 0`
#: therefore does not fire, the code divides by 1.78e-15, and the amplified noise is reported as
#: a distance of 0.476 instead of the ceiling 0.163 -- a 2.9x error in exactly the case this
#: instrument exists for.
#:
#: THE GUARD IS ON ptp, NOT ON sd RELATIVE TO THE MEAN, and that is a correction. The first
#: version tested `sd <= 1e-12 * max(|mean|, 1)`, which has a FALSE-POSITIVE direction that no
#: test covered: `1e12 + 0.1*randn(4000)` is 4000 distinct float64s drawn from a clean Gaussian,
#: and it was reported as `degenerate=True, distance=0.163137`. Peak-to-peak against the local
#: floating-point spacing is scale-aware and cannot do that -- `np.full(n, v)` has ptp exactly 0
#: for every v, while a real Gaussian at any magnitude has ptp many ULPs wide.
_DEGENERATE_ULPS = 4.0


def _standardise(x: np.ndarray) -> tuple[np.ndarray, bool]:
    """Centre and scale to unit sd. Returns (y, degenerate).

    A point mass has no standardisation. That is not an error to raise: it is the limiting case
    we most want a number for, so it maps to all-zeros, whose T is the analytic ceiling.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError(f"need at least 2 finite values, got {x.size}")
    mean = float(x.mean())
    spread = float(np.ptp(x))
    # Scale-aware: is the spread at the level of floating-point noise FOR THIS MAGNITUDE?
    tol = _DEGENERATE_ULPS * float(np.spacing(max(abs(mean), abs(x).max(), 1.0)))
    if spread <= tol:
        return np.zeros_like(x), True
    sd = float(x.std(ddof=1))
    if sd <= 0:                      # unreachable given the ptp test, kept as a hard floor
        return np.zeros_like(x), True
    return (x - mean) / sd, False


#: LeJEPA's quadrature, read off its own reference implementation (`MINIMAL.md` in
#: github.com/galilai-group/lejepa), NOT from either of two research briefs that disagreed
#: about it -- one had the weight right and the grid wrong, the other had both wrong:
#:
#:     t = torch.linspace(0, 3, knots); dt = 3 / (knots - 1)
#:     weights = full((knots,), 2 * dt); weights[[0, -1]] = dt     # trapezoid, doubled
#:     window = torch.exp(-t.square() / 2.0)                       # w(t) = e^{-t^2/2}
#:     statistic = (err @ (weights * window)) * N
#:
#: So: trapezoid over the HALF-line [0, 3] with the weights doubled to exploit evenness
#: (i.e. effectively [-3, 3], not [-5, 5]), an UNNORMALISED Gaussian window, and a factor N.
#: Against this module's normalised convention the scale factor is sqrt(2*pi).
LEJEPA_TMAX = 3.0
LEJEPA_KNOTS = 17


def epps_pulley(x: np.ndarray, *, beta: float = 1.0, standardise: bool = True,
                chunk: int = 2048) -> dict[str, float]:
    """Exact Epps-Pulley statistic against N(0,1), O(n^2) in time, O(n*chunk) in memory.

    `beta` is the Gaussian smoothing parameter of the weight, w = N(0, beta^2) as a density.
    beta = 1 is the textbook test; `BETA_LEJEPA` is what LeJEPA uses. Closed form, from the
    BHEP integral with `int e^{iat} w(t) dt = e^{-a^2 beta^2/2}`:

        T/n = (1/n^2) sum_{j,k} e^{-b2 d_jk^2/2} - (2g/(n b)) sum_j e^{-g2 y_j^2/2} + d/b
        g^2 = b2/(1+b2),   d^2 = b2/(1+2 b2)

    At beta = 1 this reduces to the familiar
    `1 + n/sqrt(3) + (2/n) sum_{j<k} e^{-d^2/2} - sqrt(2) sum_j e^{-y^2/4}`.

    Chunked because the full pairwise matrix at our n (6145) is 302 MB, which is the kind of
    allocation that works on the laptop and dies in a loop over four isoforms and six seeds.
    """
    y, degenerate = _standardise(x) if standardise else (
        np.asarray(x, float).ravel(), False)
    n = y.size
    b2 = float(beta) ** 2
    g = np.sqrt(b2 / (1.0 + b2))
    d = np.sqrt(b2 / (1.0 + 2.0 * b2))
    cross = 0.0
    for i in range(0, n, chunk):
        blk = y[i:i + chunk, None] - y[None, :]
        cross += float(np.exp(-0.5 * b2 * blk * blk).sum())    # includes the j==k diagonal
    single = float(np.exp(-0.5 * (g ** 2) * y * y).sum())
    dist = cross / (n * n) - (2.0 * g / (n * beta)) * single + d / beta
    return dict(stat=float(n * dist), distance=float(dist), n=int(n),
                degenerate=bool(degenerate), beta=float(beta))


def epps_pulley_lejepa(x: np.ndarray, *, knots: int = LEJEPA_KNOTS,
                       tmax: float = LEJEPA_TMAX,
                       standardise: bool = True) -> dict[str, float]:
    """LeJEPA's SIGReg statistic exactly as its reference implementation computes it.

    Trapezoid on `linspace(0, tmax, knots)` with the interior weights doubled, window
    `exp(-t^2/2)`, times N. `stat_lejepa` is their raw number; `distance` divides by N and by
    sqrt(2*pi) so it is comparable with `epps_pulley(beta=1)`, whose weight is the same Gaussian
    normalised to a density over all of R.

    The two differ only by the truncation at |t| = tmax and the quadrature, so comparing them
    measures exactly what their 17-knot choice costs. Done in `tests/test_distest.py`.
    """
    y, degenerate = _standardise(x) if standardise else (
        np.asarray(x, float).ravel(), False)
    n = y.size
    t = np.linspace(0.0, tmax, knots)
    dt = tmax / (knots - 1)
    w = np.full(knots, 2.0 * dt)
    w[0] = w[-1] = dt
    w = w * np.exp(-0.5 * t * t)
    ty = t[:, None] * y[None, :]
    re = np.cos(ty).mean(axis=1) - np.exp(-0.5 * t * t)
    im = np.sin(ty).mean(axis=1)
    raw = float(np.dot(w, re * re + im * im))
    return dict(stat_lejepa=float(n * raw), distance=float(raw / np.sqrt(2.0 * np.pi)),
                stat=float(n * raw / np.sqrt(2.0 * np.pi)), n=int(n),
                degenerate=bool(degenerate))


def _hermite_nodes(knots: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights for integral f(t) * N(t;0,1) dt ~= sum w_i f(t_i)."""
    t, w = np.polynomial.hermite_e.hermegauss(knots)
    return t, w / w.sum()               # hermegauss weights sum to sqrt(2*pi); normalise to 1


def epps_pulley_knots(x: np.ndarray, *, knots: int = 17,
                      standardise: bool = True) -> dict[str, float]:
    """The same statistic by Gauss-Hermite quadrature in `knots` nodes. O(n*knots).

    This is the form LeJEPA/LeWAM actually use (17 knots), and the only form that is affordable
    inside a training loop.

    ACCURACY, measured against the exact form rather than assumed. The integrand carries
    `cos(t*y_j)` terms, so its oscillation frequency is set by `max|y|` after standardisation,
    and 17 Gauss-Hermite nodes are exact only up to polynomial degree 33:

        max|y| ~ 3   (well-behaved sample)      relative error ~1e-3, falling as 1/n
        max|y| ~ 21  (heavy-tailed sample)      relative error ~2e-3, and NOT monotone in
                                                `knots` -- 33 knots was worse than 17, and
                                                only 257 reached machine precision

    So 17 knots is fine as a training signal and fine for ordering samples, but the exact
    `epps_pulley` is the reference for any number that gets written into a finding. At our
    n = 6145 the O(n^2) form costs well under a second, so there is no reason to quote the
    approximation in a result.
    """
    y, degenerate = _standardise(x) if standardise else (
        np.asarray(x, float).ravel(), False)
    n = y.size
    t, w = _hermite_nodes(knots)
    ty = t[:, None] * y[None, :]                       # (knots, n)
    re = np.cos(ty).mean(axis=1) - np.exp(-0.5 * t * t)
    im = np.sin(ty).mean(axis=1)
    dist = float(np.dot(w, re * re + im * im))
    return dict(stat=float(n * dist), distance=dist, n=int(n),
                degenerate=bool(degenerate))


def sigreg(Z: np.ndarray, *, n_proj: int = 256, knots: int = 17,
           seed: int = 0, standardise: bool = False) -> dict[str, float]:
    """Mean Epps-Pulley over `n_proj` random unit directions.

    `standardise=False` IS LeJEPA. `standardise=True` IS NOT, AND THE DIFFERENCE IS LOAD-BEARING.
    This defaulted to True, and that was wrong in a way that inverted a result. Standardising each
    projection removes its location and scale, so the statistic sees only SHAPE -- and the thing
    SIGReg is a regulariser for is largely a SCALE defect. Measured on one collapsed cloud:

        scale 1.0    standardise=True 0.000058    standardise=False 0.000204
        scale 0.1    standardise=True 0.000058    standardise=False 0.156892
        scale 10.0   standardise=True 0.000058    standardise=False 0.277573

    The shape-only version is identical across three orders of magnitude of collapse. So
    `standardise=True` is a legitimate *measurement* choice when the question is about shape, and
    it is the wrong default for anything claiming to be SIGReg.

    Two further corrections from LeJEPA's own `MINIMAL.md`: it uses **256** random directions, not
    the 1024 this module's docstring previously attributed to "their config", and the projections
    are fed to the test raw.

    Returns the mean and the spread across projections. The spread is the honest part: a mean
    distance that is small only because most random directions are uninformative is not the same
    as a distribution that is actually isotropic Gaussian. `n_degenerate` counts projections that
    came out as point masses, which a rank-deficient `Z` will produce.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D (n, d), got shape {Z.shape}")
    n, d = Z.shape
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((d, n_proj))
    U /= np.linalg.norm(U, axis=0, keepdims=True)
    P = Z @ U                                           # (n, n_proj)
    out, degen = [], 0
    for m in range(n_proj):
        col = P[:, m]
        if not np.isfinite(col).all():
            continue
        # Use the SAME degeneracy test as `_standardise`. The first version of this loop filtered
        # on `col.std(ddof=1) <= 0` -- the precise bug this module documents at the top and fixes
        # 60 lines above -- so `sigreg(np.full((512,16), 0.0))` RAISED while
        # `sigreg(np.full((512,16), 4.2))` returned a number from 23 of 32 projections. Same
        # collapsed input, two different answers, decided by float representability.
        r = epps_pulley_knots(col, knots=knots, standardise=standardise)
        if r["degenerate"]:
            degen += 1
        out.append(r["distance"])
    a = np.asarray(out, dtype=float)
    if not a.size:
        raise ValueError("every projection was non-finite -- Z contains NaN or inf")
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)) if a.size > 1 else 0.0,
                p95=float(np.quantile(a, 0.95)), max=float(a.max()),
                n_proj=int(a.size), n_degenerate=int(degen), n=int(n), d=int(d))


def effective_rank(A: np.ndarray, *, centre: bool = True,
                   eps: float = 0.0) -> dict[str, float]:
    """Effective rank, Roy & Vetterli 2007: exp(H(p)) with p_i = s_i / sum(s), s = singvals.

    Note p is built from the singular values THEMSELVES, not their squares -- erank of an
    isotropic Gaussian matrix is therefore well below its ambient dimension even with no
    collapse at all, which is why it is only ever read as a comparison between two matrices of
    the same shape, never against d.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got {A.shape}")
    if centre:
        A = A - A.mean(axis=0, keepdims=True)
    s = np.linalg.svd(A, compute_uv=False)
    s = s[np.isfinite(s) & (s > 0)]
    if not s.size:
        return dict(erank=0.0, n_singvals=0, s_max=0.0, stable_rank=0.0)
    # In both the RankMe and LiDAR papers the epsilon sits OUTSIDE the normalisation, so
    # sum(p) = 1 + Q*eps rather than 1. Reproduced rather than "fixed": renormalising would
    # silently give a different number from every published RankMe. The entropy bias is
    # ~Q*eps*|log eps|, i.e. ~0.005 nats at Q = 3091 with eps = 1e-7 -- a 0.5% effect, which
    # matters only when comparing matrices of different width, where every zero singular value
    # contributes a little.
    p = s / s.sum() + eps
    h = float(-(p * np.log(p)).sum())
    return dict(erank=float(np.exp(h)), n_singvals=int(s.size), s_max=float(s[0]),
                stable_rank=float((s * s).sum() / (s[0] * s[0])))


def rankme(A: np.ndarray, *, centre: bool = True, eps: float = 1e-7) -> float:
    """RankMe (Garrido et al.) -- effective rank with the smoothing epsilon they specify."""
    return effective_rank(A, centre=centre, eps=eps)["erank"]


def ep_null(n: int, *, reps: int = 2000, knots: int = 17, seed: int = 0,
            quantiles=(0.90, 0.95, 0.99)) -> dict[str, float]:
    """Monte-Carlo null distribution of the statistic at sample size n.

    Epps-Pulley's published critical values are for the raw statistic on standardised data, and
    standardising with the SAMPLE mean and sd changes the null -- so calibrate rather than look
    up. This is what makes `distance` interpretable as "how far from normal, in null units".
    """
    rng = np.random.default_rng(seed)
    v = np.array([epps_pulley_knots(rng.standard_normal(n), knots=knots)["stat"]
                  for _ in range(reps)])
    out = {f"q{int(q * 100)}": float(np.quantile(v, q)) for q in quantiles}
    out.update(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=int(n), reps=int(reps))
    return out
