"""Score a validation set in the similarity regime the blinded test set actually occupies.

Measured 2026-10-09, nearest-neighbour Tanimoto (ECFP4 counts, 2048 bits) to the training set:

    NN band      blinded TEST   our VALIDATION
    0.0-0.3            0.0%            5.1%
    0.3-0.5            4.8%           65.2%
    0.5-0.7           84.9%           25.6%
    0.7-0.9           10.1%            3.3%
    0.9-1.0            0.1%            0.9%
    median             0.587           0.439

**Two thirds of our validation sits in a band that is under 5% of the test set.** The blinded
set is an analog expansion -- compounds purchased *around* the training chemistry -- while a
scaffold split over a set that is 82% scaffold-singletons produces something close to a random,
and therefore more diverse, partition. So the headline validation rho has been measuring a
regime the board barely contains.

Restricting to `NN >= 0.5` (which the existing guards do) covers 95% of the test set and is
sound, but it throws away three quarters of the validation compounds and the power with them.
Importance weighting keeps them: each validation compound is weighted by how much more or less
common its NN band is in the test set than in validation, so the estimate is of *test-regime*
rho while still using every compound.

**That power claim is refuted -- by the Kish number this module already prints.** Measured over
48 cells (12 seeds x 4 isoforms, 2026-10-09 late, `scripts/cyp_split_match_paired.py`):

    matched_spearman vs the NN >= 0.5 restriction   Pearson r 0.9857, mean |diff| 0.0164
    matched_spearman Kish ESS                       67 of 270 scored = 0.24 n
    the NN >= 0.5 subset                            67 of 270        = 0.24 n

Because 84.9% of the test weight sits in a single band, the weighting delivers *exactly* the
effective sample size of the restriction it was written to replace, and tracks it at r = 0.986.
It keeps every compound formally, and none of the power. It also costs **2.4x** the per-seed
sd of plain rho (0.0925 against 0.0386), while being **no closer to the board's rho** than the
plain scaffold split: paired over 12 fresh seeds, `prodmatch - all` = +0.0130 +- 0.0133, CI
crossing zero.

So this function is kept as a diagnostic of *where* validation sits, and must **not** be used
as the selection instrument or to adjudicate a lever. Use plain rho on the full held-out set.
See `docs/drugrx-training/PROGRESS.md`, "The board-matched validation surface has a measured
cost and no measured benefit".

Spearman has no standard weighted form, so this resamples instead: draw with replacement in
proportion to the weights, take rho, repeat. The mean over draws estimates the matched rho and
the spread across draws is an honest statement of how much of the apparent precision was really
coming from compounds the board will never ask about.

A band the test set does not occupy gets weight zero, so a compound at NN 0.2 contributes
nothing -- which is the point, not a defect.
"""
from __future__ import annotations

import numpy as np

#: Fraction of the 750 blinded test compounds in each NN-to-train band. Measured, not assumed.
TEST_NN_BANDS: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.3, 0.000),
    (0.3, 0.5, 0.048),
    (0.5, 0.7, 0.849),
    (0.7, 0.9, 0.101),
    (0.9, 1.01, 0.001),
)


def band_of(sim: np.ndarray) -> np.ndarray:
    """Index of the NN band each similarity falls in."""
    out = np.full(len(sim), -1, dtype=int)
    for b, (lo, hi, _) in enumerate(TEST_NN_BANDS):
        out[(sim >= lo) & (sim < hi)] = b
    return out


def matching_weights(sim: np.ndarray) -> np.ndarray:
    """Per-compound weight making the validation NN histogram match the test set's.

    `w_i = p_test(band_i) / p_val(band_i)`, normalised. Empty validation bands contribute
    nothing and cannot divide by zero; test-empty bands correctly receive weight zero.
    """
    b = band_of(sim)
    w = np.zeros(len(sim), dtype=float)
    for k, (_, _, p_test) in enumerate(TEST_NN_BANDS):
        m = b == k
        n = int(m.sum())
        if n == 0 or p_test <= 0:
            continue
        w[m] = p_test / n
    s = w.sum()
    return w / s if s > 0 else w


def matched_spearman(pred: np.ndarray, truth: np.ndarray, sim: np.ndarray,
                     draws: int = 200, seed: int = 0) -> tuple[float, float, int]:
    """Spearman rho re-weighted to the test set's similarity regime.

    Returns `(mean rho over draws, sd across draws, effective sample size)`. The effective
    sample size is Kish's -- with most weight on one band it is far below the nominal count,
    and reporting it stops a matched estimate being read as if it had the full n behind it.
    """
    from scipy.stats import spearmanr

    ok = np.isfinite(pred) & np.isfinite(truth) & np.isfinite(sim)
    pred, truth, sim = pred[ok], truth[ok], sim[ok]
    w = matching_weights(sim)
    if w.sum() <= 0 or (w > 0).sum() < 10:
        return float("nan"), float("nan"), 0

    ess = int(round(w.sum() ** 2 / np.square(w).sum()))
    rng = np.random.default_rng(seed)
    n = len(pred)
    vals = []
    for _ in range(draws):
        idx = rng.choice(n, size=n, replace=True, p=w)
        # An all-identical resample has no rank variation and yields NaN; skip rather than
        # poison the mean.
        if len(np.unique(truth[idx])) < 3 or len(np.unique(pred[idx])) < 3:
            continue
        r = spearmanr(pred[idx], truth[idx]).statistic
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return float("nan"), float("nan"), ess
    return float(np.mean(vals)), float(np.std(vals, ddof=1)), ess
