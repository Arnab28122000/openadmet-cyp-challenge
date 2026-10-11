"""The point prediction that minimises expected ST-RAE, given a predictive distribution.

ST-RAE charges `relu(lo − p) + relu(p − hi)` per molecule, so the risk is convex and
piecewise-linear in `p` and its derivative is `P(hi < p) − P(lo > p)`. The optimum is the
**balance point** where

    P(lo > p) == P(hi < p)

— a weighted median over the pooled bound draws, not a mean and not a mode. This is the one
place where theory, rather than another feature, is worth something here.

**Why it matters more after the selection fix than before.** Once screen-derived labels are in,
a large share of test compounds are believed inactive, and an inactive compound's interval is an
absolute window (~[1.07, 3.52]) rather than a band around `y`. A mixture of "probably inactive,
wide free window" and "possibly active, narrow band" has a risk minimum that sits well below its
own mean: moving `p` down costs nothing until it leaves the active band, and saves the full
distance on every inactive draw. Averaging a potency estimate with an inactivity probability —
`(1−q)·2.2 + q·ŷ` — is the wrong operation and over-predicts exactly where the metric is harshest.

The caller supplies draws of `y` (ensemble members, quantiles, or a mixture) and a `bounds`
function mapping `y` to `(lo, hi)` — `scripts/cyp_per_endpoint_decode.py::bound_lookup` fits one
per endpoint from the release's own intervals.
"""

from __future__ import annotations

import numpy as np


def balance_point(lo: np.ndarray, hi: np.ndarray, w: np.ndarray | None = None) -> float:
    """The `p` minimising `sum_j w_j (relu(lo_j − p) + relu(p − hi_j))`.

    Exact rather than a grid search: the risk is piecewise linear with knots at the bounds,
    so the minimiser is at a knot. Walking the sorted knots and tracking the subgradient is
    O(n log n).

    **Ties go to the right-hand end of a flat segment.** When every draw says "inactive" the
    risk is flat across the whole free window — about `[1.07, 3.52]` — and both ends score
    zero against the draws, so the choice looks arbitrary and is not: the draws are a model's
    belief, not the truth, and the error that actually happens is a compound we called inactive
    being somewhat active. The right end is 2.45 log units closer to that, and costs nothing
    when the belief is right. (Found by a second reading of this function, which returned the
    left end; the risk value cannot distinguish them, so no test on the risk could have.)
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    w = np.ones_like(lo) if w is None else np.asarray(w, float)
    k = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(w) & (w > 0)
    lo, hi, w = lo[k], hi[k], w[k]
    if len(lo) == 0:
        return float("nan")
    # Events: passing a `lo` from the right removes its +(-1) slope contribution; passing a
    # `hi` adds +1. Slope left of everything is -sum(w); it increases by w at each lo, then
    # again by w at each hi.
    knots = np.concatenate([lo, hi])
    delta = np.concatenate([w, w])
    order = np.argsort(knots, kind="stable")
    knots, delta = knots[order], delta[order]
    slope = -w.sum() + np.cumsum(delta)
    # The minimiser is the first knot whose slope is >= 0; where that slope is exactly 0 the
    # risk is flat up to the next knot with a positive slope, and we take that right-hand end.
    idx = np.flatnonzero(slope > 0)
    if len(idx) == 0:
        return float(knots[-1])
    return float(knots[idx[0]])


def mixture_draws(y_active: np.ndarray, p_inactive: float, floor_y: float,
                  n_floor: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Draws and weights for "inactive with probability `p_inactive`, else `y_active`".

    The inactive branch is a single atom at the floor label; its *interval* is what carries
    the width, so no spread is needed on the draw itself.
    """
    y_active = np.asarray(y_active, float)
    n = len(y_active) if n_floor is None else n_floor
    draws = np.concatenate([y_active, np.full(n, floor_y)])
    w = np.concatenate([np.full(len(y_active), (1.0 - p_inactive) / max(len(y_active), 1)),
                        np.full(n, p_inactive / max(n, 1))])
    return draws, w


def decide(draws: np.ndarray, bounds, w: np.ndarray | None = None) -> float:
    """Balance point for a set of `y` draws, after mapping each to its own `(lo, hi)`."""
    lo, hi = bounds(np.asarray(draws, float))
    return balance_point(lo, hi, w)


def decide_many(draws: np.ndarray, bounds, w: np.ndarray | None = None) -> np.ndarray:
    """`decide` per row of a `(n_molecules, n_draws)` array."""
    draws = np.atleast_2d(np.asarray(draws, float))
    return np.array([decide(row, bounds, w) for row in draws])
