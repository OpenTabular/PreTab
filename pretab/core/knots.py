"""Shared knot-placement primitives for the spline transformers and selectors.

Both the spline transformers (which place knots automatically) and the
target-aware knot selectors (which place knots at decision-tree splits) rely on
the same low-level operations: converting a basis-function count to a number of
internal knots, generating uniform / quantile knots, and down-sampling an
over-full knot vector. Those primitives live here as small pure functions so the
two systems share one implementation and stay numerically consistent.

``np.quantile(x, q)`` with ``q`` in ``[0, 1]`` matches the historical
``np.percentile(x, 100 * q)`` used by the spline transformer, so moving to these
helpers leaves knot positions numerically unchanged.
"""

import numpy as np

from .exceptions import invalid_param_error

__all__ = [
    "basis_to_knots",
    "bspline_basis",
    "generate_internal_knots",
    "quantile_knots",
    "select_knots",
    "spanning_knots",
    "uniform_knots",
]


def _last_positive_span(knots: np.ndarray) -> int:
    """Index of the final knot span with non-zero width, or ``-1`` if none.

    A clamped knot vector repeats its boundary knots ``degree + 1`` times, so
    the trailing spans are degenerate (``knots[j] == knots[j + 1]``). The span
    that actually reaches the right boundary is the last one with positive
    width.
    """
    for j in range(len(knots) - 2, -1, -1):
        if knots[j] < knots[j + 1]:
            return j
    return -1


def bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int, i: int) -> np.ndarray:
    """Evaluate the ``i``-th B-spline basis function of ``degree`` over ``knots``.

    Standard Cox-de Boor recursion, with one deliberate departure: the final
    knot span of non-zero width is treated as **closed** rather than half-open.
    Under the usual ``[k_j, k_{j+1})`` convention the largest value in the data
    belongs to no span, so every basis function evaluates to zero there and the
    row loses all of its signal. Closing that span keeps the basis a partition
    of unity across the whole fitted range, right endpoint included.

    Parameters
    ----------
    x : ndarray
        Points at which to evaluate the basis function.
    knots : ndarray
        Full (padded) knot vector.
    degree : int
        Degree of the basis function.
    i : int
        Index of the basis function.
    """
    return _bspline_basis(x, knots, degree, i, _last_positive_span(knots))


def _bspline_basis(x: np.ndarray, knots: np.ndarray, degree: int, i: int, last: int) -> np.ndarray:
    """Cox-de Boor recursion with ``last`` naming the span to treat as closed."""
    if degree == 0:
        if i == last:
            return ((knots[i] <= x) & (x <= knots[i + 1])).astype(float)
        return ((knots[i] <= x) & (x < knots[i + 1])).astype(float)

    denom1 = knots[i + degree] - knots[i]
    denom2 = knots[i + degree + 1] - knots[i + 1]
    # A zero denominator marks a degenerate span from a repeated boundary knot;
    # its contribution is zero by convention.
    zero = np.zeros_like(x, dtype=float)
    term1 = zero if denom1 == 0 else (x - knots[i]) / denom1 * _bspline_basis(x, knots, degree - 1, i, last)
    term2 = (
        zero
        if denom2 == 0
        else (knots[i + degree + 1] - x) / denom2 * _bspline_basis(x, knots, degree - 1, i + 1, last)
    )
    return term1 + term2


def basis_to_knots(n_basis: int, degree: int) -> int:
    """Number of internal knots implied by ``n_basis`` basis functions of ``degree``."""
    return max(0, n_basis - degree - 1)


def uniform_knots(x: np.ndarray, n_knots: int) -> np.ndarray:
    """Return ``n_knots`` internal knots evenly spaced across the range of ``x``."""
    if n_knots <= 0:
        return np.array([])
    return np.linspace(x.min(), x.max(), n_knots + 2)[1:-1]


def quantile_knots(x: np.ndarray, n_knots: int) -> np.ndarray:
    """Return ``n_knots`` internal knots at evenly spaced quantiles of ``x``."""
    if n_knots <= 0:
        return np.array([])
    quantiles = np.linspace(0, 1, n_knots + 2)[1:-1]
    return np.quantile(x, quantiles)


def spanning_knots(x: np.ndarray, n_knots: int, strategy: str = "uniform") -> np.ndarray:
    """Return ``n_knots`` knots spanning the full range of ``x``, endpoints included.

    Unlike :func:`uniform_knots` / :func:`quantile_knots` (which return *interior*
    knots for the B/M/I spline convention), the legacy spline families
    (cubic, natural cubic, p-spline, tensor product) place ``n_knots`` points that
    span the whole range, including the minimum and maximum.

    Parameters
    ----------
    x : ndarray
        Values of a single feature.
    n_knots : int
        Number of spanning knots to place; ``<= 0`` returns an empty array.
    strategy : {"uniform", "quantile"}, default="uniform"
        ``"uniform"`` reproduces ``np.linspace(x.min(), x.max(), n_knots)`` exactly;
        ``"quantile"`` places the knots at evenly spaced data quantiles (also
        including the 0th and 100th percentiles).
    """
    if n_knots <= 0:
        return np.array([])
    if strategy == "uniform":
        return np.linspace(x.min(), x.max(), n_knots)
    if strategy == "quantile":
        return np.quantile(x, np.linspace(0, 1, n_knots))
    raise invalid_param_error(
        "spanning_knots", "strategy", strategy,
        "must be 'uniform' or 'quantile'", valid={"quantile", "uniform"},
    )


def generate_internal_knots(
    x: np.ndarray, n_knots: int, strategy: str = "quantile"
) -> np.ndarray:
    """Generate internal knots for one feature using ``strategy``.

    Parameters
    ----------
    x : ndarray
        Values of a single feature.
    n_knots : int
        Number of internal knots to place; ``<= 0`` returns an empty array.
    strategy : {"uniform", "quantile"}, default="quantile"
        Placement rule.
    """
    if strategy == "uniform":
        return uniform_knots(x, n_knots)
    if strategy == "quantile":
        return quantile_knots(x, n_knots)
    raise invalid_param_error(
        "generate_internal_knots", "strategy", strategy,
        "must be 'uniform' or 'quantile'", valid={"quantile", "uniform"},
    )


def select_knots(knots: np.ndarray, count: int) -> np.ndarray:
    """Down-sample ``knots`` to ``count`` evenly spaced entries."""
    if len(knots) <= count:
        return knots
    idx = np.linspace(0, len(knots) - 1, count).round().astype(int)
    return knots[idx]
