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
    "generate_internal_knots",
    "quantile_knots",
    "select_knots",
    "spanning_knots",
    "uniform_knots",
]


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
        "spanning_knots",
        "strategy",
        strategy,
        "must be 'uniform' or 'quantile'",
        valid={"quantile", "uniform"},
    )


def generate_internal_knots(x: np.ndarray, n_knots: int, strategy: str = "quantile") -> np.ndarray:
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
        "generate_internal_knots",
        "strategy",
        strategy,
        "must be 'uniform' or 'quantile'",
        valid={"quantile", "uniform"},
    )


def select_knots(knots: np.ndarray, count: int) -> np.ndarray:
    """Down-sample ``knots`` to ``count`` evenly spaced entries."""
    if len(knots) <= count:
        return knots
    idx = np.linspace(0, len(knots) - 1, count).round().astype(int)
    return knots[idx]
