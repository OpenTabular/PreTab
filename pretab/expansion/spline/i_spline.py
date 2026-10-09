"""
Integrated Spline (I-Spline) Transformer.

I-splines are monotonically increasing basis functions obtained by integrating
M-splines. They are useful whenever a monotone relationship between a feature and
the target should be representable.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from ...core.parameters import UNSET
from ...core.policy import RepresentationPolicy
from .base import BaseSplineTransformer


class ISplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using an I-spline (integrated spline) basis.

    Each basis function is the integral of an M-spline over the fitted range,
    which yields a monotonically increasing, non-negative function normalized to
    ``[0, 1]``. Knot placement follows the shared priority: explicit
    (``knot_locations``) > target-aware (``placement_strategy``) > automatic
    (``output_dim`` with ``placement_strategy``).

    See :class:`~pretab.expansion.spline.base.BaseSplineTransformer`
    for the full parameter description. ``include_bias`` defaults to False here.
    Because I-splines start at zero, a bias term may be useful for a non-zero
    intercept.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ISplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> ISplineTransformer(output_dim=8).fit_transform(X).shape
    (50, 8)
    """

    _representation_family = "ispline"

    def __init__(
        self,
        output_dim=UNSET,
        degree: int = 3,
        include_bias: bool = False,
        knot_locations: np.ndarray | None = None,
        target_aware: bool = False,
        placement_strategy: str = "quantile",
        task: Literal["regression", "classification"] | None = None,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
        policy: RepresentationPolicy | dict | None = None,
    ):
        super().__init__(
            output_dim=output_dim,
            degree=degree,
            include_bias=include_bias,
            knot_locations=knot_locations,
            target_aware=target_aware,
            placement_strategy=placement_strategy,
            task=task,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
            policy=policy,
        )

    def _feature_suffix(self) -> str:
        return "is"

    def _ispline_columns(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        """
        Evaluate every I-spline basis function at ``x`` in closed form.

        ``I_k(x)`` is the integral of the normalized M-spline ``M_k`` from the left
        boundary, i.e. the antiderivative of the B-spline ``B_k`` divided by its
        full-range integral ``(t_{k+p+1} - t_k) / (p + 1)``. The antiderivative of
        a B-spline is itself a B-spline of degree ``p + 1``, so the basis is exact
        for any knot spacing (a fixed quadrature grid cannot resolve basis
        functions whose support is narrower than the grid step). Values below /
        above the knot range are 0 / 1. A degenerate basis function with
        zero-width support (only possible for a knot vector with a repeated
        knot, e.g. one fitted before boundary knots were de-duplicated)
        integrates to the step function at that knot, its limit as the support
        shrinks.
        """
        n_coef = len(knots) - self.degree - 1
        lower, upper = knots[0], knots[-1]
        antiderivative = BSpline(knots, np.eye(n_coef), self.degree).antiderivative()
        total = (knots[self.degree + 1 : self.degree + 1 + n_coef] - knots[:n_coef]) / (self.degree + 1)

        x_clipped = np.clip(x, lower, upper)
        values = antiderivative(x_clipped) - antiderivative(lower)
        # Every basis function is fully integrated at the right boundary; pin it
        # instead of evaluating there, where a repeated boundary knot would put
        # the evaluation on a zero-width interval.
        values = np.where(x_clipped[:, None] >= upper, total, values)
        live = total > 0
        values[:, live] = values[:, live] / total[live]
        values[:, ~live] = (x_clipped[:, None] >= knots[:n_coef][~live]).astype(float)
        return np.clip(values, 0.0, 1.0)

    def _ispline_basis(self, x: np.ndarray, knots: np.ndarray, basis_idx: int) -> np.ndarray:
        """Compute a single I-spline basis function (see :meth:`_ispline_columns`)."""
        n_coef = len(knots) - self.degree - 1
        if basis_idx >= n_coef:
            return np.zeros(len(x))
        return self._ispline_columns(np.asarray(x, dtype=float), knots)[:, basis_idx]

    def _design_matrix(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        design = self._ispline_columns(np.asarray(x, dtype=float), knots)
        return np.nan_to_num(design, nan=0.0)
