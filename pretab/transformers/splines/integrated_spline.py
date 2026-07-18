"""
Integrated Spline (I-Spline) Transformer.

I-splines are monotonically increasing basis functions obtained by integrating
M-splines. They are useful whenever a monotone relationship between a feature and
the target should be representable.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from ...core.params import UNSET
from .base_spline import BaseSplineTransformer


class ISplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using an I-spline (integrated spline) basis.

    Each basis function is the integral of an M-spline over the fitted range,
    which yields a monotonically increasing, non-negative function normalized to
    ``[0, 1]``. Knot placement follows the shared priority: target-aware
    (``selector``) > explicit (``knot_locations``) > automatic
    (``output_dim`` with ``strategy``).

    See :class:`~pretab.transformers.splines.base_spline.BaseSplineTransformer`
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

    def __init__(
        self,
        output_dim=UNSET,
        degree: int = 3,
        strategy=UNSET,
        include_bias: bool = False,
        knot_locations: np.ndarray | None = None,
        selector=UNSET,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        task: Literal["regression", "classification"] | None = None,
        knot_strategy=UNSET,
        knot_selector=UNSET,
    ):
        super().__init__(
            output_dim=output_dim,
            degree=degree,
            strategy=strategy,
            include_bias=include_bias,
            knot_locations=knot_locations,
            selector=selector,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            task=task,
            knot_strategy=knot_strategy,
            knot_selector=knot_selector,
        )

    def _feature_suffix(self) -> str:
        return "is"

    def _ispline_basis(self, x: np.ndarray, knots: np.ndarray, basis_idx: int) -> np.ndarray:
        """
        Compute a single I-spline basis function.

        The M-spline is evaluated on a fine grid, integrated with the trapezoidal
        rule to build a cumulative integral, then linearly interpolated at the
        requested points and normalized by the full-range integral.
        """
        x_min_knot = knots[0]
        x_max_knot = knots[-1]
        n_coef = len(knots) - self.degree - 1
        if basis_idx >= n_coef:
            return np.zeros(len(x))

        grid = np.linspace(x_min_knot, x_max_knot, 200)
        coef = np.zeros(n_coef)
        coef[basis_idx] = 1.0
        spline = BSpline(knots, coef, self.degree, extrapolate=False)
        m_values = np.nan_to_num(spline(grid), nan=0.0)

        knot_span = knots[basis_idx + self.degree + 1] - knots[basis_idx]
        if knot_span > 1e-10:
            m_values = m_values * (self.degree + 1) / knot_span

        cumulative = np.zeros(len(grid))
        for i in range(1, len(grid)):
            dx = grid[i] - grid[i - 1]
            cumulative[i] = cumulative[i - 1] + 0.5 * (m_values[i - 1] + m_values[i]) * dx

        ispline_values = np.interp(x, grid, cumulative, left=0.0, right=cumulative[-1])

        max_integral = cumulative[-1]
        if max_integral > 1e-10:
            ispline_values = ispline_values / max_integral
        return ispline_values

    def _design_matrix(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        n_coef = len(knots) - self.degree - 1
        design = np.zeros((len(x), n_coef))
        for i in range(n_coef):
            design[:, i] = self._ispline_basis(x, knots, i)
        return np.nan_to_num(design, nan=0.0)
