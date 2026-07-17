"""
M-Spline Transformer.

M-splines are non-negative spline basis functions derived from B-splines and
normalized so that each basis function integrates to one. They are the building
block for I-splines.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from .base_spline import BaseSplineTransformer


class MSplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using an M-spline basis expansion.

    M-splines are non-negative basis functions obtained by scaling each B-spline
    by ``(degree + 1) / knot_span`` so that it integrates to one. Knot placement
    follows the same priority as the other spline transformers: target-aware
    (``knot_selector``) > explicit (``knot_locations``) > automatic
    (``n_basis_functions`` with ``knot_strategy``).

    See :class:`~pretab.transformers.splines.base_spline.BaseSplineTransformer`
    for the full parameter description. ``include_bias`` defaults to False here.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import MSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> MSplineTransformer(n_basis_functions=8).fit_transform(X).shape
    (50, 8)
    """

    def __init__(
        self,
        n_basis_functions: int = 5,
        degree: int = 3,
        knot_strategy: str = "quantile",
        include_bias: bool = False,
        knot_locations: np.ndarray | None = None,
        knot_selector=None,
        adaptive: bool = False,
        min_basis_functions: int | None = None,
        max_basis_functions: int | None = None,
        n_knots: int | None = None,
        task: Literal["regression", "classification"] | None = None,
    ):
        super().__init__(
            n_basis_functions=n_basis_functions,
            degree=degree,
            knot_strategy=knot_strategy,
            include_bias=include_bias,
            knot_locations=knot_locations,
            knot_selector=knot_selector,
            adaptive=adaptive,
            min_basis_functions=min_basis_functions,
            max_basis_functions=max_basis_functions,
            n_knots=n_knots,
            task=task,
        )

    def _feature_suffix(self) -> str:
        return "ms"

    def _mspline_basis(self, x: np.ndarray, knots: np.ndarray, basis_idx: int) -> np.ndarray:
        """Compute a single M-spline basis function."""
        n_coef = len(knots) - self.degree - 1
        coef = np.zeros(n_coef)
        coef[basis_idx] = 1.0
        spline = BSpline(knots, coef, self.degree, extrapolate=False)
        values = np.nan_to_num(spline(x), nan=0.0)

        knot_span = knots[basis_idx + self.degree + 1] - knots[basis_idx]
        if knot_span > 1e-6:
            values = values * (self.degree + 1) / knot_span
            values = np.clip(values, 0.0, 1e6)
        else:
            values = np.zeros(len(x))
        return np.maximum(values, 0.0)

    def _design_matrix(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        n_coef = len(knots) - self.degree - 1
        design = np.zeros((len(x), n_coef))
        for i in range(n_coef):
            design[:, i] = self._mspline_basis(x, knots, i)
        design = np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(design, -1e3, 1e3)
