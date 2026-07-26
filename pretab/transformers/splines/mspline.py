"""
M-Spline Transformer.

M-splines are non-negative spline basis functions derived from B-splines and
normalized so that each basis function integrates to one. They are the building
block for I-splines.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from ...core.params import UNSET
from .base_spline import BaseSplineTransformer


class MSplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using an M-spline basis expansion.

    M-splines are non-negative basis functions obtained by scaling each B-spline
    by ``(degree + 1) / knot_span`` so that it integrates to one. Knot placement
    follows the same priority as the other spline transformers: explicit
    (``knot_locations``) > target-aware (``placement_strategy``) > automatic
    (``output_dim`` with ``placement_strategy``).

    See :class:`~pretab.transformers.splines.base_spline.BaseSplineTransformer`
    for the full parameter description. ``include_bias`` defaults to False here.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import MSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> MSplineTransformer(output_dim=8).fit_transform(X).shape
    (50, 8)
    """

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
