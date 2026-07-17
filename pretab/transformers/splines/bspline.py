"""
B-Spline Transformer.

B-splines are locally supported polynomial basis functions defined by a knot
vector. They form the foundation for the M-spline and I-spline transformers.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from .base_spline import BaseSplineTransformer


class BSplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using a B-spline basis expansion.

    Supports three knot placement strategies with a fixed priority:
    target-aware (``knot_selector``) > explicit (``knot_locations``) >
    automatic (``n_basis_functions`` with ``knot_strategy``). Multi-column input
    is expanded column by column and the results are stacked horizontally.

    See :class:`~pretab.transformers.splines.base_spline.BaseSplineTransformer`
    for the full parameter description. ``include_bias`` defaults to True here.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import BSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> BSplineTransformer(n_basis_functions=8).fit_transform(X).shape
    (50, 9)
    """

    def __init__(
        self,
        n_basis_functions: int = 5,
        degree: int = 3,
        knot_strategy: str = "quantile",
        include_bias: bool = True,
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
        return "bs"

    def _design_matrix(self, x: np.ndarray, knots: np.ndarray) -> np.ndarray:
        n_basis = len(knots) - self.degree - 1
        design = np.zeros((len(x), n_basis))
        for i in range(n_basis):
            coef = np.zeros(n_basis)
            coef[i] = 1.0
            spline = BSpline(knots, coef, self.degree, extrapolate=False)
            design[:, i] = spline(x)
        return np.nan_to_num(design, nan=0.0)
