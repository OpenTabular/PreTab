"""
B-Spline Transformer.

B-splines are locally supported polynomial basis functions defined by a knot
vector. They form the foundation for the M-spline and I-spline transformers.
"""

from typing import Literal

import numpy as np
from scipy.interpolate import BSpline

from ...core.parameters import UNSET
from ...core.policy import RepresentationPolicy
from .base import BaseSplineTransformer


class BSplineTransformer(BaseSplineTransformer):
    """
    Transform numerical features using a B-spline basis expansion.

    Supports three knot placement strategies with a fixed priority:
    explicit (``knot_locations``) > target-aware (``placement_strategy``) >
    automatic (``output_dim`` with ``placement_strategy``). Multi-column input
    is expanded column by column and the results are stacked horizontally.

    See :class:`~pretab.expansion.spline.base.BaseSplineTransformer`
    for the full parameter description. ``include_bias`` defaults to False: a
    B-spline basis over a clamped knot vector is a partition of unity (every row
    sums to 1), so prepending a bias column makes it an exact linear combination
    of the others and the design rank-deficient. Pass ``include_bias=True`` to
    add the column anyway if a downstream model requires it.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import BSplineTransformer
    >>> X = np.linspace(0, 1, 50).reshape(-1, 1)
    >>> BSplineTransformer(output_dim=8).fit_transform(X).shape
    (50, 8)
    """

    _representation_family = "bspline"

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
