import numpy as np

from ...core.parameters import UNSET
from .base import BaseCenterExpansion


class ReLUExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies ReLU basis expansion to input features using fixed or data-driven center placement.

    This transformer expands each feature using a set of ReLU activation functions centered at fixed positions,
    which can be either uniformly/quantile spaced or determined by a target-aware selector based on the target.

    Parameters
    ----------
    output_dim : int, default=6
        Number of ReLU centers (output columns) per feature.

    target_aware : bool, default=False
        Whether to place centers with a target-aware selector (requires `y`).

    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}, optional
        Selector when `target_aware=True` (`"cart"` or `"lightgbm"`); spacing when
        `target_aware=False` (`"uniform"` or `"quantile"`). If left unset, resolves
        to `"cart"` on the target-aware path and `"quantile"` otherwise.

    task : {"regression", "classification"}, default="regression"
        Task type for the target-aware selector used to place centers.

    adaptive : bool, default=False
        If True (with `target_aware=True`), the per-feature number of centers may
        vary within `[min_output_dim, max_output_dim]` instead of being fixed to
        `output_dim`. Has no effect on the `quantile` / `uniform` paths.

    min_output_dim : int or None, default=None
        Lower bound on the per-feature number of centers in adaptive mode.

    max_output_dim : int or None, default=None
        Upper bound on the per-feature number of centers in adaptive mode.

    random_state : int or None, default=None
        Random state forwarded to the target-aware selector for reproducibility.

    Attributes
    ----------
    centers_ : list of ndarray
        A list of arrays containing the center locations for each input feature.

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    For a feature :math:`x` and centers :math:`c_i`, each output column applies a
    rectified linear unit

    .. math::

        \phi_i(x) = \max(0,\; x - c_i),

    producing ``output_dim`` new features per original feature on the
    non-target-aware path; the target-aware default may place a data-driven number.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ReLUExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = ReLUExpansionTransformer(output_dim=3, target_aware=False, placement_strategy="uniform")
    >>> transformer.fit(X)
    ReLUExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "relu"
    _representation_family = "relu"

    def __init__(
        self,
        output_dim=UNSET,
        target_aware: bool = False,
        placement_strategy=UNSET,
        task: str = "regression",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        super().__init__(
            output_dim=output_dim,
            target_aware=target_aware,
            placement_strategy=placement_strategy,
            task=task,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
        )

    def _expand_column(self, x_col, centers):
        return np.maximum(0, x_col - centers[np.newaxis, :])
