import numpy as np

from ...core.parameters import UNSET
from .base import BaseCenterExpansion


class RBFExpansionTransformer(BaseCenterExpansion):
    r"""
    Radial Basis Function (RBF) feature expansion for numerical tabular data.

    This transformer expands each feature into a set of RBF (Gaussian) basis functions
    centered at fixed points. The centers can be determined either by a target-aware
    selector (based on supervised splits) or based on quantiles or uniform spacing.

    Parameters
    ----------
    output_dim : int, default=6
        Number of RBF centers (output columns) per feature.

    gamma : float, default=1.0
        Width parameter of the RBF kernel. Larger values make the kernel narrower.

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
        `output_dim`. Has no effect on the `quantile` / `uniform` paths. When both
        `min_output_dim` and `max_output_dim` are set, `output_dim` is ignored
        entirely.

    min_output_dim : int or None, default=None
        Lower bound on the per-feature number of centers in adaptive mode.

    max_output_dim : int or None, default=None
        Upper bound on the per-feature number of centers in adaptive mode.

    random_state : int or None, default=None
        Random state forwarded to the target-aware selector for reproducibility.

    Attributes
    ----------
    centers_ : list of ndarray
        List of arrays containing center locations for each feature.

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    For a feature :math:`x` and centers :math:`c_i`, each output column is a
    Gaussian radial basis function

    .. math::

        \phi_i(x) = \exp\left(-\gamma (x - c_i)^2\right),

    producing ``output_dim`` new features per original feature on the
    non-target-aware path; the target-aware default may place a data-driven number.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import RBFExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = RBFExpansionTransformer(output_dim=3, gamma=0.5, target_aware=False, placement_strategy="uniform")
    >>> transformer.fit(X)
    RBFExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "rbf"
    _representation_family = "rbf"
    _representation_local_support = True

    def __init__(
        self,
        output_dim=UNSET,
        gamma: float = 1.0,
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
        self.gamma = gamma

    def _expand_column(self, x_col, centers):
        return np.exp(-self.gamma * (x_col - centers[np.newaxis, :]) ** 2)
