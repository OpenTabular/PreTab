import numpy as np

from ...core.params import UNSET
from ._base import BaseCenterExpansion


class RBFExpansionTransformer(BaseCenterExpansion):
    r"""
    Radial Basis Function (RBF) feature expansion for numerical tabular data.

    This transformer expands each feature into a set of RBF (Gaussian) basis functions
    centered at fixed points. The centers can be determined either by a target-aware
    selector (based on supervised splits) or based on quantiles or uniform spacing.

    Parameters
    ----------
    output_dim : int, default=10
        Number of RBF centers (output columns) per feature.

    gamma : float, default=1.0
        Width parameter of the RBF kernel. Larger values make the kernel narrower.

    target_aware : bool, default=True
        Whether to place centers with a target-aware selector (requires `y`).

    task : {"regression", "classification"}, default="regression"
        Task type for the target-aware selector used to place centers.

    placement_strategy : {"cart", "lightgbm", "uniform", "quantile"}, default="cart"
        Selector when `target_aware=True` (`"cart"` or `"lightgbm"`); spacing when
        `target_aware=False` (`"uniform"` or `"quantile"`).

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

    def __init__(
        self,
        output_dim=UNSET,
        gamma: float = 1.0,
        target_aware: bool = True,
        task: str = "regression",
        placement_strategy: str = "cart",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        super().__init__(
            output_dim=output_dim,
            target_aware=target_aware,
            task=task,
            placement_strategy=placement_strategy,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
        )
        self.gamma = gamma

    def _expand_column(self, x_col, centers):
        return np.exp(-self.gamma * (x_col - centers[np.newaxis, :]) ** 2)
