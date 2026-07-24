import numpy as np
from scipy.special import expit

from ...core.params import UNSET
from ._base import BaseCenterExpansion


class SigmoidExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies sigmoid basis expansion to input features using specified or data-driven center placement.

    Each feature is expanded using a set of sigmoid functions centered at various locations, creating
    a smooth, nonlinear transformation that is especially useful for capturing saturating or threshold-like behavior.

    Parameters
    ----------
    output_dim : int, default=10
        Number of sigmoid centers (output columns) per feature.

    scale : float, default=1.0
        Controls the sharpness of the sigmoid transition. Smaller values yield sharper transitions.

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
        A list containing the sigmoid center locations for each input feature.

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    For a feature :math:`x` and center :math:`c`, the transformation is

    .. math::

        \sigma\!\left(\frac{x - c}{s}\right) = \frac{1}{1 + \exp\!\left(-\frac{x - c}{s}\right)},

    where :math:`s` is ``scale``. This produces ``output_dim`` new features per
    original feature on the non-target-aware path; the target-aware default may
    place a data-driven number.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import SigmoidExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = SigmoidExpansionTransformer(output_dim=3, target_aware=False, placement_strategy="uniform")
    >>> transformer.fit(X)
    SigmoidExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "sigmoid"

    def __init__(
        self,
        output_dim=UNSET,
        scale: float = 1.0,
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
        self.scale = scale

    def _expand_column(self, x_col, centers):
        return expit((x_col - centers[np.newaxis, :]) / self.scale)
