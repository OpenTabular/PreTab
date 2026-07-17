import numpy as np

from ...core.params import UNSET
from .._center_expansion import BaseCenterExpansion


class SigmoidExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies sigmoid basis expansion to input features using specified or data-driven center placement.

    Each feature is expanded using a set of sigmoid functions centered at various locations, creating
    a smooth, nonlinear transformation that is especially useful for capturing saturating or threshold-like behavior.

    Parameters
    ----------
    n_centers : int, default=10
        Number of sigmoid centers per feature.

    scale : float, default=1.0
        Controls the sharpness of the sigmoid transition. Smaller values yield sharper transitions.

    use_decision_tree : bool, default=True
        If True, uses a decision tree to determine the center locations based on the input `X` and target `y`.

    task : {"regression", "classification"}, default="regression"
        Type of prediction task. Required for decision tree-based center selection.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy to determine center placement when `use_decision_tree=False`.

    Attributes
    ----------
    centers_ : list of ndarray
        A list containing the sigmoid center locations for each input feature.

    Notes
    -----
    For a feature :math:`x` and center :math:`c`, the transformation is

    .. math::

        \sigma\!\left(\frac{x - c}{s}\right) = \frac{1}{1 + \exp\!\left(-\frac{x - c}{s}\right)},

    where :math:`s` is ``scale``. This produces ``n_centers`` new features per
    original feature.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import SigmoidExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = SigmoidExpansionTransformer(n_centers=3, use_decision_tree=False)
    >>> transformer.fit(X)
    SigmoidExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "sigmoid"

    def __init__(
        self,
        n_basis=UNSET,
        scale: float = 1.0,
        use_target=UNSET,
        task: str = "regression",
        strategy="uniform",
        n_centers=UNSET,
        use_decision_tree=UNSET,
    ):
        super().__init__(
            n_basis=n_basis,
            use_target=use_target,
            task=task,
            strategy=strategy,
            n_centers=n_centers,
            use_decision_tree=use_decision_tree,
        )
        self.scale = scale

    def _expand_column(self, x_col, centers):
        return 1 / (1 + np.exp(-(x_col - centers[np.newaxis, :]) / self.scale))
