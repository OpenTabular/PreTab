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
    output_dim : int, default=10
        Number of sigmoid centers (output columns) per feature.

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

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    For a feature :math:`x` and center :math:`c`, the transformation is

    .. math::

        \sigma\!\left(\frac{x - c}{s}\right) = \frac{1}{1 + \exp\!\left(-\frac{x - c}{s}\right)},

    where :math:`s` is ``scale``. This produces ``output_dim`` new features per
    original feature (on the default, non-decision-tree path; a decision tree may
    place a data-driven number).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import SigmoidExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = SigmoidExpansionTransformer(output_dim=3, use_decision_tree=False)
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
        use_target=UNSET,
        task: str = "regression",
        strategy="uniform",
        use_decision_tree=UNSET,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
    ):
        super().__init__(
            output_dim=output_dim,
            use_target=use_target,
            task=task,
            strategy=strategy,
            use_decision_tree=use_decision_tree,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
        )
        self.scale = scale

    def _expand_column(self, x_col, centers):
        return 1 / (1 + np.exp(-(x_col - centers[np.newaxis, :]) / self.scale))
