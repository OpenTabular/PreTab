import numpy as np

from .._center_expansion import BaseCenterExpansion


class TanhExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies hyperbolic tangent (tanh) basis expansion to input features using specified or learned center locations.

    This transformer expands each input feature into multiple tanh-activated features, useful for capturing
    nonlinear and saturating patterns in the data.

    Parameters
    ----------
    n_centers : int, default=10
        Number of tanh centers per feature.

    scale : float, default=1.0
        Controls the sharpness of the tanh transitions. Smaller values make the activation sharper.

    use_decision_tree : bool, default=True
        If True, uses a decision tree to determine the tanh center locations based on the input `X` and target `y`.

    task : {"regression", "classification"}, default="regression"
        Type of prediction task. Required for decision tree-based center selection.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy to determine center placement when `use_decision_tree=False`.

    Attributes
    ----------
    centers_ : list of ndarray
        A list of center values for each input feature used in the tanh expansion.

    Notes
    -----
    Each original feature :math:`x` is transformed into ``n_centers`` features of
    the form

    .. math::

        \tanh\!\left(\frac{x - c}{s}\right),

    where :math:`c` is a center value and :math:`s` (``scale``) controls the
    spread of the activation.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import TanhExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = TanhExpansionTransformer(n_centers=3, use_decision_tree=False)
    >>> transformer.fit(X)
    TanhExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "tanh"

    def __init__(
        self,
        n_centers=10,
        scale: float = 1.0,
        use_decision_tree=True,
        task: str = "regression",
        strategy="uniform",
    ):
        super().__init__(
            n_centers=n_centers,
            use_decision_tree=use_decision_tree,
            task=task,
            strategy=strategy,
        )
        self.scale = scale

    def _expand_column(self, x_col, centers):
        return np.tanh((x_col - centers[np.newaxis, :]) / self.scale)
