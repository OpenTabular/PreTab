import numpy as np

from .._center_expansion import BaseCenterExpansion


class ReLUExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies ReLU basis expansion to input features using fixed or data-driven center placement.

    This transformer expands each feature using a set of ReLU activation functions centered at fixed positions,
    which can be either uniformly/quantile spaced or determined by a decision tree based on the target.

    Parameters
    ----------
    n_centers : int, default=10
        Number of ReLU centers per feature.

    use_decision_tree : bool, default=True
        If True, uses a decision tree to determine center locations based on the input `X` and target `y`.

    task : {"regression", "classification"}, default="regression"
        Task type used for center selection when `use_decision_tree=True`.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy used to determine center locations when `use_decision_tree=False`.

    Attributes
    ----------
    centers_ : list of ndarray
        A list of arrays containing the center locations for each input feature.

    Notes
    -----
    For a feature :math:`x` and centers :math:`c_i`, each output column applies a
    rectified linear unit

    .. math::

        \phi_i(x) = \max(0,\; x - c_i),

    producing ``n_centers`` new features per original feature.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ReLUExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = ReLUExpansionTransformer(n_centers=3, use_decision_tree=False)
    >>> transformer.fit(X)
    ReLUExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "relu"

    def __init__(
        self,
        n_centers=10,
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

    def _expand_column(self, x_col, centers):
        return np.maximum(0, x_col - centers[np.newaxis, :])
