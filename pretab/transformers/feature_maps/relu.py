import numpy as np

from ...core.params import UNSET
from .._center_expansion import BaseCenterExpansion


class ReLUExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies ReLU basis expansion to input features using fixed or data-driven center placement.

    This transformer expands each feature using a set of ReLU activation functions centered at fixed positions,
    which can be either uniformly/quantile spaced or determined by a decision tree based on the target.

    Parameters
    ----------
    output_dim : int, default=10
        Number of ReLU centers (output columns) per feature.

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

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    For a feature :math:`x` and centers :math:`c_i`, each output column applies a
    rectified linear unit

    .. math::

        \phi_i(x) = \max(0,\; x - c_i),

    producing ``output_dim`` new features per original feature (on the default,
    non-decision-tree path; a decision tree may place a data-driven number).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ReLUExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = ReLUExpansionTransformer(output_dim=3, use_decision_tree=False)
    >>> transformer.fit(X)
    ReLUExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "relu"

    def __init__(
        self,
        output_dim=UNSET,
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

    def _expand_column(self, x_col, centers):
        return np.maximum(0, x_col - centers[np.newaxis, :])
