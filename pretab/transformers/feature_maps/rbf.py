import numpy as np

from ...core.params import UNSET
from ._base import BaseCenterExpansion


class RBFExpansionTransformer(BaseCenterExpansion):
    r"""
    Radial Basis Function (RBF) feature expansion for numerical tabular data.

    This transformer expands each feature into a set of RBF (Gaussian) basis functions
    centered at fixed points. The centers can be determined either by a decision tree
    (based on supervised splits) or based on quantiles or uniform spacing.

    Parameters
    ----------
    output_dim : int, default=10
        Number of RBF centers (output columns) per feature.

    gamma : float, default=1.0
        Width parameter of the RBF kernel. Larger values make the kernel narrower.

    use_decision_tree : bool, default=True
        Whether to use a decision tree to select center locations based on `y`.

    task : {"regression", "classification"}, default="regression"
        Type of task for the decision tree used to find center locations.

    strategy : {"uniform", "quantile"}, default="uniform"
        Strategy for choosing centers when not using a decision tree.

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

    producing ``output_dim`` new features per original feature (on the default,
    non-decision-tree path; a decision tree may place a data-driven number).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import RBFExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = RBFExpansionTransformer(output_dim=3, gamma=0.5, use_decision_tree=False)
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
        use_target=UNSET,
        task: str = "regression",
        strategy="uniform",
        use_decision_tree=UNSET,
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
        random_state: int | None = None,
        selector: str = "cart",
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
            selector=selector,
        )
        self.gamma = gamma

    def _expand_column(self, x_col, centers):
        return np.exp(-self.gamma * (x_col - centers[np.newaxis, :]) ** 2)
