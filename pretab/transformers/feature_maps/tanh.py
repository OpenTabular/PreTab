import numpy as np

from ...core.parameters import UNSET
from .base import BaseCenterExpansion


class TanhExpansionTransformer(BaseCenterExpansion):
    r"""
    Applies hyperbolic tangent (tanh) basis expansion to input features using specified or learned center locations.

    This transformer expands each input feature into multiple tanh-activated features, useful for capturing
    nonlinear and saturating patterns in the data.

    Parameters
    ----------
    output_dim : int, default=6
        Number of tanh centers (output columns) per feature.

    scale : float, default=1.0
        Controls the sharpness of the tanh transitions. Smaller values make the activation sharper.

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
        A list of center values for each input feature used in the tanh expansion.

    total_output_dim_ : int
        Total number of output columns across all features (fitted).

    Notes
    -----
    Each original feature :math:`x` is transformed into ``output_dim`` features of
    the form

    .. math::

        \tanh\!\left(\frac{x - c}{s}\right),

    where :math:`c` is a center value and :math:`s` (``scale``) controls the
    spread of the activation on the non-target-aware path; the target-aware
    default may place a data-driven number.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import TanhExpansionTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> transformer = TanhExpansionTransformer(output_dim=3, target_aware=False, placement_strategy="uniform")
    >>> transformer.fit(X)
    TanhExpansionTransformer(...)
    >>> transformer.transform(X).shape
    (3, 3)
    """

    _feature_suffix_value = "tanh"
    _representation_family = "tanh"

    def __init__(
        self,
        output_dim=UNSET,
        scale: float = 1.0,
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
        self.scale = scale

    def _expand_column(self, x_col, centers):
        return np.tanh((x_col - centers[np.newaxis, :]) / self.scale)
