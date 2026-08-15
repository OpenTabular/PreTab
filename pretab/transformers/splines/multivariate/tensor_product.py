import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ....core.knots import bspline_basis
from ....core.parameters import UNSET
from ....exceptions import InvalidParamError
from ..mixins import SplineBasisMixin


class TensorProductSplineTransformer(SplineBasisMixin, TransformerMixin, BaseEstimator):
    r"""
    Tensor Product Spline Transformer for multivariate smooth basis expansion.

    This transformer generates tensor-product B-spline basis functions for
    multivariate input, allowing for smooth modeling of complex feature
    interactions. It supports regularization via difference penalties in each
    marginal dimension, suitable for additive models and structured regression.

    ``output_dim`` here is **per marginal dimension**: each of the :math:`d` input
    features gets a marginal B-spline basis of :math:`m := \mathtt{output\_dim}`
    columns (using :math:`K = \mathtt{output\_dim} - p - 1` interior knots with a
    :math:`p + 1` boundary pad, exactly like :class:`PSplineTransformer`), and the
    full tensor-product design has

    .. math::

        \text{total columns} = \mathtt{output\_dim}^{\,d}

    columns (``(output_dim + 1) ** d`` when ``include_bias=True``). This is the one
    family where the total output width is *not* ``n_features * output_dim``.

    Parameters
    ----------
    output_dim : int, default=4
        Number of non-bias output columns **per marginal dimension** (:math:`m`).
        Must be at least ``degree + 1``. The interior knots per dimension is
        ``output_dim - degree - 1``.

    degree : int, default=3
        Degree of the B-spline basis functions.

    diff_order : int, default=2
        Order of the finite difference penalty used to enforce smoothness along each input dimension.

    include_bias : bool, default=False
        If True, prepend a constant column to each marginal basis before the
        tensor product. The bias term is left unpenalized.

    placement_strategy : {"uniform", "quantile"}, default="uniform"
        Interior-knot placement rule for each marginal dimension. ``"uniform"``
        spaces knots evenly; ``"quantile"`` uses data quantiles.

        .. note::
           The tensor-product spline is a penalized (difference-penalty) spline
           per marginal, exactly like :class:`~pretab.transformers.splines.p_spline.PSplineTransformer`,
           so it assumes **equally-spaced** knots and is *unsupervised-only*:
           target-aware placement does not apply and only ``"uniform"`` /
           ``"quantile"`` spacing is accepted.

    adaptive : bool, default=False
        Retained for API parity but a no-op for this unsupervised-only family: the
        per-marginal width is always fixed to ``output_dim``.

    min_output_dim : int or None, default=None
        Unused for this unsupervised-only family (kept for API parity).

    max_output_dim : int or None, default=None
        Unused for this unsupervised-only family (kept for API parity).

    Attributes
    ----------
    dim_ : int
        Number of input features (marginal dimensions).

    knots_ : list of ndarray
        Full padded knot sequence for each marginal dimension.

    n_knots_ : list of int
        Number of interior knots for each marginal dimension
        (``output_dim - degree - 1`` on the default, non-selector path).

    bases_ : list of ndarray
        Marginal B-spline basis matrices, each of shape
        ``(n_samples, output_dim (+1 if include_bias))``.

    penalties_ : list of ndarray
        Univariate difference penalties for each marginal basis.

    X_design_ : ndarray of shape (n_samples, output_dim ** dim\_)
        Full tensor-product design matrix computed during ``fit``.

    n_features_in_ : int
        Number of input features seen during ``fit``.

    total_output_dim_ : int
        Total number of output columns (fitted); equals
        ``output_dim ** dim_`` (``(output_dim + 1) ** dim_`` with a bias).

    Notes
    -----
    - Uses einsum-based reshaping to build the tensor product basis efficiently.
    - Supports arbitrary number of input dimensions.
    - Commonly used in structured additive models and GAMs where smooth surfaces are
      desired [1]_, [2]_.

    References
    ----------
    .. [1] Eilers, P.H.C. and Marx, B.D. (2003). "Multivariate calibration with
       temperature interaction using two-dimensional penalized signal regression".
    .. [2] Wood, S.N. (2017). "Generalized Additive Models: An Introduction with R".

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> from pretab.transformers import TensorProductSplineTransformer
    >>> X = rng.random((20, 2))
    >>> transformer = TensorProductSplineTransformer(output_dim=6)
    >>> Xt = transformer.fit_transform(X)
    >>> Xt.shape
    (20, 36)
    >>> transformer.total_output_dim_
    36
    """

    _representation_family = "tensorspline"
    _representation_scope = "multivariate"
    _representation_local_support = True

    def __init__(
        self,
        output_dim=UNSET,
        degree=3,
        diff_order=2,
        include_bias=False,
        placement_strategy: str = "uniform",
        adaptive: bool = False,
        min_output_dim=UNSET,
        max_output_dim=UNSET,
    ):
        self.output_dim = output_dim
        self.degree = degree
        self.diff_order = diff_order
        self.include_bias = include_bias
        self.placement_strategy = placement_strategy
        self.adaptive = adaptive
        self.min_output_dim = min_output_dim
        self.max_output_dim = max_output_dim

    def _pad_knots(self, inner):
        return np.concatenate((np.repeat(inner[0], self.degree), inner, np.repeat(inner[-1], self.degree)))

    def _basis_matrix(self, x, knots):
        n_basis = len(knots) - self.degree - 1
        B = np.zeros((len(x), n_basis))
        for i in range(n_basis):
            B[:, i] = bspline_basis(x, knots, self.degree, i)
        if self.include_bias:
            B = np.hstack([np.ones((len(x), 1)), B])
        return B

    def _difference_penalty(self, n_basis):
        D = np.eye(n_basis)
        for _ in range(self.diff_order):
            D = np.diff(D, n=1, axis=0)
        return D.T @ D

    def fit(self, X, y=None):
        X = self._validate_allow_nan(X, reset=True)
        output_dim = self._resolve_param("output_dim", default=4)
        if self.placement_strategy not in ("uniform", "quantile"):
            raise InvalidParamError(
                f"Invalid placement_strategy. Choose 'uniform' or 'quantile'. Got {self.placement_strategy!r}."
            )
        strategy = self.placement_strategy

        if output_dim < self.degree + 1:
            raise InvalidParamError(
                f"output_dim must be >= degree + 1 = {self.degree + 1} for the tensor-product "
                f"spline basis, got {output_dim}"
            )

        self.dim_ = X.shape[1]
        self.knots_ = []
        self.bases_ = []
        self.penalties_ = []
        self.n_knots_ = []

        min_interior, max_interior = self._adaptive_interior_bounds(
            output_dim, None, floor=self.degree + 1, offset=self.degree + 1
        )

        for d in range(self.dim_):
            knots = self._place_bspline_knots(
                X[:, d], y, output_dim, self.degree, strategy, None, None, min_interior, max_interior
            )
            basis = self._basis_matrix(X[:, d], knots)
            penalty = self._difference_penalty(len(knots) - self.degree - 1)
            if self.include_bias:
                penalty = np.pad(penalty, ((1, 0), (1, 0)))
            self.knots_.append(knots)
            self.bases_.append(basis)
            self.penalties_.append(penalty)
            self.n_knots_.append(max(0, len(knots) - 2 * (self.degree + 1)))

        n_samples = X.shape[0]
        design = self.bases_[0]
        for b in self.bases_[1:]:
            design = np.einsum("ni,nj->nij", design, b).reshape(n_samples, -1)
        self.X_design_ = design

        return self

    def transform(self, X):
        check_is_fitted(self, "X_design_")
        X = self._validate_allow_nan(X, reset=False)

        bases = []
        for d in range(self.dim_):
            basis = self._basis_matrix(X[:, d], self.knots_[d])
            bases.append(basis)

        n_samples = X.shape[0]
        design = bases[0]
        for b in bases[1:]:
            design = np.einsum("ni,nj->nij", design, b).reshape(n_samples, -1)
        return design

    def get_feature_names_out(self, input_features=None):
        """Return names for the interaction basis as ``tp_{feat0 i}_{feat1 j}...``."""
        check_is_fitted(self, "bases_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        sizes = [b.shape[1] for b in self.bases_]
        names = []
        for multi_index in np.ndindex(*sizes):
            parts = [f"{input_features[d]}{multi_index[d]}" for d in range(self.dim_)]
            names.append("tp_" + "_".join(parts))
        return np.asarray(names, dtype=object)

    def get_penalty_matrices(self):
        """Return the Kronecker-structured penalty matrices.

        Returns
        -------
        penalties : list of ndarray
            One full penalty matrix per marginal direction, each formed as a
            Kronecker product of a marginal difference penalty with identity
            matrices for the remaining dimensions.
        """
        kron_penalties = []
        for i, Si in enumerate(self.penalties_):
            mats = [np.eye(b.shape[1]) for j, b in enumerate(self.bases_) if j != i]
            P = Si
            for M in mats:
                P = np.kron(P, M)
            kron_penalties.append(P)
        return kron_penalties

    def get_penalty_matrix(self, feature_index=0):
        """Return the marginal difference penalty for one input dimension.

        Provided for signature parity with the other spline transformers; use
        :meth:`get_penalty_matrices` for the full Kronecker-structured penalties.

        Parameters
        ----------
        feature_index : int, default=0
            Index of the marginal dimension whose difference penalty is returned.

        Returns
        -------
        P : ndarray
            The marginal difference penalty matrix for ``feature_index``.
        """
        check_is_fitted(self, "penalties_")
        return self.penalties_[feature_index]
