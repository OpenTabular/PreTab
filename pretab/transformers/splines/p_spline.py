import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


def bspline_basis(x, knots, degree, i):
    if degree == 0:
        return np.where((x >= knots[i]) & (x < knots[i + 1]), 1.0, 0.0)
    else:
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]

        term1 = (
            0.0
            if denom1 == 0
            else (x - knots[i]) / denom1 * bspline_basis(x, knots, degree - 1, i)
        )
        term2 = (
            0.0
            if denom2 == 0
            else (knots[i + degree + 1] - x)
            / denom2
            * bspline_basis(x, knots, degree - 1, i + 1)
        )

        return term1 + term2


class PSplineTransformer(BaseEstimator, TransformerMixin):
    """
    P-spline Transformer for smooth spline basis expansion with penalization.

    This transformer expands each input feature into a set of B-spline basis functions
    and stores a corresponding penalty matrix for regularization. It is useful in
    Generalized Additive Models (GAMs) where smoothness is enforced through penalties.

    Parameters
    ----------
    n_knots : int, default=20
        Number of interior knots to place uniformly across the range of each feature.

    degree : int, default=3
        Degree of the B-spline basis functions (e.g., 3 for cubic splines).

    diff_order : int, default=2
        The order of the difference penalty used to compute the smoothness penalty matrix.
        For example, 2 corresponds to a second-order difference penalty (encouraging smooth second derivatives).

    Attributes
    ----------
    knots_ : list of ndarray
        List of extended knot sequences (with added boundary knots) for each feature.

    penalty_ : list of ndarray
        List of penalty matrices (D^T D) for each feature, where D is the differencing matrix.

    n_basis_ : list of int
        Number of B-spline basis functions generated for each feature.

    n_features_in_ : int
        Number of input features seen during `fit`.

    Methods
    -------
    get_penalty_matrix(feature_index=0)
        Returns the penalty matrix associated with the specified feature. This matrix
        can be used for Tikhonov-style regularization to enforce smoothness of the fitted spline.

    Notes
    -----
    - Boundary knots are added automatically to ensure proper spline behavior near the edges.
    - Internally, this transformer uses recursive B-spline basis construction.
    - This implementation supports multi-dimensional inputs and stacks transformed features horizontally.
    """

    def __init__(self, n_knots=20, degree=3, diff_order=2):
        self.n_knots = n_knots
        self.degree = degree
        self.diff_order = diff_order

    def fit(self, X, y=None):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        self.knots_ = []
        self.penalty_ = []
        self.n_basis_ = []

        for i in range(X.shape[1]):
            x = X[:, i]
            xmin, xmax = x.min(), x.max()
            inner_knots = np.linspace(xmin, xmax, self.n_knots)
            knots = np.concatenate(
                (
                    np.repeat(inner_knots[0], self.degree),
                    inner_knots,
                    np.repeat(inner_knots[-1], self.degree),
                )
            )
            n_basis = len(knots) - self.degree - 1
            D = np.eye(n_basis)
            for _ in range(self.diff_order):
                D = np.diff(D, n=1, axis=0)
            self.knots_.append(knots)
            self.n_basis_.append(n_basis)
            self.penalty_.append(D.T @ D)

        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        original_dim = np.shape(X)[1] if np.ndim(X) == 2 else 1
        X = check_array(
            X, dtype=np.float64, ensure_2d=True, ensure_all_finite="allow-nan"
        )
        if X.shape[1] < original_dim:
            warnings.warn(
                "Some input features were dropped during check_array validation.",
                UserWarning,
            )

        all_basis = []
        for i in range(X.shape[1]):
            x = X[:, i]
            basis = np.zeros((len(x), self.n_basis_[i]))
            for j in range(self.n_basis_[i]):
                basis[:, j] = bspline_basis(x, self.knots_[i], self.degree, j)
            all_basis.append(basis)

        return np.hstack(all_basis)

    def get_penalty_matrix(self, feature_index=0):
        return self.penalty_[feature_index]
