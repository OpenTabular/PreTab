import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist

class ThinPlateSplineTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn-style transformer implementing 1D thin plate regression splines (TPRS),
    including both the radial basis and the low-rank polynomial null space.
    Penalty matrix is defined for the radial basis part only, matching mgcv behavior.
    """

    def __init__(self, n_knots=10):
        self.n_knots = n_knots

    def _tps_basis(self, r):
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.where(r == 0, 0, np.log(r))
            basis = r**2 * log_r
            basis[r == 0] = 0
        return basis

    def fit(self, X, y=None):
        x = np.asarray(X).reshape(-1, 1)
        self.n_samples_ = x.shape[0]

        # Choose knots evenly across the domain
        self.knots_ = np.linspace(x.min(), x.max(), self.n_knots).reshape(-1, 1)

        # Design matrix: radial basis functions
        r = cdist(x, self.knots_, metric="euclidean")
        self.radial_ = self._tps_basis(r)

        # Null space: intercept and linear term
        self.null_space_ = np.hstack([np.ones_like(x), x])

        # Full design matrix
        self.X_design_ = np.hstack([self.radial_, self.null_space_])

        # Penalty matrix: only for radial part
        knot_distances = cdist(self.knots_, self.knots_, metric="euclidean")
        self.penalty_ = self._tps_basis(knot_distances)

        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1, 1)

        # Radial part
        r = cdist(x, self.knots_, metric="euclidean")
        radial = self._tps_basis(r)

        # Null space part
        null_space = np.hstack([np.ones_like(x), x])

        return np.hstack([radial, null_space])

    def get_penalty_matrix(self):
        return self.penalty_
