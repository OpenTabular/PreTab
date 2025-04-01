import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

class ExactThinPlateSplineTransformer(BaseEstimator, TransformerMixin):
    """
    Exact 1D Thin Plate Regression Spline (TPRS) basis, matching mgcv's bs="tp".
    Uses full data as knots, projects out the null space, and reduces rank via eigendecomposition.
    """

    def __init__(self, n_basis=10):
        self.n_basis = n_basis  # number of basis functions to keep (k in mgcv)

    def _tps_kernel(self, r):
        with np.errstate(divide='ignore', invalid='ignore'):
            log_r = np.where(r == 0, 0, np.log(r))
            K = r**2 * log_r
            K[r == 0] = 0
        return K

    def fit(self, X, y=None):
        x = np.asarray(X).reshape(-1, 1)
        self.x_ = x
        n = x.shape[0]

        # Null space (intercept and linear term)
        Z = np.hstack([np.ones_like(x), x])
        self.Z_ = Z

        # Radial basis matrix (full kernel)
        r = cdist(x, x, metric="euclidean")
        K = self._tps_kernel(r)

        # Projection matrix to remove null space: P = I - Z(Z^T Z)^-1 Z^T
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        P = np.eye(n) - Z @ ZTZ_inv @ Z.T
        KP = P @ K @ P  # penalized kernel, null space removed

        # Eigendecomposition
        eigvals, eigvecs = eigh(KP)
        idx = np.argsort(eigvals)[::-1]  # sort descending
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Keep top `n_basis` components
        self.eigvals_ = eigvals[:self.n_basis]
        self.basis_ = eigvecs[:, :self.n_basis] * np.sqrt(n)  # scale for stability

        # Store full penalty matrix in reduced basis
        self.penalty_ = np.diag(self.eigvals_)

        return self

    def transform(self, X):
        x_new = np.asarray(X).reshape(-1, 1)
        r_new = cdist(x_new, self.x_, metric="euclidean")
        K_new = self._tps_kernel(r_new)

        # Remove null space
        Z = self.Z_
        ZTZ_inv = np.linalg.pinv(Z.T @ Z)
        P_new = np.eye(Z.shape[0]) - Z @ ZTZ_inv @ Z.T
        K_new_proj = K_new @ P_new

        # Project onto learned basis
        return K_new_proj @ self.basis_

    def get_penalty_matrix(self):
        return self.penalty_
