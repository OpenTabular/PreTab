import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ThinPlateSplineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_knots=10):
        self.n_knots = n_knots

    def _radial_basis(self, r):
        return (r ** 2) * np.log(r + 1e-12)

    def fit(self, X, y=None):
        x = np.asarray(X).reshape(-1, 1)
        self.centers_ = np.linspace(x.min(), x.max(), self.n_knots).reshape(-1, 1)
        self.r_ = np.linalg.norm(x - self.centers_.T, axis=2)
        self.X_design_ = self._radial_basis(self.r_)
        return self

    def transform(self, X):
        x = np.asarray(X).reshape(-1, 1)
        r = np.linalg.norm(x - self.centers_.T, axis=2)
        return self._radial_basis(r)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_penalty_matrix(self):
        # Identity matrix: TPS penalizes weights directly
        n = self.X_design_.shape[1]
        return np.eye(n)
