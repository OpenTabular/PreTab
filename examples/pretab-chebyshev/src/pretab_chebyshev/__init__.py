"""A minimal, self-contained PreTab extension package.

Demonstrates the full third-party extension workflow: subclass
:class:`pretab.BaseRepresentation`, expose the class through the
``pretab.representations`` entry-point group (see ``pyproject.toml``), and let it
be discovered, validated, and used exactly like a built-in representation.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.validation import check_is_fitted

from pretab import BaseRepresentation

__all__ = ["ChebyshevRepresentation"]


class ChebyshevRepresentation(BaseRepresentation):
    """Expand each numerical feature into a Chebyshev polynomial basis.

    Every input column is rescaled to ``[-1, 1]`` using the training-data range,
    then expanded into ``T_1 ... T_degree`` Chebyshev polynomials (the constant
    ``T_0`` term is dropped to avoid a redundant bias column). This yields
    ``degree`` output columns per input feature.

    Parameters
    ----------
    degree : int, default=5
        Number of Chebyshev polynomials produced per feature.
    """

    representation_name = "chebyshev"
    feature_kind = "numerical"
    scope = "univariate"
    supervision = "unsupervised"

    def __init__(self, degree=5):
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(self._validate(X, reset=True), dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def _rescale(self, X):
        span = self.data_max_ - self.data_min_
        span = np.where(span == 0.0, 1.0, span)
        return np.clip(2.0 * (X - self.data_min_) / span - 1.0, -1.0, 1.0)

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        z = self._rescale(np.asarray(self._validate(X, reset=False), dtype=float))
        theta = np.arccos(z)
        blocks = [
            np.column_stack([np.cos(k * theta[:, j]) for k in range(1, self.degree + 1)]) for j in range(z.shape[1])
        ]
        return np.hstack(blocks)

    def _output_sizes(self):
        return [self.degree] * self.n_features_in_
