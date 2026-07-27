import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import PretabDataError


class ContinuousOrdinalTransformer(TransformerMixin, BaseEstimator):
    """Encode categorical features as continuous integer values.

    Each unique category within a feature is assigned an integer based on its
    sorted order. Unknown or missing categories are mapped to ``0``. This is
    useful for models that can only handle numerical input.

    Attributes
    ----------
    mapping_ : list of dict
        One dictionary per feature mapping original categories to integers.

    Notes
    -----
    Categories are numbered starting at ``1`` in sorted order; the value ``0`` is
    reserved for categories not seen during ``fit`` (and for ``None``).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ContinuousOrdinalTransformer
    >>> X = np.array([["a", "x"], ["b", "y"], ["a", "x"]], dtype=object)
    >>> transformer = ContinuousOrdinalTransformer()
    >>> transformer.fit_transform(X).shape
    (3, 2)
    """

    def fit(self, X, y=None):
        """Learn the mapping from categories to integers for each feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Coerce to a 2D object array first. Iterating ``X.T`` directly yields
        # the *column labels* of the transpose for a DataFrame -- i.e. the
        # original row index -- which silently produced one mapping per row.
        X = self._as_2d(X)
        # Fit should determine the mapping from original categories to sequential integers starting from 0
        self.mapping_ = [
            {category: i + 1 for i, category in enumerate(np.unique(X[:, j]))}
            for j in range(X.shape[1])
        ]
        for mapping in self.mapping_:
            mapping[None] = 0  # Assign 0 to unknown values
        self.n_features_in_ = len(self.mapping_)
        return self

    @staticmethod
    def _as_2d(X) -> np.ndarray:
        """Return ``X`` as a 2D object ndarray, accepting frames, arrays and lists."""
        array = np.asarray(X, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def transform(self, X):
        """Apply the learned category-to-integer mapping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data with integer values.
        """
        check_is_fitted(self, "mapping_")
        # As in ``fit``: iterating a DataFrame directly yields column *names*,
        # not rows. Normalize first, then index by position.
        X = self._as_2d(X)
        if X.shape[1] != len(self.mapping_):
            raise PretabDataError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"is expecting {len(self.mapping_)} features as input."
            )
        # Allocating up front keeps the output 2D even for zero rows, where the
        # comprehension used to collapse to shape ``(0,)``.
        X_transformed = np.zeros(X.shape, dtype=int)
        for col, mapping in enumerate(self.mapping_):
            X_transformed[:, col] = [mapping.get(value, 0) for value in X[:, col]]
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names (unchanged from the input).

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The names of the output features after transformation.
        """
        check_is_fitted(self, "mapping_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing/unknown categories are handled (mapped to 0)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
