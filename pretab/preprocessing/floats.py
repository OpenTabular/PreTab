import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names, _check_feature_names_in, check_is_fitted


class NoTransformer(TransformerMixin, BaseEstimator):
    """Pass-through transformer that returns the input unchanged.

    Retains compatibility with the scikit-learn pipeline API without modifying
    the data.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.
    feature_names_in_ : ndarray of str
        Names of the input columns seen during ``fit``. Defined only when ``X``
        was a DataFrame with string column names; ``transform`` then requires
        the same columns in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import NoTransformer
    >>> X = np.array([[1.0], [2.0], [3.0]])
    >>> NoTransformer().fit_transform(X).shape
    (3, 1)
    """

    def fit(self, X, y=None):
        """Fit the transformer (no operation is performed).

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
        _check_feature_names(self, X, reset=True)
        # np.shape reads ``.shape`` directly, so a sparse matrix (the one-hot
        # encoder's output) reports its real width instead of the 0-d object
        # array that np.asarray makes of it.
        shape = np.shape(X)
        self.n_features_in_ = shape[1] if len(shape) > 1 else 1
        return self

    def transform(self, X):
        """Return the input data unprocessed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X : array-like
            The same input data, unmodified.
        """
        check_is_fitted(self, "n_features_in_")
        _check_feature_names(self, X, reset=False)
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. ``None`` uses ``feature_names_in_`` when the
            transformer was fitted on a DataFrame, else ``x0, x1, ...``. Explicit
            names need one entry per input feature and, after a DataFrame fit,
            must equal ``feature_names_in_``.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.

        Raises
        ------
        ValueError
            If ``input_features`` does not have one entry per input feature, or
            differs from ``feature_names_in_``.
        """
        check_is_fitted(self, "n_features_in_")
        return np.asarray(_check_feature_names_in(self, input_features), dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values pass through unchanged."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.allow_nan = True
        return tags


class ToFloatTransformer(TransformerMixin, BaseEstimator):
    """Convert input data to floating-point type.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.
    feature_names_in_ : ndarray of str
        Names of the input columns seen during ``fit``. Defined only when ``X``
        was a DataFrame with string column names; ``transform`` then requires
        the same columns in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ToFloatTransformer
    >>> X = np.array([[1], [2], [3]])
    >>> ToFloatTransformer().fit_transform(X).dtype
    dtype('float64')
    """

    def fit(self, X, y=None):
        """Fit the transformer (records the feature count only).

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
        _check_feature_names(self, X, reset=True)
        # np.shape reads ``.shape`` directly, so a sparse matrix (the one-hot
        # encoder's output) reports its real width instead of the 0-d object
        # array that np.asarray makes of it.
        shape = np.shape(X)
        self.n_features_in_ = shape[1] if len(shape) > 1 else 1
        return self

    def transform(self, X):
        """Cast the input data to float.

        Parameters
        ----------
        X : ndarray
            The input data to convert.

        Returns
        -------
        X_float : ndarray
            The input data cast to ``float``.
        """
        check_is_fitted(self, "n_features_in_")
        _check_feature_names(self, X, reset=False)
        return X.astype(float)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. ``None`` uses ``feature_names_in_`` when the
            transformer was fitted on a DataFrame, else ``x0, x1, ...``. Explicit
            names need one entry per input feature and, after a DataFrame fit,
            must equal ``feature_names_in_``.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.

        Raises
        ------
        ValueError
            If ``input_features`` does not have one entry per input feature, or
            differs from ``feature_names_in_``.
        """
        check_is_fitted(self, "n_features_in_")
        return np.asarray(_check_feature_names_in(self, input_features), dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values pass through the float cast."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.allow_nan = True
        return tags
