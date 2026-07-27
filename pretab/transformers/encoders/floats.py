import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import PretabDataError


class NoTransformer(TransformerMixin, BaseEstimator):
    """Pass-through transformer that returns the input unchanged.

    Retains compatibility with the scikit-learn pipeline API without modifying
    the data.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.

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
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
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
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names.

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values pass through unchanged."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class ToFloatTransformer(TransformerMixin, BaseEstimator):
    """Convert input data to floating-point type.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.

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
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
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
        return X.astype(float)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names.

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values pass through the float cast."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


class RaiseOnNaNTransformer(TransformerMixin, BaseEstimator):
    """Pass data through, raising a clear error if it contains missing values.

    Sits at the head of the numerical pipeline when
    ``Preprocessor(handle_missing="error")`` drops the imputer, so the policy is
    enforced in one place for every numerical method.

    Without it the policy depended on whether the chosen transformer happened to
    notice NaN: the plain scikit-learn scalers ignore missing values by design,
    the PreTab expansion families declare ``allow_nan`` so they let them through,
    and only ``PLETransformer`` actually raised. Everything else silently emitted
    a NaN-contaminated feature matrix -- and the unsupervised feature-map path was
    worse than pass-through, because ``np.percentile`` over a column containing
    NaN makes *every* center NaN.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import RaiseOnNaNTransformer
    >>> RaiseOnNaNTransformer().fit_transform(np.array([[1.0], [2.0]])).shape
    (2, 1)
    """

    def fit(self, X, y=None):
        """Check ``X`` for missing values and record the feature count.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to check.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = self._check_no_nan(X)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return the input unchanged, raising if it contains missing values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to check.

        Returns
        -------
        X : ndarray
            The validated input, as a float array.

        Raises
        ------
        pretab.core.exceptions.PretabDataError
            If ``X`` contains any NaN.
        """
        check_is_fitted(self, "n_features_in_")
        return self._check_no_nan(X)

    @staticmethod
    def _check_no_nan(X) -> np.ndarray:
        """Coerce to a 2D float array and reject missing values."""
        array = np.asarray(X, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if np.isnan(array).any():
            raise PretabDataError(
                "Input contains NaN, but handle_missing='error' was requested.\n"
                "Fix: pass handle_missing='median' to impute missing values before "
                "the numerical method, or remove them from the input."
            )
        return array

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
            The output feature names.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values are rejected, not passed through."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
