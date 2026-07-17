import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NoTransformer(TransformerMixin, BaseEstimator):
    """Pass-through transformer that returns the input unchanged.

    Retains compatibility with the scikit-learn pipeline API without modifying
    the data.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features. Always set to 1 for this transformer.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers.utils.floats import NoTransformer
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
        self.n_features_in_ = 1
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
        return X

    def get_feature_names_out(self, input_features=None):
        """Return the original feature names.

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The original feature names.
        """
        if input_features is None:
            raise ValueError(
                "input_features must be provided to generate feature names."
            )
        return np.array(input_features)


class ToFloatTransformer(TransformerMixin, BaseEstimator):
    """Convert input data to floating-point type.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features. Always set to 1 for this transformer.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers.utils.floats import ToFloatTransformer
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
        self.n_features_in_ = 1
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
        return X.astype(float)
