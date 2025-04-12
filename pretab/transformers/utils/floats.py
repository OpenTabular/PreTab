import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NoTransformer(TransformerMixin, BaseEstimator):
    """A transformer that does not preprocess the data but retains compatibility with the sklearn pipeline API. It
    simply returns the input data as is.

    Methods:
        fit(X, y=None): Fits the transformer to the data (no operation).
        transform(X): Returns the input data unprocessed.
        get_feature_names_out(input_features=None): Returns the original feature names.
    """

    def fit(self, X, y=None):
        """Fits the transformer to the data. No operation is performed.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        """Returns the input data unprocessed.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform.

        Returns:
            X (array-like): The same input data, unmodified.
        """
        return X

    def get_feature_names_out(self, input_features=None):
        """Returns the original feature names.

        Parameters:
            input_features (list of str or None): The names of the input features.

        Returns:
            feature_names (array of shape (n_features,)): The original feature names.
        """
        if input_features is None:
            raise ValueError(
                "input_features must be provided to generate feature names."
            )
        return np.array(input_features)


class ToFloatTransformer(TransformerMixin, BaseEstimator):
    """A transformer that converts input data to float type."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        return X.astype(float)
