import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class OneHotFromOrdinalTransformer(TransformerMixin, BaseEstimator):
    """Convert ordinal-encoded features into a one-hot encoded representation.

    This is useful when features have already been ordinal-encoded and a one-hot
    representation is required for model training.

    Attributes
    ----------
    max_bins_ : ndarray of shape (n_features,)
        Array containing the maximum bin index (plus one) for each feature, which
        determines the width of the one-hot block for that feature.

    Notes
    -----
    Each input feature ``i`` is expanded into ``max_bins_[i]`` columns, so the
    total number of output columns is the sum of ``max_bins_`` across all
    features.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import OneHotFromOrdinalTransformer
    >>> X = np.array([[0, 1], [1, 0], [2, 1]])
    >>> transformer = OneHotFromOrdinalTransformer()
    >>> transformer.fit_transform(X).shape
    (3, 5)
    """

    def fit(self, X, y=None):
        """Learn the maximum bin index for each feature from the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data containing ordinal-encoded features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        self.max_bins_ = np.max(X, axis=0).astype(int) + 1  # Find the maximum bin index for each feature
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """Convert ordinal-encoded features into one-hot encoded format.

        Uses the ``max_bins_`` learned during fitting.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data containing ordinal-encoded features.

        Returns
        -------
        X_one_hot : ndarray of shape (n_samples, n_output_features)
            The one-hot encoded features.

        Notes
        -----
        Codes outside the range learned during :meth:`fit` (negative values or
        values ``>= max_bins_[i]``) are treated as unknown and encoded as an
        all-zero row, mirroring scikit-learn's ``handle_unknown="ignore"``.
        """
        check_is_fitted(self, "max_bins_")
        X = np.asarray(X)
        # Initialize an empty list to hold the one-hot encoded arrays
        one_hot_encoded = []
        for i, max_bins in enumerate(self.max_bins_):
            max_bins = int(max_bins)
            codes = X[:, i].astype(int)
            # Codes outside the fitted range map to an all-zero row instead of
            # raising an IndexError on np.eye indexing.
            in_range = (codes >= 0) & (codes < max_bins)
            feature_one_hot = np.zeros((codes.shape[0], max_bins))
            feature_one_hot[np.nonzero(in_range)[0], codes[in_range]] = 1.0
            one_hot_encoded.append(feature_one_hot)
        # Concatenate the one-hot encoded features horizontally
        return np.hstack(one_hot_encoded)

    def get_feature_names_out(self, input_features=None):
        """Generate feature names for the one-hot encoded output.

        Parameters
        ----------
        input_features : list of str
            The names of the input features that were ordinal-encoded.

        Returns
        -------
        feature_names : ndarray of shape (n_output_features,)
            The names of the one-hot encoded features.
        """
        feature_names = []
        for i, max_bins in enumerate(self.max_bins_):
            feature_names.extend([f"{input_features[i]}_bin_{j}" for j in range(int(max_bins))])  # type: ignore
        return np.array(feature_names)

    def __sklearn_tags__(self):
        """Ordinal integer input is required; missing values are not supported."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        return tags
