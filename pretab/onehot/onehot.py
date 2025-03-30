import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotFromOrdinalTransformer(TransformerMixin, BaseEstimator):
    """A transformer that takes ordinal-encoded features and converts them into one-hot encoded format. This is useful
    in scenarios where features have been pre-encoded with ordinal encoding and a one-hot representation is required for
    model training.

    Attributes:
        max_bins_ (ndarray of shape (n_features,)): An array containing the maximum bin index for each feature,
                                                    determining the size of the one-hot encoded array for that feature.

    Methods:
        fit(X, y=None): Learns the maximum bin index for each feature.
        transform(X): Converts ordinal-encoded features into one-hot format.
        get_feature_names_out(input_features=None): Returns the feature names after one-hot encoding.
    """

    def fit(self, X, y=None):
        """Learns the maximum bin index for each feature from the data.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit, containing ordinal-encoded features.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        self.max_bins_ = (
            np.max(X, axis=0).astype(int) + 1
        )  # Find the maximum bin index for each feature
        return self

    def transform(self, X):
        """Transforms ordinal-encoded features into one-hot encoded format based on the `max_bins_` learned during
        fitting.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform,
            containing ordinal-encoded features.

        Returns:
            X_one_hot (ndarray of shape (n_samples, n_output_features)): The one-hot encoded features.
        """
        # Initialize an empty list to hold the one-hot encoded arrays
        one_hot_encoded = []
        for i, max_bins in enumerate(self.max_bins_):
            # Convert each feature to one-hot using its max_bins
            feature_one_hot = np.eye(max_bins)[X[:, i].astype(int)]
            one_hot_encoded.append(feature_one_hot)
        # Concatenate the one-hot encoded features horizontally
        return np.hstack(one_hot_encoded)

    def get_feature_names_out(self, input_features=None):
        """Generates feature names for the one-hot encoded features based on the input feature names and the number of
        bins.

        Parameters:
            input_features (list of str): The names of the input features that were ordinal-encoded.

        Returns:
            feature_names (array of shape (n_output_features,)): The names of the one-hot encoded features.
        """
        feature_names = []
        for i, max_bins in enumerate(self.max_bins_):
            feature_names.extend([f"{input_features[i]}_bin_{j}" for j in range(int(max_bins))])  # type: ignore
        return np.array(feature_names)
