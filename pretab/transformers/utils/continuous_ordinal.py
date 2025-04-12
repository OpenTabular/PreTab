import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ContinuousOrdinalTransformer(BaseEstimator, TransformerMixin):
    """This encoder converts categorical features into continuous integer values. Each unique category within a feature
    is assigned a unique integer based on its order of appearance in the dataset. This transformation is useful for
    models that can only handle continuous data.

    Attributes:
        mapping_ (list of dicts): A list where each element is a dictionary mapping original categories to integers
                                  for a single feature.

    Methods:
        fit(X, y=None): Learns the mapping from original categories to integers.
        transform(X): Applies the learned mapping to the data.
        get_feature_names_out(input_features=None): Returns the input features after transformation.
    """

    def fit(self, X, y=None):
        """Learns the mapping from original categories to integers for each feature.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to fit.
            y (ignored): Not used, present for API consistency by convention.

        Returns:
            self: Returns the instance itself.
        """
        # Fit should determine the mapping from original categories to sequential integers starting from 0
        self.mapping_ = [
            {category: i + 1 for i, category in enumerate(np.unique(col))}
            for col in X.T
        ]
        for mapping in self.mapping_:
            mapping[None] = 0  # Assign 0 to unknown values
        return self

    def transform(self, X):
        """Transforms the categories in X to their corresponding integer values based on the learned mapping.

        Parameters:
            X (array-like of shape (n_samples, n_features)): The input data to transform.

        Returns:
            X_transformed (ndarray of shape (n_samples, n_features)): The transformed data with integer values.
        """
        # Transform the categories to their mapped integer values
        X_transformed = np.array(
            [
                [self.mapping_[col].get(value, 0) for col, value in enumerate(row)]
                for row in X
            ]
        )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Returns the names of the transformed features.

        Parameters:
            input_features (list of str): The names of the input features.

        Returns:
            input_features (array of shape (n_features,)): The names of the output features after transformation.
        """
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features
