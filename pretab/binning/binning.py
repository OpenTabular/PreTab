import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CustomBinTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, bins):
        # bins can be a scalar (number of bins) or array-like (bin edges)
        self.bins = bins

    def fit(self, X, y=None):
        # Fit doesn't need to do anything as we are directly using provided bins
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        if isinstance(self.bins, int):
            # Calculate equal width bins based on the range of the data and number of bins
            _, bins = pd.cut(X.squeeze(), bins=self.bins, retbins=True)
        else:
            # Use predefined bins
            bins = self.bins

        # Apply the bins to the data
        binned_data = pd.cut(  # type: ignore
            X.squeeze(),
            bins=np.sort(np.unique(bins)),  # type: ignore
            labels=False,
            include_lowest=True,
        )
        return np.expand_dims(np.array(binned_data), 1)

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
