import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CustomBinTransformer(TransformerMixin, BaseEstimator):
    """
    Custom binning transformer for one-dimensional numerical features.

    This transformer bins continuous values into discrete intervals, using either a fixed number of equal-width bins
    or a user-provided array of bin edges. It is compatible with scikit-learn pipelines.

    Parameters
    ----------
    bins : int or array-like
        If int, defines the number of equal-width bins. If array-like, defines the bin edges to use directly.

    Attributes
    ----------
    n_features_in_ : int
        The number of input features. Always set to 1 for this transformer.
    """

    def __init__(self, bins):
        # bins can be a scalar (number of bins) or array-like (bin edges)
        self.bins = bins

    def fit(self, X, y=None):
        """
        Fit the transformer on the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input data.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        # Fit doesn't need to do anything as we are directly using provided bins
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        """
        Transform the data using the specified binning strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            Input data to transform.

        Returns
        -------
        X_binned : ndarray of shape (n_samples, 1)
            Binned data with integer bin indices.
        """

        X = np.asarray(X)  # Ensures squeeze works and consistent input
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Input must be a 2D array with shape (n_samples, 1).")

        if X.shape[0] <= 2:
            raise ValueError("Input must have more than 2 observations.")

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
