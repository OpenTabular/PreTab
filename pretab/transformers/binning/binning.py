from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ...core.exceptions import InvalidParamError
from ...core.params import UNSET, AliasResolverMixin


class CustomBinTransformer(AliasResolverMixin, TransformerMixin, BaseEstimator):
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

    Notes
    -----
    This transformer operates on a single feature of shape ``(n_samples, 1)``. When
    ``bins`` is an integer, equal-width bin edges are computed from the data range;
    when it is an array-like, the provided edges are used directly. The output
    contains integer bin indices.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import CustomBinTransformer
    >>> X = np.linspace(0, 1, 10).reshape(-1, 1)
    >>> transformer = CustomBinTransformer(bins=4)
    >>> transformer.fit_transform(X).shape
    (10, 1)
    """

    _param_aliases: ClassVar[dict[str, str]] = {"bins": "n_basis"}

    def __init__(self, n_basis=UNSET, bins=UNSET):
        # A basis count (int) yields equal-width bins; an array-like is used as bin edges.
        self.n_basis = n_basis
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

        bins_spec = self._resolve_param("n_basis", default=UNSET)
        if bins_spec is UNSET:
            raise InvalidParamError("CustomBinTransformer requires 'n_basis' (or the legacy 'bins').")

        if isinstance(bins_spec, int):
            # Calculate equal width bins based on the range of the data and number of bins
            _, bins = pd.cut(X.squeeze(), bins=bins_spec, retbins=True)
        else:
            # Use predefined bins
            bins = bins_spec

        # Apply the bins to the data
        binned_data = pd.cut(  # type: ignore
            X.squeeze(),
            bins=np.sort(np.unique(bins)),  # type: ignore
            labels=False,
            include_lowest=True,
        )
        return np.expand_dims(np.array(binned_data), 1)

    def get_feature_names_out(self, input_features=None):
        """Return the names of the transformed features.

        Parameters
        ----------
        input_features : list of str
            The names of the input features.

        Returns
        -------
        input_features : ndarray of shape (n_features,)
            The names of the output features after transformation.
        """
        if input_features is None:
            raise ValueError("input_features must be specified")
        return input_features
