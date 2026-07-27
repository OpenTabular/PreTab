from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import InvalidParamError, PretabDataError
from ...core.params import UNSET, AliasResolverMixin


class CustomBinTransformer(AliasResolverMixin, TransformerMixin, BaseEstimator):
    """
    Custom binning transformer for one-dimensional numerical features.

    This transformer bins continuous values into discrete intervals, using either a fixed number of equal-width bins
    or a user-provided array of bin edges. It is compatible with scikit-learn pipelines.

    Parameters
    ----------
    output_dim : int or array-like
        If int, defines the number of equal-width bins. If array-like, defines
        the bin edges to use directly. Note that ``output_dim`` here is the
        number of *bins*, not the number of output columns: this transformer
        always emits a single ordinal column of integer bin indices. The bin
        count only becomes an output width after a subsequent one-hot encoding.

    Attributes
    ----------
    n_features_in_ : int
        The number of input features seen during ``fit`` (expected to be 1).

    bin_edges_ : list of ndarray
        Bin edges learned during ``fit``, one array per input column. Reused
        verbatim by every ``transform`` call so a given value always receives the
        same code.

    total_output_dim_ : int
        Total number of output columns (fitted). Always ``1`` because the output
        is a single ordinal column.

    Notes
    -----
    This transformer operates on a single feature of shape ``(n_samples, 1)``. When
    ``output_dim`` is an integer, equal-width bin edges are computed from the data
    range **at fit time**; when it is an array-like, the provided edges are used
    directly. The output contains integer bin indices in a single column, so its
    width is ``1`` regardless of ``output_dim`` -- this is a documented exception
    to the exact-width contract that the fixed-basis families follow.

    Values outside the fitted range are clipped into the first or last bin, so
    ``transform`` never emits ``NaN`` into what is otherwise an integer column.

    The input must be numeric: binning is performed with :func:`pandas.cut`, so
    string / categorical data cannot be processed and raises a
    :class:`~pretab.core.exceptions.PretabDataError`. Encode such columns with a
    categorical method (e.g. ``"int"`` or ``"one-hot"``) before binning.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import CustomBinTransformer
    >>> X = np.linspace(0, 1, 10).reshape(-1, 1)
    >>> transformer = CustomBinTransformer(output_dim=4)
    >>> transformer.fit_transform(X).shape
    (10, 1)
    """

    _param_aliases: ClassVar[dict[str, str]] = {}

    def __init__(self, output_dim=UNSET):
        # An int yields equal-width bins; an array-like is used as bin edges.
        self.output_dim = output_dim

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
        X = self._as_numeric(np.asarray(X))
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.total_output_dim_ = 1

        bins_spec = self._resolve_param("output_dim", default=UNSET)
        if bins_spec is UNSET:
            raise InvalidParamError("CustomBinTransformer requires 'output_dim'.")

        # Edges are learned here and reused verbatim by ``transform``. Deriving
        # them inside ``transform`` instead made the encoding depend on the batch
        # being transformed, so the same value got different codes depending on
        # which other rows travelled with it.
        self.bin_edges_ = [
            self._edges_for(X[:, j], bins_spec) for j in range(X.shape[1])
        ]
        return self

    @staticmethod
    def _as_numeric(X: np.ndarray) -> np.ndarray:
        """Return ``X`` as a float array, raising a clear error on non-numeric input."""
        if np.issubdtype(X.dtype, np.number):
            return X
        try:
            return X.astype(np.float64)
        except (ValueError, TypeError) as exc:
            raise PretabDataError(
                "CustomBinTransformer requires numeric input: it bins continuous "
                "values with pandas.cut and cannot process string/categorical "
                "data. Encode string columns with a categorical method (e.g. "
                "'int' or 'one-hot') before binning."
            ) from exc

    @staticmethod
    def _edges_for(column: np.ndarray, bins_spec) -> np.ndarray:
        """Resolve the bin edges for one column from an int count or explicit edges."""
        if isinstance(bins_spec, int):
            _, edges = pd.cut(column, bins=bins_spec, retbins=True)
        else:
            edges = bins_spec
        return np.sort(np.unique(np.asarray(edges)))

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

        check_is_fitted(self, "bin_edges_")

        X = np.asarray(X)  # Ensures squeeze works and consistent input
        if X.ndim != 2 or X.shape[1] != 1:
            raise PretabDataError("Input must be a 2D array with shape (n_samples, 1).")

        X = self._as_numeric(X)

        edges = self.bin_edges_[0]
        # Clip into the fitted range so unseen extremes land in the first / last
        # bin rather than becoming NaN in an otherwise integer column.
        values = np.clip(X[:, 0], edges[0], edges[-1])

        binned_data = pd.cut(  # type: ignore
            values,
            bins=edges,
            labels=False,
            include_lowest=True,
        )
        return np.expand_dims(np.asarray(binned_data, dtype=int), 1)

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
            raise InvalidParamError("input_features must be specified")
        return input_features
