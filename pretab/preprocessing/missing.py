import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names, _check_feature_names_in, check_is_fitted


class MissingStateIndicator(TransformerMixin, BaseEstimator):
    """Emit a binary ``__missing`` column marking where the input was missing.

    Used by the ``missing_policy="separate_state"`` path: the column is produced
    on the *raw* input (before imputation) and kept separate from the ordinary
    representation basis, so a downstream model can learn a dedicated response to
    missingness rather than confounding it with an imputed value.

    Unlike :class:`sklearn.impute.MissingIndicator`, this works on both numeric
    and object (categorical) columns via :func:`pandas.isna` and always emits one
    column per input feature.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during ``fit``.
    feature_names_in_ : ndarray of str
        Names of the input columns seen during ``fit``. Defined only when ``X``
        was a DataFrame with string column names; ``transform`` then requires
        the same columns in the same order.

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import MissingStateIndicator
    >>> X = np.array([[1.0], [np.nan], [3.0]])
    >>> MissingStateIndicator().fit_transform(X)
    array([[0.],
           [1.],
           [0.]])
    """

    def fit(self, X, y=None):
        """Record the input feature count.

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
        _check_feature_names(self, X, reset=True)
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        return self

    def transform(self, X):
        """Return a float mask (``1.0`` where missing, ``0.0`` otherwise).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to inspect.

        Returns
        -------
        mask : ndarray of shape (n_samples, n_features)
            The missingness indicator as ``float``.
        """
        check_is_fitted(self, "n_features_in_")
        _check_feature_names(self, X, reset=False)
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.isna(X).astype(float)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names, each suffixed with ``__missing``.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names. ``None`` uses ``feature_names_in_`` when the
            transformer was fitted on a DataFrame, else ``x0, x1, ...``. Explicit
            names need one entry per input feature and, after a DataFrame fit,
            must equal ``feature_names_in_``.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.

        Raises
        ------
        ValueError
            If ``input_features`` does not have one entry per input feature, or
            differs from ``feature_names_in_``.
        """
        check_is_fitted(self, "n_features_in_")
        names = np.asarray(_check_feature_names_in(self, input_features))
        return np.asarray([f"{name}__missing" for name in names], dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values are expected (they are the signal)."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.allow_nan = True
        return tags
