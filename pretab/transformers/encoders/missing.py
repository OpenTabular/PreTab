import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


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
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return pd.isna(X).astype(float)

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names, each suffixed with ``__missing``.

        Parameters
        ----------
        input_features : list of str or None
            The names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        feature_names : ndarray of shape (n_features,)
            The output feature names.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray([f"{name}__missing" for name in input_features], dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing values are expected (they are the signal)."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.allow_nan = True
        return tags
