import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.representation import RepresentationSpecMixin
from ...exceptions import PretabDataError

_NOT_ORDINAL_MSG = (
    "OneHotFromOrdinalTransformer requires input that is already ordinal-encoded "
    "(non-negative integer codes); got values that cannot be cast to int. Use "
    "categorical_method='one-hot' to one-hot encode raw categories directly, or "
    "'int' to ordinal-encode first."
)


class OneHotFromOrdinalTransformer(RepresentationSpecMixin, TransformerMixin, BaseEstimator):
    """Convert ordinal-encoded features into a one-hot encoded representation.

    This is useful when features have already been ordinal-encoded and a one-hot
    representation is required for model training.

    .. deprecated:: 1.0.0
        ``OneHotFromOrdinalTransformer`` is deprecated and will be removed in a
        future release. Use the ``"one-hot"`` categorical method (backed by
        scikit-learn's :class:`~sklearn.preprocessing.OneHotEncoder`), which
        one-hot encodes raw categories directly without a separate
        ordinal-encoding step.

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

    _representation_family = "onehot"
    _representation_component_kind = "category"

    def __init__(self):
        warnings.warn(
            "OneHotFromOrdinalTransformer is deprecated and will be removed in a "
            "future release. Use the 'one-hot' categorical method (sklearn's "
            "OneHotEncoder), which one-hot encodes raw categories directly.",
            DeprecationWarning,
            stacklevel=2,
        )

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
        X = np.asarray(X)
        try:
            codes = X.astype(int)
        except (TypeError, ValueError) as exc:
            raise PretabDataError(_NOT_ORDINAL_MSG) from exc
        self.max_bins_ = np.max(codes, axis=0) + 1  # Find the maximum bin index for each feature
        self.n_features_in_ = X.shape[1]
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
        try:
            X = X.astype(int)
        except (TypeError, ValueError) as exc:
            raise PretabDataError(_NOT_ORDINAL_MSG) from exc
        # Initialize an empty list to hold the one-hot encoded arrays
        one_hot_encoded = []
        for i, max_bins in enumerate(self.max_bins_):
            max_bins = int(max_bins)
            codes = X[:, i]
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
