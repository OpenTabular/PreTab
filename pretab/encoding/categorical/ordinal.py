import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names, check_is_fitted

from ...core.representation import RepresentationSpecMixin
from ...core.validation import resolve_input_features
from ...exceptions import PretabDataError


def _is_missing_value(value) -> bool:
    """Return True for ``None`` or a NaN float; False for any other value."""
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


class ContinuousOrdinalTransformer(RepresentationSpecMixin, TransformerMixin, BaseEstimator):
    """Encode categorical features as continuous integer values.

    Each unique category within a feature is assigned an integer based on its
    sorted order. Unknown, missing (``None`` or NaN), or unseen categories are
    mapped to ``0``. This is useful for models that can only handle numerical
    input.

    Attributes
    ----------
    mapping_ : list of dict
        One dictionary per feature mapping original categories to integers.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    feature_names_in_ : ndarray of str
        Names of the input columns seen during ``fit``. Defined only when ``X``
        was a DataFrame with string column names; ``transform`` then requires
        the same columns in the same order.

    Notes
    -----
    Categories are numbered starting at ``1`` in sorted order; the value ``0`` is
    reserved for categories not seen during ``fit`` (and for ``None`` / NaN).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ContinuousOrdinalTransformer
    >>> X = np.array([["a", "x"], ["b", "y"], ["a", "x"]], dtype=object)
    >>> transformer = ContinuousOrdinalTransformer()
    >>> transformer.fit_transform(X).shape
    (3, 2)
    """

    _representation_family = "ordinal"
    _representation_component_kind = "category"

    def fit(self, X, y=None):
        """Learn the mapping from categories to integers for each feature.

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
        # Coerce to 2-D ndarray so DataFrame.T / DataFrame row-iteration work correctly
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.mapping_ = []
        for j in range(X.shape[1]):
            column = X[:, j]
            # Missing values (None / NaN) are excluded before sorting: np.unique
            # cannot compare a NaN or None against a string category.
            non_missing = np.array([v for v in column if not _is_missing_value(v)], dtype=object)
            categories = np.unique(non_missing) if non_missing.size else np.array([], dtype=object)
            mapping = {category: i + 1 for i, category in enumerate(categories)}
            mapping[None] = 0  # Assign 0 to unknown/missing values
            self.mapping_.append(mapping)
        self.n_features_in_ = len(self.mapping_)
        return self

    def transform(self, X):
        """Apply the learned category-to-integer mapping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_features)
            The transformed data with integer values.
        """
        check_is_fitted(self, "mapping_")
        _check_feature_names(self, X, reset=False)
        # Coerce to 2-D ndarray so DataFrame row-iteration and empty-input shape both work
        X = np.asarray(X, dtype=object)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise PretabDataError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        out = np.zeros(X.shape, dtype=int)
        for j, mapping in enumerate(self.mapping_):
            out[:, j] = [mapping.get(v, 0) for v in X[:, j]]
        return out

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names (unchanged from the input).

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
            The names of the output features after transformation.

        Raises
        ------
        InvalidParamError
            If ``input_features`` does not have one entry per input feature.
        ValueError
            If ``input_features`` differs from ``feature_names_in_``.
        """
        check_is_fitted(self, "mapping_")
        return np.asarray(resolve_input_features(self, input_features), dtype=object)

    def __sklearn_tags__(self):
        """Declare that missing/unknown categories are handled (mapped to 0)."""
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.allow_nan = True
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags
