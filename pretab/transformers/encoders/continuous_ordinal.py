import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...core.exceptions import InvalidParamError


class ContinuousOrdinalTransformer(TransformerMixin, BaseEstimator):
    """Encode categorical features as continuous integer values.

    Each unique category within a feature is assigned an integer based on its
    sorted order. Unknown or missing categories are mapped to ``0``. This is
    useful for models that can only handle numerical input.

    Attributes
    ----------
    mapping_ : list of dict
        One dictionary per feature mapping original categories to integers.

    Notes
    -----
    Categories are numbered starting at ``1`` in sorted order; the value ``0`` is
    reserved for categories not seen during ``fit`` (and for ``None``).

    Examples
    --------
    >>> import numpy as np
    >>> from pretab.transformers import ContinuousOrdinalTransformer
    >>> X = np.array([["a", "x"], ["b", "y"], ["a", "x"]], dtype=object)
    >>> transformer = ContinuousOrdinalTransformer()
    >>> transformer.fit_transform(X).shape
    (3, 2)
    """

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
        # Fit should determine the mapping from original categories to sequential integers starting from 0
        self.mapping_ = [
            {category: i + 1 for i, category in enumerate(np.unique(col))}
            for col in X.T
        ]
        for mapping in self.mapping_:
            mapping[None] = 0  # Assign 0 to unknown values
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
        # Transform the categories to their mapped integer values
        X_transformed = np.array(
            [
                [self.mapping_[col].get(value, 0) for col, value in enumerate(row)]
                for row in X
            ]
        )
        return X_transformed

    def get_feature_names_out(self, input_features=None):
        """Return the output feature names (unchanged from the input).

        Parameters
        ----------
        input_features : list of str
            The names of the input features.

        Returns
        -------
        input_features : ndarray of shape (n_features,)
            The names of the output features after transformation.
        """
        check_is_fitted(self, "mapping_")
        if input_features is None:
            raise InvalidParamError("input_features must be specified")
        return input_features

    def __sklearn_tags__(self):
        """Declare that missing/unknown categories are handled (mapped to 0)."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
