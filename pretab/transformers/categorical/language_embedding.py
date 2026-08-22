import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ...exceptions import OptionalDependencyError, PretabConfigError, PretabDataError


class LanguageEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """Encode categorical text features into embeddings using a pre-trained language model.

    Each text value is mapped to a dense embedding vector produced by a
    SentenceTransformer model, allowing free-text categorical columns to be used
    by downstream numerical models.

    Parameters
    ----------
    model_name : str, default="paraphrase-MiniLM-L3-v2"
        Name of the SentenceTransformer model to load when ``model`` is None.
    model : object, optional
        A preloaded SentenceTransformer model instance. When provided,
        ``model_name`` is ignored.

    Attributes
    ----------
    model_ : object
        The SentenceTransformer model used to compute embeddings, resolved during
        ``fit`` from ``model`` or by loading ``model_name``.
    n_features_in_ : int
        Number of input features seen during ``fit``.
    embedding_dim_ : int
        Dimensionality of the embeddings produced by ``model_``.

    Notes
    -----
    Requires the optional ``sentence-transformers`` dependency. Install it with
    ``pip install sentence-transformers`` or pass a preloaded ``model``.

    Examples
    --------
    >>> from pretab.transformers import LanguageEmbeddingTransformer
    >>> transformer = LanguageEmbeddingTransformer()  # doctest: +SKIP
    >>> embeddings = transformer.fit_transform([["red"], ["blue"], ["green"]])  # doctest: +SKIP
    >>> embeddings.shape[0]  # doctest: +SKIP
    3
    """

    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", model=None):
        """Store parameters only; the embedding model is loaded lazily in ``fit``."""
        self.model_name = model_name
        self.model = model  # Allow user to pass a preloaded model

    def _resolve_model(self):
        """Return the preloaded ``model`` or load one from ``model_name``."""
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise OptionalDependencyError(
                "sentence-transformers is not installed. Install it via `pip install sentence-transformers` or provide a preloaded model."
            ) from e
        return SentenceTransformer(self.model_name)

    def fit(self, X, y=None):
        """Load the embedding model and record the number of input features.

        Parameters
        ----------
        X : array-like
            Input categorical text features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted transformer.
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1] if X.ndim > 1 else 1
        self.model_ = self._resolve_model()
        # Read the embedding dim without calling encode() so call-count stays
        # predictable; fall back to the 'dim' attribute used by test stubs.
        if hasattr(self.model_, "get_sentence_embedding_dimension"):
            self.embedding_dim_ = int(self.model_.get_sentence_embedding_dimension())
        else:
            self.embedding_dim_ = int(getattr(self.model_, "dim", 0))
        return self

    def transform(self, X):
        """Transform text features into numerical embeddings.

        Each column is encoded independently and the resulting embeddings are
        concatenated horizontally, so the output always has one row per input
        sample regardless of the number of text columns.

        Parameters
        ----------
        X : array-like
            A 1D or 2D array-like of categorical text features.

        Returns
        -------
        embeddings : ndarray of shape (n_samples, n_features * embedding_dim)
            The concatenated embeddings for each text input.
        """
        if getattr(self, "model_", None) is None:
            raise PretabConfigError("Model is not initialized. Call `fit` before `transform`.")

        # Normalise to a 2D array of strings so each column is encoded on its own
        # and the row count is preserved (a flat encode would return
        # n_samples * n_features rows).
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != self.n_features_in_:
            raise PretabDataError(
                f"X has {arr.shape[1]} features, but {type(self).__name__} "
                f"is expecting {self.n_features_in_} features as input."
            )
        arr = arr.astype(str)

        column_embeddings = [self.model_.encode(arr[:, i].tolist(), convert_to_numpy=True) for i in range(arr.shape[1])]
        return np.hstack(column_embeddings)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()  # type: ignore[attr-defined]
        tags.input_tags.categorical = True
        tags.input_tags.string = True
        return tags

    def get_feature_names_out(self, input_features=None):
        """Return output feature names: one per embedding dimension per input column.

        Parameters
        ----------
        input_features : array-like of str or None
            Input feature names. When ``None``, names of the form ``x0, x1, ...``
            are generated.

        Returns
        -------
        feature_names_out : ndarray of str, shape (n_features_in_ * embedding_dim_,)
        """
        check_is_fitted(self, ["n_features_in_", "embedding_dim_"])
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        names = [f"{col}_emb{j}" for col in input_features for j in range(self.embedding_dim_)]
        return np.asarray(names, dtype=object)
