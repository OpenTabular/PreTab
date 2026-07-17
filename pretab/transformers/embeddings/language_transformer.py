import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


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
    model : object
        The SentenceTransformer model used to compute embeddings.
    n_features_in_ : int
        Number of input features seen during ``fit``.

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
        """Initialize the transformer and load the embedding model if needed."""
        self.model_name = model_name
        self.model = model  # Allow user to pass a preloaded model

        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is not installed. Install it via `pip install sentence-transformers` or provide a preloaded model."
                ) from e

    def fit(self, X, y=None):
        """Record the number of input features (no fitting required).

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
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        return self

    def transform(self, X):
        """Transform text features into numerical embeddings.

        Parameters
        ----------
        X : array-like
            A 1D or 2D array-like of categorical text features.

        Returns
        -------
        embeddings : ndarray of shape (n_samples, embedding_dim)
            The embedding for each text input.
        """
        if isinstance(X, np.ndarray):
            X = (
                X.flatten().astype(str).tolist()
            )  # Convert to a list of strings if passed as an array
        elif isinstance(X, list):
            X = [str(x) for x in X]  # Ensure everything is a string

        if self.model is None:
            raise ValueError(
                "Model is not initialized. Ensure that the model is properly loaded."
            )
        embeddings = self.model.encode(
            X, convert_to_numpy=True
        )  # Get sentence embeddings
        return embeddings
