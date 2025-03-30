import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LanguageEmbeddingTransformer(TransformerMixin, BaseEstimator):
    """A transformer that encodes categorical text features into embeddings using a pre-trained language model."""

    def __init__(self, model_name="paraphrase-MiniLM-L3-v2", model=None):
        """
        Initializes the transformer with a language embedding model.

        Parameters:
        - model_name (str): The name of the SentenceTransformer model to use (if model is None).
        - model (object, optional): A preloaded SentenceTransformer model instance.
        """
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
        """Fit method (not required for a transformer but included for compatibility)."""
        self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        return self

    def transform(self, X):
        """
        Transforms input categorical text features into numerical embeddings.

        Parameters:
        - X: A 1D or 2D array-like of categorical text features.

        Returns:
        - A 2D numpy array with embeddings for each text input.
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
