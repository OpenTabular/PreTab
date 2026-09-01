# Embeddings

Embeddings map a categorical or text column to a dense vector produced by a pretrained model,
rather than recoding it into a small discrete representation the way
[categorical encoding](categorical_encoding.md) does. For high-cardinality text categories
(product titles, free-text tags, descriptions), a pretrained sentence embedding captures
semantic similarity that integer or one-hot encoding cannot. Similar labels land near each
other in the embedding space.

```python
from pretab.transformers import LanguageEmbeddingTransformer

t = LanguageEmbeddingTransformer(model_name="paraphrase-MiniLM-L3-v2")
X = [["red running shoes"], ["blue jacket"], ["red running shoes"]]   # 3 rows, 1 column
t.fit_transform(X).shape
# (3, embedding_dim_): one row per input; embedding_dim_ is set from the loaded
# model's own dimensionality once fitted, e.g. via t.embedding_dim_
```

Constructor highlights: `model_name="paraphrase-MiniLM-L3-v2"`, or pass a preloaded `model`
(useful for tests or a custom embedding backend without pulling in `sentence-transformers`).
Any object with a `.encode(X)` method works for the `transform` step; `embedding_dim_` is
read from `get_sentence_embedding_dimension()` when the model exposes it (as a real
`SentenceTransformer` does), falling back to a `dim` attribute, or to `0` if neither is
present. The registry key is `pretrained`.

```{important}
Language embeddings require the optional `embeddings` extra, which pulls in
`sentence-transformers`. Install it with `pip install "pretab[embeddings]"`. Without it,
requesting `pretrained` raises a clear `OptionalDependencyError`.
```

```{note}
The output width is fixed by the underlying model, not by any PreTab parameter, and is exposed
after fitting as `embedding_dim_`. It does not depend on `n_samples` or on how many distinct
categories are present. Swapping `model_name` for a different model changes the output width
accordingly.
```

```{tip}
Embeddings shine when category labels carry meaning as text. If the labels are opaque codes
with no semantic content, [integer encoding](categorical_encoding.md#integer-ordinal-encoding)
is simpler and just as effective.
```

## Where to go next

- [Categorical encoding](categorical_encoding.md) for compact integer and one-hot alternatives.
- [Installation](../getting_started/installation.md) for the `embeddings` extra.
- [Missing values](../core_concepts/missing_values.md) for categorical imputation.
