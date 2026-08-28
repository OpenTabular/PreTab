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
X2 = t.fit_transform(x)
```

Constructor highlights: `model_name="paraphrase-MiniLM-L3-v2"`, or pass a preloaded `model`.
The registry key is `pretrained`.

```{important}
Language embeddings require the optional `embeddings` extra, which pulls in
`sentence-transformers`. Install it with `pip install "pretab[embeddings]"`. Without it,
requesting `pretrained` raises a clear `OptionalDependencyError`.
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
