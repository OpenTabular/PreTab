# Categorical

Categorical features range from a handful of labels to free text with thousands of distinct
values. PreTab covers the spectrum: compact integer encoding, explicit one-hot, and pretrained
language embeddings for high-cardinality text. All of them handle unseen categories without
raising.

## Integer (ordinal) encoding

The default categorical method maps each category to an integer. It is compact and works well
as an input to models that consume category indices, such as embedding layers.

```python
from pretab.transformers import ContinuousOrdinalTransformer

t = ContinuousOrdinalTransformer()
X2 = t.fit_transform(x)
```

Unseen categories at transform time map to a reserved slot rather than raising, so a model in
production never crashes on a new label.

```{note}
Integer encoding imposes an order on the codes. Feed it to models that treat the code as an
index (trees, embedding layers), not to a plain linear model that would read the codes as
magnitudes.
```

## One-hot encoding

One-hot encoding produces one indicator column per category, the right choice when the
downstream model should treat categories as unordered.

```python
pre = Preprocessor(categorical_method="one-hot")
```

The alias `ohe` resolves to `one-hot`. There is also `onehot_from_ordinal`, which one-hot
encodes an already integer-coded column.

```{warning}
One-hot width grows with cardinality. A column with thousands of categories produces thousands
of columns. Use the [output budget](../core_concepts/outputs_and_inspection.md) to cap it, or
prefer integer encoding or embeddings for high-cardinality columns.
```

## Language embeddings

For high-cardinality text categories (product titles, free-text tags, descriptions), a
pretrained sentence embedding captures semantic similarity that integer or one-hot encoding
cannot. Similar labels land near each other in the embedding space.

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
with no semantic content, integer encoding is simpler and just as effective.
```

## Choosing a categorical method

| If the column is... | Reach for... |
| --- | --- |
| Low cardinality, unordered | One-hot |
| Fed to a tree or embedding layer | Integer |
| High-cardinality meaningful text | Language embedding |

## Where to go next

- [Missing values](../core_concepts/missing_values.md) for categorical imputation.
- [Configuration](../core_concepts/configuration.md) to set categorical methods per column.
- [Installation](../getting_started/installation.md) for the `embeddings` extra.
