# Categorical encoding

Categorical encoding maps a category to codes or indicators a model can consume directly.
PreTab covers compact integer encoding and explicit one-hot encoding, both of which handle
unseen categories without raising. For high-cardinality text where the labels themselves carry
meaning, see [Embeddings](embeddings.md) instead.

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
prefer integer encoding or [embeddings](embeddings.md) for high-cardinality columns.
```

## Choosing a categorical method

| If the column is... | Reach for... |
| --- | --- |
| Low cardinality, unordered | One-hot |
| Fed to a tree or embedding layer | Integer |
| High-cardinality meaningful text | [Language embedding](embeddings.md) |

## Where to go next

- [Embeddings](embeddings.md) for high-cardinality text categories.
- [Missing values](../core_concepts/missing_values.md) for categorical imputation.
- [Configuration](../core_concepts/configuration.md) to set categorical methods per column.
