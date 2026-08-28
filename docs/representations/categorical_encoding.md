# Categorical encoding

Categorical encoding maps a category to codes or indicators a model can consume directly.
PreTab covers compact integer encoding and explicit one-hot encoding, both of which handle
unseen categories without raising. For high-cardinality text where the labels themselves carry
meaning, see [Embeddings](embeddings.md) instead.

## Integer (ordinal) encoding

The default categorical method maps each category to an integer. It is compact and works well
as an input to models that consume category indices, such as embedding layers.

```python
import numpy as np
from pretab.transformers import ContinuousOrdinalTransformer

X = np.array([["a"], ["b"], ["a"], ["c"]])   # (4, 1)
t = ContinuousOrdinalTransformer()
t.fit_transform(X).ravel()
# array([1, 2, 1, 3])  codes start at 1; output shape stays (4, 1)
t.transform(np.array([["unseen"]])).ravel()
# array([0])  unseen categories map to the reserved 0 code
```

Unseen categories at transform time map to a reserved slot (code `0`) rather than raising, so a
model in production never crashes on a new label.

```{note}
Integer encoding imposes an order on the codes. Feed it to models that treat the code as an
index (trees, embedding layers), not to a plain linear model that would read the codes as
magnitudes.
```

## One-hot encoding

One-hot encoding produces one indicator column per category, the right choice when the
downstream model should treat categories as unordered.

```python
import pandas as pd
from pretab import Preprocessor

df = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
pre = Preprocessor(categorical_method="one-hot")
pre.fit(df)
pre.get_feature_names_out()
# array(['cat_color_blue', 'cat_color_green', 'cat_color_red'], dtype=object)
```

The alias `ohe` resolves to `one-hot`. There is also `onehot_from_ordinal`, which one-hot
encodes an already integer-coded column.

```{warning}
One-hot width grows with cardinality: a column with `k` distinct categories produces `k`
output columns (3 in the example above). A column with thousands of categories produces
thousands of columns. Use the [output budget](../core_concepts/outputs_and_inspection.md) to
cap it, or prefer integer encoding or [embeddings](embeddings.md) for high-cardinality columns.
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
