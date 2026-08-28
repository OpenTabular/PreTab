# Preprocessing utilities

Preprocessing utilities don't expand or recode a feature. They prepare it for the rest of the
pipeline, converting types or flagging missingness before it reaches the transformer that does
the actual representation work. `Preprocessor` wires these in automatically; most users never
instantiate them directly, but they are part of the public API for anyone building a custom
`ColumnTransformer` or pipeline by hand.

```{note}
This page is distinct from [`pretab.preprocessor`](../api/preprocessor.rst), the module that
holds the top-level `Preprocessor` facade. `pretab.preprocessing` is the package for these
smaller supporting transformers.
```

## Pass-through and type conversion

`NoTransformer` returns its input unchanged. It backs the `"none"` categorical and numerical
methods, letting a column skip representation entirely while still satisfying the
scikit-learn transformer API.

```python
from pretab.transformers import NoTransformer

t = NoTransformer()
X2 = t.fit_transform(X)  # X2 is X, unmodified
```

`ToFloatTransformer` casts its input to floating point. `Preprocessor` appends it after
one-hot encoding so the categorical block has the same dtype as the rest of the design matrix.

```python
from pretab.transformers import ToFloatTransformer

t = ToFloatTransformer()
t.fit_transform(X).dtype  # dtype('float64')
```

## Missing-value flagging

`MissingStateIndicator` emits a binary column marking where the input was missing, computed on
the raw data before imputation. `Preprocessor` uses it when `missing_policy="separate_state"`:
the indicator is kept apart from the imputed representation basis, so a downstream model can
learn a dedicated response to missingness instead of confusing it with an imputed value.

```python
import numpy as np
from pretab.transformers import MissingStateIndicator

X = np.array([[1.0], [np.nan], [3.0]])
MissingStateIndicator().fit_transform(X)
# array([[0.], [1.], [0.]])
```

```{tip}
Unlike `sklearn.impute.MissingIndicator`, `MissingStateIndicator` works on both numeric and
object (categorical) columns and always emits one column per input feature.
```

## Where to go next

- [Missing values](../core_concepts/missing_values.md) for the full `missing_policy` behavior.
- [Configuration](../core_concepts/configuration.md) for how `Preprocessor` builds its pipelines.
