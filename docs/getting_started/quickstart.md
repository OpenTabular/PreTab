# Quickstart

This page walks through the two ways to use pretab:

1. the high-level `Preprocessor` (`pretab.preprocessor.Preprocessor`), which builds a full
   scikit-learn pipeline from a config, and
2. the individual transformers, which behave like any other `sklearn` transformer.

## Using the `Preprocessor`

The `Preprocessor` detects feature types automatically and applies per-feature
preprocessing. It returns either a dictionary of feature blocks (default) or a single
stacked array.

```python
import numpy as np
import pandas as pd

from pretab.preprocessor import Preprocessor

# Simulated tabular dataset
df = pd.DataFrame({
    "age": np.random.randint(18, 65, size=100),
    "income": np.random.normal(60000, 15000, size=100).astype(int),
    "job": np.random.choice(["nurse", "engineer", "scientist", "teacher"], size=100),
    "city": np.random.choice(["Berlin", "Munich", "Hamburg", "Cologne"], size=100),
    "experience": np.random.randint(0, 40, size=100),
})
y = np.random.randn(100, 1)

# Optional per-feature preprocessing config
config = {
    "age": "ple",
    "income": "rbf",
    "experience": "quantile",
    "job": "one-hot",
    "city": "none",
}

preprocessor = Preprocessor(feature_preprocessing=config, task="regression")

# Fit and transform into a dictionary of feature arrays
X_dict = preprocessor.fit_transform(df, y)

# ... or get a single stacked array instead
X_array = preprocessor.transform(df, return_array=True)

# Inspect the resolved feature metadata
preprocessor.get_feature_info(verbose=True)
```

```{tip}
When no per-feature config is provided, the `Preprocessor` falls back to the global
`numerical_preprocessing` and `categorical_preprocessing` strategies. See the
[User Guide](../user_guide/preprocessing.md) for the full list of options.
```

## Using individual transformers

Every transformer follows the standard `sklearn` `fit` / `transform` API, so it can be
dropped into a `Pipeline` or `ColumnTransformer`.

```python
import numpy as np

from pretab.transformers import PLETransformer

x = np.random.randn(100, 1)
y = np.random.randn(100, 1)

x_ple = PLETransformer(n_bins=15, task="regression").fit_transform(x, y)
assert x_ple.shape[1] == 15
```

```{note}
`PLETransformer` is supervised: it uses the target `y` during `fit` to place its bin
edges. Always pass `y` when fitting it, or any pipeline that includes it.
```

For spline transformers, the penalty matrix can be extracted with
`get_penalty_matrix()`:

```python
import numpy as np

from pretab.transformers import ThinPlateSplineTransformer

x = np.random.randn(100, 1)

tp = ThinPlateSplineTransformer(n_basis=15)
x_tp = tp.fit_transform(x)
assert x_tp.shape[1] == 15

penalty = tp.get_penalty_matrix()
```

## Next steps

- Learn about the available strategies in the [User Guide](../user_guide/preprocessing.md).
- Browse every class in the [API Reference](../api/index.rst).
