# Inside an sklearn Pipeline

The **standalone transformers** return plain arrays and follow the `sklearn` API exactly, so
they drop straight into a `ColumnTransformer` and `Pipeline`, and work with
`cross_val_score`, `GridSearchCV`, and every other `sklearn` utility. The high-level
`Preprocessor` composes the same way (it returns a single stacked array by default), but this
page focuses on wiring each column's transformer by hand for fine-grained, per-transformer
control: addressable hyperparameters (`step__param`), one estimator per column. The
[mixed-column example](#configure-mixed-columns-with-preprocessor) then uses `Preprocessor`
to configure, inspect and save the fitted preprocessing step.

This tutorial builds the regression task from the
[nonlinear regression tutorial](nonlinear_regression.md) as a single, self-contained
`Pipeline`.

## Build the pipeline

Each column gets its own transformer inside a `ColumnTransformer`, and the whole thing feeds
a `Ridge` regressor.

```python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge

from pretab.transformers import (
    NaturalCubicSplineTransformer,
    PLETransformer,
    RBFExpansionTransformer,
)

rng = np.random.default_rng(0)
n = 4000
age = rng.uniform(18, 70, n)
income = rng.normal(60_000, 15_000, n)
tenure = rng.uniform(0, 40, n)
city = rng.choice(["Berlin", "Munich", "Hamburg", "Cologne"], n)

city_effect = pd.Series(city).map(
    {"Berlin": 5.0, "Munich": 8.0, "Hamburg": 3.0, "Cologne": 6.0}
).to_numpy()
y = (
    12 * np.sin(age / 8)
    + 0.00004 * (income - 60_000) ** 2 / 1000
    + np.sqrt(tenure) * 3
    + city_effect
    + rng.normal(0, 2, n)
)
df = pd.DataFrame({"age": age, "income": income, "tenure": tenure, "city": city})

features = ColumnTransformer([
    ("age", NaturalCubicSplineTransformer(output_dim=10), ["age"]),
    ("income", PLETransformer(output_dim=12, task="regression"), ["income"]),
    ("tenure", RBFExpansionTransformer(output_dim=10), ["tenure"]),
    ("city", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])

model = Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])
```

```{note}
`PLETransformer` is supervised, so it uses `y` during `fit` to place its bins. Because it
lives inside the `Pipeline`, `cross_val_score` and `GridSearchCV` pass the training labels
through automatically, so there is nothing extra to wire up.
```

## Cross-validate

Because the pipeline is a standard `sklearn` estimator, `cross_val_score` works directly.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, df, y, cv=5, scoring="r2")
print(f"5-fold R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

```text
5-fold R2: 0.948 +/- 0.005
```

## Tune with GridSearchCV

Transformer hyper-parameters are addressable with the usual `step__param` syntax, so they
can be tuned alongside the model. Here we search the spline width for `age` and the `Ridge`
regularization strength together.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "features__age__output_dim": [6, 10, 16],
    "ridge__alpha": [0.1, 1.0, 10.0],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring="r2")
grid.fit(df, y)

print("best params:", grid.best_params_)
print(f"best CV R2:  {grid.best_score_:.3f}")
```

```text
best params: {'features__age__output_dim': 6, 'ridge__alpha': 0.1}
best CV R2:  0.949
```

Every pretab transformer participates in the search grid just like a native `sklearn` step.

## Configure mixed columns with Preprocessor

Run this example in the
{download}`companion notebook <../../examples/per-feature-configuration.ipynb>`.

`Preprocessor` can select column types and build the preprocessing step for you.
This independent example uses three numerical columns, a Boolean column and a
string category. Keep their dtypes in a `DataFrame`: Booleans are categorical,
while these floating-point columns are numerical.

```python
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab import Preprocessor, RepresentationConfig

frame = pd.DataFrame({
    "age": np.linspace(18.0, 70.0, 80),
    "income": np.linspace(25000.0, 90000.0, 80),
    "tenure": np.linspace(0.0, 30.0, 80),
    "active": [True, False] * 40,
    "city": ["Oslo", "Lima", "Perth", "Oslo"] * 20,
})
target = 0.4 * frame["age"] + 2.0 * frame["active"]
```

### Request local settings

Method strings, mappings and immutable `RepresentationConfig` values can coexist.
The three numerical columns use RBF expansions with different widths. The Boolean
uses integer encoding, and the city encoder drops its first learned category.
Override declaration order does not determine output order.

```python
pre = Preprocessor(
    numerical_method="none",
    output_dim=7,
    target_aware=False,
    placement_strategy="uniform",
    scaling="none",
    output_format="dense",
    random_state=17,
    feature_preprocessing={
        "city": {"method": "one-hot", "params": {"drop": "first"}},
        "tenure": RepresentationConfig(method="rbf", output_dim=5),
        "active": "int",
        "income": {"method": "rbf", "output_dim": 3, "params": {"gamma": 0.25}},
        "age": RepresentationConfig(method="RBF", output_dim=2),
    },
)
requested = pre.get_params(deep=False)
assert requested["output_dim"] == 7
assert requested["feature_preprocessing"]["age"].output_dim == 2
```

### Read the resolved configuration

`get_resolved_config()` normalizes aliases and applies configuration precedence
without fitting. The constructor-wide width remains seven, but each numerical
override supplies its own width. This report does not contain learned categories
or guarantee the realized output width.

```python
resolved = pre.get_resolved_config()
assert resolved["output_dim"] == 7
assert resolved["feature_preprocessing"]["age"]["method"] == "rbf"
assert resolved["feature_preprocessing"]["age"]["output_dim"] == 2
assert resolved["feature_preprocessing"]["income"]["params"]["gamma"] == 0.25
assert not hasattr(pre, "output_dims_")
```

### Inspect the fitted state and width

Fit inside the pipeline so training and prediction use the same preprocessing.
The fitted dimensions account for the learned city vocabulary and its dropped
category, not just the requested RBF widths.

```python
mixed_model = Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])
mixed_model.fit(frame, target)

assert pre.numerical_features_ == ["age", "income", "tenure"]
assert pre.categorical_features_ == ["active", "city"]
expected_dims = {"age": 2, "income": 3, "tenure": 5, "active": 1, "city": 2}
assert pre.output_dims_ == expected_dims
assert list(pre.output_dims_) == list(expected_dims)
assert pre.total_output_dim_ == 13

numerical, categorical, embeddings = pre.get_feature_info(verbose=False)
assert {name: info["dimension"] for name, info in numerical.items()} == {
    "age": 2, "income": 3, "tenure": 5,
}
assert {name: info["dimension"] for name, info in categorical.items()} == {
    "active": 1, "city": 2,
}
assert embeddings == {}
matrix = pre.transform(frame, return_array=True)
assert isinstance(matrix, np.ndarray)
assert matrix.shape == (80, 13)
assert pre.get_feature_names_out().shape == (13,)
predictions = mixed_model.predict(frame)
assert isinstance(predictions, np.ndarray)
assert predictions.shape == (80,)
```

```{important}
Use fitted attributes and inspection methods to size downstream inputs. Numerical
outputs precede categorical outputs, with source-column order within each kind.
Changing constructor parameters after fitting changes the request, not the
learned transformers. Fit again to apply a new request.
```

### Save and restore the fitted preprocessor

`to_spec()` captures the fitted preprocessing state, including the typed local
settings. Pass the JSON-decoded specification to `from_spec()` to restore it
without fitting again. This saves the preprocessing step, not the Ridge model.

```python
spec = json.loads(json.dumps(pre.to_spec(), allow_nan=False))
replayed = Preprocessor.from_spec(spec)

assert replayed.output_dims_ == expected_dims
assert replayed.total_output_dim_ == 13
np.testing.assert_array_equal(replayed.get_feature_names_out(), pre.get_feature_names_out())
np.testing.assert_array_equal(replayed.transform(frame, return_array=True), matrix)
assert replayed.fingerprint_ == pre.fingerprint_
```

For precedence and copy semantics, see
[Structured per-feature configuration](../core_concepts/configuration.md#structured-per-feature-configuration).
For artifact trust and persistence details, see
[Reproducibility](../core_concepts/reproducibility.md).

## When to use which

- **Standalone transformers** (this page) compose inside one `Pipeline` and integrate with
  cross-validation and grid search, with each column's transformer addressable individually
  (`step__param`). Reach for them when you want fine control over one column's transformer.
- **The `Preprocessor`** (the [nonlinear regression tutorial](nonlinear_regression.md)) also
  composes inside a `Pipeline` (it returns a single stacked array by default), reads a
  `DataFrame`, detects feature types automatically, and configures every column from a single
  config. Reach for it when you want per-column strategies without wiring each one by hand.

## Next steps

- Browse every transformer in the [API reference](../api/index.rst).
- Review the method catalogue in the [representations overview](../representations/overview.md).
