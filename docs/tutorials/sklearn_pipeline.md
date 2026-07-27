# Inside an sklearn Pipeline

The high-level `Preprocessor` returns a dict by default, so it is used as an explicit
feature-building step (call it with `return_array=True`, then fit the model on the arrays).
The **standalone transformers**, on the other hand, return plain arrays and follow the
`sklearn` API exactly, so they drop straight into a `ColumnTransformer` and `Pipeline`, and
work with `cross_val_score`, `GridSearchCV`, and every other `sklearn` utility.

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
5-fold R2: 0.920 +/- 0.007
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
best CV R2:  0.921
```

Every pretab transformer participates in the search grid just like a native `sklearn` step.

## When to use which

- **Standalone transformers** (this page) compose inside one `Pipeline` and integrate with
  cross-validation and grid search. Reach for them when you want a single estimator object.
- **The `Preprocessor`** (the [nonlinear regression tutorial](nonlinear_regression.md)) reads
  a `DataFrame`, detects feature types automatically, and configures every column from a
  single config. Reach for it when you want per-column strategies without wiring each one by
  hand.

## Next steps

- Browse every transformer in the [API reference](../api/index.rst).
- Review the method catalogue in the [representations overview](../representations/overview.md).
