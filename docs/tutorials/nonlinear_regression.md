# Nonlinear regression

PreTab is most useful as the feature layer in front of a model. This walkthrough builds the
same small regression task twice: once with plain scaling and once with PreTab, using the
**same linear model** both times. Only the representation changes, which makes the effect
easy to see.

## The dataset

We simulate a tabular dataset with three numeric columns and one categorical column, where the
target depends on each feature in a nonlinear way.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 4000

age = rng.uniform(18, 70, n)
income = rng.normal(60_000, 15_000, n)
tenure = rng.uniform(0, 40, n)
city = rng.choice(["Berlin", "Munich", "Hamburg", "Cologne"], n)

city_effect = pd.Series(city).map(
    {"Berlin": 5.0, "Munich": 8.0, "Hamburg": 3.0, "Cologne": 6.0}
).to_numpy()
target = (
    12 * np.sin(age / 8)                       # wave in age
    + 0.00004 * (income - 60_000) ** 2 / 1000  # quadratic in income
    + np.sqrt(tenure) * 3                       # diminishing returns on tenure
    + city_effect                               # per-city offset
    + rng.normal(0, 2, n)                       # noise
)

df = pd.DataFrame({"age": age, "income": income, "tenure": tenure, "city": city})

X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.25, random_state=42
)
```

The target curves with `age`, bends quadratically with `income`, and flattens out with
`tenure`. A plain linear model only sees a single straight-line term per column, so it has no
way to represent these shapes. That is exactly the gap PreTab fills.

```{warning}
Fit every transformer on the **training split only**, then apply it to the test split with
`transform`. Supervised expansions such as PLE read `y` while fitting, so fitting on the full
dataset would leak test information and inflate your scores. See
[Target awareness](../core_concepts/target_awareness.md).
```

## Baseline: scaling and Ridge

First, a conventional pipeline: scale the numeric columns, one-hot the categorical one, and fit
a `Ridge` regressor.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error

baseline = ColumnTransformer([
    ("num", MinMaxScaler(), ["age", "income", "tenure"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
])

X_tr = baseline.fit_transform(X_train)
X_te = baseline.transform(X_test)

model = Ridge(alpha=1.0).fit(X_tr, y_train)
pred = model.predict(X_te)

print(f"features: {X_tr.shape[1]}")
print(f"R2:  {r2_score(y_test, pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
```

```text
features: 7
R2:  0.124
MAE: 11.20
```

With one straight-line term per numeric column, `Ridge` can only fit a global slope. It misses
every curve in the target, and the $R^2$ of `0.124` is barely better than predicting the mean.

## With PreTab

Now swap the scaler for a `Preprocessor` that gives each column an expressive basis: a B-spline
for `age`, piecewise-linear encoding for `income`, radial basis functions for `tenure`, and
one-hot for `city`. Everything else stays the same.

```python
from pretab import Preprocessor

pre = Preprocessor(
    feature_preprocessing={
        "age": "bspline",
        "income": "ple",
        "tenure": "rbf",
        "city": "one-hot",
    },
    task="regression",
    output_dim=12,
)

X_tr = pre.fit_transform(X_train, y_train)
X_te = pre.transform(X_test)

model = Ridge(alpha=1.0).fit(X_tr, y_train)
pred = model.predict(X_te)

print(f"features: {X_tr.shape[1]}")
print(f"R2:  {r2_score(y_test, pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
```

```text
features: 40
R2:  0.983
MAE: 1.78
```

The data and the `Ridge` model are unchanged, but the expressive features let it capture the
nonlinear structure. The $R^2$ jumps from `0.124` to `0.983` and the mean absolute error drops
from `11.20` to `1.78`.

```{tip}
`Preprocessor.transform` returns a single stacked array by default, so it drops straight into
a plain scikit-learn estimator or `Pipeline`. Pass `output_structure="blocks"` (or
`return_array=False` for a single call) for the dict-of-feature-blocks form instead. See the
[sklearn pipeline tutorial](sklearn_pipeline.md) for wiring each column's transformer by hand.
```

## What actually changed

The `Preprocessor` expands four raw columns into 40 features. Inspect the resolved layout with
`get_feature_info`:

```python
pre.get_feature_info()
```

```text
feature  kind         pipeline                        dim   cats
----------------------------------------------------------------
age      numerical    imputer -> minmax -> bspline     12      -
income   numerical    imputer -> minmax -> ple         12      -
tenure   numerical    imputer -> minmax -> rbf         12      -
city     categorical  imputer -> onehot -> to_float     4      4
```

Each numeric column is imputed, scaled, then expanded into a basis the linear model can weight
independently: 12 spline coefficients for `age`, 12 PLE bins for `income`, and 12 RBF bumps for
`tenure`, while `city` becomes four one-hot columns. To trace any single output column back to
its source, use [feature lineage](../core_concepts/outputs_and_inspection.md).

## Where to go next

- Do the same for a classifier in the
  [leakage-safe classification tutorial](target_aware_classification.md).
- Wire PreTab transformers into a full `Pipeline` with cross-validation and grid search in the
  [sklearn pipeline tutorial](sklearn_pipeline.md).
- Measure one representation against another in
  [comparing representations](comparing_representations.md).
