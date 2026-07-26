# End-to-end example

pretab is most useful as the *feature layer* in front of a model. This walkthrough builds
the same small regression task twice: once with plain scaling and once with pretab, using
the **same linear model** both times. The only thing that changes is how the raw columns
are turned into features, which makes the effect of pretab easy to see.

## The dataset

We simulate a tabular dataset with three numeric columns and one categorical column.

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

# The target depends on each feature in a *nonlinear* way.
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
df.head()
```

```text
         age        income     tenure     city
0  51.122008  38220.981430  31.495899   Munich
1  32.028909  61219.946532  18.424959  Cologne
2  20.130623  49018.509912  28.795777   Munich
3  18.859437  42292.103104  22.590428   Munich
4  60.290052  40930.830263  39.569339  Cologne
```

The target curves with `age`, bends quadratically with `income`, and flattens out with
`tenure`. A plain linear model only sees a single straight-line term per column, so it has
no way to represent these shapes. That is exactly the gap pretab fills.

```{warning}
Fit every transformer on the **training split only**, then apply it to the test split with
`transform`. Supervised expansions such as PLE and RBF read `y` while fitting, so fitting
them on the full dataset would leak test information and inflate your scores.
```

## Baseline: scaling + Ridge

First, a conventional pipeline: scale the numeric columns, one-hot the categorical one, and
fit a `Ridge` regressor.

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

With one straight-line term per numeric column, `Ridge` can only fit a global slope. It
misses every curve in the target, and the $R^2$ of `0.124` is barely better than predicting
the mean.

## With pretab

Now swap the scaler for a `Preprocessor` that gives each column an expressive basis. It uses
a B-spline for `age`, piecewise-linear encoding (PLE) for `income`, radial basis functions
for `tenure`, and one-hot for `city`. Everything else stays the same.

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

X_tr = pre.fit_transform(X_train, y_train, return_array=True)
X_te = pre.transform(X_test, return_array=True)

model = Ridge(alpha=1.0).fit(X_tr, y_train)
pred = model.predict(X_te)

print(f"features: {X_tr.shape[1]}")
print(f"R2:  {r2_score(y_test, pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
```

```text
features: 41
R2:  0.968
MAE: 2.16
```

The data and the `Ridge` model are unchanged, but the expressive features let it capture the
nonlinear structure. The $R^2$ jumps from `0.124` to `0.968` and the mean absolute error
drops from `11.20` to `2.16`.

```{tip}
`Preprocessor.transform` returns a **dict of feature blocks** by default. When you feed a
plain estimator, call it yourself with `return_array=True` to get a single stacked matrix,
then hand the arrays to the model. If you would rather compose everything inside one
`sklearn` `Pipeline`, use the standalone transformers instead. See the
[sklearn Pipeline tutorial](../tutorials/sklearn_pipeline.md).
```

## What actually changed

The `Preprocessor` expands four raw columns into 41 features. Inspect the resolved layout
with `get_feature_info`:

```python
pre.get_feature_info()
```

```text
feature  kind         pipeline                        dim   cats
----------------------------------------------------------------
age      numerical    imputer -> minmax -> bspline     13      -
income   numerical    imputer -> minmax -> ple         12      -
tenure   numerical    imputer -> minmax -> rbf         12      -
city     categorical  imputer -> onehot -> to_float     4      4
```

Each numeric column is imputed, scaled, then expanded into a basis that a linear model can
weight independently: 13 spline coefficients for `age`, 12 PLE bins for `income`, and 12 RBF
bumps for `tenure`, while `city` becomes four one-hot columns. The model is unchanged, and
only the representation improved.

## Next steps

- Do the same for a classifier in the [classification tutorial](../tutorials/classification.md).
- Wire pretab transformers into a full `sklearn` `Pipeline` with cross-validation and
  grid search in the [sklearn Pipeline tutorial](../tutorials/sklearn_pipeline.md).
- Review every strategy string in the [User Guide](../user_guide/preprocessing.md).
