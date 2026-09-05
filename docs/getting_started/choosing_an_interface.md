# Choosing an interface

PreTab exposes the same representations through two surfaces: the high-level `Preprocessor`
and the standalone transformers. They share the same underlying code, so the choice is about
ergonomics, not capability. This page helps you pick.

## The two surfaces at a glance

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} `Preprocessor`
Reads a `DataFrame`, detects numerical and categorical columns, and applies a strategy per
column from a single configuration object. Returns a single stacked array by default.
:::

:::{grid-item-card} Standalone transformers
Plain scikit-learn transformers you import from `pretab.transformers`. Each one returns a
NumPy array and slots into a `Pipeline`, `ColumnTransformer`, or any scikit-learn utility.
:::

::::

## Reach for the `Preprocessor` when

- You start from a `DataFrame` and want **automatic feature-type detection** rather than
  wiring every column by hand.
- You want to configure **many columns from one place**, either with global
  `numerical_method` / `categorical_method` defaults or a per-column `feature_preprocessing`
  map.
- You want the **framework services** that live at this level: feature lineage, output-format
  control, missing-value policy, output budgets, serialization, and a reproducibility
  fingerprint.

```python
from pretab import Preprocessor

pre = Preprocessor(feature_preprocessing={
    "age": "naturalspline",
    "income": "ple",
    "city": "one-hot",
})
X = pre.fit_transform(df, y)   # one stacked array, or output_structure="blocks" for a dict
```

## Reach for standalone transformers when

- You want a **single estimator object** that composes cleanly inside one `Pipeline`.
- You rely on **scikit-learn model selection**: `cross_val_score`, `GridSearchCV`, and
  `step__param` hyperparameter addressing all work out of the box.
- You need **fine control** over one column's transformer and its parameters.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from pretab.transformers import NaturalCubicSplineTransformer, PLETransformer

features = ColumnTransformer([
    ("age", NaturalCubicSplineTransformer(output_dim=10), ["age"]),
    ("income", PLETransformer(output_dim=12), ["income"]),
])
model = Pipeline([("features", features), ("ridge", Ridge())])
```

```{note}
The `Preprocessor` returns a single stacked array by default, so it drops directly into a
plain `Pipeline`/`ColumnTransformer` like any other scikit-learn transformer. Pass
`output_structure="blocks"` (or `return_array=False` for a single call) when you want the
dict-of-feature-blocks form instead, for inspection or per-block downstream heads.
```

A `Preprocessor` composes the same way as the standalone transformers above:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from pretab import Preprocessor

model = Pipeline([
    ("pretab", Preprocessor(feature_preprocessing={"age": "naturalspline", "income": "ple"})),
    ("ridge", Ridge()),
])
model.fit(df, y)
model.predict(df)
```

## A note on multivariate methods

The tensor-product spline, thin-plate spline, random Fourier features, and Nyström map model
several columns **jointly**. They are standalone-only and are not selectable per column
through `Preprocessor(numerical_method=...)`. Use them directly as transformers over a block
of columns. See [Multivariate features](../tutorials/multivariate_features.md).

## They interoperate

The choice is not exclusive. A `Preprocessor` can live inside a larger `Pipeline`, and
standalone transformers can preprocess columns you then hand to a `Preprocessor`. Pick the
surface that keeps the intent of your code clearest.

## Where to go next

- [Configuration](../core_concepts/configuration.md) documents every `Preprocessor` knob.
- [scikit-learn pipelines](../tutorials/sklearn_pipeline.md) shows the standalone route with
  cross-validation and grid search.
- [Representations](../representations/overview.md) is the full method catalogue.
