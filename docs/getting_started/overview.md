# Overview

PreTab is a representation and basis-expansion framework for tabular data. It takes raw
numerical and categorical columns and turns them into model-ready features that expose
structure a plain estimator cannot see on its own. Every strategy speaks the standard
scikit-learn `fit` / `transform` API, so PreTab drops into the pipelines and tooling you
already use.

## The problem PreTab solves

Most tabular models receive one straight-line term per numerical column. A linear model,
a logistic regression, or a plain additive model can only weight that single slope, so any
curve, threshold, saturation, or periodic pattern in the data is invisible to it. The usual
response is to hand-craft features: bucket an age column, add a squared income term, encode
the hour of day as a pair of sine and cosine values. That work is repetitive, easy to get
wrong, and rarely reproducible.

PreTab makes those representations first-class. Instead of writing feature code by hand you
declare intent once, for example "expand `age` with a spline, encode `income` with
piecewise-linear bins, treat `hour` as periodic", and PreTab fits the corresponding basis
per column, tracks where every output column came from, and hands back a clean matrix.

The examples on this page use the same small regression dataset.

```python
import numpy as np
import pandas as pd

from pretab import Preprocessor

rng = np.random.default_rng(0)
n_rows = 64
df = pd.DataFrame(
  {
    "age": rng.uniform(18, 65, size=n_rows),
    "income": rng.uniform(25_000, 90_000, size=n_rows),
    "hour": rng.integers(0, 24, size=n_rows),
    "city": rng.choice(["Oslo", "Rome"], size=n_rows),
  }
)
y = np.sin(df["age"] / 10) + np.log(df["income"]) + np.cos(2 * np.pi * df["hour"] / 24)

pre = Preprocessor(
  task="regression",
  random_state=0,
  feature_preprocessing={
    "age": "naturalspline",
    "income": "ple",
    "hour": "fourier",
    "city": "one-hot",
  },
)
X = pre.fit_transform(df, y)
assert X.shape == (len(df), len(pre.get_feature_names_out()))
```

## Two ways to use it

`Preprocessor` and standalone transformers share many of the same representations. Use
`Preprocessor` to configure a whole table, or compose standalone transformers when you need
to control each pipeline step or use a standalone-only method.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} The high-level `Preprocessor`
Detects column types from a `DataFrame`, applies a strategy per column, and returns
model-ready blocks or a single stacked array. Reach for it when you want per-column
strategies from one config.
:::

:::{grid-item-card} Standalone transformers
Import a transformer from `pretab.transformers` and use it on its own or inside a
`Pipeline` or `ColumnTransformer`.
:::

::::

**Reach for the `Preprocessor` when** you start from a `DataFrame` and want automatic
feature-type detection, want to configure many columns from one place (global
`numerical_method` / `categorical_method` defaults or a per-column `feature_preprocessing`
map), or want the framework services that live at this level: feature lineage,
output-format control, missing-value policy, output budgets, serialization, and a
reproducibility fingerprint.

The fitted `Preprocessor` above can also return the same four features as separate blocks.

```python
blocks = pre.transform(df, return_array=False)
assert set(blocks) == {"num_age", "num_income", "num_hour", "cat_city"}
assert sum(block.shape[1] for block in blocks.values()) == X.shape[1]
```

**Reach for standalone transformers** when you want a single estimator object that composes
cleanly inside one `Pipeline`, rely on scikit-learn model selection (`cross_val_score`,
`GridSearchCV`, `step__param` hyperparameter addressing), or need fine control over one
column's transformer and its parameters.

This pipeline uses just `age` and `income`. It fits their representations and Ridge on the
training split, then transforms and predicts on held-out rows.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pretab.transformers import NaturalCubicSplineTransformer, PLETransformer

features = ColumnTransformer([
    ("age", NaturalCubicSplineTransformer(output_dim=10), ["age"]),
  ("income", PLETransformer(output_dim=12, task="regression", random_state=0), ["income"]),
])
model = Pipeline([("features", features), ("ridge", Ridge())])
df_train, df_test, y_train, _ = train_test_split(df, y, random_state=0)
model.fit(df_train, y_train)
assert model.predict(df_test).shape == (len(df_test),)
```

```{note}
The `Preprocessor` returns a single stacked array by default, so it drops directly into a
plain `Pipeline`/`ColumnTransformer` like any other scikit-learn transformer, and composes the
same way as the standalone transformers above. Pass `output_structure="blocks"` (or
`return_array=False` for a single call) when you want the dict-of-feature-blocks form
instead, for inspection or per-block downstream heads.
```

The choice is not exclusive: a `Preprocessor` can live inside a larger `Pipeline`, and
standalone transformers can preprocess columns you then hand to a `Preprocessor`. Pick the
surface that keeps the intent of your code clearest.

```{note}
The tensor-product spline, thin-plate spline, random Fourier features, and Nyström map model
several columns jointly. Use `Preprocessor(feature_groups=...)` for tensor-product or
thin-plate splines, or fit their standalone transformers directly. Random Fourier features
and Nyström remain standalone-only. None is a per-column `numerical_method`. See
[Multivariate features](../tutorials/multivariate_features.md).
```

## How this compares to scikit-learn's preprocessing transformers

PreTab is not a competitor to scikit-learn. Every transformer subclasses `BaseEstimator` and
`TransformerMixin` and drops into the same `Pipeline` and `ColumnTransformer` you already use.
The real question is what PreTab adds where scope overlaps with scikit-learn's own
`SplineTransformer`, `KBinsDiscretizer`, `PolynomialFeatures`, and `TargetEncoder`.

| Capability               | scikit-learn                                                                     | PreTab                                                                                                                                                                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Knot placement           | `SplineTransformer` places `"uniform"` or `"quantile"` knots during `fit`        | Configured via `output_dim` (how many basis functions you want); PreTab derives the knot count from it and places them with `placement_strategy`, optionally target-aware (a CART or LightGBM model places them where the target changes fastest) |
| How many basis functions | You pick a fixed count                                                           | `adaptive=True` searches a width in `[min_output_dim, max_output_dim]` from the data                                                                                                                                                              |
| Target-aware fitting     | `TargetEncoder.fit_transform` cross-fits; `fit` followed by `transform` does not | `CrossFittedTransformer` makes out-of-fold training features when the output width is consistent across folds. Direct standalone fits that use `y` can emit `LeakageWarning`.                                                                     |
| Feature provenance       | `get_feature_names_out()` returns names only                                     | A typed `RepresentationSpec` per transformer plus a `FeatureLineage` record per output column (family, component, target usage)                                                                                                                   |
| Persistence              | `pickle` / `joblib`, which execute arbitrary code on load                        | `to_spec()` / `from_spec()`: a versioned JSON spec for supported fitted state, plus a stable `fingerprint_`; restore trusted specs in the same environment                                                                                        |
| Choosing per column      | Hand-assemble a `ColumnTransformer` yourself                                     | One `Preprocessor(feature_preprocessing={...})`, validated against a capability registry so incompatible combinations (a required-target method without `y`, for example) raise a typed error at fit time                                         |

```{warning}
A fit inside `Preprocessor` or a scikit-learn `Pipeline` does not emit the standalone
`LeakageWarning`. That does not make the fit leakage-safe by itself. For evaluation, fit
the whole pipeline on training rows only, or cross-validate the unfitted pipeline. See
[Target awareness](../core_concepts/target_awareness.md).
```

```{note}
Piecewise-linear encoding (`ple`) and the neural-style basis maps (`rbf`, `relu`, `sigmoid`,
`tanh`, deterministic `fourier`) have no scikit-learn equivalent. `rff` and `nystroem` are thin
wrappers around scikit-learn's own `RBFSampler` and `Nystroem`. Both operate on whole
feature blocks and are available as standalone transformers only.
```

## When to reach for PreTab

PreTab is a good fit when any of the following is true.

- You want an existing model, from a simple linear model (Ridge, logistic regression, a
  GAM) to a deep learning architecture, to capture non-linear structure through the input
  representation rather than through added model complexity.
- You need **expressive numerical representations** such as splines, radial basis maps,
  Fourier features, or piecewise-linear encoding without wiring each one by hand.
- You want **per-column control** over preprocessing from a single configuration object.
- You care about **reproducibility and inspection**: knowing exactly which input produced
  each output column, serializing a fitted representation, and getting a stable fingerprint.
- You are **researching representations** and want a common, typed intermediate form
  (`RepresentationSpec` plus feature lineage) shared across every family.

```{tip}
The right representation depends on what sits downstream: linear and additive models gain
the most, tree ensembles (gradient boosting, random forests) usually gain the least since
they already partition each feature on their own, and neural networks benefit from methods
like PLE and learned embeddings. See
[Choosing a method](../representations/choosing_a_method.md#start-from-the-model) for the
full breakdown.
```

## What PreTab is not

Knowing the boundaries is as useful as knowing the features. PreTab deliberately does not
try to be an everything-library.

- **Not a modelling library.** PreTab supplies representations, not predictors. Use your own
  estimator downstream. [`RepresentationSearchCV`](../api/search_and_cross_fitting.rst)
  fits that estimator while comparing candidate representations by cross-validation.
- **Not a time-series toolkit.** Generic lag and rolling-window utilities were removed on
  purpose. PreTab keeps the periodic encoding that expresses cyclic structure (hour, day,
  month) but leaves sequence modelling to dedicated libraries.
- **Not a data-cleaning suite.** It offers principled, centrally-defined policies for
  missing values, constant columns, and out-of-range inputs, but it is not a substitute for
  domain-specific data validation.
- **Not a guaranteed accuracy win.** An expressive basis in front of a model that is already
  flexible, or on a feature with no non-linear signal, can add columns without adding value.
  The [failure modes](../representations/choosing_a_method.md#when-basis-expansion-does-not-help)
  section is explicit about where it does not help.

## What's new in PreTab 1.1

- **More control per feature.** Set a [width or method options](../core_concepts/configuration.md)
  for individual columns, or use a [`FeatureGroup`](../tutorials/multivariate_features.md)
  to fit a joint spline across numerical columns.
- **Inspect what was fitted.** Read [actual widths and output lineage](../core_concepts/outputs_and_inspection.md)
  and use [`PenaltySpec`](../representations/spline_expansions.md) to align supported
  spline penalties with fitted basis coefficients.
- **Use Polars directly.** Pass an [eager `polars.DataFrame`](../core_concepts/configuration.md#polars-dataframe-input)
  to `Preprocessor` without PyArrow.
- **Export strict JSON.** [Schema-v1 specs](../core_concepts/reproducibility.md) can
  record fit-time settings and output layout. Supported 1.0.0 specs still load in 1.1.

**Upgrading:** PreTab 1.1 supports Python 3.11 through 3.13. On Python 3.10,
stay on PreTab 1.0.0 or [upgrade Python](installation.md#python-310-migration).

```{note}
Use fitted attributes and methods in code rather than parsing verbose inspection
text, which may change. Only the thin-plate spline's legacy penalty matrix is
experimental, not its transform. The matrix is not guaranteed positive semidefinite.
```

## Where to go next

- [Installation](installation.md) sets up PreTab and its optional extras.
- [Quickstart](quickstart.md) fits your first `Preprocessor` in a few minutes.
- [Core concepts](../core_concepts/feature_representation.md) explains the ideas that run
  through the whole library.
- [Representations](../representations/overview.md) is the catalogue of every method.
