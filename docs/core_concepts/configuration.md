# Configuration

The `Preprocessor` is configured through a small, predictable set of parameters. This page
covers the four ways to express intent: global defaults, per-feature overrides, presets, and
reading back the resolved configuration. The mechanics of width and placement live in
[Resolution and placement](resolution_and_placement.md), and target usage in
[Target awareness](target_awareness.md).

## Global defaults

The simplest configuration sets one strategy for every numerical column and one for every
categorical column.

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_method="ple",     # applied to every numerical column
    categorical_method="int",   # applied to every categorical column
)
```

The defaults are `numerical_method="ple"` and `categorical_method="int"`. The full list of
strategy strings is in the [representation comparison](../representations/comparison_table.md).

## NumPy array input

`Preprocessor` also accepts a plain `numpy.ndarray`, not just a `DataFrame`. Columns are
named `feature_0`, `feature_1`, ... in position order, then detected as numerical or
categorical exactly as they would be for a `DataFrame`.

```python
import numpy as np
from pretab import Preprocessor

X = np.random.default_rng(0).normal(size=(100, 3))
y = np.random.default_rng(0).normal(size=100)

pre = Preprocessor(numerical_method="ple").fit(X, y)
pre.numerical_features_
```

```text
['feature_0', 'feature_1', 'feature_2']
```

```{tip}
`feature_preprocessing` works the same way on array input: target the synthetic name, for
example `{"feature_0": "rbf"}`. Check `numerical_features_` / `categorical_features_` after
`fit` (or `get_feature_info()`) to confirm the names PreTab assigned before writing the
overrides, rather than guessing the column order.
```

## Per-feature overrides

Columns rarely want identical treatment. The `feature_preprocessing` dict assigns a strategy
to individual columns and takes precedence over the global defaults for those columns.

```python
pre = Preprocessor(
    numerical_method="ple",              # default for numerical columns not listed
    feature_preprocessing={
        "age": "naturalspline",
        "income": "rbf",
        "city": "one-hot",
    },
)
```

```{note}
A per-feature entry is resolved in the correct namespace for its detected column kind. You do
not need to state whether a column is numerical or categorical; PreTab already knows from
feature-type detection.
```

### Columns not listed in `feature_preprocessing`

`feature_preprocessing` only overrides the columns it names. Any numerical column left out
falls back to `numerical_method`, and any categorical column left out falls back to
`categorical_method`. This is real method resolution, not just a logging detail: the fallback
column is fit with the global default's transformer, exactly as if you had listed it yourself.

```python
pre = Preprocessor(
    numerical_method="bspline",          # applies to every numerical column not listed below
    feature_preprocessing={
        "income": "rbf",                 # overrides the default for this one column
        "city": "one-hot",
    },
).fit(df, y)
```

| Column   | Kind        | Listed in `feature_preprocessing`? | Resolved method                     |
| -------- | ----------- | ---------------------------------- | ----------------------------------- |
| `age`    | numerical   | no                                 | `bspline` (from `numerical_method`) |
| `income` | numerical   | yes, `"rbf"`                       | `rbf`                               |
| `city`   | categorical | yes, `"one-hot"`                   | `one-hot`                           |
| `region` | categorical | no                                 | `int` (from `categorical_method`)   |

```{tip}
Don't infer the resolved method per column from the constructor arguments alone. Call
`get_feature_info(verbose=True)` after `fit` for the definitive per-column table, or set
`verbose=2` (or higher) on the `Preprocessor` to log the same table at fit time. See
[Fit-time logging](outputs_and_inspection.md#fit-time-logging).
```

## Presets

Presets are transparent, named bundles of parameters for common intents. They set the same
knobs you could set by hand, so nothing is hidden, and each one resolves to a fixed,
documented set of values. `numerical_method` is the one exception: every preset resolves it
from `task` instead of a fixed value, since a spline basis and piecewise-linear encoding suit
regression and classification differently.

| Preset       | `numerical_method` | `categorical_method` | `output_dim` | `adaptive` | `min_output_dim` | `max_output_dim` |
| ------------ | ------------------ | -------------------- | ------------ | ---------- | ---------------- | ---------------- |
| `"standard"` | task-dependent     | `"int"`              | `7`          | `False`    | -                | -                |
| `"expanded"` | task-dependent     | `"one-hot"`          | `10`         | `False`    | -                | -                |
| `"adaptive"` | task-dependent     | `"int"`              | -            | `True`     | `7`              | `15`             |

`numerical_method` resolves to `"bspline"` when `task="regression"` (the default) and to
`"ple"` when `task="classification"`, for every preset:

```python
standard_regression = Preprocessor(preset="standard", task="regression")
standard_classification = Preprocessor(preset="standard", task="classification")

standard_regression.get_resolved_config()["numerical_method"]        # "bspline"
standard_classification.get_resolved_config()["numerical_method"]    # "ple"
```

```{note}
The regression/classification split follows Kumar et al. (2026),
["From Uniform to Learned Knots: A Study of Spline-Based Numerical Encodings for Tabular Deep
Learning"](https://openreview.net/pdf?id=str7wQt9Qc), *Transactions on Machine Learning
Research*. See the [representation references](../representations/references.md) for the full
citation.
```

```python
standard = Preprocessor(preset="standard")
expanded = Preprocessor(preset="expanded")

standard.get_resolved_config()["categorical_method"]   # "int": compact integer codes
expanded.get_resolved_config()["categorical_method"]    # "one-hot": one column per category
expanded.get_resolved_config()["output_dim"]             # 10: wider than "standard"
```

So `"standard"` is the balanced default (`output_dim=7`, integer-coded categoricals),
`"expanded"` widens the representation to `output_dim=10` and one-hot-encodes categoricals
instead, and `"adaptive"` lets each feature pick its own width between `min_output_dim=7` and
`max_output_dim=15` rather than using a fixed `output_dim`.

```{tip}
A preset is a starting point, not a lock. Any parameter you pass alongside a preset overrides
the preset's value for that knob, including `numerical_method` itself, for example
`Preprocessor(preset="expanded", numerical_method="rbf")`.
```

## Reading the resolved configuration

Because global defaults, per-feature overrides, and presets interact, PreTab lets you read
back exactly what will be used. `get_resolved_config()` returns the fully resolved settings
as a plain dict.

```python
pre = Preprocessor(preset="expanded", feature_preprocessing={"age": "bspline"})
pre.get_resolved_config()
```

This is the authoritative answer to "what did my configuration actually become", and it is
useful in tests and reproducible experiments.

## How the layers combine

The resolution order is deterministic. Later layers win.

1. Library defaults.
2. A `preset`, if given.
3. Explicit constructor arguments (`numerical_method`, `output_dim`, and so on).
4. Per-column `feature_preprocessing` entries, for the columns they name.

```{warning}
Configuration is validated at `fit` time, not silently coerced. An invalid combination, such
as a method that requires the target used with `target_aware=False`, raises a typed error.
This is intentional: it surfaces mistakes early rather than producing a quietly wrong
representation.
```

## Key parameters at a glance

The parameters below are the ones you reach for most. Each links to the page that explains it
in depth.

| Parameter                                                                 | Default                                | Covered in                                                           |
| ------------------------------------------------------------------------- | -------------------------------------- | -------------------------------------------------------------------- |
| `numerical_method`, `categorical_method`                                  | `"ple"`, `"int"`                       | this page                                                            |
| `feature_preprocessing`                                                   | `None`                                 | this page                                                            |
| `output_dim`                                                              | `7`                                    | [Resolution and placement](resolution_and_placement.md)              |
| `adaptive`, `min_output_dim`, `max_output_dim`                            | `False`, `5`, `10`                     | [Resolution and placement](resolution_and_placement.md)              |
| `target_aware`, `placement_strategy`                                      | `True`, `"cart"`                       | [Target awareness](target_awareness.md)                              |
| `numerical_imputation`, `categorical_imputation`, `add_missing_indicator` | `"median"`, `"most_frequent"`, `False` | [Missing values](missing_values.md)                                  |
| `output_format`, `dtype`                                                  | `"dense"`, `None`                      | [Outputs and inspection](outputs_and_inspection.md)                  |
| `output_structure`                                                        | `"matrix"`                             | [Outputs and inspection](outputs_and_inspection.md)                  |
| `verbose`                                                                 | `0`                                    | [Outputs and inspection](outputs_and_inspection.md#fit-time-logging) |
| `random_state`                                                            | `None`                                 | [Reproducibility](reproducibility.md)                                |

## Where to go next

- [Resolution and placement](resolution_and_placement.md) for width and location.
- [Target awareness](target_awareness.md) for supervised placement.
- [Representations](../representations/overview.md) for what each method does.
