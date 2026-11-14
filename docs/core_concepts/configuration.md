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

At `transform`, an array or a list of rows is matched by position, so the training and serving
containers may differ, as in scikit-learn. A `Preprocessor` fitted on a string-labelled
`DataFrame` accepts the same data as an array of the same width (for example `df.to_numpy()`
in NumPy-based serving or an explainer such as `shap.KernelExplainer`). It warns that `X`
does not have valid feature names and re-infers the numeric columns of an object array.
A fit on non-string column labels accepts arrays by position without that warning.
One fitted on an array accepts a `DataFrame` or dict of the same width the same way,
with a warning that `X` has feature names when its labels differ from the fitted synthetic
labels. After a `DataFrame` fit, subsequent frames are still matched by label, even when
the fitted labels are literally `feature_0`, `feature_1`, .... Input matched by position
with a different number of columns raises the usual
`X has 4 features, but Preprocessor is expecting 3 features as input.` error.

## Polars DataFrame input

A `polars.DataFrame` is accepted as input to the `Preprocessor` and `RepresentationSearchCV`
wherever a pandas `DataFrame` is (temporal columns are not supported), so a scikit-learn
`Pipeline` configured with `set_output(transform="polars")` can hand its polars output straight
to the `Preprocessor`. PreTab reads the frame through pandas with the same column names, order
and dtypes, then detects column types as usual: `Float*` columns are numerical, `Int*` /
`UInt*` columns follow the integer cardinality rule (an integer column with nulls becomes
float, as in pandas), and `String` (`Utf8` on older Polars), `Categorical`, `Enum` and `Boolean` columns are
categorical. Polars nulls become missing values, so imputation and `missing_policy` treat them
like `NaN` in a pandas frame. The conversion does not need `pyarrow`.

```python
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

X = pl.DataFrame({"age": [25.0, None, 47.0, 33.0], "income": [3.1e4, 5.2e4, None, 4.4e4]})
y = [0.0, 1.0, 1.0, 0.0]

pipe = Pipeline([("impute", SimpleImputer()), ("pretab", Preprocessor(numerical_method="minmax"))])
pipe.set_output(transform="polars")
pipe.fit_transform(X, y)   # a polars.DataFrame with columns num_age, num_income
```

## Per-feature overrides

Columns rarely want identical treatment. The `feature_preprocessing` dict assigns a strategy
to individual columns and takes precedence over the global defaults for those columns.
Each value can be a method string, a mapping with a `method` key, or a
{class}`~pretab.RepresentationConfig`. You can mix these forms in the same dict.

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

## Structured per-feature configuration

These three examples use the same training data and a scikit-learn regression pipeline.
`income` is deliberately left out of each override map, so it uses the global `"none"`
method and contributes one column.

```python
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab import Preprocessor, RepresentationConfig

frame = pd.DataFrame({
    "age": np.linspace(18.0, 70.0, 40),
    "income": np.linspace(25000.0, 90000.0, 40),
    "city": ["Oslo", "Lima"] * 20,
})
target = 0.4 * frame["age"] + 0.0002 * frame["income"]
common = {
    "numerical_method": "none",
    "output_dim": 7,
    "target_aware": False,
    "placement_strategy": "uniform",
    "scaling": "none",
    "random_state": 17,
}
```

### Method strings

Strings select the representation and retain the applicable constructor-wide controls.
Here `"RBF"` is an alias for `"rbf"`, and `"OHE"` selects one-hot encoding.

```python
legacy = Preprocessor(
    **common,
    feature_preprocessing={"age": "RBF", "city": "OHE"},
)
legacy_model = Pipeline([("pre", legacy), ("model", Ridge())]).fit(frame, target)

assert legacy.output_dims_ == {"age": 7, "income": 1, "city": 2}
assert legacy.transform(frame, return_array=True).shape == (40, 10)
assert legacy_model.predict(frame).shape == (40,)
```

### Mappings with local settings

A mapping adds controls for that feature. `age` requests three RBF outputs and a local
`gamma`; it does not change `income` or the two learned city categories.

```python
mapped = Preprocessor(
    **common,
    feature_preprocessing={
        "age": {"method": "rbf", "output_dim": 3, "params": {"gamma": 0.5}},
        "city": "one-hot",
    },
)
mapped_model = Pipeline([("pre", mapped), ("model", Ridge())]).fit(frame, target)
report = mapped.get_resolved_config()

assert report["output_dim"] == 7
assert report["feature_preprocessing"]["age"]["output_dim"] == 3
assert report["feature_preprocessing"]["age"]["params"]["gamma"] == 0.5
assert mapped.output_dims_ == {"age": 3, "income": 1, "city": 2}
assert mapped.transform(frame, return_array=True).shape == (40, 6)
assert mapped_model.predict(frame).shape == (40,)
```

Family parameters can also appear beside `method`, for example
`{"method": "rbf", "gamma": 0.5}`. Normalization collects them under `params`.
Do not supply the same parameter in both places. Common controls such as `output_dim`
belong at the top level, not inside `params`.

### Immutable values and JSON

`RepresentationConfig` detaches its nested inputs immediately. Its dictionary form can be
stored as JSON, reconstructed, and used with `clone` and `Pipeline`.

```python
params = {"gamma": 0.5}
value = RepresentationConfig(method="RBF", output_dim=3, params=params)
params["gamma"] = 9.0
assert value.params["gamma"] == 0.5

payload = json.loads(json.dumps(value.to_dict(), allow_nan=False))
restored = RepresentationConfig.from_dict(payload)
assert restored == value

typed = Preprocessor(
    **common,
    feature_preprocessing={"age": restored, "city": {"method": "OHE"}},
)
fresh = clone(typed)
assert fresh.get_params() == typed.get_params()
typed_model = Pipeline([("pre", fresh), ("model", Ridge())]).fit(frame, target)

assert fresh.output_dims_ == {"age": 3, "income": 1, "city": 2}
np.testing.assert_array_equal(
    fresh.transform(frame, return_array=True),
    mapped.transform(frame, return_array=True),
)
assert typed_model.predict(frame).shape == (40,)
```

```{important}
Raw mappings passed to `Preprocessor` remain live constructor parameters, as required by
scikit-learn. Edits before fitting affect that fit. Normalization and fitting detach the
representation settings; editing a raw mapping afterward does not refit the learned
transformers. Use `RepresentationConfig` when the request itself should be immutable from
construction onward. Neither form changes an already fitted graph without another fit.
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
`"ple"` when `task="classification"`, for every preset. `task` is matched exactly: any other
value, such as `"Regression"`, raises an `InvalidParamError` from `fit` and
`get_resolved_config()` instead of falling back to the classification preset.

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
Research*. See the [representations overview](../representations/overview.md) for the full
citation list.
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

`get_resolved_config()` returns a detached plain dict for the current request. Outer keys
retain the preset-expanded constructor settings. Each declared `feature_preprocessing`
entry becomes a mapping with its canonical method, applicable inherited controls, local
settings, and `params`. Omitted, inapplicable controls are not added.

```python
pre = Preprocessor(preset="expanded", feature_preprocessing={"age": "bspline"})
pre.get_resolved_config()
```

You can edit the returned dict, including nested containers, without changing the estimator.
The report includes only declared overrides, not a row for every input column. It can be read
before fitting; actual column existence and compatibility with detected feature kinds are
checked at fit time.

```{note}
The report stays a current-request view after fitting or freezing. It is not a record of the
last fit, learned knots, or realized output widths. Changing a live raw request can change
the report while the fitted transformer continues to produce its original output. Use
`output_dims_`, `total_output_dim_`, and `get_feature_info()` for fitted results.
```

## How the layers combine

The resolution order is deterministic. Later layers win.

1. Library defaults.
2. A `preset`, if given.
3. Explicit constructor arguments (`numerical_method`, `output_dim`, and so on).
4. Per-column `feature_preprocessing` entries, for the columns they name.

Within a structured entry, inheritance is per field, not an all-or-nothing replacement.

| Setting             | Effective choice                                                                                                                                     |
| :------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| Method              | The local `method`, otherwise the global method for the detected kind.                                                                               |
| Common controls     | Explicit local values win. Omitted values inherit constructor settings only when the method supports them; other defaults belong to the transformer. |
| Family `params`     | Local parameters win over applicable inherited `degree`, `task`, or an explicit `random_state`. Other family defaults are left to the transformer.   |
| Fixed local width   | An explicit local `output_dim` disables inherited adaptive sizing unless that entry also explicitly sets `adaptive=True`.                            |
| Adaptive bounds     | Omitted bounds inherit only when adaptive sizing is active. Explicit bounds with fixed sizing are rejected; unsupported controls are not forwarded.  |
| Run-wide processing | Imputation, scaling, missing indicators, and output formatting remain constructor-wide settings, not representation parameters.                      |

Omission is different from an explicit value. For example, a local
`params={"random_state": None}` overrides an inherited seed where the method accepts it.
Passing `None` is valid only for options that accept it; it does not mean "inherit".

```{warning}
Value construction validates portable data and field types. `get_resolved_config()` and
`fit` also validate method names, supported options, and target/placement combinations.
Unknown keys and invalid methods raise typed errors with the feature's configuration path.
A local override does not make an invalid constructor-wide task or placement combination
valid. Actual input-column and feature-kind checks still require `fit`.
```

## Using existing string configurations

No syntax change is required for supported legacy configuration. Existing method strings,
aliases, global parameters, and per-column string overrides remain supported. Adopt mappings
or typed values only where you need local settings; other entries can stay strings.

Equivalent fitted outputs do not imply that differently written requests have identical
fingerprints. Keep the requested configuration and dependency versions with your experiment;
see [Reproducibility](reproducibility.md) for fitted artifacts and upgrade precautions.

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
