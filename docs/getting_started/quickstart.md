# Quickstart

This page fits your first representation in a few minutes. It covers the two ways to use
PreTab: the high-level `Preprocessor` that builds a full pipeline from a config, and the
individual transformers that behave like any other scikit-learn step.

## Install

```bash
pip install pretab
```

See [Installation](installation.md) for optional extras such as language embeddings and
LightGBM-based placement.

## Fit a `Preprocessor`

The `Preprocessor` inspects a `DataFrame`, decides which columns are numerical and which are
categorical, and applies a strategy per column. It returns a single stacked array by default,
or a dict of per-feature blocks on request.

```python
import numpy as np
import pandas as pd

from pretab import Preprocessor

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "age": rng.integers(18, 65, size=200),
    "income": rng.normal(60_000, 15_000, size=200).astype(int),
    "experience": rng.integers(0, 40, size=200),
    "job": rng.choice(["nurse", "engineer", "scientist", "teacher"], size=200),
    "city": rng.choice(["Berlin", "Munich", "Hamburg", "Cologne"], size=200),
})
y = np.sin(df["age"] / 10) + df["income"] / 1e5 + rng.normal(0, 0.1, size=200)

config = {
    "age": "ple",              # supervised piecewise-linear encoding
    "income": "rbf",           # radial basis feature map
    "experience": "naturalspline",
    "job": "one-hot",
    "city": "int",             # integer (ordinal) codes
}
pre = Preprocessor(feature_preprocessing=config, task="regression", random_state=0)

# Fit and transform into a single stacked array
X = pre.fit_transform(df, y)
assert X.shape == (200, 26)
print(X.shape)
```

```text
(200, 26)
```

```{tip}
When no per-feature config is given, the `Preprocessor` falls back to its global
`numerical_method` (default `"ple"`) and `categorical_method` (default `"int"`). See
[Configuration](../core_concepts/configuration.md) for every knob.
```

Ask for a dict of per-feature blocks instead, for inspection or per-block downstream heads:

```python
blocks = pre.transform(df, return_array=False)
assert sum(block.shape[1] for block in blocks.values()) == X.shape[1]
for name, block in blocks.items():
  print(name, block.shape)
```

```text
num_age (200, 7)
num_income (200, 7)
num_experience (200, 7)
cat_job (200, 4)
cat_city (200, 1)
```

## Inspect what was built

Call `pre.get_feature_info(verbose=True)` for a table of fitted pipelines and widths.
Use `verbose=False` to read the numerical, categorical, and grouped feature information
in code. The lineage records trace output columns back to their inputs.

```python
_, categorical_info, _ = pre.get_feature_info(verbose=False)
print(categorical_info["job"]["preprocessing"])

lineage = pre.get_feature_lineage()
first = lineage[0]
print(first.output_feature, first.source_features, first.uses_target)
assert [record.output_feature for record in lineage] == list(pre.get_feature_names_out())
```

```text
normalize_missing -> imputer -> onehot -> to_float
num_age_ple0 ('age',) True
```

The first output column comes from `age` and was fitted using the target. The lineage
covers every output column, in the same order as `get_feature_names_out()`.
See [Outputs and inspection](../core_concepts/outputs_and_inspection.md) for the full
contract.

## Use a transformer on its own

Import individual transformers from `pretab.transformers`. They follow the scikit-learn
API and work inside a `Pipeline` or `ColumnTransformer`.

```python
import numpy as np

from pretab.transformers import PLETransformer

rng = np.random.default_rng(0)
values = rng.normal(size=(200, 1))
target = values[:, 0] ** 2

ple = PLETransformer(output_dim=15, task="regression", random_state=0)
features = ple.fit_transform(values, target)
assert features.shape == (len(values), 15)
print(features.shape[1])
```

```text
15
```

```{warning}
`PLETransformer` uses the target to place its bin edges. The direct fit above emits a
`LeakageWarning`: do not fit on all rows before splitting them to evaluate a model. Put the
transformer and predictor in one pipeline and fit them on the training split only, or
cross-validate the unfitted pipeline. See [Target awareness](../core_concepts/target_awareness.md)
for out-of-fold training features.
```

Spline families with a smoothness penalty expose it through `get_penalty_matrix()`.
For natural cubic splines, the matrix penalizes the integrated squared curvature.

```python
import numpy as np

from pretab.transformers import NaturalCubicSplineTransformer

rng = np.random.default_rng(0)
values = rng.normal(size=(200, 1))
spline = NaturalCubicSplineTransformer(output_dim=8)
features = spline.fit_transform(values)
penalty = spline.get_penalty_matrix()
assert features.shape == (len(values), 8)
assert penalty.shape == (features.shape[1], features.shape[1])
print(features.shape, penalty.shape)
```

```text
(200, 8) (8, 8)
```

The multivariate thin-plate spline models several columns jointly and is sized by
`n_components` rather than `output_dim`:

```python
import numpy as np

from pretab.transformers import ThinPlateSplineTransformer

rng = np.random.default_rng(0)
values = rng.normal(size=(200, 2))
thin_plate = ThinPlateSplineTransformer(n_components=10, random_state=0)
features = thin_plate.fit_transform(values)
penalty = thin_plate.get_penalty_matrix()
assert features.shape == (len(values), 10)
assert penalty.shape == (features.shape[1], features.shape[1])
print(features.shape, penalty.shape)
```

```text
(200, 10) (10, 10)
```

```{warning}
`ThinPlateSplineTransformer.get_penalty_matrix()` is experimental: the retained
eigenvalues are not guaranteed non-negative, so the returned penalty is not guaranteed
positive semi-definite. Calling it emits a `ConfigWarning` to make this explicit.
```

## Next steps

- See PreTab lift a linear model, baseline versus PreTab, in the
  [non-linear regression tutorial](../tutorials/nonlinear_regression.md).
- Decide between the two surfaces in [Overview](overview.md#two-ways-to-use-it).
- Browse every method in [Representations](../representations/overview.md).
- Learn the shared ideas in [Core concepts](../core_concepts/feature_representation.md).
