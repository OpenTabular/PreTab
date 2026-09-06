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
X.shape
```

```{tip}
When no per-feature config is given, the `Preprocessor` falls back to its global
`numerical_method` (default `"ple"`) and `categorical_method` (default `"int"`). See
[Configuration](../core_concepts/configuration.md) for every knob.
```

Ask for a dict of per-feature blocks instead, for inspection or per-block downstream heads:

```python
X_dict = pre.transform(df, return_array=False)   # {"num_age": ..., "cat_city": ...}
```

## Inspect what was built

Every fitted representation is self-describing. Read the resolved layout, or trace each
output column back to its source.

```python
pre.get_feature_info(verbose=True)   # human-readable table of per-feature pipelines

lineage = pre.get_feature_lineage()  # one record per output column
lineage[0]
```

The lineage covers every output column, and the names line up with `get_feature_names_out`.
See [Outputs and inspection](../core_concepts/outputs_and_inspection.md) for the full
contract.

## Use a transformer on its own

Every strategy is also importable from `pretab.transformers` and follows the scikit-learn
API, so it drops into a `Pipeline` or `ColumnTransformer`.

```python
import numpy as np

from pretab.transformers import PLETransformer

x = np.random.randn(200, 1)
y = np.random.randn(200)

x_ple = PLETransformer(output_dim=15, task="regression").fit_transform(x, y)
x_ple.shape[1]   # number of piecewise-linear bins
```

```{note}
`PLETransformer` is supervised: it reads the target `y` during `fit` to place its bin edges.
Always pass `y` when fitting it, or any pipeline that contains it. See
[Target awareness](../core_concepts/target_awareness.md).
```

Spline families that carry a smoothness penalty expose it through `get_penalty_matrix()`:

```python
import numpy as np

from pretab.transformers import NaturalCubicSplineTransformer

x = np.random.randn(200, 1)
spline = NaturalCubicSplineTransformer(output_dim=8)
spline.fit_transform(x)

penalty = spline.get_penalty_matrix()   # integrated-curvature penalty for GAM-style fitting
```

The multivariate thin-plate spline models several columns jointly and is sized by
`n_components` rather than `output_dim`:

```python
import numpy as np

from pretab.transformers import ThinPlateSplineTransformer

x = np.random.randn(200, 2)              # two input columns, modelled together
tp = ThinPlateSplineTransformer(n_components=10)
features = tp.fit_transform(x)
penalty = tp.get_penalty_matrix()
```

## Next steps

- See PreTab lift a linear model, baseline versus PreTab, in the
  [non-linear regression tutorial](../tutorials/nonlinear_regression.md).
- Decide between the two surfaces in [Choosing an interface](choosing_an_interface.md).
- Browse every method in [Representations](../representations/overview.md).
- Learn the shared ideas in [Core concepts](../core_concepts/feature_representation.md).
