# PreTab

**PreTab** is a modular, [scikit-learn](https://scikit-learn.org/)-compatible representation
and preprocessing library for tabular data. Its focus is feature representation and
expansion: splines, neural basis maps, piecewise-linear encoding, kernel approximations, and
language embeddings, each shipped as a standalone transformer that speaks the standard
`fit` / `transform` API. Every PreTab transformer subclasses `BaseEstimator` and
`TransformerMixin`, so it drops into a `Pipeline` or `ColumnTransformer` alongside any
sklearn-native transformer you already use.

```{note}
These docs are for PreTab {{ version }}.
```

## Highlights

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} 🔢 Numerical preprocessing
Spline expansions (B-splines, natural cubic, thin-plate, tensor-product, P-splines),
neural basis maps (RBF, ReLU, sigmoid, tanh), custom binning, and Piecewise Linear
Encoding (PLE).
:::

:::{grid-item-card} 🌤 Categorical preprocessing
Ordinal and one-hot encodings, plus pretrained language embeddings for high-cardinality or
semantic columns.
:::

:::{grid-item-card} 🔧 Composable pipelines
Fully compatible with `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`;
compose standalone representations with scikit-learn transformers and estimators.
:::

:::{grid-item-card} 🧠 Smart defaults
Automatic feature-type detection (numerical vs. categorical) with support for
`pandas.DataFrame`, eager `polars.DataFrame` and `numpy.ndarray` inputs.
:::

::::

## See it in action

```python
import numpy as np
import pandas as pd
from pretab import Preprocessor

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "age": rng.integers(18, 65, size=100),
    "income": rng.normal(60_000, 15_000, size=100).astype(int),
    "city": rng.choice(["Berlin", "Munich", "Hamburg"], size=100),
})
y = rng.standard_normal(100)

# One strategy per feature type: PLE for numerics, integer codes for categoricals
pre = Preprocessor(numerical_method="ple", categorical_method="int", task="regression")
X = pre.fit_transform(df, y)
assert X.shape == (100, 15)
assert len(pre.get_feature_lineage()) == X.shape[1]
```

`Preprocessor` detects column types and returns one array by default. Pass
`output_structure="blocks"` to get a dictionary of feature blocks instead. After fitting,
`get_feature_lineage()` traces each output column back to its input.

### Mix strategies per column

Use `feature_preprocessing` to choose a different strategy for each column. The output
still has one row per input sample:

```python
pre = Preprocessor(
    feature_preprocessing={
        "age": "ple",
        "income": "rbf",
        "city": "one-hot",
    },
    task="regression",
)
X = pre.fit_transform(df, y)
assert X.shape == (100, 17)
```

See [What's new in PreTab 1.1](getting_started/overview.md#whats-new-in-pretab-11)
for per-feature widths, grouped splines, and other 1.1 changes.

## Get started

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Overview
:link: getting_started/overview
:link-type: doc
What PreTab is, what it is not, and where it fits.
:::

:::{grid-item-card} Installation
:link: getting_started/installation
:link-type: doc
Install PreTab from PyPI or from source.
:::

:::{grid-item-card} Quickstart
:link: getting_started/quickstart
:link-type: doc
Fit and transform a dataset in a few lines.
:::

:::{grid-item-card} Nonlinear regression
:link: tutorials/nonlinear_regression
:link-type: doc
See PreTab lift a linear model, baseline vs. PreTab.
:::

:::{grid-item-card} Representations
:link: representations/overview
:link-type: doc
The full catalogue of spline and functional expansions, kernel approximations, and encoders.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
The `Preprocessor` and every transformer.
:::

::::

## Project links

- **Source code**: <https://github.com/OpenTabular/PreTab>
- **PyPI**: <https://pypi.org/project/pretab/>
- **Issue tracker**: <https://github.com/OpenTabular/PreTab/issues>
