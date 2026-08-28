# PreTab

**PreTab** is a modular, extensible, and [scikit-learn](https://scikit-learn.org/)-compatible
preprocessing library for tabular data. It supports **all `sklearn` transformers** out of the
box and extends them with a rich set of custom encoders, splines, and neural basis expansions.

```{note}
These docs are for PreTab {{ version }}. The project is under active development and the
public API may evolve while the major version is `0`.
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
Ordinal and one-hot encodings, pretrained language embeddings, and helpers such as
`OneHotFromOrdinalTransformer`.
:::

:::{grid-item-card} 🔧 Composable pipelines
Fully compatible with `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`;
accepts any sklearn-native transformer and its hyperparameters.
:::

:::{grid-item-card} 🧠 Smart defaults
Automatic feature-type detection (numerical vs. categorical) with support for both
`pandas.DataFrame` and `numpy.ndarray` inputs.
:::

::::

## See it in action

```python
import numpy as np
import pandas as pd
from pretab import Preprocessor

df = pd.DataFrame({
    "age": np.random.randint(18, 65, size=100),
    "income": np.random.normal(60_000, 15_000, size=100).astype(int),
    "city": np.random.choice(["Berlin", "Munich", "Hamburg"], size=100),
})
y = np.random.randn(100)

# One strategy per feature type: PLE for numerics, integer codes for categoricals
pre = Preprocessor(numerical_method="ple", categorical_method="int")
X = pre.fit_transform(df, y)          # dict of model-ready feature blocks

{k: v.shape for k, v in X.items()}
# {'num_age': (100, 7), 'num_income': (100, 7), 'cat_city': (100, 1)}
```

`Preprocessor` detects the column types, fits a strategy per column, and returns model-ready
arrays, either as a dict of blocks or, with `return_array=True`, a single stacked matrix.
Inspect the resolved layout at any time with `get_feature_info(verbose=True)`:

```text
feature  kind         pipeline                        dim   cats
----------------------------------------------------------------
age      numerical    imputer -> minmax -> ple          7      -
income   numerical    imputer -> minmax -> ple          7      -
city     categorical  imputer -> continuous_ordinal     1      4
```

### Mix strategies per column

Columns rarely want the same treatment. Pass a `feature_preprocessing` map to give each
column its own strategy, and a single `fit` still returns one coherent feature set:

```python
pre = Preprocessor(feature_preprocessing={
    "age": "ple",       # piecewise-linear encoding
    "income": "rbf",    # radial-basis expansion
    "city": "one-hot",  # one-hot categorical
})
X = pre.fit_transform(df, y)

{k: v.shape for k, v in X.items()}
# {'num_age': (100, 7), 'num_income': (100, 7), 'cat_city': (100, 3)}
```

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
