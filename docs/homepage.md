# pretab

**pretab** is a modular, extensible, and [scikit-learn](https://scikit-learn.org/)-compatible
preprocessing library for tabular data. It supports **all `sklearn` transformers** out of the
box and extends them with a rich set of custom encoders, splines, and neural basis expansions.

```{note}
These docs are for pretab {{ version }}. The project is under active development and the
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

## Get started

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Installation
:link: getting_started/installation
:link-type: doc
Install pretab from PyPI or from source.
:::

:::{grid-item-card} Quickstart
:link: getting_started/quickstart
:link-type: doc
Fit and transform a dataset in a few lines.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
The `Preprocessor` and every transformer.
:::

::::

## Project links

- **Source code**: <https://github.com/OpenTabular/PreTab>
- **PyPI**: <https://pypi.org/project/pretab/>
- **Issue tracker**: <https://github.com/OpenTabular/PreTab/issues>
