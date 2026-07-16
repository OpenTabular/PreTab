# Preprocessing overview

The `Preprocessor` (`pretab.preprocessor.Preprocessor`) is the main entry point. It
inspects a `pandas.DataFrame`, decides which columns are numerical and which are
categorical, and builds a scikit-learn `ColumnTransformer` that applies a chosen strategy
per feature.

## How feature types are detected

By default, columns are classified automatically:

- **Numerical** — continuous columns, and integer columns with enough distinct values.
- **Categorical** — string/object columns, and low-cardinality integer columns.

The behaviour is controlled by several constructor arguments:

`cat_cutoff`
: Threshold that decides whether an integer column is treated as categorical.

`treat_all_integers_as_numerical`
: When `True`, every integer column is treated as numerical regardless of cardinality.

`min_unique_vals`
: Minimum number of distinct values a column needs to be treated as numerical.

## Choosing strategies

There are two ways to configure preprocessing:

1. **Globally** via `numerical_preprocessing` and `categorical_preprocessing`.
2. **Per feature** via the `feature_preprocessing` dict, which overrides the global
   defaults for specific columns.

```python
from pretab.preprocessor import Preprocessor

# Global strategy for every column of a given type
preprocessor = Preprocessor(
    numerical_preprocessing="ple",
    categorical_preprocessing="int",
)

# Or override individual columns
preprocessor = Preprocessor(
    feature_preprocessing={
        "age": "ple",
        "income": "rbf",
        "city": "one-hot",
    },
)
```

## Numerical strategies

| Strategy          | Transformer                      | Notes |
| ----------------- | -------------------------------- | ----- |
| `standardization` | `StandardScaler`                 | Zero mean, unit variance |
| `minmax`          | `MinMaxScaler`                   | Scaled to `[-1, 1]` |
| `quantile`        | `QuantileTransformer`            | Rank-based normalisation |
| `robust`          | `RobustScaler`                   | Robust to outliers |
| `polynomial`      | `PolynomialFeatures`             | Polynomial interactions |
| `box-cox`         | `PowerTransformer`               | Positive inputs only |
| `yeo-johnson`     | `PowerTransformer`               | Handles zero/negative values |
| `ple`             | `PLETransformer`                 | Piecewise linear encoding |
| `custombin`       | `CustomBinTransformer`           | Rule- or tree-based binning |
| `rbf`             | `RBFExpansionTransformer`        | Radial basis functions |
| `relu`            | `ReLUExpansionTransformer`       | ReLU basis expansion |
| `sigmoid`         | `SigmoidExpansionTransformer`    | Sigmoid basis expansion |
| `tanh`            | `TanhExpansionTransformer`       | Tanh basis expansion |
| `cubicspline`     | `CubicSplineTransformer`         | B-spline basis |
| `naturalspline`   | `NaturalCubicSplineTransformer`  | Natural cubic spline |
| `pspline`         | `PSplineTransformer`             | Penalised B-spline |
| `tensorspline`    | `TensorProductSplineTransformer` | Tensor-product spline |
| `tprs`            | `ThinPlateSplineTransformer`     | Thin-plate regression spline |
| `none`            | `NoTransformer`                  | Pass-through |

## Categorical strategies

| Strategy              | Transformer                    | Notes |
| --------------------- | ------------------------------ | ----- |
| `int`                 | `ContinuousOrdinalTransformer` | Integer/ordinal encoding (default) |
| `one-hot`             | `OneHotEncoder`                | One-hot encoding |
| `onehot_from_ordinal` | `OneHotFromOrdinalTransformer` | One-hot from pre-encoded ordinals |
| `pretrained`          | `LanguageEmbeddingTransformer` | Pretrained language embeddings |
| `custombin`           | `CustomBinTransformer`         | Binning of categorical codes |
| `none`                | `NoTransformer`                | Pass-through |

```{note}
The `pretrained` strategy requires the optional `sentence-transformers` dependency.
Install it with `pip install "pretab[embeddings]"`.
```

## Output format

`fit_transform` and `transform` return a dictionary that maps each feature to its
transformed array (keys are prefixed with `num_` or `cat_`). Pass `return_array=True`
to `transform` to receive a single stacked `numpy.ndarray` instead.

```python
X_dict = preprocessor.fit_transform(df, y)   # {"num_age": ..., "cat_city": ...}
X_array = preprocessor.transform(df, return_array=True)   # single ndarray
```

Use `get_feature_info(verbose=True)` to inspect the resolved strategy and output
dimensionality of every feature.

## Using transformers directly

Every transformer listed above is also importable from `pretab.transformers` and works as
a standalone scikit-learn transformer, so it can be composed into any `Pipeline` or
`ColumnTransformer`. See the [Quickstart](../getting_started/quickstart.md) for examples,
and the [API Reference](../api/index.rst) for the full parameter list of each class.
