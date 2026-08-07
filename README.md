<div align="center">
  <img src="./docs/logo/pretab-logo.png" width="900" />

[![PyPI](https://img.shields.io/pypi/v/pretab)](https://pypi.org/project/pretab)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pretab)
[![Python](https://img.shields.io/pypi/pyversions/pretab)](https://pypi.org/project/pretab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OpenTabular/PreTab/blob/main/LICENSE)
[![docs build](https://readthedocs.org/projects/pretab/badge/?version=latest)](https://pretab.readthedocs.io/en/latest/?badge=latest)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://pretab.readthedocs.io/en/latest/)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/OpenTabular/PreTab/issues)

[📘 Documentation](https://pretab.readthedocs.io) |
[🚀 Getting Started](https://pretab.readthedocs.io/en/latest/getting_started/quickstart.html) |
[📖 Representations](https://pretab.readthedocs.io/en/latest/representations/overview.html) |
[🤔 Report Issues](https://github.com/OpenTabular/PreTab/issues)

</div>

# PreTab: Tabular Preprocessing Made Simple

**PreTab** is a modular, scikit-learn compatible representation and preprocessing library
for tabular data. A single `Preprocessor` detects numerical and categorical columns and
turns them into model-ready features. Every strategy it uses (splines, neural basis
expansions, piecewise-linear encoding, binning, kernel approximations, and language
embeddings) is also available as a standalone transformer. Because it speaks the sklearn
API, PreTab drops straight into `Pipeline` and `ColumnTransformer` workflows and accepts any
sklearn transformer alongside its own.

Beyond the transformers themselves, every fitted representation is self-describing: it
reports per-output-column lineage, guards supervised methods against leakage, serializes to
a portable versioned spec, and can be extended with your own representations through a
public, discoverable protocol.

## Why PreTab?

- **Familiar interface.** A scikit-learn `fit`/`transform`/`fit_transform` API that drops
  into existing pipelines and works with both `pandas.DataFrame` and `numpy.ndarray`
  inputs.
- **Automatic feature handling.** Feature-type detection and per-feature strategies let you
  describe intent once instead of wiring transformers by hand.
- **Beyond scaling.** Spline bases, neural basis maps, piecewise-linear encoding, and
  kernel approximations turn raw numerical columns into expressive representations.
- **Categoricals done right.** Ordinal and one-hot encoding and pretrained language
  embeddings cover both low- and high-cardinality columns.
- **Self-describing and reproducible.** Every fit produces per-column feature lineage and
  serializes to a portable spec with a stable fingerprint, so you always know what a fitted
  preprocessor does and can reproduce it exactly.
- **Leakage-safe by default.** Supervised representations declare their target usage and
  warn when fit outside a controlled context, with a cross-fitting wrapper for out-of-fold
  training features.
- **Composable and extensible.** Every strategy is a standalone transformer you can import,
  compose, or subclass; register your own representation and it behaves like a built-in.

## 🏃 Quickstart

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

# Global strategies: PLE for numerics, integer codes for categoricals
preprocessor = Preprocessor(numerical_method="ple", categorical_method="int")

X = preprocessor.fit_transform(df, y)   # dict of transformed feature blocks

print({k: v.shape for k, v in X.items()})
# {'num_age': (100, 7), 'num_income': (100, 7), 'cat_city': (100, 1)}
```

> **Note:** PreTab accepts a `pandas.DataFrame` or a `numpy.ndarray` and infers numerical
> versus categorical columns either way.

> **Tip:** Swap the global methods for a `feature_preprocessing` map, for example
> `{"age": "ple", "income": "rbf", "city": "one-hot"}`, and PreTab fits each column with its
> own strategy in a single pass. See [Usage](#usage) for a full example.

## Available Transformers

PreTab groups its transformers into three families. Each one follows the standard `fit` /
`transform` API and is importable from `pretab.transformers`.

### Splines

| Transformer                        | Basis                                 | Best for                               |
| ----------------------------------- | -------------------------------------- | ---------------------------------------- |
| `BSplineTransformer`                | B-spline basis                        | General-purpose smooth nonlinearity    |
| `MSplineTransformer`                | Non-negative B-spline basis           | Density-like, non-negative bases       |
| `ISplineTransformer`                | Monotone integrated spline            | Effects that must not reverse          |
| `CubicRegressionSplineTransformer`  | Cubic regression spline               | GAM-style additive smooth terms        |
| `NaturalCubicSplineTransformer`     | Natural cubic spline                  | Smooth effects with linear tails       |
| `PSplineTransformer`                | Penalized B-spline                    | Smoothness via a difference penalty    |
| `TensorProductSplineTransformer`    | Tensor-product spline (multivariate)  | Smooth interactions across 2+ features |
| `ThinPlateSplineTransformer`        | Thin-plate spline (multivariate)      | Smooth surfaces across 2+ features     |

### Feature maps

| Transformer                        | Basis                                   | Best for                              |
| ------------------------------------ | ------------------------------------------ | ---------------------------------------- |
| `RBFExpansionTransformer`            | Radial basis functions                  | Localized, kernel-like features       |
| `ReLUExpansionTransformer`           | ReLU basis                              | Piecewise-linear neural features      |
| `SigmoidExpansionTransformer`        | Sigmoid basis                           | Smooth saturating features            |
| `TanhExpansionTransformer`           | Tanh basis                              | Zero-centered saturating features     |
| `FourierFeatureTransformer`          | Sine/cosine basis                       | Periodic or cyclic numerical effects  |
| `RandomFourierFeaturesTransformer`   | Random Fourier features (multivariate)  | Scalable RBF-kernel approximation     |
| `NystroemFeaturesTransformer`        | Nystroem kernel map (multivariate)      | Landmark-based kernel approximation   |

### Encoding and binning

| Transformer                    | Method                                  | Best for                              |
| ------------------------------- | ------------------------------------------ | ---------------------------------------- |
| `PLETransformer`                | Piecewise-linear encoding (supervised)  | Strong numerical encoding for models  |
| `NumericBinningTransformer`     | Uniform/quantile binning, tree-driven   | Discretizing numerical columns        |
| `ContinuousOrdinalTransformer`  | Integer (ordinal) encoding              | Compact codes for categoricals        |
| `LanguageEmbeddingTransformer`  | Pretrained language embeddings          | High-cardinality, semantic columns    |

> **Warning:** `OneHotFromOrdinalTransformer` is deprecated. Use
> `categorical_method="one-hot"` (backed by `sklearn.preprocessing.OneHotEncoder`) instead.

> **Note:** Inside the `Preprocessor` you select these by short name, for example `"ple"`,
> `"rbf"`, `"one-hot"`, `"pretrained"`. See
> [Representations](https://pretab.readthedocs.io/en/latest/representations/overview.html) for
> the full catalogue and [comparison table](https://pretab.readthedocs.io/en/latest/representations/comparison_table.html).

## 📚 Documentation

**Full documentation:** [pretab.readthedocs.io](https://pretab.readthedocs.io)

### Quick Links

- **[Getting Started](https://pretab.readthedocs.io/en/latest/getting_started/installation.html)**: Installation and quickstart
- **[Core Concepts](https://pretab.readthedocs.io/en/latest/core_concepts/feature_representation.html)**: Configuration, resolution, target awareness, reproducibility
- **[Representations](https://pretab.readthedocs.io/en/latest/representations/overview.html)**: The full method catalogue and how to choose one
- **[Tutorials](https://pretab.readthedocs.io/en/latest/tutorials/nonlinear_regression.html)**: Worked, end-to-end examples
- **[API Reference](https://pretab.readthedocs.io/en/latest/api/index.html)**: The `Preprocessor` and every transformer
- **[Developer Guide](https://pretab.readthedocs.io/en/latest/developer_guide/contributing.html)**: Contributing, testing, and releases

## 🛠️ Installation

**Basic installation:**

```bash
pip install pretab
```

**With optional extras:**

```bash
pip install "pretab[embeddings]"   # adds sentence-transformers, for the `pretrained` strategy
pip install "pretab[lightgbm]"     # adds lightgbm, for placement_strategy="lightgbm"
pip install "pretab[all]"          # both of the above
```

> **Note:** The core install has no heavy dependencies. Each extra is opt-in and only
> needed if you use the corresponding strategy. PreTab requires Python 3.10 to 3.13.

**From source:**

```bash
git clone https://github.com/OpenTabular/PreTab
cd PreTab
pip install -e ".[dev]"
```

## Usage

### The Preprocessor

The `Preprocessor` is the high-level entry point. Set a global strategy per feature type,
or override individual columns with `feature_preprocessing`.

```python
from pretab import Preprocessor

# Per-feature configuration overrides the global defaults
preprocessor = Preprocessor(
    feature_preprocessing={
        "age": "ple",
        "income": "rbf",
        "experience": "quantile",
        "city": "one-hot",
    },
    task="regression",
)

X_dict = preprocessor.fit_transform(df, y)               # {"num_age": ..., "cat_city": ...}
X_array = preprocessor.transform(df, return_array=True)  # single stacked ndarray

preprocessor.get_feature_info(verbose=True)              # inspect resolved strategies
```

`get_feature_info(verbose=True)` prints the resolved layout so you can confirm every column
at a glance:

```text
feature     kind         pipeline                        dim   cats
-------------------------------------------------------------------
age         numerical    imputer -> minmax -> ple          7      -
income      numerical    imputer -> minmax -> rbf          7      -
experience  numerical    imputer -> minmax -> quantile     1      -
city        categorical  imputer -> onehot -> to_float     4      4
```

> **Note:** `transform` returns a dict of feature blocks by default (keys prefixed `num_`
> and `cat_`), or a single stacked array when you pass `return_array=True`.

### Standalone transformers

Each transformer works on its own and composes with any sklearn estimator.

```python
import numpy as np

from pretab.transformers import PLETransformer

x = np.random.randn(100, 1)
y = np.random.randn(100, 1)

x_ple = PLETransformer(output_dim=15, task="regression").fit_transform(x, y)
assert x_ple.shape[1] == 15
```

> **Important:** `PLETransformer` is supervised. It uses the target `y` during `fit` to
> place its bin edges and raises if you omit it, so always pass `y` when fitting it directly.

### Inside an sklearn Pipeline

Because every transformer follows the sklearn API, you can drop them into a `Pipeline` or
`ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab.transformers import NaturalCubicSplineTransformer, RBFExpansionTransformer

features = ColumnTransformer([
    ("age", NaturalCubicSplineTransformer(output_dim=10), ["age"]),
    ("income", RBFExpansionTransformer(), ["income"]),
])

model = Pipeline([("features", features), ("ridge", Ridge())])
model.fit(df[["age", "income"]], y)
```

### Spline penalty matrices

Spline transformers expose their penalty matrix for penalized (smoothing) models.

```python
import numpy as np

from pretab.transformers import NaturalCubicSplineTransformer

x = np.random.randn(100, 1)

spline = NaturalCubicSplineTransformer(output_dim=10)
x_spline = spline.fit_transform(x)
penalty = spline.get_penalty_matrix()   # (output_dim, output_dim) smoothing penalty
```

## Advanced Features

### Automatic feature-type detection

By default PreTab inspects each column and classifies it as numerical or categorical.
String and object columns are treated as categorical, low-cardinality integer columns are
categorical, and integer columns with enough distinct values stay numerical. Tune the
behavior with `cat_cutoff` and `treat_all_integers_as_numerical`.

```python
preprocessor = Preprocessor(
    treat_all_integers_as_numerical=False,
    cat_cutoff=0.03,
)
```

### Language embeddings for categoricals

The `pretrained` strategy encodes categorical values with a sentence-transformer, which
helps with high-cardinality or semantically rich columns.

```python
preprocessor = Preprocessor(
    feature_preprocessing={"job_title": "pretrained"},
)
```

> **Note:** Install with `pip install "pretab[embeddings]"` before using the `pretrained`
> strategy.

### Numeric binning

`NumericBinningTransformer` (selected as `"custombin"`) discretizes a numerical column into
uniformly- or quantile-spaced bins, with `ordinal`, `onehot`, or `soft` output encodings.

```python
preprocessor = Preprocessor(
    numerical_method="custombin",
    output_dim=32,
)
```

### Feature lineage and inspection

Every fitted `Preprocessor` can explain itself. `get_feature_info` summarizes the resolved
per-column pipeline, and `get_feature_lineage` maps every output column back to its source
feature, representation family, and component.

```python
preprocessor.get_feature_info(verbose=True)      # resolved strategies, widths, categories
lineage = preprocessor.get_feature_lineage()     # one record per output column
```

### Leakage-safe supervised representations

Methods like `PLETransformer` place their bins using the target. PreTab warns when a
supervised transformer is fit outside a `Pipeline` or cross-validation context, and ships a
cross-fitting wrapper that produces out-of-fold training features.

```python
from pretab import CrossFittedTransformer
from pretab.transformers import PLETransformer

cf = CrossFittedTransformer(PLETransformer(), n_folds=5)
X_train_features = cf.fit_transform(x_train, y_train)   # out-of-fold, leakage-free
```

> **Warning:** Fitting a supervised transformer on the same rows you later evaluate on
> leaks target information into the features. `CrossFittedTransformer` removes that leakage
> from the training features themselves; inside a `Pipeline`, cross-validation already
> keeps each fold's fit confined to its training data.

### Serialization and reproducibility

A fitted preprocessor serializes to a portable, versioned JSON spec, a safer alternative to
`pickle` that never executes arbitrary code on load, and reports a stable fingerprint for
tracking exactly what was fitted.

```python
preprocessor.to_spec("representation.json")
restored = Preprocessor.from_spec("representation.json")

preprocessor.fingerprint_          # stable sha256 hash of the fitted representation
```

### Extending PreTab

Add your own representation by subclassing `BaseRepresentation`, then register it so it
behaves like a built-in, selectable via `Preprocessor(numerical_method=...)`.

```python
from pretab import BaseRepresentation, register_representation

class MyRepresentation(BaseRepresentation):
    representation_name = "my_representation"
    feature_kind = "numerical"
    scope = "univariate"
    supervision = "unsupervised"
    # implement fit / transform / _output_sizes

register_representation("my_representation", MyRepresentation)
```

> **Tip:** See the
> [custom representation tutorial](https://pretab.readthedocs.io/en/latest/tutorials/custom_representation.html)
> for a complete, runnable example.

## 📄 License

PreTab is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## 🤝 Contributing

Contributions are welcome, whether you are fixing bugs, adding transformers, or improving
the docs. Clone the repository and install it in editable mode as shown in the Installation
section above, then see the
[Contributing Guide](https://pretab.readthedocs.io/en/latest/developer_guide/contributing.html)
and our
[Code of Conduct](https://github.com/OpenTabular/PreTab/blob/main/CODE_OF_CONDUCT.md).

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/OpenTabular/PreTab/issues)
- **Discussions:** [GitHub Discussions](https://github.com/OpenTabular/PreTab/discussions)


