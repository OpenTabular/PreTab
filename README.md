<div align="center">
  <img src="./docs/images/logo/pretab.png" width="900" />

[![PyPI](https://img.shields.io/pypi/v/pretab)](https://pypi.org/project/pretab)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pretab)
[![Python](https://img.shields.io/pypi/pyversions/pretab)](https://pypi.org/project/pretab)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/OpenTabular/PreTab/blob/main/LICENSE)
[![docs build](https://readthedocs.org/projects/pretab/badge/?version=latest)](https://pretab.readthedocs.io/en/latest/?badge=latest)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://pretab.readthedocs.io/en/latest/)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/OpenTabular/PreTab/issues)

[📘 Documentation](https://pretab.readthedocs.io) |
[🚀 Getting Started](https://pretab.readthedocs.io/en/latest/getting_started/quickstart.html) |
[📖 User Guide](https://pretab.readthedocs.io/en/latest/user_guide/preprocessing.html) |
[🤔 Report Issues](https://github.com/OpenTabular/PreTab/issues)

</div>

# PreTab: Tabular Preprocessing Made Simple

**PreTab** is a modular, scikit-learn compatible preprocessing library for tabular data. A
single `Preprocessor` detects numerical and categorical columns and turns them into
model-ready features, and every strategy is also available as a standalone transformer:
splines, neural basis expansions, piecewise-linear encoding, binning, language embeddings,
and temporal features. Because it speaks the sklearn API, PreTab drops straight into
`Pipeline` and `ColumnTransformer` workflows and accepts any sklearn transformer alongside
its own.

## Why PreTab?

- **Familiar interface.** A scikit-learn `fit`/`transform`/`fit_transform` API that drops
  into existing pipelines and works with both `pandas.DataFrame` and `numpy.ndarray`
  inputs.
- **Automatic feature handling.** Feature-type detection and per-feature strategies let you
  describe intent once instead of wiring transformers by hand.
- **Beyond scaling.** Spline bases, neural basis maps, and piecewise-linear encoding turn
  raw numerical columns into expressive representations.
- **Categoricals done right.** Ordinal and one-hot encoding, pretrained language
  embeddings, and custom binning cover both low- and high-cardinality columns.
- **Composable and extensible.** Every strategy is a standalone transformer you can import,
  compose, or subclass, and any sklearn transformer works out of the box.

## 🏃 Quickstart

```python
import numpy as np
import pandas as pd

from pretab.preprocessor import Preprocessor

df = pd.DataFrame({
    "age": np.random.randint(18, 65, size=100),
    "income": np.random.normal(60_000, 15_000, size=100).astype(int),
    "city": np.random.choice(["Berlin", "Munich", "Hamburg"], size=100),
})
y = np.random.randn(100)

# Global strategies: PLE for numerics, integer codes for categoricals
preprocessor = Preprocessor(numerical_preprocessing="ple", categorical_preprocessing="int")

X = preprocessor.fit_transform(df, y)   # dict of transformed feature blocks
```

> **That's it.** PreTab detects feature types, fits a strategy per column, and returns
> ready-to-use arrays.

> **Works with pandas and numpy.** Pass a DataFrame or an array, and PreTab infers
> numerical vs. categorical columns for you.

## Available Transformers

PreTab groups its transformers into four families. Each one follows the standard `fit` /
`transform` API and is importable from `pretab.transformers`.

### Splines

| Transformer                      | Basis                        | Best for                             |
| -------------------------------- | ---------------------------- | ------------------------------------ |
| `CubicSplineTransformer`         | B-spline basis               | Smooth non-linear numerical effects  |
| `NaturalCubicSplineTransformer`  | Natural cubic spline         | Smooth effects with linear tails     |
| `PSplineTransformer`             | Penalized B-spline           | Smoothness with a penalty matrix     |
| `TensorProductSplineTransformer` | Tensor-product spline        | Interactions between two features    |
| `ThinPlateSplineTransformer`     | Thin-plate regression spline | Smooth multivariate surfaces         |

### Feature maps

| Transformer                   | Basis                  | Best for                            |
| ----------------------------- | ---------------------- | ----------------------------------- |
| `RBFExpansionTransformer`     | Radial basis functions | Localized, kernel-like features     |
| `ReLUExpansionTransformer`    | ReLU basis             | Piecewise-linear neural features    |
| `SigmoidExpansionTransformer` | Sigmoid basis          | Smooth saturating features          |
| `TanhExpansionTransformer`    | Tanh basis             | Zero-centered saturating features   |

### Encoding and binning

| Transformer                    | Method                                 | Best for                              |
| ------------------------------ | -------------------------------------- | ------------------------------------- |
| `PLETransformer`               | Piecewise linear encoding (supervised) | Strong numerical encoding for models  |
| `CustomBinTransformer`         | Rule- or tree-based binning            | Discretizing numerical or code values |
| `OneHotFromOrdinalTransformer` | One-hot from ordinal codes             | One-hot on pre-encoded categoricals   |
| `LanguageEmbeddingTransformer` | Pretrained language embeddings         | High-cardinality, semantic columns    |

### Temporal

| Transformer               | Method                    | Best for                            |
| ------------------------- | ------------------------- | ----------------------------------- |
| `CyclicalTimeTransformer` | Sine/cosine encoding      | Hour, day, month and cyclic fields  |
| `LagFeatureTransformer`   | Lagged values             | Time-series lag features            |
| `RollingStatsTransformer` | Rolling window statistics | Moving averages and rolling summary |

> **Strategy strings.** Inside the `Preprocessor` you select these by short name (for
> example `"ple"`, `"rbf"`, `"one-hot"`, `"pretrained"`). See the
> [User Guide](https://pretab.readthedocs.io/en/latest/user_guide/preprocessing.html) for
> the full list.

## 📚 Documentation

**Full documentation:** [pretab.readthedocs.io](https://pretab.readthedocs.io)

### Quick Links

- **[Getting Started](https://pretab.readthedocs.io/en/latest/getting_started/installation.html)**: Installation and quickstart
- **[User Guide](https://pretab.readthedocs.io/en/latest/user_guide/preprocessing.html)**: Feature detection, strategies, and outputs
- **[API Reference](https://pretab.readthedocs.io/en/latest/api/index.html)**: The `Preprocessor` and every transformer
- **[Developer Guide](https://pretab.readthedocs.io/en/latest/developer_guide/contributing.html)**: Contributing, versioning, and releases

## 🛠️ Installation

**Basic installation:**

```bash
pip install pretab
```

**With language-embedding support:**

```bash
pip install "pretab[embeddings]"   # adds sentence-transformers
```

> **Lightweight by default.** The `embeddings` extra pulls in `sentence-transformers` and
> PyTorch, so install it only if you use the `pretrained` categorical strategy.

> **Requirements:** Python 3.10 to 3.13.

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
from pretab.preprocessor import Preprocessor

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

> **Two output formats.** `transform` returns a dict of feature blocks by default (keys
> prefixed `num_` and `cat_`), or a single stacked array when you pass `return_array=True`.

### Standalone transformers

Each transformer works on its own and composes with any sklearn estimator.

```python
import numpy as np

from pretab.transformers import PLETransformer

x = np.random.randn(100, 1)
y = np.random.randn(100, 1)

x_ple = PLETransformer(n_bins=15, task="regression").fit_transform(x, y)
assert x_ple.shape[1] == 15
```

> **Some transformers are supervised.** `PLETransformer` uses the target `y` during `fit`
> to place its bin edges, so pass `y` whenever you fit it.

### Inside an sklearn Pipeline

Because every transformer follows the sklearn API, you can drop them into a `Pipeline` or
`ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from pretab.transformers import NaturalCubicSplineTransformer, RBFExpansionTransformer

features = ColumnTransformer([
    ("age", NaturalCubicSplineTransformer(n_knots=10), ["age"]),
    ("income", RBFExpansionTransformer(), ["income"]),
])

model = Pipeline([("features", features), ("ridge", Ridge())])
model.fit(df[["age", "income"]], y)
```

### Spline penalty matrices

Spline transformers expose their penalty matrix for penalized (smoothing) models.

```python
import numpy as np

from pretab.transformers import ThinPlateSplineTransformer

x = np.random.randn(100, 1)

tp = ThinPlateSplineTransformer(n_basis=15)
x_tp = tp.fit_transform(x)
penalty = tp.get_penalty_matrix()   # (n_basis, n_basis) smoothing penalty
```

## Advanced Features

### Automatic feature-type detection

By default PreTab inspects each column and classifies it as numerical or categorical.
String and object columns are treated as categorical, low-cardinality integer columns are
categorical, and integer columns with enough distinct values stay numerical. Tune the
behavior with `cat_cutoff`, `min_unique_vals`, and `treat_all_integers_as_numerical`.

```python
preprocessor = Preprocessor(
    treat_all_integers_as_numerical=False,
    cat_cutoff=0.03,
    min_unique_vals=5,
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

> **Optional dependency.** Install with `pip install "pretab[embeddings]"` before using the
> `pretrained` strategy.

### Custom binning

`CustomBinTransformer` supports both rule-based edges and tree-based bins learned from the
target.

```python
preprocessor = Preprocessor(
    numerical_preprocessing="custombin",
    use_decision_tree_bins=True,
    n_bins=32,
)
```

## 📄 License

PreTab is licensed under the MIT License. See [LICENSE](./LICENSE) for details.

## 🤝 Contributing

Contributions are welcome, whether you are fixing bugs, adding transformers, or improving
the docs. See the
[Contributing Guide](https://pretab.readthedocs.io/en/latest/developer_guide/contributing.html)
to get started, and please follow our
[Code of Conduct](https://github.com/OpenTabular/PreTab/blob/main/CODE_OF_CONDUCT.md).

```bash
git clone https://github.com/OpenTabular/PreTab
cd PreTab
pip install -e ".[dev]"
```

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/OpenTabular/PreTab/issues)
- **Discussions:** [GitHub Discussions](https://github.com/OpenTabular/PreTab/discussions)


