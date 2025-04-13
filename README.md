# 📦 pretab

**pretab** is a modular, extensible, and `scikit-learn`-compatible preprocessing library for tabular data. It supports **all `sklearn` transformers** out of the box, and extends functionality with a rich set of custom encoders, splines, and neural basis expansions.

---

## ✨ Features

- 🔢 **Numerical preprocessing** via:
  - Polynomial and spline expansions: `B-splines`, `natural cubic splines`, `thin plate splines`, `tensor product splines`, `P-splines`
  - Neural-inspired basis: `RBF`, `ReLU`, `Sigmoid`, `Tanh`
  - Custom binning: rule-based or tree-based
  - Piecewise Linear Encoding (`PLE`)

- 🌤 **Categorical preprocessing**:
  - Ordinal encodings
  - One-hot encodings
  - Language embeddings (`pretrained vectorizers`)
  - Custom encoders like `OneHotFromOrdinalTransformer`

- 🔧 **Composable pipeline interface**:
  - Fully compatible with `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`
  - Accepts all sklearn-native transformers and parameters seamlessly

- 🧠 **Smart preprocessing**:
  - Automatically detects feature types (categorical vs numerical)
  - Supports both `pandas.DataFrame` and `numpy.ndarray` inputs

- 🧪 Comprehensive test coverage

- 🤝 Community-driven and open to contributions

---

## 💠 Installation

Install via pip:

```bash
pip install pretab
```

Or install in editable mode for development:

```bash
git clone https://github.com/OpenTabular/pretab.git
cd pretab
pip install -e .
```

---

## 🚀 Quickstart

```python
import pandas as pd
import numpy as np
from pretab.preprocessor import Preprocessor

# Simulated tabular dataset
df = pd.DataFrame({
    "age": np.random.randint(18, 65, size=100),
    "income": np.random.normal(60000, 15000, size=100).astype(int),
    "job": np.random.choice(["nurse", "engineer", "scientist", "teacher", "artist", "manager"], size=100),
    "city": np.random.choice(["Berlin", "Munich", "Hamburg", "Cologne"], size=100),
    "experience": np.random.randint(0, 40, size=100)
})

y = np.random.randn(100, 1)

# Optional feature-specific preprocessing config
config = {
    "age": "ple",
    "income": "rbf",
    "experience": "quantile",
    "job": "one-hot",
    "city": "none"
}

# Initialize Preprocessor
preprocessor = Preprocessor(
    feature_preprocessing=config,
    task="regression"
)

# Fit and transform the data into a dictionary of feature arrays
X_dict = preprocessor.fit_transform(df, y)

# Optionally get a stacked array instead of a dictionary
X_array = preprocessor.transform(df, return_array=True)

# Get feature metadata
preprocessor.get_feature_info(verbose=True)
```

---

## 🪰 Included Transformers

pretab includes both sklearn-native and custom-built transformers:

### 🌈 Splines
- `CubicSplineTransformer`
- `NaturalCubicSplineTransformer`
- `PSplineTransformer`
- `TensorProductSplineTransformer`
- `ThinPlateSplineTransformer`

### 🧠 Feature Maps
- `RBFExpansionTransformer`
- `ReLUExpansionTransformer`
- `SigmoidExpansionTransformer`
- `TanhExpansionTransformer`

### 📊 Encodings and Binning
- `PLETransformer`
- `CustomBinTransformer`
- `OneHotFromOrdinalTransformer`
- `ContinuousOrdinalTransformer`
- `LanguageEmbeddingTransformer`

### 🔧 Utilities
- `NoTransformer`
- `ToFloatTransformer`

> Plus: **any `sklearn` transformer** can be passed directly with full support for hyperparameters.

### Using Transformers
Using the transformers follows the standard sklearn.preprocessing steps. I.e. using PLE
```python
import numpy as np
from pretab.transformers import PLETransformer

x = np.random.randn(100, 1)
y = np.random.randn(100, 1)

x_ple = PLETransformer(n_bins=15, task="regression").fit_transform(x, y)

assert x_ple.shape[1] == 15
```

For splines, the penalty matrices can be extracted via `.get_penalty_matrix()`

```python
import numpy as np
from pretab.transformers import ThinPlateSplineTransformer

x = np.random.randn(100, 1)

tp = ThinPlateSplineTransformer(n_basis=15)

x_tp = tp.fit_transform(x)

assert x_tp.shape[1] == 15

penalty = tp.get_penalty_matrix()
```

---

## 🧪 Running Tests

```bash
pytest --maxfail=2 --disable-warnings -v
```

---

## 🤝 Contributing

pretab is community-driven! Whether you’re fixing bugs, adding new encoders, or improving the docs — contributions are welcome.

```bash
git clone https://github.com/OpenTabular/pretab.git
cd pretab
pip install -e ".[dev]"
```

Then create a pull request 🚀

---

## 📄 License

MIT License. See [LICENSE](./LICENSE) for details.

---

## ❤️ Acknowledgements

pretab builds on the strengths of:
- [`scikit-learn`](https://scikit-learn.org)

---


