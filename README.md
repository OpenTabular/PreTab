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
from pretab import Preprocessor

df = pd.DataFrame({
    "age": [22, 35, 46, 59],
    "income": [40000, 52000, 98000, 87000],
    "job": ["nurse", "engineer", "scientist", "teacher"]
})

# Optional feature-specific config
config = {
    "age": "ple",
    "income": "rbf",
    "job": "one-hot"
}

preprocessor = Preprocessor(
    feature_preprocessing=config,
    task="regression"
)

# Fit and transform
X_dict = preprocessor.fit_transform(df)

# Optionally get stacked array
X_array = preprocessor.transform(df, return_dict=False)

# Get feature info
preprocessor.get_feature_info()
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


