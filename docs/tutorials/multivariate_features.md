# Multivariate features

Most representations transform one column at a time. Some relationships, though, live in the
interaction between columns: a smooth surface over latitude and longitude, or a kernel over
many inputs at once. PreTab's multivariate methods model several columns jointly. This tutorial
shows how to use them.

## Which methods are multivariate

Four methods operate on several inputs together rather than per column.

`tensorspline`
: Tensor-product spline. A smooth basis over a small number of inputs, capturing their
interaction on a grid.

`tprs`
: Thin-plate regression spline. A smooth surface over two or more inputs, from the generalized
additive model literature. It also accepts a single input, though a univariate spline family
is usually a more natural fit there.

`rff`
: Random Fourier features. A scalable approximation to a shift-invariant kernel.

`nystroem`
: Nyström kernel map. A landmark-based kernel approximation.

```{warning}
These four are standalone transformers. They are not available as a per-column
`numerical_method` on `Preprocessor`, because they need the whole input block. Fit them
directly on the columns you want to model jointly.
```

## A smooth surface with thin-plate splines

Suppose the target is a smooth function of two coordinates. A per-column expansion cannot see
the interaction, but a thin-plate spline models the surface directly.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from pretab.transformers import ThinPlateSplineTransformer

rng = np.random.default_rng(0)
n = 3000
X = rng.uniform(-3, 3, size=(n, 2))
y = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2)) * 5 + rng.normal(0, 0.2, n)

model = Pipeline([
    ("tps", ThinPlateSplineTransformer(n_components=20)),
    ("ridge", Ridge(alpha=1.0)),
])

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-fold R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

The thin-plate basis captures the radial bump over the two coordinates jointly, something two
separate one-dimensional splines cannot do.

```{tip}
Use `n_components` to trade accuracy for cost. More landmarks give a richer surface at higher
memory and compute. Start modest and increase only if validation improves.
```

## A scalable kernel with random Fourier features

When you want kernel-style flexibility over many inputs on a large dataset, random Fourier
features approximate an RBF kernel without forming the full kernel matrix.

```python
from pretab.transformers import RandomFourierFeaturesTransformer

model = Pipeline([
    ("rff", RandomFourierFeaturesTransformer(n_components=200, gamma=0.5)),
    ("ridge", Ridge(alpha=1.0)),
])

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-fold R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

```{note}
Random Fourier features and Nyström both approximate a kernel machine. Random Fourier features
scale to large data with random projections; Nyström samples landmark points and is often more
accurate at a given width. Try both.
```

## Combining multivariate and per-column methods

You can mix a joint block for interacting columns with per-column expansions for the rest,
using a `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from pretab.transformers import PLETransformer
import pandas as pd

df = pd.DataFrame({"lat": X[:, 0], "lon": X[:, 1], "size": rng.uniform(0, 100, n)})

features = ColumnTransformer([
    ("geo", ThinPlateSplineTransformer(n_components=20), ["lat", "lon"]),
    ("size", PLETransformer(output_dim=10, task="regression"), ["size"]),
])

model = Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])
```

The thin-plate spline handles the geographic interaction while PLE handles the standalone
`size` column, each with the representation that suits it.

## Where to go next

- [Spline expansions](../representations/spline_expansions.md) for the tensor-product and
  thin-plate details.
- [Kernel approximation](../representations/kernel_approximation.md) for random Fourier
  features and Nyström.
- [Representations overview](../representations/overview.md) for the underlying theory and supporting notes.
