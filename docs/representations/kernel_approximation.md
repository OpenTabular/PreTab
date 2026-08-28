# Kernel approximation

Kernel approximation builds an explicit, low-dimensional feature map whose inner products
approximate an implicit kernel, so a linear model downstream can behave like a kernel machine
without ever forming the full kernel matrix. PreTab wraps the two standard approaches. Both are
**multivariate, standalone transformers**: they operate on the whole input matrix and are not
selectable per column through `Preprocessor`.

## Random Fourier features

Approximates a shift-invariant kernel (by default the RBF kernel) with random projections,
following Rahimi and Recht. This makes kernel-style models scale to large datasets, since the
cost of the approximation does not grow with the number of training points the way an exact
kernel method's does.

```python
from pretab.transformers import RandomFourierFeaturesTransformer

t = RandomFourierFeaturesTransformer(n_components=100, gamma=1.0)
X2 = t.fit_transform(X)
```

Constructor highlights: `n_components=100`, `gamma=1.0`, `random_state`.

```{tip}
`n_components` trades approximation quality for cost. More components track the true kernel
more closely at the price of a wider output; start around 100 and increase if validation
performance is still improving.
```

## Nyström

Approximates a kernel by sampling landmark points from the training data and projecting onto
them, following Williams and Seeger. It supports several kernels through `kernel`, and is often
more accurate than random Fourier features at a given output width because the landmarks adapt
to the data rather than being drawn at random.

```python
from pretab.transformers import NystroemFeaturesTransformer

t = NystroemFeaturesTransformer(n_components=100, kernel="rbf")
X2 = t.fit_transform(X)
```

Constructor highlights: `n_components=100`, `kernel="rbf"`, `gamma=None`, `degree=3`,
`coef0=1`, `random_state`.

```{note}
Both methods approximate the same idea from different angles: random Fourier features draw a
random basis independent of the data, while Nyström samples landmarks from the data itself.
When in doubt, try both and compare with cross-validation.
```

```{warning}
Random Fourier features and Nyström are multivariate and operate on the whole input matrix.
They are not available as a per-column `numerical_method`; fit them standalone or combine them
with per-column methods through a `ColumnTransformer`.
```

## Where to go next

- [Functional expansions](functional_expansions.md) for the per-column basis functions.
- [Spline expansions](spline_expansions.md) for the multivariate tensor-product and thin-plate
  splines, another way to model several inputs jointly.
- [References](references.md) for the kernel-approximation literature.
