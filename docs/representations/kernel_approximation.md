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
kernel method's does. For an input vector $x$, each output column draws a random weight vector
$w_k \sim \mathcal{N}(0,\ 2\gamma I)$ and offset $b_k \sim \mathrm{Uniform}(0, 2\pi)$ at fit time,
then computes

$$
\phi_k(x) = \sqrt{\frac{2}{n_{\text{components}}}}\ \cos\!\big(w_k^\top x + b_k\big).
$$

The inner product $\phi(x)^\top \phi(x')$ approximates the RBF kernel
$\exp(-\gamma \lVert x - x' \rVert^2)$ in expectation over the random draw.

```python
import numpy as np
from pretab.transformers import RandomFourierFeaturesTransformer

X = np.random.default_rng(0).uniform(size=(200, 3))   # (200, 3): the whole feature block
t = RandomFourierFeaturesTransformer(n_components=100, gamma=1.0)
t.fit_transform(X).shape
# (200, 100): n_components columns, independent of the number of input features
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
to the data rather than being drawn at random. For landmarks $z_1, \dots, z_m$ (the sampled
training rows) and kernel function $K$, let $k_m(x) = \big(K(x, z_1), \dots, K(x, z_m)\big)$ be
the vector of kernel evaluations between $x$ and every landmark. The output is

$$
\phi(x) = K_{mm}^{-1/2}\ k_m(x),
$$

where $K_{mm}$ is the $m \times m$ kernel matrix between the landmarks themselves, and
$K_{mm}^{-1/2}$ is computed once at fit time via its eigendecomposition. The inner product
$\phi(x)^\top \phi(x')$ approximates $K(x, x')$.

```python
import numpy as np
from pretab.transformers import NystroemFeaturesTransformer

X = np.random.default_rng(0).uniform(size=(200, 3))
t = NystroemFeaturesTransformer(n_components=100, kernel="rbf")
t.fit_transform(X).shape
# (200, 100)
```

Constructor highlights: `n_components=100`, `kernel="rbf"`, `gamma=None`, `degree=3`,
`coef0=1`, `random_state`.

```{warning}
Nyström samples its landmarks from the training rows, so `n_components` cannot exceed
`n_samples`. If you fit on fewer rows than `n_components` (for example a small
cross-validation fold), scikit-learn silently clamps `n_components` down to `n_samples` and
emits a `UserWarning` rather than raising: the fitted output width is `min(n_components,
n_samples_seen_in_fit)`. Random Fourier features have no such limit, since they draw a random
basis instead of sampling training rows.
```

```{note}
Both methods approximate the same idea from different angles: random Fourier features draw a
random basis independent of the data, while Nyström samples landmarks from the data itself.
When in doubt, try both and compare with cross-validation.
```

```{note}
Nyström's approximation error is not uniformly bounded across the kernel matrix. In
particular, the self-similarity entries $K(x, x)$ on the diagonal can be approximated far
less accurately than typical off-diagonal entries, depending on how well the sampled
landmarks happen to cover that row. This is an inherent property of the Nyström method
itself (also present in plain `sklearn.kernel_approximation.Nystroem`), not something
specific to PreTab's wrapper. If your downstream model is sensitive to diagonal accuracy,
increase `n_components` or try random Fourier features instead.
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
