# Spline expansions

Splines are piecewise-polynomial bases with local support. They turn a single numerical column
into a set of smooth, overlapping basis functions, so a linear model on top can bend to follow
the data while staying stable. PreTab ships the full family, from the workhorse B-spline to the
multivariate thin-plate spline.

## The idea

A spline places a set of **knots** along the range of a feature and builds basis functions
between them. The transformed feature is the vector of basis values,

$$
x \mapsto \big(B_1(x),\ B_2(x),\ \dots,\ B_K(x)\big),
$$

where each $B_k$ is nonzero only near a few knots. Local support is what keeps splines stable:
a point in one region does not disturb the fit in another. Width is set by `output_dim` and
knot positions by `placement_strategy` (see
[Resolution and placement](../core_concepts/resolution_and_placement.md)).

```{important}
For every univariate spline in this section, `output_dim` is the number of output columns
**per input feature**, not the total. A `(n_samples, 3)` input produces
`(n_samples, 3 * output_dim)` output (plus one extra column per feature if
`include_bias=True`). Feature names are suffixed per input, for example `x0_bs0, x0_bs1, ...`
for a B-spline on column `x0`.
```

## B-spline

The B-spline is the default general-purpose smooth basis. Its functions are non-negative,
sum to one, and each spans only `degree + 1` knot intervals. For knots $\tau$ and degree $p$,
the basis follows the standard Cox-de Boor recursion,

$$
B_{i,0}(x) = \begin{cases} 1 & \tau_i \le x < \tau_{i+1} \\ 0 & \text{otherwise} \end{cases}, \qquad
B_{i,p}(x) = \frac{x - \tau_i}{\tau_{i+p} - \tau_i} B_{i,p-1}(x) + \frac{\tau_{i+p+1} - x}{\tau_{i+p+1} - \tau_{i+1}} B_{i+1,p-1}(x).
$$

```python
import numpy as np
from pretab.transformers import BSplineTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)   # (50, 1)
t = BSplineTransformer(output_dim=8, degree=3, placement_strategy="quantile")
t.fit_transform(X).shape
# (50, 8)
```

Constructor highlights: `output_dim`, `degree=3`, `include_bias=False`, `knot_locations=None`
(pass explicit knots to override placement), `target_aware=False`, `placement_strategy="quantile"`,
`adaptive`, `random_state`.

**Parameter impact.**

`degree`
: Sets the minimum usable `output_dim`: PreTab requires `output_dim >= degree + 1` (a cubic,
`degree=3`, needs at least 4 columns) and raises a typed error otherwise. Higher degree gives
smoother, wider-support basis functions at the same `output_dim`; `degree=1` recovers
piecewise-linear segments.

`output_dim`
: The exact per-feature output width (unlike the cubic/natural/tensor families below, no
conversion is applied). More columns track finer local detail and increase overfitting risk.

`include_bias`
: Defaults to `False`. A B-spline basis over a clamped knot vector already sums to 1 in every
row (a partition of unity), so prepending a bias column makes the design exactly
rank-deficient. Set `include_bias=True` only if a downstream model specifically needs an
explicit intercept column; it adds one extra output column.

```{tip}
Cubic (`degree=3`) B-splines with quantile knots are a strong default for smooth regression.
Increase `output_dim` for more wiggle, decrease it to regularize.
```

```{note}
Every spline in this family exposes `get_penalty_matrix(feature_index=0, diff_order=2)`, a
`D^T D` second-difference penalty matrix of shape `(output_dim, output_dim)` (plus the bias
row/column left unpenalized when `include_bias=True`). It is not limited to the "penalized"
splines further down this page. Note that `diff_order` is specific to this B/M/I family; the
cubic, natural cubic, P-spline, tensor-product, and thin-plate variants expose
`get_penalty_matrix(feature_index=0)` with their own fixed penalty construction instead.
```

## M-spline and I-spline

These two share the B-spline machinery but target special shapes.

M-spline
: A non-negative spline basis (`include_bias=False`). Useful when the components themselves
should be non-negative, for example as a density-like basis. Built by rescaling each B-spline
basis function so it integrates to one over its support,

$$
M_k(x) = \frac{p + 1}{\tau_{k+p+1} - \tau_k}\, B_k(x).
$$

I-spline
: The integral of an M-spline, giving a **monotone** basis. A model with non-negative
coefficients on an I-spline basis is guaranteed monotone in the input, which is valuable when
domain knowledge says a relationship cannot reverse.

$$
I_k(x) = \int_{\tau_k}^{x} M_k(t)\, dt.
$$

```python
import numpy as np
from pretab.transformers import ISplineTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
t = ISplineTransformer(output_dim=10, degree=3)  # monotone basis
t.fit_transform(X).shape
# (50, 10)
```

Both share the same `degree`/`output_dim` constraint and parameter set as the B-spline above
(`output_dim >= degree + 1`, `include_bias=False` by default).

```{note}
I-splines only guarantee monotonicity when the downstream coefficients are constrained to be
non-negative. Pair them with a non-negative linear model.
```

## Cubic regression and natural cubic splines

These are penalized-ready cubic bases with a clear knot interpretation, and both expose a
smoothing penalty through `get_penalty_matrix()`.

Cubic regression spline
: A cubic basis parameterized at the knots (`cubicspline`), convenient for GAM-style additive
models. Requires `output_dim >= 3`. The basis stacks the polynomial terms with one truncated
cubic term per interior knot $\kappa_j$,

$$
\big(x,\ x^2,\ x^3,\ (x - \kappa_1)_+^3,\ \dots,\ (x - \kappa_K)_+^3\big), \qquad (z)_+ = \max(0, z).
$$

Natural cubic spline
: A cubic spline constrained to be **linear beyond the boundary knots** (`naturalspline`).
The linear tails reduce the wild behaviour ordinary cubics show near the edges of the data.
Requires `output_dim >= 2`. For knots $\xi_1, \dots, \xi_T$ ($\xi_1$, $\xi_T$ the boundary
knots), the basis stacks $x$ with one constrained term per interior knot $\xi_k$,

$$
d_k(x) = \frac{(x - \xi_k)_+^3 - (x - \xi_T)_+^3}{\xi_T - \xi_k}, \qquad
N_k(x) = d_k(x) - \frac{\xi_T - \xi_k}{\xi_T - \xi_1} d_1(x) - \frac{\xi_k - \xi_1}{\xi_T - \xi_1} d_T(x).
$$

```python
import numpy as np
from pretab.transformers import NaturalCubicSplineTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
t = NaturalCubicSplineTransformer(output_dim=8)
X2 = t.fit_transform(X)
X2.shape                     # (50, 8): output width equals output_dim exactly
t.n_knots_                    # [7]: fitted interior-knot count per feature (output_dim - 1)
t.get_penalty_matrix().shape  # (8, 8)
```

```{tip}
Prefer the natural cubic spline when your feature has sparse data near its extremes; the linear
tails behave far better than an unconstrained cubic there.
```

## Penalized spline (P-spline)

The P-spline combines a B-spline basis with a difference penalty on adjacent coefficients,
following Eilers and Marx. Instead of controlling smoothness only through the number of knots,
it uses many knots and a penalty of order `diff_order` to keep the fit smooth. The basis
functions $B_k$ are exactly the B-spline basis above; fitting a linear model with coefficients
$\beta$ on top penalizes the loss with

$$
\lambda\, \beta^\top D^\top D\, \beta,
$$

where $D$ is the `diff_order`-th order difference operator (the same matrix returned by
`get_penalty_matrix()`) and $\lambda$ is chosen by the downstream penalized model, not by
`PSplineTransformer` itself.

```python
import numpy as np
from pretab.transformers import PSplineTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
t = PSplineTransformer(output_dim=8, degree=3, diff_order=2)
t.fit_transform(X).shape      # (50, 8)
t.get_penalty_matrix().shape  # (8, 8)
```

Constructor highlights: `output_dim`, `degree=3`, `diff_order=2`, `include_bias=False`,
`placement_strategy="uniform"`, `adaptive`. The P-spline is unsupervised; it does not read the
target.

**Parameter impact.** `diff_order` sets the order of the penalty: `diff_order=1` penalizes
changes in level between adjacent coefficients (favors flat fits), `diff_order=2` (the default)
penalizes changes in slope (favors locally-linear fits), and higher orders favor progressively
smoother curves. Requires `output_dim >= degree + 1`, the same floor as B-spline.

```{note}
The P-spline decouples smoothness from knot count. Use a generous `output_dim` and let the
penalty do the regularizing. Its penalty matrix plugs directly into penalized linear models.
```

## Multivariate splines

Two families model several inputs jointly. They are used standalone, not selected per column
through `Preprocessor`.

### Tensor-product spline

Builds a joint basis over multiple inputs as the tensor product of per-axis bases, capturing
interactions on a smooth grid. Each per-axis marginal is a B-spline basis (above), and the
joint basis function for multi-index $(k_1, \dots, k_d)$ over $d$ input columns $x_1, \dots, x_d$
is their product,

$$
\Phi_{k_1, \dots, k_d}(x_1, \dots, x_d) = \prod_{j=1}^{d} B_{k_j}(x_j).
$$

It exposes an anisotropic penalty per marginal via
`get_penalty_matrix(feature_index=...)`.

```python
import numpy as np
from pretab.transformers import TensorProductSplineTransformer

rng = np.random.default_rng(0)
X2 = rng.uniform(-3, 3, size=(200, 2))   # two input columns, e.g. lat/lon
t = TensorProductSplineTransformer(output_dim=5, degree=3, diff_order=2)
t.fit_transform(X2).shape                # (200, 25)
t.get_penalty_matrix(feature_index=0).shape  # (5, 5), one marginal
```

```{warning}
`output_dim` here is **per input dimension**, and the total width is `output_dim ** n_dims`.
With two columns and `output_dim=5` the result has `5 ** 2 = 25` columns; with three columns it
would be `125`. Keep `output_dim` small as the number of joint inputs grows, or the output width
explodes.
```

### Thin-plate spline

A thin-plate regression spline, the smooth-surface method from generalized additive models. It
places landmarks (by default with k-means) and forms a low-rank basis from the leading
eigenvectors of a projected radial-kernel matrix between the data and the landmarks. The radial
kernel $\eta(r)$ depends on the input dimension $d$:

$$
\eta(r) = \begin{cases} r^3 & d = 1 \\ r^2 \log r & d = 2 \\ r & d \ge 3 \end{cases}
$$

where $r = \lVert x - z_j \rVert$ is the distance from $x$ to landmark $z_j$. This follows the
low-rank thin-plate regression spline of Wood (2003).

```python
import numpy as np
from pretab.transformers import ThinPlateSplineTransformer

rng = np.random.default_rng(0)
X2 = rng.uniform(-3, 3, size=(200, 2))
t = ThinPlateSplineTransformer(n_components=10, landmark_strategy="kmeans")
t.fit_transform(X2).shape   # (200, 10): output width is n_components, not input-dependent
```

Constructor highlights: `n_components=10`, `landmark_strategy="kmeans"`, `rank_strategy="eigen"`,
`include_bias=False`, `random_state`.

```{warning}
Unlike the other spline families on this page, `get_penalty_matrix()` on
`ThinPlateSplineTransformer` is experimental: the retained eigenvalues are not guaranteed
non-negative, so the returned penalty is not guaranteed positive semi-definite, and a
`ConfigWarning` is raised on every call. `transform()` is unaffected; avoid this penalty for
actual smoothing regularization until it is fully fixed.
```

```{warning}
The tensor-product and thin-plate splines are multivariate. They are standalone transformers
and are not available as a per-column `numerical_method`. Fit them directly on the columns you
want to model jointly.
```

## Where to go next

- [Functional expansions](functional_expansions.md) for non-spline bases.
- [Multivariate features tutorial](../tutorials/multivariate_features.md) for a worked joint
  model.
- [Representations overview](overview.md) for the primary spline literature and supporting notes.
