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
Without explicit knots, `output_dim` is the number of output columns for a univariate spline
**per input feature**, not the total. A `(n_samples, 3)` input produces
`(n_samples, 3 * output_dim)` output (plus one extra column per feature if
`include_bias=True`). Feature names are suffixed per input, for example `age_bs0, age_bs1, ...`
for a B-spline fitted on a DataFrame column `age` (`x0_bs0, x0_bs1, ...` for array input).
For B/M/I splines with explicit `knot_locations`, a feature
with `K` retained unique interior knots instead gets `K + degree + 1` non-bias columns;
different feature ranges can therefore produce different widths.
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
(pass explicit interior knots to override placement, `output_dim`, and the adaptive bounds:
repeated knots are merged, and knots on or outside a feature's fitted range are dropped with
a warning rather than replaced), `target_aware=False`, `placement_strategy="quantile"`,
`adaptive`, `random_state`.

**Parameter impact.**

`degree`
: For automatic placement, sets the minimum usable `output_dim`: PreTab requires
`output_dim >= degree + 1` (a cubic,
`degree=3`, needs at least 4 columns) and raises a typed error otherwise. Higher degree gives
smoother, wider-support basis functions at the same `output_dim`; `degree=1` recovers
piecewise-linear segments.

`output_dim`
: The exact per-feature output width (unlike the cubic/natural/tensor families below, no
conversion is applied). Ignored when `knot_locations` is provided. More columns track finer
local detail and increase overfitting risk.

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
The B/M/I family exposes `get_penalty_matrix(feature_index=0, diff_order=2)` and
the corresponding typed `get_penalty_spec()`. The returned matrix covers the
fitted output width, including a zero row and column for the leading bias if present.
See [Typed spline penalties](#typed-spline-penalties) for the other families.
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

where $D$ is the `diff_order`-th order difference operator. `get_penalty_matrix()`
returns $D^\top D$, padded for the optional unpenalized bias. The downstream model,
not `PSplineTransformer`, chooses $\lambda$ and adds the penalty to its loss.

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

Two families model several inputs jointly. Use them through
`Preprocessor(feature_groups=...)` for exclusive numerical groups, or fit their
standalone transformers directly. Neither is a per-column `numerical_method`.

### Tensor-product spline

Builds a joint basis over multiple inputs as the tensor product of per-axis bases, capturing
interactions on a smooth grid. Each per-axis marginal is a B-spline basis (above), and the
joint basis function for multi-index $(k_1, \dots, k_d)$ over $d$ input columns $x_1, \dots, x_d$
is their product,

$$
\Phi_{k_1, \dots, k_d}(x_1, \dots, x_d) = \prod_{j=1}^{d} B_{k_j}(x_j).
$$

`get_penalty_matrix(feature_index=...)` returns a marginal matrix.
`get_penalty_specs()` returns one full joint matrix per source direction, ready
to apply to the coefficients of the tensor basis.

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

```{note}
Use `feature_groups` to select the joint sources inside `Preprocessor`. A group
has one fitted width and one output block, including its raw missing states.
The [multivariate tutorial](../tutorials/multivariate_features.md) shows how to
configure groups and inspect the actual width rather than assigning it to each source.
```

## Typed spline penalties

Fit the spline first, then call its typed accessor. Each {py:class}`pretab.PenaltySpec`
pairs a penalty matrix with the ordered `coefficient_names` and zero-based
`coefficient_indices` of the fitted transformer's output. `axis` identifies the
input direction. It does not index the coefficients. The legacy matrix accessors
remain available, and neither accessor changes the fitted basis or trains a model.

| Spline family                | Typed accessor                                       | Matrix convention                                                                                           |
| ---------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| B-spline, M-spline, I-spline | `get_penalty_spec(feature_index=0, diff_order=None)` | Coefficient differences. Defaults to order 2; the accessor accepts an explicit order.                       |
| P-spline                     | `get_penalty_spec(feature_index=0)`                  | Coefficient differences of fitted order 1 or 2. Higher orders remain available through the legacy accessor. |
| Natural cubic                | `get_penalty_spec(feature_index=0)`                  | Approximate integrated squared second derivative, using finite differences and trapezoidal integration.     |
| Cubic regression             | `get_penalty_spec(feature_index=0)`                  | Integrated squared second derivative, using piecewise Simpson integration, exact in exact arithmetic.       |
| Tensor-product               | `get_penalty_specs()`                                | One full joint coefficient-difference matrix per fitted axis, using the fitted order.                       |
| Thin-plate                   | None                                                 | Experimental legacy matrix only; positive semidefiniteness is not guaranteed.                               |

For $m$ selected coefficients, `matrix` is a detached, read-only `float64`
array of shape $(m, m)$. Let $P$ denote that matrix and $\beta$ the coefficient
vector in the recorded order. The unweighted penalty is $\beta^\top P\beta$,
with no implicit factor of $1/2$. Symmetry, positive semidefiniteness and
numerical rank use the recorded tolerance `atol + rtol * spectral_norm`, where
the norm is of the symmetric part of $P$. Eigenvalues above that threshold
count toward `rank`; `nullity` is $m$ minus `rank`. Accepted entries are not
rounded, clipped, or symmetrized. A penalty can be singular without being invalid.

For coefficient differences, $P = D^\top D$ and the penalty is
$\lVert D\beta\rVert_2^2$. A difference penalty of order $r$ on $q$ non-bias
coefficients has rank $q-r$ and nullity $r$ when $1 \le r < q$; a leading
bias is unpenalized and adds one null direction. Differences operate on the
ordered coefficients, not derivatives in input units. The two curvature
penalties instead integrate over their fitted input domains. The linear term
and any explicit bias have zero curvature. Scaling inputs before fitting can
change these curvature matrices, even if the output width stays the same.

The tensor matrices share the full joint coefficient order (C order, with the
last declared axis varying fastest). Each direction applies a marginal
difference operator along its axis, holding the others fixed. With marginal
bias enabled, that direction leaves its own bias index unpenalized; other
directions may still penalize such terms. Apply the matrices to the same
joint coefficients and add their quadratic forms with nonnegative weights
$\lambda_j$. Do not substitute marginal matrices or add matrices defined in
different coefficient coordinates without first aligning them.

### Fit with a quadratic penalty

This example fits a P-spline basis, then uses its penalty in a least-squares
model. Because the example uses the **sum** of squared residuals, the normal
equations add $\lambda P$ to the design's Gram matrix. A mean-reduced loss uses
a different effective weight, so choose the weight for your loss convention.

```python
import numpy as np
from pretab.transformers import PSplineTransformer

source = np.linspace(0.0, 1.0, 80)
features = source.reshape(-1, 1)
target = np.sin(2.0 * np.pi * source)
spline = PSplineTransformer(output_dim=6, diff_order=2, include_bias=False).fit(features)
penalty = spline.get_penalty_spec()
indices = np.asarray(penalty.coefficient_indices)
assert tuple(spline.get_feature_names_out()[indices]) == penalty.coefficient_names
design = spline.transform(features)[:, indices]

weight = 0.1
matrix = penalty.matrix
coefficients = np.linalg.solve(design.T @ design + weight * matrix, design.T @ target)
residuals = design @ coefficients - target
roughness = coefficients @ matrix @ coefficients
loss = residuals @ residuals + weight * roughness

assert np.isfinite(loss)
np.testing.assert_allclose(roughness, np.sum(np.diff(coefficients, n=2) ** 2))
```

### Use in DeepTab and other estimators

Apply the quadratic form to the downstream weights that multiply the spline
basis columns directly, not to the input values or activations. `coefficient_indices`
are local to the fitted transformer's output. In a combined model input, use the
fitted block layout and output lineage to find the corresponding columns. For
a tensor group, keep the full joint basis in one block; do not split the penalty
by source or include raw missing indicators in it. The
[multivariate tutorial](../tutorials/multivariate_features.md) shows how to
inspect that fitted layout. This is consumer guidance, not a DeepTab configuration API.

For a tensor-based training loop, copy the read-only array to a fixed tensor
on the coefficients' device and dtype. If you have several coefficient vectors
for the same basis (for example, several output units), compute the form for each
vector and reduce the results consistently with your task loss. Recheck the
alignment after refitting or changing the design. `to_summary()` can log the
convention, coordinates, numerical rank and matrix checksum without including
the matrix. It is not a way to restore the matrix or train from the summary alone.

## Where to go next

- [Functional expansions](functional_expansions.md) for non-spline bases.
- [Multivariate features tutorial](../tutorials/multivariate_features.md) for a worked joint
  model.
- [Representations overview](overview.md) for the primary spline literature and supporting notes.
