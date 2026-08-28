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

## B-spline

The B-spline is the default general-purpose smooth basis. Its functions are non-negative,
sum to one, and each spans only `degree + 1` knot intervals.

```python
from pretab.transformers import BSplineTransformer

t = BSplineTransformer(output_dim=13, degree=3, placement_strategy="quantile")
```

Constructor highlights: `output_dim`, `degree=3`, `include_bias=False`, `knot_locations=None`
(pass explicit knots to override placement), `target_aware=False`, `placement_strategy="quantile"`,
`adaptive`, `random_state`.

```{tip}
Cubic (`degree=3`) B-splines with quantile knots are a strong default for smooth regression.
Increase `output_dim` for more wiggle, decrease it to regularize.
```

## M-spline and I-spline

These two share the B-spline machinery but target special shapes.

M-spline
: A non-negative spline basis (`include_bias=False`). Useful when the components themselves
  should be non-negative, for example as a density-like basis.

I-spline
: The integral of an M-spline, giving a **monotone** basis. A model with non-negative
  coefficients on an I-spline basis is guaranteed monotone in the input, which is valuable when
  domain knowledge says a relationship cannot reverse.

```python
from pretab.transformers import ISplineTransformer

t = ISplineTransformer(output_dim=10, degree=3)  # monotone basis
```

```{note}
I-splines only guarantee monotonicity when the downstream coefficients are constrained to be
non-negative. Pair them with a non-negative linear model.
```

## Cubic regression and natural cubic splines

These are penalized-ready cubic bases with a clear knot interpretation, and both expose a
smoothing penalty through `get_penalty_matrix()`.

Cubic regression spline
: A cubic basis parameterized at the knots (`cubicspline`), convenient for GAM-style additive
  models.

Natural cubic spline
: A cubic spline constrained to be **linear beyond the boundary knots** (`naturalspline`).
  The linear tails reduce the wild behaviour ordinary cubics show near the edges of the data.

```python
from pretab.transformers import NaturalCubicSplineTransformer

t = NaturalCubicSplineTransformer(output_dim=12)
penalty = t.get_penalty_matrix()   # for smoothing penalties
```

```{tip}
Prefer the natural cubic spline when your feature has sparse data near its extremes; the linear
tails behave far better than an unconstrained cubic there.
```

## Penalized spline (P-spline)

The P-spline combines a B-spline basis with a difference penalty on adjacent coefficients,
following Eilers and Marx. Instead of controlling smoothness only through the number of knots,
it uses many knots and a penalty of order `diff_order` to keep the fit smooth.

```python
from pretab.transformers import PSplineTransformer

t = PSplineTransformer(output_dim=20, degree=3, diff_order=2)
penalty = t.get_penalty_matrix()
```

Constructor highlights: `output_dim`, `degree=3`, `diff_order=2`, `include_bias=False`,
`placement_strategy="uniform"`, `adaptive`. The P-spline is unsupervised; it does not read the
target.

```{note}
The P-spline decouples smoothness from knot count. Use a generous `output_dim` and let the
penalty do the regularizing. Its penalty matrix plugs directly into penalized linear models.
```

## Multivariate splines

Two families model several inputs jointly. They are used standalone, not selected per column
through `Preprocessor`.

### Tensor-product spline

Builds a joint basis over multiple inputs as the tensor product of per-axis bases, capturing
interactions on a smooth grid. It exposes an anisotropic penalty.

```python
from pretab.transformers import TensorProductSplineTransformer

t = TensorProductSplineTransformer(output_dim=8, degree=3, diff_order=2)
X2 = t.fit_transform(X[["lat", "lon"]])
```

### Thin-plate spline

A thin-plate regression spline, the smooth-surface method from generalized additive models. It
places landmarks (by default with k-means) and forms a low-rank basis.

```python
from pretab.transformers import ThinPlateSplineTransformer

t = ThinPlateSplineTransformer(n_components=10, landmark_strategy="kmeans")
X2 = t.fit_transform(X[["lat", "lon"]])
```

Constructor highlights: `n_components=10`, `landmark_strategy="kmeans"`, `rank_strategy="eigen"`,
`include_bias=False`, `random_state`.

```{warning}
The tensor-product and thin-plate splines are multivariate. They are standalone transformers
and are not available as a per-column `numerical_method`. Fit them directly on the columns you
want to model jointly.
```

## Where to go next

- [Functional expansions](functional_expansions.md) for non-spline bases.
- [Multivariate features tutorial](../tutorials/multivariate_features.md) for a worked joint
  model.
- [References](references.md) for the primary spline literature.
