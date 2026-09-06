# Representations overview

This section is the catalogue of every representation PreTab ships. Each family turns raw
columns into an expressive basis, and they all share the same vocabulary and the same
scikit-learn API. Start here to see the landscape, then dive into the family that fits your
data.

## The families

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Spline expansions
:link: spline_expansions
:link-type: doc
Smooth, locally-supported bases: B, M, I, cubic regression, natural cubic, penalized
(P-spline), and the multivariate tensor-product and thin-plate splines.
:::

:::{grid-item-card} Functional expansions
:link: functional_expansions
:link-type: doc
Basis functions from machine learning: radial (RBF), ReLU, sigmoid, tanh, and deterministic
Fourier features.
:::

:::{grid-item-card} Kernel approximation
:link: kernel_approximation
:link-type: doc
Multivariate kernel machines without the full kernel matrix: random Fourier features and
Nyström.
:::

:::{grid-item-card} Numerical encoding
:link: numerical_encoding
:link-type: doc
Discretization and recoding: numeric binning, supervised piecewise-linear encoding (PLE), and
periodic encoding.
:::

:::{grid-item-card} Categorical encoding
:link: categorical_encoding
:link-type: doc
Ordinal and one-hot encoding for categories, handling unseen values without raising.
:::

:::{grid-item-card} Embeddings
:link: embeddings
:link-type: doc
Pretrained language embeddings for high-cardinality text categories.
:::

::::

## Shared terminology

Every family is described with the same terms, introduced in
[Preprocessing and representation](../core_concepts/feature_representation.md).

| Term          | Meaning                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scope`       | `univariate` methods transform one column at a time. `multivariate` methods such as tensor-product spline, thin-plate spline, random Fourier features, and Nyström model several columns jointly and are used standalone, not per column through `Preprocessor`. |
| `supervision` | Whether the method uses the target: `forbidden`, `optional`, or `required`. See [Target awareness](../core_concepts/target_awareness.md).                                                                                                                        |
| `output_dim`  | The width of the expansion. See [Resolution and placement](../core_concepts/resolution_and_placement.md).                                                                                                                                                        |
| `placement`   | Where the knots, centers, or edges are placed, chosen via `target_aware` and `placement_strategy`.                                                                                                                                                               |

## How to select a method

There are two ways to pick.

- **By intent**: read [Choosing a method](choosing_a_method.md) for practical guidance,
  including where basis expansion does not help.
- **By capability**: read the [comparison table](comparison_table.md) to filter families by
  feature kind, scope, supervision, and adaptivity.

You can also query the registry in code:

```python
from pretab import list_representations

list_representations(feature_kind="numerical", supervised=True)
```

## Supporting utilities

Some transformers in this section do not expand or recode a feature. Instead, they prepare
it for the rest of the pipeline.

- **Pass-through and type conversion**: `NoTransformer` leaves a column unchanged, and
  `ToFloatTransformer` converts it to floating point while keeping the same width. These are
  the minimal support operations that let a pipeline keep a column untouched or normalize its
  dtype without changing its structure.
- **Missing-value flagging**: `MissingStateIndicator` emits a binary mask marking where the
  input was missing, computed before imputation. `Preprocessor` uses this when
  `missing_policy="separate_state"` so a downstream model can learn a dedicated response to
  missingness instead of confusing it with an imputed value.

These utilities are part of the public API for custom pipelines and column-wise preprocessing,
but most users never instantiate them directly because `Preprocessor` wires them in
automatically.

## References

The representations in PreTab rest on established literature. The primary sources behind the
families are grouped below so each method is traceable to its origin.

### Splines and penalized splines

Eilers, P. H. C., and Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
_Statistical Science_, 11(2), 89-121.

Eilers, P. H. C., and Marx, B. D. (2003). Multivariate calibration with temperature
interaction using two-dimensional penalized signal regression. _Chemometrics and Intelligent
Laboratory Systems_, 66(2), 159-174.

These papers introduce the P-spline and its tensor-product extension, which underpin
`PSplineTransformer` and `TensorProductSplineTransformer`.

Kumar, M., Thielmann, A. F., Weisser, C., and Säfken, B. (2026). From uniform to learned
knots: A study of spline-based numerical encodings for tabular deep learning. _(TMLR)_.

This study motivates the task-dependent preset defaults used by `Preprocessor`:
`"bspline"` for regression and `"ple"` for classification.

### Thin-plate and generalized additive models

Wahba, G. (1990). _Spline Models for Observational Data_. Society for Industrial and Applied
Mathematics.

Wood, S. N. (2003). Thin plate regression splines. _Journal of the Royal Statistical Society:
Series B_, 65(1), 95-114.

Wood, S. N. (2017). _Generalized Additive Models: An Introduction with R_ (2nd ed.).
Chapman and Hall/CRC.

### Kernel approximations

Williams, C. K. I., and Seeger, M. (2001). Using the Nyström method to speed up kernel
machines. _Advances in Neural Information Processing Systems_, 13.

Rahimi, A., and Recht, B. (2007). Random features for large-scale kernel machines.
_Advances in Neural Information Processing Systems_, 20.

### Piecewise-linear encoding

Gorishniy, Y., Rubachev, I., and Babenko, A. (2022). On embeddings for numerical features in
tabular deep learning. _Advances in Neural Information Processing Systems_, 35.

This paper motivates the piecewise-linear encoding used by `PLETransformer`.

## Where to go next

- [Spline expansions](spline_expansions.md), [Functional expansions](functional_expansions.md),
  [Kernel approximation](kernel_approximation.md), [Numerical encoding](numerical_encoding.md),
  [Categorical encoding](categorical_encoding.md), [Embeddings](embeddings.md), and
  [Comparison table](comparison_table.md) for the families and capability filters.
- [Choosing a method](choosing_a_method.md) for guidance and failure modes.
- [Configuration](../core_concepts/configuration.md) for the preset and pipeline behavior that
  wires these representations together.
