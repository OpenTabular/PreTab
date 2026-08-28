# Comparison table

Use this page to filter representations by capability. It is a static reference; for a live,
queryable view use `list_representations(...)` against the registry. The registry is the single
source of truth, and these tables mirror it.

## Reading the columns

`Key`
: The string you pass to `numerical_method`, `categorical_method`, or per-feature config.

`Scope`
: `univariate` (one column) or `multivariate` (several columns jointly).

`Target`
: `forbidden`, `optional` (used when `target_aware=True`), or `required`.

`Adaptive`
: Supports data-driven width selection between `min_output_dim` and `max_output_dim`.

`Penalty`
: Exposes `get_penalty_matrix()` for smoothing penalties.

`Selectable`
: Can be chosen through `Preprocessor` as a per-column method.

## Numerical: scalers and simple transforms

| Method | Key | Scope | Target | Selectable |
| --- | --- | --- | --- | --- |
| Standardization | `standardization` | univariate | forbidden | yes |
| Min-max scaling | `minmax` | univariate | forbidden | yes |
| Robust scaling | `robust` | univariate | forbidden | yes |
| Quantile transform | `quantile` | univariate | forbidden | yes |
| Polynomial features | `polynomial` | univariate | forbidden | yes |
| Box-Cox | `box-cox` | univariate | forbidden | yes |
| Yeo-Johnson | `yeo-johnson` | univariate | forbidden | yes |
| Passthrough | `none` | univariate | forbidden | yes |

## Spline expansions

| Method | Key | Scope | Target | Adaptive | Penalty | Selectable |
| --- | --- | --- | --- | --- | --- | --- |
| B-spline | `bspline` | univariate | optional | yes | yes | yes |
| M-spline | `mspline` | univariate | optional | yes | yes | yes |
| I-spline | `ispline` | univariate | optional | yes | yes | yes |
| Cubic regression spline | `cubicspline` | univariate | optional | yes | yes | yes |
| Natural cubic spline | `naturalspline` | univariate | optional | yes | yes | yes |
| Penalized spline (P-spline) | `pspline` | univariate | forbidden | yes | yes | yes |
| Tensor-product spline | `tensorspline` | multivariate | forbidden | yes | yes | no |
| Thin-plate spline | `tprs` | multivariate | forbidden | no | yes | no |

```{note}
The multivariate splines (`tensorspline`, `tprs`) model several inputs jointly and are used
standalone, not selected per column through `Preprocessor`. The alias `thinplate` resolves to
`tprs`.
```

## Functional expansions

| Method | Key | Scope | Target | Adaptive | Selectable |
| --- | --- | --- | --- | --- | --- |
| RBF expansion | `rbf` | univariate | optional | yes | yes |
| ReLU expansion | `relu` | univariate | optional | yes | yes |
| Sigmoid expansion | `sigmoid` | univariate | optional | yes | yes |
| Tanh expansion | `tanh` | univariate | optional | yes | yes |
| Fourier features | `fourier` | univariate | forbidden | no | yes |

## Kernel approximation

| Method | Key | Scope | Target | Adaptive | Selectable |
| --- | --- | --- | --- | --- | --- |
| Random Fourier features | `rff` | multivariate | forbidden | no | no |
| Nyström kernel map | `nystroem` | multivariate | forbidden | no | no |

```{note}
Random Fourier features and Nyström model the whole input matrix jointly and are used
standalone, not selected per column through `Preprocessor`.
```

## Numerical encoding

| Method | Key | Scope | Target | Adaptive | Selectable |
| --- | --- | --- | --- | --- | --- |
| Numeric binning | `custombin` | univariate | forbidden | no | yes |
| Piecewise-linear encoding (PLE) | `ple` | univariate | required | yes | yes |
| Periodic encoding | n/a | univariate | forbidden | no | no |

```{important}
PLE is the only numerical method that **requires** the target. It always places its bins
against `y`, so it must be fit with a target and is best used with cross-fitting. See
[Target awareness](../core_concepts/target_awareness.md).
```

```{note}
Periodic encoding has no registry key: it takes a required per-feature `period`, so it is not
selectable through `Preprocessor`. Instantiate `PeriodicEncodingTransformer` directly.
```

## Categorical encoding

| Method | Key | Scope | Target | Selectable |
| --- | --- | --- | --- | --- |
| Ordinal (integer) encoding | `int` | univariate | forbidden | yes |
| One-hot encoding | `one-hot` | univariate | forbidden | yes |
| One-hot from ordinal | `onehot_from_ordinal` | univariate | forbidden | yes |
| Passthrough | `none` | univariate | forbidden | yes |

```{note}
The alias `ohe` resolves to `one-hot`.
```

## Embeddings

| Method | Key | Scope | Target | Selectable |
| --- | --- | --- | --- | --- |
| Pretrained language embedding | `pretrained` | univariate | forbidden | yes |

```{note}
`pretrained` requires the optional `embeddings` extra.
```

## Where to go next

- [Choosing a method](choosing_a_method.md) for guidance on which of these to reach for.
- [Spline expansions](spline_expansions.md), [Functional expansions](functional_expansions.md),
  [Kernel approximation](kernel_approximation.md), [Numerical encoding](numerical_encoding.md),
  [Categorical encoding](categorical_encoding.md), [Embeddings](embeddings.md) for the details.
