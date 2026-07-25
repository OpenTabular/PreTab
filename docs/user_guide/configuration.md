# Hyperparameter and configuration guide

Every numerical transformer in PreTab shares a small set of hyperparameters. This guide explains
what each one does, the default it ships with, and how to choose a value that suits your data. If
there is one setting worth learning first, it is `output_dim`, which controls how wide each
feature becomes after transformation.

```{note}
When you work through the `Preprocessor`, its single `output_dim` (default `7`) is forwarded to
**every** numerical method. The per-transformer defaults listed below therefore only take effect
when you build a transformer directly, for example `RBFExpansionTransformer()`.
```

## The `output_dim` width knob

`output_dim` is the main capacity control. It sets the number of non-bias output columns produced
for each input feature: bins for PLE and binning, centers for the feature maps, and basis
functions for the splines. A larger value captures finer structure in a feature, at the cost of
more columns and a higher chance of overfitting. A smaller value is more compact and regularises
the representation.

The defaults are aligned on a moderate value of `6`. It is expressive enough for most features
while staying compact, and it clears the minimum width that every spline basis requires. The
tensor-product spline is the single exception: its columns multiply across marginal dimensions,
so it defaults to its smallest valid width to keep the output from exploding.

| Method | Class | Default | Minimum (floor) |
| --- | --- | --- | --- |
| PLE | `PLETransformer` | `6` | `1` (upper bound on bins; actual count is data-dependent) |
| RBF map | `RBFExpansionTransformer` | `6` | `1` |
| ReLU map | `ReLUExpansionTransformer` | `6` | `1` |
| Sigmoid map | `SigmoidExpansionTransformer` | `6` | `1` |
| Tanh map | `TanhExpansionTransformer` | `6` | `1` |
| Cubic spline | `CubicSplineTransformer` | `6` | `3` (3 polynomial terms + interior knots) |
| Natural cubic | `NaturalCubicSplineTransformer` | `6` | `2` (places `output_dim + 1` knots) |
| B/M/I splines | `BSplineTransformer`, `MSplineTransformer`, `ISplineTransformer` | `6` | `degree + 1` (=`4`), capped at `50` |
| P-spline | `PSplineTransformer` | `6` | `degree + 1` (=`4`) |
| Tensor product | `TensorProductSplineTransformer` | `4` | `degree + 1` (=`4`) **per marginal** |
| Thin-plate | `ThinPlateSplineTransformer` | `6` | `1` |
| Preprocessor (shared) | `Preprocessor` | `7` | overrides the per-transformer defaults above |

```{warning}
Each spline enforces its own minimum width. Requesting fewer basis functions than the floor in
the table raises an error at `fit` time instead of silently clamping, so keep `output_dim` at or
above that floor. The floor is `degree + 1` for the B, M, I, P-spline, and tensor-product bases.
```

```{tip}
For the tensor-product spline the width grows as the product across marginals. A 2-D input with
`output_dim=4` already produces `4 × 4 = 16` columns, so raise it in small steps and keep an eye
on the total column count.
```

## Adaptive sizing

PLE and the feature maps can size each feature from the data instead of using one fixed width.
This helps when your features differ a lot in complexity and you would rather not tune
`output_dim` by hand.

`adaptive`
: When `True`, the width for each feature is chosen from the data and kept inside
  `[min_output_dim, max_output_dim]`. Fixed-width methods such as the plain scalers ignore this
  flag.

`min_output_dim`, `max_output_dim`
: The lower and upper bounds that apply only when `adaptive=True`. They are ignored otherwise.

## Target-aware placement

Some methods can place their bins, centers, or knots using the target `y`. This tends to sharpen
the representation where the target actually changes, at the cost of needing labels at `fit`
time.

`target_aware`
: Whether placement uses the target. PLE is inherently target-aware. Every other family defaults
  to `target_aware=False`, which is fully unsupervised and fits without `y`.

`placement_strategy`
: How units are placed. The valid values depend on `target_aware`, as shown below.

| `target_aware` | Allowed `placement_strategy` | Default when unset |
| --- | --- | --- |
| `True` | `cart`, `lightgbm` | `cart` |
| `False` | `uniform`, `quantile` | `quantile` |

```{warning}
The two rows of this table are mutually exclusive. Combining them, for example
`target_aware=True` with `placement_strategy="quantile"`, raises an error. Leave
`placement_strategy` unset to get the sensible default for whichever mode you picked.
```

`task`
: Either `"regression"` or `"classification"`. It is only consulted by target-aware placement,
  which uses it to fit the selector against `y`.

## Spline-specific parameters

`degree`
: Degree of the spline basis, where `3` is cubic. It also sets the `output_dim` floor of
  `degree + 1` for the B, M, I, P-spline, and tensor-product bases, so a lower degree lowers the
  minimum width.

`include_bias`
: When `True`, a constant intercept column is prepended to the output. The bias term is left
  unpenalised.

## Reproducibility

`random_state`
: Seeds the target-aware selectors and any stochastic placement so that repeated fits produce
  identical output. Set it to an integer whenever you need deterministic results, for example in
  tests or published experiments.
