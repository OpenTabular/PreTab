# Choosing a method

This page gives practical guidance for picking a representation, and it is honest about where
representations do not help. If you read only one page in this section, read this one.

## Start from the model

The right representation depends on what sits downstream.

Linear and additive models
: These gain the most from expansion. A linear model on top of a spline or PLE basis can fit
smooth nonlinearities while staying interpretable. This is the primary use case for PreTab.

Gradient-boosted trees
: Trees already partition each feature, so raw or lightly-scaled inputs are usually enough.
Expansion rarely helps and often adds noise. See
[when it does not help](#when-basis-expansion-does-not-help).

Neural networks
: PLE and learned embeddings are effective front-ends, echoing the tabular deep-learning
literature. Splines can help shallow networks.

## Match the method to the signal

| If the relationship is...         | Reach for...                             |
| --------------------------------- | ---------------------------------------- |
| Smooth and curved                 | B-spline, natural cubic spline, P-spline |
| Monotone (must not reverse)       | I-spline                                 |
| Sharp, threshold-like             | PLE, numeric binning, ReLU expansion     |
| Local bumps around centers        | RBF expansion                            |
| Periodic (known period)           | Periodic encoding, Fourier features      |
| A smooth surface over two inputs  | Tensor-product or thin-plate spline      |
| A general kernel over many inputs | Random Fourier features, Nyström         |

```{tip}
When unsure, start with the `"standard"` preset (min-max scaling, integer categoricals, and
`numerical_method` resolved from `task`: `"bspline"` for regression, `"ple"` for
classification) and compare against another method. The
[comparing representations tutorial](../tutorials/comparing_representations.md) shows how to
measure the difference instead of guessing.
```

## Match the method to the target

- If the relationship between a feature and the target is what you want to capture, a
  **target-aware** method (PLE, or a spline with `target_aware=True`) places its units where
  the target changes. Always fit these leakage-safely, see
  [Target awareness](../core_concepts/target_awareness.md).
- If you only want a flexible unsupervised basis, an **unsupervised** method (P-spline,
  Fourier, quantile-placed spline) avoids target usage entirely.

## Control the width

More columns means more flexibility and more overfitting risk. Start narrow and widen only if
validation improves. Turn on `adaptive=True` to let the data choose a width between
`min_output_dim` and `max_output_dim`. See
[Resolution and placement](../core_concepts/resolution_and_placement.md).

## When basis expansion does not help

Expansion is a tool, not a default. There are clear cases where it adds cost without value,
and pretending otherwise would be dishonest.

Tree ensembles already handle nonlinearity
: Gradient-boosted trees and random forests split each feature into regions on their own.
Feeding them a spline or binning basis usually leaves accuracy unchanged while multiplying
the column count. Prefer raw or scaled inputs for these models.

Truly linear relationships
: If a feature enters the target linearly, scaling is enough. A spline will fit the same line
with extra parameters and a little more variance.

Very small samples
: A wide expansion on a few hundred rows overfits. Keep `output_dim` small, or skip expansion
and rely on a scaled input.

Extrapolation beyond the fitted range
: Bases are fitted on the training range, and each spline family has its own default
transform-time behavior for values beyond it: B/M/I-spline, P-spline, and tensor-product
clip to the fitted range, while natural-cubic and cubic-regression extrapolate smoothly
(their basis is defined for any input). If your test data lies well beyond training, an
extrapolated value carries no real signal. Pass
`policy=RepresentationPolicy(out_of_range="clip")` (or `"warn"` / `"error"`) directly to any
of these spline transformers to override their default for that instance. This is
standalone-transformer-only: it is not yet threaded through `Preprocessor`. See the
edge-case behaviour below.

Pure noise features
: Expanding a feature that carries no signal only gives the model more ways to fit noise. Drop
the feature instead.

```{warning}
Basis expansion changes the geometry of your features, not the information in them. If a
feature does not carry the signal, no representation will create it. Measure, do not assume.
```

## Edge-case behaviour

PreTab is explicit about degenerate inputs rather than failing silently.

- **Constant column**: the splines that need spread (B/M/I, cubic, natural cubic, P-spline,
  tensor-product) raise a typed error rather than fitting a meaningless basis. Numeric
  binning falls back to a single bin instead of raising, and PLE and the feature maps place
  all their bins or centers at the same value, a degenerate but still valid basis.
- **Out-of-range input at transform**: B/M/I-spline, P-spline, and tensor-product clip
  values beyond the fitted range by default; natural-cubic and cubic-regression extrapolate
  smoothly by default. Pass `policy=RepresentationPolicy(out_of_range=...)` directly to any
  of these transformers to override its default with `"clip"`, `"warn"`, or `"error"`
  instead. This is standalone-transformer-only for now, not yet available through
  `Preprocessor`.
- **Unseen category**: `ContinuousOrdinalTransformer` (`"int"`) maps an unseen category to a
  reserved code rather than raising; one-hot encoding (`"one-hot"`) instead emits an all-zero
  row for it.
- **NaN into a finite-only method**: raises a typed error unless imputation is configured. See
  [Missing values](../core_concepts/missing_values.md).

## Non-goals

To set expectations, PreTab deliberately does not do the following.

- It is **not** a feature-selection library. It represents the features you give it; it does
  not decide which features to keep.
- It is **not** a modelling library. It produces representations; you bring the estimator.
- It does **not** invent signal. It reshapes existing information into a more learnable form.

## Where to go next

- [Comparison table](comparison_table.md) to filter by capability.
- [Spline expansions](spline_expansions.md), [Functional expansions](functional_expansions.md),
  [Kernel approximation](kernel_approximation.md), [Numerical encoding](numerical_encoding.md),
  [Categorical encoding](categorical_encoding.md), and [Embeddings](embeddings.md) for the
  details.
- [Comparing representations](../tutorials/comparing_representations.md) to measure the choice.
