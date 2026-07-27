# Representations overview

This section is the catalogue of every representation PreTab ships. Each family turns raw
columns into an expressive basis, and they all share the same vocabulary and the same
scikit-learn API. Start here to see the landscape, then dive into the family that fits your
data.

## The families

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Splines
:link: splines
:link-type: doc
Smooth, locally-supported bases: B, M, I, cubic regression, natural cubic, penalized
(P-spline), and the multivariate tensor-product and thin-plate splines.
:::

:::{grid-item-card} Feature maps
:link: feature_maps
:link-type: doc
Basis functions from machine learning: radial (RBF), ReLU, sigmoid, tanh, deterministic
Fourier, and the kernel approximations (random Fourier features, Nyström).
:::

:::{grid-item-card} Binning and PLE
:link: binning_and_ple
:link-type: doc
Discretization: numeric binning with several encodings, and supervised piecewise-linear
encoding (PLE).
:::

:::{grid-item-card} Categorical
:link: categorical
:link-type: doc
Ordinal and one-hot encoding, plus pretrained language embeddings for high-cardinality text.
:::

::::

## Shared terminology

Every family is described with the same terms, introduced in
[Preprocessing and representation](../core_concepts/feature_representation.md).

`scope`
: `univariate` methods transform one column at a time. `multivariate` methods (tensor-product
  spline, thin-plate spline, random Fourier features, Nyström) model several columns jointly
  and are used standalone, not per column through `Preprocessor`.

`supervision`
: `forbidden`, `optional`, or `required` target usage. See
  [Target awareness](../core_concepts/target_awareness.md).

`output_dim`
: The width of the expansion. See
  [Resolution and placement](../core_concepts/resolution_and_placement.md).

`placement`
: Where the knots, centers, or edges go, chosen by `target_aware` and `placement_strategy`.

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

## A note on scientific grounding

Every family rests on established theory, from B-splines and P-splines to thin-plate
regression splines and random Fourier features. The [references](references.md) page collects
the primary sources for each, so the representations are traceable to their literature.

## Where to go next

- [Splines](splines.md), [Feature maps](feature_maps.md), [Binning and PLE](binning_and_ple.md),
  [Categorical](categorical.md) for the families.
- [Comparison table](comparison_table.md) to filter by capability.
- [Choosing a method](choosing_a_method.md) for guidance and failure modes.
