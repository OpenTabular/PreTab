# Preprocessing and representation

PreTab draws a deliberate line between two ideas that are often blurred together:
*preprocessing* and *representation*. The distinction shapes the whole library.

## Preprocessing prepares a column

Preprocessing makes a column safe and comparable for a model, without changing what it
*means*. Standardizing to zero mean and unit variance, imputing a missing value, casting to
float, and one-hot encoding a category are all preprocessing: each keeps a one-to-one
relationship with the original signal.

## Representation exposes structure

A representation expands a column into a new basis that exposes structure a plain estimator
cannot weight on its own: spline coefficients, a bank of radial bumps, piecewise-linear bins,
or a sine/cosine pair. The model gets several coordinates to weight instead of one slope, so
it can express curves, thresholds, saturation, and periodicity.

```{note}
This is the load-bearing idea in PreTab: the model is often fine, the *representation* is
what is missing. A linear model with an expressive basis can fit shapes that the same model
on raw columns cannot.
```

## Why the distinction matters

- **Scaling composes with representation.** A numeric column is imputed and scaled first
  (preprocessing), then expanded into a basis (representation). `Preprocessor` wires this
  order for you.
- **Representations are self-describing.** Every fitted PreTab representation carries a typed
  [`RepresentationSpec`](../api/preprocessor.rst) and per-output-column
  [lineage](outputs_and_inspection.md), so you always know which input and which component
  produced each output column. Plain scikit-learn transformers used through
  `feature_preprocessing` (`StandardScaler`, `OneHotEncoder`, and so on) do not implement
  `get_representation_spec`; lineage falls back to step metadata for those columns instead.
- **Some representations use the target.** Placing bins or knots where the target actually
  changes is a supervised decision, which is why leakage safety is a first-class concern.

```{warning}
Target-aware placement can leak information if fit outside a proper train/validation split.
See [Target awareness](target_awareness.md) for how PreTab guards against this.
```

## The shared vocabulary

Every representation family is described with the same small set of terms.

`family`
: The kind of representation, for example spline, feature map, binning, periodic, or
  categorical.

`scope`
: Whether the representation transforms one column at a time (`univariate`) or models several
  columns jointly (`multivariate`), such as the tensor-product and thin-plate splines.

`supervision`
: Whether placement can (`optional`) or must (`required`) use the target, or never does
  (`forbidden`).

`output_dim`
: The width of the expansion, that is the number of basis functions, centers, or bins per
  input feature. See [Resolution and placement](resolution_and_placement.md).

`locations`
: The data-driven positions the basis is anchored at: knots for splines, centers for feature
  maps, edges for bins.

## The intermediate representation

Every family's fitted state is captured in one typed object, `RepresentationSpec`, the common
form across the whole catalogue. It records the family, input and output features, scope,
supervision, width, degree, and locations, and round-trips to and from a plain dict. Feature
lineage then maps each output column back to its source, making a fitted PreTab pipeline
fully inspectable and serializable.

```python
from pretab.transformers import NaturalCubicSplineTransformer
import numpy as np

transformer = NaturalCubicSplineTransformer(output_dim=6).fit(np.random.randn(100, 1))
spec = transformer.get_representation_spec()
spec.family, spec.output_features, spec.locations
```

## Where to go next

- [Configuration](configuration.md) covers how you request representations.
- [Resolution and placement](resolution_and_placement.md) explains width and location.
- [Outputs and inspection](outputs_and_inspection.md) covers lineage and output formats.
- [Representations](../representations/overview.md) is the full catalogue of families.
