# Preprocessing and representation

PreTab draws a deliberate line between two ideas that are often blurred together:
*preprocessing* and *representation*. Understanding the distinction explains why the library
is shaped the way it is, and it is the key to using it well.

## Preprocessing prepares a column

Preprocessing makes a column safe and comparable for a model. It does not change what the
column *means*, only its scale, dtype, or completeness. Standardizing to zero mean and unit
variance, imputing a missing value, casting to float, and one-hot encoding a category are all
preprocessing. Each keeps a one-to-one relationship with the original signal.

## Representation changes what the model can see

A representation expands a column into a new basis that exposes structure a plain estimator
cannot weight on its own. A single numeric column becomes a set of spline coefficients, a
bank of radial bumps, a stack of piecewise-linear bins, or a pair of sine and cosine values.
The model now has several coordinates to weight where it previously had one slope, so it can
express curves, thresholds, saturation, and periodicity.

```{note}
This is the load-bearing idea in PreTab: the model is often fine, the *representation* is
what is missing. A linear model with an expressive basis can fit shapes that the same model
on raw columns cannot.
```

## Why the distinction matters

Keeping the two separate has practical consequences that show up all over the API.

- **Scaling composes with representation.** A numeric column is typically imputed and scaled
  first (preprocessing), then expanded into a basis (representation). The `Preprocessor`
  wires this order for you.
- **Representations are self-describing.** Because an expansion is a real modelling choice,
  every fitted representation carries a typed [`RepresentationSpec`](../api/preprocessor.rst)
  and per-output-column [lineage](outputs_and_inspection.md), so you always know which input
  and which component produced each output column.
- **Some representations use the target.** Placing bins or knots where the target actually
  changes is a supervised decision, which is why leakage safety is a first-class concern. See
  [Target awareness](target_awareness.md).

## The shared vocabulary

Every representation family in PreTab is described with the same small set of terms. Learning
them once pays off across the whole catalogue.

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

All of this is captured in one typed object, the `RepresentationSpec`, which is the common
intermediate form across every family. It records the family, input and output features,
scope, supervision, width, degree, and locations, and it round-trips to and from a plain
dict. Feature lineage then maps each individual output column back to its source. Together
they make a fitted PreTab pipeline fully inspectable and serializable.

```python
spec = transformer.get_representation_spec()
spec.family, spec.output_features, spec.locations
```

## Where to go next

- [Configuration](configuration.md) covers how you request representations.
- [Resolution and placement](resolution_and_placement.md) explains width and location.
- [Outputs and inspection](outputs_and_inspection.md) covers lineage and output formats.
- [Representations](../representations/overview.md) is the full catalogue of families.
