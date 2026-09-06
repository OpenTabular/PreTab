# Preprocessing and representation

PreTab draws a deliberate line between two ideas that are often blurred together:
_preprocessing_ and _representation_. The distinction shapes the whole library.

- **Preprocessing prepares a column** without changing what it _means_.
- **Representation exposes structure** a plain estimator cannot weight on its own.

| Aspect                         | Preprocessing                                              | Representation                                                                     |
| ------------------------------ | ---------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Purpose                        | Make a column safe and comparable for a model              | Expose structure a plain estimator cannot weight on its own                        |
| Relationship to input          | One-to-one: same signal, rescaled or cleaned               | One-to-many: expands into a new basis                                              |
| Output width per column        | Unchanged (one-hot excepted, one per category)             | Grows to `output_dim` basis columns                                                |
| PreTab methods                 | `minmax`, `standardization`, `robust`, `one-hot`, imputers | `bspline`, `naturalspline`, `ple`, `rbf`, `fourier`, and the rest of the catalogue |
| Changes what the column means? | No                                                         | Yes: re-expresses it in a new coordinate system                                    |

The same input column makes this concrete. Scaling keeps the column a single coordinate;
a spline basis turns it into several.

```python
import numpy as np
from pretab import Preprocessor

age = np.random.default_rng(0).uniform(18, 65, size=100).reshape(-1, 1)
y = np.random.default_rng(0).normal(size=100)

scaled = Preprocessor(numerical_method="minmax").fit_transform(age, y)
scaled.shape
```

```text
(100, 1)
```

```python
expanded = Preprocessor(
    numerical_method="bspline", output_dim=6,
    target_aware=False, placement_strategy="quantile",
).fit_transform(age, y)
expanded.shape
```

```text
(100, 6)
```

`minmax` preprocesses `age` into a single rescaled column: same meaning, safe range.
`bspline` represents the same column as 6 local basis functions a linear model can weight
independently, capturing shapes a single rescaled slope cannot.

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

| Term          | Meaning                                                                                                                                                                          |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `family`      | The kind of representation, for example spline, feature map, binning, periodic, or categorical.                                                                                  |
| `scope`       | Whether the representation transforms one column at a time (`univariate`) or models several columns jointly (`multivariate`), such as the tensor-product and thin-plate splines. |
| `supervision` | Whether placement can (`optional`) or must (`required`) use the target, or never does (`forbidden`).                                                                             |
| `output_dim`  | The width of the expansion: the number of basis functions, centers, or bins per input feature. See [Resolution and placement](resolution_and_placement.md).                      |
| `locations`   | The data-driven positions the basis is anchored at: knots for splines, centers for feature maps, edges for bins.                                                                 |

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
