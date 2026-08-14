# Binning and PLE

Discretization turns a continuous feature into regions. It captures sharp, threshold-like
effects that smooth bases blur, and it is the natural representation when a feature acts in
steps. PreTab offers unsupervised numeric binning and supervised piecewise-linear encoding
(PLE).

## Numeric binning

Numeric binning splits a feature into intervals and encodes which interval each value falls
into. You choose how the edges are placed and how the result is encoded.

```python
from pretab.transformers import NumericBinningTransformer

t = NumericBinningTransformer(output_dim=8, encode="onehot", placement_strategy="quantile")
```

The `encode` parameter selects the output form.

`"ordinal"`
: A single integer column giving the bin index.

`"onehot"`
: One indicator column per bin.

`"soft"`
: A soft assignment that spreads each value across neighbouring bins, so the boundaries are not
  hard. This keeps a little of the smoothness that hard binning discards.

Edge placement follows `placement_strategy`: `"uniform"` for equal-width bins, `"quantile"`
for equal-frequency bins. See
[Resolution and placement](../core_concepts/resolution_and_placement.md).

```{tip}
Quantile edges give every bin a similar number of samples, which is usually more stable than
equal-width bins when the feature is skewed.
```

## Piecewise-linear encoding

PLE is the flagship supervised representation. It fits a decision tree of the feature against
the target, reads the split points as bin edges, and encodes each value as its **linear
position within its bin**. The result is a piecewise-linear function that bends exactly where
the target changes, following the tabular deep-learning work of Gorishniy and colleagues.

```python
from pretab.transformers import PLETransformer

t = PLETransformer(output_dim=12, task="regression")
X2 = t.fit_transform(x, y)   # y is required
```

Constructor highlights: `output_dim`, `placement_strategy="cart"`, `task="regression"`,
`adaptive`, `random_state=51`, and the tree controls `max_depth`, `min_samples_split`,
`min_samples_leaf`.

```{important}
PLE **requires** the target. It places its edges using `y`, so it must be fit with a target
and should be fit leakage-safely, ideally with cross-fitting. See
[Target awareness](../core_concepts/target_awareness.md).
```

### Why piecewise-linear rather than one-hot

Plain binning throws away where a value sits inside its bin; two values in the same interval
become identical. PLE keeps the within-bin position as a linear ramp, so it retains fine
resolution while still capturing the sharp transitions the tree found. That combination is why
it works so well as a front-end for both linear models and neural networks.

```{tip}
PLE is a strong default for numerical features, and it is the default `numerical_method` on
`Preprocessor`. Reach for it first when you have a supervised task and want the representation
to follow the target.
```

## Binning versus PLE

| | Numeric binning | PLE |
| --- | --- | --- |
| Uses the target | No | Yes (required) |
| Within-bin resolution | Lost (hard) or blurred (soft) | Preserved (linear) |
| Edge placement | Uniform or quantile | Target-driven (tree splits) |
| Best for | Unsupervised, known step structure | Supervised sharp effects |

## Where to go next

- [Target awareness](../core_concepts/target_awareness.md) for fitting PLE safely.
- [Splines](splines.md) for smooth alternatives to binning.
- [References](references.md) for the PLE source.
