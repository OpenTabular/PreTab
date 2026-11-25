# Numerical encoding

Encoding recodes a numeric value rather than expanding it into a smooth basis. PreTab covers
three flavors: unsupervised discretization (numeric binning), supervised piecewise-linear
encoding (PLE), and periodic encoding for values that wrap around a known cycle. Discretization
captures sharp, threshold-like effects that smooth bases blur, and is the natural choice when a
feature acts in steps.

## Numeric binning

Numeric binning splits a feature into intervals and encodes which interval each value falls
into. You choose how the edges are placed and how the result is encoded.

```python
import numpy as np
from pretab.transformers import NumericBinningTransformer

X = np.random.default_rng(0).uniform(size=(100, 1))   # (100, 1)
onehot = NumericBinningTransformer(output_dim=8, encode="onehot", placement_strategy="quantile")
assert onehot.fit_transform(X).shape == (100, 8)

ordinal = NumericBinningTransformer(output_dim=8, encode="ordinal", placement_strategy="quantile")
assert ordinal.fit_transform(X).shape == (100, 1)

tied = np.repeat([0, 1, 2], 3).reshape(-1, 1)
for encoding in ("onehot", "soft"):
  binner = NumericBinningTransformer(output_dim=8, encode=encoding, placement_strategy="quantile")
  assert binner.fit_transform(tied).shape == (9, 2)
  assert binner.n_bins_ == [2]
```

`output_dim` requests a bin count per input feature. Quantile placement merges duplicate
edges, so tied data can yield fewer bins than requested. A constant feature falls back to
one bin. Inspect `n_bins_` for the fitted counts. The `encode` parameter determines how those
bins become output columns:

`"ordinal"`
: One integer column per input feature, giving the bin index regardless of the bin count.

`"onehot"`
: One indicator column per fitted bin.

`"soft"`
: A soft assignment that spreads each value across neighbouring bins, so the boundaries are not
hard. It emits one column per fitted bin, like `"onehot"`, but with fractional memberships.

Edge placement follows `placement_strategy`: `"uniform"` for equal-width bins, `"quantile"`
for equal-frequency bins. See
[Resolution and placement](../core_concepts/resolution_and_placement.md).

```{tip}
When the feature has few ties, quantile edges give bins similar sample counts, which can be
more stable than equal-width bins on skewed data.
```

## Piecewise-linear encoding

PLE is the flagship supervised representation. It fits a decision tree of the feature against
the target, reads the split points as bin edges, and encodes each value as its **linear
position within its bin**. The result is a piecewise-linear function that bends exactly where
the target changes, following the tabular deep-learning work of Gorishniy and colleagues.

```python
import numpy as np
from pretab.transformers import PLETransformer

X = np.random.default_rng(0).uniform(size=(100, 1))
y = np.random.default_rng(0).integers(0, 2, size=100)
t = PLETransformer(output_dim=12, task="classification")
assert t.fit_transform(X, y).shape == (100, 12)
assert t.total_output_dim_ == 12
```

Constructor highlights: `output_dim`, `placement_strategy="cart"`, `task="regression"`,
`adaptive`, and `random_state=51`.

```{important}
PLE **requires** the target. It places its edges using `y`, so it must be fit with a target
and should be fit leakage-safely, ideally with cross-fitting. See
[Target awareness](../core_concepts/target_awareness.md).
```

```{warning}
The standalone fit above emits `LeakageWarning` because PLE uses `y` to place its bins. For
evaluation, split before fitting or pass the entire scikit-learn `Pipeline` to
cross-validation. A plain pipeline suppresses the warning but still uses each training
row's target when fitting its bins. For out-of-fold training features, wrap PLE in
`pretab.CrossFittedTransformer` and use `fit_transform(X, y)`.
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

## Periodic encoding

When a feature wraps around a known cycle, such as hour of day or month of year, the periodic
encoder maps each value onto its position on that cycle using sine and cosine harmonics. This
keeps the boundary continuous, so December and January sit next to each other instead of at
opposite ends of a number line.

```python
import numpy as np
from pretab.exceptions import PretabDataError
from pretab.transformers import PeriodicEncodingTransformer

X = np.array([[0], [6], [12], [18], [24]])   # hour-of-day style values
t = PeriodicEncodingTransformer(period=24, harmonics=2)  # e.g. hour of day
encoded = t.fit_transform(X)
assert encoded.shape == (5, 4)
np.testing.assert_allclose(encoded[0], encoded[-1], atol=1e-12)
np.testing.assert_allclose(t.transform([[25], [-1]]), t.transform([[1], [23]]), atol=1e-12)

with np.testing.assert_raises(PretabDataError):
  PeriodicEncodingTransformer(period=24).fit([[0], [25]])
```

Constructor highlights: `period` (required, the cycle length), `harmonics=1`,
`include_original=False`.

**Parameter impact.** Output width is `2 * harmonics` (one sine/cosine pair per harmonic), plus
one extra column when `include_original=True`. Higher `harmonics` lets the encoding represent
finer-grained sub-cycles (for example distinguishing morning from afternoon within a day), at
the cost of a wider output.

```{important}
Supply the period explicitly. `fit` does not infer it and raises `PretabDataError` if a
training value falls outside `[0, period]`. Both endpoints are accepted and their harmonic
columns agree up to floating-point precision. After fitting, `transform` accepts values
outside that interval and the sine and cosine terms wrap around the cycle. If
`include_original=True`, the extra raw column is not wrapped.
```

```{note}
Periodic encoding is a standalone time-series utility. It is not wired into `Preprocessor`
because it requires a per-feature `period`, so apply it directly to the relevant cyclical
column. [Fourier features](functional_expansions.md#fourier-features) instead set their base
frequency from the fitted range; they do not infer a cycle length from the data.
```

## Binning versus PLE

|                       | Numeric binning                    | PLE                         |
| --------------------- | ---------------------------------- | --------------------------- |
| Uses the target       | No                                 | Yes (required)              |
| Within-bin resolution | Lost (hard) or blurred (soft)      | Preserved (linear)          |
| Edge placement        | Uniform or quantile                | Target-driven (tree splits) |
| Best for              | Unsupervised, known step structure | Supervised sharp effects    |

## Where to go next

- [Target awareness](../core_concepts/target_awareness.md) for fitting PLE safely.
- [Spline expansions](spline_expansions.md) for smooth alternatives to binning.
- [Functional expansions](functional_expansions.md) for Fourier features, the deterministic
  alternative to periodic encoding.
- [Representations overview](overview.md) for the literature and supporting notes.
