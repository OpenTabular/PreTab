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
onehot.fit_transform(X).shape
# (100, 8): one column per bin

ordinal = NumericBinningTransformer(output_dim=8, encode="ordinal", placement_strategy="quantile")
ordinal.fit_transform(X).shape
# (100, 1): a single integer column, regardless of output_dim
```

The `encode` parameter selects the output form, and it changes the output **width**, not just
the values: `"onehot"` produces `output_dim` columns, while `"ordinal"` and `"soft"` behave
differently from each other despite both accepting the same `output_dim`.

`"ordinal"`
: A single integer column giving the bin index (output width is always 1, independent of
`output_dim`).

`"onehot"`
: One indicator column per bin (output width equals `output_dim`).

`"soft"`
: A soft assignment that spreads each value across neighbouring bins, so the boundaries are not
hard (output width equals `output_dim`, same shape as `"onehot"` but with fractional
membership instead of a single 1).

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
import numpy as np
from pretab.transformers import PLETransformer

X = np.random.default_rng(0).uniform(size=(100, 1))
y = np.random.default_rng(0).integers(0, 2, size=100)
t = PLETransformer(output_dim=12, task="classification")
t.fit_transform(X, y).shape
# (100, 12): output_dim is exact here (no adaptive clamping)
t.total_output_dim_
# 12
```

Constructor highlights: `output_dim`, `placement_strategy="cart"`, `task="regression"`,
`adaptive`, and `random_state=51`.

```{important}
PLE **requires** the target. It places its edges using `y`, so it must be fit with a target
and should be fit leakage-safely, ideally with cross-fitting. See
[Target awareness](../core_concepts/target_awareness.md).
```

```{warning}
Because PLE always reads `y` to place its bins, fitting it directly on data you will also train
on emits a `LeakageWarning`. Fit it inside a scikit-learn `Pipeline` or wrap it in
`pretab.CrossFittedTransformer` so the bin edges never see the rows they will later transform
for training.
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
from pretab.transformers import PeriodicEncodingTransformer

X = np.array([[0], [6], [12], [18], [24]])   # hour-of-day style values
t = PeriodicEncodingTransformer(period=24, harmonics=2)  # e.g. hour of day
t.fit_transform(X).shape
# (5, 4): 2 columns (sin, cos) per harmonic
```

Constructor highlights: `period` (required, the cycle length), `harmonics=1`,
`include_original=False`.

**Parameter impact.** Output width is `2 * harmonics` (one sine/cosine pair per harmonic), plus
one extra column when `include_original=True`. Higher `harmonics` lets the encoding represent
finer-grained sub-cycles (for example distinguishing morning from afternoon within a day), at
the cost of a wider output.

```{warning}
PreTab has no mechanism to detect the period automatically from the data. `period` is a
required constructor argument with no default, and `fit` only validates that values fall
within `[0, period]`, it never infers the cycle length. You must know and supply the period
yourself (24 for hour of day, 7 for day of week, 12 for month of year, and so on).
```

```{important}
Valid input is the **closed interval** `[0, period]`: both endpoints are accepted, and by
construction they map to the identical `(sin, cos)` pair, since `x=0` and `x=period` are the
same point on the cycle. Values outside `[0, period]` raise a `PretabDataError` at fit and
transform, there is no silent wrap-around or clamping.
```

```{note}
Periodic encoding is a standalone time-series utility. It is not wired into `Preprocessor`
because it requires a per-feature `period`, so apply it directly to the relevant cyclical
column. Use [Fourier features](functional_expansions.md#fourier-features) instead when you want
the model to search across a set of frequencies rather than commit to one known period.
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
