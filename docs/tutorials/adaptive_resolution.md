# Adaptive resolution

Adaptive resolution lets supported representations adjust their width at fit time. This tutorial
shows how to set a search window and how to read the fitted width.

## The idea

Adaptive-capable methods accept three settings:

`adaptive=True`
: Enables width adaptation on supported placement paths.

`min_output_dim` and `max_output_dim`
: The requested lower and upper bounds. The final width also depends on the method and data.

When a supervised search runs, two explicit bounds define its search window. If a bound is
omitted, `output_dim` helps determine the missing edge. The requested `output_dim` must still
be valid for the method, even when both bounds are set.

## A worked example

We fit a spline with adaptive width on two signals of different complexity and inspect what each
one chose.

```python
import warnings

import numpy as np

from pretab.exceptions import LeakageWarning
from pretab.transformers import BSplineTransformer

rng = np.random.default_rng(0)
n = 3000
x = rng.uniform(0, 10, n)

simple = 0.5 * x + rng.normal(0, 0.3, n)              # nearly linear
wiggly = np.sin(x * 2) * 3 + rng.normal(0, 0.3, n)    # high-frequency

for name, y in [("simple", simple), ("wiggly", wiggly)]:
    t = BSplineTransformer(
        adaptive=True, min_output_dim=5, max_output_dim=20,
        target_aware=True, placement_strategy="cart", task="regression",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LeakageWarning)
        t.fit(x.reshape(-1, 1), y)
    print(f"{name:8s} -> selected width {t.total_output_dim_}")
```

```text
simple   -> selected width 20
wiggly   -> selected width 20
```

Both fits reach the upper bound in this example. Adaptive width reflects the splits the CART
selector finds under its tree settings. A more irregular signal does not necessarily get more
columns.

```{note}
Fitting a target-aware transformer directly like this, outside a `Pipeline`, normally emits a
`LeakageWarning`; it is suppressed above because this is a one-off illustrative fit whose
output is never used to train a downstream model. See
[Target awareness](../core_concepts/target_awareness.md) for when the warning matters.
```

## Adaptation without a target

B-, M-, and I-splines can also adapt without a target. They start from `output_dim` and adjust
the unsupervised knots toward the requested bounds. For example, a target-free B-spline with
bounds `[4, 6]` can keep a requested width of 5 or reduce a requested width of 12 to 6.
By contrast, natural cubic splines and maps such as RBF and ReLU keep their ordinary fixed
sizing on the target-free path. PLE always uses the target and does not require a
`target_aware` flag.

```{note}
B-, M-, and I-splines cap the number of interior knots by the number of distinct input
values. A column with few distinct values can produce fewer columns than `min_output_dim`.
Use the fitted width for any downstream model or size check.
```

## Adaptive across a whole preprocessor

`Preprocessor` passes the global settings to eligible features, which are fitted independently.
This example applies the target-free B-spline settings to two columns with enough distinct values.

```python
import pandas as pd
from pretab import Preprocessor

df = pd.DataFrame({"value": x, "squared": x**2})

pre = Preprocessor(
    numerical_method="bspline",
    output_dim=12,
    adaptive=True,
    min_output_dim=4,
    max_output_dim=6,
    target_aware=False,
    placement_strategy="quantile",
)
pre.fit(df)
assert pre.output_dims_ == {"value": 6, "squared": 6}
assert pre.total_output_dim_ == 12
```

`output_dims_` and `total_output_dim_` describe the fitted output, not the requested width
or bounds. For a per-feature breakdown, call `pre.get_feature_info()` after fitting.

```{note}
The [comparison table](../representations/comparison_table.md) marks which methods support
adaptive resolution. An unsupported method such as P-spline ignores the global adaptive
setting. An explicit per-feature `RepresentationConfig("pspline", adaptive=True)` is
rejected when `Preprocessor` validates its configuration.
```

## Where to go next

- [Placement strategies](../core_concepts/resolution_and_placement.md#placement-where-the-units-go)
  for how the locations of knots and centers are chosen.
- [Comparing representations](comparing_representations.md) to measure adaptive against fixed.
- [Choosing a method](../representations/choosing_a_method.md) for width guidance.
