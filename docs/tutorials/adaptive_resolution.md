# Adaptive resolution

Picking the width of an expansion by hand is guesswork. Adaptive resolution lets the data
choose it for you, within bounds you set. This tutorial shows how to turn it on and how to read
the width that was selected.

## The idea

Every adaptive-capable method accepts three parameters that turn a fixed width into a searched
one.

`adaptive=True`
: Enables data-driven width selection.

`min_output_dim` and `max_output_dim`
: The lower and upper bounds of the search. The method picks a width in this range.

When adaptive is on and both bounds are set, `output_dim` is ignored completely: the fitted
width comes only from `[min_output_dim, max_output_dim]` and the data. If you leave one bound
unset, `output_dim` fills in for it (as the missing lower or upper edge of the search), so it
still matters in that case. See
[Resolution and placement](../core_concepts/resolution_and_placement.md) for the mechanics.

```{warning}
Setting `output_dim` alongside `adaptive=True` with both `min_output_dim` and `max_output_dim`
is harmless but silently has no effect. It is easy to assume it caps or anchors the search; it
does not. Drop it, or drop one of the two bounds if you meant `output_dim` to anchor the window.
```

## A worked example

We fit a spline with adaptive width on two signals of different complexity and inspect what each
one chose.

```python
import warnings
import numpy as np
import pandas as pd
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
        warnings.simplefilter("ignore")  # one-off fit, not reused to train a model
        t.fit(x.reshape(-1, 1), y)
    print(f"{name:8s} -> selected width {t.total_output_dim_}")
```

```text
simple   -> selected width 15
wiggly   -> selected width 15
```

Both widths land inside the `[5, 20]` window without you having to guess a number up front.
The two happen to match here because the underlying CART selector's split count is governed
more by its own tree depth and minimum-samples settings than by how wiggly the signal looks;
with noisier or smaller data, or a narrower window, the two searches can land on different
widths. The bound is what you control directly, the exact count inside it is data-driven.

```{note}
Fitting a target-aware transformer directly like this, outside a `Pipeline`, normally emits a
`LeakageWarning`; it is suppressed above because this is a one-off illustrative fit whose
output is never used to train a downstream model. See
[Target awareness](../core_concepts/target_awareness.md) for when the warning matters.
```

```{note}
Adaptive resolution only takes effect on the target-aware placement path
(`target_aware=True`, paired with `placement_strategy="cart"` or `"lightgbm"`). With the
default `target_aware=False`, `adaptive=True` is a silent no-op and the transformer keeps its
ordinary fixed `output_dim` width.
```

```{tip}
Set `min_output_dim` and `max_output_dim` to a range you consider reasonable, then let the data
place the width inside it. This is more robust than committing to a single `output_dim` across
features of different complexity.
```

## Adaptive across a whole preprocessor

The same switch works at the `Preprocessor` level, so every eligible column adapts
independently.

```python
import pandas as pd
from pretab import Preprocessor

df = pd.DataFrame({"simple": simple, "wiggly": wiggly})

pre = Preprocessor(
    numerical_method="bspline",
    adaptive=True,
    min_output_dim=5,
    max_output_dim=15,
)
pre.fit(df, wiggly)
pre.get_feature_info()
```

Each numerical column receives a width suited to its own complexity, visible in the resolved
feature info.

```{note}
Adaptive resolution is available for B/M/I splines, the freely-placed cubic and natural cubic
splines, PLE, and the RBF, ReLU, sigmoid, and tanh feature maps. The penalized P-spline and the
tensor-product and thin-plate splines have a fixed structure and ignore the adaptive flag. The
[comparison table](../representations/comparison_table.md) marks which methods adapt.
```

## Where to go next

- [Resolution and placement](../core_concepts/resolution_and_placement.md) for how width and
  placement interact.
- [Comparing representations](comparing_representations.md) to measure adaptive against fixed.
- [Choosing a method](../representations/choosing_a_method.md) for width guidance.
