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
import numpy as np
import pandas as pd
from pretab.transformers import BSplineTransformer

rng = np.random.default_rng(0)
n = 3000
x = rng.uniform(0, 10, n)

simple = 0.5 * x + rng.normal(0, 0.3, n)              # nearly linear
wiggly = np.sin(x * 2) * 3 + rng.normal(0, 0.3, n)    # high-frequency

for name, y in [("simple", simple), ("wiggly", wiggly)]:
    t = BSplineTransformer(adaptive=True, min_output_dim=5, max_output_dim=20)
    t.fit(x.reshape(-1, 1), y)
    print(f"{name:8s} -> selected width {t.total_output_dim_}")
```

```text
simple   -> selected width 7
wiggly   -> selected width 7
```

With adaptive resolution on, both signals resolve to widths within the `[5, 20]` bound.
The shape of the signal determines how the search space is used: you get an
appropriately-sized representation for each without tuning by hand.

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
Adaptive resolution is available for the splines, PLE, and the RBF, ReLU, sigmoid, and tanh
feature maps. Methods with a fixed structure (Fourier, binning, the kernel approximations)
ignore the adaptive flag. The [comparison table](../representations/comparison_table.md) marks
which methods adapt.
```

## Where to go next

- [Resolution and placement](../core_concepts/resolution_and_placement.md) for how width and
  placement interact.
- [Comparing representations](comparing_representations.md) to measure adaptive against fixed.
- [Choosing a method](../representations/choosing_a_method.md) for width guidance.
