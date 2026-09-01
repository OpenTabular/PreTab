# Resolution and placement

Two questions define any basis expansion: *how many* units to use, and *where* to put them.
PreTab keeps these separate on purpose. Resolution answers "how many" (the output width), and
placement answers "where" (the knots, centers, or bin edges). This page explains both and how
they combine.

## Resolution: the `output_dim` width

`output_dim` is the main capacity control. It sets the number of non-bias output columns per
input feature: basis functions for the splines, centers for the feature maps, and bins for
PLE. A larger value captures finer structure at the cost of more columns and a higher chance
of overfitting. A smaller value is more compact and regularizes the representation.

```{note}
When you configure through the `Preprocessor`, its single `output_dim` (default `7`) is
forwarded to **every** numerical method. Per-transformer defaults only apply when you build a
transformer directly, for example `RBFExpansionTransformer()`.
```

```{warning}
Numeric binning (`custombin`) is the one exception: `output_dim` sets the number of *bins*,
but with the default `encode="ordinal"` each input feature still emits a single output
column holding the bin index. Pass `encode="onehot"` or `encode="soft"` if you want
`output_dim` to also control the output width.
```

### Spline width has a floor

Each spline enforces a minimum width tied to its degree. Requesting fewer basis functions
than the floor raises an error at `fit` time rather than silently clamping, so keep
`output_dim` at or above the floor.

| Family | Minimum width (floor) |
| --- | --- |
| B, M, I, P-spline, tensor-product | `degree + 1` (so `4` at the default cubic degree) |
| Cubic regression spline | `3` (three polynomial terms plus interior knots) |
| Natural cubic spline | `2` (places `output_dim + 1` knots) |
| Feature maps, PLE, binning | `1` |

```{warning}
For the tensor-product spline the width grows as the **product** across marginal dimensions.
A 2-D input with `output_dim=4` already produces `4 x 4 = 16` columns, so raise it in small
steps and watch the total column count.
```

## Adaptive sizing

Some features are simple and some are complex, and one fixed width rarely suits all of them.
PLE, the feature maps, and the freely-placed knot splines can size each feature from the data
instead.

`adaptive`
: When `True`, the width for each feature is chosen from the data and kept inside
  `[min_output_dim, max_output_dim]`. Fixed-width methods such as the plain scalers ignore
  this flag.

`min_output_dim`, `max_output_dim`
: The lower and upper bounds that apply only when `adaptive=True` (defaults `5` and `10`).
  They are ignored otherwise.

```{note}
When `adaptive=True` and both `min_output_dim` and `max_output_dim` are set, `output_dim` has
no effect at all: the window comes entirely from the two bounds. `output_dim` only matters in
adaptive mode when one of the bounds is left unset, where it fills in for the missing one.
```

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_method="rbf",
    adaptive=True,
    min_output_dim=4,
    max_output_dim=12,
)
```

See the [adaptive resolution tutorial](../tutorials/adaptive_resolution.md) for a worked
example.

## Placement: where the units go

Placement decides the actual positions of the basis units. PreTab centralizes this in one
placement subsystem so no transformer re-implements it, and it is driven by two parameters.

`target_aware`
: Whether placement uses the target `y`.

`placement_strategy`
: How the positions are chosen. Valid values depend on `target_aware`.

| `target_aware` | Allowed `placement_strategy` | Meaning |
| --- | --- | --- |
| `False` | `"uniform"` | Evenly spaced across the observed range. |
| `False` | `"quantile"` | Spaced by data density, more units where data is dense. |
| `True` | `"cart"` | Split points from a per-feature decision tree fit against `y`. |
| `True` | `"lightgbm"` | Split points aggregated from gradient-boosted trees (needs the `lightgbm` extra). |

```{warning}
The unsupervised and target-aware rows are mutually exclusive. Combining them, for example
`target_aware=True` with `placement_strategy="quantile"`, raises an error.
`Preprocessor.placement_strategy` defaults to `"cart"` (paired with `target_aware=True`), so
switching to `target_aware=False` also means passing `placement_strategy="uniform"` or
`"quantile"` explicitly, otherwise the mismatched default raises an error.
```

Target-aware placement is a supervised decision and carries leakage considerations. See
[Target awareness](target_awareness.md).

## Resolution and placement are independent

Keeping the two axes separate is what makes the system predictable. You choose a width
(resolution) and, separately, a rule for positions (placement). The same `"quantile"`
placement works at width `5` or width `20`; the same width works with uniform or
target-aware placement. Not every method honours every strategy: the penalized P-spline
assumes a regular geometry and is `"uniform"` only, PLE always places against the target, and
the thin-plate spline uses landmark points rather than ordinary knots. These per-method rules
are enforced from the capability registry, so an invalid request fails loudly.

## Method-specific placement rules

| Method | Placement behaviour |
| --- | --- |
| PLE | Target-aware always (`"cart"` or `"lightgbm"`). |
| P-spline | `"uniform"` only, unsupervised (the difference penalty assumes regular knots). |
| Feature maps, freely-placed knot splines | Any of the four strategies. |
| Thin-plate spline | Landmark points (k-means), not ordinary knots. |
| Fourier features | Frequencies derived from the data, not placement knots. |

## Where to go next

- [Target awareness](target_awareness.md) for supervised placement and leakage safety.
- [Representations](../representations/overview.md) for how each family uses its locations.
- [Comparing representations](../tutorials/comparing_representations.md) to see width and
  strategy trade-offs measured.
