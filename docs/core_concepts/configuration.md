# Configuration

The `Preprocessor` is configured through a small, predictable set of parameters. This page
covers the four ways to express intent: global defaults, per-feature overrides, presets, and
reading back the resolved configuration. The mechanics of width and placement live in
[Resolution and placement](resolution_and_placement.md), and target usage in
[Target awareness](target_awareness.md).

## Global defaults

The simplest configuration sets one strategy for every numerical column and one for every
categorical column.

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_method="ple",     # applied to every numerical column
    categorical_method="int",   # applied to every categorical column
)
```

The defaults are `numerical_method="ple"` and `categorical_method="int"`. The full list of
strategy strings is in the [representation comparison](../representations/comparison_table.md).

## Per-feature overrides

Columns rarely want identical treatment. The `feature_preprocessing` dict assigns a strategy
to individual columns and takes precedence over the global defaults for those columns.

```python
pre = Preprocessor(
    numerical_method="ple",              # default for numerical columns not listed
    feature_preprocessing={
        "age": "naturalspline",
        "income": "rbf",
        "city": "one-hot",
    },
)
```

```{note}
A per-feature entry is resolved in the correct namespace for its detected column kind. You do
not need to state whether a column is numerical or categorical; PreTab already knows from
feature-type detection.
```

## Presets

Presets are transparent, named bundles of parameters for common intents. They set the same
knobs you could set by hand, so nothing is hidden, and each one resolves to a fixed,
documented set of values:

| Preset | `numerical_method` | `categorical_method` | `output_dim` | `adaptive` | `max_output_dim` |
| --- | --- | --- | --- | --- | --- |
| `"standard"` | `"ple"` | `"int"` | `7` | `False` | `10` |
| `"expanded"` | `"ple"` | `"one-hot"` | `16` | `False` | `10` |
| `"adaptive"` | `"ple"` | `"int"` | `7` | `True` | `16` |

```python
standard = Preprocessor(preset="standard")
expanded = Preprocessor(preset="expanded")

standard.get_resolved_config()["categorical_method"]   # "int": compact integer codes
expanded.get_resolved_config()["categorical_method"]    # "one-hot": one column per category
expanded.get_resolved_config()["output_dim"]             # 16: wider representations than "standard"
```

So `"standard"` is the balanced default (PLE numerics, integer-coded categoricals, `output_dim=7`),
`"expanded"` widens the representation and one-hot-encodes categoricals instead, and
`"adaptive"` lets each feature pick its own width between `min_output_dim` and `max_output_dim`
rather than using a fixed `output_dim`.

```{tip}
A preset is a starting point, not a lock. Any parameter you pass alongside a preset overrides
the preset's value for that knob, for example `Preprocessor(preset="expanded", output_dim=32)`.
```

## Reading the resolved configuration

Because global defaults, per-feature overrides, and presets interact, PreTab lets you read
back exactly what will be used. `get_resolved_config()` returns the fully resolved settings
as a plain dict.

```python
pre = Preprocessor(preset="expanded", feature_preprocessing={"age": "bspline"})
pre.get_resolved_config()
```

This is the authoritative answer to "what did my configuration actually become", and it is
useful in tests and reproducible experiments.

## How the layers combine

The resolution order is deterministic. Later layers win.

1. Library defaults.
2. A `preset`, if given.
3. Explicit constructor arguments (`numerical_method`, `output_dim`, and so on).
4. Per-column `feature_preprocessing` entries, for the columns they name.

```{warning}
Configuration is validated at `fit` time, not silently coerced. An invalid combination, such
as a method that requires the target used with `target_aware=False`, raises a typed error.
This is intentional: it surfaces mistakes early rather than producing a quietly wrong
representation.
```

## Key parameters at a glance

The parameters below are the ones you reach for most. Each links to the page that explains it
in depth.

| Parameter | Default | Covered in |
| --- | --- | --- |
| `numerical_method`, `categorical_method` | `"ple"`, `"int"` | this page |
| `feature_preprocessing` | `None` | this page |
| `output_dim` | `7` | [Resolution and placement](resolution_and_placement.md) |
| `adaptive`, `min_output_dim`, `max_output_dim` | `False`, `5`, `10` | [Resolution and placement](resolution_and_placement.md) |
| `target_aware`, `placement_strategy` | `True`, `"cart"` | [Target awareness](target_awareness.md) |
| `numerical_imputation`, `categorical_imputation`, `add_missing_indicator` | `"median"`, `"most_frequent"`, `False` | [Missing values](missing_values.md) |
| `output_format`, `dtype` | `"dense"`, `None` | [Outputs and inspection](outputs_and_inspection.md) |
| `random_state` | `None` | [Reproducibility](reproducibility.md) |

## Where to go next

- [Resolution and placement](resolution_and_placement.md) for width and location.
- [Target awareness](target_awareness.md) for supervised placement.
- [Representations](../representations/overview.md) for what each method does.
