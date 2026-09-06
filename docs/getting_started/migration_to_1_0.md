# Migrating to 1.0

PreTab 1.0 is the first stable release. Because the previously published API (`0.0.3`) was
never declared stable, 1.0 takes a one-time, deliberate cleanup: intention-revealing class
names, non-overlapping parameters, and a smaller, sharper scope. This page maps the old
surface to the new one so you can upgrade in a single pass.

```{important}
1.0 contains breaking changes relative to `0.0.3`. There are no compatibility shims. Update
the names and parameters below, then re-fit. Pin `pretab<1` if you need the old behaviour
while you migrate.
```

## Renamed transformers

The classes gained names that say what they compute.

| Old name (`0.0.3`)        | New name (`1.0`)                   | Notes                                               |
| ------------------------- | ---------------------------------- | --------------------------------------------------- |
| `CustomBinTransformer`    | `NumericBinningTransformer`        | Numeric-only, now stateful (learns edges in `fit`). |
| `CyclicalTimeTransformer` | `PeriodicEncodingTransformer`      | Sine and cosine harmonics for cyclic values.        |
| `CubicSplineTransformer`  | `CubicRegressionSplineTransformer` | Disambiguated from the generic cubic B-spline.      |

## Removed transformers

Generic time-series utilities are out of scope for a representation framework.

| Removed                   | Replacement                          |
| ------------------------- | ------------------------------------ |
| `LagFeatureTransformer`   | Use a dedicated time-series library. |
| `RollingStatsTransformer` | Use a dedicated time-series library. |

```{note}
Cyclic time structure is still first-class through `PeriodicEncodingTransformer` and the
`"fourier"` feature map. Only the generic lag and rolling-window helpers were removed.
```

## Deprecated

| Symbol                         | Status                                   | Do this instead                                                                     |
| ------------------------------ | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `OneHotFromOrdinalTransformer` | Deprecated, emits a `DeprecationWarning` | Use the `"one-hot"` categorical method, which wraps scikit-learn's `OneHotEncoder`. |

## Parameter changes on `Preprocessor`

### Placement is now two clean knobs

The overlapping `selector` / `strategy` / `use_target` arguments are gone. Placement is
controlled by exactly two parameters that validate strictly against each other.

| Old                                                          | New                                                |
| ------------------------------------------------------------ | -------------------------------------------------- |
| `use_target=True/False`, plus ad-hoc `selector` / `strategy` | `target_aware: bool` and `placement_strategy: str` |

The valid combinations are fixed:

| `target_aware` | Allowed `placement_strategy` |
| -------------- | ---------------------------- |
| `False`        | `"uniform"`, `"quantile"`    |
| `True`         | `"cart"`, `"lightgbm"`       |

```{warning}
Mixing the two rows, for example `target_aware=True` with `placement_strategy="quantile"`,
raises an error rather than silently guessing. `placement_strategy` defaults to `"cart"`
(paired with the `target_aware=True` default), so switching to `target_aware=False` also
means passing `placement_strategy="uniform"` or `"quantile"` explicitly.
```

See [Resolution and placement](../core_concepts/resolution_and_placement.md) for the full
model.

### Missing-value handling is explicit

The single `handle_missing` flag was replaced by three explicit parameters.

| Old                  | New                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------- |
| `handle_missing=...` | `numerical_imputation="median"`, `categorical_imputation="most_frequent"`, `add_missing_indicator=False` |

Set an imputation strategy to `None` to disable it for that kind. See
[Missing values](../core_concepts/missing_values.md).

## Renamed optional extra

| Old install                   | New install                      |
| ----------------------------- | -------------------------------- |
| `pip install "pretab[knots]"` | `pip install "pretab[lightgbm]"` |

The rename matches `placement_strategy="lightgbm"`. The `embeddings` and `all` extras are
unchanged. See [Installation](installation.md).

## Thin-plate spline parameters

The thin-plate spline moved to landmark-based terminology and is sized by rank, not by a
fixed `output_dim`.

| Old                                          | New                                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ThinPlateSplineTransformer(output_dim=...)` | `ThinPlateSplineTransformer(n_components=..., landmark_strategy="kmeans", rank_strategy="eigen")` |

## What is new in 1.0

Upgrading also unlocks capabilities that did not exist in `0.0.3`.

- **New representations**: `FourierFeatureTransformer`, `RandomFourierFeaturesTransformer`,
  and `NystroemFeaturesTransformer`.
- **A typed intermediate form**: `RepresentationSpec` plus per-output-column
  [feature lineage](../core_concepts/outputs_and_inspection.md).
- **Leakage-safe supervision**: `CrossFittedTransformer`, `RepresentationSearchCV`, and a
  `LeakageWarning`. See [Target awareness](../core_concepts/target_awareness.md).
- **Portable serialization**: `to_spec` / `from_spec`, a stable `fingerprint_`, and a frozen
  lifecycle. See [Reproducibility](../core_concepts/reproducibility.md).
- **Presets and discovery**: `Preprocessor(preset=...)` and `list_representations(...)`.
- **Central edge-case policy** and **output budgets** on `Preprocessor`.

## Upgrade checklist

1. Rename the three renamed transformer classes.
2. Remove any use of `LagFeatureTransformer` / `RollingStatsTransformer`.
3. Replace `handle_missing` with the three explicit imputation parameters.
4. Replace `use_target` / `selector` / `strategy` with `target_aware` and
   `placement_strategy`.
5. Swap `ThinPlateSplineTransformer(output_dim=...)` for `n_components`.
6. Update `pretab[knots]` to `pretab[lightgbm]` in your dependencies.
7. Re-fit and confirm the resolved layout with `get_feature_info(verbose=True)`.
