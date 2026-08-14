# Overview

PreTab is a representation and basis-expansion framework for tabular data. It takes raw
numerical and categorical columns and turns them into model-ready features that expose
structure a plain estimator cannot see on its own. Every strategy speaks the standard
scikit-learn `fit` / `transform` API, so PreTab drops into the pipelines and tooling you
already use.

## The problem PreTab solves

Most tabular models receive one straight-line term per numerical column. A linear model,
a logistic regression, or a plain additive model can only weight that single slope, so any
curve, threshold, saturation, or periodic pattern in the data is invisible to it. The usual
response is to hand-craft features: bucket an age column, add a squared income term, encode
the hour of day as a pair of sine and cosine values. That work is repetitive, easy to get
wrong, and rarely reproducible.

PreTab makes those representations first-class. Instead of writing feature code by hand you
declare intent once, for example "expand `age` with a spline, encode `income` with
piecewise-linear bins, treat `hour` as periodic", and PreTab fits the corresponding basis
per column, tracks where every output column came from, and hands back a clean matrix.

```python
from pretab import Preprocessor

pre = Preprocessor(feature_preprocessing={
    "age": "naturalspline",   # smooth non-linear effect
    "income": "ple",          # supervised piecewise-linear encoding
    "hour": "fourier",        # periodic representation
    "city": "one-hot",        # categorical
})
X = pre.fit_transform(df, y)
```

## How this compares to scikit-learn's preprocessing transformers

PreTab is not a competitor to scikit-learn. Every transformer subclasses `BaseEstimator` and
`TransformerMixin` and drops into the same `Pipeline` and `ColumnTransformer` you already use.
The real question is what PreTab adds where scope overlaps with scikit-learn's own
`SplineTransformer`, `KBinsDiscretizer`, `PolynomialFeatures`, and `TargetEncoder`.

| Capability | scikit-learn | PreTab |
| --- | --- | --- |
| Knot / threshold placement | Uniform or quantile, fixed before fitting | Optionally target-aware: a CART or LightGBM model places knots where the target changes fastest (`placement_strategy="cart"`) |
| How many basis functions | You pick a fixed count | `adaptive=True` searches a width in `[min_output_dim, max_output_dim]` from the data |
| Leakage safety | `TargetEncoder` cross-fits internally; nothing else does, and nothing warns you | Every supervised representation emits a `LeakageWarning` outside a `Pipeline`, and any of them can be wrapped in `CrossFittedTransformer` |
| Feature provenance | `get_feature_names_out()` returns names only | A typed `RepresentationSpec` per transformer plus a `FeatureLineage` record per output column (family, component, target usage) |
| Persistence | `pickle` / `joblib`, which execute arbitrary code on load | `to_spec()` / `from_spec()`: a versioned JSON schema that never runs estimator code, plus a stable `fingerprint_` |
| Choosing per column | Hand-assemble a `ColumnTransformer` yourself | One `Preprocessor(feature_preprocessing={...})`, validated against a capability registry so incompatible combinations (a required-target method without `y`, for example) raise a typed error at fit time |

```{note}
Piecewise-linear encoding (`ple`) and the neural-style basis maps (`rbf`, `relu`, `sigmoid`,
`tanh`, deterministic `fourier`) have no scikit-learn equivalent. `rff` and `nystroem` are thin
wrappers around scikit-learn's own `RBFSampler` and `Nystroem`, exposed through the same
`Preprocessor` interface as every other method.
```

## When to reach for PreTab

PreTab is a good fit when any of the following is true.

- You pair a **simple or linear model** (Ridge, logistic regression, a GAM, a linear layer)
  with tabular data and want it to capture non-linear structure.
- You need **expressive numerical representations** such as splines, radial basis maps,
  Fourier features, or piecewise-linear encoding without wiring each one by hand.
- You want **per-column control** over preprocessing from a single configuration object.
- You care about **reproducibility and inspection**: knowing exactly which input produced
  each output column, serializing a fitted representation, and getting a stable fingerprint.
- You are **researching representations** and want a common, typed intermediate form
  (`RepresentationSpec` plus feature lineage) shared across every family.

```{tip}
Basis expansion helps most when the model downstream is comparatively simple. A rich,
already-non-linear model such as gradient boosting can learn many of these shapes on its
own, so the marginal benefit of an explicit basis is smaller there. See
[Choosing a method](../representations/choosing_a_method.md) for the trade-offs.
```

## What PreTab is not

Knowing the boundaries is as useful as knowing the features. PreTab deliberately does not
try to be an everything-library.

- **Not a modelling library.** PreTab produces features. It does not fit predictors, tune
  models, or select features for you. It sits *in front of* an estimator.
- **Not a time-series toolkit.** Generic lag and rolling-window utilities were removed on
  purpose. PreTab keeps the periodic encoding that expresses cyclic structure (hour, day,
  month) but leaves sequence modelling to dedicated libraries.
- **Not a data-cleaning suite.** It offers principled, centrally-defined policies for
  missing values, constant columns, and out-of-range inputs, but it is not a substitute for
  domain-specific data validation.
- **Not a guaranteed accuracy win.** An expressive basis in front of a model that is already
  flexible, or on a feature with no non-linear signal, can add columns without adding value.
  The [failure modes](../representations/choosing_a_method.md#when-basis-expansion-does-not-help)
  section is explicit about where it does not help.

## Two ways to use it

PreTab exposes the same capabilities through two surfaces.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} The high-level `Preprocessor`
Detects column types from a `DataFrame`, applies a strategy per column, and returns
model-ready blocks or a single stacked array. Reach for it when you want per-column
strategies from one config.
:::

:::{grid-item-card} Standalone transformers
Every strategy is also a plain scikit-learn transformer you can import and compose inside a
`Pipeline` or `ColumnTransformer`. Reach for them when you want a single estimator object.
:::

::::

The [Choosing an interface](choosing_an_interface.md) page explains which to pick.

## Where to go next

- [Installation](installation.md) sets up PreTab and its optional extras.
- [Quickstart](quickstart.md) fits your first `Preprocessor` in a few minutes.
- [Core concepts](../core_concepts/feature_representation.md) explains the ideas that run
  through the whole library.
- [Representations](../representations/overview.md) is the catalogue of every method.
