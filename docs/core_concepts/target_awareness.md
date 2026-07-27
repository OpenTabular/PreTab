# Target awareness

Some representations place their bins, centers, or knots using the target `y`. Positioning
units where the target actually changes sharpens the representation, but it also reads labels
at `fit` time, which introduces a leakage risk if done carelessly. PreTab makes target usage
explicit and gives you leakage-safe tools. This page explains the contract.

## Which methods use the target

Every method declares how it uses `y` through three levels.

`forbidden`
: The method never uses the target. The scalers, one-hot, ordinal encoding, the Fourier map,
  and the P-spline are all unsupervised.

`optional`
: The method uses the target only when `target_aware=True`. The feature maps (RBF, ReLU,
  sigmoid, tanh) and the freely-placed knot splines (B, M, I, cubic, natural) are in this
  group.

`required`
: The method always places against the target. Piecewise-linear encoding (PLE) is the primary
  example and needs `y` at every fit.

```python
from pretab.transformers import PLETransformer

t = PLETransformer()
t.requires_y      # True: PLE always needs y
t.is_supervised   # True
```

```{warning}
A `required` method fitted without `y`, or with `target_aware=False`, raises a typed error
rather than silently producing an unsupervised result. Always pass `y` to a pipeline that
contains PLE.
```

## The fitted-usage flag

After fitting, a transformer reports whether it actually consumed the target through
`uses_target_`. This is the ground truth for an individual fit, and it flows into the
[`RepresentationSpec`](outputs_and_inspection.md) so a serialized representation records
whether it was supervised.

```python
t = PLETransformer().fit(x, y)
t.uses_target_    # True
```

## Leakage safety

Fitting a supervised transformer on your full dataset and then evaluating on part of it leaks
target information and inflates scores. PreTab warns when it detects this pattern.

```{important}
A supervised transformer emits a `LeakageWarning` when it is fit with a target **outside** a
cross-validation or `Pipeline` context. Inside a scikit-learn `Pipeline`, `ColumnTransformer`,
`Preprocessor`, or a cross-fitting wrapper, the warning is suppressed because those contexts
already keep the fit confined to the training fold.
```

The safe patterns are:

- Put the supervised transformer **inside a `Pipeline`**, so `cross_val_score` and
  `GridSearchCV` fit it on the training fold only.
- Use the `Preprocessor`, which fits its imputers and supervised expansions on the training
  data you pass to `fit`.
- Wrap it in a `CrossFittedTransformer` when you want out-of-fold training features.

## Cross-fitted features

`CrossFittedTransformer` removes leakage from the training features themselves. It produces
out-of-fold values for the training rows (each row is transformed by a model that did not see
it) while `transform` on new data uses a model fit on all the training data.

```python
from pretab import CrossFittedTransformer
from pretab.transformers import PLETransformer

cf = CrossFittedTransformer(PLETransformer(), n_folds=5)
X_train_features = cf.fit_transform(x_train, y_train)  # out-of-fold, leakage-free
X_test_features = cf.transform(x_test)                 # uses the all-data model
```

The fitted spec records `cross_fitted=True` and the number of folds, so the choice is
visible and serializable.

```{note}
Cross-fitting matters most for strongly supervised encodings such as PLE, where the target
directly determines the bins. For unsupervised methods it is unnecessary.
```

## Searching over representations

`RepresentationSearchCV` cross-validates a downstream estimator over a set of candidate
numerical methods and refits the best one. It is a convenient way to let the data choose the
representation without leaking through the selection.

```python
from pretab import RepresentationSearchCV
```

See the [target-aware classification tutorial](../tutorials/target_aware_classification.md)
for an end-to-end, leakage-safe evaluation.

## Where to go next

- [Resolution and placement](resolution_and_placement.md) for the placement strategies.
- [Reproducibility](reproducibility.md) for how supervised state is recorded and serialized.
- [Leakage-safe classification](../tutorials/target_aware_classification.md) for a worked
  example.
