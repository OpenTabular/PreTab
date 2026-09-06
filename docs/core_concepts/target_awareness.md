# Target awareness

Some representations place their bins, centers, or knots using the target `y`. Positioning
units where the target actually changes sharpens the representation, but it also reads labels
at `fit` time, which introduces a leakage risk if done carelessly. PreTab makes target usage
explicit and gives you leakage-safe tools. This page explains the contract.

## Which methods use the target

Every method declares how it uses `y` through three levels.

| Level        | Meaning                                                        | Numerical methods                                                                                   | Categorical methods |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------- |
| `forbidden`  | Never uses the target                                           | The scalers (`minmax`, `standardization`, `robust`, `quantile`, `box-cox`, `yeo-johnson`, `polynomial`), `custombin`, `fourier`, `pspline` | `int`, `one-hot`, `onehot_from_ordinal`, `pretrained` |
| `optional`   | Uses the target only when `target_aware=True`                   | The feature maps (`rbf`, `relu`, `sigmoid`, `tanh`) and the freely-placed knot splines (`bspline`, `mspline`, `ispline`, `cubicspline`, `naturalspline`) | -                     |
| `required`   | Always places against the target                                 | `ple` (piecewise-linear encoding)                                                                        | -                     |

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

Even inside a `Pipeline`, fitting a supervised transformer once on the full training set and
then using it to build the *training* features for the same rows introduces a subtle leak:
each row's PLE bins were placed using its own target value. `CrossFittedTransformer` fixes
this for the training features specifically. It splits the training data into folds, fits a
fresh copy of the transformer on all folds *except* one, and uses that copy to transform the
held-out fold, so every training row is transformed by a model that never saw its own target.
`transform` on genuinely new data (a validation or test set) instead uses one model fit on all
the training data, since there is no leakage risk there.

```python
import numpy as np

from pretab import CrossFittedTransformer
from pretab.transformers import PLETransformer

rng = np.random.default_rng(0)
x_train = rng.uniform(-3.0, 3.0, size=(200, 1))
y_train = rng.normal(size=200)

# Naive: one PLE fit on all the training data, then used to transform that same data.
naive = PLETransformer(output_dim=10, random_state=0).fit(x_train, y_train).transform(x_train)

# Cross-fitted: each row is transformed by a model that did not see its own target.
cf = CrossFittedTransformer(PLETransformer(output_dim=10, random_state=0), n_folds=5, random_state=0)
out_of_fold = cf.fit_transform(x_train, y_train)

changed = (~np.all(naive == out_of_fold, axis=1)).sum()
changed, len(x_train)
```

```text
(193, 200)
```

193 of the 200 training rows get different feature values once out-of-fold cross-fitting is
used, which is the leakage `CrossFittedTransformer` removes: the naive version handed a
downstream model features that were partly informed by the very target it is trying to
predict.

```{tip}
Use `CrossFittedTransformer` when you need the **training features themselves** to be
leakage-free, for example to feed a second-stage model or to report an honest training-set
metric. A supervised transformer inside a plain `Pipeline` already keeps cross-validation
honest for `cross_val_score` / `GridSearchCV`, since each fold refits from scratch; you only
need cross-fitting when you build the training features once and reuse them directly.
```

The fitted spec records `cross_fitted=True` and the number of folds, so the choice is
visible and serializable.

```{note}
Cross-fitting matters most for strongly supervised encodings such as PLE, where the target
directly determines the bins. For unsupervised methods it is unnecessary.
```

## Searching over representations

Choosing a numerical method by comparing validation scores is itself a form of model
selection, and doing it carelessly (for example scoring each candidate on the same data used
to fit it) leaks information the same way an unguarded supervised transformer does.
`RepresentationSearchCV` cross-validates a downstream estimator over a set of candidate
`numerical_method` values, refits the best one on all the data, and keeps every candidate's
scoring honest by fitting a fresh `Preprocessor` per fold.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pretab import RepresentationSearchCV

rng = np.random.default_rng(0)
X = pd.DataFrame({"x": rng.uniform(-3, 3, size=300)})
y = np.sin(X["x"]) + rng.normal(0, 0.1, size=300)

search = RepresentationSearchCV(
    estimator=Ridge(),
    methods=["minmax", "ple", "bspline", "rbf"],
    cv=5,
    random_state=0,
)
search.fit(X, y)
search.cv_results_
search.best_method_
```

```text
{'minmax': 0.636, 'ple': 0.949, 'bspline': 0.973, 'rbf': 0.841}
'bspline'
```

`bspline` scored highest across the 5 folds for this sine-shaped signal, so `search.
best_preprocessor_` and `search.best_estimator_` are refit on all the data with `bspline` and
ready to call `.predict(X_new)`.

```{note}
This is deliberately narrow: it only searches the single `numerical_method` axis with one
global method for every numerical column, not a per-column `feature_preprocessing` search.
Use it to answer "which single method suits this dataset" before committing to a
`Preprocessor` configuration, not as a general hyperparameter search.
```

See the [target-aware classification tutorial](../tutorials/target_aware_classification.md)
for an end-to-end, leakage-safe evaluation.

## Where to go next

- [Resolution and placement](resolution_and_placement.md) for the placement strategies.
- [Reproducibility](reproducibility.md) for how supervised state is recorded and serialized.
- [Leakage-safe classification](../tutorials/target_aware_classification.md) for a worked
  example.
