# Leakage-safe classification

The [nonlinear regression tutorial](nonlinear_regression.md) put PreTab in front of a
regressor. The same idea works for classification, with one added concern: when the
representation is supervised, the evaluation must keep it from seeing the test labels. This
tutorial shows an expressive classifier and how to evaluate it without leakage.

Here the target has a **circular** decision boundary, where the positive class sits near the
origin of two coordinates, plus a categorical `plan` effect. A plain `LogisticRegression`
draws a single straight boundary and struggles; radial basis features let it curve around the
circle.

## The dataset

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(1)
n = 4000

x1 = rng.uniform(-3, 3, n)
x2 = rng.uniform(-3, 3, n)
hours = rng.uniform(0, 60, n)
plan = rng.choice(["free", "pro", "team"], n, p=[0.5, 0.3, 0.2])

plan_effect = pd.Series(plan).map({"free": -0.5, "pro": 0.3, "team": 1.0}).to_numpy()
logit = 3.0 - (x1**2 + x2**2) + 0.02 * (hours - 30) + plan_effect + rng.normal(0, 0.5, n)
prob = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({"x1": x1, "x2": x2, "hours": hours, "plan": plan})

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, random_state=42, stratify=y
)
```

The positive class lives inside a disk around the origin, shifted by the plan. The classes are
imbalanced, roughly one positive to three negatives.

## Baseline: scaling and LogisticRegression

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

baseline = ColumnTransformer([
    ("num", MinMaxScaler(), ["x1", "x2", "hours"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
])

X_tr = baseline.fit_transform(X_train)
X_te = baseline.transform(X_test)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
proba = clf.predict_proba(X_te)[:, 1]

print(f"accuracy: {accuracy_score(y_test, clf.predict(X_te)):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
```

```text
accuracy: 0.742
ROC AUC:  0.569
```

Accuracy looks acceptable only because the classes are imbalanced; the model mostly predicts
the majority class. The `ROC AUC` of `0.569` shows it has barely learned to rank positives
above negatives, because a straight boundary cannot enclose the disk.

```{warning}
On imbalanced data, accuracy misleads. A model that always predicts the majority class already
scores around `0.74` here. Prefer threshold-independent metrics such as `ROC AUC`, or
precision and recall, to judge whether a classifier has genuinely learned.
```

## With PreTab

Give every numeric column a radial basis expansion and keep the same classifier.

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_method="rbf",
    categorical_method="one-hot",
    task="classification",
    target_aware=True,
    output_dim=10,
)

X_tr = pre.fit_transform(X_train, y_train)
X_te = pre.transform(X_test)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
proba = clf.predict_proba(X_te)[:, 1]

print(f"accuracy: {accuracy_score(y_test, clf.predict(X_te)):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
```

```text
accuracy: 0.870
ROC AUC:  0.927
```

The RBF features let the linear classifier bend around the circular boundary. Accuracy rises
from `0.742` to `0.870`, and the `ROC AUC` jumps from `0.569` to `0.927`.

```{note}
`target_aware=True` lets supervised expansions use `y` during `fit` to place their basis
functions where they best separate the classes. Because we fit on the training split and only
`transform` the test split, no test label reaches the representation.
```

## Leakage-safe cross-validation

The split above is honest because the representation was fit on the training rows only. To make
that guarantee automatic across folds, put the transformers inside a `Pipeline`. scikit-learn
then fits every step, including the supervised expansion, on each training fold in turn.

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from pretab.transformers import RBFExpansionTransformer

features = ColumnTransformer([
    ("x1", RBFExpansionTransformer(output_dim=10, target_aware=True, task="classification"), ["x1"]),
    ("x2", RBFExpansionTransformer(output_dim=10, target_aware=True, task="classification"), ["x2"]),
    ("hours", RBFExpansionTransformer(output_dim=10, target_aware=True, task="classification"), ["hours"]),
    ("plan", OneHotEncoder(handle_unknown="ignore"), ["plan"]),
])

model = Pipeline([("features", features), ("clf", LogisticRegression(max_iter=1000))])

scores = cross_val_score(model, df, y, cv=5, scoring="roc_auc")
print(f"5-fold ROC AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
```

```{important}
A supervised transformer fit outside a cross-validation or `Pipeline` context emits a
`LeakageWarning`. Inside the `Pipeline` here, the warning is suppressed because each fold fits
the representation on training data only. For the strongest guarantee on training features
themselves, wrap the transformer in `CrossFittedTransformer`. See
[Target awareness](../core_concepts/target_awareness.md).
```

## Where to go next

- See the regression version in the [nonlinear regression tutorial](nonlinear_regression.md).
- Compose transformers with cross-validation and grid search in the
  [sklearn pipeline tutorial](sklearn_pipeline.md).
- Read [Target awareness](../core_concepts/target_awareness.md) for the full leakage model.
