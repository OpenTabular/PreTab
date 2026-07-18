# Classification

The [end-to-end example](../getting_started/end_to_end.md) showed pretab in front of a
regressor. The same idea works for classification: give a linear classifier an expressive
feature basis and it can learn decision boundaries that a raw model cannot.

Here the target has a **ring-shaped** boundary, where the positive class sits near the origin
of two coordinates, plus a categorical `plan` effect. A plain `LogisticRegression` draws a
single straight boundary and struggles; radial basis features let it curve around the ring.

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

# Positive class lives inside a ring around the origin, shifted by the plan.
plan_effect = pd.Series(plan).map({"free": -0.5, "pro": 0.3, "team": 1.0}).to_numpy()
logit = 3.0 - (x1**2 + x2**2) + 0.02 * (hours - 30) + plan_effect + rng.normal(0, 0.5, n)
prob = 1 / (1 + np.exp(-logit))
y = (rng.uniform(0, 1, n) < prob).astype(int)

df = pd.DataFrame({"x1": x1, "x2": x2, "hours": hours, "plan": plan})
print("class balance:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, random_state=42, stratify=y
)
```

```text
class balance: {0: 2966, 1: 1034}
```

## Baseline: scaling + LogisticRegression

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

print(f"features: {X_tr.shape[1]}")
print(f"accuracy: {accuracy_score(y_test, clf.predict(X_te)):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
```

```text
features: 6
accuracy: 0.742
ROC AUC:  0.569
```

Accuracy looks acceptable only because the classes are imbalanced, since the model mostly
predicts the majority class. The `ROC AUC` of `0.569` shows it has barely learned to rank
positives above negatives, because a straight boundary cannot enclose the ring.

```{warning}
On imbalanced data, accuracy can be misleading. A model that always predicts the majority
class would already score around `0.74` here. Prefer threshold-independent metrics such as
`ROC AUC`, or precision and recall, to judge whether a classifier has genuinely learned.
```

## With pretab

Give every numeric column a radial basis expansion and keep the same classifier.

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_method="rbf",
    categorical_method="one-hot",
    task="classification",
    use_target=True,
    output_dim=10,
)

X_tr = pre.fit_transform(X_train, y_train, return_array=True)
X_te = pre.transform(X_test, return_array=True)

clf = LogisticRegression(max_iter=1000).fit(X_tr, y_train)
proba = clf.predict_proba(X_te)[:, 1]

print(f"features: {X_tr.shape[1]}")
print(f"accuracy: {accuracy_score(y_test, clf.predict(X_te)):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, proba):.3f}")
```

```text
features: 33
accuracy: 0.872
ROC AUC:  0.927
```

The RBF features let the linear classifier bend around the ring. Accuracy rises from `0.742`
to `0.872`, and the `ROC AUC` jumps from `0.569` to `0.927`, a much better separation of
the two classes.

```{note}
`use_target=True` lets supervised expansions (like RBF and PLE) use `y` during `fit` to
place their basis functions where they best separate the classes, so always pass `y` when
fitting.
```

## What changed

```python
pre.get_feature_info()
```

```text
feature  kind         pipeline                        dim   cats
----------------------------------------------------------------
x1       numerical    imputer -> minmax -> rbf         10      -
x2       numerical    imputer -> minmax -> rbf         10      -
hours    numerical    imputer -> minmax -> rbf         10      -
plan     categorical  imputer -> onehot -> to_float     3      3
```

Three numeric columns become 30 RBF features and `plan` becomes three one-hot columns, 33
in total, turning an unsolvable linear problem into an easy one.

## Next steps

- See the regression version in the [end-to-end example](../getting_started/end_to_end.md).
- Compose transformers inside a single `sklearn` `Pipeline` with cross-validation in the
  [sklearn Pipeline tutorial](sklearn_pipeline.md).
