# Comparing representations

Choosing a representation should be an experiment, not a guess. This tutorial evaluates several
numerical methods on the same task with the same model, so the only thing that varies is the
basis. The pattern generalizes to any dataset you have.

## The setup

We reuse a simple nonlinear regression target and hold the model fixed at a `Ridge` regressor.
Each candidate method is fit leakage-safely inside cross-validation.

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 3000
x = rng.uniform(0, 10, n)
y = np.sin(x) * 3 + 0.3 * x + rng.normal(0, 0.4, n)
df = pd.DataFrame({"x": x})
```

## Sweep the candidates

We compare a scaled baseline against a spline, a feature map, and PLE. Each transformer goes
inside a `Pipeline` so cross-validation fits it per fold.

```python
from sklearn.preprocessing import MinMaxScaler
from pretab.transformers import (
    BSplineTransformer,
    RBFExpansionTransformer,
    PLETransformer,
)

candidates = {
    "minmax (baseline)": MinMaxScaler(),
    "bspline": BSplineTransformer(output_dim=12),
    "rbf": RBFExpansionTransformer(output_dim=12),
    "ple": PLETransformer(output_dim=12, task="regression"),
}

results = {}
for name, transformer in candidates.items():
    features = ColumnTransformer([("x", transformer, ["x"])])
    model = Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])
    scores = cross_val_score(model, df, y, cv=5, scoring="r2")
    results[name] = (scores.mean(), scores.std())

for name, (mean, std) in results.items():
    print(f"{name:20s} R2 = {mean:.3f} +/- {std:.3f}")
```

```text
minmax (baseline)    R2 = 0.111 +/- 0.030
bspline              R2 = 0.966 +/- 0.002
rbf                  R2 = 0.965 +/- 0.003
ple                  R2 = 0.955 +/- 0.005
```

The scaled baseline fits a straight line and cannot follow the sine. Every expansion captures
it, with the spline slightly ahead on this smooth signal.

```{tip}
Fix everything except the representation. The same model, the same folds, the same metric.
That isolates the effect of the basis so the comparison is fair.
```

## Weigh width against accuracy

More columns can buy accuracy, but they also cost memory and overfitting headroom. Estimate the
output width before you commit, using the `Preprocessor` budget tools.

```python
from pretab import Preprocessor

for method in ["bspline", "rbf", "ple"]:
    pre = Preprocessor(numerical_method=method, output_dim=12).fit(df, y)
    shape = pre.estimate_output_shape(df)
    print(f"{method:8s} -> {shape[1]} columns")
```

```{note}
A method that wins by a hair but doubles the column count may not be worth it. Read the width
from `estimate_output_shape` and factor it into the decision. See
[Outputs and inspection](../core_concepts/outputs_and_inspection.md).
```

## When nothing beats the baseline

If every expansion ties the scaled baseline, the relationship is probably already linear, or
the feature carries little signal. That is a real and useful result. Do not add columns that do
not earn their place, see
[when basis expansion does not help](../representations/choosing_a_method.md#when-basis-expansion-does-not-help).

## Where to go next

- [Adaptive resolution](adaptive_resolution.md) to let the data pick the width.
- [Choosing a method](../representations/choosing_a_method.md) for guidance behind the numbers.
- [Comparison table](../representations/comparison_table.md) to filter candidates by capability.
