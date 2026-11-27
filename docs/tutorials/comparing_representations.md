# Comparing representations

Compare numerical representations on the same task, with the same folds and downstream model.
This tutorial measures both validation scores and fitted widths for each candidate.

## The setup

We reuse a simple nonlinear regression target and hold the model fixed at a `Ridge` regressor.
Each candidate method is fit leakage-safely inside cross-validation.

```python
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

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
    results[name] = cross_validate(model, df, y, cv=5, scoring="r2", return_estimator=True)

for name, result in results.items():
    scores = result["test_score"]
    print(f"{name:20s} R2 = {scores.mean():.3f} +/- {scores.std():.3f}")
```

```text
minmax (baseline)    R2 = 0.111 +/- 0.030
bspline              R2 = 0.966 +/- 0.002
rbf                  R2 = 0.965 +/- 0.003
ple                  R2 = 0.942 +/- 0.004
```

The scaled baseline fits a straight line and cannot follow the sine. Every expansion captures
it, with the spline slightly ahead on this smooth signal.

```{tip}
Fix everything except the representation. The same model, the same folds, the same metric.
That isolates the effect of the basis so the comparison is fair.
```

## Weigh width against accuracy

Wider representations use more memory and can overfit. Count the output features of the same
fitted pipelines that produced the validation scores. Widths can differ across folds, so check
each one rather than assuming the requested `output_dim` was realized.

```python
for name, result in results.items():
    widths = [
        len(fold.named_steps["features"].get_feature_names_out())
        for fold in result["estimator"]
    ]
    print(f"{name:20s} fitted widths: {widths}")
```

```{note}
Here the baseline has one column and every expansion has 12 in each fold. When comparing
`Preprocessor` candidates, you can read `total_output_dim_` after fitting or call
`estimate_output_shape(X)` on a fitted instance to include the row count for new input. See
[Outputs and inspection](../core_concepts/outputs_and_inspection.md) for output budgets.
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
