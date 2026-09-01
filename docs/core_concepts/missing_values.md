# Missing values

Missing data is handled explicitly in PreTab, never silently. You control it with a small set
of imputation parameters and, when you need finer behaviour, a single `missing_policy`. This
page explains both and the rule that ties them together: imputers are fit on the training
data only, and no rows are ever dropped.

## Imputation parameters

Three parameters on `Preprocessor` control the common case.

`numerical_imputation`
: Strategy for numerical columns. Default `"median"`. Set to `None` to disable.

`categorical_imputation`
: Strategy for categorical columns. Default `"most_frequent"`. Set to `None` to disable.

`add_missing_indicator`
: When `True`, adds a binary indicator column marking where a value was missing. Default
  `False`.

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_imputation="median",
    categorical_imputation="most_frequent",
    add_missing_indicator=True,
)
```

```{note}
Setting an imputation strategy to `None` disables imputation for that column kind. The
missing values then reach the transformer directly: scikit-learn scalers, the splines, and
the feature maps (`rbf`, `relu`, `sigmoid`, `tanh`) tolerate `NaN` and pass it straight into
the basis, so an affected row's output is itself undefined. Genuinely finite-only
representations such as PLE, numeric binning, periodic encoding, Fourier features, and the
kernel approximations (`rff`, `nystroem`) raise a typed error instead. That is intentional, an
expansion of an undefined value has no meaning for those methods.
```

```{note}
Requesting `add_missing_indicator=True` while imputation is disabled for that column kind does
not raise. It routes through the same `__missing` indicator branch used by
`missing_policy="separate_state"` below, since `SimpleImputer`'s own indicator only takes
effect when the imputer runs.
```

## Fit on train, apply to test

Imputers learn their fill values from the data passed to `fit`, and only that data. When you
later call `transform` on new rows, the stored fill values are reused. This keeps the split
clean and prevents test statistics from leaking into training.

```{important}
PreTab never drops rows to deal with missing values. Every input row produces an output row.
This preserves alignment with your target and any parallel arrays.
```

## The `missing_policy` control

For finer control, `missing_policy` selects one of five behaviours for the whole
preprocessor.

| Policy | Behaviour |
| --- | --- |
| `"error"` | Reject any missing value at `fit` and `transform`. |
| `"propagate"` | Pass missing values through to the transformer unchanged. |
| `"impute"` | Fill using the imputation parameters above. |
| `"impute_with_indicator"` | Impute and add a missing indicator column. |
| `"separate_state"` | Impute the basis, and add a dedicated `__missing` column that does not activate the ordinary basis. |

```python
pre = Preprocessor(missing_policy="separate_state")
```

### Separate state

`"separate_state"` is the most expressive option. For each affected column it keeps the
imputed value flowing into the normal basis and, in parallel, emits a `__missing` indicator
that a model can weight on its own. This lets the model learn a distinct effect for
"missing" without corrupting the shape learned on observed values.

```{tip}
Reach for `"separate_state"` when missingness is itself informative, for example a field that
users leave blank for a meaningful reason. Reach for plain `"impute"` when a value is missing
purely at random.
```

## Choosing an approach

- **Missing at random, not informative**: `numerical_imputation` / `categorical_imputation`
  (the default), no indicator.
- **Missingness may carry signal**: add `add_missing_indicator=True`, or use
  `missing_policy="separate_state"`.
- **Missing values are a data error you want to catch**: `missing_policy="error"`.
- **You will handle missingness upstream**: `missing_policy="propagate"` with imputation
  disabled.

## Where to go next

- [Configuration](configuration.md) for how these parameters combine with the rest.
- [Edge-case behaviour](../representations/choosing_a_method.md) for constant columns,
  out-of-range inputs, and unseen categories.
- [Outputs and inspection](outputs_and_inspection.md) to see indicator columns in the
  lineage.
