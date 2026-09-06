# Missing values

Missing data is handled explicitly, never silently: a small set of imputation parameters
covers the common case, and a single `missing_policy` gives finer control when you need it.

## Imputation parameters

| Parameter                | Meaning                                                           | Default           |
| ------------------------ | ----------------------------------------------------------------- | ----------------- |
| `numerical_imputation`   | Strategy for numerical columns. `None` disables it.               | `"median"`        |
| `categorical_imputation` | Strategy for categorical columns. `None` disables it.             | `"most_frequent"` |
| `add_missing_indicator`  | Adds a binary indicator column marking where a value was missing. | `False`           |

```python
from pretab import Preprocessor

pre = Preprocessor(
    numerical_imputation="median",
    categorical_imputation="most_frequent",
    add_missing_indicator=True,
)
```

```{note}
Disabling imputation lets `NaN` reach the transformer directly. Scalers, splines, and the
feature maps (`rbf`, `relu`, `sigmoid`, `tanh`) tolerate it, so an affected row's output is
itself undefined; finite-only methods (PLE, numeric binning, periodic encoding, Fourier
features, `rff`, `nystroem`) raise a typed error instead, since expanding an undefined value
has no meaning for them. `add_missing_indicator=True` still works with imputation disabled: it
routes through the same `__missing` branch `missing_policy="separate_state"` uses below.
```

```{important}
Imputers fit their fill values only on the data passed to `fit`, and reuse those values
unchanged at `transform`. PreTab never drops a row for missing values: every input row
produces an output row.
```

## The `missing_policy` control

For finer control, `missing_policy` selects one of five behaviours for the whole
preprocessor.

| Policy                    | Behaviour                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------- |
| `"error"`                 | Reject any missing value at `fit` and `transform`.                                                  |
| `"propagate"`             | Pass missing values through to the transformer unchanged.                                           |
| `"impute"`                | Fill using the imputation parameters above.                                                         |
| `"impute_with_indicator"` | Impute and add a missing indicator column.                                                          |
| `"separate_state"`        | Impute the basis, and add a dedicated `__missing` column that does not activate the ordinary basis. |

```python
pre = Preprocessor(missing_policy="separate_state")
```

### Separate state

`"separate_state"` is the most expressive option: it keeps the imputed value flowing into the
normal basis while emitting a parallel `__missing` indicator a model can weight on its own, so
it can learn a distinct effect for "missing" without corrupting the shape learned on observed
values.

```{tip}
Reach for `"separate_state"` (or `add_missing_indicator=True`) when missingness itself carries
signal. Reach for plain `"impute"` when a value is missing purely at random, `"error"` to catch
missingness as a data bug, and `"propagate"` (with imputation disabled) to handle it upstream
yourself.
```

## Where to go next

- [Configuration](configuration.md) for how these parameters combine with the rest.
- [Edge-case behaviour](../representations/choosing_a_method.md) for constant columns,
  out-of-range inputs, and unseen categories.
- [Outputs and inspection](outputs_and_inspection.md) to see indicator columns in the
  lineage.
