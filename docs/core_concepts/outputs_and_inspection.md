# Outputs and inspection

A representation is only useful if you can read what it produced. PreTab returns model-ready
output in the format you ask for, names every column, and can trace each output column back to
the exact input and component that created it. This page covers output shapes, formats,
feature names, lineage, and the output budget.

## Output shapes

`fit_transform` and `transform` return a single stacked `numpy.ndarray` by default. Pass
`output_structure="blocks"` (or `return_array=False` for a single call) to receive a
dictionary that maps each feature to its transformed block instead, with keys prefixed
`num_` or `cat_`.

```python
X_array = pre.fit_transform(df, y)                                   # one stacked ndarray
X_dict = Preprocessor(output_structure="blocks").fit_transform(df, y)  # {"num_age": ..., "cat_city": ...}
```

```{note}
The array form is what a plain scikit-learn estimator expects, and is what lets a
`Preprocessor` drop straight into a `Pipeline`. The dict form is convenient for inspection and
for feeding blocks to different model heads; pass `return_array` explicitly to override
`output_structure` for a single call without changing the estimator's configuration.
```

### Example: one column expands into several

A single input column rarely maps to a single output column. Most numerical methods (splines,
feature maps, PLE) expand one feature into `output_dim` basis columns, so the matrix width
grows well beyond the number of input columns. A B-spline makes this concrete: `output_dim=6`
turns the one `age` column into 6 local basis functions, each capturing a different region of
its range.

```python
df = pd.DataFrame({"age": [25, 40, 63, 51]})
y = [0.1, 0.9, 0.3, 0.6]

pre = Preprocessor(numerical_method="bspline", output_dim=6,
                    target_aware=False, placement_strategy="quantile").fit(df, y)
out = pre.transform(df)
out.shape
```

```text
(4, 6)
```

```text
array([[1.   , 0.   , 0.   , 0.   , 0.   , 0.   ],
       [0.   , 0.179, 0.593, 0.228, 0.   , 0.   ],
       [0.   , 0.   , 0.   , 0.   , 0.   , 1.   ],
       [0.   , 0.   , 0.165, 0.607, 0.229, 0.   ]])
```

4 input rows, 1 input column, but 6 output columns: every row's `age` value is spread across
the basis functions whose local support it falls into (each row sums to 1, since a B-spline
basis is a partition of unity). `get_feature_names_out()` shows exactly where each column came
from:

```python
list(pre.get_feature_names_out())
```

```text
['num_age_bs0', 'num_age_bs1', 'num_age_bs2', 'num_age_bs3', 'num_age_bs4', 'num_age_bs5']
```

All six still trace back to the single `age` input, which is exactly what `output_structure=
"blocks"` reflects: one dict entry per **input** feature, holding its full expanded block, not
one entry per output column.

```python
pre_blocks = Preprocessor(numerical_method="bspline", output_dim=6,
                           target_aware=False, placement_strategy="quantile",
                           output_structure="blocks").fit(df, y)
pre_blocks.transform(df)["num_age"].shape
```

```text
(4, 6)
```

```{tip}
This is why a wide expansion (a spline or feature map with a large `output_dim`, or several
expanded columns) can produce a much wider matrix than the input `DataFrame` had columns.
Use `get_feature_info(verbose=True)` (below) or `estimate_output_shape(df)` to see the total
width before committing, especially with several expanded columns at once.
```

## Output format and dtype


Two parameters control the physical layout of the stacked output.

`output_format`
: One of `"dense"`, `"sparse"`, or `"auto"`. `"auto"` picks sparse when it saves memory (for
  example wide one-hot blocks) and dense otherwise. Default `"dense"`.

`dtype`
: The floating-point precision of the output, for example `numpy.float32` to halve memory.

```python
pre = Preprocessor(output_format="auto", dtype="float32")
```

After fitting, `output_report_` summarizes what was produced: the chosen format, dimensions,
density, and memory saved.

```python
pre.fit(df, y)
pre.output_report_
```

### DataFrame output

PreTab honours the scikit-learn output API, so you can request pandas or polars frames.

```python
pre.set_output(transform="pandas")   # or "polars"
```

```{note}
Polars output requires the optional `polars` extra (`pip install "pretab[polars]"`) and is
loaded lazily: if polars is not installed, requesting it raises a clear
`OptionalDependencyError` rather than failing deep in the call stack. See
[Installation](../getting_started/installation.md#optional-extras).
```

## Choosing your output settings

Four things independently affect what `transform` / `fit_transform` return: `output_structure`
(top-level shape), `return_array` (a per-call override of it), `output_format` (dense vs.
sparse), and `set_output` (scikit-learn's own DataFrame protocol). This section is the
decision guide: what each one controls, how they interact, and which combination fits a given
use case, with the literal output shown for each rather than just a description of it.

The examples below all share the same tiny, fixed input so the printed output is directly
comparable:

```python
import numpy as np
import pandas as pd
from pretab import Preprocessor

df = pd.DataFrame({"age": [25, 40, 63], "city": ["A", "B", "A"]})
y = [0.1, 0.9, 0.3]
```

### Parameter impact at a glance

| Parameter | Values | Controls | Set at |
| --- | --- | --- | --- |
| `output_structure` | `"matrix"` (default), `"blocks"` | Whether `transform()` returns one stacked array or a dict of per-feature blocks, when `return_array` is not passed. | Constructor |
| `return_array` | `True`, `False`, `None` (default) | Overrides `output_structure` for a single call. `None` resolves from `output_structure`. | Per call |
| `output_format` | `"dense"` (default), `"sparse"`, `"auto"` | Whether the array (or each block) is a NumPy array or a SciPy CSR matrix. `"auto"` picks sparse only when it saves memory. | Constructor |
| `dtype` | e.g. `numpy.float32`, `None` (default) | Floating-point precision of the output; also what `estimate_memory()` assumes. | Constructor |
| `set_output(transform=...)` | `"default"`, `"pandas"`, `"polars"` | Wraps the array in a DataFrame. **Takes priority over `output_structure` and `return_array` entirely**: once set, `transform()` always returns a DataFrame, never a dict or a bare array. | Method call, before `transform` |

### Which setting should I use?

Feeding a plain scikit-learn estimator or building a `Pipeline`
: Use the default (`output_structure="matrix"`, no `return_array` override). `Preprocessor()`
  composes directly: `Pipeline([("pretab", Preprocessor(...)), ("model", Ridge())])` just
  works, since `transform(X)` already returns a single array.

  ```python
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot").fit(df, y)
  out = pre.transform(df)
  type(out), out.shape
  ```

  ```text
  (<class 'numpy.ndarray'>, (3, 3))
  ```

  ```text
  array([[0.        , 1.        , 0.        ],
         [0.39473684, 0.        , 1.        ],
         [1.        , 1.        , 0.        ]])
  ```

  Column order matches `get_feature_names_out()`: the scaled `age`, then the one-hot `city_A`
  / `city_B` columns.

Inspecting per-feature blocks, or feeding different blocks to different model heads
: Set `output_structure="blocks"` on the constructor (so it stays the estimator's default
  everywhere it's reused), or pass `return_array=False` for a one-off call without changing
  the estimator's configuration.

  ```python
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot",
                      output_structure="blocks").fit(df, y)
  out = pre.transform(df)
  out
  ```

  ```text
  {'num_age': array([[0.        ],
                      [0.39473684],
                      [1.        ]]),
   'cat_city': array([[1., 0.],
                       [0., 1.],
                       [1., 0.]])}
  ```

  The same estimator still returns an array for a single call if you ask for one:

  ```python
  pre.transform(df, return_array=True)   # ndarray, shape (3, 3); this call only
  ```

Passing external `embeddings`
: **Embeddings require dict output.** They are separate named blocks, so they cannot be
  stacked into a single matrix. Use `output_structure="blocks"` (or `return_array=False`) on
  every `transform` call that passes `embeddings`; the default `"matrix"` raises
  `IncompatibleParamsError` the moment `embeddings` is supplied.

  ```python
  emb = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot",
                      output_structure="blocks").fit(df, y, embeddings=emb)
  out = pre.transform(df, embeddings=emb)
  sorted(out.keys()), out["embedding_1"].shape
  ```

  ```text
  (['cat_city', 'embedding_1', 'num_age'], (3, 2))
  ```

Large, mostly one-hot-encoded categorical data
: Set `output_format="sparse"` (or `"auto"` to let PreTab decide per fit). This applies
  independently of `output_structure`: a sparse `"matrix"` is a single stacked
  `scipy.sparse.csr_matrix`, and a sparse `"blocks"` dict holds a CSR matrix per feature.

  ```python
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot",
                      output_format="sparse").fit(df, y)
  pre.transform(df)
  ```

  ```text
  <Compressed Sparse Row sparse matrix of dtype 'float64'
          with 5 stored elements and shape (3, 3)>
  ```

Halving memory on a large dataset
: Set `dtype=numpy.float32`. `estimate_memory()` and the `max_dense_memory` budget both
  reflect the configured `dtype`, not a hardcoded `float64` assumption, so budgets sized for
  the real, cast output are honored correctly.

  ```python
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot",
                      dtype=np.float32).fit(df, y)
  pre.transform(df).dtype
  ```

  ```text
  dtype('float32')
  ```

Feeding a library that expects a DataFrame, or wanting column names attached to the output
: Call `pre.set_output(transform="pandas")` (or `"polars"`). This overrides everything else:
  `transform()` always returns a DataFrame from that point on, regardless of `output_structure`
  or any `return_array` passed to an individual call.

  ```python
  pre = Preprocessor(numerical_method="minmax", categorical_method="one-hot").fit(df, y)
  pre.set_output(transform="pandas").transform(df)
  ```

  ```text
      num_age  cat_city_A  cat_city_B
  0  0.000000         1.0         0.0
  1  0.394737         0.0         1.0
  2  1.000000         1.0         0.0
  ```

```{warning}
`set_output` always wins. If a downstream step unexpectedly receives a DataFrame instead of
the array or dict you configured, check whether `set_output` was called anywhere upstream
(including by a cloned copy inside a `Pipeline`/`GridSearchCV`).
```

## Feature names

Every representation names its output columns, and the names are stable and descriptive.
`get_feature_names_out()` returns them in output order.

```python
pre.get_feature_names_out()
```

Use `get_feature_info(verbose=True)` for a human-readable table of the resolved per-feature
pipeline, output width, and category count.

```text
feature  kind         pipeline                        dim   cats
----------------------------------------------------------------
age      numerical    imputer -> minmax -> bspline     13      -
income   numerical    imputer -> minmax -> ple         12      -
city     categorical  imputer -> onehot -> to_float     4      4
```

## Feature lineage

Lineage is the flagship inspection feature. `get_feature_lineage()` returns one record per
output column, mapping it back to its origin.

```python
lineage = pre.get_feature_lineage()
lineage[0]
```

Each `FeatureLineage` record carries:

- the **source input column(s)** the output came from,
- the **representation** family that produced it,
- the **component** it corresponds to (a basis function, knot, center, frequency, interval,
  or category),
- whether the **target was used** to fit it,
- whether it is an **interaction** across several inputs.

```{tip}
Lineage covers every output column and the names line up with `get_feature_names_out()`. This
makes a fitted `Preprocessor` fully auditable, which is invaluable when you interpret a linear
model fit on top of the expansion.
```

## Output budget

Expansions can multiply columns quickly, especially wide splines or high-cardinality one-hot.
The output budget lets you cap the blast radius and estimate cost before committing.

| Parameter | Effect |
| --- | --- |
| `max_output_features` | Cap on total output columns. |
| `max_features_per_input` | Cap on columns produced from any single input. |
| `max_dense_memory` | Cap on dense output memory. |
| `overflow_policy` | What to do on overflow, default `"error"`. |

```python
pre = Preprocessor(max_output_features=500, overflow_policy="error")

pre.estimate_output_shape(df)   # predicted (n_rows, n_cols) without transforming
pre.estimate_memory(df)         # predicted dense memory in bytes
```

```{warning}
With `overflow_policy="error"`, exceeding a budget raises `OutputBudgetError` at `fit`. Use
`estimate_output_shape` and `estimate_memory` first when you work with wide expansions or
large data.
```

## Where to go next

- [Reproducibility](reproducibility.md) to serialize and fingerprint the fitted output.
- [Representations](../representations/overview.md) for what each family emits.
- [Comparing representations](../tutorials/comparing_representations.md) to measure width and
  memory.
