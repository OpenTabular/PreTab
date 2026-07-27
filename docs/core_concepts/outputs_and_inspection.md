# Outputs and inspection

A representation is only useful if you can read what it produced. PreTab returns model-ready
output in the format you ask for, names every column, and can trace each output column back to
the exact input and component that created it. This page covers output shapes, formats,
feature names, lineage, and the output budget.

## Output shapes

`fit_transform` and `transform` return a dictionary that maps each feature to its transformed
block, with keys prefixed `num_` or `cat_`. Pass `return_array=True` to receive a single
stacked `numpy.ndarray` instead.

```python
X_dict = pre.fit_transform(df, y)              # {"num_age": ..., "cat_city": ...}
X_array = pre.transform(df, return_array=True) # one stacked ndarray
```

```{note}
The dict form is convenient for inspection and for feeding blocks to different model heads.
The array form is what a plain scikit-learn estimator expects. Choose per call.
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
Polars output is loaded lazily. If polars is not installed, requesting it raises a clear
`OptionalDependencyError` rather than failing deep in the call stack.
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
