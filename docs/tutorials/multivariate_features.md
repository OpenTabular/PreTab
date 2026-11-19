# Multivariate features

Most representations transform one column at a time. Some relationships, though, live in the
interaction between columns: a smooth surface over latitude and longitude, or a kernel over
many inputs at once. PreTab's multivariate methods model several columns jointly. This tutorial
shows how to use them.

## Which methods are multivariate

Four methods operate on several inputs together rather than per column.

`tensorspline`
: Tensor-product spline. A smooth basis over a small number of inputs, capturing their
interaction on a grid.

`tprs`
: Thin-plate regression spline. A smooth surface over two or more inputs, from the generalized
additive model literature. It also accepts a single input, though a univariate spline family
is usually a more natural fit there.

`rff`
: Random Fourier features. A scalable approximation to a shift-invariant kernel.

`nystroem`
: Nyström kernel map. A landmark-based kernel approximation.

```{note}
Use `feature_groups` on `Preprocessor` to apply `tensorspline` or `tprs` to named
numerical columns jointly. Neither is a per-column `numerical_method`. Random
Fourier features and Nyström remain standalone transformers; use a scikit-learn
`ColumnTransformer` to combine them with other representations.
```

## Grouped preprocessing with Preprocessor

A feature group owns its source columns and produces one joint block. Other
columns keep their per-column representations. Here latitude and longitude share
a tensor-product basis, size receives an RBF expansion, and city is one-hot encoded.

```python
import numpy as np
import pandas as pd

from pretab import FeatureGroup, Preprocessor, RepresentationConfig

group_rng = np.random.default_rng(17)
data = pd.DataFrame({
  "longitude": group_rng.uniform(-3, 3, 96),
  "latitude": group_rng.uniform(-2, 2, 96),
  "size": group_rng.uniform(20, 200, 96),
  "city": np.resize(["Lima", "Oslo", "Rome"], 96),
})
data.loc[0, "latitude"] = np.nan

location = FeatureGroup(
  name="location",
  columns=("latitude", "longitude"),
  representation=RepresentationConfig(
    method="tensorspline", output_dim=3, params={"degree": 2},
  ),
)
pre = Preprocessor(
  numerical_method="none",
  categorical_method="one-hot",
  feature_preprocessing={"size": {"method": "rbf", "output_dim": 3}},
  feature_groups=[location],
  target_aware=False,
  placement_strategy="uniform",
  missing_policy="separate_state",
  output_structure="blocks",
  output_format="dense",
)
blocks = pre.fit_transform(data)
assert list(blocks) == ["num_size", "group_location", "cat_city"]
```

Each group must contain at least two detected numerical columns. A source may
belong to only one group, and cannot also have an explicit `feature_preprocessing`
entry. Categorical and Boolean columns cannot be group sources. Set
`treat_all_integers_as_numerical=True` when low-cardinality integer sources
would otherwise be detected as categorical.
See [Configuration](../core_concepts/configuration.md) for feature detection settings.

Group order and source order have different meanings. Output blocks appear as
ungrouped numerical columns, groups in declaration order, then categorical columns.
The ordered `columns` tuple determines a group's input axes; it does not need to
match the DataFrame's column order. Named inference columns are matched by label.
Arrays and lists instead use fitted input positions and must have the same width.

### Read the fitted width

Tensor-product `output_dim` is the width **per input axis**, not the total group
width. Three basis functions on each of two axes produce $3^2 = 9$ joint basis
columns. With `missing_policy="separate_state"`, two additional raw missing-state
columns follow that basis, in declared source order. The group therefore has
eleven output columns, even when one source has no missing values during fitting.

```python
assert pre.output_dims_["group_location"] == 11
assert blocks["group_location"].shape == (len(data), 11)
np.testing.assert_array_equal(blocks["group_location"][0, -2:], [1.0, 0.0])
assert sum(pre.output_dims_.values()) == pre.total_output_dim_

names = pre.get_feature_names_out()
lineage = pre.get_feature_lineage()
assert len(names) == len(lineage) == pre.total_output_dim_
location_lineage = [item for item in lineage if item.group_id == "group_location"]
assert len(location_lineage) == 11
assert all(
  item.source_features == ("latitude", "longitude")
  for item in location_lineage if item.component != "indicator"
)
assert not any(item.uses_target for item in location_lineage)
```

Names encode the group and its ordered sources. Joint basis columns trace back to
both inputs; each missing indicator traces back to its own raw source. The two
sources do not receive duplicate copies of the joint width in `output_dims_` or
`get_feature_info()`.

```{warning}
Tensor-product width grows multiplicatively with the number of sources. Inspect
the fitted width and use output budgets before passing a large joint basis to a
model. `max_features_per_input` applies to the whole group, including raw missing
states, not to an average width per source.
```

```python
bounded = pre.clone_unfitted().set_params(
  max_features_per_input=16,
  max_output_features=64,
)
bounded.fit(data)
assert bounded.output_dims_["group_location"] == 11
```

The built-in tensor and thin-plate groups do not use the target. Tensor-product
splines use local `output_dim`; thin-plate splines use `n_components` to size the
joint basis. To replace the tensor basis with a six-column thin-plate basis:

```python
thin_plate = FeatureGroup(
  name="location",
  columns=("latitude", "longitude"),
  representation=RepresentationConfig(
    method="tprs",
    params={
      "n_components": 6,
      "landmark_strategy": "subsample",
      "random_state": 17,
    },
  ),
)
thin_pre = pre.clone_unfitted().set_params(feature_groups=[thin_plate])
thin_blocks = thin_pre.fit_transform(data)
assert thin_blocks["group_location"].shape[1] == 8
```

The thin-plate group has six joint basis columns plus the same two raw indicators.
Fitted widths are authoritative; they also include categorical encodings and
missing-state columns outside the group.

### Mapping syntax

An ordered mapping is equivalent to the typed group declaration. Representation
options belong inside `representation`, not beside `columns`.

```python
mapped = pre.clone_unfitted().set_params(feature_groups={
  "location": {
    "columns": ["latitude", "longitude"],
    "representation": {"method": "tensorspline", "output_dim": 3, "degree": 2},
  },
})
mapped_blocks = mapped.fit_transform(data)
for key in blocks:
  np.testing.assert_array_equal(mapped_blocks[key], blocks[key])
```

Typed values detach their settings when constructed. Raw constructor mappings
remain live until normalization or fitting. After fitting, `set_params` changes
the requested configuration, not the existing fitted representation. Refit an
unfrozen instance, or use `clone_unfitted()` to learn a different representation.

### Block widths for DeepTab

A block-based model needs one input width per output block, not one joint width
for each raw source. Read those widths from fitted output:

```python
block_widths = {name: values.shape[1] for name, values in blocks.items()}
assert list(block_widths) == ["num_size", "group_location", "cat_city"]
assert block_widths["group_location"] == 11
assert "latitude" not in block_widths and "longitude" not in block_widths
assert sum(block_widths.values()) == pre.total_output_dim_
print(block_widths)
```

`group_location` is one model input with eleven coordinates. Do not divide eleven
by the two sources or create two identical joint inputs. This example stops at
PreTab's dense block-output boundary. It does not construct or fit a DeepTab
model, and `block_widths` is not a DeepTab configuration object.

```{note}
Dictionary output is not supported for out-of-fold assembly by
`CrossFittedTransformer`. Use matrix output for that training workflow. An
ordinary all-data `fit` followed by a dictionary `transform` is not a substitute
for out-of-fold features when another representation uses the target.
```

### Preserve a fitted group

Specs retain fitted group order, learned parameters, output metadata and frozen
state. The example uses strict JSON and reloads the preprocessor without fitting:

```python
import json

pre.freeze()
restored = Preprocessor.from_spec(json.loads(json.dumps(pre.to_spec(), allow_nan=False)))
restored_blocks = restored.transform(data[data.columns[::-1]])
assert restored.is_frozen()
assert restored.fingerprint_ == pre.fingerprint_
np.testing.assert_array_equal(restored.get_feature_names_out(), names)
for key in blocks:
  np.testing.assert_array_equal(restored_blocks[key], blocks[key])
```

Use specs only from trusted sources; loading can import supported library code.
A spec does not guarantee compatibility across dependency versions. See
[Reproducibility](../core_concepts/reproducibility.md) for the persistence contract.

## A smooth surface with thin-plate splines

Suppose the target is a smooth function of two coordinates. A per-column expansion cannot see
the interaction, but a thin-plate spline models the surface directly.

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from pretab.transformers import ThinPlateSplineTransformer

rng = np.random.default_rng(0)
n = 3000
X = rng.uniform(-3, 3, size=(n, 2))
y = np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2)) * 5 + rng.normal(0, 0.2, n)

model = Pipeline([
    ("tps", ThinPlateSplineTransformer(n_components=20)),
    ("ridge", Ridge(alpha=1.0)),
])

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-fold R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

The thin-plate basis captures the radial bump over the two coordinates jointly, something two
separate one-dimensional splines cannot do.

```{tip}
Use `n_components` to trade accuracy for cost. More landmarks give a richer surface at higher
memory and compute. Start modest and increase only if validation improves.
```

## A scalable kernel with random Fourier features

When you want kernel-style flexibility over many inputs on a large dataset, random Fourier
features approximate an RBF kernel without forming the full kernel matrix.

```python
from pretab.transformers import RandomFourierFeaturesTransformer

model = Pipeline([
    ("rff", RandomFourierFeaturesTransformer(n_components=200, gamma=0.5)),
    ("ridge", Ridge(alpha=1.0)),
])

scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"5-fold R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

```{note}
Random Fourier features and Nyström both approximate a kernel machine. Random Fourier features
scale to large data with random projections; Nyström samples landmark points and is often more
accurate at a given width. Try both.
```

## Combining multivariate and per-column methods

You can mix a joint block for interacting columns with per-column expansions for the rest,
using a `ColumnTransformer`.

```python
from sklearn.compose import ColumnTransformer
from pretab.transformers import PLETransformer
import pandas as pd

df = pd.DataFrame({"lat": X[:, 0], "lon": X[:, 1], "size": rng.uniform(0, 100, n)})

features = ColumnTransformer([
    ("geo", ThinPlateSplineTransformer(n_components=20), ["lat", "lon"]),
    ("size", PLETransformer(output_dim=10, task="regression"), ["size"]),
])

model = Pipeline([("features", features), ("ridge", Ridge(alpha=1.0))])
```

The thin-plate spline handles the geographic interaction while PLE handles the standalone
`size` column, each with the representation that suits it.

## Where to go next

- [Spline expansions](../representations/spline_expansions.md) for the tensor-product and
  thin-plate details.
- [Kernel approximation](../representations/kernel_approximation.md) for random Fourier
  features and Nyström.
- [Representations overview](../representations/overview.md) for the underlying theory and supporting notes.
