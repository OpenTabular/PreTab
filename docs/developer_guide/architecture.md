# Architecture contract

PreTab separates representation algorithms, the composition that selects and wires
them, and the public `Preprocessor` that owns the fitting lifecycle. Use these
boundaries when adding a representation or a consumer of fitted state.

## Dependency directions

- `pretab.core` owns shared estimator, validation, and representation primitives.
  It depends on core peers and shared exceptions, not on registered methods or
  the facade. `pretab.placement` uses core and placement peers, not concrete
  representations.
- Expansion, encoding, kernel approximation, embedding, and preprocessing modules
  use their own implementations and lower-level core, placement, or exception
  facilities as needed. They do not import composition, the facade, extension
  registration, or the public reexport modules.
- `pretab.compose` selects and assembles concrete implementations. The registry is
  the source of method capabilities; config resolves requests and the factory builds
  pipelines from the resolved settings. Add a method capability there rather than
  duplicating it in the facade or an output consumer.
- `Preprocessor` orchestrates fitting, inference, and lifecycle operations through
  composition and core. The extension API registers through the same registry and
  uses the core representation contract.

Two composition modules intentionally refer to the facade: `pretab.compose.search`
constructs candidates and `pretab.compose.serialize` imports `Preprocessor` locally
to restore one. These are upper-layer exceptions, not a dependency for a representation
implementation. The codec's controlled class resolution is separate from static imports.

Use the documented public imports in application and plugin code. `pretab` exports
the facade, configuration values, and extension functions. `pretab.transformers`
reexports concrete transformers from their implementation families. Inside PreTab,
import from the owner rather than routing implementation dependencies through these
convenience exports. See the [API reference](../api/preprocessor.rst) and
[extensibility API](../api/extension.rst) for the supported entry points.

## Requests, resolution, and fitted state

Keep these views distinct:

- `get_params()` exposes the current constructor requests, preserving spellings and
  caller-owned raw mappings. Typed requests such as `RepresentationConfig` detach
  their input data at construction.
- `get_resolved_config()` normalizes current requests and applies defaults and
  inheritance. It does not fit, inspect the input, or recover learned state.
- The last successful fit records its effective settings separately. Supported specs
  expose them as `resolved_config` when available; legacy specs need not have that
  record. The fitted estimators and their actual layout are authoritative for inference.

Changing a request with `set_params()` does not replace the fitted representation.
This example checks the live and recorded methods before the next fit:

```python
import numpy as np
import pandas as pd

from pretab import Preprocessor, RepresentationConfig

frame = pd.DataFrame({"value": np.linspace(0.0, 1.0, 32)})
pre = Preprocessor(
    feature_preprocessing={"value": RepresentationConfig(method="rbf", output_dim=3)},
    target_aware=False,
    placement_strategy="uniform",
).fit(frame)
fitted_output = pre.transform(frame, return_array=True)

pre.set_params(feature_preprocessing={"value": "minmax"})
assert pre.get_params(deep=False)["feature_preprocessing"] == {"value": "minmax"}
current = pre.get_resolved_config()["feature_preprocessing"]
recorded = pre.to_spec()["resolved_config"]["feature_preprocessing"]
assert current["value"]["method"] == "minmax"
assert recorded["value"]["method"] == "rbf"
np.testing.assert_array_equal(pre.transform(frame, return_array=True), fitted_output)
```

## Fitted layout and lineage

The fitted `ColumnTransformer` owns the ordered blocks and their output spans.
Composition inspection reads the fitted entries and `output_indices_`; output
formatting consumes those spans. Do not infer the number or order of columns from
a request, refit a block for inspection, or use a leaf estimator's basis width as
the width of its whole pipeline. Adaptive sizing, joint bases, and missing-state
columns can all make the actual width differ from the request.

Use `output_dims_` and `total_output_dim_` when sizing a consumer, and
`get_feature_names_out()` and `get_feature_lineage()` when labeling it. A group
owns its sources and contributes one block. Joint basis columns record all of the
group's sources in declared order; each raw missing-state column records only its
own source but retains the same group ID. The example requests three basis
functions per axis, yielding nine joint columns and two missing-state columns.

```python
from pretab import FeatureGroup

group_frame = frame.assign(other=np.sin(4.0 * frame["value"]))
group_frame.loc[0, "value"] = np.nan
group = FeatureGroup(
  name="pair",
  columns=("other", "value"),
  representation=RepresentationConfig(method="tensorspline", output_dim=3, params={"degree": 2}),
)
grouped = Preprocessor(
  feature_groups=[group],
  target_aware=False,
  placement_strategy="uniform",
  missing_policy="separate_state",
  output_structure="blocks",
  output_format="dense",
)
blocks = grouped.fit_transform(group_frame)
lineage = grouped.get_feature_lineage()

assert list(blocks) == ["group_pair"]
assert blocks["group_pair"].shape[1] == grouped.output_dims_["group_pair"] == 11
assert len(lineage) == len(grouped.get_feature_names_out()) == grouped.total_output_dim_ == 11
assert all(item.group_id == "group_pair" for item in lineage)
assert [item.source_features for item in lineage if item.component != "indicator"] == [group.columns] * 9
assert [item.source_features for item in lineage if item.component == "indicator"] == [("other",), ("value",)]
```

## Input-only state and supervised refitting

When `CrossFittedTransformer` wraps a `Preprocessor`, it uses the fitted block's
actual target use to decide what to refit. A target-consuming representation and
any steps after it are refit on the fold's training rows. Input-only state from
the full input to the wrapper stays fitted: column routing, target-free blocks
and their vocabularies, imputers, scalers, and separate missing branches. A group
still uses its ordered sources as one block. This keeps category identities and
indicator columns aligned across folds without letting a representation use the
held-out row's target. The refitted block must retain its fitted width.

```{note}
Cross-fitting produces a matrix, not a dictionary of blocks. It protects the
held-out targets within the data passed to the wrapper. For evaluation, still
fit the whole pipeline within each outer training split.
```

## Optional dependencies and persistence

Keep optional packages at the feature boundary. LightGBM placement and language
embeddings use the lazy dependency helpers in `pretab.core.dependencies`; Polars
input and output are handled at the input and output boundaries. Do not make a
base import or an unrelated method require an extra. Preserve an actionable
`OptionalDependencyError` for the path that needs the missing package.

Registration makes a representation selectable for fitting and inference. It does
not register a persistence codec or extend the schema-v1 import policy. If a
fitted estimator's module is outside that policy, `to_spec()` rejects it with
`PretabSerializationError` before opening a destination, even if it is nested
inside a supported pipeline. `fingerprint_` and `reproducibility_report()` raise
the same error rather than dropping state. Use supported built-in representations
for JSON replay, or a trusted pickle or joblib checkpoint in a matching environment
if all components support it. See [Reproducibility](../core_concepts/reproducibility.md)
for the artifact and trusted-source limits.
