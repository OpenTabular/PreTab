# Reproducibility

A representation you cannot reproduce is a representation you cannot trust in production or in
a paper. PreTab treats reproducibility as a contract: deterministic fitting, a portable
declarative spec, a stable fingerprint, and an immutable lifecycle. This page covers all four.

## Deterministic fitting

`random_state` seeds every stochastic step: the target-aware selectors, k-means landmark
placement, and the randomized feature maps. Set it to an integer whenever you need repeatable
output, for example in tests or published experiments.

```python
from pretab import Preprocessor

pre = Preprocessor(numerical_method="rbf", random_state=0)
```

```{note}
With a fixed `random_state`, repeated fits on the same data produce identical output. Methods
with no stochastic component ignore the seed. The standalone kernel approximations
(`RandomFourierFeaturesTransformer`, `NystroemFeaturesTransformer`) accept `random_state`
directly; they are fit outside `Preprocessor` since they operate on the whole feature block.
```

(portable-serialization)=

## Portable serialization

`to_spec()` exports a fitted `Preprocessor` as a JSON-compatible dictionary and can also
write it to a file. `from_spec()` restores it from either form without refitting.

```python
pre.fit(df, y)
spec = pre.to_spec()                 # returns a dict
pre.to_spec("representation.json")   # or writes JSON to a path

restored = Preprocessor.from_spec("representation.json")
```

### What's in a spec

- `schema_version`, `pretab_version`, and `library_versions` identify the format and the
  environment that wrote the spec.
- `params` holds constructor requests at export time. These can change after fit.
- `resolved_config` holds effective settings recorded at the last successful fit, including
  normalized declared feature overrides and groups in declared order. It is absent if the
  fit has no settings record.
- `feature_names_out` and `feature_lineage` give the output column order and fitted
  provenance, including group identity and ordered sources where applicable.
- `output_dims` and `total_output_dim` give actual per-block and total widths, including
  separate missing indicators. These are not requested `output_dim` values.
- `representations` describes blocks that expose representation metadata. Group entries
  include `group_id`, ordered `columns`, fitted `dimension`, and a half-open `output_slice`.
  Supported typed penalties add compact `penalties` metadata, including coefficient
  coordinates and a matrix checksum, not the matrix itself.
- `state` is the encoded fitted object state. This is what the reader restores; the
  descriptive fields alone cannot recreate the learned transformation.

Use `get_resolved_config()` to inspect effective settings for the **current** requests.
It is not a substitute for the recorded fit-time `resolved_config`. Don't edit a spec to
change a fit: the reader checks recorded configuration and layout fields against the
stored state where applicable and raises `PretabSerializationError` on mismatches.
Those checks do not authenticate the artifact.

### Compatibility

The current writer still uses `schema_version: 1`. It writes strict JSON, encoding
non-finite floats such as the default imputer's `np.nan` as tagged values, for example
`{"__float__": "nan"}`, instead of bare `NaN`. Standard JSON parsers can inspect new specs;
restoring the fitted object requires PreTab.

The current reader loads supported PreTab 1.0.0 schema-v1 specs, including legacy bare
`NaN` tokens and specs without the added metadata. It does not invent a missing fit-time
settings record. **Downgrades are not supported:** PreTab 1.0.0 cannot decode the new
float tags, even though the schema number is unchanged. Keep the original artifact and
its environment for rollback. The change to tagged encoding can affect fingerprints for
state containing non-finite values without changing transformed values.

```{important}
Load specs only from trusted sources. Reconstruction restricts imports to `pretab`,
`scikit-learn`, `numpy`, and `scipy`, reconstructs dataclasses only from an exact, closed
allow-list, and bypasses estimator initialization and pickle hooks. These restrictions are
not a sandbox: importing library modules can still execute code. Recorded versions are
informational, not a guarantee of compatibility. Use the original environment for exact
replay, and verify fitted outputs when upgrading.
```

### Spec or pickle?

Use a spec when you need an inspectable JSON artifact for supported fitted PreTab state.
Supported built-in representations reproduce `transform` bit-for-bit in the same environment.
The spec covers preprocessing, not a downstream model. Registering a third-party
representation enables fitting and transforming, not spec persistence. `to_spec()` raises
`PretabSerializationError` and names the unsupported estimator if its module is outside
the spec's allowed modules, even when it is nested in an otherwise supported pipeline.
The fitted object is unchanged, and `to_spec(path)` raises before opening the destination.
Use supported built-in representations when you need JSON replay; don't rely on specs
to persist third-party representations or pretrained language models.

Use pickle or joblib when you need to persist a larger Python object, such as a complete
scikit-learn pipeline, and every component supports that mechanism. Use a matching
environment to restore the checkpoint; neither format guarantees cross-version replay.
Unpickling can execute arbitrary code: only load trusted pickle or joblib artifacts.
The same trusted-source requirement applies to specs.

```{note}
Restoring a PreTab 1.0.0 `Preprocessor` from a spec or trusted pickle repairs
the stale feature count in pass-through and float-cast helpers directly after
sparse one-hot encoders. With unchanged dependencies, these objects retain their
transformed values and names without refitting, and can again export specs and
fingerprints. Feature information reports the fitted layout; previously incorrect
dimension metadata can change. Standalone helper validation is unchanged.
Fingerprints can differ from 1.0.0 because the fitted metadata and JSON encoding
have been corrected. This targeted repair is not a general guarantee of
cross-version compatibility.
```

When refitting after an upgrade, check output names and widths before reusing a
trained model. If the feature layout changes, refit the preprocessing and retrain
the model together.

## Fingerprint

`fingerprint_` is a SHA-256 hash over a canonical view of the fitted representation: the
resolved config, the schema, the fitted parameters, the output-column order, the seeds, the
library versions, and the output precision.

```python
pre.fit(df, y)
pre.fingerprint_
```

Inference reports and advisory lifecycle flags are excluded from the fingerprint.
The fingerprint is deterministic within a process and across processes, and it survives a
`to_spec` / `from_spec` round-trip. Two preprocessors with the same fingerprint will produce
the same output; a change to config, data, seed, or version changes the fingerprint.

```{tip}
Log the fingerprint alongside model metrics. If it changes unexpectedly between runs, your
representation changed, which is exactly the signal you want before you chase a metric
regression.
```

`reproducibility_report()` returns a structured summary for logging: the fingerprint,
versions, seed, output dtype and format, output widths, and the per-feature families.

```python
pre.reproducibility_report()
```

`fingerprint_` and `reproducibility_report()` require supported spec state. Both raise
`PretabSerializationError` on unsupported estimator state rather than returning a digest
or partial report.

## Immutable lifecycle

A fitted representation moves through a small set of explicit states, which prevents
accidental mutation of something you intend to deploy.

| State      | Meaning                                     |
| ---------- | ------------------------------------------- |
| `UNFITTED` | Constructed, not yet fit.                   |
| `FITTED`   | Fit and ready to transform.                 |
| `FROZEN`   | Locked against parameter changes.           |
| `STALE`    | Marked as no longer current, with a reason. |

```python
pre.freeze()          # lock it
pre.is_frozen()       # True
pre.set_params(...)   # raises FrozenRepresentationError while frozen
```

Freezing is useful when a representation is validated and about to ship. To make a fresh,
unfrozen copy, use `clone_unfitted()`. To retrain, `refit(X, y)` returns a **new** fitted
object and leaves the original untouched, and `mark_stale(reason)` records why an existing one
should no longer be used.

```{warning}
Attempts to refit a frozen preprocessor, change its parameters, or change its
`set_output` format raise `FrozenRepresentationError`. Use `clone_unfitted()`
for a mutable copy or `refit()` to produce a new fitted object.
```

`set_output(transform=None)` leaves the configuration unchanged. Repeating an
explicitly selected format, such as `"pandas"`, is also allowed after freezing.
Selecting a format for the first time is a change, even if it matches the
current global default.

```{note}
Freezing protects instance settings, not sklearn's global configuration. If no
instance format is set, `sklearn.set_config(transform_output="pandas")` can
still change the returned container without changing the fingerprint. Call
`set_output(transform="default")`, or select another supported format, before
`freeze()` to make the output container independent of global settings.
```

## Where to go next

- [Outputs and inspection](outputs_and_inspection.md) for the output the fingerprint covers.
- [Target awareness](target_awareness.md) for how supervised state is recorded.
- [Production lifecycle](../developer_guide/release.md) for versioning and release discipline.
