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

## Portable serialization

`to_spec` writes a fitted `Preprocessor` to a versioned, declarative schema, and `from_spec`
reconstructs it. The spec records the schema and library versions, the resolved parameters,
and the per-representation fitted state (parameters, knots, centers, columns, scaling).

```python
spec = pre.to_spec()                 # returns a dict
pre.to_spec("representation.json")   # or writes JSON to a path

restored = Preprocessor.from_spec("representation.json")
```

```{important}
`from_spec` is a safe alternative to pickle. Reconstruction imports only from `pretab`,
`scikit-learn`, `numpy`, `scipy`, and builtins, and it never executes arbitrary estimator
code. A spec from an untrusted source cannot run code the way an untrusted pickle can.
```

A round-trip reproduces `transform` bit-for-bit, so a spec is a faithful, human-readable
record of a fitted representation.

## Fingerprint

`fingerprint_` is a SHA-256 hash over a canonical view of the fitted representation: the
resolved config, the schema, the fitted parameters, the output-column order, the seeds, the
library versions, and the output precision.

```python
pre.fit(df, y)
pre.fingerprint_
```

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

## Immutable lifecycle

A fitted representation moves through a small set of explicit states, which prevents
accidental mutation of something you intend to deploy.

| State | Meaning |
| --- | --- |
| `UNFITTED` | Constructed, not yet fit. |
| `FITTED` | Fit and ready to transform. |
| `FROZEN` | Locked against parameter changes. |
| `STALE` | Marked as no longer current, with a reason. |

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
`set_params` on a frozen preprocessor raises `FrozenRepresentationError`. This is deliberate:
a deployed representation should not silently change shape. Use `refit` to produce a new
object instead of mutating the old one.
```

## Where to go next

- [Outputs and inspection](outputs_and_inspection.md) for the output the fingerprint covers.
- [Target awareness](target_awareness.md) for how supervised state is recorded.
- [Production lifecycle](../developer_guide/release.md) for versioning and release discipline.
