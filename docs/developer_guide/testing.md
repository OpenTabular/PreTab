# Testing

PreTab has a comprehensive test suite that gates every change. This page explains how the tests
are organized and how to run them.

## Running the tests

The suite runs with coverage through a single recipe.

```bash
just test     # poetry run pytest --cov=pretab tests/
```

To run a subset while developing, invoke pytest directly.

```bash
poetry run pytest tests/transformers/            # one area
poetry run pytest tests/transformers/test_bspline.py::test_output_shape   # one test
poetry run pytest -k "spline and not tensor"     # by keyword
```

## Layout

Tests mirror the structure of the package, so a change in one area maps to an obvious test
directory.

| Directory | Covers |
| --- | --- |
| `tests/core/` | Base classes, adaptive resolution, supervised logic, logging. |
| `tests/transformers/` | Every representation, per family. |
| `tests/placement/` | Knot and edge placement strategies. |
| `tests/compose/` | Registry, feature detection, config resolution, serialization. |
| `tests/extension/` | The public extensibility surface and conformance. |
| `tests/integration/` | End-to-end `Preprocessor` and pipeline behaviour. |
| `tests/regression/` | Pinned outputs that guard against silent numerical drift. |

```{note}
Regression tests pin known-good output. If one fails after a deliberate change to a
representation, update the pinned values in the same commit and call it out in the pull
request, so the change is reviewed rather than hidden.
```

## Markers

The suite defines a `smoke` marker for fast end-to-end sanity checks that run as a dedicated CI
gate.

```bash
poetry run pytest -m smoke        # only the smoke checks
poetry run pytest -m "not smoke"  # everything else
```

## Coverage

`just test` measures coverage over the `pretab` package. Keep new code covered, and prefer a
focused test that exercises the behaviour over one that merely touches lines.

```bash
poetry run pytest --cov=pretab --cov-report=term-missing tests/
```

## Testing a custom representation

If you extend PreTab, run the conformance suite in your own tests. It verifies your class obeys
the representation contract, the same one the built-ins satisfy.

```python
from pretab import check_representation
from my_package import MyRepresentation

def test_conforms():
    check_representation(MyRepresentation)
```

```{important}
`check_representation` raises `RepresentationConformanceError` on any violation. Wiring it into
your test suite keeps a future refactor from silently breaking compatibility with `Preprocessor`.
```

## Before you push

Run the full local gate, which mirrors CI.

```bash
just test     # tests with coverage
just check    # lint, format, type-check across all files
just docs     # strict docs build
```

## Where to go next

- [Contributing](contributing.md) for the full pull-request workflow.
- [Writing a custom representation](../tutorials/custom_representation.md) for the conformance
  suite in context.
- [Documentation](documentation.md) for the docs build the last command runs.
