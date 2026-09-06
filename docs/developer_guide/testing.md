# Testing

PreTab has a comprehensive test suite that gates every change. This page explains how the tests
are organized and how to run them.

## Running the tests

The suite runs with coverage through a single recipe.

```bash
just test     # poetry run pytest --cov=pretab --cov-branch --cov-fail-under=90 tests/
```

To run a subset while developing, invoke pytest directly.

```bash
poetry run pytest tests/expansion/            # one area
poetry run pytest tests/expansion/spline/test_spline_expansions.py -k bspline   # one test
poetry run pytest -k "spline and not tensor"     # by keyword
```

## Layout

Tests mirror the structure of the package, so a change in one area maps to an obvious test
directory.

| Directory                                                                                | Covers                                                                                                                                                      |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/core/`                                                                            | Base classes, adaptive resolution, supervised logic, logging.                                                                                               |
| `tests/expansion/`, `tests/encoding/`, `tests/kernel_approximation/`, `tests/embedding/` | Every representation family, split by kind (splines and functional expansions, numerical/categorical encoders, kernel approximations, language embeddings). |
| `tests/transformers/`                                                                    | Cross-family contracts: sklearn compatibility, feature names, output dimensions, parameter aliases, encoder counts.                                         |
| `tests/placement/`                                                                       | Knot and edge placement strategies.                                                                                                                         |
| `tests/compose/`                                                                         | Registry, feature detection, config resolution, serialization.                                                                                              |
| `tests/extension/`                                                                       | The public extensibility surface and conformance.                                                                                                           |
| `tests/integration/`                                                                     | End-to-end `Preprocessor` and pipeline behaviour.                                                                                                           |
| `tests/regression/`                                                                      | Pinned outputs that guard against silent numerical drift.                                                                                                   |
| `tests/doc_snippets/`                                                                    | Executes the `docs/tutorials/*.md` code fences, so the tutorials cannot silently rot.                                                                       |

```{note}
Regression tests pin known-good output. If one fails after a deliberate change to a
representation, update the pinned values in the same commit and call it out in the pull
request, so the change is reviewed rather than hidden.
```

## Testing mathematical correctness, not just shape

A representation-heavy library like PreTab has a failure mode that shape and dtype checks
cannot catch: a basis function, penalty matrix, or encoding can be computed with the wrong
formula and still produce output of the right shape, the right dtype, and finite values. A
test that only asserts `X.shape == (n, k)` or `np.isfinite(out).all()` will pass on both the
correct and the incorrect implementation.

When a representation has a closed-form mathematical property, test that property directly
instead of (or in addition to) its shape:

- **Known identities.** A B-spline basis should sum to `1` at every point (partition of
  unity); an M-spline should integrate to `1` over its own support; an I-spline should be
  monotonically non-decreasing and bounded in `[0, 1]`.
- **Independent reference values.** A penalty matrix or a hand-derivable formula (a
  particular basis value at a particular knot, say) can be checked against a value computed
  a different way, for example a fine numerical quadrature or a direct closed-form
  substitution, not just re-derived with the same code path the implementation itself uses.
- **Boundary behaviour.** Values at, or just past, a fitted range's edge are where
  clipping-versus-extrapolation bugs and off-by-one integration bounds hide. Test a value
  exactly at the boundary and one just beyond it, not only values safely inside the range.
- **Realistic missing-data shapes.** A mixed object array with an actual `NaN`/`None` among
  string categories (the ordinary shape of a pandas column with missing values) is a
  different code path than an all-numeric array with `NaN`, and needs its own test if a
  transformer declares `allow_nan=True`.

```{warning}
Shape/symmetry/finiteness assertions are still useful as a first line of defense, but they
are not sufficient proof that a mathematical implementation is correct: they pass equally
well on a subtly wrong formula as on a correct one. Pair them with at least one value-level
assertion for anything that has a defined mathematical property to check against.
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

Run `just check` and `just test` locally; together they cover most of what CI checks, though
CI additionally runs across the full Python 3.10-3.13 matrix, builds the package, and
enforces a branch-coverage threshold.

```bash
just test     # tests with coverage
just check    # lint, format, type-check across all files
just docs     # strict docs build
just quickstart  # end-to-end sanity check: same script CI's smoke job runs
```

## Where to go next

- [Contributing](contributing.md) for the full pull-request workflow.
- [Writing a custom representation](../tutorials/custom_representation.md) for the conformance
  suite in context.
- [Documentation](documentation.md) for the docs build the last command runs.
