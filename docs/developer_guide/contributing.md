# Contributing

Thanks for contributing to pretab. This page covers environment setup, the local
workflow, and what a pull request needs to pass review.

## Code of Conduct

All contributors are expected to follow the project
[Code of Conduct](https://github.com/OpenTabular/PreTab/blob/main/CODE_OF_CONDUCT.md),
which sets the standard for respectful and inclusive participation.

## Setting up the development environment

The project uses [Poetry](https://python-poetry.org/docs/) for dependency management and
the [just](https://just.systems/man/en/) command runner for common tasks (the `justfile`
defines testing, building, and formatting).

1. Clone the repository:

```bash
git clone https://github.com/OpenTabular/PreTab
cd PreTab
```

2. Install the prerequisites: `pip install poetry` and `just` (see the
   [just install guide](https://just.systems/man/en/packages.html), e.g. `brew install just`).

3. Install dependencies and register the pre-commit hooks:

```bash
just install
```

Without `just`, run the same steps directly:

```bash
poetry install
poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
```

4. To work on the docs, also install the docs group with `poetry install --with docs`.

## How to contribute

1. Branch off `main` with a short, descriptive name.
2. Make your changes, keeping each pull request to a single logical focus.
3. Add or update tests, and run the full check suite locally before pushing:

```bash
just test     # full suite with coverage
just check    # ruff format, ruff lint, and pyright, via the pre-commit and pre-push hooks
just docs     # build HTML docs (warnings treated as errors)
```

4. Commit using Conventional Commits via `just commit`. If `just check` reformats files,
   commit those separately with `style: apply ruff formatting`.
5. Open a pull request to `main`, reference any related issues, and address review
   feedback until approved and merged.

## Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code quality
automatically. `just install` registers all three hook types so each fires at the right
time:

| Stage        | Hook                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| `commit-msg` | Validates the message against Conventional Commits.                     |
| `pre-commit` | `ruff` format and lint, plus file hygiene (whitespace, EOF, conflicts). |
| `pre-push`   | `pyright` type checking (slower, so deferred to push). Also runs in CI. |

```{important}
Run `just check` before opening a PR. It runs every `pre-commit`- and `pre-push`-stage hook
(ruff format, ruff lint, pyright, and file hygiene checks) against every file. It does not
validate the commit message itself, that is the `commit-msg` hook, checked when you actually
run `git commit` or `just commit`.
```

Individual recipes are available when you want to run one step:

| Command       | Action                                          |
| ------------- | ----------------------------------------------- |
| `just lint`   | Lint and auto-fix with ruff.                    |
| `just format` | Run the ruff formatter.                         |
| `just types`  | Run the pyright type checker.                   |
| `just test`   | Run the test suite with coverage.               |
| `just docs`   | Build the HTML documentation.                   |
| `just check`  | Run all hooks across all files (commit + push). |

## Testing

PreTab has a comprehensive test suite that gates every change.

### Running the tests

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

### Layout

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

### Testing mathematical correctness, not just shape

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

### Markers

The suite defines a `smoke` marker for fast end-to-end sanity checks that run as a dedicated CI
gate.

```bash
poetry run pytest -m smoke        # only the smoke checks
poetry run pytest -m "not smoke"  # everything else
```

### Coverage

`just test` measures coverage over the `pretab` package. Keep new code covered, and prefer a
focused test that exercises the behaviour over one that merely touches lines.

```bash
poetry run pytest --cov=pretab --cov-report=term-missing tests/
```

### Testing a custom representation

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

### Before you push

Run `just check` and `just test` locally; together they cover most of what CI checks, though
CI additionally runs across the full Python 3.10-3.13 matrix, builds the package, and
enforces a branch-coverage threshold.

```bash
just test     # tests with coverage
just check    # lint, format, type-check across all files
just docs     # strict docs build
just quickstart  # end-to-end sanity check: same script CI's smoke job runs
```

## Documentation

The documentation is part of the codebase and is held to the same standard as the code. It is
built with [Sphinx](https://www.sphinx-doc.org/) and hosted on
[Read the Docs](https://about.readthedocs.com/).

### Building the docs

```bash
just docs                           # build HTML into docs/_build/html
open docs/_build/html/index.html    # macOS; use xdg-open on Linux
```

```{important}
The build runs with `-W`, so **warnings are treated as errors**. A broken cross-reference, an
orphaned page, or a malformed directive fails the build. Run `just docs` before opening a pull
request that touches documentation.
```

To work on the docs, install the docs dependency group.

```bash
poetry install --with docs
```

### Structure

The `docs/` tree is organized by reader intent.

| Section            | Purpose                                                                                                                  |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `getting_started/` | Install, first model, migration.                                                                                         |
| `core_concepts/`   | The mental model: representation, configuration, resolution, target awareness, missing values, outputs, reproducibility. |
| `representations/` | The method catalogue, comparison table, and selection guidance.                                                          |
| `tutorials/`       | Task-oriented, worked examples.                                                                                          |
| `api/`             | Autogenerated reference from docstrings.                                                                                 |
| `developer_guide/` | Contributing (setup, testing, documentation, versioning) and the release process.                                        |

### MyST Markdown and reStructuredText

Prose pages are written in [MyST Markdown](https://myst-parser.readthedocs.io/) (`.md`); the
API pages are reStructuredText (`.rst`) so they can drive `autosummary`. Use callout directives
to highlight important information.

````markdown
```{note}
A neutral aside.
```

```{tip}
A helpful suggestion.
```

```{warning}
Something that can bite the reader.
```

```{important}
A guarantee or constraint the reader must not miss.
```
````

Math uses standard MyST syntax, inline as `$...$` and display as `$$...$$`.

### Adding a page

Every page must be reachable from a `toctree`, or the strict build fails with an orphan-document
error.

1. Create the `.md` file in the appropriate section.
2. Add its filename (without extension) to the relevant `toctree`, either in `index.rst` or the
   section's own index.
3. Cross-link to and from sibling pages with relative links.
4. Run `just docs` and fix any warnings.

```{warning}
A cross-reference to a page that does not exist fails the strict build. When you link to a page,
make sure the target exists, and when you remove a page, remove every link to it.
```

### The API reference

The API pages document public classes and functions from their numpy-style docstrings through
`autodoc` and `autosummary`. There is no prose to write for a new public class; instead, add its
name to the appropriate `autosummary` block under `docs/api/` and keep its docstring accurate.

```{note}
Because the reference is generated from docstrings, an accurate docstring is documentation.
Update the docstring in the same change that alters the behaviour.
```

### Writing style

The documentation aims to be precise and natural, and to read well for beginners, practitioners,
and researchers alike. A few conventions keep it consistent.

- Separate sections with headings, not horizontal rules.
- Avoid stray transitional text between sections; let the headings carry the structure.
- Prefer active, concrete sentences over filler.
- Ground every claim in the real API. If you are unsure of a parameter name or default, check
  the source.
- Add a callout where it genuinely helps, not on every paragraph.

## Versioning

pretab follows [Semantic Versioning 2.0](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/) to automate version bumps and
changelog generation via [commitizen](https://commitizen-tools.github.io/commitizen/).

From `1.0.0` onward, `feat!:` and `BREAKING CHANGE:` commits bump the major version, following
standard SemVer.

### Version format

```
MAJOR.MINOR.PATCH
```

| Segment | When it increments                                                         |
| ------- | -------------------------------------------------------------------------- |
| `MAJOR` | Breaking change (`feat!:` or `BREAKING CHANGE:` footer)                    |
| `MINOR` | New backwards-compatible feature (`feat:`)                                 |
| `PATCH` | Backwards-compatible bug fix (`fix:`) or performance improvement (`perf:`) |

Release candidates use the suffix `rcN`, e.g. `1.0.0rc1`.

The version is defined **in one place only**, `pyproject.toml`, and read at runtime via
`importlib.metadata` in `pretab/_version.py`, so it never needs to be hard-coded in the
package.

```{note}
`major_version_zero` is `false` in the commitizen config, so `feat!:` / `BREAKING CHANGE:`
commits bump the **major** version, in line with standard SemVer.
```

### Commit types and their effect

| Commit type | Example                                    | Version bump |
| ----------- | ------------------------------------------ | ------------ |
| `feat`      | `feat(splines): add B-spline knots option` | Minor        |
| `fix`       | `fix(binning): handle empty bins`          | Patch        |
| `perf`      | `perf(ple): vectorise bin assignment`      | Patch        |
| `feat!`     | `feat!: drop Python 3.9 support`           | Major        |
| `docs`      | `docs: update API reference`               | None         |
| `test`      | `test: add spline round-trip test`         | None         |
| `ci`        | `ci: add Python 3.13 to matrix`            | None         |
| `refactor`  | `refactor: simplify feature detection`     | None         |
| `style`     | `style: apply ruff formatting`             | None         |
| `chore`     | `chore: update pre-commit revisions`       | None         |

Commit messages that do not match any of these types do not trigger a version bump. See
[CONVENTIONAL_COMMITS.md](https://github.com/OpenTabular/PreTab/blob/main/CONVENTIONAL_COMMITS.md)
for the full list of pretab scopes.

### Making a conventional commit

Use commitizen's interactive prompt rather than writing the message by hand:

```bash
just commit      # opens the cz commit wizard
```

Or write the message directly:

```bash
git commit -m "feat(feature-maps): add Gaussian RBF centers"
git commit -m "fix(preprocessor): validate output_dim > 0"
```

The `commit-msg` pre-commit hook validates every commit message against the conventional
commits format and rejects non-conforming messages.

### Bumping the version

Version bumps are driven by commitizen, wrapped in `just` recipes. Preview first with the
`-preview` (dry-run) variant, then apply. Each apply recipe updates `version` in
`pyproject.toml`, appends to `CHANGELOG.md`, and creates the bump commit and tag.

| Goal              | Preview                | Apply          |
| ----------------- | ---------------------- | -------------- |
| Stable release    | `just bump-preview`    | `just bump`    |
| Release candidate | `just bump-rc-preview` | `just bump-rc` |

The next version is inferred from the conventional commits since the last tag. To force a
level when it is not auto-detected, append the increment, e.g. `just bump --increment MINOR`.

### Changelog

`CHANGELOG.md` at the repository root is the authoritative changelog, updated automatically
by the bump recipes. Changes are grouped under their commit types (`feat`, `fix`,
`perf`, ...) with the subject line of every matching commit since the previous release.

### Tags

Release tags follow `vMAJOR.MINOR.PATCH` (or `vMAJOR.MINOR.PATCHrcN` for RCs) and trigger
the PyPI publish workflows. See [Release process](release.md) for the full end-to-end
procedure.

## Release workflow

For the end-to-end release procedure (version bump, tags, PyPI publishing), see
**[Release process](release.md)**.

## Issue tracker

Report bugs, request features, or ask for help on the
[Issue Tracker](https://github.com/OpenTabular/PreTab/issues). Search existing issues
before opening a new one.

## License

By contributing, you agree that your contributions are licensed under the project
[LICENSE](https://github.com/OpenTabular/PreTab/blob/main/LICENSE).
