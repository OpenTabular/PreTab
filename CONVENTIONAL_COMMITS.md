# Conventional Commits Quick Reference

`pretab` uses [Conventional Commits](https://www.conventionalcommits.org/) with
[Commitizen](https://commitizen-tools.github.io/commitizen/) to drive automated
version bumps and `CHANGELOG.md` generation.

## Commit Format

```
<type>(<scope>): <subject>
```

## Types

| Type       | Description             | Version Bump  |
| ---------- | ----------------------- | ------------- |
| `feat`     | New feature             | Minor (1.x.0) |
| `fix`      | Bug fix                 | Patch (1.0.x) |
| `docs`     | Documentation only      | None          |
| `style`    | Code style/formatting   | None          |
| `refactor` | Code refactoring        | None          |
| `perf`     | Performance improvement | Patch         |
| `test`     | Adding/updating tests   | None          |
| `build`    | Build system changes    | None          |
| `ci`       | CI/CD changes           | None          |
| `chore`    | Other changes           | None          |

> From `1.0.0` onward (`major_version_zero = false` in `pyproject.toml`), a
> breaking change (`feat!:` or a `BREAKING CHANGE:` footer) bumps the **major**
> version, following standard SemVer.

## Scopes (Optional)

Common scopes in this project:

- `preprocessor`: The top-level `Preprocessor`
- `transformers`: Transformer implementations
- `binning`: Numeric binning
- `splines`: Spline expansions (B/M/I, cubic, natural cubic, P-spline, tensor product, thin plate)
- `feature-maps`: Neural basis expansions (RBF, ReLU, sigmoid, tanh) and Fourier features
- `embeddings`: Language / pretrained embeddings
- `ple`: Piecewise linear encoding
- `onehot`: One-hot encoders
- `placement`: Knot / center / bin-edge placement (uniform, quantile, CART, LightGBM)
- `cross-fitting`: `CrossFittedTransformer` and leakage-safe fitting
- `search`: `RepresentationSearchCV`
- `serialization`: `to_spec` / `from_spec` and fingerprints
- `logging`: Verbosity and the `"pretab"` logger
- `ci`: CI/CD related
- `deps`: Dependencies

## Examples

```bash
# Feature (minor bump: 1.0.0 → 1.1.0)
git commit -m "feat(splines): add B-spline basis transformer"

# Bug fix (patch bump: 1.0.0 → 1.0.1)
git commit -m "fix(preprocessor): handle all-NaN numerical columns"

# Performance (patch bump)
git commit -m "perf(feature-maps): vectorize RBF distance computation"

# Documentation (no bump)
git commit -m "docs: document custom binning strategies"

# Breaking change (major bump: 1.0.0 → 2.0.0)
git commit -m "feat!: rename fit_transform embeddings argument

BREAKING CHANGE: embeddings must now be passed as a keyword argument."
```

## Breaking Changes

Use `!` after the type and explain the change in the footer:

```
feat!: change transform() return type

BREAKING CHANGE: transform() now returns a dict of arrays keyed by feature
name instead of a single concatenated array. Pass return_array=True for the
previous behaviour.
```

## Quick Commands

```bash
# Interactive, guided commit (recommended)
just commit

# Preview the next stable version bump
just bump-preview

# Apply the version bump (updates version, CHANGELOG, commit, and tag)
just bump

# Release-candidate bump (rcN)
just bump-rc-preview
just bump-rc

# View the changelog
cat CHANGELOG.md
```

## Multi-line Commits

```bash
# Opens your editor
git commit

# In the editor:
feat(feature-maps): add a log-spaced frequency strategy

Adds frequency_strategy="log_spaced" to FourierFeatureTransformer, spreading
the frequencies in octaves above the fundamental.

Closes #123
```

## Pre-commit Hook

Commit messages are validated automatically by the Commitizen `commit-msg`
hook. If a message is rejected:

1. Check the format: `type(scope): description`
2. Use an allowed type only
3. Keep the header under 72 characters
4. Don't end the subject with a period
