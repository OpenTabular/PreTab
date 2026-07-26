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
| `feat`     | New feature             | Minor (0.x.0) |
| `fix`      | Bug fix                 | Patch (0.0.x) |
| `docs`     | Documentation only      | None          |
| `style`    | Code style/formatting   | None          |
| `refactor` | Code refactoring        | None          |
| `perf`     | Performance improvement | Patch         |
| `test`     | Adding/updating tests   | None          |
| `build`    | Build system changes    | None          |
| `ci`       | CI/CD changes           | None          |
| `chore`    | Other changes           | None          |

> While the project is in the `0.x` series (`major_version_zero = true`), a
> breaking change bumps the **minor** version rather than the major version.

## Scopes (Optional)

Common scopes in this project:

- `preprocessor`: The top-level `Preprocessor`
- `transformers`: Transformer implementations
- `binning`: Custom / tree-based binning
- `splines`: Spline expansions (cubic, natural cubic, P-spline, tensor product, thin plate)
- `feature-maps`: Neural basis expansions (RBF, ReLU, sigmoid, tanh)
- `embeddings`: Language / pretrained embeddings
- `ple`: Piecewise linear encoding
- `onehot`: One-hot encoders
- `temporal`: Cyclic, lag, and rolling-stats transformers
- `utils`: General utilities
- `ci`: CI/CD related
- `deps`: Dependencies

## Examples

```bash
# Feature (minor bump: 0.1.0 → 0.2.0)
git commit -m "feat(splines): add B-spline basis transformer"

# Bug fix (patch bump: 0.1.0 → 0.1.1)
git commit -m "fix(preprocessor): handle all-NaN numerical columns"

# Performance (patch bump)
git commit -m "perf(feature-maps): vectorize RBF distance computation"

# Documentation (no bump)
git commit -m "docs: document custom binning strategies"

# Breaking change (minor bump while in 0.x)
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
feat(temporal): add rolling-window statistics transformer

Adds a transformer that computes rolling mean/std/min/max over a configurable
window for time-ordered features.

Closes #123
```

## Pre-commit Hook

Commit messages are validated automatically by the Commitizen `commit-msg`
hook. If a message is rejected:

1. Check the format: `type(scope): description`
2. Use an allowed type only
3. Keep the header under 72 characters
4. Don't end the subject with a period
