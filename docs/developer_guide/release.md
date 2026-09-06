# Release process

pretab is published to [PyPI](https://pypi.org/project/pretab/) with a release-candidate
step on [TestPyPI](https://test.pypi.org/project/pretab/) first. The whole flow is driven
by [commitizen](https://commitizen-tools.github.io/commitizen/) version bumps and Git tags;
publishing itself runs on GitHub Actions using
[PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC), so no API
tokens are stored anywhere.

For the SemVer rules and commit conventions that decide the next version, see
[Versioning](contributing.md#versioning).

## Overview

| Stage             | Trigger                      | Workflow               | Target   |
| ----------------- | ---------------------------- | ---------------------- | -------- |
| Build check       | `workflow_dispatch` (manual) | `build-check.yml`      | Artifact |
| Release candidate | Tag `vX.Y.ZrcN`              | `publish-testpypi.yml` | TestPyPI |
| Stable release    | Tag `vX.Y.Z`                 | `publish-pypi.yml`     | PyPI     |

## What runs when

`ci.yml` and `docs.yml` trigger on both `main` and `release/**` branches, so a commit
pushed straight to a release branch (step 4 below) gets the same feedback as a commit on
`main`, without waiting for the next RC tag.

| Event                |   `ci.yml`    |      `docs.yml`      | Publish workflow       |
| -------------------- | :-----------: | :------------------: | ---------------------- |
| PR into `main`       |      ✅       | ✅ (docs paths only) | -                      |
| Push to `main`       |      ✅       |          ✅          | -                      |
| Push to `release/**` |      ✅       |          ✅          | -                      |
| Push tag `vX.Y.ZrcN` | ✅ (via `qa`) |   ✅ (via `docs`)    | `publish-testpypi.yml` |
| Push tag `vX.Y.Z`    | ✅ (via `qa`) |   ✅ (via `docs`)    | `publish-pypi.yml`     |

Every publish workflow re-runs the full `ci.yml`/`docs.yml` gate on the exact tagged
commit before anything is uploaded, rather than trusting whatever last passed on `main`.

## QA gate checks

| Check                                                | Job in `ci.yml`                |
| ---------------------------------------------------- | ------------------------------ |
| Lint (`ruff check`, `ruff format --check`)           | `lint`                         |
| Type checking (`pyright`)                            | `typecheck`                    |
| Package build + `twine check`                        | `build`                        |
| Full test matrix (3.10-3.13, 3 OSes)                 | `tests`                        |
| Minimum supported dependency versions                | `minimum-deps`                 |
| Smoke test + quickstart script                       | `smoke`                        |
| Coverage (fails under 90%)                           | `coverage`                     |
| Optional extras (`embeddings`, `lightgbm`, `polars`) | `optional-deps`                |
| Docs build (`sphinx-build -W --keep-going`)          | `docs.yml` (separate workflow) |
| Wheel install + import smoke test                    | publish workflow's own steps   |

## Prerequisites

- You are a maintainer with permission to push tags to `main`.
- The `pypi-publish` and `testpypi-publish` GitHub Environments are configured with
  tag-based protection and trusted publishing enabled.
- Your working tree is clean and `main` is up to date.

## Step-by-step

### 1. Prepare a release branch

```bash
git checkout main
git pull
git checkout -b release/vX.Y.Z
```

### 2. Run the full QA suite

```bash
just check    # lint, format, type-check
just test     # tests with coverage
just docs     # docs build (warnings as errors)
```

### 3. Cut a release candidate

Preview the version commitizen infers from the commits since the last tag, then apply it:

```bash
just bump-rc-preview     # dry run to inspect the proposed version and changelog
just bump-rc             # applies the bump: updates pyproject.toml + CHANGELOG.md, commits, tags vX.Y.ZrcN
```

Push the branch and the RC tag:

```bash
git push origin release/vX.Y.Z
git push origin vX.Y.ZrcN
```

The `publish-testpypi.yml` workflow publishes the candidate to TestPyPI and opens a
GitHub pre-release.

### 4. Verify the release candidate

Install the candidate from TestPyPI in a clean environment and smoke-test it:

```bash
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "pretab==X.Y.ZrcN"
python -c "import pretab; print(pretab.__version__)"
```

If the candidate has problems, fix them on the branch and produce the next RC:

```bash
just bump-rc-preview     # confirm the proposed version (e.g. 1.0.0rc2)
just bump-rc             # apply the bump: pyproject.toml, CHANGELOG.md, commit, tag
git push origin release/vX.Y.Z
git push origin vX.Y.ZrcN
```

Repeat for each additional RC (`rc3`, `rc4`, ...) until the build is clean.

### 5. Merge to main

Open a pull request from `release/vX.Y.Z` into `main` and merge it once approved and green.

### 6. Cut the stable release

From an up-to-date `main`:

```bash
git checkout main
git pull
just bump-preview        # dry run
just bump                # applies the bump: commits and tags vX.Y.Z
git push origin main
git push origin vX.Y.Z
```

The `publish-pypi.yml` workflow builds the package, verifies the tag matches the project
version, publishes to PyPI, and creates the GitHub Release.

See [What runs when](#what-runs-when) and [QA gate checks](#qa-gate-checks) above for
exactly what both publishing workflows wait on before uploading.

### 7. Confirm

```bash
pip install --upgrade pretab
python -c "import pretab; print(pretab.__version__)"
```

Check the release on [PyPI](https://pypi.org/project/pretab/) and the auto-generated
GitHub Release notes.

```{tip}
Need a build artifact without publishing? Run the `build-check.yml` workflow manually from
the GitHub Actions tab. It builds the wheel and sdist, runs `twine check`, and performs an
import smoke test.
```
