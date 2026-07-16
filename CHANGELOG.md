# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

---

## Unreleased

### Changes

- Migrated packaging from setuptools to Poetry (`pyproject.toml`, `poetry.lock`); removed `setup.py`, `requirements.txt`, and `MANIFEST.in`
- Dynamic versioning: the version is now sourced from `pyproject.toml` via `importlib.metadata`; removed the hardcoded `__version__.py`
- Adopted a Poetry + OIDC release pipeline publishing to PyPI (`v*.*.*`) and TestPyPI (`v*.*.*rc*`), plus a manual `build-check` dry-run workflow
- Added a `justfile` and pre-commit configuration for the local development workflow
- Added project meta documentation: `CHANGELOG.md`, `CONVENTIONAL_COMMITS.md`, and `CODE_OF_CONDUCT.md`
