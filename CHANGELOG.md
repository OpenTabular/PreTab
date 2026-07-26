# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

---

## Unreleased

> **Note:** `0.1.0` is an internal development marker for the pre-1.0 restructure
> line and **will not be published**. The next released version is **1.0.0** (a
> deliberate major release due to the package restructure and breaking API
> changes; `major_version_zero` will be flipped to `false` at that point). The
> entries below are curated from the conventional commits since `v0.0.2` and are
> the reconciliation source for the 1.0.0 roadmap.

### Feat

- **representation**: add typed `RepresentationSpec` and per-output-column `FeatureLineage` (exported from `pretab`); every transformer family exposes `get_representation_spec()` and `Preprocessor.get_feature_lineage()` maps each output column to its source feature(s), representation family, component, and target-usage flag
- **transformers**: add `FourierFeatureTransformer` (deterministic sine/cosine feature map with `harmonic` / `log_spaced` / `random` frequencies), selectable as the `"fourier"` numerical method
- **transformers**: add `RandomFourierFeaturesTransformer` and `NystroemFeaturesTransformer` — standalone multivariate kernel-approximation feature maps (`"rff"` / `"nystroem"`)
- **binning**: make `NumericBinningTransformer` a stateful, multi-feature encoder with learned `bin_edges_` and `encode` (`ordinal` / `onehot` / `soft`) plus `placement_strategy` (`uniform` / `quantile`) options
- **transformers**: add `harmonics` and `include_original` options to `PeriodicEncodingTransformer` for multi-harmonic periodic encodings
- update default output_dim
- unsupervised feature-map default
- wire custombin output_dim
- unify knot / bins API
- **pipeline**: make cubic and natural splines target-aware
- **pipeline**: use selector and adaptive setting to splines
- **pipeline**: accept preprocessing method name variations
- **preprocessor**: expose total_output_dim_, output_dims_ attribute
- **preprocessor**: add random_state parameter
- **preprocessor**: add numerical_imputation / categorical_imputation / add_missing_indicator parameters (replacing handle_missing)
- **sklearn-compat**: enforce n_features consistency, fix mixin order/tags
- **exceptions**: route all raises through typed exceptions
- **logging**: add verbose level, route warnings
- **adaptive**: unify adaptive/fixed output size across expansion families via AdaptiveResolutionMixin
- **preprocessor**: rename constructor params to the canonical vocabulary and add adaptive parameter
- **pipeline**: thread output_dim through registry and Preprocessor
- **ple,binning**: rename count knob to output_dim and set total_output_dim_
- **feature_maps**: rename count knob to output_dim
- **splines**: invert output_dim to knots per family and expose n_knots_
- **core**: make output_dim the canonical width param and add total_output_dim_
- add include_bias and feature_index parity to thin-plate spline
- add strategy/selector/task/include_bias parity to knot splines
- add selector-aware spanning-knot placement to spline mixin
- add spanning_knots primitive for full-range spline knots
- accept n_basis alias in custom binning transformer
- accept n_basis and use_target aliases in feature maps and ple
- accept n_basis alias across spline transformers
- add core params AliasResolverMixin and canonical vocabulary
- add pretab.pipeline package with declarative registry
- add transformers/encoders package
- add core knot-placement primitives
- add core center identification and BaseCenterExpansion
- add core foundation with base, validation, logging, exceptions
- export Preprocessor at package root
- add bspline, mspline, ispline transformers with shared spline base
- add target-aware knot selectors

### Fix

- **embeddings**: encode each text column separately
- **categorical**: ignore unknown categories in one-hot
- **onehot**: handle out-of-range codes
- **feature_maps**: use a numerically stable sigmoid to avoid overflow
- **pipeline**: make optional keyword arguments explicitly optional
- **transformers**: generate default feature names when none are given
- **transformers**: report the actual number of input features
- **embeddings**: resolve language model in fit

### Refactor

- **splines**: reformulate `ThinPlateSplineTransformer` as a multivariate low-rank thin-plate regression spline (landmark selection + eigen/Nyström basis via `n_components` / `landmark_strategy` / `rank_strategy`, replacing the univariate `output_dim` form)
- **transformers**: rename `CustomBinTransformer` → `NumericBinningTransformer`, `CyclicalTimeTransformer` → `PeriodicEncodingTransformer`, and `CubicSplineTransformer` → `CubicRegressionSplineTransformer` (intention-revealing public names)
- **transformers**: remove `LagFeatureTransformer` and `RollingStatsTransformer` (row-count-changing time-series utilities outside the tabular scope)
- **splines**: restrict `PSplineTransformer` to `placement_strategy="uniform"` (penalized splines require equally-spaced knots)
- **compose**: exclude the multivariate `tensorspline` / `tprs` methods from the per-column `Preprocessor` whitelist (they remain available as standalone transformers)
- **categorical**: deprecate `OneHotFromOrdinalTransformer` (use the `"one-hot"` categorical method backed by scikit-learn's `OneHotEncoder`)
- consistent param order
- remove dead selection helpers
- **ple**: use location selectors for thresholds
- add shared resolve_locations helper
- **feature_maps**: move strategy/task validation from __init__ to fit
- **transformers**: drop utils, move BaseCenterExpansion to feature_maps
- **preprocessor**: make it a compliant sklearn estimator
- canonicalize pipeline kwargs to avoid deprecation warnings
- point preprocessor at pretab.pipeline
- turn pretab.utils into pipeline shims
- export encoders and repoint internal imports
- turn transformers/utils encoders into shims
- add sklearn compliance to one-hot transformer
- put lag and rolling-stats transformers on core.base
- put cyclic transformer on core.base
- delegate spline knot math to core.knots
- reduce feature-map transformers to kernel hooks
- retarget spline transformers onto core base
- vectorize ple encoding and remove eval
- source package version dynamically from metadata

### Build & Tooling

- Migrated packaging from setuptools to Poetry (`pyproject.toml`, `poetry.lock`); removed `setup.py`, `requirements.txt`, and `MANIFEST.in`
- Dynamic versioning: the version is now sourced from `pyproject.toml` via `importlib.metadata`; removed the hardcoded `__version__.py`
- Adopted a Poetry + OIDC release pipeline publishing to PyPI (`v*.*.*`) and TestPyPI (`v*.*.*rc*`), plus a manual `build-check` dry-run workflow
- Added a `justfile` and pre-commit configuration for the local development workflow
- Added project meta documentation: `CHANGELOG.md`, `CONVENTIONAL_COMMITS.md`, and `CODE_OF_CONDUCT.md`
