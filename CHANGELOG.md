# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

## v1.0.0 (2026-09-06)

First stable release, consolidating the release-candidate cycle (`rc1`-`rc5`) below into one
entry. `1.0.0` is a deliberate, one-time breaking cleanup of the previously published (never
declared stable) `0.0.x` API: intention-revealing class names, non-overlapping parameters, and
a smaller, sharper scope. See the
[migration guide](https://pretab.readthedocs.io/en/latest/getting_started/migration_to_1_0.html)
for the full old-to-new mapping.

Highlights:

- Package restructured into `pretab.expansion`, `pretab.encoding`, `pretab.embedding`,
  `pretab.kernel_approximation`, `pretab.preprocessing`, and `pretab.compose`, with a
  capability registry driving `Preprocessor`.
- New representations: Fourier, random Fourier, and Nystroem feature maps, plus rewritten
  binning, periodic encoding, and thin-plate spline transformers.
- A typed `RepresentationSpec` and per-output-column feature lineage for every representation.
- Leakage-safe supervision: `CrossFittedTransformer`, `RepresentationSearchCV`, and a
  `LeakageWarning`.
- Portable serialization: `to_spec`/`from_spec`, a stable `fingerprint_`, and a frozen fitted
  lifecycle.
- Explicit `missing_policy` replacing the old `handle_missing` flag.
- `Preprocessor(preset=...)` (`"standard"`, `"expanded"`, `"adaptive"`) and a
  `list_representations(...)` discovery API.
- A central `RepresentationPolicy` for edge cases, plus output budgets and sparse/DataFrame
  output.
- A public extension protocol: subclass `BaseRepresentation`, validate with
  `check_representation`, and register with `register_representation`.

### Feat

- PreTab 1.0.0 restructuring
- PreTab 1.0.0 restructure and refactoring
- add quickstart script as a ci smoke test and reviewer artifact
- **extension**: add representation protocol, discovery, and presets
- **serialize**: add to_spec/from_spec, fingerprint, and frozen lifecycle
- **missing**: add missing_policy and edge-case tests
- **output**: add output budgets and sparse/dataframe output
- **policy**: add RepresentationPolicy and edge-case contract
- **supervised**: add leakage-safe contract, cross-fitting, and representation search
- **representation**: add RepresentationSpec and feature lineage
- **transformers**: add Fourier, random-Fourier, and Nystroem feature maps
- **transformers**: rewrite binning, periodic encoding, and thin-plate spline
- **params**: replace handle_missing with imputation params
- prep 1.0.0 release
- restructure package toward 1.0.0 release
- update default output_dim
- unsupervised feature-map default
- wire custombin output_dim
- unify knot / bins API
- **pipeline**: make cubic and natural splines target-aware
- **pipeline**: use selector and adaptive setting to splines
- **pipeline**: accept preprocessing method name variations
- **preprocessor**: expose total*output_dim*, output*dims* attribute
- **preprocessor**: add random_state, handle_missing parameters
- **sklearn-compat**: enforce n_features consistency, fix mixin order/tags
- **exceptions**: route all raises through typed exceptions
- **logging**: add verbose level, route warnings
- **adaptive**: unify adaptive/fixed output size across expansion families via AdaptiveResolutionMixin
- **preprocessor**: rename constructor params to the canonical vocabulary and add adaptive parameter
- **pipeline**: thread output_dim through registry and Preprocessor
- **ple,binning**: rename count knob to output*dim and set total_output_dim*
- **feature_maps**: rename count knob to output_dim
- **splines**: invert output*dim to knots per family and expose n_knots*
- **core**: make output*dim the canonical width param and add total_output_dim*
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
- default Preprocessor output to a single array

### Fix

- make present task dependent
- fit summary to report resolved methods
- **types**: silence optional lightgbm import and narrow array_equal args
- **tests**: narrow transform return type for pyright csr_matrix ufunc check
- formatting
- linting and formatting
- **embeddings**: add get_feature_names_out to LanguageEmbeddingTransformer
- **preprocessor**: collapse duplicated feature name in output column names
- **docs**: define missing dataset in two tutorial snippets
- **types**: resolve remaining type errors across pretab
- **types**: annotate cloned estimators in cross-fitting and search
- **embeddings**: encode each text column separately
- **categorical**: ignore unknown categories in one-hot
- **onehot**: handle out-of-range codes
- **feature_maps**: use a numerically stable sigmoid to avoid overflow
- **pipeline**: make optional keyword arguments explicitly optional
- **transformers**: generate default feature names when none are given
- **transformers**: report the actual number of input features
- **embeddings**: resolve language model in fit
- missclassify numerical to cat feature based on name
- enfore feature count for cat transformer
- unassigned int cat cuttoff
- missing policy feature inspection, embedding silient failure
- sparse to dense conversion for sparse output
- custom representation usage for cat features
- supplied parameter override preset
- appropriate error message for out_dim and min_out_dim (#38)
- raise error for duplicate columns (#37)
- follow sklearn contract for binning (#36)
- reject unrecognized scalar (#35)
- validate embedding against fitted dimension (#34)
- set include_bias to False as default (#33)
- **embeddings**: accept list input in LanguageEmbeddingTransformer.fit (issue #21)
- **core**: raise on mismatched input_features length in get_feature_names_out (issue #21)
- **locations**: keep importance aligned with locations after sort/dedupe (issue #21)
- keep location provided by tree when suplementing
- provide appropriate error for onehot_from_ordinal input
- **splines,transformers**: close B-spline final span and fix ContinuousOrdinalTransformer DataFrame input (issues #12, #14)
- **selectors**: make \_enforce_spacing order-independent to fix lightgbm clustering (issue #10)
- **tests**: sort imports in test_adaptive_output_dim
- add missing indicator
- validate embedding during fit
- remove unused ple parameters
- raise scikit-learn minimum and use scipy trapezoid
- validate cross-fitting task name
- reuse identical CV folds across search candidates
- correct serialization safety claims and bypass **new** hooks
- reject fit on a frozen preprocessor
- clip P-spline and tensor-product out-of-range transforms via policy
- document periodic transform wrap-around and correct binning docstring
- add missing polars optional dependency and its tests
- numerical_method=none no longer applies scaling
- validate feature_preprocessing keys against input columns
- remove unimplemented resolution stubs from public placement API
- use configured dtype in output-budget memory estimate
- drop unwired policy fields and encoding/embedding NaN bugs
- **spline**: correct penalty matrices and validate diff_order
- **ple**: remove boundary-bin discontinuity in PLETransformer
- **serialize**: reject unsafe classes in from_spec

### Perf

- **splines**: drop retained training design matrices (issue #19)
- **preprocessor**: slice dict blocks from output*indices* instead of re-transforming (issue #20)

### Refactor

- **core**: rename typing module to \_typing and add estimator protocols
- **transformers**: move Fourier and kernel-approximation maps into feature_maps/
- **transformers**: rename core transformers, drop temporal utils
- **compose**: add capability registry and slim preprocessor
- **placement**: centralize location placement and migrate transformers to it
- **layout**: restructure package toward 1.0 and drop compat shims
- consistent param order
- remove dead selection helpers
- **ple**: use location selectors for thresholds
- add shared resolve_locations helper
- **feature_maps**: move strategy/task validation from `__init__` to fit
- **transformers**: drop utils, move BaseCenterExpansion ti feature_maps
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
- **preprocessing**: move floats and missing encoders to pretab.preprocessing
- **embedding**: move language embedding to pretab.embedding
- **kernel-approximation**: split kernel approximations into pretab.kernel_approximation
- **encoding**: move categorical encoders to pretab.encoding.categorical
- **encoding**: move numerical encoders to pretab.encoding.numerical
- **expansion**: move functional expansions to pretab.expansion.functional
- **expansion**: move spline transformers to pretab.expansion.spline

## v0.0.2 (2025-04-13)

## v0.0.1 (2025-04-12)
