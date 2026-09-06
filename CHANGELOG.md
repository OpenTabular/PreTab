# CHANGELOG

All notable changes to this project will be documented in this file.

This project adheres to [Semantic Versioning](https://semver.org/) and uses
[Conventional Commits](https://www.conventionalcommits.org/).

Going forward, this file is updated automatically by `cz bump` on each release.

## v1.0.0 (2026-09-06)

### Fix

- make present task dependent
- fit summary to report resolved methods

## v1.0.0rc5 (2026-09-06)

## v1.0.0rc4 (2026-09-06)

### Feat

- default Preprocessor output to a single array

### Fix

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
- raise scikit-learn minimum and use scipy trapezoid
- use configured dtype in output-budget memory estimate
- drop unwired policy fields and encoding/embedding NaN bugs
- **spline**: correct penalty matrices and validate diff_order
- **ple**: remove boundary-bin discontinuity in PLETransformer
- **serialize**: reject unsafe classes in from_spec

## v1.0.0rc3 (2026-09-01)

### Fix

- **tests**: sort imports in test_adaptive_output_dim
- add missing indicator
- validate embedding during fit
- remove unused ple parameters

### Refactor

- **preprocessing**: move floats and missing encoders to pretab.preprocessing
- **embedding**: move language embedding to pretab.embedding
- **kernel-approximation**: split kernel approximations into pretab.kernel_approximation
- **encoding**: move categorical encoders to pretab.encoding.categorical
- **encoding**: move numerical encoders to pretab.encoding.numerical
- **expansion**: move functional expansions to pretab.expansion.functional
- **expansion**: move spline transformers to pretab.expansion.spline

## v1.0.0rc2 (2026-08-21)

### Fix

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

### Perf

- **splines**: drop retained training design matrices (issue #19)
- **preprocessor**: slice dict blocks from output*indices* instead of re-transforming (issue #20)

## v1.0.0rc1 (2026-08-15)

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

### Fix

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

## v0.0.2 (2025-04-13)

## v0.0.1 (2025-04-12)
