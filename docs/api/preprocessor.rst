Preprocessor
============

The high-level entry point. :class:`~pretab.Preprocessor` reads a ``DataFrame``,
detects feature types, resolves a per-column representation from a single
configuration, and produces model-ready output with full lineage.

.. currentmodule:: pretab

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Preprocessor

Representation values
---------------------

:class:`~pretab.RepresentationConfig` stores a representation request with
read-only, detached parameters and JSON-compatible dictionary conversion.
Constructing the value does not fit an estimator or resolve method aliases.
Pass it as a value in ``Preprocessor(feature_preprocessing={...})``; strings
and mappings with a ``method`` key are accepted there as well.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentationConfig

Fields and conversion
~~~~~~~~~~~~~~~~~~~~~

``method``
    Required method name or alias. Validation against the selected method
    happens during configuration resolution or fitting.

``output_dim``
    Positive requested width, subject to the representation's constraints.
    An explicit local width suppresses inherited adaptive sizing unless
    local ``adaptive=True`` also opts in.

``target_aware``, ``placement_strategy``
    Local target-use and placement controls, where the method supports them.

``adaptive``, ``min_output_dim``, ``max_output_dim``
    Local adaptive sizing and bounds. Bounds must be positive integers or
    ``None`` where accepted. Explicit bounds with fixed sizing are invalid.

``params``
    Family-specific constructor options, such as ``{"gamma": 0.5}`` for
    RBF. Common fields belong at the top level, not in this mapping.
    One-hot entries accept encoder options such as ``categories``, ``drop``,
    ``handle_unknown``, and ``sparse_output``. For example,
    ``{"method": "one-hot", "params": {"drop": "first", "sparse_output": False}}``
    drops one category and requests dense encoding. Encoder ``dtype`` values
    must be portable, such as ``"float32"``, rather than NumPy scalar classes.
    Use the preprocessor's ``dtype`` to control the final output dtype;
    encoder settings do not bypass subsequent output conversion.

``name``
    Optional nonempty metadata label, default ``None``. It does not rename
    input columns or fitted output features.

Omitted common controls retain ``UNSET`` until resolution. An explicit local
value takes precedence over an applicable constructor-wide value; ``None``
is an explicit value only for fields that accept it, not an inheritance marker.
Imputation, scaling, missing indicators, and output formatting remain run-wide.

Nested ``params`` values may contain strings, booleans, integers, finite floats,
``None``, sequences, and mappings with string keys. Construction copies them
into read-only storage. Callables, arbitrary objects, sets, nonfinite floats,
and circular references are rejected.

``to_dict()`` returns detached dict/list containers, omits ``UNSET`` fields,
and retains explicit ``None`` values. It preserves the requested method
spelling; it does not resolve an alias. The result can be passed through
``json.dumps`` / ``json.loads`` and reconstructed with ``from_dict()``.
``from_dict()`` also accepts family parameters beside ``method`` and collects
them under ``params``. Supplying the same option in both places is an error,
even when the values agree.

.. note::

    This is a value object, not a scikit-learn estimator. It has no fitted state
    or ``set_params`` method. Replace an entry through the containing
    preprocessor's ``set_params(feature_preprocessing=...)``, or through
    ``pre__feature_preprocessing`` when its pipeline step is named ``pre``.
    The replacement is used by the next fit; it does not refit automatically.
    Frozen preprocessors reject configuration replacement.

Requested and resolved settings
-------------------------------

``get_params()`` preserves the raw constructor arguments, including caller-owned
override mappings. Raw mapping edits before fit remain observable. Typed values
detach on construction; raw representation settings detach during normalization
and fitting.

:meth:`~pretab.Preprocessor.get_resolved_config` returns a detached current-request
dict, retaining the existing outer keys and expanding presets. Declared overrides
become canonical method mappings with applicable inherited controls and family
parameters. It does not enumerate undeclared columns, validate actual column
existence or detected kind before fit, or report learned widths or knot state.
Reading it never fits or transforms data. After fit or freeze it still describes
the current request, not the last successful fit. Use ``output_dims_``,
``total_output_dim_``, and :meth:`~pretab.Preprocessor.get_feature_info` for fitted
results.

Value construction validates field types and portable data. Resolution additionally
validates method names, option applicability, and target/placement combinations;
invalid feature declarations report their configuration path. Actual input checks
belong to fitting.

See the :doc:`configuration guide <../core_concepts/configuration>` for the
precedence table and three executable examples. Supported legacy string
configuration needs no syntax change, and can be mixed with structured entries.

Configuration, output, and reproducibility
------------------------------------------

Fitted preprocessors using string, mapping, or
:class:`~pretab.RepresentationConfig` entries can be saved with
:meth:`~pretab.Preprocessor.to_spec` and restored with
:meth:`~pretab.Preprocessor.from_spec` in the same environment. Restored names,
lineage, widths, output values, and fingerprints retain the fitted state,
including adaptive sizing and separate missing indicators. Representation
summaries describe the representation itself, excluding separate indicators;
output budgets and reported output dimensions include those indicators.

.. warning::

    Load specs only from trusted sources. Dependency versions recorded in a
    spec do not guarantee compatibility with a different environment.

Supporting types returned or consumed by the preprocessor.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentationSpec
   FeatureLineage
   RepresentationPolicy
