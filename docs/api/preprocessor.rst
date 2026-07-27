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

Configuration, output, and reproducibility
------------------------------------------

Supporting types returned or consumed by the preprocessor.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentationSpec
   FeatureLineage
   RepresentationPolicy
