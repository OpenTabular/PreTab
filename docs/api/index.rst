API Reference
=============

This page documents the public API of pretab: the high-level
``pretab.preprocessor.Preprocessor`` and every transformer exported from
``pretab.transformers``.

Preprocessor
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pretab.preprocessor.Preprocessor

Encoders and binning
--------------------

.. currentmodule:: pretab.transformers

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   PLETransformer
   CustomBinTransformer
   OneHotFromOrdinalTransformer
   LanguageEmbeddingTransformer

Feature maps
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RBFExpansionTransformer
   ReLUExpansionTransformer
   SigmoidExpansionTransformer
   TanhExpansionTransformer

Splines
-------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   CubicSplineTransformer
   NaturalCubicSplineTransformer
   PSplineTransformer
   TensorProductSplineTransformer
   ThinPlateSplineTransformer

Temporal
--------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   CyclicalTimeTransformer
   LagFeatureTransformer
   RollingStatsTransformer
