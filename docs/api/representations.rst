Representations
===============

Every built-in transformer exported from ``pretab.transformers``. These are the
standalone, scikit-learn compatible representations. For a capability-oriented
view, see the :doc:`comparison table <../representations/comparison_table>`.

.. currentmodule:: pretab.transformers

Splines
-------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   BSplineTransformer
   MSplineTransformer
   ISplineTransformer
   CubicRegressionSplineTransformer
   NaturalCubicSplineTransformer
   PSplineTransformer
   TensorProductSplineTransformer
   ThinPlateSplineTransformer

Feature maps
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RBFExpansionTransformer
   ReLUExpansionTransformer
   SigmoidExpansionTransformer
   TanhExpansionTransformer
   FourierFeatureTransformer
   PeriodicEncodingTransformer
   RandomFourierFeaturesTransformer
   NystroemFeaturesTransformer

Binning and piecewise-linear encoding
-------------------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   NumericBinningTransformer
   PLETransformer

Categorical
-----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ContinuousOrdinalTransformer
   OneHotFromOrdinalTransformer
   LanguageEmbeddingTransformer

Utility transformers
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   MissingStateIndicator
   NoTransformer
   ToFloatTransformer
