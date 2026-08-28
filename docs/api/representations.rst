Representations
===============

Every built-in transformer exported from ``pretab.transformers``. These are the
standalone, scikit-learn compatible representations. For a capability-oriented
view, see the :doc:`comparison table <../representations/comparison_table>`.

.. currentmodule:: pretab.transformers

Spline expansions
------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   BSplineTransformer
   MSplineTransformer
   ISplineTransformer
   CubicRegressionSplineTransformer
   NaturalCubicSplineTransformer
   PSplineTransformer

Canonical import: ``pretab.expansion.spline``.

Multivariate splines
---------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   TensorProductSplineTransformer
   ThinPlateSplineTransformer

Canonical import: ``pretab.expansion.spline.multivariate``.

Functional expansions
----------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RBFExpansionTransformer
   ReLUExpansionTransformer
   SigmoidExpansionTransformer
   TanhExpansionTransformer
   FourierFeatureTransformer

Canonical import: ``pretab.expansion.functional``.

Kernel approximation
----------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RandomFourierFeaturesTransformer
   NystroemFeaturesTransformer

Canonical import: ``pretab.kernel_approximation``.

Numerical encoding
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   NumericBinningTransformer
   PLETransformer
   PeriodicEncodingTransformer

Canonical import: ``pretab.encoding.numerical``.

Categorical encoding
-----------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   ContinuousOrdinalTransformer
   OneHotFromOrdinalTransformer

Canonical import: ``pretab.encoding.categorical``.

Embeddings
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   LanguageEmbeddingTransformer

Canonical import: ``pretab.embedding``.

Preprocessing utilities
--------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   MissingStateIndicator
   NoTransformer
   ToFloatTransformer

Canonical import: ``pretab.preprocessing``.

