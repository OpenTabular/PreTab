Search and cross-fitting
========================

Tools for selecting a representation and for producing leakage-free supervised
features. See :doc:`../core_concepts/target_awareness` for the leakage model.

.. currentmodule:: pretab

Representation search
---------------------

Cross-validate a downstream estimator over candidate numerical methods and refit
the best one.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentationSearchCV

Cross-fitting
-------------

Produce out-of-fold training features from a supervised transformer while
transforming new data with an all-data model.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   CrossFittedTransformer
