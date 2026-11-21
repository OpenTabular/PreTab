Search and cross-fitting
========================

Tools for selecting a representation and for producing leakage-free supervised
features. See :doc:`../core_concepts/target_awareness` for the leakage model.

.. currentmodule:: pretab

Representation search
---------------------

Cross-validate a downstream estimator over candidate numerical methods and refit
the best one.

Classifier searches use ``task="classification"`` for preprocessing unless
``preprocessor_params`` specifies ``task``. The search's keyword-only
``task_validation`` checks the downstream classifier or regressor task and active
candidate placement tasks. A candidate's own policy in
``preprocessor_params["task_validation"]`` is independent. Turning one off does
not turn the other off.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   RepresentationSearchCV

Cross-fitting
-------------

Produce out-of-fold training features from a supervised transformer while
transforming new data with an all-data model.

The wrapper's keyword-only ``task_validation`` checks its splitter's configured
task. It does not replace the wrapped estimator's policy. If a later pipeline
step needs transformed inputs to determine target use, it checks when those
inputs are available, before training its representations and before any folds.

Both wrappers accept ``"off"`` (default), ``"warn"`` (``ConfigWarning``), and
``"raise"`` (``InvalidParamError``). Matching checks for the same task, mode,
target and use need not repeat in the folds of one fit. An independent fit checks
again. See :doc:`../core_concepts/resolution_and_placement` for policy selection,
suppression and target-type limits.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   CrossFittedTransformer
