Extensibility
=============

The supported surface for adding, registering, discovering, and validating your
own representations. See the :doc:`custom representation tutorial
<../tutorials/custom_representation>` for a worked example.

.. currentmodule:: pretab

Base class and registration
---------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   BaseRepresentation
   register_representation
   list_representations
   check_representation
   load_entry_point_representations

Exceptions and warnings
-----------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   FrozenRepresentationError
   LeakageWarning
   OutputBudgetError
   PretabSerializationError
   PretabWarning
   RepresentationConformanceError

Logging
-------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   configure_logging
   set_verbosity
