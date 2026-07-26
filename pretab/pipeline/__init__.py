"""Pipeline assembly layer: build scikit-learn transformer steps per strategy.

``get_numerical_transformer_steps`` and ``get_categorical_transformer_steps``
turn a strategy name plus keyword arguments into an ordered list of
``(name, transformer)`` steps. The available numerical strategies are declared
in :mod:`pretab.pipeline.registry`.
"""

from .categorical import get_categorical_transformer_steps
from .numerical import get_numerical_transformer_steps

__all__ = [
    "get_categorical_transformer_steps",
    "get_numerical_transformer_steps",
]
