from ._version import __version__
from .compose.search import RepresentationSearchCV
from .core.logging import configure_logging, set_verbosity
from .core.policy import RepresentationPolicy
from .core.representation import FeatureLineage, RepresentationSpec
from .core.supervised import CrossFittedTransformer
from .exceptions import (
    FrozenRepresentationError,
    LeakageWarning,
    OutputBudgetError,
    PretabSerializationError,
    PretabWarning,
    RepresentationConformanceError,
)
from .extension import (
    BaseRepresentation,
    check_representation,
    list_representations,
    load_entry_point_representations,
    register_representation,
)
from .preprocessor import Preprocessor

__all__ = [
    "BaseRepresentation",
    "CrossFittedTransformer",
    "FeatureLineage",
    "FrozenRepresentationError",
    "LeakageWarning",
    "OutputBudgetError",
    "Preprocessor",
    "PretabSerializationError",
    "PretabWarning",
    "RepresentationConformanceError",
    "RepresentationPolicy",
    "RepresentationSearchCV",
    "RepresentationSpec",
    "__version__",
    "check_representation",
    "configure_logging",
    "list_representations",
    "load_entry_point_representations",
    "register_representation",
    "set_verbosity",
]
