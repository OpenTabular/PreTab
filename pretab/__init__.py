from ._version import __version__
from .compose.search import RepresentationSearchCV
from .core.logging import configure_logging, set_verbosity
from .core.policy import RepresentationPolicy
from .core.representation import FeatureLineage, RepresentationSpec
from .core.supervised import CrossFittedTransformer
from .exceptions import LeakageWarning, OutputBudgetError, PretabWarning
from .preprocessor import Preprocessor

__all__ = [
    "CrossFittedTransformer",
    "FeatureLineage",
    "LeakageWarning",
    "OutputBudgetError",
    "Preprocessor",
    "PretabWarning",
    "RepresentationPolicy",
    "RepresentationSearchCV",
    "RepresentationSpec",
    "__version__",
    "configure_logging",
    "set_verbosity",
]
