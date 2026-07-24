from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ..core.exceptions import invalid_param_error
from ..transformers.binning import CustomBinTransformer
from ..transformers.embeddings import LanguageEmbeddingTransformer
from ..transformers.encoders.continuous_ordinal import ContinuousOrdinalTransformer
from ..transformers.encoders.floats import NoTransformer, ToFloatTransformer
from ..transformers.onehot import OneHotFromOrdinalTransformer
from .registry import CATEGORICAL_ALIASES, CATEGORICAL_METHODS, resolve_method


def get_categorical_transformer_steps(
    method: str,
    add_imputer: bool = True,
    imputer_strategy: str = "most_frequent",
    imputer_kwargs: dict | None = None,
    **kwargs,
):
    """
    Returns a list of (name, transformer) steps for a given categorical preprocessing method.
    """
    method = resolve_method(method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES)
    steps = []

    if add_imputer:
        imputer_kwargs = imputer_kwargs or {}
        steps.append(
            ("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs))
        )

    if method == "int":
        steps.append(("continuous_ordinal", ContinuousOrdinalTransformer()))
    elif method == "one-hot":
        # Default to ignoring unseen categories so transform never crashes on
        # categories absent at fit time; callers can override via kwargs.
        onehot_kwargs = {"handle_unknown": "ignore", **kwargs}
        steps.append(("onehot", OneHotEncoder(**onehot_kwargs)))
        steps.append(("to_float", ToFloatTransformer()))
    elif method == "pretrained":
        steps.append(("pretrained", LanguageEmbeddingTransformer()))
    elif method == "none":
        steps.append(("none", NoTransformer()))
    elif method == "custombin":
        steps.append(("custombin", CustomBinTransformer(**kwargs)))
    elif method == "onehot_from_ordinal":
        steps.append(("onehot_from_ordinal", OneHotFromOrdinalTransformer()))
    else:
        raise invalid_param_error(
            "get_categorical_transformer_steps", "method", method,
            "unrecognized categorical preprocessing method",
            valid=set(CATEGORICAL_METHODS),
        )

    return steps
