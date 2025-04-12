from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ..transformers.utils.continuous_ordinal import ContinuousOrdinalTransformer
from ..transformers.utils.floats import ToFloatTransformer, NoTransformer
from ..transformers.binning import CustomBinTransformer
from ..transformers.onehot import OneHotFromOrdinalTransformer
from ..transformers.embeddings import LanguageEmbeddingTransformer


def get_categorical_transformer_steps(
    method: str,
    add_imputer: bool = True,
    imputer_strategy: str = "most_frequent",
    imputer_kwargs: dict = None,
    **kwargs,
):
    """
    Returns a list of (name, transformer) steps for a given categorical preprocessing method.
    """
    method = method.lower()
    steps = []

    if add_imputer:
        imputer_kwargs = imputer_kwargs or {}
        steps.append(
            ("imputer", SimpleImputer(strategy=imputer_strategy, **imputer_kwargs))
        )

    if method == "int":
        steps.append(("continuous_ordinal", ContinuousOrdinalTransformer()))
    elif method == "one-hot":
        steps.append(("onehot", OneHotEncoder(**kwargs)))
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
        raise ValueError(f"Unknown categorical transformer method: {method}")

    return steps
