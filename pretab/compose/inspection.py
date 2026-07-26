"""Introspect a fitted ColumnTransformer for output layout and feature metadata.

These helpers back the Preprocessor's ``transform`` slicing and its
``get_feature_info`` reporting: :func:`get_output_slices` computes each
transformer's contiguous span in the stacked output, :func:`build_feature_info`
collects per-feature preprocessing / dimension / category metadata, and
:func:`build_transformer_summary` renders that metadata as an aligned table.
"""

import numpy as np

from ..core.logging import get_logger

logger = get_logger(__name__)

__all__ = ["build_feature_info", "build_transformer_summary", "get_output_slices"]


def get_output_slices(column_transformer, X):
    """Return ordered ``(name, start, width)`` spans for each output block.

    The width of each transformer's block is obtained by transforming its input
    columns, matching the order in which the fitted ColumnTransformer stacks its
    outputs.
    """
    slices = []
    start = 0
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "transform"):
            width = transformer.transform(X[columns]).shape[1]
        else:
            width = 1
        slices.append((name, start, width))
        start += width
    return slices


def build_feature_info(column_transformer, *, embeddings, embedding_dimensions):
    """Collect per-feature metadata (preprocessing, dimension, categories).

    Returns a ``(numerical_info, categorical_info, embedding_info)`` tuple of
    dicts keyed by feature name.
    """
    numerical_feature_info = {}
    categorical_feature_info = {}

    embedding_feature_info = (
        {
            key: {"preprocessing": None, "dimension": dim, "categories": None}
            for key, dim in embedding_dimensions.items()
        }
        if embeddings
        else {}
    )

    for (
        name,
        transformer_pipeline,
        columns,
    ) in column_transformer.transformers_:
        steps = [step[0] for step in transformer_pipeline.steps]

        for feature_name in columns:
            preprocessing_type = " -> ".join(steps)
            dimension = None
            categories = None

            if "discretizer" in steps or any(
                step in steps
                for step in [
                    "standardization",
                    "minmax",
                    "quantile",
                    "polynomial",
                    "splines",
                    "box-cox",
                ]
            ):
                last_step = transformer_pipeline.steps[-1][1]
                if hasattr(last_step, "transform"):
                    dummy_input = np.zeros((1, 1)) + 1e-05
                    try:
                        transformed_feature = last_step.transform(dummy_input)
                        dimension = transformed_feature.shape[1]
                    except (ValueError, TypeError, AttributeError, IndexError) as exc:
                        logger.debug(
                            "Could not introspect output width of %r: %s",
                            feature_name,
                            exc,
                        )
                        dimension = None
                numerical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": None,
                }

            elif "continuous_ordinal" in steps:
                step = transformer_pipeline.named_steps["continuous_ordinal"]
                categories = len(step.mapping_[columns.index(feature_name)])
                dimension = 1
                categorical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": categories,
                }

            elif "onehot" in steps:
                step = transformer_pipeline.named_steps["onehot"]
                if hasattr(step, "categories_"):
                    categories = sum(len(cat) for cat in step.categories_)
                    dimension = categories
                categorical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": categories,
                }

            else:
                last_step = transformer_pipeline.steps[-1][1]
                if hasattr(last_step, "transform"):
                    dummy_input = np.zeros((1, 1))
                    try:
                        transformed_feature = last_step.transform(dummy_input)
                        dimension = transformed_feature.shape[1]
                    except (ValueError, TypeError, AttributeError, IndexError) as exc:
                        logger.debug(
                            "Could not introspect output width of %r: %s",
                            feature_name,
                            exc,
                        )
                        dimension = None
                if "cat" in name:
                    categorical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": None,
                    }
                else:
                    numerical_feature_info[feature_name] = {
                        "preprocessing": preprocessing_type,
                        "dimension": dimension,
                        "categories": None,
                    }

    return numerical_feature_info, categorical_feature_info, embedding_feature_info


def build_transformer_summary(numerical_info, categorical_info, embedding_info):
    """Build aligned, human-readable rows describing the fitted feature layout."""
    rows = []
    for feat, info in numerical_info.items():
        rows.append((str(feat), "numerical", str(info["preprocessing"]), info["dimension"], info["categories"]))
    for feat, info in categorical_info.items():
        rows.append((str(feat), "categorical", str(info["preprocessing"]), info["dimension"], info["categories"]))
    for feat, info in embedding_info.items():
        rows.append((str(feat), "embedding", "-", info["dimension"], info["categories"]))
    if not rows:
        return []

    feat_w = max(len("feature"), *(len(r[0]) for r in rows))
    kind_w = max(len("kind"), *(len(r[1]) for r in rows))
    pipe_w = max(len("pipeline"), *(len(r[2]) for r in rows))
    header = f"{'feature':<{feat_w}}  {'kind':<{kind_w}}  {'pipeline':<{pipe_w}}  {'dim':>4}  {'cats':>5}"
    lines = [header, "-" * len(header)]
    for feat, kind, pipe, dim, cats in rows:
        dim_s = "-" if dim is None else str(dim)
        cats_s = "-" if cats is None else str(cats)
        lines.append(f"{feat:<{feat_w}}  {kind:<{kind_w}}  {pipe:<{pipe_w}}  {dim_s:>4}  {cats_s:>5}")
    return lines
