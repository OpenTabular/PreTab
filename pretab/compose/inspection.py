"""Introspect a fitted ColumnTransformer for output layout and feature metadata.

These helpers back the Preprocessor's ``transform`` slicing and its
``get_feature_info`` reporting: :func:`get_output_slices` computes each
transformer's contiguous span in the stacked output, :func:`build_feature_info`
collects per-feature preprocessing / dimension / category metadata, and
:func:`build_transformer_summary` renders that metadata as an aligned table.
"""

import numpy as np
from sklearn.pipeline import FeatureUnion

from ..core.logging import get_logger
from ..core.representation import FeatureLineage

logger = get_logger(__name__)

__all__ = [
    "build_feature_info",
    "build_feature_lineage",
    "build_transformer_summary",
    "clean_feature_names",
    "get_output_slices",
]


def get_output_slices(column_transformer):
    """Return ordered ``(name, start, width)`` spans for each output block.

    Reads widths from ``output_indices_`` — the fitted index map that
    ``ColumnTransformer`` already maintains — so no second transform is needed.
    """
    indices = column_transformer.output_indices_
    slices = []
    for name, transformer, _columns in column_transformer.transformers_:
        if transformer == "drop":
            continue
        span = indices.get(name)
        if span is None:
            continue
        width = span.stop - span.start
        if width == 0:
            continue
        slices.append((name, span.start, width))
    return slices


def clean_feature_names(column_transformer, names):
    """Collapse the per-feature name that sklearn's ColumnTransformer duplicates.

    Each per-column step is named ``f"{kind}_{feature}"`` (see ``compose/factory.py``),
    and every PreTab transformer's own ``get_feature_names_out`` already bakes the
    input feature name into each output column, so sklearn's default
    ``f"{step}__{inner}"`` naming doubles it, e.g. ``"num_age__age_bs0"``. This
    collapses that back to ``"num_age_bs0"``, leaving passthrough/remainder columns
    and any name it cannot confidently match unchanged.
    """
    step_to_feature = {
        name: columns[0]
        for name, _transformer, columns in column_transformer.transformers_
        if name != "remainder" and len(columns) == 1
    }
    cleaned = []
    for raw in names:
        raw = str(raw)
        step_name, sep, inner_name = raw.partition("__")
        feature = step_to_feature.get(step_name)
        if not sep or feature is None:
            cleaned.append(raw)
            continue
        if inner_name == feature or inner_name.startswith(f"{feature}_"):
            kind_prefix = step_name[: -(len(feature) + 1)] if step_name.endswith(f"_{feature}") else ""
            cleaned.append(f"{kind_prefix}_{inner_name}" if kind_prefix else inner_name)
        else:
            cleaned.append(raw)
    return cleaned


def _separate_state_branches(transformer):
    """Return the representation and missing branches of a separate-state union."""
    if not isinstance(transformer, FeatureUnion):
        return None
    branches = dict(transformer.transformer_list)
    if "representation" not in branches or "missing" not in branches:
        return None
    return branches["representation"], branches["missing"]


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
        separate_state = _separate_state_branches(transformer_pipeline)
        if separate_state is not None:
            representation_pipeline, _missing_indicator = separate_state
            steps = [step[0] for step in representation_pipeline.steps]
            preprocessing_type = f"representation({' -> '.join(steps)}) + missing"
            span = column_transformer.output_indices_.get(name)
            separate_state_dimension = None if span is None else span.stop - span.start
        else:
            representation_pipeline = transformer_pipeline
            steps = [step[0] for step in representation_pipeline.steps]
            preprocessing_type = " -> ".join(steps)
            separate_state_dimension = None

        for feature_name in columns:
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
                last_step = representation_pipeline.steps[-1][1]
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
                if separate_state_dimension is not None:
                    dimension = separate_state_dimension
                numerical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": None,
                }

            elif "continuous_ordinal" in steps:
                step = representation_pipeline.named_steps["continuous_ordinal"]
                categories = len(step.mapping_[columns.index(feature_name)])
                dimension = separate_state_dimension if separate_state_dimension is not None else 1
                categorical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": categories,
                }

            elif "onehot" in steps:
                step = representation_pipeline.named_steps["onehot"]
                if hasattr(step, "categories_"):
                    categories = sum(len(cat) for cat in step.categories_)
                    dimension = categories
                if separate_state_dimension is not None:
                    dimension = separate_state_dimension
                categorical_feature_info[feature_name] = {
                    "preprocessing": preprocessing_type,
                    "dimension": dimension,
                    "categories": categories,
                }

            else:
                last_step = representation_pipeline.steps[-1][1]
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
                if separate_state_dimension is not None:
                    dimension = separate_state_dimension
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


# Mapping from pipeline step name to (family, component) for representation-bearing
# scikit-learn steps that do not expose ``get_representation_spec``.
_STEP_FAMILY = {
    "standardization": ("standardization", "raw"),
    "scaler": ("standardization", "raw"),
    "minmax": ("minmax", "raw"),
    "robust": ("robust", "raw"),
    "quantile": ("quantile", "raw"),
    "polynomial": ("polynomial", "basis"),
    "boxcox": ("box_cox", "raw"),
    "yeojohnson": ("yeo_johnson", "raw"),
    "onehot": ("onehot", "category"),
    "pretrained": ("language_embedding", "embedding"),
}


def _resolve_block_representation(pipeline, columns):
    """Return ``(family, component, uses_target, is_interaction)`` for a block.

    The representation-bearing step is the last pipeline step exposing a
    ``get_representation_spec`` (a PreTab transformer) or a known scikit-learn
    step name; helper steps such as imputers and float casts are skipped.
    """
    steps = pipeline.steps if hasattr(pipeline, "steps") else [("_", pipeline)]
    for step_name, transformer in reversed(steps):
        if hasattr(transformer, "get_representation_spec"):
            spec = transformer.get_representation_spec(input_features=list(columns))
            return spec.family, spec.component_kind, spec.uses_target, spec.is_interaction
        if step_name in _STEP_FAMILY:
            family, component = _STEP_FAMILY[step_name]
            return family, component, False, False
    return "passthrough", "raw", False, False


def _passthrough_source(columns, offset, feature_names_in):
    """Resolve the source feature name for a passthrough / remainder column."""
    column = columns[offset] if offset < len(columns) else columns[-1]
    if isinstance(column, (int, np.integer)) and feature_names_in is not None:
        return str(feature_names_in[column])
    return str(column)


def build_feature_lineage(column_transformer):
    """Return per-output-column :class:`FeatureLineage` records.

    Each record maps one output column of the fitted ColumnTransformer back to
    its source feature(s), representation family, and component, covering 100%
    of the transformed columns in ``get_feature_names_out`` order.
    """
    output_names = clean_feature_names(
        column_transformer, [str(name) for name in column_transformer.get_feature_names_out()]
    )
    output_indices = column_transformer.output_indices_
    feature_names_in = getattr(column_transformer, "feature_names_in_", None)
    records = []
    for name, transformer, columns in column_transformer.transformers_:
        span = output_indices.get(name)
        if span is None:
            continue
        width = span.stop - span.start
        if width == 0:
            continue
        if transformer == "passthrough" or name == "remainder":
            for offset in range(width):
                index = span.start + offset
                records.append(
                    FeatureLineage(
                        output_feature=output_names[index],
                        output_index=index,
                        source_features=(_passthrough_source(columns, offset, feature_names_in),),
                        family="passthrough",
                        component="raw",
                        component_index=offset,
                        uses_target=False,
                        is_interaction=False,
                    )
                )
            continue

        separate_state = _separate_state_branches(transformer)
        if separate_state is not None:
            representation_pipeline, _missing_indicator = separate_state
            union_names = [str(value) for value in transformer.get_feature_names_out(list(columns))]
            representation_width = sum(value.startswith("representation__") for value in union_names)
            missing_width = sum(value.startswith("missing__") for value in union_names)
            if representation_width + missing_width != width:
                representation_width = width - missing_width

            family, component, uses_target, is_interaction = _resolve_block_representation(
                representation_pipeline, columns
            )
            source_features = tuple(str(column) for column in columns)
            for offset in range(representation_width):
                index = span.start + offset
                records.append(
                    FeatureLineage(
                        output_feature=output_names[index],
                        output_index=index,
                        source_features=source_features,
                        family=family,
                        component=component,
                        component_index=offset,
                        uses_target=uses_target,
                        is_interaction=is_interaction,
                    )
                )
            for offset in range(missing_width):
                index = span.start + representation_width + offset
                records.append(
                    FeatureLineage(
                        output_feature=output_names[index],
                        output_index=index,
                        source_features=source_features,
                        family="missing_state",
                        component="indicator",
                        component_index=offset,
                        uses_target=False,
                        is_interaction=False,
                    )
                )
            continue
        family, component, uses_target, is_interaction = _resolve_block_representation(transformer, columns)
        source_features = tuple(str(column) for column in columns)
        for offset in range(width):
            index = span.start + offset
            records.append(
                FeatureLineage(
                    output_feature=output_names[index],
                    output_index=index,
                    source_features=source_features,
                    family=family,
                    component=component,
                    component_index=offset,
                    uses_target=uses_target,
                    is_interaction=is_interaction,
                )
            )
    records.sort(key=lambda record: record.output_index)
    return records
