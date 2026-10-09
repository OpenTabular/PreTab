"""Introspect a fitted ColumnTransformer for output layout and feature metadata.

These helpers back the Preprocessor's ``transform`` slicing and its
``get_feature_info`` reporting: :func:`get_output_slices` computes each
transformer's contiguous span in the stacked output, :func:`build_feature_info`
collects per-feature preprocessing / dimension / category metadata, and
:func:`build_transformer_summary` renders that metadata as an aligned table.
"""

import copy
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline

from ..core.logging import get_logger
from ..core.representation import FeatureLineage
from .registry import TRANSFORMER_REGISTRY

logger = get_logger(__name__)

__all__ = [
    "block_name",
    "block_uses_target",
    "build_feature_info",
    "build_feature_lineage",
    "build_transformer_summary",
    "clean_feature_names",
    "feature_names_out",
    "get_output_slices",
    "refit_block_representation",
    "representation_leaf",
]


def _block(step_name, columns):
    """Return ``(kind, feature)`` for a per-column ``num_*`` / ``cat_*`` step, else ``None``."""
    if step_name == "remainder" or len(columns) != 1:
        return None
    kind, sep, _ = str(step_name).partition("_")
    if not sep or kind not in ("num", "cat"):
        return None
    return kind, str(columns[0])


def block_name(step_name, columns):
    """Return the public block name (``"num_<label>"`` / ``"cat_<label>"``) of a step.

    The name is built from the step's kind and its column label rather than read
    from the step name itself, which differs for labels scikit-learn cannot embed
    in a step name (see :func:`pretab.compose.factory._step_names`). Steps that are
    not per-column representation blocks keep their own name.
    """
    block = _block(step_name, columns)
    return step_name if block is None else f"{block[0]}_{block[1]}"


def get_output_slices(column_transformer):
    """Return ordered ``(name, start, width)`` spans for each output block.

    Reads widths from ``output_indices_`` — the fitted index map that
    ``ColumnTransformer`` already maintains — so no second transform is needed.
    ``name`` is the public block name (see :func:`block_name`).
    """
    indices = column_transformer.output_indices_
    slices = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop":
            continue
        span = indices.get(name)
        if span is None:
            continue
        width = span.stop - span.start
        if width == 0:
            continue
        slices.append((block_name(name, columns), span.start, width))
    return slices


def feature_names_out(column_transformer, input_features=None):
    """Return the cleaned output feature names of a fitted ColumnTransformer.

    ``input_features`` -- one name per input column, in input order -- replaces the
    column labels the ColumnTransformer was fitted on (e.g. the synthetic
    ``feature_i`` labels of an array input), following scikit-learn's contract
    for an estimator fitted without feature names. Each step's names are rebuilt
    from its fitted transformer with the new labels, so names that embed a label
    anywhere (e.g. polynomial terms) are renamed consistently.
    """
    fitted_labels = getattr(column_transformer, "feature_names_in_", None)
    if input_features is None or fitted_labels is None:
        raw_names = column_transformer.get_feature_names_out(input_features)
        return clean_feature_names(column_transformer, [str(name) for name in raw_names])

    rename = {str(label): str(name) for label, name in zip(fitted_labels, input_features, strict=True)}
    raw_names = []
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == "drop" or len(columns) == 0:
            continue
        labels = [rename[str(column)] for column in columns]
        if transformer == "passthrough":
            inner = labels
        else:
            inner = _without_feature_names(transformer).get_feature_names_out(labels)
        raw_names.extend(f"{name}__{inner_name}" for inner_name in inner)
    return clean_feature_names(column_transformer, raw_names, rename=rename)


def _without_feature_names(estimator) -> Any:
    """Return a shallow copy of a fitted step that accepts any input feature names.

    The ColumnTransformer hands each step a DataFrame column, so scikit-learn
    estimators inside it record ``feature_names_in_`` and only accept those exact
    names in ``get_feature_names_out``. Dropping the attribute on a shallow copy
    (and on copies of any nested pipeline / union steps) makes them fall back to
    the length check scikit-learn applies to estimators fitted without feature
    names, while the fitted step itself is left untouched.
    """
    if not hasattr(estimator, "get_params"):
        return estimator
    stripped = copy.copy(estimator)
    stripped.__dict__.pop("feature_names_in_", None)
    if isinstance(stripped, Pipeline):
        stripped.steps = [(name, _without_feature_names(step)) for name, step in stripped.steps]
    elif isinstance(stripped, FeatureUnion):
        stripped.transformer_list = [
            (name, _without_feature_names(transformer)) for name, transformer in stripped.transformer_list
        ]
    return stripped


def clean_feature_names(column_transformer, names, *, rename=None):
    """Collapse the per-feature name that sklearn's ColumnTransformer duplicates.

    Each per-column step is a ``num_*`` / ``cat_*`` block for one feature (see
    ``compose/factory.py``), and every PreTab transformer's own
    ``get_feature_names_out`` already bakes the input feature name into each output
    column, so sklearn's default ``f"{step}__{inner}"`` naming doubles it, e.g.
    ``"num_age__age_bs0"``. This collapses that back to ``"num_age_bs0"``. Other
    names of a block keep the public block name as their prefix, and
    passthrough/remainder columns are left unchanged. ``rename`` maps fitted
    column labels to the names ``names`` were built from (see
    :func:`feature_names_out`).
    """
    rename = rename or {}
    step_to_columns = {
        name: [rename.get(str(column), column) for column in columns]
        for name, _transformer, columns in column_transformer.transformers_
        if name != "remainder" and len(columns) == 1
    }
    cleaned = []
    for raw in names:
        raw = str(raw)
        step_name, sep, inner_name = raw.partition("__")
        columns = step_to_columns.get(step_name)
        if not sep or columns is None:
            cleaned.append(raw)
            continue
        block = _block(step_name, columns)
        if block is not None:
            kind_prefix, feature = block
            public_name = f"{kind_prefix}_{feature}"
        else:
            feature = str(columns[0])
            kind_prefix = step_name[: -(len(feature) + 1)] if step_name.endswith(f"_{feature}") else ""
            public_name = step_name
        if inner_name == feature or inner_name.startswith(f"{feature}_"):
            cleaned.append(f"{kind_prefix}_{inner_name}" if kind_prefix else inner_name)
        else:
            cleaned.append(f"{public_name}__{inner_name}")
    return cleaned


def _separate_state_branches(transformer):
    """Return the representation and missing branches of a separate-state union."""
    if not isinstance(transformer, FeatureUnion):
        return None
    branches = dict(transformer.transformer_list)
    if "representation" not in branches or "missing" not in branches:
        return None
    return branches["representation"], branches["missing"]


def representation_leaf(transformer):
    """Return the last step of a fitted per-column block's representation.

    For a separate-state / missing-indicator union this is the last step of its
    ``"representation"`` pipeline (the raw missing indicator beside it is not
    part of the representation); for a pipeline it is the last step, and any
    other block is returned unchanged.
    """
    separate_state = _separate_state_branches(transformer)
    representation = transformer if separate_state is None else separate_state[0]
    return representation.steps[-1][1] if hasattr(representation, "steps") else representation


def _probe_input(step, value):
    """Return a one-row probe input for measuring a fitted step's output width.

    A step fitted directly on the ColumnTransformer's DataFrame column records
    ``feature_names_in_`` and, like scikit-learn, warns on input without feature
    names, so its probe is a DataFrame carrying the fitted names.
    """
    names = getattr(step, "feature_names_in_", None)
    if names is None:
        return np.full((1, 1), value)
    return pd.DataFrame(np.full((1, len(names)), value), columns=names)


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
                    dummy_input = _probe_input(last_step, 1e-05)
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
                position = columns.index(feature_name)
                # An encoder that received no column (a 0-width block) has no mapping.
                has_mapping = position < len(step.mapping_)
                categories = len(step.mapping_[position]) if has_mapping else None
                dimension = separate_state_dimension if separate_state_dimension is not None else int(has_mapping)
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
                    dummy_input = _probe_input(last_step, 0.0)
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
                if name.startswith("cat_"):
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
    ``get_representation_spec`` (a PreTab transformer), a known scikit-learn
    step name, or a registered method that consumes the target; helper steps
    such as imputers and float casts are skipped.
    """
    steps = pipeline.steps if hasattr(pipeline, "steps") else [("_", pipeline)]
    for step_name, transformer in reversed(steps):
        if hasattr(transformer, "get_representation_spec"):
            spec = transformer.get_representation_spec(input_features=list(columns))
            return spec.family, spec.component_kind, spec.uses_target, spec.is_interaction
        if step_name in _STEP_FAMILY:
            family, component = _STEP_FAMILY[step_name]
            return family, component, False, False
        registered = TRANSFORMER_REGISTRY.get(step_name)
        if registered is not None and registered.target_usage != "forbidden":
            # A registered class without a RepresentationSpec (e.g. scikit-learn's
            # TargetEncoder): its declared supervision says whether it used y, so
            # cross-fitting refits it per fold instead of reusing the all-data fit.
            uses_target = registered.target_usage == "required" or bool(getattr(transformer, "target_aware", False))
            component = "category" if registered.is_categorical else "basis"
            return step_name, component, uses_target, registered.is_multivariate
    return "passthrough", "raw", False, False


def block_uses_target(transformer, columns) -> bool:
    """Whether a fitted per-column block's representation consumed the target ``y``.

    For a separate-state / missing-indicator union the representation branch
    decides; helper steps such as imputers and scalers never use the target.
    """
    separate_state = _separate_state_branches(transformer)
    representation = separate_state[0] if separate_state is not None else transformer
    return bool(_resolve_block_representation(representation, columns)[2])


def refit_block_representation(transformer, X, y):
    """Return a copy of a fitted per-column block whose representation is refit on ``(X, y)``.

    For a separate-state / missing-indicator union only the representation
    branch is refit; the fitted missing branch is kept, since it never uses the
    target and a ``MissingIndicator`` refit on rows without missing values would
    drop its column (and so change the block's width). Any other block is cloned
    and refit as a whole. The fitted ``transformer`` itself is left untouched.
    """
    separate_state = _separate_state_branches(transformer)
    if separate_state is None:
        return cast(Any, clone(transformer)).fit(X, y)
    representation = cast(Any, clone(separate_state[0])).fit(X, y)
    union = copy.copy(transformer)
    union.transformer_list = [
        (name, representation if name == "representation" else branch) for name, branch in transformer.transformer_list
    ]
    return union


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
            representation_pipeline, missing_indicator = separate_state
            # The missing branch comes last in the union; read its width from the
            # fitted indicator, which may emit no column (MissingIndicator only
            # marks features that had missing values at fit).
            missing_width = len(missing_indicator.get_feature_names_out([str(column) for column in columns]))
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
