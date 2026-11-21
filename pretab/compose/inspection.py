"""Introspect a fitted ColumnTransformer for output layout and feature metadata.

These helpers back the Preprocessor's ``transform`` slicing and its
``get_feature_info`` reporting: :func:`get_output_slices` computes each
transformer's contiguous span in the stacked output, :func:`build_feature_info`
collects per-feature preprocessing / dimension / category metadata, and
:func:`build_transformer_summary` renders that metadata as an aligned table.
"""

import copy
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import quote

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import FeatureUnion, Pipeline

from ..core._typing import TransformerLike
from ..core.representation import FeatureLineage
from .registry import TRANSFORMER_REGISTRY

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


@dataclass(frozen=True)
class _FittedBlock:
    """Read-only block metadata retaining the fitted transformer and selector references."""

    step_name: str
    transformer: Any
    columns: Any
    _column_transformer: Any

    @property
    def public_name(self) -> str:
        return block_name(self.step_name, self.columns)

    @property
    def group_id(self) -> str | None:
        return self.step_name if self.step_name.startswith("group_") else None

    @property
    def span(self) -> slice:
        return self._column_transformer.output_indices_[self.step_name]


def _iter_fitted_blocks(column_transformer) -> Iterator[_FittedBlock]:
    """Yield every fitted block in order, leaving filtering and span access to callers."""
    for name, transformer, columns in column_transformer.transformers_:
        yield _FittedBlock(name, transformer, columns, column_transformer)


def get_output_slices(column_transformer):
    """Return ordered ``(name, start, width)`` spans for each output block.

    Reads widths from ``output_indices_`` — the fitted index map that
    ``ColumnTransformer`` already maintains — so no second transform is needed.
    ``name`` is the public block name (see :func:`block_name`).
    """
    indices = column_transformer.output_indices_
    slices = []
    for block in _iter_fitted_blocks(column_transformer):
        if block.transformer == "drop" or block.step_name not in indices:
            continue
        span = block.span
        if span is None:
            continue
        width = span.stop - span.start
        if width == 0:
            continue
        slices.append((block.public_name, span.start, width))
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
    for block in _iter_fitted_blocks(column_transformer):
        name, transformer, columns = block.step_name, block.transformer, block.columns
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
    column labels to the input feature names ``names`` were built from (see
    :func:`feature_names_out`); a step of a renamed column is then reported under
    the new name, e.g. ``num_feature_0`` as ``num_age``.
    Group names retain their public block key and prepend percent-escaped source
    labels in declared order to the leaf output name.
    """
    rename = rename or {}
    step_to_columns = {}
    group_ids = set()
    for block in _iter_fitted_blocks(column_transformer):
        if block.step_name == "remainder":
            continue
        if block.group_id is not None:
            group_ids.add(block.group_id)
        elif len(block.columns) != 1:
            continue
        step_to_columns[block.step_name] = [rename.get(str(column), column) for column in block.columns]
    cleaned = []
    for raw in names:
        raw = str(raw)
        step_name, sep, inner_name = raw.partition("__")
        columns = step_to_columns.get(step_name)
        if not sep or columns is None:
            cleaned.append(raw)
            continue
        if step_name in group_ids:
            sources = "+".join(quote(str(column), safe="") for column in columns)
            cleaned.append(f"{step_name}__{sources}__{inner_name}")
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


def build_feature_info(column_transformer, *, embeddings, embedding_dimensions):
    """Collect per-feature metadata (preprocessing, dimension, categories).

    Dimensions come from the fitted output spans without evaluating transformers.
    Category counts describe fitted vocabularies, not emitted widths.
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

    for block in _iter_fitted_blocks(column_transformer):
        name, transformer_pipeline, columns = block.step_name, block.transformer, block.columns
        if transformer_pipeline == "drop" or len(columns) == 0:
            continue
        span = block.span
        dimension = span.stop - span.start
        separate_state = _separate_state_branches(transformer_pipeline)
        if separate_state is not None:
            representation_pipeline, _missing_indicator = separate_state
            steps = [step[0] for step in representation_pipeline.steps]
            preprocessing_type = f"representation({' -> '.join(steps)}) + missing"
        else:
            representation_pipeline = transformer_pipeline
            steps = (
                [step[0] for step in representation_pipeline.steps]
                if isinstance(representation_pipeline, Pipeline)
                else [type(representation_pipeline).__name__]
            )
            preprocessing_type = " -> ".join(steps)

        group_id = block.group_id
        if group_id is not None:
            numerical_feature_info[group_id] = {
                "preprocessing": preprocessing_type,
                "dimension": dimension,
                "categories": None,
                "group": group_id.removeprefix("group_"),
                "source_features": tuple(columns),
            }
            continue
        for feature_name in columns:
            categories = None
            if "continuous_ordinal" in steps:
                step = representation_pipeline.named_steps["continuous_ordinal"]
                position = columns.index(feature_name)
                # An encoder that received no column (a 0-width block) has no mapping.
                has_mapping = position < len(step.mapping_)
                categories = len(step.mapping_[position]) if has_mapping else None
            elif "onehot" in steps:
                step = representation_pipeline.named_steps["onehot"]
                if hasattr(step, "categories_"):
                    categories = sum(len(cat) for cat in step.categories_)
            feature_info = (
                categorical_feature_info
                if name.startswith("cat_") or "continuous_ordinal" in steps or "onehot" in steps
                else numerical_feature_info
            )
            feature_info[feature_name] = {
                "preprocessing": preprocessing_type,
                "dimension": dimension,
                "categories": categories,
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


def _locate_block_representation(pipeline, columns):
    """Return ``(step_index, (family, component, uses_target, is_interaction))`` for a block.

    The representation-bearing step is the last pipeline step exposing a
    ``get_representation_spec`` (a PreTab transformer), a known scikit-learn
    step name, or a registered method that consumes the target; helper steps
    such as imputers and float casts are skipped. ``columns`` names the spec's
    inputs; with None the step names its own. ``step_index`` is that step's
    position in ``pipeline.steps`` (0 for a bare transformer), or None when the
    block has no representation step. Known scikit-learn names are a fallback:
    trailing scalers must not hide a representation's declared target use.
    """
    steps = pipeline.steps if hasattr(pipeline, "steps") else [("_", pipeline)]
    legacy_step = None
    for index in reversed(range(len(steps))):
        step_name, transformer = steps[index]
        if hasattr(transformer, "get_representation_spec"):
            # A step fitted on more inputs than the block has columns (behind the
            # imputer's built-in missing indicator) names its own inputs; its
            # get_feature_names_out rejects the block's shorter column list.
            input_features = None
            if columns is not None and getattr(transformer, "n_features_in_", len(columns)) == len(columns):
                input_features = list(columns)
            spec = transformer.get_representation_spec(input_features=input_features)
            return index, (spec.family, spec.component_kind, spec.uses_target, spec.is_interaction)
        registered = TRANSFORMER_REGISTRY.get(step_name)
        if (
            registered is not None
            and isinstance(transformer, registered.transformer_cls)
            and registered.target_usage != "forbidden"
        ):
            # A registered class without a RepresentationSpec (e.g. scikit-learn's
            # TargetEncoder): its declared supervision says whether it used y, so
            # cross-fitting refits it per fold instead of reusing the all-data fit.
            uses_target = registered.target_usage == "required" or bool(getattr(transformer, "target_aware", False))
            component = "category" if registered.is_categorical else "basis"
            return index, (step_name, component, uses_target, registered.is_multivariate)
        if step_name in _STEP_FAMILY and legacy_step is None:
            family, component = _STEP_FAMILY[step_name]
            legacy_step = index, (family, component, False, False)
    return legacy_step if legacy_step is not None else (None, ("passthrough", "raw", False, False))


def _resolve_block_representation(pipeline, columns):
    """Return ``(family, component, uses_target, is_interaction)`` for a block."""
    return _locate_block_representation(pipeline, columns)[1]


def _block_uses_task(transformer) -> bool:
    """Check a constructed block for a target consumer configured with a task."""
    separate_state = _separate_state_branches(transformer)
    representation = separate_state[0] if separate_state is not None else transformer
    for name, step in representation.steps:
        spec = TRANSFORMER_REGISTRY.get(name)
        if spec is None or not isinstance(step, spec.transformer_cls) or getattr(step, "task", None) is None:
            continue
        if spec.target_usage == "required" or (
            spec.target_usage == "optional" and getattr(step, "target_aware", False)
        ):
            return True
    return False


def block_uses_target(transformer) -> bool:
    """Whether a fitted single-feature or joint block's representation consumed ``y``.

    For a separate-state / missing-indicator union the representation branch decides; helper steps
    such as imputers and scalers never use the target. The spec is read without
    input names, since only its target use matters here.
    """
    separate_state = _separate_state_branches(transformer)
    representation = separate_state[0] if separate_state is not None else transformer
    return bool(_resolve_block_representation(representation, None)[2])


def refit_block_representation(transformer, X, y) -> TransformerLike:
    """Copy a fitted block and refit its representation on its ordered inputs and ``y``.

    Only the representation step and the steps after it are refit. The steps
    before it (the imputer with its missing indicator, scalers) never use the
    target, so they are kept from the all-data fit and transform ``X`` for the
    refit: a ``SimpleImputer(add_indicator=True)`` refit on rows without missing
    values would drop its indicator column and so change the block's width. For
    a separate-state union the missing branch is kept as well. The fitted
    ``transformer`` itself is left untouched.
    """
    separate_state = _separate_state_branches(transformer)
    if separate_state is not None:
        representation = refit_block_representation(separate_state[0], X, y)
        union = copy.copy(transformer)
        union.transformer_list = [
            (name, representation if name == "representation" else branch)
            for name, branch in transformer.transformer_list
        ]
        return union
    index = _locate_block_representation(transformer, None)[0]
    if not isinstance(transformer, Pipeline) or index is None or index == 0:
        refit = cast(TransformerLike, clone(transformer))
        refit.fit(X, y)
        return refit
    refit = cast(Pipeline, clone(transformer[index:]))
    refit.fit(transformer[:index].transform(X), y)
    block = copy.copy(transformer)
    block.steps = [*transformer.steps[:index], *refit.steps]
    return cast(TransformerLike, block)


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
    for block in _iter_fitted_blocks(column_transformer):
        name, transformer, columns = block.step_name, block.transformer, block.columns
        if name not in output_indices:
            continue
        span = block.span
        if span is None:
            continue
        width = span.stop - span.start
        if width == 0:
            continue
        group_id = block.group_id
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
                        group_id=group_id,
                    )
                )
            indicator = missing_indicator.steps[-1][1] if isinstance(missing_indicator, Pipeline) else missing_indicator
            missing_sources = getattr(indicator, "features_", range(len(columns)))
            for offset in range(missing_width):
                index = span.start + representation_width + offset
                records.append(
                    FeatureLineage(
                        output_feature=output_names[index],
                        output_index=index,
                        source_features=(str(columns[missing_sources[offset]]),)
                        if group_id is not None
                        else source_features,
                        family="missing_state",
                        component="indicator",
                        component_index=offset,
                        uses_target=False,
                        is_interaction=False,
                        group_id=group_id,
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
                    group_id=group_id,
                )
            )
    records.sort(key=lambda record: record.output_index)
    return records
