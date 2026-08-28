"""Single capability registry for every preprocessing method.

This module is the one place that answers *what a method is and what it can do*:
the transformer class to instantiate, the constructor arguments it accepts, and
the capability flags (feature kind, arity, target usage, valid placement
strategies, adaptive-resolution support, preprocessor compatibility, optional
dependency). The composition layer (:mod:`pretab.compose.config`,
:mod:`pretab.compose.factory`) and the public contract tests all derive their
behaviour from this table rather than from scattered per-family lists.

Name resolution (aliases + separator/case-insensitive matching) also lives here
so both the numerical and categorical sides resolve user-supplied method names
through a single implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from ..expansion.functional.fourier import FourierFeatureTransformer
from ..expansion.functional.rbf import RBFExpansionTransformer
from ..expansion.functional.relu import ReLUExpansionTransformer
from ..expansion.functional.sigmoid import SigmoidExpansionTransformer
from ..expansion.functional.tanh import TanhExpansionTransformer
from ..expansion.spline.b_spline import BSplineTransformer
from ..expansion.spline.cubic_regression import CubicRegressionSplineTransformer
from ..expansion.spline.i_spline import ISplineTransformer
from ..expansion.spline.m_spline import MSplineTransformer
from ..expansion.spline.multivariate.tensor_product import (
    TensorProductSplineTransformer,
)
from ..expansion.spline.multivariate.thin_plate import (
    ThinPlateSplineTransformer,
)
from ..expansion.spline.natural_cubic import NaturalCubicSplineTransformer
from ..expansion.spline.p_spline import PSplineTransformer
from ..transformers.categorical.language_embedding import (
    LanguageEmbeddingTransformer,
)
from ..transformers.categorical.legacy import OneHotFromOrdinalTransformer
from ..transformers.categorical.ordinal import ContinuousOrdinalTransformer
from ..transformers.encoders.floats import NoTransformer
from ..transformers.feature_maps.kernel_approx import (
    NystroemFeaturesTransformer,
    RandomFourierFeaturesTransformer,
)
from ..transformers.numerical.binning import NumericBinningTransformer
from ..transformers.numerical.piecewise import PLETransformer

__all__ = [
    "CATEGORICAL_ALIASES",
    "CATEGORICAL_METHODS",
    "NUMERICAL_ALIASES",
    "NUMERICAL_METHODS",
    "TRANSFORMER_REGISTRY",
    "TransformerSpec",
    "categorical_method_names",
    "get_spec",
    "numerical_method_names",
    "placement_strategies_for",
    "register_spec",
    "resolve_method",
    "supports_adaptive_resolution",
    "supports_target_aware",
]

# Canonical feature kinds a method can apply to.
NUMERICAL = "numerical"
CATEGORICAL = "categorical"

# Canonical placement strategy names, split by supervision.
_UNSUPERVISED_STRATEGIES = frozenset({"uniform", "quantile"})
_UNIFORM_ONLY = frozenset({"uniform"})
_TARGET_AWARE_STRATEGIES = frozenset({"cart", "lightgbm"})
_ALL_STRATEGIES = _UNSUPERVISED_STRATEGIES | _TARGET_AWARE_STRATEGIES


@dataclass(frozen=True)
class TransformerSpec:
    """Declarative capability record for a single preprocessing method.

    Parameters
    ----------
    name : str
        Canonical method name (the registry key).
    transformer_cls : type
        The scikit-learn-compatible transformer class to instantiate. Methods
        that build extra steps (e.g. ``one-hot`` appends a float cast, the power
        transforms prepend a positive-range scaler) record their primary class
        here; the extra wiring lives in :mod:`pretab.compose.factory`.
    allowed_args : tuple of str
        Constructor argument names the method accepts. Used to filter the shared
        Preprocessor keyword arguments down to what the class understands.
    feature_kind : frozenset of str
        The feature kinds the method applies to (``"numerical"`` and/or
        ``"categorical"``). Only ``none`` (passthrough) applies to both.
    arity : {"univariate", "multivariate"}
        Whether the method transforms one column at a time (``"univariate"``) or
        jointly models several columns (``"multivariate"`` -- the tensor-product
        and thin-plate splines).
    target_usage : {"forbidden", "optional", "required"}
        How the method uses the supervised target ``y`` for basis placement.
        ``"required"`` methods (PLE) always place against ``y``; ``"optional"``
        methods (feature maps and freely-placed knot splines) place against ``y``
        only when ``target_aware`` is set; ``"forbidden"`` methods never use it.
    placement_strategies : frozenset of str
        The placement strategies the method honours. Empty for methods with no
        data-driven placement.
    supports_adaptive_resolution : bool
        Whether the method can size each feature's output dimension from the data
        (within ``[min_output_dim, max_output_dim]``) instead of a fixed width.
    preprocessor_compatible : bool
        Whether the method can be selected per column through :class:`Preprocessor`.
    optional_dependency : str or None
        The optional extra that must be installed for the method to run
        (``pip install pretab[<extra>]``), or ``None`` when always available.
    periodic : bool
        Whether the representation encodes a periodic signal (e.g. the Fourier
        feature map). Surfaced through :func:`pretab.list_representations`.
    sparse_output : bool
        Whether the method can emit a sparse matrix (e.g. one-hot). Surfaced
        through :func:`pretab.list_representations`.
    """

    name: str
    transformer_cls: type
    allowed_args: tuple[str, ...] = ()
    feature_kind: frozenset[str] = field(default_factory=lambda: frozenset({NUMERICAL}))
    arity: str = "univariate"
    target_usage: str = "forbidden"
    placement_strategies: frozenset[str] = frozenset()
    supports_adaptive_resolution: bool = False
    preprocessor_compatible: bool = True
    optional_dependency: str | None = None
    periodic: bool = False
    sparse_output: bool = False

    @property
    def is_numerical(self) -> bool:
        """Whether the method applies to numerical columns."""
        return NUMERICAL in self.feature_kind

    @property
    def is_categorical(self) -> bool:
        """Whether the method applies to categorical columns."""
        return CATEGORICAL in self.feature_kind

    @property
    def is_multivariate(self) -> bool:
        """Whether the method jointly models several columns."""
        return self.arity == "multivariate"

    @property
    def target_aware_capable(self) -> bool:
        """Whether the method can place basis units against ``y``."""
        return self.target_usage in ("optional", "required")

    @property
    def requires_target(self) -> bool:
        """Whether the method always needs ``y`` for placement."""
        return self.target_usage == "required"

    @property
    def requires_y(self) -> bool:
        """Alias of :attr:`requires_target` matching the transformer contract."""
        return self.requires_target

    @property
    def is_supervised(self) -> bool:
        """Whether the method can consume ``y`` (optional or required)."""
        return self.target_aware_capable


def _spec(name, cls, allowed_args=(), **kwargs):
    """Construct a :class:`TransformerSpec`, normalising ``allowed_args``."""
    return TransformerSpec(name=name, transformer_cls=cls, allowed_args=tuple(allowed_args), **kwargs)


# Feature-map centers and freely-placed knot splines share the same placement
# capability: optional target awareness across all four strategies, plus adaptive
# resolution.
_BOTH_MODE = {
    "target_usage": "optional",
    "placement_strategies": _ALL_STRATEGIES,
    "supports_adaptive_resolution": True,
}

# name -> capability spec. Numerical methods first (preserving the historical
# ordering), then the categorical-only methods.
_SPECS: tuple[TransformerSpec, ...] = (
    # --- numerical: scalers / distribution transforms (no placement) ---
    _spec("standardization", StandardScaler),
    _spec("minmax", MinMaxScaler),
    _spec("quantile", QuantileTransformer, ("n_quantiles", "output_distribution", "random_state")),
    _spec("polynomial", PolynomialFeatures, ("degree", "interaction_only", "include_bias")),
    _spec("robust", RobustScaler),
    _spec("box-cox", PowerTransformer),
    _spec("yeo-johnson", PowerTransformer),
    # --- numerical: piecewise-linear encoding (always target-aware) ---
    _spec(
        "ple",
        PLETransformer,
        ("output_dim", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        target_usage="required",
        placement_strategies=_TARGET_AWARE_STRATEGIES,
        supports_adaptive_resolution=True,
    ),
    # --- numerical: binning (unsupervised uniform / quantile edge placement) ---
    _spec(
        "custombin",
        NumericBinningTransformer,
        ("output_dim", "encode"),
        placement_strategies=_UNSUPERVISED_STRATEGIES,
    ),
    # --- numerical: feature maps (optional target-aware, adaptive) ---
    _spec(
        "rbf",
        RBFExpansionTransformer,
        ("output_dim", "gamma", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    _spec(
        "relu",
        ReLUExpansionTransformer,
        ("output_dim", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    _spec(
        "sigmoid",
        SigmoidExpansionTransformer,
        ("output_dim", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    _spec(
        "tanh",
        TanhExpansionTransformer,
        ("output_dim", "scale", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    # --- numerical: deterministic Fourier feature map (univariate, unsupervised) ---
    _spec(
        "fourier",
        FourierFeatureTransformer,
        ("n_frequencies", "frequency_strategy", "include_original", "random_state"),
        periodic=True,
    ),
    # --- numerical: freely-placed knot splines (optional target-aware, adaptive) ---
    _spec(
        "cubicspline",
        CubicRegressionSplineTransformer,
        (
            "output_dim",
            "degree",
            "include_bias",
            "task",
            "adaptive",
            "min_output_dim",
            "max_output_dim",
            "random_state",
        ),
        **_BOTH_MODE,
    ),
    _spec(
        "naturalspline",
        NaturalCubicSplineTransformer,
        ("output_dim", "include_bias", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    # --- numerical: penalized splines (equally-spaced knots, unsupervised only) ---
    _spec(
        "pspline",
        PSplineTransformer,
        ("output_dim", "degree", "diff_order"),
        placement_strategies=_UNIFORM_ONLY,
    ),
    _spec(
        "tensorspline",
        TensorProductSplineTransformer,
        ("output_dim", "degree", "diff_order"),
        arity="multivariate",
        placement_strategies=_UNSUPERVISED_STRATEGIES,
        preprocessor_compatible=False,
    ),
    # --- numerical: kernel-based thin-plate spline (knot-free, multivariate) ---
    _spec(
        "tprs",
        ThinPlateSplineTransformer,
        ("n_components", "landmark_strategy", "rank_strategy", "random_state"),
        arity="multivariate",
        preprocessor_compatible=False,
    ),
    # --- numerical: kernel-approximation feature maps (multivariate, standalone) ---
    _spec(
        "rff",
        RandomFourierFeaturesTransformer,
        ("n_components", "gamma", "random_state"),
        arity="multivariate",
        preprocessor_compatible=False,
    ),
    _spec(
        "nystroem",
        NystroemFeaturesTransformer,
        ("n_components", "kernel", "gamma", "degree", "coef0", "random_state"),
        arity="multivariate",
        preprocessor_compatible=False,
    ),
    # --- numerical: B / M / I spline bases (optional target-aware, adaptive) ---
    _spec(
        "bspline",
        BSplineTransformer,
        ("degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    _spec(
        "mspline",
        MSplineTransformer,
        ("degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    _spec(
        "ispline",
        ISplineTransformer,
        ("degree", "task", "adaptive", "min_output_dim", "max_output_dim", "random_state"),
        **_BOTH_MODE,
    ),
    # --- numerical / categorical: passthrough ---
    _spec("none", NoTransformer, feature_kind=frozenset({NUMERICAL, CATEGORICAL})),
    # --- categorical-only methods ---
    _spec("int", ContinuousOrdinalTransformer, feature_kind=frozenset({CATEGORICAL})),
    _spec("one-hot", OneHotEncoder, feature_kind=frozenset({CATEGORICAL}), sparse_output=True),
    _spec("onehot_from_ordinal", OneHotFromOrdinalTransformer, feature_kind=frozenset({CATEGORICAL})),
    _spec(
        "pretrained",
        LanguageEmbeddingTransformer,
        feature_kind=frozenset({CATEGORICAL}),
        optional_dependency="embeddings",
    ),
)

TRANSFORMER_REGISTRY: dict[str, TransformerSpec] = {spec.name: spec for spec in _SPECS}


# ---------------------------------------------------------------------------
# Derived views (kept in sync with the registry; do not edit by hand).
# ---------------------------------------------------------------------------
def numerical_method_names() -> frozenset[str]:
    """Return the set of canonical numerical method names."""
    return frozenset(name for name, spec in TRANSFORMER_REGISTRY.items() if spec.is_numerical)


def categorical_method_names() -> frozenset[str]:
    """Return the set of canonical categorical method names."""
    return frozenset(name for name, spec in TRANSFORMER_REGISTRY.items() if spec.is_categorical)


# Derived lookup tables consumed by the factory and config layers.
# ``NUMERICAL_METHODS`` maps a numerical method to ``(class, allowed_args_list)``;
# ``CATEGORICAL_METHODS`` is the set of categorical method names. Methods flagged
# ``preprocessor_compatible=False`` (the multivariate tensor-product / thin-plate
# splines) are standalone-only and deliberately excluded from the per-column
# ``Preprocessor`` whitelist.
NUMERICAL_METHODS: dict[str, tuple[type, list[str]]] = {
    name: (spec.transformer_cls, list(spec.allowed_args))
    for name, spec in TRANSFORMER_REGISTRY.items()
    if spec.is_numerical and spec.preprocessor_compatible
}
# Mutable so :func:`pretab.register_representation` can extend the categorical
# whitelist in place and have the config / factory layers (which import this
# name) observe the addition immediately.
CATEGORICAL_METHODS: set[str] = set(categorical_method_names())


def get_spec(method: str) -> TransformerSpec:
    """Return the :class:`TransformerSpec` for a canonical ``method`` name.

    Raises
    ------
    KeyError
        If ``method`` is not a registered canonical method name. Callers that
        accept user input should resolve the name via :func:`resolve_method`
        first.
    """
    return TRANSFORMER_REGISTRY[method]


def placement_strategies_for(method: str) -> frozenset[str]:
    """Return the placement strategies a canonical ``method`` honours."""
    return TRANSFORMER_REGISTRY[method].placement_strategies


def supports_adaptive_resolution(method: str) -> bool:
    """Return whether a canonical ``method`` supports adaptive output sizing."""
    return TRANSFORMER_REGISTRY[method].supports_adaptive_resolution


def supports_target_aware(method: str) -> bool:
    """Return whether ``method`` supports target-aware basis placement.

    Accepts an alias or separator variant; resolves it first. True for the
    feature maps, PLE, and the freely-placed knot splines (``bspline`` /
    ``mspline`` / ``ispline`` / ``cubicspline`` / ``naturalspline``); False for
    the penalized and kernel-based splines and every non-placement method.
    """
    resolved = resolve_method(method, TRANSFORMER_REGISTRY, NUMERICAL_ALIASES)
    spec = TRANSFORMER_REGISTRY.get(resolved)
    return bool(spec and spec.target_aware_capable)


def register_spec(spec: TransformerSpec, *, override: bool = False) -> TransformerSpec:
    """Insert a :class:`TransformerSpec` into the live registry.

    Updates the registry and the derived ``NUMERICAL_METHODS`` /
    ``CATEGORICAL_METHODS`` views in place, so the config and factory layers
    (which import those names) immediately observe the new method. This backs the
    public :func:`pretab.register_representation`.

    Parameters
    ----------
    spec : TransformerSpec
        The capability record to register.
    override : bool, default=False
        Whether replacing an already-registered name is allowed.

    Raises
    ------
    TypeError
        If ``spec`` is not a :class:`TransformerSpec`.
    ValueError
        If ``spec.name`` is already registered and ``override`` is False.
    """
    if not isinstance(spec, TransformerSpec):
        raise TypeError(f"expected a TransformerSpec, got {type(spec).__name__}")
    if spec.name in TRANSFORMER_REGISTRY and not override:
        raise ValueError(f"method {spec.name!r} is already registered; pass override=True to replace it.")
    # Drop any stale derived-view entries before re-inserting (supports override).
    NUMERICAL_METHODS.pop(spec.name, None)
    CATEGORICAL_METHODS.discard(spec.name)
    TRANSFORMER_REGISTRY[spec.name] = spec
    if spec.is_numerical and spec.preprocessor_compatible:
        NUMERICAL_METHODS[spec.name] = (spec.transformer_cls, list(spec.allowed_args))
    if spec.is_categorical:
        CATEGORICAL_METHODS.add(spec.name)
    return spec


# ---------------------------------------------------------------------------
# Name resolution (aliases + separator/case-insensitive matching).
# ---------------------------------------------------------------------------
def _squash(name: str) -> str:
    """Collapse a method name for separator/case-insensitive comparison.

    Lowercases, trims surrounding whitespace, and drops the ``-``, ``_`` and
    space separators so ``"One-Hot"``, ``"one_hot"`` and ``"onehot"`` all map to
    the same key. Canonical names that only differ by a separator (``"box-cox"``
    vs ``"boxcox"``, ``"cubicspline"`` vs ``"cubic spline"``) therefore match
    without needing an explicit alias entry.
    """
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


# Genuine synonyms / abbreviations that are *not* just separator variants of a
# canonical name (those are handled by :func:`_squash`). Keys are already
# squashed; values are canonical numerical method names.
NUMERICAL_ALIASES = {
    "standard": "standardization",
    "standardize": "standardization",
    "standardscaler": "standardization",
    "std": "standardization",
    "zscore": "standardization",
    "minmaxscaler": "minmax",
    "quantiletransformer": "quantile",
    "poly": "polynomial",
    "robustscaler": "robust",
    "piecewiselinear": "ple",
    "bin": "custombin",
    "binning": "custombin",
    "cubic": "cubicspline",
    "natural": "naturalspline",
    "naturalcubic": "naturalspline",
    "tensor": "tensorspline",
    "tensorproduct": "tensorspline",
    "tensorproductspline": "tensorspline",
    "thinplate": "tprs",
    "thinplatespline": "tprs",
    "fourierfeatures": "fourier",
    "randomfourier": "rff",
    "randomfourierfeatures": "rff",
    "rbfsampler": "rff",
    "nystrom": "nystroem",
    "passthrough": "none",
    "identity": "none",
    "raw": "none",
}

# Genuine synonyms / abbreviations for the categorical methods (keys squashed).
CATEGORICAL_ALIASES = {
    "integer": "int",
    "ordinal": "int",
    "label": "int",
    "labelencoder": "int",
    "ordinalencoder": "int",
    "ohe": "one-hot",
    "dummy": "one-hot",
    "onehotencoder": "one-hot",
    "embedding": "pretrained",
    "embeddings": "pretrained",
    "language": "pretrained",
    "llm": "pretrained",
    "passthrough": "none",
    "identity": "none",
    "raw": "none",
}


def resolve_method(name, canonical, aliases):
    """Resolve a user-supplied method name to its canonical spelling.

    Matching is case-insensitive, ignores ``-`` / ``_`` / space separators, and
    honours the explicit ``aliases`` map of synonyms and abbreviations. An
    unrecognized name is returned lowercased and stripped so the caller's own
    "unrecognized method" error lists the canonical options.

    Parameters
    ----------
    name : str
        The method name the user supplied.
    canonical : set or dict
        The canonical method names (``NUMERICAL_METHODS`` keys or
        ``CATEGORICAL_METHODS``).
    aliases : dict
        Squashed-alias to canonical-name mapping for this side of the pipeline.
    """
    key = name.strip().lower()
    if key in canonical:
        return key

    squashed = _squash(name)
    for canon in canonical:
        if _squash(canon) == squashed:
            return canon
    if squashed in aliases:
        return aliases[squashed]
    return key
