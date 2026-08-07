"""Typed representation metadata and feature-lineage records.

``RepresentationSpec`` is the machine-readable description of the basis /
encoding a fitted transformer produces (family, scope, supervision, knot or
center locations, ...).  ``RepresentationSpecMixin`` gives every PreTab
transformer a default ``get_representation_spec`` implementation driven by a
handful of class-attribute hooks, so concrete transformers usually only declare
their family and a couple of flags.  ``FeatureLineage`` records the per-output
column provenance assembled by the preprocessor.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "FeatureLineage",
    "RepresentationSpec",
    "RepresentationSpecMixin",
]


@dataclass(frozen=True)
class RepresentationSpec:
    """Typed description of the representation a transformer produces.

    Attributes
    ----------
    family : str
        Representation family identifier, e.g. ``"bspline"``, ``"rbf"``,
        ``"piecewise_linear"``.
    component_kind : str
        Nature of a single output column, e.g. ``"basis"``, ``"center"``,
        ``"frequency"``, ``"interval"``, ``"category"``, ``"raw"``.
    scope : str
        ``"univariate"`` when each input feature is expanded independently or
        ``"multivariate"`` for interaction / joint bases.
    supervision : str
        ``"unsupervised"``, ``"supervised"``, or ``"optional"`` (target used
        only when ``target_aware`` is enabled).
    uses_target : bool
        Whether the fitted transformer actually consumed ``y``.
    is_interaction : bool
        Whether output columns mix multiple input features.
    input_features : tuple of str
        Names of the input features consumed.
    output_features : tuple of str
        Names of the produced output columns (matches
        ``get_feature_names_out``).
    output_dim : int
        Number of output columns (``len(output_features)``).
    degree : int or None
        Polynomial / spline degree when applicable.
    include_bias : bool
        Whether an explicit intercept / bias column is included.
    periodic : bool
        Whether the representation encodes a periodic signal.
    period : float or None
        Period length when ``periodic`` is True.
    local_support : bool
        Whether individual basis functions have compact (local) support.
    location_kind : str or None
        Semantic label for ``locations`` (``"knots"``, ``"centers"``,
        ``"bin_edges"``, ``"thresholds"``, ``"frequencies"``, ``"landmarks"``).
    locations : tuple of tuple of float or None
        Fitted knot / center / threshold locations, one inner tuple per input
        feature (or per landmark row for multivariate bases).
    dtype : str
        Output dtype of the transformed array.
    cross_fitted : bool
        Whether the representation was produced with out-of-fold cross-fitting
        (see :class:`~pretab.core.supervised.CrossFittedTransformer`).
    n_folds : int or None
        Number of cross-fitting folds when ``cross_fitted`` is True.
    """

    family: str
    component_kind: str
    scope: str
    supervision: str
    uses_target: bool
    is_interaction: bool
    input_features: tuple[str, ...]
    output_features: tuple[str, ...]
    output_dim: int
    degree: int | None
    include_bias: bool
    periodic: bool
    period: float | None
    local_support: bool
    location_kind: str | None
    locations: tuple[tuple[float, ...], ...] | None
    dtype: str = "float64"
    cross_fitted: bool = False
    n_folds: int | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation."""
        return {
            "family": self.family,
            "component_kind": self.component_kind,
            "scope": self.scope,
            "supervision": self.supervision,
            "uses_target": self.uses_target,
            "is_interaction": self.is_interaction,
            "input_features": list(self.input_features),
            "output_features": list(self.output_features),
            "output_dim": self.output_dim,
            "degree": self.degree,
            "include_bias": self.include_bias,
            "periodic": self.periodic,
            "period": self.period,
            "local_support": self.local_support,
            "location_kind": self.location_kind,
            "locations": (None if self.locations is None else [list(group) for group in self.locations]),
            "dtype": self.dtype,
            "cross_fitted": self.cross_fitted,
            "n_folds": self.n_folds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RepresentationSpec":
        """Reconstruct a ``RepresentationSpec`` from :meth:`to_dict` output."""
        locations = data.get("locations")
        n_folds = data.get("n_folds")
        return cls(
            family=data["family"],
            component_kind=data["component_kind"],
            scope=data["scope"],
            supervision=data["supervision"],
            uses_target=bool(data["uses_target"]),
            is_interaction=bool(data["is_interaction"]),
            input_features=tuple(data["input_features"]),
            output_features=tuple(data["output_features"]),
            output_dim=int(data["output_dim"]),
            degree=None if data["degree"] is None else int(data["degree"]),
            include_bias=bool(data["include_bias"]),
            periodic=bool(data["periodic"]),
            period=None if data["period"] is None else float(data["period"]),
            local_support=bool(data["local_support"]),
            location_kind=data["location_kind"],
            locations=(None if locations is None else tuple(tuple(float(v) for v in group) for group in locations)),
            dtype=data.get("dtype", "float64"),
            cross_fitted=bool(data.get("cross_fitted", False)),
            n_folds=None if n_folds is None else int(n_folds),
        )


@dataclass(frozen=True)
class FeatureLineage:
    """Provenance record for a single output column of a fitted preprocessor.

    Attributes
    ----------
    output_feature : str
        Name of the produced output column (matches
        ``Preprocessor.get_feature_names_out``).
    output_index : int
        Position of the column in the transformed array.
    source_features : tuple of str
        Input column(s) this output column is derived from.
    family : str
        Representation family that produced the column.
    component : str
        Component kind of the column (``"basis"``, ``"center"``, ...).
    component_index : int
        Index of the column within its representation block.
    uses_target : bool
        Whether the producing transformer consumed ``y``.
    is_interaction : bool
        Whether the column mixes multiple input features.
    """

    output_feature: str
    output_index: int
    source_features: tuple[str, ...]
    family: str
    component: str
    component_index: int
    uses_target: bool
    is_interaction: bool

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary representation."""
        return {
            "output_feature": self.output_feature,
            "output_index": self.output_index,
            "source_features": list(self.source_features),
            "family": self.family,
            "component": self.component,
            "component_index": self.component_index,
            "uses_target": self.uses_target,
            "is_interaction": self.is_interaction,
        }


def _as_location_tuple(value) -> tuple[tuple[float, ...], ...]:
    """Normalise fitted location arrays into a tuple of float tuples.

    Handles both list-of-1D-array layouts (one array per input feature) and 2D
    arrays (one row per landmark) by iterating the outer axis.
    """
    return tuple(tuple(float(v) for v in np.asarray(group).ravel()) for group in value)


class RepresentationSpecMixin:
    """Provide a default ``get_representation_spec`` from class-attribute hooks.

    Concrete transformers declare their family and a few flags via the
    ``_representation_*`` class attributes; fitted knot / center / threshold
    locations are auto-detected from the first matching fitted attribute.
    Transformers with bespoke needs override the ``_representation_*`` helper
    methods (or ``get_representation_spec`` itself).
    """

    _representation_family: str = "unknown"
    _representation_component_kind: str = "basis"
    _representation_scope: str = "univariate"
    _representation_supervision: str = "unsupervised"
    _representation_local_support: bool = False

    _REPRESENTATION_LOCATION_ATTRS: tuple[tuple[str, str], ...] = (
        ("knots_", "knots"),
        ("centers_", "centers"),
        ("bin_edges_", "bin_edges"),
        ("thresholds_", "thresholds"),
        ("frequencies_", "frequencies"),
        ("landmarks_", "landmarks"),
    )

    @property
    def requires_y(self) -> bool:
        """Whether this transformer mandates ``y`` at fit time.

        ``True`` for inherently supervised representations (e.g. PLE); ``False``
        for unsupervised and optionally target-aware families.
        """
        return self._representation_supervision == "supervised"

    @property
    def is_supervised(self) -> bool:
        """Whether this transformer consumes ``y`` given its configuration.

        ``True`` when the target is mandatory (:attr:`requires_y`) or when an
        optionally target-aware family has ``target_aware=True``.
        """
        return self.requires_y or bool(getattr(self, "target_aware", False))

    @property
    def uses_target_(self) -> bool:
        """Fitted flag: whether the last ``fit`` consumed the target ``y``.

        Available only after ``fit``. Because target-aware placement requires
        ``y`` at fit time (a supervised fit without ``y`` raises), a fitted
        supervised transformer always reports ``True``.
        """
        check_is_fitted(self, "n_features_in_")
        return self.is_supervised

    def _representation_uses_target(self) -> bool:
        """Return whether the fitted transformer consumed the target."""
        return self.is_supervised

    def _representation_cross_fitting(self) -> tuple[bool, int | None]:
        """Return ``(cross_fitted, n_folds)`` for the representation."""
        return False, None

    def _representation_degree(self) -> int | None:
        """Return the polynomial / spline degree, if the transformer has one."""
        degree = getattr(self, "degree", None)
        return None if degree is None else int(degree)

    def _representation_periodic(self) -> tuple[bool, float | None]:
        """Return ``(periodic, period)`` for the representation."""
        return False, None

    def _representation_locations(self) -> tuple[str | None, tuple[tuple[float, ...], ...] | None]:
        """Auto-detect fitted location arrays from known fitted attributes."""
        for attr, kind in self._REPRESENTATION_LOCATION_ATTRS:
            value = getattr(self, attr, None)
            if value is None:
                continue
            return kind, _as_location_tuple(value)
        return None, None

    def get_representation_spec(self, input_features=None) -> RepresentationSpec:
        """Return the typed :class:`RepresentationSpec` for this fitted transformer.

        Parameters
        ----------
        input_features : list of str or None
            Names of the input features. When ``None``, names of the form
            ``x0, x1, ...`` are generated.

        Returns
        -------
        RepresentationSpec
            The representation metadata describing the produced columns.
        """
        check_is_fitted(self, "n_features_in_")
        if input_features is None:
            inputs = tuple(f"x{i}" for i in range(self.n_features_in_))
        else:
            inputs = tuple(str(feature) for feature in input_features)
        output_features = tuple(str(name) for name in self.get_feature_names_out(list(inputs)))
        location_kind, locations = self._representation_locations()
        periodic, period = self._representation_periodic()
        cross_fitted, n_folds = self._representation_cross_fitting()
        scope = self._representation_scope
        return RepresentationSpec(
            family=self._representation_family,
            component_kind=self._representation_component_kind,
            scope=scope,
            supervision=self._representation_supervision,
            uses_target=self._representation_uses_target(),
            is_interaction=scope == "multivariate",
            input_features=inputs,
            output_features=output_features,
            output_dim=len(output_features),
            degree=self._representation_degree(),
            include_bias=bool(getattr(self, "include_bias", False)),
            periodic=periodic,
            period=period,
            local_support=self._representation_local_support,
            location_kind=location_kind,
            locations=locations,
            dtype="float64",
            cross_fitted=cross_fitted,
            n_folds=n_folds,
        )
