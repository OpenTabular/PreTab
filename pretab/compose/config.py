"""Normalized, validated configuration for a :class:`Preprocessor` run.

:class:`PreprocessorConfig` is the frozen, canonical view of the user-supplied
Preprocessor parameters. It normalizes the global method names (resolving
aliases and separator/case variants, mapping ``None`` to ``"none"``) and
validates the global ``target_aware`` / ``placement_strategy`` contract up front.
The user's original constructor arguments stay untouched on the estimator (as
scikit-learn requires); this object is the internal, normalized counterpart the
composition layer consumes.

Per-column overrides in ``feature_preprocessing`` are kept verbatim because the
namespace they resolve in (numerical vs categorical) depends on the column type,
which is only known after feature detection; :meth:`PreprocessorConfig.method_for`
resolves them in the correct namespace at build time.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.parameters import validate_placement
from ..exceptions import IncompatibleParamsError
from .registry import (
    CATEGORICAL_ALIASES,
    CATEGORICAL_METHODS,
    NUMERICAL_ALIASES,
    NUMERICAL_METHODS,
    resolve_method,
)

__all__ = ["PreprocessorConfig"]


def _normalize_method(method, canonical, aliases) -> str:
    """Resolve a global method name to canonical form, mapping ``None`` to ``"none"``."""
    if method is None:
        return "none"
    return resolve_method(method, canonical, aliases)


@dataclass(frozen=True)
class PreprocessorConfig:
    """Frozen, normalized configuration derived from Preprocessor parameters.

    Built via :meth:`from_params`, which normalizes the global method names and
    validates the placement contract. All other knobs are carried through as-is
    for the factory and orchestration layers.
    """

    numerical_method: str
    categorical_method: str
    feature_preprocessing: dict
    output_dim: int
    degree: int
    target_aware: bool
    placement_strategy: str
    task: str
    adaptive: bool
    min_output_dim: int
    max_output_dim: int
    random_state: int | None
    scaling: str | None
    cat_cutoff: float | int
    treat_all_integers_as_numerical: bool
    numerical_imputation: str | None
    categorical_imputation: str | None
    add_missing_indicator: bool
    verbose: int

    @classmethod
    def from_params(
        cls,
        *,
        numerical_method,
        categorical_method,
        feature_preprocessing,
        output_dim,
        degree,
        target_aware,
        placement_strategy,
        task,
        adaptive,
        min_output_dim,
        max_output_dim,
        random_state,
        scaling,
        cat_cutoff,
        treat_all_integers_as_numerical,
        numerical_imputation,
        categorical_imputation,
        add_missing_indicator,
        verbose,
    ) -> PreprocessorConfig:
        """Normalize and validate raw Preprocessor parameters into a config.

        Raises
        ------
        InvalidParamError
            If the ``target_aware`` / ``placement_strategy`` combination is
            invalid (via :func:`~pretab.core.parameters.validate_placement`).
        IncompatibleParamsError
            If ``add_missing_indicator`` is requested while both imputation
            strategies are disabled, since the indicator is produced by the
            imputation step.
        """
        validate_placement(target_aware, placement_strategy)
        if add_missing_indicator and numerical_imputation is None and categorical_imputation is None:
            raise IncompatibleParamsError(
                "add_missing_indicator=True requires numerical_imputation or categorical_imputation "
                "to be set; the missing-value indicator is produced by the imputation step."
            )
        return cls(
            numerical_method=_normalize_method(numerical_method, NUMERICAL_METHODS, NUMERICAL_ALIASES),
            categorical_method=_normalize_method(categorical_method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES),
            feature_preprocessing=dict(feature_preprocessing or {}),
            output_dim=output_dim,
            degree=degree,
            target_aware=target_aware,
            placement_strategy=placement_strategy,
            task=task,
            adaptive=adaptive,
            min_output_dim=min_output_dim,
            max_output_dim=max_output_dim,
            random_state=random_state,
            scaling=scaling,
            cat_cutoff=cat_cutoff,
            treat_all_integers_as_numerical=treat_all_integers_as_numerical,
            numerical_imputation=numerical_imputation,
            categorical_imputation=categorical_imputation,
            add_missing_indicator=add_missing_indicator,
            verbose=verbose,
        )

    @staticmethod
    def resolve_numerical(method) -> str:
        """Resolve a numerical method name to its canonical spelling."""
        return resolve_method(method, NUMERICAL_METHODS, NUMERICAL_ALIASES)

    @staticmethod
    def resolve_categorical(method) -> str:
        """Resolve a categorical method name to its canonical spelling."""
        return resolve_method(method, CATEGORICAL_METHODS, CATEGORICAL_ALIASES)

    def method_for(self, feature, *, is_numerical: bool) -> str:
        """Return the resolved method for ``feature`` given its detected kind.

        A per-column override in ``feature_preprocessing`` wins over the global
        default; the chosen name is resolved in the numerical or categorical
        namespace according to ``is_numerical``.
        """
        default = self.numerical_method if is_numerical else self.categorical_method
        raw = self.feature_preprocessing.get(feature, default)
        return self.resolve_numerical(raw) if is_numerical else self.resolve_categorical(raw)

    @property
    def seed_kwargs(self) -> dict:
        """Return ``{"random_state": ...}`` only when a seed was explicitly set.

        Leaving it empty when ``random_state`` is ``None`` preserves each
        transformer's own default seed, matching the historical behaviour where a
        seed is forwarded only when the user pins one.
        """
        return {} if self.random_state is None else {"random_state": self.random_state}
