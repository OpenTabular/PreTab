"""Phase 15: the legacy count parameter names are hard-removed (``TypeError``).

Phase 15 collapses every family-specific width name (``n_basis`` /
``n_basis_functions`` / ``n_knots`` / ``n_bins`` / ``n_centers`` / ``bins`` and
their ``min_*`` / ``max_*`` variants) into a single ``output_dim`` knob. Passing
any of the removed names now raises ``TypeError`` at construction -- there is no
``FutureWarning`` grace period. The non-count aliases kept from Phase 8/9
(``knot_strategy`` / ``knot_selector`` for the knot splines, ``use_decision_tree``
for the feature maps) still resolve to their canonical name with a
``FutureWarning``.
"""

import numpy as np
import pytest

from pretab.core.exceptions import InvalidParamError
from pretab.transformers import (
    BSplineTransformer,
    CubicSplineTransformer,
    CustomBinTransformer,
    ISplineTransformer,
    MSplineTransformer,
    NaturalCubicSplineTransformer,
    PLETransformer,
    PSplineTransformer,
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
    TensorProductSplineTransformer,
    ThinPlateSplineTransformer,
)


@pytest.fixture
def X():
    rng = np.random.RandomState(0)
    return rng.uniform(-3, 3, size=(200, 1))


@pytest.fixture
def Xy():
    rng = np.random.RandomState(0)
    X = rng.uniform(0, 1, size=(200, 1))
    y = np.sin(X[:, 0]) + 0.1 * rng.randn(200)
    return X, y


# (transformer, removed constructor name) - passing any legacy count spelling
# must raise TypeError at construction (hard removal, no FutureWarning window).
REMOVED_COUNT_CASES = [
    (CubicSplineTransformer, "n_basis"),
    (CubicSplineTransformer, "n_knots"),
    (NaturalCubicSplineTransformer, "n_basis"),
    (NaturalCubicSplineTransformer, "n_knots"),
    (PSplineTransformer, "n_basis"),
    (PSplineTransformer, "n_knots"),
    (TensorProductSplineTransformer, "n_basis"),
    (TensorProductSplineTransformer, "n_knots"),
    (BSplineTransformer, "n_basis"),
    (BSplineTransformer, "n_basis_functions"),
    (MSplineTransformer, "n_basis"),
    (MSplineTransformer, "n_basis_functions"),
    (ISplineTransformer, "n_basis"),
    (ISplineTransformer, "n_basis_functions"),
    (ThinPlateSplineTransformer, "n_basis"),
    (RBFExpansionTransformer, "n_centers"),
    (RBFExpansionTransformer, "n_basis"),
    (ReLUExpansionTransformer, "n_centers"),
    (SigmoidExpansionTransformer, "n_centers"),
    (TanhExpansionTransformer, "n_centers"),
    (PLETransformer, "n_basis"),
    (PLETransformer, "n_bins"),
    (PLETransformer, "min_basis"),
    (PLETransformer, "max_basis"),
    (PLETransformer, "min_bins"),
    (PLETransformer, "max_bins"),
    (CustomBinTransformer, "n_basis"),
    (CustomBinTransformer, "bins"),
]


@pytest.mark.parametrize(("cls", "removed"), REMOVED_COUNT_CASES)
def test_removed_count_name_raises_typeerror(cls, removed):
    with pytest.raises(TypeError):
        cls(**{removed: 6})


# Every family accepts the canonical output_dim knob.
OUTPUT_DIM_CLASSES = [
    (CubicSplineTransformer, 8),
    (NaturalCubicSplineTransformer, 6),
    (PSplineTransformer, 8),
    (TensorProductSplineTransformer, 5),
    (BSplineTransformer, 8),
    (MSplineTransformer, 8),
    (ISplineTransformer, 8),
    (ThinPlateSplineTransformer, 6),
]


@pytest.mark.parametrize(("cls", "output_dim"), OUTPUT_DIM_CLASSES)
def test_output_dim_is_accepted(cls, output_dim, X):
    transformer = cls(output_dim=output_dim).fit(X)
    assert transformer.total_output_dim_ == transformer.transform(X).shape[1]


# --- retained non-count aliases still resolve with a FutureWarning ---

KNOT_SPLINES = [
    CubicSplineTransformer,
    NaturalCubicSplineTransformer,
    PSplineTransformer,
    TensorProductSplineTransformer,
    BSplineTransformer,
    MSplineTransformer,
    ISplineTransformer,
]


@pytest.mark.parametrize("cls", KNOT_SPLINES)
def test_knot_strategy_alias_emits_futurewarning(cls, X):
    with pytest.warns(FutureWarning):
        cls(output_dim=8, knot_strategy="uniform").fit(X)


@pytest.mark.parametrize("cls", KNOT_SPLINES)
def test_knot_strategy_canonical_conflict_raises(cls, X):
    with pytest.raises(InvalidParamError):
        cls(output_dim=8, strategy="uniform", knot_strategy="uniform").fit(X)


FEATURE_MAP_CLASSES = [
    RBFExpansionTransformer,
    ReLUExpansionTransformer,
    SigmoidExpansionTransformer,
    TanhExpansionTransformer,
]


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_use_decision_tree_alias_emits_futurewarning(cls, X):
    with pytest.warns(FutureWarning):
        cls(output_dim=4, use_decision_tree=False).fit(X)


@pytest.mark.parametrize("cls", FEATURE_MAP_CLASSES)
def test_use_target_canonical_conflict_raises(cls, X):
    with pytest.raises(InvalidParamError):
        cls(output_dim=4, use_target=False, use_decision_tree=False).fit(X)
