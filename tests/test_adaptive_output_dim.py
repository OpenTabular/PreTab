"""Per-method audit of the ``output_dim`` / ``adaptive`` sizing contract.

These tests pin down, for every numerical preprocessing method, how many
columns a single input feature expands to:

* **Fixed mode** (``adaptive=False``) -- the per-feature width must equal
  ``output_dim`` for the width-driven families (PLE, feature maps, splines),
  and the constant natural width for the scalers / binning.
* **Adaptive mode** (``adaptive=True``) -- width must be data-driven inside
  ``[min_output_dim, max_output_dim]`` for the adaptive-capable families.

They also document, via ``xfail`` markers, the two places where the
``Preprocessor`` does *not* forward the adaptive knobs yet (all spline families
and the categorical ``custombin`` path), so those gaps stay visible and any
future fix flips the marker.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from pretab.core.exceptions import InvalidParamError
from pretab.preprocessor import Preprocessor
from pretab.transformers.splines.bspline import BSplineTransformer
from pretab.transformers.splines.integrated_spline import ISplineTransformer
from pretab.transformers.splines.knot_selectors import CARTKnotSelector
from pretab.transformers.splines.mspline import MSplineTransformer

OUTPUT_DIM = 6

# Per-feature width each method produces at ``output_dim=6`` in fixed mode.
FIXED_WIDTH = {
    # plain scalers / rank transforms / passthrough -> one column
    "standardization": 1,
    "minmax": 1,
    "quantile": 1,
    "robust": 1,
    "box-cox": 1,
    "yeo-johnson": 1,
    "none": 1,
    # polynomial degree 3 with bias -> [1, x, x^2, x^3]
    "polynomial": 4,
    # binning collapses to a single integer-coded column
    "custombin": 1,
    # width-driven expansions -> exactly output_dim
    "ple": OUTPUT_DIM,
    "rbf": OUTPUT_DIM,
    "relu": OUTPUT_DIM,
    "sigmoid": OUTPUT_DIM,
    "tanh": OUTPUT_DIM,
    "cubicspline": OUTPUT_DIM,
    "naturalspline": OUTPUT_DIM,
    "pspline": OUTPUT_DIM,
    "tensorspline": OUTPUT_DIM,
    "tprs": OUTPUT_DIM,
    "mspline": OUTPUT_DIM,
    "ispline": OUTPUT_DIM,
    # B-spline defaults to include_bias=True -> output_dim + 1
    "bspline": OUTPUT_DIM + 1,
}

# Families that honor the adaptive window when driven through the Preprocessor.
ADAPTIVE_VIA_PREPROCESSOR = ["ple", "rbf", "relu", "sigmoid", "tanh"]

# Spline families whose adaptive/min/max knobs are silently dropped by the
# Preprocessor today (transformer-level support exists; the wiring does not).
# B/M/I spline families whose adaptive window + selector choice the Preprocessor
# now forwards (Phase 5). Each has a native ``spline_type`` for the knot selector.
BMI_SPLINE_METHODS = ["bspline", "mspline", "ispline"]

# Legacy spline families whose adaptive/selector knobs the Preprocessor still
# drops -- they need the Phase 6 ``spline_type`` mapping to become target-aware.
LEGACY_SPLINE_METHODS = [
    "cubicspline",
    "naturalspline",
    "pspline",
    "tensorspline",
    "tprs",
]

# All spline families (kept for callers that want the full set).
SPLINE_METHODS = LEGACY_SPLINE_METHODS + BMI_SPLINE_METHODS


@pytest.fixture
def data():
    """A single positive numerical feature and a step-shaped regression target."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.1, 5.0, 300)
    X = pd.DataFrame({"x": x})
    y = (x > 2.5).astype(float) + rng.normal(0, 0.05, x.size)
    return X, y


def _num_width(X, y, method, **kwargs):
    """Fit a Preprocessor on one numerical feature and return its block width."""
    pre = Preprocessor(numerical_method=method, categorical_method="none", **kwargs)
    out = cast("dict[str, np.ndarray]", pre.fit(X, y).transform(X))
    return out["num_x"].shape[1]


# --------------------------------------------------------------------------- #
# Fixed mode: width must equal the declared constant for every method.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", sorted(FIXED_WIDTH))
def test_fixed_width_matches_expected(data, method):
    X, y = data
    width = _num_width(
        X, y, method,
        output_dim=OUTPUT_DIM, adaptive=False, use_target=True, task="regression",
    )
    assert width == FIXED_WIDTH[method], f"{method}: got {width}, want {FIXED_WIDTH[method]}"


@pytest.mark.parametrize("output_dim", [4, 6, 10])
def test_ple_fixed_width_tracks_output_dim(data, output_dim):
    """PLE in fixed mode produces exactly ``output_dim`` bins."""
    X, y = data
    width = _num_width(
        X, y, "ple",
        output_dim=output_dim, adaptive=False, use_target=True, task="regression",
    )
    assert width == output_dim


@pytest.mark.parametrize("output_dim", [4, 8])
def test_custombin_respects_bin_count(data, output_dim):
    """Numerical ``custombin`` yields a single column of at most output_dim codes."""
    X, y = data
    pre = Preprocessor(numerical_method="custombin", categorical_method="none", output_dim=output_dim)
    block = cast("dict[str, np.ndarray]", pre.fit(X, y).transform(X))["num_x"]
    assert block.shape[1] == 1
    assert int(block.max()) < output_dim
    assert len(np.unique(block)) <= output_dim


# --------------------------------------------------------------------------- #
# Adaptive mode via the Preprocessor: PLE + feature maps honor the window.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ADAPTIVE_VIA_PREPROCESSOR)
def test_adaptive_width_within_window(data, method):
    X, y = data
    fixed = _num_width(
        X, y, method,
        output_dim=10, adaptive=False, use_target=True, task="regression",
    )
    adaptive = _num_width(
        X, y, method,
        output_dim=10, adaptive=True, min_output_dim=3, max_output_dim=5,
        use_target=True, task="regression",
    )
    assert fixed == 10
    assert 3 <= adaptive <= 5
    assert adaptive < fixed


# --------------------------------------------------------------------------- #
# Adaptive mode at the transformer level: the B/M/I splines honor the window.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cls", "spline_type"),
    [
        (BSplineTransformer, "bspline"),
        (MSplineTransformer, "mspline"),
        (ISplineTransformer, "ispline"),
    ],
)
def test_spline_adaptive_transformer_level(data, cls, spline_type):
    X, y = data
    Xv = X.to_numpy()
    fixed = cls(output_dim=10).fit_transform(Xv, y).shape[1]
    selector = CARTKnotSelector(degree=3, spline_type=spline_type)
    adaptive = cls(
        output_dim=10, adaptive=True, min_output_dim=4, max_output_dim=5,
        selector=selector, task="regression",
    ).fit_transform(Xv, y).shape[1]
    assert adaptive < fixed
    assert adaptive <= 5 + 1  # allow the optional bias column


# --------------------------------------------------------------------------- #
# B/M/I splines honor the adaptive window + selector choice via the Preprocessor.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", BMI_SPLINE_METHODS)
def test_bmi_spline_adaptive_via_preprocessor(data, method):
    """B/M/I splines size each feature inside the adaptive window through the pipeline."""
    X, y = data
    fixed = _num_width(
        X, y, method,
        output_dim=10, adaptive=False, use_target=True, task="regression",
    )
    adaptive = _num_width(
        X, y, method,
        output_dim=10, adaptive=True, min_output_dim=4, max_output_dim=6,
        use_target=True, task="regression",
    )
    assert adaptive < fixed
    assert 4 <= adaptive <= 6 + 1  # allow the optional bias column


def test_bmi_spline_selector_choice_via_preprocessor(data):
    """The Preprocessor ``selector`` knob reaches B/M/I splines (LightGBM path)."""
    pytest.importorskip("lightgbm")
    X, y = data
    adaptive = _num_width(
        X, y, "bspline",
        output_dim=10, adaptive=True, min_output_dim=4, max_output_dim=6,
        use_target=True, task="regression", selector="lightgbm",
    )
    assert 4 <= adaptive <= 6 + 1  # allow the optional bias column


# --------------------------------------------------------------------------- #
# Known gaps -- documented as strict xfails so a future fix is flagged.
# --------------------------------------------------------------------------- #
@pytest.mark.xfail(
    reason="Preprocessor does not forward adaptive/min/max to the legacy spline "
    "families yet (Phase 6: they still need a selector spline_type mapping)",
    strict=True,
)
@pytest.mark.parametrize("method", LEGACY_SPLINE_METHODS)
def test_spline_adaptive_via_preprocessor_gap(data, method):
    X, y = data
    fixed = _num_width(
        X, y, method,
        output_dim=10, adaptive=False, use_target=True, task="regression",
    )
    adaptive = _num_width(
        X, y, method,
        output_dim=10, adaptive=True, min_output_dim=3, max_output_dim=5,
        use_target=True, task="regression",
    )
    assert adaptive < fixed


@pytest.mark.xfail(
    reason="Preprocessor does not pass output_dim to categorical custombin",
    raises=InvalidParamError,
    strict=True,
)
def test_categorical_custombin_via_preprocessor_gap():
    Xcat = pd.DataFrame({"g": np.array(["a", "b", "c"] * 100)})
    pre = Preprocessor(numerical_method="none", categorical_method="custombin", output_dim=4)
    pre.fit_transform(Xcat)
