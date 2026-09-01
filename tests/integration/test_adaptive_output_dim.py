"""Per-method audit of the ``output_dim`` / ``adaptive`` sizing contract.

These tests pin down, for every numerical preprocessing method, how many
columns a single input feature expands to:

* **Fixed mode** (``adaptive=False``) -- the per-feature width must equal
  ``output_dim`` for the width-driven families (PLE, feature maps, splines),
  and the constant natural width for the scalers / binning.
* **Adaptive mode** (``adaptive=True``) -- width must be data-driven inside
  ``[min_output_dim, max_output_dim]`` for the adaptive-capable families.

``custombin`` is numeric-only: it is selectable as a numerical method (and
forwards ``output_dim``), while selecting it as a categorical method is rejected.
"""

from typing import cast

import numpy as np
import pandas as pd
import pytest

from pretab.exceptions import InvalidParamError
from pretab.expansion.spline.b_spline import BSplineTransformer
from pretab.expansion.spline.i_spline import ISplineTransformer
from pretab.expansion.spline.m_spline import MSplineTransformer
from pretab.preprocessor import Preprocessor

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
    # deterministic Fourier map: 2 * default n_frequencies (5) sine/cosine columns
    "fourier": 10,
    # width-driven expansions -> exactly output_dim
    "ple": OUTPUT_DIM,
    "rbf": OUTPUT_DIM,
    "relu": OUTPUT_DIM,
    "sigmoid": OUTPUT_DIM,
    "tanh": OUTPUT_DIM,
    "cubicspline": OUTPUT_DIM,
    "naturalspline": OUTPUT_DIM,
    "pspline": OUTPUT_DIM,
    "mspline": OUTPUT_DIM,
    "ispline": OUTPUT_DIM,
    # B-spline: include_bias defaults to False (a bias column would be collinear
    # with the partition-of-unity basis), so width == output_dim like its siblings
    "bspline": OUTPUT_DIM,
}

# Families that honor the adaptive window when driven through the Preprocessor.
ADAPTIVE_VIA_PREPROCESSOR = ["ple", "rbf", "relu", "sigmoid", "tanh"]

# B/M/I spline families whose adaptive window + selector choice the Preprocessor
# forwards (Phase 5). Each has a native ``spline_type`` for the knot selector.
BMI_SPLINE_METHODS = ["bspline", "mspline", "ispline"]

# Legacy freely-placed knot splines the Preprocessor also routes through the
# selector + adaptive window (Phase 6), via the ``"bspline"`` knot mapping.
TARGET_AWARE_LEGACY_SPLINE_METHODS = [
    "cubicspline",
    "naturalspline",
]

# Fixed-only spline families: target-aware placement does not apply. The
# penalized ``pspline`` assumes equally-spaced knots for its difference penalty,
# so it stays fixed-width regardless of the adaptive / selector knobs. (The
# multivariate ``tensorspline`` / ``tprs`` are standalone-only and not selectable
# through the Preprocessor, so they are exercised in their own transformer tests.)
FIXED_ONLY_SPLINE_METHODS = ["pspline"]

# All spline families (kept for callers that want the full set).
SPLINE_METHODS = TARGET_AWARE_LEGACY_SPLINE_METHODS + FIXED_ONLY_SPLINE_METHODS + BMI_SPLINE_METHODS


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
        X,
        y,
        method,
        output_dim=OUTPUT_DIM,
        adaptive=False,
        target_aware=True,
        task="regression",
    )
    assert width == FIXED_WIDTH[method], f"{method}: got {width}, want {FIXED_WIDTH[method]}"


@pytest.mark.parametrize("output_dim", [4, 6, 10])
def test_ple_fixed_width_tracks_output_dim(data, output_dim):
    """PLE in fixed mode produces exactly ``output_dim`` bins."""
    X, y = data
    width = _num_width(
        X,
        y,
        "ple",
        output_dim=output_dim,
        adaptive=False,
        target_aware=True,
        task="regression",
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
        X,
        y,
        method,
        output_dim=10,
        adaptive=False,
        target_aware=True,
        task="regression",
    )
    adaptive = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=True,
        min_output_dim=3,
        max_output_dim=5,
        target_aware=True,
        task="regression",
    )
    assert fixed == 10
    assert 3 <= adaptive <= 5
    assert adaptive < fixed


# --------------------------------------------------------------------------- #
# Adaptive mode at the transformer level: the B/M/I splines honor the window.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cls",
    [BSplineTransformer, MSplineTransformer, ISplineTransformer],
)
def test_spline_adaptive_transformer_level(data, cls):
    X, y = data
    Xv = X.to_numpy()
    fixed = cls(output_dim=10).fit_transform(Xv, y).shape[1]
    adaptive = (
        cls(
            output_dim=10,
            adaptive=True,
            min_output_dim=4,
            max_output_dim=5,
            target_aware=True,
            placement_strategy="cart",
            task="regression",
        )
        .fit_transform(Xv, y)
        .shape[1]
    )
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
        X,
        y,
        method,
        output_dim=10,
        adaptive=False,
        target_aware=True,
        task="regression",
    )
    adaptive = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=True,
        min_output_dim=4,
        max_output_dim=6,
        target_aware=True,
        task="regression",
    )
    assert adaptive < fixed
    assert 4 <= adaptive <= 6 + 1  # allow the optional bias column


def test_bmi_spline_selector_choice_via_preprocessor(data):
    """The Preprocessor ``selector`` knob reaches B/M/I splines (LightGBM path)."""
    pytest.importorskip("lightgbm")
    X, y = data
    adaptive = _num_width(
        X,
        y,
        "bspline",
        output_dim=10,
        adaptive=True,
        min_output_dim=4,
        max_output_dim=6,
        target_aware=True,
        task="regression",
        placement_strategy="lightgbm",
    )
    assert 4 <= adaptive <= 6 + 1  # allow the optional bias column


# --------------------------------------------------------------------------- #
# Legacy knot splines honor the adaptive window via the Preprocessor (Phase 6).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", TARGET_AWARE_LEGACY_SPLINE_METHODS)
def test_legacy_spline_adaptive_via_preprocessor(data, method):
    """Legacy knot splines size each feature inside the adaptive window through the pipeline."""
    X, y = data
    fixed = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=False,
        target_aware=True,
        task="regression",
    )
    adaptive = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=True,
        min_output_dim=4,
        max_output_dim=6,
        target_aware=True,
        task="regression",
    )
    assert adaptive < fixed
    assert 4 <= adaptive <= 6


@pytest.mark.parametrize("method", FIXED_ONLY_SPLINE_METHODS)
def test_fixed_only_spline_ignores_adaptive(data, method):
    """Penalized splines are not target-aware: the adaptive window is a no-op.

    ``pspline`` needs equally-spaced knots for its difference penalty, so the
    Preprocessor never routes it through the selector / adaptive path -- the width
    stays ``output_dim``.
    """
    X, y = data
    fixed = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=False,
        target_aware=True,
        task="regression",
    )
    adaptive = _num_width(
        X,
        y,
        method,
        output_dim=10,
        adaptive=True,
        min_output_dim=4,
        max_output_dim=6,
        target_aware=True,
        task="regression",
    )
    assert fixed == adaptive == 10


# --------------------------------------------------------------------------- #
# Categorical custombin: numeric-only, no longer selectable on the categorical side.
# --------------------------------------------------------------------------- #
def test_categorical_custombin_no_longer_selectable():
    """custombin is numeric-only: selecting it as a categorical method is rejected."""
    # Integer codes with few unique values -> detected as categorical (ratio < cat_cutoff).
    Xcat = pd.DataFrame({"g": np.array([0, 1, 2, 3, 4, 5] * 50)})
    pre = Preprocessor(numerical_method="none", categorical_method="custombin", output_dim=4)
    with pytest.raises(InvalidParamError):
        pre.fit_transform(Xcat)
