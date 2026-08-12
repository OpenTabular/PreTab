"""Unit tests for :mod:`pretab.compose.factory`."""

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer

from pretab.compose.factory import (
    _placement_kwargs,
    build_column_transformer,
    get_categorical_transformer_steps,
    get_numerical_transformer_steps,
)
from pretab.compose.registry import get_spec
from pretab.exceptions import ConfigWarning, InvalidParamError


def _names(steps):
    return [name for name, _ in steps]


# --------------------------------------------------------------------------- #
# numerical step assembly
# --------------------------------------------------------------------------- #
def test_imputer_is_first_step_by_default():
    assert _names(get_numerical_transformer_steps("standardization"))[0] == "imputer"


def test_imputer_omitted_when_disabled():
    assert "imputer" not in _names(get_numerical_transformer_steps("standardization", add_imputer=False))


def test_none_method_uses_noop_step():
    assert _names(get_numerical_transformer_steps("none", add_imputer=False)) == ["noop"]


def test_box_cox_scales_positive_first():
    assert _names(get_numerical_transformer_steps("box-cox", add_imputer=False)) == ["scale_positive", "boxcox"]


def test_scaling_injected_only_when_different_from_method():
    with_scaler = _names(get_numerical_transformer_steps("ple", add_imputer=False, scaling="standardization"))
    assert "scaler" in with_scaler
    same = _names(get_numerical_transformer_steps("standardization", add_imputer=False, scaling="standardization"))
    assert "scaler" not in same
    assert same.count("standardization") == 1


def test_bmi_spline_output_dim_is_clamped_with_warning():
    with pytest.warns(ConfigWarning):
        get_numerical_transformer_steps("bspline", add_imputer=False, output_dim=100)


def test_unknown_numerical_method_raises():
    with pytest.raises(InvalidParamError):
        get_numerical_transformer_steps("does-not-exist", add_imputer=False)


# --------------------------------------------------------------------------- #
# categorical step assembly
# --------------------------------------------------------------------------- #
def test_one_hot_appends_to_float():
    assert _names(get_categorical_transformer_steps("one-hot", add_imputer=False)) == ["onehot", "to_float"]


def test_int_uses_continuous_ordinal():
    assert _names(get_categorical_transformer_steps("int", add_imputer=False)) == ["continuous_ordinal"]


def test_unknown_categorical_method_raises():
    with pytest.raises(InvalidParamError):
        get_categorical_transformer_steps("does-not-exist", add_imputer=False)


# --------------------------------------------------------------------------- #
# placement kwargs by capability class
# --------------------------------------------------------------------------- #
def test_placement_optional_forwards_target_aware_and_strategy():
    spec = get_spec("rbf")  # target_usage == optional
    assert _placement_kwargs(spec, {"target_aware": False}) == {"target_aware": False}
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "quantile"}) == {
        "target_aware": False,
        "placement_strategy": "quantile",
    }


def test_placement_required_only_when_target_aware_supervised():
    spec = get_spec("ple")  # target_usage == required
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "cart"}) == {
        "placement_strategy": "cart"
    }
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "cart"}) == {}


def test_placement_forbidden_uses_unsupervised_only():
    spec = get_spec("pspline")  # target_usage == forbidden, unsupervised placement
    assert _placement_kwargs(spec, {"target_aware": False, "placement_strategy": "uniform"}) == {
        "placement_strategy": "uniform"
    }
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "uniform"}) == {}


def test_placement_absent_when_method_has_no_strategies():
    spec = get_spec("standardization")  # no placement strategies
    assert _placement_kwargs(spec, {"target_aware": True, "placement_strategy": "cart"}) == {}


# --------------------------------------------------------------------------- #
# ColumnTransformer assembly
# --------------------------------------------------------------------------- #
def test_build_column_transformer_prefixes_and_passthrough(make_config):
    ct = build_column_transformer(make_config(), ["age"], ["city"])
    assert isinstance(ct, ColumnTransformer)
    assert [name for name, _, _ in ct.transformers] == ["num_age", "cat_city"]
    assert ct.remainder == "passthrough"


def test_build_column_transformer_fits_and_transforms(make_config, sample_frame):
    ct = build_column_transformer(make_config(), ["age"], ["city"])
    out = ct.fit_transform(sample_frame, np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(sample_frame)
