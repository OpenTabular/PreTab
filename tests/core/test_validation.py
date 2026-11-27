"""Contracts for shared target inspection, independent of estimator wiring."""

import warnings

import numpy as np
import pandas as pd
import pytest

from pretab.core import validation
from pretab.exceptions import ConfigWarning, InvalidParamError


@pytest.mark.parametrize("mode", ["off", "warn", "raise"])
def test_task_validation_accepts_exact_modes(mode):
    validation._validate_task_validation(mode, "Example")


@pytest.mark.parametrize("mode", ["WARN", " off", "", None, True, False, 1, ["warn"], np.array(["warn"])])
def test_task_validation_rejects_invalid_modes(mode):
    with pytest.raises(InvalidParamError, match=r"(?s)Example\.task_validation.*'off', 'warn', or 'raise'"):
        validation._validate_task_validation(mode, "Example")


@pytest.mark.parametrize(
    ("values", "inconsistent_task", "observed"),
    [
        pytest.param([0.1, 1.25, 3.5], "classification", "continuous", id="continuous"),
        pytest.param([[0.1], [1.25], [3.5]], "classification", "continuous", id="column-vector"),
        pytest.param(["private-a", "private-b", "private-a"], "regression", "binary", id="binary-text"),
        pytest.param(["private-a", "private-b", "private-c"], "regression", "multiclass", id="multiclass-text"),
        pytest.param([0, 1, 0], None, None, id="binary-integer"),
        pytest.param([0, 1, 2], None, None, id="multiclass-integer"),
        pytest.param([True, False, True], None, None, id="boolean"),
        pytest.param([0.0, 1.0, 0.0], None, None, id="binary-integral-float"),
        pytest.param([0.0, 1.0, 2.0], None, None, id="multiclass-integral-float"),
        pytest.param(["0.5", "1e3", "-2"], None, None, id="numeric-text"),
        pytest.param([[0, 1], [1, 0], [1, 1]], None, None, id="multilabel"),
        pytest.param([[0.1, 1.1], [1.2, 2.3], [4.5, 0.3]], None, None, id="continuous-multioutput"),
        pytest.param([0.1, np.nan, 3.5], None, None, id="missing"),
        pytest.param(np.array([0, "private-a", 2], dtype=object), None, None, id="mixed-object"),
    ],
)
@pytest.mark.parametrize("container", ["numpy", "pandas"])
@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("mode", ["off", "warn", "raise"])
def test_task_target_policy_matrix(values, inconsistent_task, observed, container, task, mode):
    array = np.asarray(values)
    target = (pd.Series(array) if array.ndim == 1 else pd.DataFrame(array)) if container == "pandas" else array
    original = np.asarray(target).copy()

    def inspect_target():
        validation._validate_task_target(target, task=task, mode=mode, estimator="Example", role="placement")

    if mode == "off" or task != inconsistent_task:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            inspect_target()
    else:
        suggested = "regression" if task == "classification" else "classification"
        match = rf"(?s)Example.*placement.*task='{task}'.*'{observed}'.*task='{suggested}'"
        if mode == "warn":
            with pytest.warns(ConfigWarning, match=match) as recorded:
                inspect_target()
            assert len(recorded) == 1
            message = str(recorded[0].message)
        else:
            with pytest.raises(InvalidParamError, match=match) as caught:
                inspect_target()
            message = str(caught.value)
        assert "task_validation='off'" in message
        assert "private-" not in message

    np.testing.assert_array_equal(np.asarray(target), original)


@pytest.mark.parametrize("target", [[0, 1, 2], [True, False], [0.0, 1.0, 2.0], ["0.5", "1e3", "-2"]])
def test_task_target_leaves_ambiguous_regression_quiet(target):
    validation._validate_task_target(target, task="regression", mode="raise", estimator="Example", role="placement")


@pytest.mark.parametrize(
    ("target", "mode"),
    [
        (np.array([0.1, 1.25, 3.5]), "off"),
        (None, "raise"),
        (np.array([0.5, np.nan]), "raise"),
        (np.array([0.5, np.inf]), "raise"),
        (np.array([0.5, 1e30]), "raise"),
        ([int(np.iinfo(np.int64).max), 0.5], "raise"),
        (pd.Series([0.5, pd.NA], dtype="Float64"), "raise"),
        (pd.DataFrame({"a": [1, 0], "b": [0, 1]}), "raise"),
    ],
)
def test_task_target_skips_indeterminate_inputs_before_sklearn(target, mode, monkeypatch):
    def unexpected_inspection(target):
        pytest.fail("target should not be sent to sklearn")

    monkeypatch.setattr(validation, "type_of_target", unexpected_inspection)
    validation._validate_task_target(target, task="classification", mode=mode, estimator="Example", role="placement")
