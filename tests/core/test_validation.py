"""Contracts for shared target inspection, independent of estimator wiring."""

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
    ("target", "task", "observed", "suggested"),
    [
        (np.array([0.1, 1.25, 3.5]), "classification", "continuous", "regression"),
        (pd.DataFrame({"response": [0.1, 1.25, 3.5]}), "classification", "continuous", "regression"),
        (pd.Series(["private-a", "private-b", "private-a"]), "regression", "binary", "classification"),
    ],
)
@pytest.mark.parametrize(("mode", "exception"), [("warn", ConfigWarning), ("raise", InvalidParamError)])
def test_task_target_diagnoses_only_clear_mismatches(target, task, observed, suggested, mode, exception):
    check = pytest.warns if mode == "warn" else pytest.raises
    with check(exception, match=rf"(?s)Example.*placement.*task='{task}'.*'{observed}'.*task='{suggested}'") as caught:
        validation._validate_task_target(target, task=task, mode=mode, estimator="Example", role="placement")
    message = str(caught.value if isinstance(caught, pytest.ExceptionInfo) else caught[0].message)
    assert "task_validation='off'" in message
    assert "private-" not in message


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
