"""Phase 13 -- verbosity contract and warning-category tests.

Covers the single ``verbose`` entry point on :class:`~pretab.Preprocessor`
(usable directly or forwarded by an embedding host such as DeepTab), the
``core.logging`` helpers, and the sweep of data/config warnings onto
:class:`~pretab.PretabWarning`.
"""

import io
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor, PretabWarning, configure_logging, set_verbosity
from pretab.exceptions import ConfigWarning


@pytest.fixture
def sample_data():
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "num1": rng.rand(60),
            "num2": rng.rand(60) * 10,
            "cat1": rng.choice(["a", "b", "c"], size=60),
        }
    )
    y = df["num1"] * 2 + df["num2"] * 0.1
    return df, y


@pytest.fixture(autouse=True)
def _reset_pretab_logger():
    """Isolate every test from the process-wide ``"pretab"`` logger state."""
    logger = logging.getLogger("pretab")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers = [logging.NullHandler()]
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    try:
        yield
    finally:
        logger.handlers = saved_handlers
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


# --------------------------------------------------------------------------- #
# Preprocessor.fit verbosity levels
# --------------------------------------------------------------------------- #
def test_default_fit_is_silent(sample_data, capsys):
    X, y = sample_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Preprocessor(numerical_method="ple").fit(X, y).transform(X)
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""
    # PreTab must not attach a real handler when running silently.
    logger = logging.getLogger("pretab")
    assert all(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_verbose_1_logs_fit_summary(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=1).fit(X, y)
    assert "fit complete" in caplog.text
    # Level 1 stays at the one-line summary -- no DEBUG per-feature table.
    assert [r for r in caplog.records if r.levelno == logging.DEBUG] == []


def test_verbose_2_logs_feature_table(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=2).fit(X, y)
    assert "fit complete" in caplog.text  # summary still emitted
    debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "feature" in debug_text  # table header
    assert "pipeline" in debug_text


def test_verbose_3_logs_internal_decisions(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=3).fit(X, y)
    debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    # Level 3 surfaces fitted internals (e.g. PLE thresholds / output width).
    assert "thresholds_" in debug_text or "total_output_dim_" in debug_text


def test_verbose_true_behaves_like_level_1(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=True).fit(X, y)
    assert "fit complete" in caplog.text
    assert [r for r in caplog.records if r.levelno == logging.DEBUG] == []


def test_verbose_false_is_silent(sample_data, capsys):
    X, y = sample_data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Preprocessor(numerical_method="ple", verbose=False).fit(X, y)
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_verbose_survives_get_params_and_clone(sample_data):
    from sklearn.base import clone

    pre = Preprocessor(numerical_method="ple", verbose=2)
    assert pre.get_params()["verbose"] == 2
    cloned = clone(pre)
    assert isinstance(cloned, Preprocessor)
    assert cloned.get_params()["verbose"] == 2


# --------------------------------------------------------------------------- #
# get_feature_info rendering
# --------------------------------------------------------------------------- #
def test_get_feature_info_returns_dicts_silently(sample_data, capsys):
    X, y = sample_data
    pre = Preprocessor(numerical_method="ple").fit(X, y)
    capsys.readouterr()  # drop anything from fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = pre.get_feature_info(verbose=False)
    out = capsys.readouterr()
    assert out.out == ""
    assert "pipeline" not in out.err
    assert isinstance(info, tuple) and len(info) == 3
    assert all(isinstance(d, dict) for d in info)


def test_get_feature_info_logs_table_when_verbose(sample_data, caplog):
    X, y = sample_data
    pre = Preprocessor(numerical_method="ple").fit(X, y)
    caplog.clear()
    caplog.set_level(logging.INFO, logger="pretab")
    pre.get_feature_info(verbose=True)
    assert "pipeline" in caplog.text


# --------------------------------------------------------------------------- #
# core.logging helpers
# --------------------------------------------------------------------------- #
def test_set_verbosity_sets_logger_level():
    logger = logging.getLogger("pretab")
    set_verbosity(2)
    assert logger.level == logging.DEBUG
    set_verbosity(0)
    assert logger.level == logging.WARNING
    set_verbosity(1)
    assert logger.level == logging.INFO


def test_configure_logging_attaches_stream_handler_when_none():
    logger = logging.getLogger("pretab")
    configure_logging(1)
    assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler) for h in logger.handlers)
    assert logger.level == logging.INFO


def test_configure_logging_respects_existing_handler():
    logger = logging.getLogger("pretab")
    host_handler = logging.StreamHandler(io.StringIO())
    logger.addHandler(host_handler)
    logger.setLevel(logging.CRITICAL)
    before = logger.handlers[:]
    configure_logging(2)
    # A host that already owns a handler wins: no new handler, level untouched.
    assert logger.handlers == before
    assert logger.level == logging.CRITICAL


# --------------------------------------------------------------------------- #
# warning categories (PretabWarning family)
# --------------------------------------------------------------------------- #
def test_output_dim_clamp_warns_config_warning(sample_data):
    X, y = sample_data
    with pytest.warns(ConfigWarning):
        Preprocessor(numerical_method="bspline", output_dim=100).fit(X, y)


def test_config_warning_is_a_pretab_warning(sample_data):
    X, y = sample_data
    with pytest.warns(PretabWarning):
        Preprocessor(numerical_method="bspline", output_dim=100).fit(X, y)
