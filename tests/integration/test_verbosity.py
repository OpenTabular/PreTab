"""Phase 13 -- verbosity contract and warning-category tests.

Covers the single ``verbose`` entry point on :class:`~pretab.Preprocessor`
(usable directly or forwarded by an embedding host such as DeepTab), the
``core.logging`` helpers, and the sweep of data/config warnings onto
:class:`~pretab.PretabWarning`.
"""

import contextlib
import io
import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from pretab import Preprocessor, PretabWarning, configure_logging, set_verbosity
from pretab.exceptions import ConfigWarning, DataWarning


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


def test_verbose_1_summary_reflects_per_feature_overrides(sample_data, caplog):
    """Regression guard: the fit summary must report the methods actually
    resolved per column, not just the global numerical_method/categorical_method,
    once feature_preprocessing overrides diverge from the global default."""
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(
        feature_preprocessing={"num2": "rbf"},
        numerical_method="ple",
        verbose=1,
    ).fit(X, y)
    assert "mixed: ple, rbf" in caplog.text
    assert "(ple)" not in caplog.text


def test_verbose_1_summary_stays_single_method_without_overrides(sample_data, caplog):
    """No feature_preprocessing overrides -> the summary still names the one
    global method actually used, unchanged from before the mixed-method fix."""
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=1).fit(X, y)
    assert "(ple)" in caplog.text
    assert "mixed" not in caplog.text


def test_verbose_2_logs_feature_table(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=2).fit(X, y)
    assert "fit complete" in caplog.text  # summary still emitted
    debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "feature" in debug_text  # table header
    assert "pipeline" in debug_text


def test_verbose_2_lists_a_column_without_observed_values(sample_data, caplog):
    X, y = sample_data
    X = X.assign(empty=pd.Series([np.nan] * len(X), dtype=object))
    caplog.set_level(logging.DEBUG, logger="pretab")
    with pytest.warns(DataWarning, match="no observed"):
        Preprocessor(numerical_method="ple", verbose=2).fit(X, y)
    rows = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(row.startswith("empty ") for row in rows)


def test_verbose_3_logs_internal_decisions(sample_data, caplog):
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=3).fit(X, y)
    debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    # Level 3 surfaces fitted internals (e.g. PLE thresholds / output width).
    assert "thresholds_" in debug_text or "total_output_dim_" in debug_text


@pytest.mark.parametrize("options", [{"add_missing_indicator": True}, {"missing_policy": "separate_state"}])
def test_verbose_3_logs_internal_decisions_of_missing_state_blocks(sample_data, caplog, options):
    """These blocks are a FeatureUnion of the representation and the missing
    indicator; the union itself has no fitted thresholds, so none were logged."""
    X, y = sample_data
    caplog.set_level(logging.DEBUG, logger="pretab")
    Preprocessor(numerical_method="ple", verbose=3, **options).fit(X, y)
    debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "num_num1.thresholds_" in debug_text


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


@contextlib.contextmanager
def _root_handlers(*handlers):
    """Run with exactly ``handlers`` on the root logger (none: a plain Python process).

    pytest adds its capture handlers to the root logger when a test body starts,
    so this is used inside the test body; the original handlers are restored after.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = list(handlers)
    try:
        yield
    finally:
        root.handlers = saved


def _own_handlers():
    return [h for h in logging.getLogger("pretab").handlers if not isinstance(h, logging.NullHandler)]


def test_configure_logging_attaches_stream_handler_when_none():
    logger = logging.getLogger("pretab")
    with _root_handlers():
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


def test_configure_logging_raises_the_level_on_later_calls():
    """Regression guard for issue #67: the first call pinned the verbosity."""
    with _root_handlers():
        configure_logging(1)
        configure_logging(2)
    assert logging.getLogger("pretab").level == logging.DEBUG
    assert len(_own_handlers()) == 1  # the handler is reused, never duplicated


def test_fit_verbose_2_after_verbose_1_logs_the_feature_table(sample_data, capsys):
    X, y = sample_data
    with _root_handlers():
        Preprocessor(numerical_method="ple", verbose=1).fit(X, y)
        capsys.readouterr()
        Preprocessor(numerical_method="ple", verbose=2).fit(X, y)
    err = capsys.readouterr().err
    assert "fit complete" in err
    assert "pipeline" in err


def test_get_feature_info_does_not_lower_a_debug_level(sample_data):
    X, y = sample_data
    pre = Preprocessor(numerical_method="ple").fit(X, y)
    with _root_handlers():
        configure_logging(2)
        pre.get_feature_info(verbose=True)
    assert logging.getLogger("pretab").level == logging.DEBUG


def test_configure_logging_defers_to_a_root_handler(sample_data, capsys):
    """Regression guard for issue #67: a host root handler printed every line twice."""
    X, y = sample_data
    stream = io.StringIO()  # a host root handler, as logging.basicConfig installs
    with _root_handlers(logging.StreamHandler(stream)):
        Preprocessor(numerical_method="ple", verbose=1).fit(X, y)
        assert _own_handlers() == []
    assert stream.getvalue().count("fit complete") == 1
    assert "fit complete" not in capsys.readouterr().err


def test_configure_logging_hands_over_to_a_host_configured_later():
    with _root_handlers():
        configure_logging(1)
    assert len(_own_handlers()) == 1
    with _root_handlers(logging.StreamHandler(io.StringIO())):
        configure_logging(1)
    assert _own_handlers() == []


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
