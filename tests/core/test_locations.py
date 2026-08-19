"""Unit tests for the shared ``resolve_locations`` / ``trim_to_count`` utility.

These cover the generic skeleton directly (the per-family characterization lives
in ``test_selection_characterization.py``), including the importance-ranked trim
path that the Stage 2 selectors will use.
"""

import numpy as np

from pretab.core.locations import resolve_locations, trim_to_count


def test_trim_to_count_noop_when_within_count():
    locs = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(trim_to_count(locs, 5), locs)


def test_trim_to_count_by_even_spacing():
    locs = np.arange(10.0)
    # select_knots picks indices linspace(0, 9, 5).round() = [0, 2, 4, 7, 9].
    np.testing.assert_array_equal(trim_to_count(locs, 5), [0.0, 2.0, 4.0, 7.0, 9.0])


def test_trim_to_count_by_importance_keeps_top_in_location_order():
    locs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    importance = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    # Top two by importance are indices 1 and 3; returned in ascending location.
    np.testing.assert_array_equal(trim_to_count(locs, 2, importance), [1.0, 3.0])


def test_resolve_in_window_passthrough():
    locs = np.array([0.1, 0.4, 0.7])
    out = resolve_locations(locs, min_count=2, max_count=5)
    np.testing.assert_array_equal(out, [0.1, 0.4, 0.7])


def test_resolve_over_max_trims_by_spacing():
    locs = np.arange(10.0)
    out = resolve_locations(locs, min_count=1, max_count=5)
    np.testing.assert_array_equal(out, [0.0, 2.0, 4.0, 7.0, 9.0])


def test_resolve_over_max_trims_by_importance():
    locs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    importance = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    out = resolve_locations(locs, min_count=1, max_count=2, importance=importance)
    np.testing.assert_array_equal(out, [1.0, 3.0])


def test_resolve_under_min_calls_supplement():
    calls = {}

    def supplement(current, target):
        calls["args"] = (current.copy(), target)
        return np.array([1.0, 2.0, 3.0])

    out = resolve_locations(np.array([5.0]), min_count=3, max_count=8, supplement=supplement)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])
    assert calls["args"][1] == 3


def test_resolve_dedupe_toggle():
    locs = np.array([2.0, 1.0, 1.0, 3.0])
    deduped = resolve_locations(locs, min_count=0, max_count=10, dedupe=True)
    kept = resolve_locations(locs, min_count=0, max_count=10, dedupe=False)
    np.testing.assert_array_equal(deduped, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(kept, [1.0, 1.0, 2.0, 3.0])


def test_resolve_importance_stays_aligned_after_sort():
    """Regression guard for issue #21: importance must track its own location,

    not the position it happened to occupy in the caller's unsorted input.
    """
    locs = np.array([5.0, 1.0, 3.0])
    importance = np.array([0.1, 9.9, 0.2])  # location 1.0 is by far the most important
    out = resolve_locations(locs, min_count=1, max_count=1, importance=importance)
    np.testing.assert_array_equal(out, [1.0])
