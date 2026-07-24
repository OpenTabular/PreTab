"""Stage 2 · Phase 0 characterization tests.

These lock the *current* numeric behaviour of the feature-map and PLE
trim/supplement helpers before the Stage 2 refactor re-expresses them in terms of
the shared ``resolve_locations`` utility. The refactor (Phase 1) must keep these
outputs bit-for-bit identical -- if any of these values change, the "no behaviour
change" guarantee has been broken.

The values below were captured from the implementation as of Phase 0:

* feature maps: ``BaseCenterExpansion._adjust_centers`` / ``_supplement_centers``
* PLE: ``PLETransformer._adjust_thresholds`` / ``_supplement_thresholds`` /
  ``_select_thresholds``
"""

import numpy as np

from pretab.transformers.feature_maps.rbf import RBFExpansionTransformer
from pretab.transformers.ple.ple import PLETransformer

# --------------------------------------------------------------------------- #
# Feature maps: center trim (over max) / supplement (under min).
# --------------------------------------------------------------------------- #


def test_fm_adjust_centers_over_max_trims_by_spacing():
    """Too many centers are down-sampled by even spacing (``select_knots``)."""
    fm = RBFExpansionTransformer()
    x = np.linspace(0.0, 1.0, 101)
    centers = np.round(np.linspace(0.0, 0.9, 10), 6)
    out = fm._adjust_centers(x, centers, 2, 5)
    np.testing.assert_allclose(out, [0.0, 0.2, 0.4, 0.7, 0.9])


def test_fm_adjust_centers_under_min_supplements():
    """Too few centers are back-filled from quantile+uniform candidates."""
    fm = RBFExpansionTransformer()
    x = np.linspace(0.0, 1.0, 101)
    out = fm._adjust_centers(x, np.array([0.5]), 4, 8)
    np.testing.assert_allclose(out, [0.0, 1 / 3, 2 / 3, 1.0])


def test_fm_supplement_centers_blends_quantile_and_uniform():
    fm = RBFExpansionTransformer()
    x = np.linspace(0.0, 1.0, 101)
    out = fm._supplement_centers(x, np.array([0.5]), 4)
    np.testing.assert_allclose(out, [0.0, 1 / 3, 2 / 3, 1.0])


def test_fm_adjust_centers_in_window_is_unchanged():
    fm = RBFExpansionTransformer()
    x = np.linspace(0.0, 1.0, 101)
    out = fm._adjust_centers(x, np.array([0.1, 0.4, 0.7]), 2, 5)
    np.testing.assert_allclose(out, [0.1, 0.4, 0.7])


# --------------------------------------------------------------------------- #
# PLE: threshold trim (over max) / supplement (under min).
# --------------------------------------------------------------------------- #


def test_ple_select_thresholds_even_spacing():
    ple = PLETransformer()
    th = np.round(np.linspace(1.0, 9.0, 9), 6)
    np.testing.assert_allclose(ple._select_thresholds(th, 3), [1.0, 5.0, 9.0])


def test_ple_adjust_thresholds_over_max_trims():
    """``max_bins=4`` caps thresholds at 3 and trims by even spacing."""
    ple = PLETransformer()
    feat = np.linspace(0.0, 10.0, 201)
    th = np.round(np.linspace(1.0, 9.0, 9), 6)
    out = ple._adjust_thresholds(feat, th, min_bins=2, max_bins=4)
    np.testing.assert_allclose(out, [1.0, 5.0, 9.0])


def test_ple_adjust_thresholds_under_min_supplements():
    """``min_bins=5`` requires 4 thresholds and back-fills from candidates."""
    ple = PLETransformer()
    feat = np.linspace(0.0, 10.0, 201)
    out = ple._adjust_thresholds(feat, np.array([5.0]), min_bins=5, max_bins=8)
    np.testing.assert_allclose(out, [2.0, 4.0, 6.0, 8.0])


def test_ple_supplement_thresholds_blends_percentile_and_linspace():
    ple = PLETransformer()
    feat = np.linspace(0.0, 10.0, 201)
    out = ple._supplement_thresholds(feat, np.array([5.0]), 4)
    np.testing.assert_allclose(out, [2.0, 4.0, 6.0, 8.0])
