"""Unit tests for ``BasePreTabTransformer.get_feature_names_out`` input validation."""

import numpy as np
import pytest

from pretab.exceptions import InvalidParamError
from pretab.transformers import CubicRegressionSplineTransformer


def test_get_feature_names_out_rejects_wrong_length_input_features():
    """Regression guard for issue #21: a mismatched input_features length must

    raise instead of silently truncating the output.
    """
    transformer = CubicRegressionSplineTransformer(output_dim=5).fit(np.random.rand(20, 2))
    with pytest.raises(InvalidParamError, match="2 entries"):
        transformer.get_feature_names_out(["only_one_name"])


def test_get_feature_names_out_accepts_matching_length_input_features():
    transformer = CubicRegressionSplineTransformer(output_dim=5).fit(np.random.rand(20, 2))
    names = transformer.get_feature_names_out(["a", "b"])
    assert len(names) == 10
