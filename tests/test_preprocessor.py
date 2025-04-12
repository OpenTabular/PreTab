import pytest
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from pretab.preprocessor import Preprocessor  # Adjust the import as needed


@pytest.fixture
def sample_data():
    df = pd.DataFrame(
        {
            "num1": np.linspace(0, 1, 100),
            "num2": np.random.randn(100),
            "cat1": np.random.choice(["A", "B", "C"], size=100),
            "cat2": np.random.randint(0, 5, size=100),
        }
    )
    y = df["num1"] * 2 + df["num2"] + np.random.randn(100) * 0.1
    return df, y


def test_fit_transform_returns_dict(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    assert isinstance(out, dict)
    assert all(isinstance(k, str) for k in out)
    assert all(isinstance(v, np.ndarray) for v in out.values())
    assert sum(v.shape[0] for v in out.values()) == 4 * len(X)


def test_transform_array_output(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    out = pre.transform(X, return_array=True)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(X)
    assert out.ndim == 2


def test_transform_raises_before_fit(sample_data):
    X, _ = sample_data
    pre = Preprocessor()
    with pytest.raises(NotFittedError):
        _ = pre.transform(X)


def test_embedding_integration(sample_data):
    X, y = sample_data
    embed = np.random.rand(len(X), 10)
    pre = Preprocessor()
    out = pre.fit_transform(X, y, embeddings=embed)
    assert "embedding_1" in out
    assert out["embedding_1"].shape == (len(X), 10)


def test_multiple_embeddings(sample_data):
    X, y = sample_data
    embeds = [np.random.rand(len(X), 3), np.random.rand(len(X), 7)]
    pre = Preprocessor()
    out = pre.fit_transform(X, y, embeddings=embeds)
    assert "embedding_1" in out and "embedding_2" in out
    assert out["embedding_1"].shape[1] == 3
    assert out["embedding_2"].shape[1] == 7


def test_feature_info_returns_three_dicts(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    pre.fit(X, y)
    info = pre.get_feature_info(verbose=False)
    assert isinstance(info, tuple)
    assert len(info) == 3
    assert all(isinstance(d, dict) for d in info)


def test_dict_output_shapes_add_up(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    shapes = [v.shape for v in out.values()]
    assert all(s[0] == len(X) for s in shapes)


def test_dict_keys_reflect_column_names(sample_data):
    X, y = sample_data
    pre = Preprocessor()
    out = pre.fit_transform(X, y)
    expected_prefixes = ["num_", "cat_"]
    for k in out:
        if "embedding" not in k:
            assert any(k.startswith(p) for p in expected_prefixes)
