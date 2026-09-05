# Installation

pretab supports Python 3.10 to 3.13, with the following minimum core dependency versions:

| Dependency | Minimum |
| --- | --- |
| `numpy` | 1.24 |
| `pandas` | 2.0 |
| `scipy` | 1.10 |
| `scikit-learn` | 1.6 |

```{note}
`scikit-learn>=1.6` is required for the `__sklearn_tags__` tag-dispatch API pretab's
transformers use. A dedicated CI job installs exactly these minimum versions and runs the
test suite against them, so this floor is verified, not just declared.
```

## From PyPI

```bash
pip install pretab
```

### Optional extras

Language-embedding features depend on
[`sentence-transformers`](https://www.sbert.net/). Install them with the `embeddings`
extra (or the convenience `all` extra):

```bash
pip install "pretab[embeddings]"
```

```{note}
The `embeddings` extra installs `sentence-transformers` and its deep-learning
dependencies (including PyTorch), so it is a sizeable download. Add it only if you plan
to use the `pretrained` categorical strategy.
```

The `lightgbm` extra enables the gradient-boosted `placement_strategy="lightgbm"` for
supervised knot, center, and threshold selection:

```bash
pip install "pretab[lightgbm]"
```

Use the convenience `all` extra to install every optional dependency at once:

```bash
pip install "pretab[all]"
```

## From source

pretab uses [Poetry](https://python-poetry.org/) for dependency management and
[just](https://just.systems/) as a command runner.

```bash
git clone https://github.com/OpenTabular/PreTab
cd PreTab
just install
```

Without `just`, run the equivalent steps directly:

```bash
poetry install
poetry run pre-commit install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push
```

To check that everything works end to end, run the quickstart script. It exercises mixed
preprocessing, feature lineage, leakage-safe cross-fitting, serialization, and more in a few
seconds, and doubles as the reviewer smoke test:

```bash
just quickstart   # or: python scripts/quickstart.py
```

To work on the documentation, also install the docs group:

```bash
poetry install --with docs
```

## Verify the installation

```python
import pretab

print(pretab.__version__)
```
