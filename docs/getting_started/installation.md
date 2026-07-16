# Installation

pretab supports Python 3.10 – 3.13.

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

To work on the documentation, also install the docs group:

```bash
poetry install --with docs
```

## Verify the installation

```python
import pretab

print(pretab.__version__)
```
