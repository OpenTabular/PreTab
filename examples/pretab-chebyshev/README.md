# pretab-chebyshev

An example, self-contained [PreTab](../../README.md) extension package. It adds a
`chebyshev` representation that expands each numerical feature into a Chebyshev
polynomial basis, and shows the complete third-party extension workflow.

This directory is a **sibling package** (it lives next to PreTab, not inside it).
In a real project it would be its own repository published to PyPI; it is kept
here only as a runnable reference.

## What it demonstrates

- Subclassing `pretab.BaseRepresentation` and declaring `representation_name`,
  `feature_kind`, `scope`, and `supervision`.
- Advertising the class through the `pretab.representations` entry-point group
  (see `pyproject.toml`) so it is auto-discoverable once installed.
- Passing the PreTab conformance suite (`pretab.check_representation`).
- Being selected by name through `Preprocessor(numerical_method="chebyshev")`.

## Install

```bash
cd examples/pretab-chebyshev
pip install -e .
```

## Use

Auto-discover every installed extension via the entry-point group:

```python
import pretab

pretab.load_entry_point_representations()   # registers "chebyshev"
"chebyshev" in pretab.list_representations(feature_kind="numerical")  # True
```

Or register the class directly, without relying on entry points:

```python
from pretab import register_representation
from pretab_chebyshev import ChebyshevRepresentation

register_representation("chebyshev", ChebyshevRepresentation, allowed_args=("degree",))
```

Then use it like any built-in method:

```python
import numpy as np, pandas as pd
from pretab import Preprocessor

X = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(-1, 1, 20)})
pre = Preprocessor(numerical_method="chebyshev", categorical_method="none", degree=4,
                   target_aware=False, placement_strategy="uniform")
out = pre.fit_transform(X, return_array=True)   # shape (20, 8)
```

## Validate

```python
from pretab import check_representation
from pretab_chebyshev import ChebyshevRepresentation

check_representation(ChebyshevRepresentation)   # raises on any contract violation
```
