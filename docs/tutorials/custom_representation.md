# Writing a custom representation

PreTab is registry-driven, and the registry is open. You can add your own representation, have
it validated against the same contract as the built-ins, and select it by name through
`Preprocessor`. This tutorial walks the full extension workflow using a Chebyshev polynomial
basis as the running example.

```{note}
A complete, installable version of this example lives in the repository under
`examples/pretab-chebyshev/`. Use it as a template for a standalone extension package.
```

## Subclass `BaseRepresentation`

`BaseRepresentation` gives you the shared scikit-learn contract: NaN-aware validation, estimator
tags, `get_feature_names_out`, and a typed `RepresentationSpec`. You implement `fit`,
`transform`, and one sizing hook, and declare a small amount of metadata.

```python
import numpy as np
from sklearn.utils.validation import check_is_fitted
from pretab import BaseRepresentation


class ChebyshevRepresentation(BaseRepresentation):
    """Expand each numerical feature into a Chebyshev polynomial basis."""

    representation_name = "chebyshev"
    feature_kind = "numerical"
    scope = "univariate"
    supervision = "unsupervised"

    def __init__(self, degree=5):
        self.degree = degree

    def fit(self, X, y=None):
        X = np.asarray(self._validate(X, reset=True), dtype=float)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def _rescale(self, X):
        span = self.data_max_ - self.data_min_
        span = np.where(span == 0.0, 1.0, span)
        return np.clip(2.0 * (X - self.data_min_) / span - 1.0, -1.0, 1.0)

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        z = self._rescale(np.asarray(self._validate(X, reset=False), dtype=float))
        theta = np.arccos(z)
        blocks = [
            np.column_stack([np.cos(k * theta[:, j]) for k in range(1, self.degree + 1)])
            for j in range(z.shape[1])
        ]
        return np.hstack(blocks)

    def _output_sizes(self):
        return [self.degree] * self.n_features_in_
```

The four class attributes are the declarative contract.

`representation_name`
: The name you will select it by, for example `numerical_method="chebyshev"`.

`feature_kind`
: `"numerical"` or `"categorical"`.

`scope`
: `"univariate"` (one column at a time) or `"multivariate"` (jointly).

`supervision`
: `"unsupervised"`, `"optional"` (uses `y` only when `target_aware=True`), or `"supervised"`
(always needs `y`).

```{tip}
Implement `_output_sizes` to return the number of output columns each input contributes. The
base class uses it to generate correct feature names and to power the output budget. If your
naming is bespoke, override `get_feature_names_out` directly instead.
```

## Validate against the contract

Before registering, run the conformance suite. It checks that your class is constructible with
defaults, raises `NotFittedError` before `fit`, returns deterministic output across a
`clone` and refit, and produces a `RepresentationSpec` and feature names consistent with its
declared metadata.

```python
from pretab import check_representation

check_representation(ChebyshevRepresentation)   # raises on any contract violation
```

```{important}
`check_representation` raises `RepresentationConformanceError` with a specific message when the
contract is broken. Run it in your test suite so a future change cannot silently break
compatibility.
```

## Register it

Registration adds the class to the capability registry under its name, making it selectable
through `Preprocessor` and visible to `list_representations`.

```python
from pretab import register_representation, Preprocessor

register_representation(
    "chebyshev",
    ChebyshevRepresentation,
    allowed_args=("degree",),
    supports_adaptive_resolution=False,
)

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
df = pd.DataFrame({"x": rng.uniform(-3, 3, size=500)})
y = np.cos(df["x"] * 2) + rng.normal(0, 0.1, size=500)

pre = Preprocessor(numerical_method="chebyshev", degree=8)
X2 = pre.fit_transform(df, y)
```

The `allowed_args` list tells `Preprocessor` which of its shared keyword arguments to pass
through to your constructor.

## Ship it as a plugin

To distribute your representation as an installable package, advertise it through the
`pretab.representations` entry-point group in your `pyproject.toml`.

```toml
[project.entry-points."pretab.representations"]
chebyshev = "pretab_chebyshev:ChebyshevRepresentation"
```

Users then load every installed plugin with one call.

```python
from pretab import load_entry_point_representations

load_entry_point_representations()   # discovers and registers installed plugins
```

```{note}
Discovery is opt-in and never runs automatically at import, so importing `pretab` stays fast
and predictable. A broken plugin is skipped with a warning rather than breaking discovery for
the others.
```

## Where to go next

- [Representations overview](../representations/overview.md) to see the built-in families your
  method joins.
- [Extensibility API](../api/extension.rst) for the full signatures.
- The `examples/pretab-chebyshev/` package for a complete, tested template.
