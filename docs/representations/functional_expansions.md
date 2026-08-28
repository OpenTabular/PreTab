# Functional expansions

Functional expansions are basis functions borrowed from machine learning rather than classical
statistics. They spread a feature across a set of activation functions (radial bumps, ReLU
ramps, sigmoids) or project it onto a deterministic Fourier basis. Together they cover local,
threshold, and periodic structure with a per-column, `Preprocessor`-selectable transformer.

```{important}
As with splines, `output_dim` (or `n_frequencies` for the Fourier map) is the number of output
columns **per input feature**. A `(n_samples, 3)` input with `output_dim=10` produces
`(n_samples, 30)` output.
```

## Radial basis functions

The RBF expansion places centers along the feature range and measures Gaussian similarity to
each,

$$
\phi_k(x) = \exp\!\big(-\gamma\,(x - c_k)^2\big).
$$

Each output is a smooth bump around a center, so a linear model on top can build up a curve
from local pieces.

```python
import numpy as np
from pretab.transformers import RBFExpansionTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)   # (50, 1)
t = RBFExpansionTransformer(output_dim=10, gamma=1.0)
t.fit_transform(X).shape
# (50, 10)
```

Constructor highlights: `output_dim`, `gamma=1.0` (bump width; larger is narrower),
`target_aware=False`, `placement_strategy`, `adaptive`, `random_state`.

```{tip}
`gamma` trades locality for coverage. Large `gamma` gives narrow, sharply local bumps; small
`gamma` gives broad, overlapping ones. Tune it alongside `output_dim`.
```

```{warning}
`target_aware=True` places centers using a supervised tree over `(X, y)`. Fitting it directly
on your full training set (outside a `Pipeline` or `pretab.CrossFittedTransformer`) raises a
`LeakageWarning`, because the center placement has already seen the labels you would then train
on. The same applies to ReLU, sigmoid, and tanh below whenever `target_aware=True`.
```

## ReLU, sigmoid, and tanh expansions

These place a set of thresholds along the range and apply an activation at each, mirroring a
single hidden layer.

ReLU
: Piecewise-linear ramps. Excellent for sharp, threshold-like effects.

Sigmoid and Tanh
: Smooth saturating steps. `scale` controls the steepness of the transition: **smaller** values
  give a sharper, more step-like transition; **larger** values spread it out.

```python
import numpy as np
from pretab.transformers import ReLUExpansionTransformer, TanhExpansionTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
relu = ReLUExpansionTransformer(output_dim=10)
tanh = TanhExpansionTransformer(output_dim=10, scale=1.0)
relu.fit_transform(X).shape   # (50, 10)
tanh.fit_transform(X).shape   # (50, 10)
```

```{note}
ReLU expansions are a natural fit when the effect of a feature turns on past a threshold, for
example a fee that applies only above a limit.
```

## Fourier features

The Fourier map represents a feature with sines and cosines at a set of frequencies, ideal for
signals with cyclical structure.

```python
import numpy as np
from pretab.transformers import FourierFeatureTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
t = FourierFeatureTransformer(n_frequencies=5, frequency_strategy="harmonic")
t.fit_transform(X).shape
# (50, 10): 2 columns (sin, cos) per frequency
```

Constructor highlights: `n_frequencies=5`, `frequency_strategy="harmonic"`,
`include_original=False`, `random_state`.

**Parameter impact.** `n_frequencies` sets the output width to `2 * n_frequencies` (one sine
and one cosine column per frequency); `include_original=True` adds one more column for the raw
value, giving `2 * n_frequencies + 1`. `frequency_strategy="harmonic"` uses integer multiples of
the base frequency (1x, 2x, 3x, ...); the alternative spacing is useful when the signal is not
a clean harmonic series.

```{tip}
Use `FourierFeatureTransformer` when you want the model to work across a set of frequencies
without committing to a single known period. If the period is known (hour of day, month of
year), the direct [periodic encoder](numerical_encoding.md#periodic-encoding) is usually simpler.
```

## Where to go next

- [Spline expansions](spline_expansions.md) for smooth statistical bases.
- [Kernel approximation](kernel_approximation.md) for the multivariate RFF and Nyström maps.
- [Numerical encoding](numerical_encoding.md) for binning, PLE, and periodic encoding.
- [References](references.md) for the underlying literature.
