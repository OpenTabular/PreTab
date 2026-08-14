# Feature maps

Feature maps are basis functions borrowed from machine learning rather than classical
statistics. They spread a feature across a set of activation functions (radial bumps, ReLU
ramps, sigmoids) or project it onto a Fourier basis, and they include the two standard
kernel approximations. Together they cover local, threshold, and periodic structure.

## Radial basis functions

The RBF expansion places centers along the feature range and measures Gaussian similarity to
each,

$$
\phi_k(x) = \exp\!\big(-\gamma\,(x - c_k)^2\big).
$$

Each output is a smooth bump around a center, so a linear model on top can build up a curve
from local pieces.

```python
from pretab.transformers import RBFExpansionTransformer

t = RBFExpansionTransformer(output_dim=10, gamma=1.0)
```

Constructor highlights: `output_dim`, `gamma=1.0` (bump width; larger is narrower),
`target_aware=False`, `placement_strategy`, `adaptive`, `random_state`.

```{tip}
`gamma` trades locality for coverage. Large `gamma` gives narrow, sharply local bumps; small
`gamma` gives broad, overlapping ones. Tune it alongside `output_dim`.
```

## ReLU, sigmoid, and tanh expansions

These place a set of thresholds along the range and apply an activation at each, mirroring a
single hidden layer.

ReLU
: Piecewise-linear ramps. Excellent for sharp, threshold-like effects.

Sigmoid and Tanh
: Smooth saturating steps. `scale` controls the steepness of the transition.

```python
from pretab.transformers import ReLUExpansionTransformer, TanhExpansionTransformer

relu = ReLUExpansionTransformer(output_dim=10)
tanh = TanhExpansionTransformer(output_dim=10, scale=1.0)
```

```{note}
ReLU expansions are a natural fit when the effect of a feature turns on past a threshold, for
example a fee that applies only above a limit.
```

## Fourier features

The Fourier map represents a feature with sines and cosines at a set of frequencies, ideal for
signals with cyclical structure.

```python
from pretab.transformers import FourierFeatureTransformer

t = FourierFeatureTransformer(n_frequencies=5, frequency_strategy="harmonic")
```

Constructor highlights: `n_frequencies=5`, `frequency_strategy="harmonic"`,
`include_original=False`, `random_state`.

### Periodic encoding

When you know the period, the periodic encoder is the direct choice. It maps a value onto its
position in a cycle of known length, so December and January sit next to each other.

```python
from pretab.transformers import PeriodicEncodingTransformer

t = PeriodicEncodingTransformer(period=12, harmonics=2)  # e.g. month of year
```

```{tip}
Use `PeriodicEncodingTransformer` when the period is known (hour of day, month of year). Use
`FourierFeatureTransformer` when you want the model to work across a set of frequencies.
```

## Kernel approximations

Two multivariate maps approximate a kernel machine without forming the full kernel matrix.
They are standalone transformers, not per-column methods.

### Random Fourier features

Approximates a shift-invariant kernel (by default the RBF kernel) with random projections,
following Rahimi and Recht. This makes kernel-style models scale to large datasets.

```python
from pretab.transformers import RandomFourierFeaturesTransformer

t = RandomFourierFeaturesTransformer(n_components=100, gamma=1.0)
X2 = t.fit_transform(X)
```

### Nyström

Approximates a kernel by sampling landmark points and projecting onto them, following Williams
and Seeger. It supports several kernels through `kernel`.

```python
from pretab.transformers import NystroemFeaturesTransformer

t = NystroemFeaturesTransformer(n_components=100, kernel="rbf")
X2 = t.fit_transform(X)
```

Constructor highlights: `n_components=100`, `kernel="rbf"`, `gamma=None`, `degree=3`,
`coef0=1`, `random_state`.

```{warning}
Random Fourier features and Nyström are multivariate and operate on the whole input matrix.
They are not available as a per-column `numerical_method`; fit them standalone.
```

## Where to go next

- [Splines](splines.md) for smooth statistical bases.
- [Binning and PLE](binning_and_ple.md) for discretization.
- [References](references.md) for the kernel-approximation literature.
