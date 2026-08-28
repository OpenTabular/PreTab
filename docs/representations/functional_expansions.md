# Functional expansions

Functional expansions are basis functions borrowed from machine learning rather than classical
statistics. They spread a feature across a set of activation functions (radial bumps, ReLU
ramps, sigmoids) or project it onto a deterministic Fourier basis. Together they cover local,
threshold, and periodic structure with a per-column, `Preprocessor`-selectable transformer.

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
