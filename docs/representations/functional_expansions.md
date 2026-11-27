# Functional expansions

Functional expansions are basis functions borrowed from machine learning rather than classical
statistics. They spread a feature across a set of activation functions (radial bumps, ReLU
ramps, sigmoids) or project it onto a deterministic Fourier basis. Together they cover local,
threshold, and periodic structure with a per-column, `Preprocessor`-selectable transformer.

```{important}
For fixed-width RBF, ReLU, sigmoid, and tanh expansions, `output_dim` counts output columns
**per input feature**: three inputs with `output_dim=10` produce 30 columns. Fourier counts
frequencies instead: three inputs with `n_frequencies=10` produce 60 sine and cosine columns,
or 63 columns when `include_original=True`.
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
`target_aware=True` places centers using a supervised tree over `(X, y)`. A standalone fit with
`y` emits `LeakageWarning`. For evaluation, split before fitting or cross-validate the whole
pipeline. A plain pipeline still uses the training rows' targets to place their centers;
wrap the expansion in `pretab.CrossFittedTransformer` for out-of-fold training features.
This also applies to ReLU, sigmoid, and tanh whenever `target_aware=True`. See
[Target awareness](../core_concepts/target_awareness.md).
```

## ReLU, sigmoid, and tanh expansions

These place a set of thresholds along the range and apply an activation at each, mirroring a
single hidden layer. For a feature $x$ and center $c_k$ (with `scale` $s$ for sigmoid and tanh),

$$
\text{ReLU: } \phi_k(x) = \max(0,\ x - c_k), \qquad
\text{Sigmoid: } \phi_k(x) = \frac{1}{1 + \exp\!\big(-(x - c_k)/s\big)}, \qquad
\text{Tanh: } \phi_k(x) = \tanh\!\big((x - c_k)/s\big).
$$

ReLU
: Piecewise-linear ramps. Excellent for sharp, threshold-like effects. With unsupervised
placement the centers are the first `output_dim` of `output_dim + 1` range-spanning points: the
first ramp starts at the training minimum, and the right endpoint is dropped because a ramp
starting at the training maximum is zero on every training row.

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
signals with cyclical structure. For a feature $x$ with fitted origin $x_0$ (the observed
minimum) and angular frequency $\omega_k$,

$$
\phi_k(x) = \big(\sin(\omega_k (x - x_0)),\ \cos(\omega_k (x - x_0))\big).
$$

The fundamental frequency is set from the feature's observed range at fit time
($2\pi / \text{range}$), and `frequency_strategy` controls how the $\omega_k$ are spread above
it: `"harmonic"` uses integer multiples $k \cdot \omega_1$; `"log_spaced"` uses octaves
$2^{k-1} \cdot \omega_1$; `"random"` draws frequencies from a half-normal distribution scaled by
$\omega_1$.

```python
import numpy as np
from pretab.transformers import FourierFeatureTransformer

X = np.linspace(0, 1, 50).reshape(-1, 1)
t = FourierFeatureTransformer(n_frequencies=5, frequency_strategy="harmonic")
encoded = t.fit_transform(X)
assert encoded.shape == (50, 10)
np.testing.assert_allclose(encoded[0], encoded[-1], atol=1e-12)
```

Constructor highlights: `n_frequencies=5`, `frequency_strategy="harmonic"`,
`include_original=False`, `random_state`.

**Parameter impact.** `n_frequencies` sets the output width to `2 * n_frequencies` (one sine
and one cosine column per frequency); `include_original=True` adds one more column for the raw
value, giving `2 * n_frequencies + 1`. `frequency_strategy="harmonic"` uses integer multiples of
the base frequency (1x, 2x, 3x, ...); the alternative spacing is useful when the signal is not
a clean harmonic series.

```{tip}
Use `FourierFeatureTransformer` for a range-based frequency basis. The fitted range sets the
base frequency; it is not an estimate of the signal's period. For example, the harmonic map
repeats every 23 units if fitted on hours 0 through 23. To encode a 24-hour cycle, use
the [periodic encoder](numerical_encoding.md#periodic-encoding) with `period=24`.
```

## Where to go next

- [Spline expansions](spline_expansions.md) for smooth statistical bases.
- [Kernel approximation](kernel_approximation.md) for the multivariate RFF and Nyström maps.
- [Numerical encoding](numerical_encoding.md) for binning, PLE, and periodic encoding.
- [Representations overview](overview.md) for the literature and supporting utilities.
