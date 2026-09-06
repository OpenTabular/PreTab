"""Basis-expansion representations.

Expansions map each numeric feature into a richer set of columns so that a linear
model can capture nonlinear structure. PreTab groups them into two families:

- :mod:`pretab.expansion.spline` for spline basis expansions such as B-spline,
  P-spline, natural cubic, and the multivariate tensor-product and thin-plate bases.
- :mod:`pretab.expansion.functional` for explicit nonlinear basis functions such as
  radial basis functions, ReLU, sigmoid, tanh, and Fourier features.

Every class here is also re-exported from :mod:`pretab.transformers`, which stays the
stable, flat public import surface.
"""
