# References

The representations in PreTab rest on established literature. This page collects the primary
sources for each family, so every method is traceable to its origin. Citations are grouped by
representation.

## Splines and penalized splines

Eilers, P. H. C., and Marx, B. D. (1996). Flexible smoothing with B-splines and penalties.
*Statistical Science*, 11(2), 89-121.

Eilers, P. H. C., and Marx, B. D. (2003). Multivariate calibration with temperature
interaction using two-dimensional penalized signal regression. *Chemometrics and Intelligent
Laboratory Systems*, 66(2), 159-174.

These two papers introduce the P-spline (B-spline basis with a difference penalty) and its
tensor-product extension, which underpin `PSplineTransformer` and
`TensorProductSplineTransformer`.

## Thin-plate and generalized additive models

Wahba, G. (1990). *Spline Models for Observational Data*. Society for Industrial and Applied
Mathematics.

Wood, S. N. (2003). Thin plate regression splines. *Journal of the Royal Statistical Society:
Series B*, 65(1), 95-114.

Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman
and Hall/CRC.

Wahba's monograph is the foundation for thin-plate splines; Wood's work gives the low-rank
thin-plate regression spline and the GAM framing that `ThinPlateSplineTransformer` follows.

## Kernel approximations

Williams, C. K. I., and Seeger, M. (2001). Using the Nyström method to speed up kernel
machines. *Advances in Neural Information Processing Systems*, 13.

Rahimi, A., and Recht, B. (2007). Random features for large-scale kernel machines. *Advances
in Neural Information Processing Systems*, 20.

These introduce the Nyström method and random Fourier features, implemented as
`NystroemFeaturesTransformer` and `RandomFourierFeaturesTransformer`.

## Piecewise-linear encoding

Gorishniy, Y., Rubachev, I., and Babenko, A. (2022). On embeddings for numerical features in
tabular deep learning. *Advances in Neural Information Processing Systems*, 35.

This paper motivates piecewise-linear encoding of numerical features for tabular models, the
basis for `PLETransformer`.

## Where to go next

- [Representations overview](overview.md) to return to the catalogue.
- [Spline expansions](spline_expansions.md), [Kernel approximation](kernel_approximation.md),
  [Numerical encoding](numerical_encoding.md) for the methods these sources describe.
