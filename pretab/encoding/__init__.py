"""Feature encoding representations.

Encoding recodes a raw column into a form a model can use directly, as opposed to
:mod:`pretab.expansion`, which expands a column into a richer basis. PreTab splits
encoding by input kind:

- :mod:`pretab.encoding.numerical` recodes numeric values (binning, PLE, periodic).
- :mod:`pretab.encoding.categorical` maps categories to codes or indicators.

Every class here is also re-exported from :mod:`pretab.transformers`.
"""
