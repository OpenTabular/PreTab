from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import _check_y, check_consistent_length


def check_X(X: Union[np.generic, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Check if input X is a numpy array or pandas dataframe and return a pandas dataframe.
    It is used for most of the module internally to check and handle incoming data.

    Parameters
    ----------
    X : Union[np.generic, np.ndarray, pd.DataFrame]
        The input data to be checked.

    Returns
    -------
    pd.DataFrame
        The input data as a pandas dataframe.

    Raises
    ------

    ValueError
        If the input X is not a 2D array or dataframe.
    TypeError
        If the input X is not a numpy array or dataframe.
    """
    if isinstance(X, pd.DataFrame):
        X = X.copy()
    elif isinstance(X, (np.generic, np.ndarray)):
        if X.ndim == 0:
            raise ValueError('Expect 2D array or dataframe')
        if X.ndim == 1:
            raise ValueError('Expected 2D array, got 1D array')

        X = pd.DataFrame(X)
        X.columns = [f"x{i}" for i in range(X.shape[1])]
    else:
        raise TypeError(
            f'Expected X to be a numpy array or dataframe, got {type(X)}.')
    return X


def check_y(
    y: Union[np.generic, np.ndarray, pd.Series],
    multi_output: bool = False,
    y_numeric: bool = False,
) -> pd.Series:
    """
    Check if y is a valid 1D array or Series and convert/return it as a Series.
    It could be utilized to target as series instead of dataframe to ML algorithms to avoid warning or errors.

    Parameters
    ----------
    y : Union[np.generic, np.ndarray, pd.Series]
        The input array or Series to check.
    multi_output : bool, optional
        Whether y should be interpreted as a multi-output problem, by default False.
        Set it to True for multivariate time series and multi-output use cases.
    y_numeric : bool, optional
        Whether to convert y to a float dtype if it is of object dtype, by default False.

    Returns
    -------
    pd.Series
        The input y as a Series, with any necessary conversions made.

    Raises
    ------
    ValueError
        If y is None, contains null values, or contains inf values.

    Notes
    -----

    - If y is already a Series, it is checked for null and infinite values. 
    - If y is not a Series, it is passed to the `_check_y` function to perform additional checks and then converted to a Series.

    """
    if y is None:
        raise ValueError(
            'y value should be passed as 1d array, but passed None')

    elif isinstance(y, pd.Series):
        if y.isnull().any():
            raise ValueError('y contains null values.')
        if y.dtype != "O" and not np.isfinite(y).all():
            raise ValueError('y contains inf values.')
        if y_numeric and y.dtype == "O":
            y = y.astype("float")
        y = y.copy()

    else:
        y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric)
        y = pd.Series(y)
    return y


def check_X_y(X: Union[np.generic, np.ndarray, pd.DataFrame],
              y: Union[np.generic, np.ndarray, pd.Series, List],
              multi_output: bool = False,
              y_numeric: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Check and preprocess input data X and target variable y for supervised learning.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or pandas DataFrame
        The input data.
    y : {array-like, sparse matrix} of shape (n_samples,) or (n_samples, n_outputs) or pandas Series
        The target variable(s).
    multi_output : bool, default=False
        Whether y is a multi-output variable.
    y_numeric : bool, default=False
        Whether y is numeric.


    Returns
    -------
    X : pandas DataFrame or numpy array of shape (n_samples, n_features)
        The preprocessed input data.
    y : pandas Series or numpy array of shape (n_samples,) or (n_samples, n_outputs)
        The preprocessed target variable(s).

    Raises
    ------
    ValueError
        If X and y have different lengths or their indexes do not match.

    Notes
    -----

    - This function checks the input data and target variable for common issues in supervised learning, such as missing values, inconsistent lengths, and mismatched indexes. 
    - It returns the X as pandas DataFrames or numpy arrays, and the target variable(s) as pandas Series or numpy arrays.
    - If X and y are both pandas objects, this function also checks that their indexes match and raises a ValueError if they do not.

    """
    def _check_X_y(X, y):
        X = check_X(X)
        y = check_y(y, multi_output=multi_output, y_numeric=y_numeric)
        check_consistent_length(X, y)
        return X, y

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X, y = _check_X_y(X, y)
        if X.index.equals(y.index) is False:
            raise ValueError('X and y indexes do not match.')

    if isinstance(X, pd.DataFrame) and not isinstance(y, pd.Series):
        X, y = _check_X_y(X, y)
        y.index = X.index

    elif not isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X, y = _check_X_y(X, y)
        X.index = y.index

    else:
        X, y = _check_X_y(X, y)
    return X, y
