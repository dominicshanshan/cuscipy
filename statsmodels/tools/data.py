"""
Compatibility tools for various data structure inputs
"""
import cupy as cp
import cudf
import numpy as np


def _check_period_index(x, freq="M"):
    from pandas import PeriodIndex, DatetimeIndex
    if not isinstance(x.index, (DatetimeIndex, PeriodIndex)):
        raise ValueError("The index must be a DatetimeIndex or PeriodIndex")

    if x.index.freq is not None:
        inferred_freq = x.index.freqstr
    else:
        inferred_freq = cudf.infer_freq(x.index)
    if not inferred_freq.startswith(freq):
        raise ValueError("Expected frequency {}. Got {}".format(inferred_freq,
                                                                freq))


def is_data_frame(obj):
    return isinstance(obj, cudf.DataFrame)


def is_design_matrix(obj):
    from patsy import DesignMatrix
    return isinstance(obj, DesignMatrix)


def _is_structured_ndarray(obj):
    return isinstance(obj, np.ndarray) and obj.dtype.names is not None


def interpret_data(data, colnames=None, rownames=None):
    """
    Convert passed data structure to form required by estimation classes

    Parameters
    ----------
    data : array_like
    colnames : sequence or None
        May be part of data structure
    rownames : sequence or None

    Returns
    -------
    (values, colnames, rownames) : (homogeneous ndarray, list)
    """
    if isinstance(data, cp.ndarray):
        if _is_structured_ndarray(data):
            import warnings
            from statsmodels.tools.sm_exceptions import recarray_warning
            warnings.warn(recarray_warning, FutureWarning)
            if colnames is None:
                colnames = data.dtype.names
            values = struct_to_ndarray(data)
        else:
            values = data

        if colnames is None:
            colnames = ['Y_%d' % i for i in range(values.shape[1])]
    elif is_data_frame(data):
        # XXX: hack
        data = data.dropna()
        values = data.values
        colnames = data.columns
        rownames = data.index
    else:  # pragma: no cover
        raise TypeError('Cannot handle input type {typ}'
                        .format(typ=type(data).__name__))

    if not isinstance(colnames, list):
        colnames = list(colnames)

    # sanity check
    if len(colnames) != values.shape[1]:
        raise ValueError('length of colnames does not match number '
                         'of columns in data')

    if rownames is not None and len(rownames) != len(values):
        raise ValueError('length of rownames does not match number '
                         'of rows in data')

    return values, colnames, rownames


def struct_to_ndarray(arr):
    return arr.view((float, (len(arr.dtype.names),)), type=cp.ndarray)


def _is_using_ndarray_type(endog, exog):
    return (type(endog) is cp.ndarray and
            (type(exog) is cp.ndarray or exog is None))


def _is_using_ndarray(endog, exog):
    return (isinstance(endog, cp.ndarray) and
            (isinstance(exog, cp.ndarray) or exog is None))


def _is_using_pandas(endog, exog):
    from statsmodels.compat.c_pandas import data_pandas_klasses as klasses
    return (isinstance(endog, klasses) or isinstance(exog, klasses))


def _is_using_cudf(endog, exog):
    from statsmodels.compat.c_pandas import data_klasses as klasses
    return (isinstance(endog, klasses) or isinstance(exog, klasses))


def _is_array_like(endog, exog):
    try:  # do it like this in case of mixed types, ie., ndarray and list
        endog = cp.asarray(endog)
        exog = cp.asarray(exog)
        return True
    except:
        return False


def _is_numpy_array_like(endog, exog):
    try:  # do it like this in case of mixed types, ie., ndarray and list
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        return True
    except:
        return False


def _is_using_patsy(endog, exog):
    # we get this when a structured array is passed through a formula
    return (is_design_matrix(endog) and
            (is_design_matrix(exog) or exog is None))


def _is_recarray(data):
    """
    Returns true if data is a recarray
    """
    return isinstance(data, np.core.recarray)
