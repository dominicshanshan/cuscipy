import cupy as cp
import numpy as np
import cudf
import pandas as pd
from pandas.util._decorators import (deprecate_kwarg, Appender, Substitution,
                                     cache_readonly)
import string

__all__ = [
    'assert_frame_equal', 'assert_index_equal', 'assert_series_equal',
    'data_klasses', 'frequencies', 'is_numeric_dtype', 'testing',
    'cache_readonly', 'deprecate_kwarg', 'Appender', 'Substitution',
    'make_dataframe', 'to_cupy',
]


try:
    from pandas.api.types import is_numeric_dtype
except ImportError:
    from pandas.core.common import is_numeric_dtype

try:
    from pandas.tseries import offsets as frequencies
except ImportError:
    from pandas.tseries import frequencies

data_klasses = (cudf.Series, cudf.DataFrame)
data_pandas_klasses = (pd.Series, pd.DataFrame)

try:
    import pandas.testing as testing
except ImportError:
    import pandas.util.testing as testing

assert_frame_equal = testing.assert_frame_equal
assert_index_equal = testing.assert_index_equal
assert_series_equal = testing.assert_series_equal


def rands_array(nchars, size, dtype="O"):
    """
    Generate an array of byte strings.
    """
    rands_chars = np.array(list(string.ascii_letters + string.digits),
                           dtype=(np.str_, 1))
    retval = (np.random.choice(rands_chars, size=nchars * np.prod(size)).view(
        (np.str_, nchars)).reshape(size))
    if dtype is None:
        return retval
    else:
        return retval.astype(dtype)


def make_dataframe():
    """
    Simple verion of pandas._testing.makeDataFrame
    """
    n = 30
    k = 4
    index = cudf.Index(rands_array(nchars=10, size=n), name=None)
    data = {
        c: cudf.Series(cp.random.randn(n), index=index)
        for c in string.ascii_uppercase[:k]
    }

    return cudf.DataFrame(data)


def to_cupy(po: cudf.DataFrame) -> cp.ndarray:
    """
    Workaround legacy pandas lacking to_numpy

    Parameters
    ----------
    po : Pandas obkect

    Returns
    -------
    ndarray
        A numpy array
    """
    return po.values
