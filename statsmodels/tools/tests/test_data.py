import cupy as cp
import numpy as np
import cudf
import pandas as pd
from statsmodels.tools import data


def test_missing_data_pandas():
    """
    Fixes GH: #144
    """
    X = cp.random.random((10, 5))
    X[1, 2] = cp.nan
    df = cudf.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    cp.testing.assert_array_equal(rnames, cp.array([0, 2, 3, 4, 5, 6, 7, 8, 9]))


def test_dataframe():
    X = cp.random.random((10, 5))
    df = cudf.DataFrame(X)
    vals, cnames, rnames = data.interpret_data(df)
    cp.testing.assert_array_equal(vals, df.values)
    cp.testing.assert_array_equal(rnames, df.index.values)
    cp.testing.assert_array_equal(cnames, df.columns.values)


def test_patsy_577():
    X = np.random.random((10, 2))
    df = pd.DataFrame(X, columns=["var1", "var2"])
    from patsy import dmatrix
    endog = dmatrix("var1 - 1", df)
    assert(data._is_using_patsy(endog, None))
    exog = dmatrix("var2 - 1", df)
    assert(data._is_using_patsy(endog, exog))
