"""
Utility functions models code
"""
import cupy as cp
import numpy.lib.recfunctions as nprf
import pandas as pd
import cudf
import numpy as np
import scipy.linalg

from statsmodels.compat.python import lzip, lmap

from statsmodels.tools.data import _is_using_cudf, _is_recarray
from statsmodels.tools.validation import array_like, bool_like, string_like


def add_trend(x, trend="c", prepend=False, has_constant='skip'):
    """
    Add a trend and/or constant to an array.

    Parameters
    ----------
    x : array_like
        Original array of data.
    trend : str {'n', 'c', 't', 'ct', 'ctt'}
        The trend to add.

        * 'n' add no trend.
        * 'c' add constant only.
        * 't' add trend only.
        * 'ct' add constant and linear trend.
        * 'ctt' add constant and linear and quadratic trend.
    prepend : bool
        If True, prepends the new data to the columns of X.
    has_constant : str {'raise', 'add', 'skip'}
        Controls what happens when trend is 'c' and a constant column already
        exists in x. 'raise' will raise an error. 'add' will add a column of
        1s. 'skip' will return the data without change. 'skip' is the default.

    Returns
    -------
    array_like
        The original data with the additional trend columns.  If x is a
        recarray or pandas Series or DataFrame, then the trend column names
        are 'const', 'trend' and 'trend_squared'.

    See Also
    --------
    statsmodels.tools.tools.add_constant
        Add a constant column to an array.

    Notes
    -----
    Returns columns as ['ctt','ct','c'] whenever applicable. There is currently
    no checking for an existing trend.
    """
    prepend = bool_like(prepend, 'prepend')
    trend = string_like(trend, 'trend', options=('n', 'c', 't', 'ct', 'ctt'))
    has_constant = string_like(has_constant, 'has_constant',
                               options=('raise', 'add', 'skip'))

    # TODO: could be generalized for trend of aribitrary order
    columns = ['const', 'trend', 'trend_squared']
    if trend == 'n':
        return x.copy()
    elif trend == "c":  # handles structured arrays
        columns = columns[:1]
        trendorder = 0
    elif trend == "ct" or trend == "t":
        columns = columns[:2]
        if trend == "t":
            columns = columns[1:2]
        trendorder = 1
    elif trend == "ctt":
        trendorder = 2

    is_recarray = _is_recarray(x)
    is_pandas = _is_using_cudf(x, None) or is_recarray
    if is_pandas or is_recarray:
        if is_recarray:
            # deprecated: remove recarray support after 0.12
            import warnings
            from statsmodels.tools.sm_exceptions import recarray_warning
            warnings.warn(recarray_warning, FutureWarning)

            descr = x.dtype.descr
            try:
                x = cudf.DataFrame.from_records(x)
            except:
                # need to convert object to string
                x = pd.DataFrame.from_records(x)
                dtypes = pd.DataFrame.from_records(x).dtypes.to_dict()
                for k in dtypes:
                    if dtypes[k] == np.dtype('O'):
                        x[k] = x[k].astype('string')
                x = cudf.from_pandas(x)
        elif isinstance(x, cudf.Series):
            x = x.to_frame()
        else:
            x = x.copy()
    else:
        x = cp.asanyarray(x)

    nobs = len(x)
    trendarr = cp.array(np.vander(np.arange(1, nobs + 1, dtype=cp.float64), trendorder + 1))
    # put in order ctt
    trendarr = cp.fliplr(trendarr)
    if trend == "t":
        trendarr = trendarr[:, 1]

    if "c" in trend:
        if is_pandas or is_recarray:
            # Mixed type protection
            def safe_is_const(s):
                try:
                    return cp.ptp(s.values) == 0.0 and np.any(s != 0.0)
                except:
                    return False
            col_const = cudf.from_pandas(x.to_pandas().apply(safe_is_const, 0)).values
        else:
            ptp0 = cp.ptp(cp.asanyarray(x), axis=0)
            col_is_const = ptp0 == 0
            nz_const = col_is_const & (x[0] != 0)
            col_const = nz_const

        if cp.any(col_const):
            if has_constant == 'raise':
                if x.ndim == 1:
                    base_err = "x is constant."
                else:
                    columns = cp.arange(x.shape[1])[col_const]
                    if isinstance(x, cudf.DataFrame):
                        columns = x.columns
                    const_cols = ", ".join([str(c) for c in columns])
                    base_err = (
                        "x contains one or more constant columns. Column(s) "
                        f"{const_cols} are constant."
                    )
                msg = (
                    f"{base_err} Adding a constant with trend='{trend}' is not allowed."
                )
                raise ValueError(msg)
            elif has_constant == 'skip':
                columns = columns[1:]
                trendarr = trendarr[:, 1:]

    order = 1 if prepend else -1
    if is_recarray or is_pandas:
        trendarr = cudf.DataFrame(trendarr, index=x.index, columns=columns)
        x = [trendarr, x]
        x = cudf.concat(x[::order], 1)
    else:
        x = [trendarr, x]
        x = cp.column_stack(x[::order])

    if is_recarray:
        x = x.to_records(index=False)
        new_descr = x.dtype.descr
        extra_col = len(new_descr) - len(descr)
        if prepend:
            descr = new_descr[:extra_col] + descr
        else:
            descr = descr + new_descr[-extra_col:]

        x = x.astype(cp.dtype(descr))

    return x

def asstr2(s):
    if isinstance(s, str):
        return s
    elif isinstance(s, bytes):
        return s.decode('latin1')
    else:
        return str(s)


def _make_dictnames(tmp_arr, offset=0):
    """
    Helper function to create a dictionary mapping a column number
    to the name in tmp_arr.
    """
    col_map = {}
    for i, col_name in enumerate(tmp_arr):
        col_map[i + offset] = col_name
    return col_map


def drop_missing(Y, X=None, axis=1):
    """
    Returns views on the arrays Y and X where missing observations are dropped.

    Y : array_like
    X : array_like, optional
    axis : int
        Axis along which to look for missing observations.  Default is 1, ie.,
        observations in rows.

    Returns
    -------
    Y : ndarray
        All Y where the
    X : ndarray

    Notes
    -----
    If either Y or X is 1d, it is reshaped to be 2d.
    """
    Y = cp.asarray(Y)
    if Y.ndim == 1:
        Y = Y[:, None]
    if X is not None:
        X = cp.array(X)
        if X.ndim == 1:
            X = X[:, None]
        keepidx = cp.logical_and(~cp.isnan(Y).any(axis),
                                 ~cp.isnan(X).any(axis))
        return Y[keepidx], X[keepidx]
    else:
        keepidx = ~cp.isnan(Y).any(axis)
        return Y[keepidx]


# TODO: needs to better preserve dtype and be more flexible
# ie., if you still have a string variable in your array you do not
# want to cast it to float
# TODO: add name validator (ie., bad names for datasets.grunfeld)
def categorical(data, col=None, dictnames=False, drop=False):
    """
    Construct a dummy matrix from categorical variables

    .. deprecated:: 0.12

       Use pandas.get_dummies instead.

    Parameters
    ----------
    data : array_like
        A structured array, recarray, array, Series or DataFrame.  This can be
        either a 1d vector of the categorical variable or a 2d array with
        the column specifying the categorical variable specified by the col
        argument.
    col : {str, int, None}
        If data is a DataFrame col must in a column of data. If data is a
        Series, col must be either the name of the Series or None. If data is a
        structured array or a recarray, `col` can be a string that is the name
        of the column that contains the variable.  For all other
        arrays `col` can be an int that is the (zero-based) column index
        number.  `col` can only be None for a 1d array.  The default is None.
    dictnames : bool, optional
        If True, a dictionary mapping the column number to the categorical
        name is returned.  Used to have information about plain arrays.
    drop : bool
        Whether or not keep the categorical variable in the returned matrix.

    Returns
    -------
    dummy_matrix : array_like
        A matrix of dummy (indicator/binary) float variables for the
        categorical data.
    dictnames :  dict[int, str], optional
        Mapping between column numbers and categorical names.

    Notes
    -----
    This returns a dummy variable for *each* distinct variable.  If a
    a structured or recarray is provided, the names for the new variable is the
    old variable name - underscore - category name.  So if the a variable
    'vote' had answers as 'yes' or 'no' then the returned array would have to
    new variables-- 'vote_yes' and 'vote_no'.  There is currently
    no name checking.

    Examples
    --------
    >>> import cupy as cp
    >>> import statsmodels.api as sm

    Univariate examples

    >>> import string
    >>> string_var = [string.ascii_lowercase[0:5],
    ...               string.ascii_lowercase[5:10],
    ...               string.ascii_lowercase[10:15],
    ...               string.ascii_lowercase[15:20],
    ...               string.ascii_lowercase[20:25]]
    >>> string_var *= 5
    >>> string_var = cp.asarray(sorted(string_var))
    >>> design = sm.tools.categorical(string_var, drop=True)

    Or for a numerical categorical variable

    >>> instr = cp.floor(cp.arange(10,60, step=2)/10)
    >>> design = sm.tools.categorical(instr, drop=True)

    With a structured array

    >>> num = cp.random.randn(25,2)
    >>> struct_ar = cp.zeros((25,1),
    ...                      dtype=[('var1', 'f4'),('var2', 'f4'),
    ...                             ('instrument','f4'),('str_instr','a5')])
    >>> struct_ar['var1'] = num[:,0][:,None]
    >>> struct_ar['var2'] = num[:,1][:,None]
    >>> struct_ar['instrument'] = instr[:,None]
    >>> struct_ar['str_instr'] = string_var[:,None]
    >>> design = sm.tools.categorical(struct_ar, col='instrument', drop=True)

    Or

    >>> design2 = sm.tools.categorical(struct_ar, col='str_instr', drop=True)
    """
    import warnings
    warnings.warn(
        "categorical is deprecated. Use pandas Categorical to represent "
        "categorical data and can get_dummies to construct dummy arrays. "
        "It will be removed after release 0.13.",
        FutureWarning
    )
    # TODO: add a NameValidator function
    if isinstance(col, (list, tuple)):
        if len(col) == 1:
            col = col[0]
        else:
            raise ValueError("Can only convert one column at a time")
    if (not isinstance(data, (cudf.DataFrame, cudf.Series)) and
            not isinstance(col, (str, int)) and
            col is not None):
        raise TypeError('col must be a str, int or None')

    # Pull out a Series from a DataFrame if provided
    if isinstance(data, cudf.DataFrame):
        if col is None:
            raise TypeError('col must be a str or int when using a DataFrame')
        elif col not in data:
            raise ValueError('Column \'{0}\' not found in data'.format(col))
        data = data[col]
        # Set col to None since we not have a Series
        col = None

    if isinstance(data, cudf.Series):
        if col is not None and data.name != col:
            raise ValueError('data.name does not match col '
                             '\'{0}\''.format(col))
        data_cat = data.astype('category')
        dummies = cudf.get_dummies(data_cat)
        col_map = {i: cat for i, cat in enumerate(dummies.columns) if
                   cat in dummies}
        if not drop:
            dummies.columns = list(dummies.columns)
            dummies = cudf.concat([dummies, data], 1)
        if dictnames:
            return dummies, col_map
        return dummies
    # catch recarrays and structured arrays
    elif data.dtype.names or data.__class__ is np.recarray:
        # deprecated: remove path after 0.12
        import warnings
        from statsmodels.tools.sm_exceptions import recarray_warning
        warnings.warn(recarray_warning, FutureWarning)
        if not col and cp.squeeze(data).ndim > 1:
            raise IndexError("col is None and the input array is not 1d")
        if isinstance(col, int):
            col = data.dtype.names[col]
        if col is None and data.dtype.names and len(data.dtype.names) == 1:
            col = data.dtype.names[0]

        tmp_arr = np.unique(data[col])

        # if the cols are shape (#,) vs (#,1) need to add an axis and flip
        _swap = True
        if data[col].ndim == 1:
            tmp_arr = tmp_arr[:, None]
            _swap = False
        tmp_dummy = (tmp_arr == np.array(data[col])).astype(float)
        if _swap:
            tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)

        if not tmp_arr.dtype.names:  # how do we get to this code path?
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr)]
        elif tmp_arr.dtype.names:
            tmp_arr = [asstr2(item) for item in np.squeeze(tmp_arr.tolist())]

        # prepend the varname and underscore, if col is numeric attribute
        # lookup is lost for recarrays...
        if col is None:
            try:
                col = data.dtype.names[0]
            except:
                col = 'var'
        # TODO: the above needs to be made robust because there could be many
        # var_yes, var_no varaibles for instance.
        tmp_arr = [col + '_' + item for item in tmp_arr]
        # TODO: test this for rec and structured arrays!!!

        if drop is True:
            if len(data.dtype) <= 1:
                if tmp_dummy.shape[0] < tmp_dummy.shape[1]:
                    tmp_dummy = np.squeeze(tmp_dummy).swapaxes(1, 0)
                dt = lzip(tmp_arr, [tmp_dummy.dtype.str]*len(tmp_arr))
                # preserve array type
                return np.array(lmap(tuple, tmp_dummy.tolist()),
                                dtype=dt).view(type(data))

            data = nprf.drop_fields(data, col, usemask=False,
                                    asrecarray=type(data) is np.recarray)
        data = nprf.append_fields(data, tmp_arr, data=tmp_dummy,
                                  usemask=False,
                                  asrecarray=type(data) is np.recarray)
        return data

    # Catch array_like for an error
    elif not isinstance(data, cp.ndarray) and not isinstance(data, np.ndarray):
        raise NotImplementedError("array_like objects are not supported")
    else:
        if isinstance(col, int):
            offset = data.shape[1]          # need error catching here?
            tmp_arr = cp.unique(data[:, col])
            tmp_dummy = (tmp_arr[:, cp.newaxis] == data[:, col]).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                offset -= 1
                data = cp.hstack([data[:, :col], data[:, col+1:]]).astype(float)
                # data = cp.delete(data, col, axis=1).astype(float)
            data = cp.column_stack((data, tmp_dummy))
            if dictnames is True:
                col_map = _make_dictnames(tmp_arr, offset)
                return data, col_map
            return data
        elif col is None and isinstance(data, np.ndarray) and np.squeeze(data).ndim == 1:
            tmp_arr = np.unique(data)
            tmp_dummy = (tmp_arr[:, None] == data).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr)
                    return tmp_dummy, col_map
                return tmp_dummy
            else:
                data = cp.column_stack((data, tmp_dummy))
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr, offset=1)
                    return data, col_map
                return data
        elif col is None and cp.squeeze(data).ndim == 1:
            tmp_arr = cp.unique(data)
            tmp_dummy = (tmp_arr[:, None] == data).astype(float)
            tmp_dummy = tmp_dummy.swapaxes(1, 0)
            if drop is True:
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr)
                    return tmp_dummy, col_map
                return tmp_dummy
            else:
                data = cp.column_stack((data, tmp_dummy))
                if dictnames is True:
                    col_map = _make_dictnames(tmp_arr, offset=1)
                    return data, col_map
                return data
        else:
            raise IndexError("The index %s is not understood" % col)


# TODO: add an axis argument to this for sysreg
def add_constant(data, prepend=True, has_constant='skip'):
    """
    Add a column of ones to an array.

    Parameters
    ----------
    data : array_like
        A column-ordered design matrix.
    prepend : bool
        If true, the constant is in the first column.  Else the constant is
        appended (last column).
    has_constant : str {'raise', 'add', 'skip'}
        Behavior if ``data`` already has a constant. The default will return
        data without adding another constant. If 'raise', will raise an
        error if any column has a constant value. Using 'add' will add a
        column of 1s if a constant column is present.

    Returns
    -------
    array_like
        The original values with a constant (column of ones) as the first or
        last column. Returned value type depends on input type.

    Notes
    -----
    When the input is recarray or a pandas Series or DataFrame, the added
    column's name is 'const'.
    """
    if _is_using_cudf(data, None) or _is_recarray(data):
        if _is_recarray(data):
            # deprecated: remove recarray support after 0.12
            import warnings
            from statsmodels.tools.sm_exceptions import recarray_warning
            warnings.warn(recarray_warning, FutureWarning)
        return add_trend(data, trend='c', prepend=prepend, has_constant=has_constant)

    # Special case for cupy
    x = cp.asanyarray(data)
    ndim = x.ndim
    if ndim == 1:
        x = x[:, None]
    elif x.ndim > 2:
        raise ValueError('Only implemented for 2-dimensional arrays')

    is_nonzero_const = cp.ptp(x, axis=0) == 0
    is_nonzero_const &= cp.all(x != 0.0, axis=0)
    if is_nonzero_const.any():
        if has_constant == 'skip':
            return x
        elif has_constant == 'raise':
            if ndim == 1:
                raise ValueError("data is constant.")
            else:
                columns = cp.arange(x.shape[1])
                cols = ",".join([str(c) for c in columns[is_nonzero_const]])
                raise ValueError(f"Column(s) {cols} are constant.")

    x = [cp.ones(x.shape[0]), x]
    x = x if prepend else x[::-1]
    return cp.column_stack(x)


def isestimable(c, d):
    """
    True if (Q, P) contrast `c` is estimable for (N, P) design `d`.

    From an Q x P contrast matrix `C` and an N x P design matrix `D`, checks if
    the contrast `C` is estimable by looking at the rank of ``vstack([C,D])``
    and verifying it is the same as the rank of `D`.

    Parameters
    ----------
    c : array_like
        A contrast matrix with shape (Q, P). If 1 dimensional assume shape is
        (1, P).
    d : array_like
        The design matrix, (N, P).

    Returns
    -------
    bool
        True if the contrast `c` is estimable on design `d`.

    Examples
    --------
    >>> d = cp.array([[1, 1, 1, 0, 0, 0],
    ...               [0, 0, 0, 1, 1, 1],
    ...               [1, 1, 1, 1, 1, 1]]).T
    >>> isestimable([1, 0, 0], d)
    False
    >>> isestimable([1, -1, 0], d)
    True
    """
    c = array_like(c, 'c', maxdim=2)
    d = array_like(d, 'd', ndim=2)
    c = c[None, :] if c.ndim == 1 else c
    if c.shape[1] != d.shape[1]:
        raise ValueError('Contrast should have %d columns' % d.shape[1])
    new = cp.vstack([c, d])
    if cp.linalg.matrix_rank(new) != cp.linalg.matrix_rank(d):
        return False
    return True


def pinv_extended(x, rcond=1e-15):
    """
    Return the pinv of an array X as well as the singular values
    used in computation.

    Code adapted from cupy.
    """
    x = cp.asarray(x)
    x = x.conjugate()
    u, s, vt = cp.linalg.svd(x, False)
    s_orig = cp.copy(s)
    m = u.shape[0]
    n = vt.shape[1]
    cutoff = rcond * s.max()
    for i in range(min(n, m)):
        if s[i] > cutoff:
            s[i] = 1./s[i]
        else:
            s[i] = 0.
    res = cp.dot(cp.transpose(vt), cp.multiply(s[:, None],
                                               cp.transpose(u)))
    return res, s_orig


def recipr(x):
    """
    Reciprocal of an array with entries less than or equal to 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
    """
    x = cp.asarray(x)
    x_flat = x.flatten()
    out_shape = x.shape
    out = cp.zeros_like(x_flat, dtype=cp.float64)
    nans = cp.isnan(x_flat)
    pos = ~nans
    pos[pos] = pos[pos] & (x_flat[pos] > 0)
    out[pos] = 1.0 / x_flat[pos]
    out[nans] = cp.nan
    return out.reshape(out_shape)


def recipr0(x):
    """
    Reciprocal of an array with entries less than 0 set to 0.

    Parameters
    ----------
    x : array_like
        The input array.

    Returns
    -------
    ndarray
        The array with 0-filled reciprocals.
    """
    x = cp.asarray(x)
    x_flat = x.flatten()
    out_shape = x.shape
    out = cp.zeros_like(x_flat, dtype=cp.float64)
    nans = cp.isnan(x_flat)
    non_zero = ~nans
    non_zero[non_zero] = non_zero[non_zero] & (x_flat[non_zero] != 0)
    out[non_zero] = 1.0 / x_flat[non_zero]
    out[nans] = cp.nan
    return out.reshape(out_shape)


def clean0(matrix):
    """
    Erase columns of zeros: can save some time in pseudoinverse.

    Parameters
    ----------
    matrix : ndarray
        The array to clean.

    Returns
    -------
    ndarray
        The cleaned array.
    """
    colsum = cp.add.reduce(matrix**2, 0)
    val = [matrix[:, i] for i in cp.flatnonzero(colsum)]
    return cp.array(cp.transpose(val))


def fullrank(x, r=None):
    """
    Return an array whose column span is the same as x.

    Parameters
    ----------
    x : ndarray
        The array to adjust, 2d.
    r : int, optional
        The rank of x. If not provided, determined by `cp.linalg.matrix_rank`.

    Returns
    -------
    ndarray
        The array adjusted to have full rank.

    Notes
    -----
    If the rank of x is known it can be specified as r -- no check
    is made to ensure that this really is the rank of x.
    """
    if r is None:
        r = cp.linalg.matrix_rank(x).item()

    v, d, u = cp.linalg.svd(x, full_matrices=False)
    order = cp.argsort(d)
    order = order[::-1]
    value = []
    for i in range(r):
        value.append(v[:, order[i]])
    value = cp.array(value)
    return cp.asarray(cp.transpose(value)).astype(cp.float64)


def unsqueeze(data, axis, oldshape):
    """
    Unsqueeze a collapsed array.

    Parameters
    ----------
    data : ndarray
        The data to unsqueeze.
    axis : int
        The axis to unsqueeze.
    oldshape : tuple[int]
        The original shape before the squeeze or reduce operation.

    Returns
    -------
    ndarray
        The unsqueezed array.

    Examples
    --------
    >>> from cupy import mean
    >>> from cupy.random import standard_normal
    >>> x = standard_normal((3,4,5))
    >>> m = mean(x, axis=1)
    >>> m.shape
    (3, 5)
    >>> m = unsqueeze(m, 1, x.shape)
    >>> m.shape
    (3, 1, 5)
    >>>
    """
    newshape = list(oldshape)
    newshape[axis] = 1
    return data.reshape(newshape)


def nan_dot(A, B):
    """
    Returns cp.dot(left_matrix, right_matrix) with the convention that
    nan * 0 = 0 and nan * x = nan if x != 0.

    Parameters
    ----------
    A, B : ndarray
    """
    # Find out who should be nan due to nan * nonzero
    should_be_nan_1 = cp.dot(cp.isnan(A), (B != 0))
    should_be_nan_2 = cp.dot((A != 0), cp.isnan(B))
    should_be_nan = should_be_nan_1 + should_be_nan_2

    # Multiply after setting all nan to 0
    # This is what happens if there were no nan * nonzero conflicts
    C = cp.dot(cp.nan_to_num(A), cp.nan_to_num(B))

    C[should_be_nan] = cp.nan

    return C


def maybe_unwrap_results(results):
    """
    Gets raw results back from wrapped results.

    Can be used in plotting functions or other post-estimation type
    routines.
    """
    return getattr(results, '_results', results)


class Bunch(dict):
    """
    Returns a dict-like object with keys accessible via attribute lookup.

    Parameters
    ----------
    *args
        Arguments passed to dict constructor, tuples (key, value).
    **kwargs
        Keyword agument passed to dict constructor, key=value.
    """
    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _ensure_2d(x, ndarray=False):
    """

    Parameters
    ----------
    x : ndarray, Series, DataFrame or None
        Input to verify dimensions, and to transform as necesary
    ndarray : bool
        Flag indicating whether to always return a cupy array. Setting False
        will return an pandas DataFrame when the input is a Series or a
        DataFrame.

    Returns
    -------
    out : ndarray, DataFrame or None
        array or DataFrame with 2 dimensiona.  One dimensional arrays are
        returned as nobs by 1. None is returned if x is None.
    names : list of str or None
        list containing variables names when the input is a pandas datatype.
        Returns None if the input is an ndarray.

    Notes
    -----
    Accepts None for simplicity
    """
    if x is None:
        return x
    is_cudf = _is_using_cudf(x, None)
    if x.ndim == 2:
        if is_cudf:
            return x, x.columns
        else:
            return x, None
    elif x.ndim > 2:
        raise ValueError('x mst be 1 or 2-dimensional.')

    name = x.name if is_cudf else None
    if ndarray:
        return cp.asarray(x)[:, None], name
    else:
        return x.to_frame(), name


def matrix_rank(m, tol=None, method="qr"):
    """
    Matrix rank calculation using QR or SVD

    Parameters
    ----------
    m : array_like
        A 2-d array-like object to test
    tol : float, optional
        The tolerance to use when testing the matrix rank. If not provided
        an appropriate value is selected.
    method : {"ip", "qr", "svd"}
        The method used. "ip" uses the inner-product of a normalized version
        of m and then computes the rank using cupy's matrix_rank.
        "qr" uses a QR decomposition and is the default. "svd" defers to
        cupy's matrix_rank.

    Returns
    -------
    int
        The rank of m.

    Notes
    -----
    When using a QR factorization, the rank is determined by the number of
    elements on the leading diagonal of the R matrix that are above tol
    in absolute value.
    """
    m = array_like(m, "m", ndim=2)
    if method == "ip":
        m = m[:, cp.any(m != 0, axis=0)]
        m = m / cp.sqrt((m ** 2).sum(0))
        m = m.T @ m
        return cp.linalg.matrix_rank(m, tol=tol, hermitian=True)
    elif method == "qr":
        r, = scipy.linalg.qr(m, mode="r")
        abs_diag = cp.abs(cp.diag(r))
        if tol is None:
            tol = abs_diag[0] * m.shape[1] * cp.finfo(float).eps
        return int((abs_diag > tol).sum())
    else:
        return cp.linalg.matrix_rank(m, tol=tol)
