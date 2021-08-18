"""
Utility function to facilitate testing.
"""
import os
import sys
import platform
import re
import gc
import operator
import warnings
from functools import partial, wraps
import shutil
import contextlib
from tempfile import mkdtemp, mkstemp
from unittest.case import SkipTest
from warnings import WarningMessage
import pprint
import cupy
import cupyx
import numpy
from numpy import errstate #TODO
from numpy.core import isnat #TODO
from cupy.testing import assert_array_equal
from cupy import(intp, float32, empty, arange, ndarray, array_repr, ndarray,array)
from io import StringIO

def gisnan(x):
    """like isnan, but always raise an error if type not supported instead of
    returning a TypeError object.
    Notes
    -----
    isnan and other ufunc sometimes return a NotImplementedType object instead
    of raising any exception. This function is a wrapper to make sure an
    exception is always raised.
    This should be removed once this problem is solved at the Ufunc level."""
    from cupy import isnan
    st = isnan(x)
    if isinstance(st, type(NotImplemented)):
        raise TypeError("isnan not supported for this type")
    return st


def gisfinite(x):
    """like isfinite, but always raise an error if type not supported instead
    of returning a TypeError object.
    Notes
    -----
    isfinite and other ufunc sometimes return a NotImplementedType object
    instead of raising any exception. This function is a wrapper to make sure
    an exception is always raised.
    This should be removed once this problem is solved at the Ufunc level."""
    from cupy import isfinite
    from numpy import errstate
    with errstate(invalid='ignore'):
        st = isfinite(x)
        if isinstance(st, type(NotImplemented)):
            raise TypeError("isfinite not supported for this type")
    return st


def gisinf(x):
    """like isinf, but always raise an error if type not supported instead of
    returning a TypeError object.
    Notes
    -----
    isinf and other ufunc sometimes return a NotImplementedType object instead
    of raising any exception. This function is a wrapper to make sure an
    exception is always raised.
    This should be removed once this problem is solved at the Ufunc level."""
    from cupy import isinf
    from numpy import errstate
    with errstate(invalid='ignore'):
        st = isinf(x)
        if isinstance(st, type(NotImplemented)):
            raise TypeError("isinf not supported for this type")
    return st





def build_err_msg(arrays, err_msg, header='Items are not equal:',
                  verbose=True, names=('ACTUAL', 'DESIRED'), precision=8):
    msg = ['\n' + header]
    if err_msg:
        if err_msg.find('\n') == -1 and len(err_msg) < 79-len(header):
            msg = [msg[0] + ' ' + err_msg]
        else:
            msg.append(err_msg)
    if verbose:
        for i, a in enumerate(arrays):

            if isinstance(a, ndarray):
                # precision argument is only needed if the objects are ndarrays
                r_func = partial(array_repr, precision=precision)
            else:
                r_func = repr

            try:
                r = r_func(a)
            except Exception as exc:
                r = f'[repr failed for <{type(a).__name__}>: {exc}]'
            if r.count('\n') > 3:
                r = '\n'.join(r.splitlines()[:3])
                r += '...'
            msg.append(f' {names[i]}: {r}')
    return '\n'.join(msg)
 
def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         precision=6, equal_nan=True, equal_inf=True):
    __tracebackhide__ = True  # Hide traceback for py.test
    from cupy import array, isnan, inf, bool_, all, max

    x = cupy.asanyarray(x)
    y = cupy.asanyarray(y)

    # original array for output formatting
    ox, oy = x, y

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPefdgFDG'

    def istime(x):
        return x.dtype.char in "Mm"

    def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
        """Handling nan/inf.
        Combine results of running func on x and y, checking that they are True
        at the same locations.
        """
        __tracebackhide__ = True  # Hide traceback for py.test

        x_id = func(x)
        y_id = func(y)
        # We include work-arounds here to handle three types of slightly
        # pathological ndarray subclasses:
        # (1) all() on `masked` array scalars can return masked arrays, so we
        #     use != True
        # (2) __eq__ on some ndarray subclasses returns Python booleans
        #     instead of element-wise comparisons, so we cast to bool_() and
        #     use isinstance(..., bool) checks
        # (3) subclasses with bare-bones __array_function__ implementations may
        #     not implement np.all(), so favor using the .all() method
        # We are not committed to supporting such subclasses, but it's nice to
        # support them if possible.
        if bool_(x_id == y_id).all() != True:
            msg = build_err_msg([x, y],
                                err_msg + '\nx and y %s location mismatch:'
                                % (hasval), verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
        # If there is a scalar, then here we know the array has the same
        # flag as it everywhere, so we should return the scalar flag.
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id

    try:
        cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + f'\n(shapes {x.shape}, {y.shape} mismatch)',
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)

        flagged = bool_(False)
        if isnumber(x) and isnumber(y):
            if equal_nan:
                flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')

            if equal_inf:
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == +inf,
                                                hasval='+inf')
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == -inf,
                                                hasval='-inf')

        elif istime(x) and istime(y):
            # If one is datetime64 and the other timedelta64 there is no point
            if equal_nan and x.dtype.type == y.dtype.type:
                flagged = func_assert_same_pos(x, y, func=isnat, hasval="NaT")

        if flagged.ndim > 0:
            x, y = x[~flagged], y[~flagged]
            # Only do the comparison if actual values are left
            if x.size == 0:
                return
        elif flagged:
            # no sense doing comparison if everything is flagged.
            return

        val = comparison(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()

        # The below comparison is a hack to ensure that fully masked
        # results, for which val.ravel().all() returns np.ma.masked,
        # do not trigger a failure (np.ma.masked != True evaluates as
        # np.ma.masked, which is falsy).
        if cond != True:
            n_mismatch = reduced.size - reduced.sum(dtype=intp)
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [
                'Mismatched elements: {} / {} ({:.3g}%)'.format(
                    n_mismatch, n_elements, percent_mismatch)]

            with errstate(invalid='ignore', divide='ignore'):
                # ignore errors for non-numeric types
                with contextlib.suppress(TypeError):
                    error = abs(x - y)
                    max_abs_error = max(error)
                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append('Max absolute difference: '
                                        + str(max_abs_error))
                    else:
                        remarks.append('Max absolute difference: '
                                        + array2string(max_abs_error))

                    # note: this definition of relative error matches that one
                    # used by assert_allclose (found in cp.isclose)
                    # Filter values where the divisor would be zero
                    nonzero = bool_(y != 0)
                    if all(~nonzero):
                        max_rel_error = array(inf)
                    else:
                        max_rel_error = max(error[nonzero] / abs(y[nonzero]))
                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append('Max relative difference: '
                                        + str(max_rel_error))
                    else:
                        remarks.append('Max relative difference: '
                                        + array2string(max_rel_error))

            err_msg += '\n' + '\n'.join(remarks)
            msg = build_err_msg([ox, oy], err_msg,
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
    except ValueError:
        import traceback
        efmt = traceback.format_exc()
        header = f'error during assertion:\n\n{efmt}\n\n{header}'

        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'), precision=precision)
        raise ValueError(msg) 
        
def np_assert_allclose(actual, desired, rtol=1e-7, atol=0, equal_nan=True,
                    err_msg='', verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    tolerance.
    The test is equivalent to ``allclose(actual, desired, rtol, atol)`` (note
    that ``allclose`` has different default values). It compares the difference
    between `actual` and `desired` to ``atol + rtol * abs(desired)``.
    .. versionadded:: 1.5.0
    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.
    See Also
    --------
    assert_array_almost_equal_nulp, assert_array_max_ulp
    Examples
    --------
    >>> x = [1e-5, 1e-3, 1e-1]
    >>> y = np.arccos(np.cos(x))
    >>> np.testing.assert_allclose(x, y, rtol=1e-5, atol=0)
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    import cupy as cp

    def compare(x, y):
        return cupy.isclose(x, y, rtol=rtol, atol=atol,
                                      equal_nan=equal_nan)

    actual, desired = cupy.asanyarray(actual), cupy.asanyarray(desired)
    header = f'Not equal to tolerance rtol={rtol:g}, atol={atol:g}'
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
                         verbose=verbose, header=header, equal_nan=equal_nan)



def assert_allclose(actual, desired, rtol=1e-7, atol=0, err_msg='',equal_nan=True,
                    verbose=True):
    """Raises an AssertionError if objects are not equal up to desired tolerance.
    Args:
         actual(numpy.ndarray or cupy.ndarray): The actual object to check.
         desired(numpy.ndarray or cupy.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting
             values are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """  # NOQA
    actual, desired = cupy.asanyarray(actual), cupy.asanyarray(desired)
    numpy.testing.assert_allclose(
        cupy.asnumpy(actual), cupy.asnumpy(desired),
        rtol=rtol, atol=atol,equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Raises an AssertionError if two objects are not equal up to desired
    precision.
    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.
    The test verifies identical shapes and that the elements of ``actual`` and
    ``desired`` satisfy.
        ``abs(desired-actual) < 1.5 * 10**(-decimal)``
    That is a looser test than originally documented, but agrees with what the
    actual implementation did up to rounding vagaries. An exception is raised
    at shape mismatch or conflicting values. In contrast to the standard usage
    in numpy, NaNs are compared like numbers, no assertion is raised if both
    objects have NaNs in the same positions.
    Parameters
    ----------
    x : array_like
        The actual object to check.
    y : array_like
        The desired, expected object.
    decimal : int, optional
        Desired precision, default is 6.
    err_msg : str, optional
      The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.
    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    Examples
    --------
    the first assert does not raise an exception
    >>> np.testing.assert_array_almost_equal([1.0,2.333,np.nan],
    ...                                      [1.0,2.333,np.nan])
    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33339,np.nan], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 3 (33.3%)
    Max absolute difference: 6.e-05
    Max relative difference: 2.57136612e-05
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33339,     nan])
    >>> np.testing.assert_array_almost_equal([1.0,2.33333,np.nan],
    ...                                      [1.0,2.33333, 5], decimal=5)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 5 decimals
    <BLANKLINE>
    x and y nan location mismatch:
     x: array([1.     , 2.33333,     nan])
     y: array([1.     , 2.33333, 5.     ])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from cupy import number, float_, result_type, array, issubdtype
    from cupy import any as npany

    def compare(x, y):
        try:
            if npany(gisinf(x)) or npany( gisinf(y)):
                xinfid = gisinf(x)
                yinfid = gisinf(y)
                if not (xinfid == yinfid).all():
                    return False
                # if one item, x and y is +- inf
                if x.size == y.size == 1:
                    return x == y
                x = x[~xinfid]
                y = y[~yinfid]
        except (TypeError, NotImplementedError):
            pass

        # make sure y is an inexact type to avoid abs(MIN_INT); will cause
        # casting of x later.
        dtype = result_type(y, 1.)
        y = array(y, dtype=dtype, copy=False, subok=True)
        z = abs(x - y)

        if not issubdtype(z.dtype, number):
            z = z.astype(float_)  # handle object arrays

        return z < 1.5 * 10.0**(-decimal)

    assert_array_compare(compare, x, y, err_msg=err_msg, verbose=verbose,
             header=('Arrays are not almost equal to %d decimals' % decimal),
             precision=decimal)




def assert_(val, msg=''):
    """
    Assert that works in release mode.
    Accepts callable msg to allow deferring evaluation until failure.
    The Python built-in ``assert`` does not work when executing code in
    optimized mode (the ``-O`` flag) - no byte-code is generated for it.
    For documentation on usage, refer to the Python documentation.
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if not val:
        try:
            smsg = msg()
        except TypeError:
            smsg = msg
        raise AssertionError(smsg)
def assert_almost_equal(actual,desired,decimal=7,err_msg='',verbose=True):
    """
    Raises an AssertionError if two items are not equal up to desired
    precision.
    .. note:: It is recommended to use one of `assert_allclose`,
              `assert_array_almost_equal_nulp` or `assert_array_max_ulp`
              instead of this function for more consistent floating point
              comparisons.
    The test verifies that the elements of ``actual`` and ``desired`` satisfy.
        ``abs(desired-actual) < 1.5 * 10**(-decimal)``
    That is a looser test than originally documented, but agrees with what the
    actual implementation in `assert_array_almost_equal` did up to rounding
    vagaries. An exception is raised at conflicting values. For ndarrays this
    delegates to assert_array_almost_equal
    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    decimal : int, optional
        Desired precision, default is 7.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    Raises
    ------
    AssertionError
      If actual and desired are not equal up to specified precision.
    See Also
    --------
    assert_allclose: Compare two array_like objects for equality with desired
                     relative and/or absolute precision.
    assert_array_almost_equal_nulp, assert_array_max_ulp, assert_equal
    Examples
    --------
    >>> import numpy.testing as npt
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334)
    >>> npt.assert_almost_equal(2.3333333333333, 2.33333334, decimal=10)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 10 decimals
     ACTUAL: 2.3333333333333
     DESIRED: 2.33333334
    >>> npt.assert_almost_equal(np.array([1.0,2.3333333333333]),
    ...                         np.array([1.0,2.33333334]), decimal=9)
    Traceback (most recent call last):
        ...
    AssertionError:
    Arrays are not almost equal to 9 decimals
    <BLANKLINE>
    Mismatched elements: 1 / 2 (50%)
    Max absolute difference: 6.66669964e-09
    Max relative difference: 2.85715698e-09
     x: array([1.         , 2.333333333])
     y: array([1.        , 2.33333334])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    from cupy import ndarray
    from cupy import iscomplexobj, real, imag

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except ValueError:
        usecomplex = False

    def _build_err_msg():
        header = ('Arrays are not almost equal to %d decimals' % decimal)
        return build_err_msg([actual, desired], err_msg, verbose=verbose,
                             header=header)

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_almost_equal(actualr, desiredr, decimal=decimal)
            assert_almost_equal(actuali, desiredi, decimal=decimal)
        except AssertionError:
            raise AssertionError(_build_err_msg())

    if isinstance(actual, (ndarray, tuple, list)) \
            or isinstance(desired, (ndarray, tuple, list)):
        return assert_array_almost_equal(actual, desired, decimal, err_msg)
    try:
        # If one of desired/actual is not finite, handle it specially here:
        # check that both are nan if any is a nan, and test for equality
        # otherwise
        if not (gisfinite(desired) and gisfinite(actual)):
            if gisnan(desired) or gisnan(actual):
                if not (gisnan(desired) and gisnan(actual)):
                    raise AssertionError(_build_err_msg())
            else:
                if not desired == actual:
                    raise AssertionError(_build_err_msg())
            return
    except (NotImplementedError, TypeError):
        pass
    if abs(desired - actual) >= 1.5 * 10.0**(-decimal):
        raise AssertionError(_build_err_msg())


def assert_equal(actual, desired, err_msg='', verbose=True):
    """
    Raises an AssertionError if two objects are not equal.
    Given two objects (scalars, lists, tuples, dictionaries or cupy arrays),
    check that all elements of these objects are equal. An exception is raised
    at the first conflicting values.
    When one of `actual` and `desired` is a scalar and the other is array_like,
    the function checks that each element of the array_like object is equal to
    the scalar.
    This function handles NaN comparisons as if NaN was a "normal" number.
    That is, AssertionError is not raised if both objects have NaNs in the same
    positions.  This is in contrast to the IEEE standard on NaNs, which says
    that NaN compared to anything must return False.
    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    err_msg : str, optional
        The error message to be printed in case of failure.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    Raises
    ------
    AssertionError
        If actual and desired are not equal.
    Examples
    --------
    >>> np.testing.assert_equal([4,5], [4,6])
    Traceback (most recent call last):
        ...
    AssertionError:
    Items are not equal:
    item=1
     ACTUAL: 5
     DESIRED: 6
    The following comparison does not raise an exception.  There are NaNs
    in the inputs, but they are in the same positions.
    >>> np.testing.assert_equal(np.array([1.0, 2.0, np.nan]), [1, 2, np.nan])
    """
    __tracebackhide__ = True  # Hide traceback for py.test
    if isinstance(desired, dict):
        if not isinstance(actual, dict):
            raise AssertionError(repr(type(actual)))
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k, i in desired.items():
            if k not in actual:
                raise AssertionError(repr(k))
            assert_equal(actual[k], desired[k], f'key={k!r}\n{err_msg}',
                         verbose)
        return
    if isinstance(desired, (list, tuple)) and isinstance(actual, (list, tuple)):
        assert_equal(len(actual), len(desired), err_msg, verbose)
        for k in range(len(desired)):
            assert_equal(actual[k], desired[k], f'item={k!r}\n{err_msg}',
                         verbose)
        return
    from cupy import ndarray, isscalar, signbit
    from cupy import iscomplexobj, real, imag
    if isinstance(actual, ndarray) or isinstance(desired, ndarray):
        return assert_array_equal(actual, desired, err_msg, verbose)
    msg = build_err_msg([actual, desired], err_msg, verbose=verbose)

    # Handle complex numbers: separate into real/imag to handle
    # nan/inf/negative zero correctly
    # XXX: catch ValueError for subclasses of ndarray where iscomplex fail
    try:
        usecomplex = iscomplexobj(actual) or iscomplexobj(desired)
    except (ValueError, TypeError):
        usecomplex = False

    if usecomplex:
        if iscomplexobj(actual):
            actualr = real(actual)
            actuali = imag(actual)
        else:
            actualr = actual
            actuali = 0
        if iscomplexobj(desired):
            desiredr = real(desired)
            desiredi = imag(desired)
        else:
            desiredr = desired
            desiredi = 0
        try:
            assert_equal(actualr, desiredr)
            assert_equal(actuali, desiredi)
        except AssertionError:
            raise AssertionError(msg)

    # isscalar test to check cases such as [np.nan] != np.nan
    if isscalar(desired) != isscalar(actual):
        raise AssertionError(msg)

    try:
        isdesnat = isnat(desired)
        isactnat = isnat(actual)
        dtypes_match = array(desired).dtype.type == array(actual).dtype.type
        if isdesnat and isactnat:
            # If both are NaT (and have the same dtype -- datetime or
            # timedelta) they are considered equal.
            if dtypes_match:
                return
            else:
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    # Inf/nan/negative zero handling
    try:
        isdesnan = gisnan(desired)
        isactnan = gisnan(actual)
        if isdesnan and isactnan:
            return  # both nan, so equal

        # handle signed zero specially for floats
        array_actual = array(actual)
        array_desired = array(desired)
        if (array_actual.dtype.char in 'Mm' or
                array_desired.dtype.char in 'Mm'):
            # version 1.18
            # until this version, gisnan failed for datetime64 and timedelta64.
            # Now it succeeds but comparison to scalar with a different type
            # emits a DeprecationWarning.
            # Avoid that by skipping the next check
            raise NotImplementedError('cannot compare to a scalar '
                                      'with a different type')

        if desired == 0 and actual == 0:
            if not signbit(desired) == signbit(actual):
                raise AssertionError(msg)

    except (TypeError, ValueError, NotImplementedError):
        pass

    try:
        # Explicitly use __eq__ for comparison, gh-2552
        if not (desired == actual):
            raise AssertionError(msg)

    except (DeprecationWarning, FutureWarning) as e:
        # this handles the case when the two types are not even comparable
        if 'elementwise == comparison' in e.args[0]:
            raise AssertionError(msg)
        else:
            raise


def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='',
                         precision=6, equal_nan=True, equal_inf=True):
    __tracebackhide__ = True  # Hide traceback for py.test
    from cupy import array, isnan, inf, bool_, all, max
    from numpy.core import object_ #array2string, #TODO
    from cupy import array_str as array2string
    x = array(x, copy=False, subok=True)
    y = array(y, copy=False, subok=True)

    # original array for output formatting
    ox, oy = x, y

    def isnumber(x):
        return x.dtype.char in '?bhilqpBHILQPefdgFDG'

    def istime(x):
        return x.dtype.char in "Mm"

    def func_assert_same_pos(x, y, func=isnan, hasval='nan'):
        """Handling nan/inf.
        Combine results of running func on x and y, checking that they are True
        at the same locations.
        """
        __tracebackhide__ = True  # Hide traceback for py.test

        x_id = func(x)
        y_id = func(y)
        # We include work-arounds here to handle three types of slightly
        # pathological ndarray subclasses:
        # (1) all() on `masked` array scalars can return masked arrays, so we
        #     use != True
        # (2) __eq__ on some ndarray subclasses returns Python booleans
        #     instead of element-wise comparisons, so we cast to bool_() and
        #     use isinstance(..., bool) checks
        # (3) subclasses with bare-bones __array_function__ implementations may
        #     not implement np.all(), so favor using the .all() method
        # We are not committed to supporting such subclasses, but it's nice to
        # support them if possible.
        if bool_(x_id == y_id).all() != True:
            msg = build_err_msg([x, y],
                                err_msg + '\nx and y %s location mismatch:'
                                % (hasval), verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
        # If there is a scalar, then here we know the array has the same
        # flag as it everywhere, so we should return the scalar flag.
        if isinstance(x_id, bool) or x_id.ndim == 0:
            return bool_(x_id)
        elif isinstance(y_id, bool) or y_id.ndim == 0:
            return bool_(y_id)
        else:
            return y_id

    try:
        cond = (x.shape == () or y.shape == ()) or x.shape == y.shape
        if not cond:
            msg = build_err_msg([x, y],
                                err_msg
                                + f'\n(shapes {x.shape}, {y.shape} mismatch)',
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)

        flagged = bool_(False)
        if isnumber(x) and isnumber(y):
            if equal_nan:
                flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')

            if equal_inf:
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == +inf,
                                                hasval='+inf')
                flagged |= func_assert_same_pos(x, y,
                                                func=lambda xy: xy == -inf,
                                                hasval='-inf')

        elif istime(x) and istime(y):
            # If one is datetime64 and the other timedelta64 there is no point
            if equal_nan and x.dtype.type == y.dtype.type:
                flagged = func_assert_same_pos(x, y, func=isnat, hasval="NaT")

        if flagged.ndim > 0:
            x, y = x[~flagged], y[~flagged]
            # Only do the comparison if actual values are left
            if x.size == 0:
                return
        elif flagged:
            # no sense doing comparison if everything is flagged.
            return

        val = comparison(x, y)

        if isinstance(val, bool):
            cond = val
            reduced = array([val])
        else:
            reduced = val.ravel()
            cond = reduced.all()

        # The below comparison is a hack to ensure that fully masked
        # results, for which val.ravel().all() returns np.ma.masked,
        # do not trigger a failure (np.ma.masked != True evaluates as
        # np.ma.masked, which is falsy).
        if cond != True:
            n_mismatch = reduced.size - reduced.sum(dtype=intp)
            n_elements = flagged.size if flagged.ndim != 0 else reduced.size
            percent_mismatch = 100 * n_mismatch / n_elements
            remarks = [
                'Mismatched elements: {} / {} ({:.3g}%)'.format(
                    n_mismatch, n_elements, percent_mismatch)]

            with errstate(invalid='ignore', divide='ignore'):
                # ignore errors for non-numeric types
                with contextlib.suppress(TypeError):
                    error = abs(x - y)
                    max_abs_error = max(error)
                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append('Max absolute difference: '
                                        + str(max_abs_error))
                    else:
                        remarks.append('Max absolute difference: '
                                        + array2string(max_abs_error))

                    # note: this definition of relative error matches that one
                    # used by assert_allclose (found in np.isclose)
                    # Filter values where the divisor would be zero
                    nonzero = bool_(y != 0)
                    if all(~nonzero):
                        max_rel_error = array(inf)
                    else:
                        max_rel_error = max(error[nonzero] / abs(y[nonzero]))
                    if getattr(error, 'dtype', object_) == object_:
                        remarks.append('Max relative difference: '
                                        + str(max_rel_error))
                    else:
                        remarks.append('Max relative difference: '
                                        + array2string(max_rel_error))

            err_msg += '\n' + '\n'.join(remarks)
            msg = build_err_msg([ox, oy], err_msg,
                                verbose=verbose, header=header,
                                names=('x', 'y'), precision=precision)
            raise AssertionError(msg)
    except ValueError:
        import traceback
        efmt = traceback.format_exc()
        header = f'error during assertion:\n\n{efmt}\n\n{header}'

        msg = build_err_msg([x, y], err_msg, verbose=verbose, header=header,
                            names=('x', 'y'), precision=precision)
        raise ValueError(msg)

class suppress_warnings:
    """
    Context manager and decorator doing much the same as
    ``warnings.catch_warnings``.
    However, it also provides a filter mechanism to work around
    https://bugs.python.org/issue4180.
    This bug causes Python before 3.4 to not reliably show warnings again
    after they have been ignored once (even within catch_warnings). It
    means that no "ignore" filter can be used easily, since following
    tests might need to see the warning. Additionally it allows easier
    specificity for testing warnings and can be nested.
    Parameters
    ----------
    forwarding_rule : str, optional
        One of "always", "once", "module", or "location". Analogous to
        the usual warnings module filter mode, it is useful to reduce
        noise mostly on the outmost level. Unsuppressed and unrecorded
        warnings will be forwarded based on this rule. Defaults to "always".
        "location" is equivalent to the warnings "default", match by exact
        location the warning warning originated from.
    Notes
    -----
    Filters added inside the context manager will be discarded again
    when leaving it. Upon entering all filters defined outside a
    context will be applied automatically.
    When a recording filter is added, matching warnings are stored in the
    ``log`` attribute as well as in the list returned by ``record``.
    If filters are added and the ``module`` keyword is given, the
    warning registry of this module will additionally be cleared when
    applying it, entering the context, or exiting it. This could cause
    warnings to appear a second time after leaving the context if they
    were configured to be printed once (default) and were already
    printed before the context was entered.
    Nesting this context manager will work as expected when the
    forwarding rule is "always" (default). Unfiltered and unrecorded
    warnings will be passed out and be matched by the outer level.
    On the outmost level they will be printed (or caught by another
    warnings context). The forwarding rule argument can modify this
    behaviour.
    Like ``catch_warnings`` this context manager is not threadsafe.
    Examples
    --------
    With a context manager::
        with np.testing.suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "Some text")
            sup.filter(module=np.ma.core)
            log = sup.record(FutureWarning, "Does this occur?")
            command_giving_warnings()
            # The FutureWarning was given once, the filtered warnings were
            # ignored. All other warnings abide outside settings (may be
            # printed/error)
            assert_(len(log) == 1)
            assert_(len(sup.log) == 1)  # also stored in log attribute
    Or as a decorator::
        sup = np.testing.suppress_warnings()
        sup.filter(module=np.ma.core)  # module must match exactly
        @sup
        def some_function():
            # do something which causes a warning in np.ma.core
            pass
    """
    def __init__(self, forwarding_rule="always"):
        self._entered = False

        # Suppressions are either instance or defined inside one with block:
        self._suppressions = []

        if forwarding_rule not in {"always", "module", "once", "location"}:
            raise ValueError("unsupported forwarding rule.")
        self._forwarding_rule = forwarding_rule

    def _clear_registries(self):
        if hasattr(warnings, "_filters_mutated"):
            # clearing the registry should not be necessary on new pythons,
            # instead the filters should be mutated.
            warnings._filters_mutated()
            return
        # Simply clear the registry, this should normally be harmless,
        # note that on new pythons it would be invalidated anyway.
        for module in self._tmp_modules:
            if hasattr(module, "__warningregistry__"):
                module.__warningregistry__.clear()

    def _filter(self, category=Warning, message="", module=None, record=False):
        if record:
            record = []  # The log where to store warnings
        else:
            record = None
        if self._entered:
            if module is None:
                warnings.filterwarnings(
                    "always", category=category, message=message)
            else:
                module_regex = module.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=category, message=message,
                    module=module_regex)
                self._tmp_modules.add(module)
                self._clear_registries()

            self._tmp_suppressions.append(
                (category, message, re.compile(message, re.I), module, record))
        else:
            self._suppressions.append(
                (category, message, re.compile(message, re.I), module, record))

        return record

    def filter(self, category=Warning, message="", module=None):
        """
        Add a new suppressing filter or apply it if the state is entered.
        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.
        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        self._filter(category=category, message=message, module=module,
                     record=False)

    def record(self, category=Warning, message="", module=None):
        """
        Append a new recording filter or apply it if the state is entered.
        All warnings matching will be appended to the ``log`` attribute.
        Parameters
        ----------
        category : class, optional
            Warning class to filter
        message : string, optional
            Regular expression matching the warning message.
        module : module, optional
            Module to filter for. Note that the module (and its file)
            must match exactly and cannot be a submodule. This may make
            it unreliable for external modules.
        Returns
        -------
        log : list
            A list which will be filled with all matched warnings.
        Notes
        -----
        When added within a context, filters are only added inside
        the context and will be forgotten when the context is exited.
        """
        return self._filter(category=category, message=message, module=module,
                            record=True)

    def __enter__(self):
        if self._entered:
            raise RuntimeError("cannot enter suppress_warnings twice.")

        self._orig_show = warnings.showwarning
        self._filters = warnings.filters
        warnings.filters = self._filters[:]

        self._entered = True
        self._tmp_suppressions = []
        self._tmp_modules = set()
        self._forwarded = set()

        self.log = []  # reset global log (no need to keep same list)

        for cat, mess, _, mod, log in self._suppressions:
            if log is not None:
                del log[:]  # clear the log
            if mod is None:
                warnings.filterwarnings(
                    "always", category=cat, message=mess)
            else:
                module_regex = mod.__name__.replace('.', r'\.') + '$'
                warnings.filterwarnings(
                    "always", category=cat, message=mess,
                    module=module_regex)
                self._tmp_modules.add(mod)
        warnings.showwarning = self._showwarning
        self._clear_registries()

        return self

    def __exit__(self, *exc_info):
        warnings.showwarning = self._orig_show
        warnings.filters = self._filters
        self._clear_registries()
        self._entered = False
        del self._orig_show
        del self._filters

    def _showwarning(self, message, category, filename, lineno,
                     *args, use_warnmsg=None, **kwargs):
        for cat, _, pattern, mod, rec in (
                self._suppressions + self._tmp_suppressions)[::-1]:
            if (issubclass(category, cat) and
                    pattern.match(message.args[0]) is not None):
                if mod is None:
                    # Message and category match, either recorded or ignored
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return
                # Use startswith, because warnings strips the c or o from
                # .pyc/.pyo files.
                elif mod.__file__.startswith(filename):
                    # The message and module (filename) match
                    if rec is not None:
                        msg = WarningMessage(message, category, filename,
                                             lineno, **kwargs)
                        self.log.append(msg)
                        rec.append(msg)
                    return

        # There is no filter in place, so pass to the outside handler
        # unless we should only pass it once
        if self._forwarding_rule == "always":
            if use_warnmsg is None:
                self._orig_show(message, category, filename, lineno,
                                *args, **kwargs)
            else:
                self._orig_showmsg(use_warnmsg)
            return

        if self._forwarding_rule == "once":
            signature = (message.args, category)
        elif self._forwarding_rule == "module":
            signature = (message.args, category, filename)
        elif self._forwarding_rule == "location":
            signature = (message.args, category, filename, lineno)

        if signature in self._forwarded:
            return
        self._forwarded.add(signature)
        if use_warnmsg is None:
            self._orig_show(message, category, filename, lineno, *args,
                            **kwargs)
        else:
            self._orig_showmsg(use_warnmsg)

    def __call__(self, func):
        """
        Function decorator to apply certain suppressions to a whole
        function.
        """
        @wraps(func)
        def new_func(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return new_func