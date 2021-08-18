import functools
import types
import warnings

import cupy as cp
from cupy import asarray, array, asanyarray


def _wrapit(obj, method, *args, **kwds):
    try:
        wrap = obj.__array_wrap__
    except AttributeError:
        wrap = None
    result = getattr(asarray(obj), method)(*args, **kwds)
    if wrap:
        if not isinstance(result, cp.ndarray):
            result = asarray(result)
        result = wrap(result)
    return result

def squeeze(a, axis=None):
    """
    Remove axes of length one from `a`.
    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        .. versionadded:: 1.7.0
        Selects a subset of the entries of length one in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.
    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`. Note that if all axes are squeezed,
        the result is a 0d array and not a scalar.
    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1
    See Also
    --------
    expand_dims : The inverse operation, adding entries of length one
    reshape : Insert, remove, and combine dimensions, and resize existing ones
    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    >>> x = np.array([[1234]])
    >>> x.shape
    (1, 1)
    >>> np.squeeze(x)
    array(1234)  # 0d array
    >>> np.squeeze(x).shape
    ()
    >>> np.squeeze(x)[()]
    1234
    """
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze', axis=axis)
    if axis is None:
        return squeeze()
    else:
        return squeeze(axis=axis)
