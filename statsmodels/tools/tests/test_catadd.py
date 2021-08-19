import cupy as cp
from cupy.testing import assert_array_equal as assert_equal
from statsmodels.tools.catadd import add_indep

from cupy import linalg


def test_add_indep():
    x1 = cp.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    x2 = cp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    x0 = cp.ones(len(x2))
    x = cp.column_stack(
        [x0, x1[:, None] * cp.arange(3), x2[:, None] * cp.arange(2)])
    varnames = ['const'] + ['var1_%d' %i for i in cp.arange(3)] \
                         + ['var2_%d' %i for i in cp.arange(2)]
    xo, vo = add_indep(x, varnames)

    assert_equal(xo, cp.column_stack((x0, x1, x2)))
    assert_equal((linalg.svd(x, compute_uv=False) > 1e-12).sum(), 3)
    assert_equal(vo, ['const', 'var1_1', 'var2_1'])
