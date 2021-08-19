from statsmodels.tools import linalg
import cupy as cp
from cupy.testing import assert_allclose
from cupyx.scipy.linalg import toeplitz


def test_stationary_solve_1d():
    b = cp.random.uniform(size=10)
    r = cp.random.uniform(size=9)
    t = cp.concatenate((cp.r_[1], r))
    tmat = toeplitz(t)
    soln = cp.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)


def test_stationary_solve_2d():
    b = cp.random.uniform(size=(10, 2))
    r = cp.random.uniform(size=9)
    t = cp.concatenate((cp.r_[1], r))
    tmat = toeplitz(t)
    soln = cp.linalg.solve(tmat, b)
    soln1 = linalg.stationary_solve(r, b)
    assert_allclose(soln, soln1, rtol=1e-5, atol=1e-5)
