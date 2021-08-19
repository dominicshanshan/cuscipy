# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:26:01 2014

Author: Josef Perktold
License: BSD-3

"""

import cupy as cp
from cupy.testing import assert_allclose, assert_array_equal
from scipy import stats

from statsmodels.tools.transform_model import StandardizeTransform


def test_standardize1():

    cp.random.seed(123)
    x = 1 + cp.random.randn(5, 4)

    transf = StandardizeTransform(x)
    xs1 = transf(x)

    assert_allclose(transf.mean, x.mean(0), rtol=1e-13)
    assert_allclose(transf.scale, x.std(0, ddof=1), rtol=1e-13)

    xs2 = cp.array(stats.zscore(x.get(), ddof=1))
    assert_allclose(xs1, xs2, rtol=1e-13, atol=1e-20)

    # check we use stored transformation
    xs4 = transf(2 * x)
    assert_allclose(xs4, (2*x - transf.mean) / transf.scale,
                    rtol=1e-13, atol=1e-20)

    # affine transform does not change standardized
    x2 = 2 * x + cp.random.randn(4)
    transf2 = StandardizeTransform(x2)
    xs3 = transf2(x2)
    assert_allclose(xs3, xs1, rtol=1e-13, atol=1e-20)

    # check constant
    x5 = cp.column_stack((cp.ones(x.shape[0]), x))
    transf5 = StandardizeTransform(x5)
    xs5 = transf5(x5)

    assert_array_equal(transf5.const_idx, 0)
    assert_array_equal(xs5[:, 0], cp.ones(x.shape[0]))
    assert_allclose(xs5[:, 1:], xs1, rtol=1e-13, atol=1e-20)
