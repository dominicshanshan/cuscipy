# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 22:28:48 2011

@author: josef
"""
import cupy as cp
from cupy.testing import assert_array_almost_equal as assert_almost_equal
from cupy.testing import assert_array_equal as assert_equal
import pytest

from statsmodels.tools.eval_measures import (
    aic,
    aic_sigma,
    aicc,
    aicc_sigma,
    bias,
    bic,
    bic_sigma,
    hqic,
    hqic_sigma,
    iqr,
    maxabs,
    meanabs,
    medianabs,
    medianbias,
    mse,
    rmse,
    rmspe,
    vare,
)


def test_eval_measures():
    # mainly regression tests
    x = cp.arange(20).reshape(4, 5)
    y = cp.ones((4, 5))

    assert_equal(iqr(x, y), 5 * cp.ones(5))
    assert_equal(iqr(x, y, axis=1), 2 * cp.ones(4))
    assert_equal(iqr(x, y, axis=None), 9)

    assert_equal(mse(x, y), cp.array([73.5, 87.5, 103.5, 121.5, 141.5]))
    assert_equal(mse(x, y, axis=1), cp.array([3.0, 38.0, 123.0, 258.0]))

    assert_almost_equal(
        rmse(x, y),
        cp.array(
            [8.5732141, 9.35414347, 10.17349497, 11.02270384, 11.89537725]
        ),
    )
    assert_almost_equal(
        rmse(x, y, axis=1),
        cp.array([1.73205081, 6.164414, 11.09053651, 16.0623784]),
    )

    err = x - y
    loc = cp.where(x != 0)
    err[loc] /= x[loc]
    err[cp.where(x == 0)] = cp.nan
    expected = cp.sqrt(cp.nanmean(err ** 2, 0) * 100)
    assert_almost_equal(rmspe(x, y), expected)
    err[cp.where(cp.isnan(err))] = 0.0
    expected = cp.sqrt(cp.nanmean(err ** 2, 0) * 100)
    assert_almost_equal(rmspe(x, y, zeros=0), expected)

    assert_equal(maxabs(x, y), cp.array([14.0, 15.0, 16.0, 17.0, 18.0]))
    assert_equal(maxabs(x, y, axis=1), cp.array([3.0, 8.0, 13.0, 18.0]))

    assert_equal(meanabs(x, y), cp.array([7.0, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(meanabs(x, y, axis=1), cp.array([1.4, 6.0, 11.0, 16.0]))
    assert_equal(meanabs(x, y, axis=0), cp.array([7.0, 7.5, 8.5, 9.5, 10.5]))

    assert_equal(medianabs(x, y), cp.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(medianabs(x, y, axis=1), cp.array([1.0, 6.0, 11.0, 16.0]))

    assert_equal(bias(x, y), cp.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(bias(x, y, axis=1), cp.array([1.0, 6.0, 11.0, 16.0]))

    assert_equal(medianbias(x, y), cp.array([6.5, 7.5, 8.5, 9.5, 10.5]))
    assert_equal(medianbias(x, y, axis=1), cp.array([1.0, 6.0, 11.0, 16.0]))

    assert_equal(vare(x, y), cp.array([31.25, 31.25, 31.25, 31.25, 31.25]))
    assert_equal(vare(x, y, axis=1), cp.array([2.0, 2.0, 2.0, 2.0]))


ics = [aic, aicc, bic, hqic]
ics_sig = [aic_sigma, aicc_sigma, bic_sigma, hqic_sigma]


@pytest.mark.parametrize("ic,ic_sig", zip(ics, ics_sig))
def test_ic_equivalence(ic, ic_sig):
    # consistency check

    assert ic(cp.array(2), 10, 2).dtype == float
    assert ic_sig(cp.array(2), 10, 2).dtype == float

    assert_almost_equal(
        ic(-10.0 / 2.0 * cp.log(2.0), 10, 2) / 10, ic_sig(2, 10, 2), decimal=14
    )

    assert_almost_equal(
        ic_sig(cp.log(2.0), 10, 2, islog=True), ic_sig(2, 10, 2), decimal=14
    )


def test_ic():
    # test information criteria

    # examples penalty directly from formula
    n = 10
    k = 2
    assert_almost_equal(aic(0, 10, 2), 2 * k, decimal=14)
    # next see Wikipedia
    assert_almost_equal(
        aicc(0, 10, 2),
        aic(0, n, k) + 2 * k * (k + 1.0) / (n - k - 1.0),
        decimal=14,
    )
    assert_almost_equal(bic(0, 10, 2), cp.log(n) * k, decimal=14)
    assert_almost_equal(hqic(0, 10, 2), 2 * cp.log(cp.log(n)) * k, decimal=14)


def test_iqr_axis(reset_randomstate):
    x1 = cp.random.standard_normal((100, 100))
    x2 = cp.random.standard_normal((100, 100))
    ax_none = iqr(x1, x2, axis=None)
    ax_none_direct = iqr(x1.ravel(), x2.ravel())
    assert_equal(ax_none, ax_none_direct)

    ax_0 = iqr(x1, x2, axis=0)
    assert ax_0.shape == (100,)
    ax_0_direct = [iqr(x1[:, i], x2[:, i]) for i in range(100)]
    assert_almost_equal(ax_0, cp.array(ax_0_direct))

    ax_1 = iqr(x1, x2, axis=1)
    assert ax_1.shape == (100,)
    ax_1_direct = [iqr(x1[i, :], x2[i, :]) for i in range(100)]
    assert_almost_equal(ax_1, cp.array(ax_1_direct))

    assert any(ax_0 != ax_1)
