"""Testing helper functions

Warning: current status experimental, mostly copy paste

Warning: these functions will be changed without warning as the need
during refactoring arises.

The first group of functions provide consistency checks

"""
from statsmodels.compat.c_pandas import assert_frame_equal, assert_series_equal

import os
import sys
from distutils.version import LooseVersion

import cupy as cp
from cupy.testing import assert_allclose

import pandas as pd


class PytestTester(object):
    def __init__(self, package_path=None):
        f = sys._getframe(1)
        if package_path is None:
            package_path = f.f_locals.get('__file__', None)
            if package_path is None:
                raise ValueError('Unable to determine path')
        self.package_path = os.path.dirname(package_path)
        self.package_name = f.f_locals.get('__name__', None)

    def __call__(self, extra_args=None, exit=False):
        try:
            import pytest
            if not LooseVersion(pytest.__version__) >= LooseVersion('3.0'):
                raise ImportError
            if extra_args is None:
                extra_args = ['--tb=short', '--disable-pytest-warnings']
            cmd = [self.package_path] + extra_args
            print('Running pytest ' + ' '.join(cmd))
            status = pytest.main(cmd)
            if exit:
                sys.exit(status)
        except ImportError:
            raise ImportError('pytest>=3 required to run the test')


def check_ttest_tvalues(results):
    # test that t_test has same results a params, bse, tvalues, ...
    res = results
    mat = cp.eye(len(res.params))
    tt = res.t_test(mat)

    assert_allclose(tt.effect, res.params, rtol=1e-12)
    # TODO: tt.sd and tt.tvalue are 2d also for single regressor, squeeze
    assert_allclose(cp.squeeze(tt.sd), res.bse, rtol=1e-10)
    assert_allclose(cp.squeeze(tt.tvalue), res.tvalues, rtol=1e-12)
    assert_allclose(tt.pvalue, res.pvalues, rtol=5e-10)
    assert_allclose(tt.conf_int(), res.conf_int(), rtol=1e-10)

    # test params table frame returned by t_test
    table_res = cp.column_stack((res.params, res.bse, res.tvalues,
                                 res.pvalues, res.conf_int()))
    table2 = tt.summary_frame().values
    assert_allclose(table2, table_res, rtol=1e-12)

    # TODO: move this to test_attributes ?
    assert(hasattr(res, 'use_t'))

    tt = res.t_test(mat[0])
    tt.summary()   # smoke test for #1323
    assert_allclose(tt.pvalue, res.pvalues[0], rtol=5e-10)
    # TODO: Adapt more of test_generic_methods.test_ttest_values here?


def check_ftest_pvalues(results):
    """
    Check that the outputs of `res.wald_test` produces pvalues that
    match res.pvalues.

    Check that the string representations of `res.summary()` and (possibly)
    `res.summary2()` correctly label either the t or z-statistic.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    res = results
    use_t = res.use_t
    k_vars = len(res.params)
    # check default use_t
    pvals = [res.wald_test(cp.eye(k_vars)[k], use_f=use_t).pvalue
             for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # automatic use_f based on results class use_t
    pvals = [res.wald_test(cp.eye(k_vars)[k]).pvalue
             for k in range(k_vars)]
    assert_allclose(pvals, res.pvalues, rtol=5e-10, atol=1e-25)

    # TODO: Separate these out into summary/summary2 tests?
    # label for pvalues in summary
    string_use_t = 'P>|z|' if use_t is False else 'P>|t|'
    summ = str(res.summary())
    assert(string_use_t in summ)

    # try except for models that do not have summary2
    try:
        summ2 = str(res.summary2())
    except AttributeError:
        pass
    else:
        assert(string_use_t in summ2)


def check_fitted(results):
    import pytest

    # ignore wrapper for isinstance check
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults

    # possibly unwrap -- GEE has no wrapper
    results = getattr(results, '_results', results)

    if isinstance(results, (GLMResults, DiscreteResults)):
        pytest.skip('Not supported for {0}'.format(type(results)))

    res = results
    fitted = res.fittedvalues
    assert_allclose(res.model.endog - fitted, res.resid, rtol=1e-12)
    assert_allclose(fitted, res.predict(), rtol=1e-12)


def check_predict_types(results):
    """
    Check that the `predict` method of the given results object produces the
    correct output type.

    Parameters
    ----------
    results : Results

    Raises
    ------
    AssertionError
    """
    res = results
    # squeeze to make 1d for single regressor test case
    p_exog = cp.squeeze(cp.asarray(res.model.exog[:2]))

    # ignore wrapper for isinstance check
    from statsmodels.genmod.generalized_linear_model import GLMResults
    from statsmodels.discrete.discrete_model import DiscreteResults

    # possibly unwrap -- GEE has no wrapper
    results = getattr(results, '_results', results)

    if isinstance(results, (GLMResults, DiscreteResults)):
        # SMOKE test only  TODO: mark this somehow
        res.predict(p_exog)
        res.predict(p_exog.tolist())
        res.predict(p_exog[0].tolist())
    else:
        fitted = res.fittedvalues[:2]
        assert_allclose(fitted, res.predict(p_exog), rtol=1e-12)
        # this needs reshape to column-vector:
        assert_allclose(fitted, res.predict(cp.squeeze(p_exog).tolist()),
                        rtol=1e-12)
        # only one prediction:
        assert_allclose(fitted[:1], res.predict(p_exog[0].tolist()),
                        rtol=1e-12)
        assert_allclose(fitted[:1], res.predict(p_exog[0]),
                        rtol=1e-12)

        # Check that pandas wrapping works as expected
        exog_index = range(len(p_exog))
        predicted = res.predict(p_exog)

        cls = pd.Series if p_exog.ndim == 1 else pd.DataFrame
        predicted_pandas = res.predict(cls(p_exog, index=exog_index))

        # predicted.ndim may not match p_exog.ndim because it may be squeezed
        #  if p_exog has only one column
        cls = pd.Series if predicted.ndim == 1 else pd.DataFrame
        predicted_expected = cls(predicted, index=exog_index)
        if isinstance(predicted_expected, pd.Series):
            assert_series_equal(predicted_expected, predicted_pandas)
        else:
            assert_frame_equal(predicted_expected, predicted_pandas)
