#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_optimize.py::test_check_grad

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_bfgs
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_bfgs_infinite

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_bfgs
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_bfgs_infinite

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_bfgs_nan

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_bfgs_nan_return
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_bfgs_numerical_jacobian
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_finite_differences
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_bfgs_gh_2169
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_bfgs_double_evaluations
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_l_bfgs_b
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_l_bfgs_b_numjac
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_l_bfgs_b_funjac
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_l_bfgs_b_maxiter
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_l_bfgs_b
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_l_bfgs_b_ftol
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_l_bfgs_maxls
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_l_bfgs_b_maxfun_interruption

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestLBFGSBBounds::test_l_bfgs_b_bounds
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestLBFGSBBounds::test_l_bfgs_b_funjac
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestLBFGSBBounds::test_minimize_l_bfgs_b_bounds
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestLBFGSBBounds::test_minimize_l_bfgs_b_incorrect_bounds
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestLBFGSBBounds::test_minimize_l_bfgs_b_bounds_FD

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[bfgs]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[l-bfgs-b]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[fmin_l_bfgs_b]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[bfgs]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[l-bfgs-b]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[bfgs]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[BFGS]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[L-BFGS-B]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[l-bfgs-b]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[bfgs]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[l-bfgs-b]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[bfgs]
