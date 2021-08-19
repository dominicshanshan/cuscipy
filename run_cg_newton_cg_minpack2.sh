#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[newton-cg]
#pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[fmin_cg]
#pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[fmin_ncg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[newton-cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[CG]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[Newton-CG]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[newton-cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[cg]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[newton-cg]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_cg
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_cg_cornercase
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_ncg_negative_maxiter
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_ncg
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_ncg_hess
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_ncg_hessp

