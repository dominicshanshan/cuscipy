#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_optimize.py::test_neldermead_adaptive

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_neldermead
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_neldermead_initial_simplex
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_neldermead_initial_simplex_bad

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_neldermead
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_neldermead_initial_simplex
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperNoDisp::test_neldermead_initial_simplex_bad

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[nelder-mead]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[nelder-mead]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[Nelder-Mead]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[nelder-mead]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[nelder-mead]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestIterationLimits::test_neldermead_limit
