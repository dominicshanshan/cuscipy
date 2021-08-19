#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_optimize.py::test_linesearch_powell
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_line_for_search
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_bounded_powell_outsidebounds
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_linesearch_powell_bounded
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_bounded_powell_vs_powell
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_onesided_bounded_powell_stability

###################################################################################################################
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_powell
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeWrapperDisp::test_powell_bounded

#####################################################################################################################
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[powell]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[fmin_powell]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[powell]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[powell]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[Powell]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[powell]

pytest -v custats/scipy/optimize/tests/test_optimize.py::TestIterationLimits::test_powell_limit

