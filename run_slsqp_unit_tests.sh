#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_optimize.py::test_minimize_multiple_constraints
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_tol_parameter[slsqp]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_minimize_callback_copies_array[fmin_slsqp]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_no_increase[slsqp]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_slsqp_respect_bounds
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_respect_maxiter[SLSQP]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_nan_values[slsqp]
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeSimple::test_duplicate_evaluations[slsqp]
pytest -v custats/scipy/optimize/tests/test_optimize.py::test_minimize_multiple_constraints
pytest -v custats/scipy/optimize/tests/test_optimize.py::TestOptimizeResultAttributes::test_attributes_present

pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_unbounded_approximated
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_unbounded_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_bounded_approximated
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_unbounded_combined
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_equality_approximated
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_equality_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_equality_given2
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_equality_given_cons_scalar
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_inequality_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_inequality_given_vector_constraints
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_bounded_constraint
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_minimize_bound_equality_given2
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_unbounded_approximated
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_unbounded_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_equality_approximated
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_equality_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_equality_given2
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_inequality_given
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_bound_equality_given2
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_scalar_constraints
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_integer_bounds

pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_array_bounds
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_obj_must_return_scalar
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_obj_returns_scalar_in_list
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_callback
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_inconsistent_linearization


pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_gh_6676
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_invalid_bounds
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_bounds_clipping
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_infeasible_initial
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_new_bounds_type
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_nested_minimization

pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_regression_5743
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_inconsistent_inequalities
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_gh1758
pytest -v custats/scipy/optimize/tests/test_slsqp.py::TestSLSQP::test_gh9640


