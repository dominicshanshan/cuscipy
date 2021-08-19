#!/bin/bash

pytest -v custats/scipy/optimize/tests/test_hessian_update_strategy.py::TestHessianUpdateStrategy::test_hessian_initialization
pytest -v custats/scipy/optimize/tests/test_hessian_update_strategy.py::TestHessianUpdateStrategy::test_rosenbrock_with_no_exception
pytest -v custats/scipy/optimize/tests/test_hessian_update_strategy.py::TestHessianUpdateStrategy::test_SR1_skip_update
pytest -v custats/scipy/optimize/tests/test_hessian_update_strategy.py::TestHessianUpdateStrategy::test_BFGS_skip_update
