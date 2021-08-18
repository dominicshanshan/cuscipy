import bfgs
import cupy
from .._constraints import Bounds


def fun(x):
    center = cupy.arange(1, 5)*3.0
    return ((x - center)**2).sum().item()


def grad(x):
    center = cupy.arange(1, 5)*3.0
    return 2.0*(x - center)


x = cupy.random.rand(4)
lb = cupy.array([-cupy.inf, 0, 10, -cupy.inf])
ub = cupy.array([-10, 2, cupy.inf, cupy.inf])
b = Bounds(lb, ub)

r, stats = bfgs.fmin(fun, grad, x, bounds=b)
print(r)
print(stats)

r, stats = bfgs.fmin(fun, grad, x, bounds=None)
print(r)
print(stats)


def fun_float(x):
    center = cupy.arange(1, 5, dtype=cupy.float32)*3.0
    return ((x - center)**2).sum().item()


def grad_float(x):
    center = cupy.arange(1, 5, dtype=cupy.float32)*3.0
    return 2.0*(x - center)


x = cupy.random.rand(4, dtype=cupy.float32)
lb = cupy.array([-cupy.inf, 0, 10, -cupy.inf], dtype=cupy.float32)
ub = cupy.array([-10, 2, cupy.inf, cupy.inf], dtype=cupy.float32)
b = Bounds(lb, ub)

r, stats = bfgs.fmin(fun_float, grad_float, x, bounds=b)
print(r)
print(stats)

r, stats = bfgs.fmin(fun_float, grad_float, x, bounds=None)
print(r)
print(stats)

