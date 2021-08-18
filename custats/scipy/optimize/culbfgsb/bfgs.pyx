# distutils: language = c++

import cupy
from libc.string cimport memset
from cupy.cuda.memory import MemoryPointer, UnownedMemory

cdef extern from "py_obj_wrapper.hpp":
    cdef cppclass PyObjWrapper:
        PyObjWrapper()
        PyObjWrapper(object) # define a constructor that takes a Python object
         # note - doesn't match c++ signature - that's fine!

cdef extern from "driver.hpp":
    void test_dsscfg_cuda(PyObjWrapper, double *, double *, double *, int *, int, double, double, double, int, int *, int *, int) except +
    void test_dsscfg_cuda_float(PyObjWrapper, float *, float *, float *, int *, int, float, float, float, int, int *, int *, int) except +
            # here I lie about the signature
            # because C++ does an automatic conversion to function pointer
            # for classes that define operator(), but Cython doesn't know that

def fmin(fun, grad, x, bounds=None, ftol=1e-8, gtol=1e-8, eps=1e-8, maxiter=1000, callback=None, m=8):

    dtype = x.dtype

    def example(a, len):
        # cdef long pointer
        mem = UnownedMemory(a, 0, a)
        memptr = MemoryPointer(mem, 0)
        arr_ndarray = cupy.ndarray((len,), dtype, memptr)
        # print(arr_ndarray)
        v = fun(arr_ndarray)
        g = grad(arr_ndarray)
        # v, g = combined(arr_ndarray)
        pointer = g.data.ptr
        if callback is not None:
            callback(cupy.copy(arr_ndarray))
        return v, pointer

    cdef PyObjWrapper f = PyObjWrapper(example)
    cdef int elements
    elements = len(x) 
    cdef long pointer, lx_pointer, ux_pointer, nbd_pointer
    pointer = x.data.ptr
    cdef int iterations, stats;

    if bounds is None:
        lx = cupy.zeros(elements, dtype)
        ux = cupy.zeros(elements, dtype)
        nbd = cupy.zeros(elements, cupy.int32)
        lx_pointer = lx.data.ptr
        ux_pointer = ux.data.ptr
        nbd_pointer = nbd.data.ptr
    else:
        lb = bounds.lb
        ub = bounds.ub
        nbd = cupy.isinf(cupy.vstack([lb, ub])).sum(axis=0)
        nbd = 2 - nbd
        nbd[nbd == 1 & cupy.isinf(lb)] = 3
        nbd = nbd.astype(cupy.int32)
        lx_pointer = lb.data.ptr
        ux_pointer = ub.data.ptr
        nbd_pointer = nbd.data.ptr
    if dtype == cupy.float64:
        test_dsscfg_cuda(f, <double*>pointer, <double*>lx_pointer, <double*>ux_pointer, <int*>nbd_pointer, elements, ftol, gtol, eps, maxiter, &iterations, &stats, m)
    elif dtype == cupy.float32:
        test_dsscfg_cuda_float(f, <float*>pointer, <float*>lx_pointer, <float*>ux_pointer, <int*>nbd_pointer, elements, ftol, <float>gtol, <float>eps, maxiter, &iterations, &stats, m)
    else:
        raise TypeError('only float32 and float64 types are supported')
    return x, (iterations, stats)
