from cupy.cuda.memory import MemoryPointer, UnownedMemory
import cupy

cdef public void call_obj(obj, double *x, double &f, unsigned long &g, int length):
    cdef double val
    ptr = <long>x
    val, grad = obj(ptr, length)
    (&f)[0] = val
    (&g)[0] = grad


cdef public void call_obj_float(obj, float *x, float &f, unsigned long &g, int length):
    cdef float val
    ptr = <long>x
    val, grad = obj(ptr, length)
    (&f)[0] = val
    (&g)[0] = grad
