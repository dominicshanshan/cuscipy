#include <Python.h>
#include <string>
#include "call_obj.h" // cython helper file

class PyObjWrapper {
public:
    // constructors and destructors mostly do reference counting
    PyObjWrapper(PyObject* o): held(o) {
        Py_XINCREF(o);
    }

    PyObjWrapper(const PyObjWrapper& rhs): PyObjWrapper(rhs.held) { // C++11 onwards only
    }

    PyObjWrapper(PyObjWrapper&& rhs): held(rhs.held) {
        rhs.held = 0;
    }

    // need no-arg constructor to stack allocate in Cython
    PyObjWrapper(): PyObjWrapper(nullptr) {
    }

    ~PyObjWrapper() {
        Py_XDECREF(held);
    }

    PyObjWrapper& operator=(const PyObjWrapper& rhs) {
        PyObjWrapper tmp = rhs;
        return (*this = std::move(tmp));
    }

    PyObjWrapper& operator=(PyObjWrapper&& rhs) {
        held = rhs.held;
        rhs.held = 0;
        return *this;
    }

    void operator()(double *x, double &f, unsigned long &g, int len) {
        if (held) { // nullptr check
            call_obj(held, x, f, g, len); // note, no way of checking for errors until you return to Python
        }
    }

    void operator()(float *x, float &f, unsigned long &g, int len) {
        if (held) { // nullptr check
            call_obj_float(held, x, f, g, len); // note, no way of checking for errors until you return to Python
        }
    }

private:
    PyObject* held;
};
