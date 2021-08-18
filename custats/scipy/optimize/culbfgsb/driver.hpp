#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>
#include <cuda_runtime.h>
#include "culbfgsb/culbfgsb.h" 

void test_dsscfg_cuda(std::function<void(double*, double&, unsigned long &, int len)> func, double *x, double *xl, double *xu, int *nbd, int elements, double ftol, double gtol, double eps, int maxiter, int *iterations, int *stats, int m);
void test_dsscfg_cuda_float(std::function<void(float*, float&, unsigned long&, int len)> func, float *x, float *xl, float *xu, int *nbd, int elements, float ftol, float gtol, float eps, int maxiter, int *iterations, int *stats, int m);
