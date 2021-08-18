/* SLSQP: Sequentional Least Squares Programming (aka sequential quadratic programming SQP)
   method for nonlinearly constrained nonlinear optimization, by Dieter Kraft (1991).
   Fortran released under a free (BSD) license by ACM to the SciPy project and used there.
   C translation via f2c + hand-cleanup and incorporation into NLopt by S. G. Johnson (2009). */

/* Table of constant values */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include "slsqp.h"
#include <stdio.h>

#ifndef SYSLQP_COMMON_CUH_
#define SYSLQP_COMMON_CUH_
#include <cuda.h>

__device__ double atomicArgMin_double(int* address, int id, double *v)
{
    int old = *address, assumed;
    double val = v[id];
    do {
        assumed = old;
        old = atomicCAS(address, assumed,
            val < v[assumed] ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double minRecord_double(int* address, int id, double *a, int a_dim1, double tau)
{
    int old = *address, assumed;
    double val = fabs(a[id + id * a_dim1]);
    do {
        assumed = old;
        old = atomicCAS(address, assumed,
            val <= tau ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double atomicArgMin2_double(int* address, int id, double *z, double *x, int *indx, double alpha)
{
    int old = *address, assumed;
    int l = indx[id];
    double val = -x[l] / (z[id] - x[l]);
    double max_v;
    do {
        assumed = old;
        int assumed_l = indx[assumed];
        if (assumed == 0){
            max_v = alpha;
        }
        else{
            max_v = -x[assumed_l] / (z[assumed] - x[assumed_l]) ;
        }
        old = atomicCAS(address, assumed,
            val <= max_v ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double atomicArgMax_double(int* address, int id, double *v, int *indx)
{
    int old = *address, assumed;
    double val = v[indx[id]];
    double max_v;
    do {
        assumed = old;
        if (assumed == 0){
            max_v = 0.0;
        }
        else{
            max_v = v[indx[assumed]];
        }
        old = atomicCAS(address, assumed,
            val > max_v ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double atomicArgMaxV_double(int* address, int id, double *v)
{
    int old = *address, assumed;
    double val = v[id];
    do {
        assumed = old;
        old = atomicCAS(address, assumed,
            val > v[assumed] ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double atomicArgMaxH_double(int* address, int id, double *h, double *a, int offset, int a_dim1)
{
    int old = *address, assumed;
    double dval = a[offset - 1 + id * a_dim1];
    h[id] -= dval * dval;
    double val = h[id];
    double oldv;
    do {
        assumed = old;
        oldv = h[old];
        old = atomicCAS(address, assumed,
            val > oldv ? id : assumed); 
    } while (assumed != old);
    return old;
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ double atomicMax_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}


  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
  static __inline__ __device__ double atomicAdd(double *address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    if (val==0.0)
      return __longlong_as_double(old);
    do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
  }


  #endif
#endif

int nlopt_isnan(double x)
{
#if defined(HAVE_ISNAN)
    return isnan(x);
#elif defined(_WIN32)
    return _isnan(x);
#else
    return (x != x);            /* might fail with aggressive optimization */
#endif
}
               
int nlopt_isinf(double x)
{
    return (fabs(x) >= HUGE_VAL * 0.99)
#if defined(HAVE_ISINF)
        || isinf(x)
#else
        || (!nlopt_isnan(x) && nlopt_isnan(x - x))
#endif
        ;
}

int nlopt_isfinite(double x)
{
    return (fabs(x) <= DBL_MAX)
#if defined(HAVE_ISFINITE)
        || isfinite(x)
#elif defined(_WIN32)
        || _finite(x)
#endif
        ;
}

/*      ALGORITHM 733, COLLECTED ALGORITHMS FROM ACM. */
/*      TRANSACTIONS ON MATHEMATICAL SOFTWARE, */
/*      VOL. 20, NO. 3, SEPTEMBER, 1994, PP. 262-281. */
/*      http://doi.acm.org/10.1145/192115.192124 */


/*      http://permalink.gmane.org/gmane.comp.python.scientific.devel/6725 */
/*      ------ */
/*      From: Deborah Cotton <cotton@hq.acm.org> */
/*      Date: Fri, 14 Sep 2007 12:35:55 -0500 */
/*      Subject: RE: Algorithm License requested */
/*      To: Alan Isaac */

/*      Prof. Issac, */

/*      In that case, then because the author consents to [the ACM] releasing */
/*      the code currently archived at http://www.netlib.org/toms/733 under the */
/*      BSD license, the ACM hereby releases this code under the BSD license. */

/*      Regards, */

/*      Deborah Cotton, Copyright & Permissions */
/*      ACM Publications */
/*      2 Penn Plaza, Suite 701** */
/*      New York, NY 10121-0701 */
/*      permissions@acm.org */
/*      212.869.7440 ext. 652 */
/*      Fax. 212.869.0481 */
/*      ------ */

/********************************* BLAS1 routines *************************/
__global__ void __copy_1(int n, double *dy, double x)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	  dy[i] = x;
}

__global__ void __assign(int n, int *indx){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride)
	  indx[i] = i;
}

__global__ void __boundary_fix__(int n, double *x, const double *xl, const double *xu)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i <= n; i += stride)
    {
        if (x[i] < xl[i])
            x[i] = xl[i];
        else if (x[i] > xu[i])
            x[i] = xu[i];
    }
}

__global__ void __zeros(int n, double *w, int offset){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride)
	  w[i + offset] = 0.0;
}

__global__ void __rnorm(int n, double *xmax, double *sum, double *rnorm){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
         rnorm[i+1] = xmax[i] * sqrt(sum[i]);
}


__global__ void __2d_copy__(int total, int n, double *a, int incx, double *w, int incy, int ig, int meq, int a_dim1)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int j = index / n + 1;
    int i = index % n;
    if (index < total)
    {
        w[ig - 1 + j + i * incy] = a[meq + j + a_dim1 + i * incx];
    }
}

__global__ void __copy_2(int n, double *dy, const double *dx, int incx, int incy, int offset)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	  dy[i*incy] = dx[i*incx + offset];
}

__global__ void __copy_2_int(int n, int *dy, const int *dx, int incx, int incy)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
	  dy[i*incy] = dx[i*incx];
}

__global__ void __daxpy_sl__(int n, double *dy, const double *dx, double da, int incx, int incy)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    dy[i*incy] += da * dx[i*incx];
}

__global__ void __ddot_sl__(int n, double *dx, int incx, double *dy, int incy, double *sum)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
        atomicAdd(sum, dx[i*incx] * dy[i*incy]);
}

__global__ void __two_vector_dot__(int total, int n, double *a, int incx, double *b, int incy, double *sum, int *indx, int iz1, int a_dim1, int npp1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int vid = index / n;
  int vele = index % n;
  int j = indx[vid + iz1];
  if (index < total)
   atomicAdd(sum+vid, a[npp1 + j * a_dim1 + vele * incx] * b[npp1+ vele * incy]);
}

__global__ void __two_vector_dot_scale__(int total, int n, double *a, int incx, double *b, int incy, double *x, double fac, int g_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int j = index / n;
  int i = index % n;
  if (index < total)
      atomicAdd(x + j + 1, fac * a[ (j + 1)* g_dim1 + 1 + i * incx] * b[i * incy]);
}

__global__ void __hfti_swap_max(int total, double *a, int lmax, int a_dim1, int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int stride = blockDim.x * gridDim.x;
    double tmp;
    for (int i = index; i <= total; i += stride)
    {
        tmp = a[i + offset * a_dim1];
        a[i + offset * a_dim1] = a[i + lmax * a_dim1];
        /* L60: */
        a[i + lmax * a_dim1] = tmp;
    }
}

__global__ void __hfti_two_vector(int total, int n, double *a, double *h, int offset, int a_dim1){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int l = index / n + offset;
    int i = index % n + offset;
    double val;
    if (index < total)
    {
        val = a[i + l * a_dim1];
        val = val * val;
        atomicAdd(h + l, val);
    }
}

__global__ void __two_vector_assign__(int n, int *indx, double *m_cl, int iz1, double *w){
  int index = blockIdx.x * blockDim.x + threadIdx.x + iz1;
  int stride = blockDim.x * gridDim.x;
  int j = indx[index];
  if (index - iz1 < n)
    w[j] = m_cl[index-iz1];
}

 
__global__ void __dnrm2_max___(int n, double *dx, int incx, double *xmax){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride){
        double xabs = fabs(dx[incx*i]);
        atomicMax_double(xmax, xabs);
  }
}

__global__ void __dnrm2_2d_max___(int total, int n, double *dx, int incx, double *xmax, int b_dim1, int offset){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n;
  for (int i = index; i < total; i += stride){
        double xabs = fabs(dx[offset + jb * b_dim1 + incx*j]);
        atomicMax_double(xmax+jb - 1, xabs);
  }
}

__global__ void __hfti_zero___(int total, int n,  double *b, int b_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n + 1;
  for (int i = index; i < total; i += stride){
      b[j + jb * b_dim1] = 0.0;
  }
}

__global__ void __hfti_set_zero___(int total, int start,  double *b, int b_dim1, int jb){
  int index = blockIdx.x * blockDim.x + threadIdx.x + start;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < total; i += stride)
  {
      b[i + jb * b_dim1] = 0.0;
  }
}


__global__ void __arg_max___(int n, int iz1, double *w, int *izmax, int *indx){
  int index = blockIdx.x * blockDim.x + threadIdx.x + iz1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
        atomicArgMax_double(izmax, i, w, indx);
  }
}

__global__ void __arg_max_v__(int n, int iz1, double *v, int *izmax){
  int index = blockIdx.x * blockDim.x + threadIdx.x + iz1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
      atomicArgMaxV_double(izmax, i, v);
  }

}

__global__ void __arg_max_h__(int n, int iz1, double *h, int *izmax, double *a, int a_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x + iz1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
        atomicArgMaxH_double(izmax, i, h, a, iz1, a_dim1);
  }
}
// __device__ double atomicArgMaxH_double(int* address, int id, double *h, double *a, int offset, int a_dim1)

__global__ void __arg_min2___(int n, double *z, double *x, int *indx, int *jj, double alpha){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
      if (z[i] <= 0.0){
        atomicArgMin2_double(jj, i, z, x, indx, alpha);
      }
  }
}

__global__ void __arg_hfti_find___(int n, double *a, double tau, int a_dim1, int *mid){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
    minRecord_double(mid, i, a, a_dim1, tau);
  }
}

__global__ void __assign_3___(int n, double *z, double *x, int *indx, double one, double alpha){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i <= n; i += stride){
        int l = indx[i];
        x[l] = (one - alpha) * x[l] + alpha * z[i];
  }
}

__global__ void __assign_4___(int n, int i__2, double *g, double *w, double *h, int g_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i = index %  (i__2 + 1) + 1;
  int j = index / (i__2 + 1) + 1;
  if (index < n){
    if (i < i__2 + 1){
        w[index + 1] = g[j + i * g_dim1];
    }
    else{
        w[index + 1] = h[j]; 
    }
  }
}
/*
    for (j = 1; j <= i__1; ++j)
    {
        i__2 = *n;
        for (i__ = 1; i__ <= i__2; ++i__)
        {
            ++iw;
            w[iw] = g[j + i__ * g_dim1];
        }
        ++iw;
        w[iw] = h__[j];
    }
*/

__global__ void __dnrm2_sum___(int n, double *dx, int incx, double *sum, double scale){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
  {
      double xs = scale * dx[incx * i];
      atomicAdd(sum, xs * xs);
  }
}

__global__ void __dnrm2_2d_sum___(int total, int n, double *dx, int incx, double *sum, double *xmax, int offset, int b_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n;
  double scale;
  for (int i = index; i < total; i += stride)
  {
    if (xmax[jb - 1] == 0){
        scale = 0.0;
    }
    else{
        scale = 1.0 / xmax[jb - 1];
    }
     double xs = scale * dx[offset + jb * b_dim1 + incx * j];
     atomicAdd(sum + jb - 1, xs * xs);
  }
}

__global__ void __lsei_dot___(int total, int n, double *g, int incx, double *x, int incy, double *h, int g_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n;
  double scale;
  for (int i = index; i < total; i += stride){
      atomicAdd(h + jb, -g[jb + g_dim1 + j * incx] * x[1 + j * incy]);
  }

}

__global__ void __lsei_dot2___(int total, int n, double *e, int incx, double *x, int incy, double *f, int e_dim1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n;
  double scale;
  for (int i = index; i < total; i += stride){
      atomicAdd(f + jb - 1, e[jb + e_dim1 + j * incx] * x[1 + j * incy]);
  }

}

__global__ void __lsei_dot3___(int total, int n, int me, double *e,  double *f,  double *g, double *w, double *d, int e_dim1, int g_dim1, int mc1){
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int jb = index / n + 1;
  int j = index % n;
  double scale;
  for (int i = index; i < total; i += stride){
      if (j < me){
        atomicAdd(d + jb, e[jb * e_dim1 + 1 + j] * f[1 + j]);
      }
      else{
        atomicAdd(d + jb, -g[jb * g_dim1 + 1 + j - me] * w[mc1 + j - me]);
      }
  }
}

__global__ void __lsei_combine___(int total, double *f, double *m){
  int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int stride = blockDim.x * gridDim.x;
   for (int i = index; i <= total; i += stride){
       f[i] = m[i -1] = f[i];
   }
}

/* apply Givens rotation */
__global__ void __dsrot_(int n, double *dx, int incx, double *dy, int incy, double c, double s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        double x = dx[incx * i], y = dy[incy * i];
        dx[incx * i] = c * x + s * y;
        dy[incy * i] = c * y - s * x;
    }
}

__global__ void __dscal_sl__(int n, double alpha, double *dx, int incx)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        dx[i*incx] *= alpha;
    }
}

__global__ void __h12_1__(int l1, int i__1, double *u, int u_dim1, double *cl)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x + l1;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i <= i__1; i += stride)
    {
        double t = u[i * u_dim1 + 1];
	    double sm = fabs(t);
        atomicMax_double(cl, sm);
    }
}

__global__ void __h12_2__(int l1, int i__1, double *u, int u_dim1, double clinv, double *cl){
    int index = blockIdx.x * blockDim.x + threadIdx.x + l1;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i <= i__1; i += stride)
    {
	double t = u[i * u_dim1 + 1] * clinv;
    atomicAdd(cl, t*t);
    }
}

__global__ void __h12_3__(int l1, int ice, int i3, int i__1, double *u, double *c__, int u_dim1, double *cl){
    int index = blockIdx.x * blockDim.x + threadIdx.x + l1;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i <= i__1; i += stride)
    {
        double cv = c__[i3 + ice * (index - l1)];
        atomicAdd(cl, cv * u[i * u_dim1 + 1]);
    }
}

__global__ void __h12_4__(int l1, int ice, int i4, int i__1, double *u, double *c__, int u_dim1, double sm){
    int index = blockIdx.x * blockDim.x + threadIdx.x + l1;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i <= i__1; i += stride)
    {
        c__[i4 + ice * (index - l1)] += sm * u[i * u_dim1 + 1];
    }
}

/*     COPIES A VECTOR, X, TO A VECTOR, Y, with the given increments */
void dcopy___(int *n_, const double *dx, int incx, 
		     double *dy, int incy)
{
     int n = *n_;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
    
     if (n <= 0) return;
     if (incx == 1 && incy == 1)
	  memcpy(dy, dx, sizeof(double) * ((unsigned) n));
     else if (incx == 0 && incy == 1) {
	  double x = dx[0];
      __copy_1<<<numBlocks, blockSize>>>(n, dy, x);
      cudaDeviceSynchronize();
     }
     else {
      __copy_2<<<numBlocks, blockSize>>>(n, dy, dx, incx, incy, 0);
      cudaDeviceSynchronize();
     }
} /* dcopy___ */

/* CONSTANT TIMES A VECTOR PLUS A VECTOR. */
void daxpy_sl__(int *n_, const double *da_, const double *dx, 
		       int incx, double *dy, int incy)
{
     int n = *n_;  
     double da = *da_;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
 
     if (n <= 0 || da == 0) return;
     __daxpy_sl__<<<numBlocks, blockSize>>>(n, dy, dx, da, incx, incy);
     cudaDeviceSynchronize();
     //for (i = 0; i < n; ++i) dy[i*incy] += da * dx[i*incx];
}

/* dot product dx dot dy. */
double ddot_sl__(int *n_, double *dx, int incx, double *dy, int incy)
{
     int n = *n_;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
     double *sum; 
     cudaMallocManaged(&sum, sizeof(double));
     sum[0] = 0.0;

     if (n <= 0) return 0;
     __ddot_sl__<<<numBlocks, blockSize>>>(n, dx, incx, dy, incy, sum);
     cudaDeviceSynchronize();
     double result = sum[0];
     cudaFree(sum);
     //for (i = 0; i < n; ++i) sum += dx[i*incx] * dy[i*incy];
     return result;
}

/* compute the L2 norm of array DX of length N, stride INCX */
double dnrm2___(int *n_, double *dx, int incx)
{
     int n = *n_;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
     double scale;
     double *xmax, *sum; 
     cudaMallocManaged(&sum, sizeof(double));
     cudaMallocManaged(&xmax, sizeof(double));
     sum[0] = 0.0;
     xmax[0] = 0.0;
     __dnrm2_max___<<<numBlocks, blockSize>>>(n, dx, incx, xmax);
     cudaDeviceSynchronize();

     if (xmax[0] == 0) {
         cudaFree(sum);
         cudaFree(xmax);
         return 0;
     }
     scale = 1.0 / xmax[0];
     __dnrm2_sum___<<<numBlocks, blockSize>>>(n, dx, incx, sum, scale);
     cudaDeviceSynchronize();
     double result = xmax[0] * sqrt(sum[0]);
     cudaFree(sum);
     cudaFree(xmax);
     return result;
}

/* compute the L2 norm of array DX of length N, stride INCX */
void two_dim_dnrm2___(int m, int n, double *dx, double *rnorm, int b_dim1, int offset)
{
     int blockSize = 256;
     int numBlocks;
     double scale;
     double *xmax, *sum; 
     cudaMallocManaged(&sum, m * sizeof(double));
     cudaMallocManaged(&xmax, m * sizeof(double));

     numBlocks = (m + blockSize - 1) / blockSize;
     __zeros<<<numBlocks, blockSize>>>(m, sum,  -1);
     __zeros<<<numBlocks, blockSize>>>(m, xmax,  -1);

     numBlocks = (m * n + blockSize - 1) / blockSize;
     __dnrm2_2d_max___<<<numBlocks, blockSize>>>(m*n, n, dx, 1, xmax, b_dim1, offset);

     //__dnrm2_max___<<<numBlocks, blockSize>>>(n, dx, incx, xmax);
     cudaDeviceSynchronize();

     // if (xmax[0] == 0) {
     //     cudaFree(sum);
     //     cudaFree(xmax);
     //     return 0;
     // }
     //scale = 1.0 / xmax[0];
     __dnrm2_2d_sum___<<<numBlocks, blockSize>>>(m*n, n, dx, 1, sum, xmax, offset, b_dim1);
     cudaDeviceSynchronize();
     numBlocks = (m + blockSize - 1) / blockSize;
     __rnorm<<<numBlocks, blockSize>>>(m, xmax, sum, rnorm);
     cudaDeviceSynchronize();
     // for (int i = 0; i < m; i++){
     //     rnorm[i+1] = xmax[i] * sqrt(sum[i]);
     // }
     // double result = xmax[0] * sqrt(sum[0]);
     cudaFree(sum);
     cudaFree(xmax);
}


/* apply Givens rotation */
void dsrot_(int n, double *dx, int incx, 
		   double *dy, int incy, double *c__, double *s_)
{
     double c = *c__, s = *s_;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
    __dsrot_<<<numBlocks, blockSize>>>(n, dx, incx, dy, incy, c, s);
     cudaDeviceSynchronize();
}

/* construct Givens rotation */
void dsrotg_(double *da, double *db, double *c, double *s)
{
     double absa, absb, roe, scale;

     absa = fabs(*da); absb = fabs(*db);
     if (absa > absb) {
	  roe = *da;
	  scale = absa;
     }
     else {
	  roe = *db;
	  scale = absb;
     }

     if (scale != 0) {
	  double r, iscale = 1 / scale;
	  double tmpa = (*da) * iscale, tmpb = (*db) * iscale;
	  r = (roe < 0 ? -scale : scale) * sqrt((tmpa * tmpa) + (tmpb * tmpb)); 
	  *c = *da / r; *s = *db / r; 
	  *da = r;
	  if (*c != 0 && fabs(*c) <= *s) *db = 1 / *c;
	  else *db = *s;
     }
     else { 
	  *c = 1; 
	  *s = *da = *db = 0;
     }
}

/* scales vector X(n) by constant da */
void dscal_sl__(int *n_, const double *da, double *dx, int incx)
{
     int n = *n_;
     double alpha = *da;
     int blockSize = 256;
     int numBlocks = (n + blockSize - 1) / blockSize;
    __dscal_sl__<<<numBlocks, blockSize>>>(n, alpha, dx, incx);
    cudaDeviceSynchronize();
}

/**************************************************************************/

const int c__0 = 0;
const int c__1 = 1;
const int c__2 = 2;

#define MIN2(a,b) ((a) <= (b) ? (a) : (b))
#define MAX2(a,b) ((a) >= (b) ? (a) : (b))

void h12_(const int *mode, int *lpivot, int *l1, 
		 int *m, double *u, const int *iue, double *up, 
		 double *c__, const int *ice, const int *icv, const int *ncv)
{
    /* Initialized data */

    const double one = 1.;
    int blockSize = 256;
    double *m_cl; 
    int numBlocks; 
    cudaMallocManaged(&m_cl, sizeof(double));
 
    /* System generated locals */
    int u_dim1, u_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    double b;
    int i__, j, i2, i3, i4;
    double cl, sm;
    int incr;
    double clinv;

/*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUN 12 */
/*     TO APPEAR IN 'SOLVING LEAST SQUARES PROBLEMS', PRENTICE-HALL, 1974 */
/*     CONSTRUCTION AND/OR APPLICATION OF A SINGLE */
/*     HOUSEHOLDER TRANSFORMATION  Q = I + U*(U**T)/B */
/*     MODE    = 1 OR 2   TO SELECT ALGORITHM  H1  OR  H2 . */
/*     LPIVOT IS THE INDEX OF THE PIVOT ELEMENT. */
/*     L1,M   IF L1 <= M   THE TRANSFORMATION WILL BE CONSTRUCTED TO */
/*            ZERO ELEMENTS INDEXED FROM L1 THROUGH M. */
/*            IF L1 > M THE SUBROUTINE DOES AN IDENTITY TRANSFORMATION. */
/*     U(),IUE,UP */
/*            ON ENTRY TO H1 U() STORES THE PIVOT VECTOR. */
/*            IUE IS THE STORAGE INCREMENT BETWEEN ELEMENTS. */
/*            ON EXIT FROM H1 U() AND UP STORE QUANTITIES DEFINING */
/*            THE VECTOR U OF THE HOUSEHOLDER TRANSFORMATION. */
/*            ON ENTRY TO H2 U() AND UP */
/*            SHOULD STORE QUANTITIES PREVIOUSLY COMPUTED BY H1. */
/*            THESE WILL NOT BE MODIFIED BY H2. */
/*     C()    ON ENTRY TO H1 OR H2 C() STORES A MATRIX WHICH WILL BE */
/*            REGARDED AS A SET OF VECTORS TO WHICH THE HOUSEHOLDER */
/*            TRANSFORMATION IS TO BE APPLIED. */
/*            ON EXIT C() STORES THE SET OF TRANSFORMED VECTORS. */
/*     ICE    STORAGE INCREMENT BETWEEN ELEMENTS OF VECTORS IN C(). */
/*     ICV    STORAGE INCREMENT BETWEEN VECTORS IN C(). */
/*     NCV    NUMBER OF VECTORS IN C() TO BE TRANSFORMED. */
/*            IF NCV <= 0 NO OPERATIONS WILL BE DONE ON C(). */
    /* Parameter adjustments */
    u_dim1 = *iue;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    --c__;

    /* Function Body */
    if (0 >= *lpivot || *lpivot >= *l1 || *l1 > *m) {
	goto L80;
    }
    cl = (d__1 = u[*lpivot * u_dim1 + 1], fabs(d__1));
    if (*mode == 2) {
	goto L30;
    }
/*     ****** CONSTRUCT THE TRANSFORMATION ****** */
    i__1 = *m;

    numBlocks = (i__1+ 1 + blockSize - 1) / blockSize;
    m_cl[0] = cl;
    __h12_1__<<<numBlocks, blockSize>>>(*l1, i__1, u, u_dim1, m_cl);
    cudaDeviceSynchronize();
    cl = m_cl[0];

    if (cl <= 0.0) {
	goto L80;
    }
    clinv = one / cl;
/* Computing 2nd power */
    d__1 = u[*lpivot * u_dim1 + 1] * clinv;
    m_cl[0] = d__1 * d__1;
    i__1 = *m;
    numBlocks = (i__1+ 1 + blockSize - 1) / blockSize;
    __h12_2__<<<numBlocks, blockSize>>>(*l1, i__1, u, u_dim1, clinv, m_cl);
    cudaDeviceSynchronize();
    sm = m_cl[0];

    cl *= sqrt(sm);
    if (u[*lpivot * u_dim1 + 1] > 0.0) {
	cl = -cl;
    }
    *up = u[*lpivot * u_dim1 + 1] - cl;
    u[*lpivot * u_dim1 + 1] = cl;
    goto L40;
/*     ****** APPLY THE TRANSFORMATION  I+U*(U**T)/B  TO C ****** */
L30:
    if (cl <= 0.0) {
	goto L80;
    }
L40:
    if (*ncv <= 0) {
	goto L80;
    }
    b = *up * u[*lpivot * u_dim1 + 1];
    if (b >= 0.0) {
	goto L80;
    }
    b = one / b;
    i2 = 1 - *icv + *ice * (*lpivot - 1);
    incr = *ice * (*l1 - *lpivot);
    i__1 = *ncv;
    for (j = 1; j <= i__1; ++j)
    {
        i2 += *icv;
        i3 = i2 + incr;
        i4 = i3;
        // sm = c__[i2] * *up;
        i__2 = *m;
        m_cl[0] = c__[i2] * *up;
        numBlocks = (i__2+ 1 + blockSize - 1) / blockSize;
        __h12_3__<<<numBlocks, blockSize>>>(*l1, *ice, i3, i__2, u, c__, u_dim1, m_cl);
        cudaDeviceSynchronize();
        sm = m_cl[0];

        if (sm == 0.0)
        {
            continue;
        }
        sm *= b;
        c__[i2] += sm * *up;
        i__2 = *m;

        numBlocks = (i__2+ 1 + blockSize - 1) / blockSize;
        __h12_4__<<<numBlocks, blockSize>>>(*l1, *ice, i4, i__2, u, c__, u_dim1, sm);
        cudaDeviceSynchronize();
    }
L80:
    cudaFree(m_cl);
    return;
} /* h12_ */

void nnls_(double *a, int *mda, int *m, int *
	n, double *b, double *x, double *rnorm, double *w, 
	double *z__, int *indx, int *mode)
{
    /* Initialized data */

    const double one = 1.;
    const double factor = .01;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    double c__;
    int i__, j, k, l;
    double s, t;
    int ii, jj, ip, iz, jz;
    double up;
    int iz1, iz2, npp1, iter;
    double wmax, alpha, asave;
    int itmax, izmax = 0, nsetp;
    double unorm;

    /*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY: */
    /*     'SOLVING LEAST SQUARES PROBLEMS'. PRENTICE-HALL.1974 */
    /*      **********   NONNEGATIVE LEAST SQUARES   ********** */
    /*     GIVEN AN M BY N MATRIX, A, AND AN M-VECTOR, B, COMPUTE AN */
    /*     N-VECTOR, X, WHICH SOLVES THE LEAST SQUARES PROBLEM */
    /*                  A*X = B  SUBJECT TO  X >= 0 */
    /*     A(),MDA,M,N */
    /*            MDA IS THE FIRST DIMENSIONING PARAMETER FOR THE ARRAY,A(). */
    /*            ON ENTRY A()  CONTAINS THE M BY N MATRIX,A. */
    /*            ON EXIT A() CONTAINS THE PRODUCT Q*A, */
    /*            WHERE Q IS AN M BY M ORTHOGONAL MATRIX GENERATED */
    /*            IMPLICITLY BY THIS SUBROUTINE. */
    /*            EITHER M>=N OR M<N IS PERMISSIBLE. */
    /*            THERE IS NO RESTRICTION ON THE RANK OF A. */
    /*     B()    ON ENTRY B() CONTAINS THE M-VECTOR, B. */
    /*            ON EXIT B() CONTAINS Q*B. */
    /*     X()    ON ENTRY X() NEED NOT BE INITIALIZED. */
    /*            ON EXIT X() WILL CONTAIN THE SOLUTION VECTOR. */
    /*     RNORM  ON EXIT RNORM CONTAINS THE EUCLIDEAN NORM OF THE */
    /*            RESIDUAL VECTOR. */
    /*     W()    AN N-ARRAY OF WORKING SPACE. */
    /*            ON EXIT W() WILL CONTAIN THE DUAL SOLUTION VECTOR. */
    /*            W WILL SATISFY W(I)=0 FOR ALL I IN SET P */
    /*            AND W(I)<=0 FOR ALL I IN SET Z */
    /*     Z()    AN M-ARRAY OF WORKING SPACE. */
    /*     INDX()AN INT WORKING ARRAY OF LENGTH AT LEAST N. */
    /*            ON EXIT THE CONTENTS OF THIS ARRAY DEFINE THE SETS */
    /*            P AND Z AS FOLLOWS: */
    /*            INDX(1)    THRU INDX(NSETP) = SET P. */
    /*            INDX(IZ1)  THRU INDX (IZ2)  = SET Z. */
    /*            IZ1=NSETP + 1 = NPP1, IZ2=N. */
    /*     MODE   THIS IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANING: */
    /*            1    THE SOLUTION HAS BEEN COMPUTED SUCCESSFULLY. */
    /*            2    THE DIMENSIONS OF THE PROBLEM ARE WRONG, */
    /*                 EITHER M <= 0 OR N <= 0. */
    /*            3    ITERATION COUNT EXCEEDED, MORE THAN 3*N ITERATIONS. */
    /* Parameter adjustments */
    --z__;
    --b;
    --indx;
    --w;
    --x;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    int numBlocks;
    int blockSize = 256;

    /* Function Body */
    /*     revised          Dieter Kraft, March 1983 */
    *mode = 2;
    if (*m <= 0 || *n <= 0)
    {
        goto L290;
    }
    *mode = 1;
    iter = 0;
    itmax = *n * 3;
    /* STEP ONE (INITIALIZE) */
    i__1 = *n;
    numBlocks = (i__1 + blockSize - 1) / blockSize;
    __assign<<<numBlocks, blockSize>>>(i__1, indx);
    cudaDeviceSynchronize();

    // for (i__ = 1; i__ <= i__1; ++i__) {
    /* L// 100: */
    //ind// x[i__] = i__;
    // }
    iz1 = 1;
    iz2 = *n;
    nsetp = 0;
    npp1 = 1;
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
/* STEP TWO (COMPUTE DUAL VARIABLES) */
/* .....ENTRY LOOP A */
L110:
    if (iz1 > iz2 || nsetp >= *m)
    {
        goto L280;
    }
    i__1 = iz2;

    double *m_cl;
    cudaMallocManaged(&m_cl, (i__1 - iz1 + 1) * sizeof(double));

    numBlocks = ((i__1 - iz1 + 1) + blockSize - 1) / blockSize;
    __copy_1<<<numBlocks, blockSize>>>((i__1 - iz1 + 1), m_cl, 0.0);

    numBlocks = ((i__1 - iz1 + 1) * (*m - nsetp) + blockSize - 1) / blockSize;
    __two_vector_dot__<<<numBlocks, blockSize>>>((i__1 - iz1 + 1) * (*m - nsetp), *m - nsetp, a, 1, b, 1, m_cl, indx, iz1, a_dim1, npp1);

    // __assign<<<numBlocks, blockSize>>>(i__1, indx);

    numBlocks = ((i__1 - iz1 + 1) + blockSize - 1) / blockSize;
    __two_vector_assign__<<<numBlocks, blockSize>>>((i__1 - iz1 + 1), indx, m_cl, iz1, w);
    cudaDeviceSynchronize();

/* STEP THREE (TEST DUAL VARIABLES) */
L130:
    wmax = 0.0;
    i__2 = iz2;

    int *m_izmax;
    cudaMallocManaged(&m_izmax, sizeof(int));

    m_izmax[0] = 0;

    numBlocks = ((i__2 - iz1 + 1) + blockSize - 1) / blockSize;
    __arg_max___<<<numBlocks, blockSize>>>(i__2, iz1, w, m_izmax, indx);
    cudaDeviceSynchronize();
    izmax = m_izmax[0];
    if (izmax == 0)
    {
        wmax = 0.0;
    }
    else
    {
        wmax = w[indx[izmax]];
    }
    /* .....EXIT LOOP A */
    if (wmax <= 0.0)
    {
        goto L280;
    }
    iz = izmax;
    j = indx[iz];
    /* STEP FOUR (TEST INDX J FOR LINEAR DEPENDENCY) */
    asave = a[npp1 + j * a_dim1];
    i__2 = npp1 + 1;
    h12_(&c__1, &npp1, &i__2, m, &a[j * a_dim1 + 1], &c__1, &up, &z__[1], &c__1, &c__1, &c__0);
    unorm = dnrm2___(&nsetp, &a[j * a_dim1 + 1], 1);
    t = factor * (d__1 = a[npp1 + j * a_dim1], fabs(d__1));
    d__1 = unorm + t;
    if (d__1 - unorm <= 0.0)
    {
        goto L150;
    }
    dcopy___(m, &b[1], 1, &z__[1], 1);
    i__2 = npp1 + 1;
    h12_(&c__2, &npp1, &i__2, m, &a[j * a_dim1 + 1], &c__1, &up, &z__[1], &c__1, &c__1, &c__1);
    if (z__[npp1] / a[npp1 + j * a_dim1] > 0.0)
    {
        goto L160;
    }
L150:
    a[npp1 + j * a_dim1] = asave;
    w[j] = 0.0;
    goto L130;
/* STEP FIVE (ADD COLUMN) */
L160:
    dcopy___(m, &z__[1], 1, &b[1], 1);
    indx[iz] = indx[iz1];
    indx[iz1] = j;
    ++iz1;
    nsetp = npp1;
    ++npp1;
    i__2 = iz2;
    for (jz = iz1; jz <= i__2; ++jz)
    {
        jj = indx[jz];
        /* L170: */
        h12_(&c__2, &nsetp, &npp1, m, &a[j * a_dim1 + 1], &c__1, &up, &a[jj * a_dim1 + 1], &c__1, mda, &c__1);
    }
    k = MIN2(npp1, *mda);
    w[j] = 0.0;
    i__2 = *m - nsetp;
    dcopy___(&i__2, &w[j], 0, &a[k + j * a_dim1], 1);
/* STEP SIX (SOLVE LEAST SQUARES SUB-PROBLEM) */
/* .....ENTRY LOOP B */
L180:
    for (ip = nsetp; ip >= 1; --ip)
    {
        if (ip == nsetp)
        {
            goto L190;
        }
        d__1 = -z__[ip + 1];
        daxpy_sl__(&ip, &d__1, &a[jj * a_dim1 + 1], 1, &z__[1], 1);
    L190:
        jj = indx[ip];
        /* L200: */
        z__[ip] /= a[ip + jj * a_dim1];
    }
    ++iter;
    if (iter <= itmax)
    {
        goto L220;
    }
L210:
    *mode = 3;
    goto L280;
/* STEP SEVEN TO TEN (STEP LENGTH ALGORITHM) */
L220:
    alpha = one;
    jj = 0;
    i__2 = nsetp;

    int *m_jj;
    cudaMallocManaged(&m_jj, sizeof(int));
    m_jj[0] = 0;

    numBlocks = ( i__2  + blockSize - 1) / blockSize;
    __arg_min2___<<<numBlocks, blockSize>>>(i__2, z__, x, indx, m_jj, alpha);
    cudaDeviceSynchronize();

    if (m_jj[0] != 0){
        jj = m_jj[0];
        l = indx[jj];
        alpha = -x[l] / (z__[jj] - x[l]);
    }

    // for (ip = 1; ip <= i__2; ++ip)
    // {
    //     if (z__[ip] > 0.0)
    //     {
    //         goto L230;
    //     }
    //     l = indx[ip];
    //     t = -x[l] / (z__[ip] - x[l]);
    //     if (alpha < t)
    //     {
    //         goto L230;
    //     }
    //     alpha = t;
    //     jj = ip;
    // L230:;
    // }
    i__2 = nsetp;

    numBlocks = ( i__2  + blockSize - 1) / blockSize;
    __assign_3___<<<numBlocks, blockSize>>>(i__2, z__, x, indx, one, alpha);
    cudaDeviceSynchronize();
    // for (ip = 1; ip <= i__2; ++ip)
    // {
    //     l = indx[ip];
    //     /* L240: */
    //     x[l] = (one - alpha) * x[l] + alpha * z__[ip];
    // }
    /* .....EXIT LOOP B */
    if (jj == 0)
    {
        goto L110;
    }
    /* STEP ELEVEN (DELETE COLUMN) */
    i__ = indx[jj];
L250:
    x[i__] = 0.0;
    ++jj;
    i__2 = nsetp;
    for (j = jj; j <= i__2; ++j)
    {
        ii = indx[j];
        indx[j - 1] = ii;
        dsrotg_(&a[j - 1 + ii * a_dim1], &a[j + ii * a_dim1], &c__, &s);
        t = a[j - 1 + ii * a_dim1];
        dsrot_(*n, &a[j - 1 + a_dim1], *mda, &a[j + a_dim1], *mda, &c__, &s);
        a[j - 1 + ii * a_dim1] = t;
        a[j + ii * a_dim1] = 0.0;
        /* L260: */
        dsrot_(1, &b[j - 1], 1, &b[j], 1, &c__, &s);
    }
    npp1 = nsetp;
    --nsetp;
    --iz1;
    indx[iz1] = i__;
    if (nsetp <= 0)
    {
        goto L210;
    }
    i__2 = nsetp;
    for (jj = 1; jj <= i__2; ++jj)
    {
        i__ = indx[jj];
        if (x[i__] <= 0.0)
        {
            goto L250;
        }
        /* L270: */
    }
    dcopy___(m, &b[1], 1, &z__[1], 1);
    goto L180;
/* STEP TWELVE (SOLUTION) */
L280:
    k = MIN2(npp1, *m);
    i__2 = *m - nsetp;
    *rnorm = dnrm2___(&i__2, &b[k], 1);
    if (npp1 > *m)
    {
        w[1] = 0.0;
        dcopy___(n, &w[1], 0, &w[1], 1);
    }
/* END OF SUBROUTINE NNLS */
L290:
    cudaFree(m_cl);
    cudaFree(m_izmax);
    cudaFree(m_jj);
    return;
} /* nnls_ */

void ldp_(double *g, int *mg, int *m, int *n, 
	double *h__, double *x, double *xnorm, double *w, 
	int *indx, int *mode)
{
    /* Initialized data */

    const double one = 1.;

    /* System generated locals */
    int g_dim1, g_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, j, n1, if__, iw, iy, iz;
    double fac;
    double rnorm;
    int iwdual;

    /*                     T */
    /*     MINIMIZE   1/2 X X    SUBJECT TO   G * X >= H. */
    /*       C.L. LAWSON, R.J. HANSON: 'SOLVING LEAST SQUARES PROBLEMS' */
    /*       PRENTICE HALL, ENGLEWOOD CLIFFS, NEW JERSEY, 1974. */
    /*     PARAMETER DESCRIPTION: */
    /*     G(),MG,M,N   ON ENTRY G() STORES THE M BY N MATRIX OF */
    /*                  LINEAR INEQUALITY CONSTRAINTS. G() HAS FIRST */
    /*                  DIMENSIONING PARAMETER MG */
    /*     H()          ON ENTRY H() STORES THE M VECTOR H REPRESENTING */
    /*                  THE RIGHT SIDE OF THE INEQUALITY SYSTEM */
    /*     REMARK: G(),H() WILL NOT BE CHANGED DURING CALCULATIONS BY LDP */
    /*     X()          ON ENTRY X() NEED NOT BE INITIALIZED. */
    /*                  ON EXIT X() STORES THE SOLUTION VECTOR X IF MODE=1. */
    /*     XNORM        ON EXIT XNORM STORES THE EUCLIDIAN NORM OF THE */
    /*                  SOLUTION VECTOR IF COMPUTATION IS SUCCESSFUL */
    /*     W()          W IS A ONE DIMENSIONAL WORKING SPACE, THE LENGTH */
    /*                  OF WHICH SHOULD BE AT LEAST (M+2)*(N+1) + 2*M */
    /*                  ON EXIT W() STORES THE LAGRANGE MULTIPLIERS */
    /*                  ASSOCIATED WITH THE CONSTRAINTS */
    /*                  AT THE SOLUTION OF PROBLEM LDP */
    /*     INDX()      INDX() IS A ONE DIMENSIONAL INT WORKING SPACE */
    /*                  OF LENGTH AT LEAST M */
    /*     MODE         MODE IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING */
    /*                  MEANINGS: */
    /*          MODE=1: SUCCESSFUL COMPUTATION */
    /*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N.LE.0) */
    /*               3: ITERATION COUNT EXCEEDED BY NNLS */
    /*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
    /* Parameter adjustments */
    int numBlocks; 
    int blockSize = 256;
 
    --indx;
    --h__;
    --x;
    g_dim1 = *mg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    --w;

    /* Function Body */
    *mode = 2;
    if (*n <= 0)
    {
        goto L50;
    }
    /*  STATE DUAL PROBLEM */
    *mode = 1;
    x[1] = 0.0;
    dcopy___(n, &x[1], 0, &x[1], 1);
    *xnorm = 0.0;
    if (*m == 0)
    {
        goto L50;
    }
    iw = 0;
    i__1 = *m;
    i__2 = *n;

    numBlocks = ( i__1 * (i__2+1)  + blockSize - 1) / blockSize;
    __assign_4___<<<numBlocks, blockSize>>>(i__1 * (i__2+1), i__2, g, w, h__, g_dim1);
    cudaDeviceSynchronize();
    iw += i__1 * (i__2+1);
    if__ = iw + 1;
    i__1 = *n;

    numBlocks = ( i__1  + blockSize - 1) / blockSize;
    __zeros<<<numBlocks, blockSize>>>(i__1, w, iw);
    cudaDeviceSynchronize();
    iw += i__1;

    w[iw + 1] = one;
    n1 = *n + 1;
    iz = iw + 2;
    iy = iz + n1;
    iwdual = iy + *m;
    /*  SOLVE DUAL PROBLEM */
    nnls_(&w[1], &n1, &n1, m, &w[if__], &w[iy], &rnorm, &w[iwdual], &w[iz], &indx[1], mode);
    if (*mode != 1)
    {
        goto L50;
    }
    *mode = 4;
    if (rnorm <= 0.0)
    {
        goto L50;
    }
    /*  COMPUTE SOLUTION OF PRIMAL PROBLEM */
    fac = one - ddot_sl__(m, &h__[1], 1, &w[iy], 1);
    d__1 = one + fac;
    if (d__1 - one <= 0.0)
    {
        goto L50;
    }
    *mode = 1;
    fac = one / fac;
    i__1 = *n;

    numBlocks = ( i__1  + blockSize - 1) / blockSize;
    __zeros<<<numBlocks, blockSize>>>(i__1, x, 0);
    numBlocks = ( i__1 * *m  + blockSize - 1) / blockSize;
    __two_vector_dot_scale__<<<numBlocks, blockSize>>>(i__1* *m, *m, g, 1, &w[iy], 1, x, fac, g_dim1);
    cudaDeviceSynchronize();

    *xnorm = dnrm2___(n, &x[1], 1);
    /*  COMPUTE LAGRANGE MULTIPLIERS FOR PRIMAL PROBLEM */
    w[1] = 0.0;
    dcopy___(m, &w[1], 0, &w[1], 1);
    daxpy_sl__(m, &fac, &w[iy], 1, &w[1], 1);
/*  END OF SUBROUTINE LDP */
L50:
    return;
} /* ldp_ */

void lsi_(double *e, double *f, double *g, 
	double *h__, int *le, int *me, int *lg, int *mg, 
	int *n, double *x, double *xnorm, double *w, int *
	jw, int *mode)
{
    /* Initialized data */

    const double epmach = 2.22e-16;
    const double one = 1.;

    /* System generated locals */
    int e_dim1, e_offset, g_dim1, g_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j;
    double t;

    /*     FOR MODE=1, THE SUBROUTINE RETURNS THE SOLUTION X OF */
    /*     INEQUALITY CONSTRAINED LINEAR LEAST SQUARES PROBLEM: */
    /*                    MIN ||E*X-F|| */
    /*                     X */
    /*                    S.T.  G*X >= H */
    /*     THE ALGORITHM IS BASED ON QR DECOMPOSITION AS DESCRIBED IN */
    /*     CHAPTER 23.5 OF LAWSON & HANSON: SOLVING LEAST SQUARES PROBLEMS */
    /*     THE FOLLOWING DIMENSIONS OF THE ARRAYS DEFINING THE PROBLEM */
    /*     ARE NECESSARY */
    /*     DIM(E) :   FORMAL (LE,N),    ACTUAL (ME,N) */
    /*     DIM(F) :   FORMAL (LE  ),    ACTUAL (ME  ) */
    /*     DIM(G) :   FORMAL (LG,N),    ACTUAL (MG,N) */
    /*     DIM(H) :   FORMAL (LG  ),    ACTUAL (MG  ) */
    /*     DIM(X) :   N */
    /*     DIM(W) :   (N+1)*(MG+2) + 2*MG */
    /*     DIM(JW):   LG */
    /*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS E, F, G, AND H. */
    /*     ON RETURN, ALL ARRAYS WILL BE CHANGED BY THE SUBROUTINE. */
    /*     X     STORES THE SOLUTION VECTOR */
    /*     XNORM STORES THE RESIDUUM OF THE SOLUTION IN EUCLIDIAN NORM */
    /*     W     STORES THE VECTOR OF LAGRANGE MULTIPLIERS IN ITS FIRST */
    /*           MG ELEMENTS */
    /*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
    /*          MODE=1: SUCCESSFUL COMPUTATION */
    /*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
    /*               3: ITERATION COUNT EXCEEDED BY NNLS */
    /*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
    /*               5: MATRIX E IS NOT OF FULL RANK */
    /*     03.01.1980, DIETER KRAFT: CODED */
    /*     20.03.1987, DIETER KRAFT: REVISED TO FORTRAN 77 */
    /* Parameter adjustments */
    --f;
    --jw;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    --w;

    /* Function Body */
    /*  QR-FACTORS OF E AND APPLICATION TO F */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        /* Computing MIN */
        i__2 = i__ + 1;
        j = MIN2(i__2, *n);
        i__2 = i__ + 1;
        i__3 = *n - i__;
        h12_(&c__1, &i__, &i__2, me, &e[i__ * e_dim1 + 1], &c__1, &t, &e[j * e_dim1 + 1], &c__1, le, &i__3);
        /* L10: */
        i__2 = i__ + 1;
        h12_(&c__2, &i__, &i__2, me, &e[i__ * e_dim1 + 1], &c__1, &t, &f[1], &c__1, &c__1, &c__1);
    }
    /*  TRANSFORM G AND H TO GET LEAST DISTANCE PROBLEM */
    *mode = 5;
    i__2 = *mg;
    for (i__ = 1; i__ <= i__2; ++i__)
    {
        i__1 = *n;
        for (j = 1; j <= i__1; ++j)
        {
            if (fabs(e[j + j * e_dim1]) < epmach)
            {
                goto L50;
            }
            /* L20: */
            i__3 = j - 1;
            g[i__ + j * g_dim1] = (g[i__ + j * g_dim1] - ddot_sl__(&i__3, &g[i__ + g_dim1], *lg, &e[j * e_dim1 + 1], 1)) / e[j + j *
                                                                                                                                     e_dim1];
        }
        /* L30: */
        h__[i__] -= ddot_sl__(n, &g[i__ + g_dim1], *lg, &f[1], 1);
    }
    /*  SOLVE LEAST DISTANCE PROBLEM */
    ldp_(&g[g_offset], lg, mg, n, &h__[1], &x[1], xnorm, &w[1], &jw[1], mode);
    if (*mode != 1)
    {
        goto L50;
    }
    /*  SOLUTION OF ORIGINAL PROBLEM */
    daxpy_sl__(n, &one, &f[1], 1, &x[1], 1);
    for (i__ = *n; i__ >= 1; --i__)
    {
        /* Computing MIN */
        i__2 = i__ + 1;
        j = MIN2(i__2, *n);
        /* L40: */
        i__2 = *n - i__;
        x[i__] = (x[i__] - ddot_sl__(&i__2, &e[i__ + j * e_dim1], *le, &x[j], 1)) / e[i__ + i__ * e_dim1];
    }
    /* Computing MIN */
    i__2 = *n + 1;
    j = MIN2(i__2, *me);
    i__2 = *me - *n;
    t = dnrm2___(&i__2, &f[j], 1);
    *xnorm = sqrt(*xnorm * *xnorm + t * t);
/*  END OF SUBROUTINE LSI */
L50:
    return;
} /* lsi_ */

void hfti_(double *a, int *mda, int *m, int *
	n, double *b, int *mdb, const int *nb, double *tau, int 
	*krank, double *rnorm, double *h__, double *g, int *
	ip)
{
    /* Initialized data */

    const double factor = .001;

    /* System generated locals */
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    int jb, kp1;
    double tmp, hmax;
    int lmax, ldiag;

    /*     RANK-DEFICIENT LEAST SQUARES ALGORITHM AS DESCRIBED IN: */
    /*     C.L.LAWSON AND R.J.HANSON, JET PROPULSION LABORATORY, 1973 JUN 12 */
    /*     TO APPEAR IN 'SOLVING LEAST SQUARES PROBLEMS', PRENTICE-HALL, 1974 */
    /*     A(*,*),MDA,M,N   THE ARRAY A INITIALLY CONTAINS THE M x N MATRIX A */
    /*                      OF THE LEAST SQUARES PROBLEM AX = B. */
    /*                      THE FIRST DIMENSIONING PARAMETER MDA MUST SATISFY */
    /*                      MDA >= M. EITHER M >= N OR M < N IS PERMITTED. */
    /*                      THERE IS NO RESTRICTION ON THE RANK OF A. */
    /*                      THE MATRIX A WILL BE MODIFIED BY THE SUBROUTINE. */
    /*     B(*,*),MDB,NB    IF NB = 0 THE SUBROUTINE WILL MAKE NO REFERENCE */
    /*                      TO THE ARRAY B. IF NB > 0 THE ARRAY B() MUST */
    /*                      INITIALLY CONTAIN THE M x NB MATRIX B  OF THE */
    /*                      THE LEAST SQUARES PROBLEM AX = B AND ON RETURN */
    /*                      THE ARRAY B() WILL CONTAIN THE N x NB SOLUTION X. */
    /*                      IF NB>1 THE ARRAY B() MUST BE DOUBLE SUBSCRIPTED */
    /*                      WITH FIRST DIMENSIONING PARAMETER MDB>=MAX(M,N), */
    /*                      IF NB=1 THE ARRAY B() MAY BE EITHER SINGLE OR */
    /*                      DOUBLE SUBSCRIPTED. */
    /*     TAU              ABSOLUTE TOLERANCE PARAMETER FOR PSEUDORANK */
    /*                      DETERMINATION, PROVIDED BY THE USER. */
    /*     KRANK            PSEUDORANK OF A, SET BY THE SUBROUTINE. */
    /*     RNORM            ON EXIT, RNORM(J) WILL CONTAIN THE EUCLIDIAN */
    /*                      NORM OF THE RESIDUAL VECTOR FOR THE PROBLEM */
    /*                      DEFINED BY THE J-TH COLUMN VECTOR OF THE ARRAY B. */
    /*     H(), G()         ARRAYS OF WORKING SPACE OF LENGTH >= N. */
    /*     IP()             INT ARRAY OF WORKING SPACE OF LENGTH >= N */
    /*                      RECORDING PERMUTATION INDICES OF COLUMN VECTORS */
    /* Parameter adjustments */
    --ip;
    --g;
    --h__;
    a_dim1 = *mda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --rnorm;
    b_dim1 = *mdb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    int numBlocks; 
    int blockSize = 256;
    int *m_izmax;
    cudaMallocManaged(&m_izmax, sizeof(int));

    /* Function Body */
    k = 0;
    ldiag = MIN2(*m, *n);
    if (ldiag <= 0)
    {
        goto L270;
    }
    /*   COMPUTE LMAX */
    i__1 = ldiag;
    for (j = 1; j <= i__1; ++j)
    {
        if (j == 1)
        {
            goto L20;
        }
        lmax = j;
        i__2 = *n;
        m_izmax[0] = j;
        numBlocks = ( (i__2 - j  + 1)  + blockSize - 1) / blockSize;
        __arg_max_h__<<<numBlocks, blockSize>>>(i__2, j, h__, m_izmax, a, a_dim1);
        cudaDeviceSynchronize();
        lmax = m_izmax[0];
        d__1 = hmax + factor * h__[lmax];
        if (d__1 - hmax > 0.0)
        {
            goto L50;
        }
    L20:
        lmax = j;
        i__2 = *n;
        i__3 = *m;
        numBlocks = ( (i__2 - j  + 1) * (i__3 - j + 1)  + blockSize - 1) / blockSize;
        __hfti_two_vector<<<numBlocks, blockSize>>>((i__2 - j  + 1) * (i__3 - j + 1), i__3 - j + 1, a, h__, j, a_dim1);
        cudaDeviceSynchronize();
        // for (l = j; l <= i__2; ++l)
        // {
        //     h__[l] = 0.0;
        //     for (i__ = j; i__ <= i__3; ++i__)
        //     {
        //         /* L30: */
        //         /* Computing 2nd power */
        //         d__1 = a[i__ + l * a_dim1];
        //         h__[l] += d__1 * d__1;
        //     }
        // }
        m_izmax[0] = j;
        numBlocks = ( (i__2 - j  + 1) + blockSize - 1) / blockSize;
        __arg_max_v__<<<numBlocks, blockSize>>>(i__2, j, h__, m_izmax);
        cudaDeviceSynchronize();

        // for (l = j; l <= i__2; ++l)
        // {
        //     /* L40: */
        //     if (h__[l] > h__[lmax])
        //     {
        //         lmax = l;
        //     }
        // }
        lmax = m_izmax[0];
        hmax = h__[lmax];
    /*   COLUMN INTERCHANGES IF NEEDED */
    L50:
        ip[j] = lmax;
        if (ip[j] == j)
        {
            goto L70;
        }
        i__2 = *m;
        numBlocks = ( i__2 + blockSize - 1) / blockSize;
        __hfti_swap_max<<<numBlocks, blockSize>>>(i__2,  a, lmax, a_dim1, j);
        cudaDeviceSynchronize();
        // for (i__ = 1; i__ <= i__2; ++i__)
        // {
        //     tmp = a[i__ + j * a_dim1];
        //     a[i__ + j * a_dim1] = a[i__ + lmax * a_dim1];
        //     /* L60: */
        //     a[i__ + lmax * a_dim1] = tmp;
        // }
        h__[lmax] = h__[j];
    /*   J-TH TRANSFORMATION AND APPLICATION TO A AND B */
    L70:
        /* Computing MIN */
        i__2 = j + 1;
        i__ = MIN2(i__2, *n);
        i__2 = j + 1;
        i__3 = *n - j;
        h12_(&c__1, &j, &i__2, m, &a[j * a_dim1 + 1], &c__1, &h__[j], &a[i__ * a_dim1 + 1], &c__1, mda, &i__3);
        /* L80: */
        i__2 = j + 1;
        h12_(&c__2, &j, &i__2, m, &a[j * a_dim1 + 1], &c__1, &h__[j], &b[b_offset], &c__1, mdb, nb);
    }
    /*   DETERMINE PSEUDORANK */
    i__2 = ldiag;
    numBlocks = ( i__2   + blockSize - 1) / blockSize;
    m_izmax[0] = -1;
    __arg_hfti_find___<<<numBlocks, blockSize>>>(i__2, a, *tau, a_dim1, m_izmax);
    cudaDeviceSynchronize();
    // for (j = 1; j <= i__2; ++j)
    // {
    //     /* L90: */
    //     if (fabs(a[j + j * a_dim1]) <= *tau)
    //     {
    //         goto L100;
    //     }
    // }
    if (m_izmax[0] >= 0)
    {
        goto L100;
    }
    k = ldiag;
    goto L110;
L100:
    k = j - 1;
L110:
    kp1 = k + 1;
    /*   NORM OF RESIDUALS */
    i__2 = *nb;

    two_dim_dnrm2___(i__2, *m -k, b, rnorm, b_dim1, kp1);
    // for (jb = 1; jb <= i__2; ++jb)
    // {
    //     /* L130: */
    //     i__1 = *m - k;
    //     rnorm[jb] = dnrm2___(&i__1, &b[kp1 + jb * b_dim1], 1);
    // }
    if (k > 0)
    {
        goto L160;
    }
    i__1 = *nb;
    i__2 = *n;

    numBlocks = ( i__1 * i__2  + blockSize - 1) / blockSize;
    m_izmax[0] = -1;
    __hfti_zero___<<<numBlocks, blockSize>>>(i__1* i__2, i__2, b, b_dim1);
    cudaDeviceSynchronize();
    // for (jb = 1; jb <= i__1; ++jb)
    // {
    //     for (i__ = 1; i__ <= i__2; ++i__)
    //     {
    //         /* L150: */
    //         b[i__ + jb * b_dim1] = 0.0;
    //     }
    // }
    goto L270;
L160:
    if (k == *n)
    {
        goto L180;
    }
    /*   HOUSEHOLDER DECOMPOSITION OF FIRST K ROWS */
    for (i__ = k; i__ >= 1; --i__)
    {
        /* L170: */
        i__2 = i__ - 1;
        h12_(&c__1, &i__, &kp1, n, &a[i__ + a_dim1], mda, &g[i__], &a[a_offset], mda, &c__1, &i__2);
    }
L180:
    i__2 = *nb;
    for (jb = 1; jb <= i__2; ++jb)
    {
        /*   SOLVE K*K TRIANGULAR SYSTEM */
        for (i__ = k; i__ >= 1; --i__)
        {
            /* Computing MIN */
            i__1 = i__ + 1;
            j = MIN2(i__1, *n);
            /* L210: */
            i__1 = k - i__;
            b[i__ + jb * b_dim1] = (b[i__ + jb * b_dim1] - ddot_sl__(&i__1, &a[i__ + j * a_dim1], *mda, &b[j + jb * b_dim1], 1)) /
                                   a[i__ + i__ * a_dim1];
        }
        /*   COMPLETE SOLUTION VECTOR */
        if (k == *n)
        {
            goto L240;
        }
        i__1 = *n;

        numBlocks = ( (i__1 - kp1 + 1)  + blockSize - 1) / blockSize;
        __hfti_set_zero___<<<numBlocks, blockSize>>>(i__1, kp1, b, b_dim1, jb);
        cudaDeviceSynchronize();
        // for (j = kp1; j <= i__1; ++j)
        // {
        //     /* L220: */
        //     b[j + jb * b_dim1] = 0.0;
        // }
        i__1 = k;
        for (i__ = 1; i__ <= i__1; ++i__)
        {
            /* L230: */
            h12_(&c__2, &i__, &kp1, n, &a[i__ + a_dim1], mda, &g[i__], &b[jb * b_dim1 + 1], &c__1, mdb, &c__1);
        }
    /*   REORDER SOLUTION ACCORDING TO PREVIOUS COLUMN INTERCHANGES */
    L240:
        for (j = ldiag; j >= 1; --j)
        {
            if (ip[j] == j)
            {
                goto L250;
            }
            l = ip[j];
            tmp = b[l + jb * b_dim1];
            b[l + jb * b_dim1] = b[j + jb * b_dim1];
            b[j + jb * b_dim1] = tmp;
        L250:;
        }
    }
L270:
    *krank = k;
    cudaFree(m_izmax);
} /* hfti_ */

void lsei_(double *c__, double *d__, double *e, 
	double *f, double *g, double *h__, int *lc, int *
	mc, int *le, int *me, int *lg, int *mg, int *n, 
	double *x, double *xnrm, double *w, int *jw, int *
	mode)
{
    /* Initialized data */

    const double epmach = 2.22e-16;

    /* System generated locals */
    int c_dim1, c_offset, e_dim1, e_offset, g_dim1, g_offset, i__1, i__2,
        i__3;
    double d__1;

    /* Local variables */
    int i__, j, k, l;
    double t;
    int ie, if__, ig, iw, mc1;
    int krank;

    /*     FOR MODE=1, THE SUBROUTINE RETURNS THE SOLUTION X OF */
    /*     EQUALITY & INEQUALITY CONSTRAINED LEAST SQUARES PROBLEM LSEI : */
    /*                MIN ||E*X - F|| */
    /*                 X */
    /*                S.T.  C*X  = D, */
    /*                      G*X >= H. */
    /*     USING QR DECOMPOSITION & ORTHOGONAL BASIS OF NULLSPACE OF C */
    /*     CHAPTER 23.6 OF LAWSON & HANSON: SOLVING LEAST SQUARES PROBLEMS. */
    /*     THE FOLLOWING DIMENSIONS OF THE ARRAYS DEFINING THE PROBLEM */
    /*     ARE NECESSARY */
    /*     DIM(E) :   FORMAL (LE,N),    ACTUAL (ME,N) */
    /*     DIM(F) :   FORMAL (LE  ),    ACTUAL (ME  ) */
    /*     DIM(C) :   FORMAL (LC,N),    ACTUAL (MC,N) */
    /*     DIM(D) :   FORMAL (LC  ),    ACTUAL (MC  ) */
    /*     DIM(G) :   FORMAL (LG,N),    ACTUAL (MG,N) */
    /*     DIM(H) :   FORMAL (LG  ),    ACTUAL (MG  ) */
    /*     DIM(X) :   FORMAL (N   ),    ACTUAL (N   ) */
    /*     DIM(W) :   2*MC+ME+(ME+MG)*(N-MC)  for LSEI */
    /*              +(N-MC+1)*(MG+2)+2*MG     for LSI */
    /*     DIM(JW):   MAX(MG,L) */
    /*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS C, D, E, F, G, AND H. */
    /*     ON RETURN, ALL ARRAYS WILL BE CHANGED BY THE SUBROUTINE. */
    /*     X     STORES THE SOLUTION VECTOR */
    /*     XNORM STORES THE RESIDUUM OF THE SOLUTION IN EUCLIDIAN NORM */
    /*     W     STORES THE VECTOR OF LAGRANGE MULTIPLIERS IN ITS FIRST */
    /*           MC+MG ELEMENTS */
    /*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
    /*          MODE=1: SUCCESSFUL COMPUTATION */
    /*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
    /*               3: ITERATION COUNT EXCEEDED BY NNLS */
    /*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
    /*               5: MATRIX E IS NOT OF FULL RANK */
    /*               6: MATRIX C IS NOT OF FULL RANK */
    /*               7: RANK DEFECT IN HFTI */
    /*     18.5.1981, DIETER KRAFT, DFVLR OBERPFAFFENHOFEN */
    /*     20.3.1987, DIETER KRAFT, DFVLR OBERPFAFFENHOFEN */
    /* Parameter adjustments */
    --d__;
    --f;
    --h__;
    --x;
    g_dim1 = *lg;
    g_offset = 1 + g_dim1;
    g -= g_offset;
    e_dim1 = *le;
    e_offset = 1 + e_dim1;
    e -= e_offset;
    c_dim1 = *lc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --w;
    --jw;
    int numBlocks;
    int blockSize = 256;
    double *m_dot;

    /* Function Body */
    *mode = 2;
    if (*mc > *n)
    {
        goto L75;
    }
    l = *n - *mc;
    mc1 = *mc + 1;
    iw = (l + 1) * (*mg + 2) + (*mg << 1) + *mc;
    ie = iw + *mc + 1;
    if__ = ie + *me * l;
    ig = if__ + *me;
    /*  TRIANGULARIZE C AND APPLY FACTORS TO E AND G */
    i__1 = *mc;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        /* Computing MIN */
        i__2 = i__ + 1;
        j = MIN2(i__2, *lc);
        i__2 = i__ + 1;
        i__3 = *mc - i__;
        h12_(&c__1, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &c__[j + c_dim1], lc, &c__1, &i__3);
        i__2 = i__ + 1;
        h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &e[e_offset], le, &c__1, me);
        /* L10: */
        i__2 = i__ + 1;
        h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &g[g_offset], lg, &c__1, mg);
    }
    /*  SOLVE C*X=D AND MODIFY F */
    *mode = 6;
    i__2 = *mc;
    for (i__ = 1; i__ <= i__2; ++i__)
    {
        if ((d__1 = c__[i__ + i__ * c_dim1], fabs(d__1)) < epmach)
        {
            goto L75;
        }
        i__1 = i__ - 1;
        x[i__] = (d__[i__] - ddot_sl__(&i__1, &c__[i__ + c_dim1], *lc, &x[1], 1)) / c__[i__ + i__ * c_dim1];
        /* L15: */
    }
    *mode = 1;
    w[mc1] = 0.0;
    i__2 = *mg; /* BUGFIX for *mc == *n: changed from *mg - *mc, SGJ 2010 */
    dcopy___(&i__2, &w[mc1], 0, &w[mc1], 1);
    if (*mc == *n)
    {
        goto L50;
    }
    i__2 = *me;
    for (i__ = 1; i__ <= i__2; ++i__)
    {
        /* L20: */
        w[if__ - 1 + i__] = f[i__] - ddot_sl__(mc, &e[i__ + e_dim1], *le, &x[1], 1);
    }
    /*  STORE TRANSFORMED E & G */
    i__2 = *me;
    for (i__ = 1; i__ <= i__2; ++i__)
    {
        /* L25: */
        dcopy___(&l, &e[i__ + mc1 * e_dim1], *le, &w[ie - 1 + i__], *me);
    }
    i__2 = *mg;
    for (i__ = 1; i__ <= i__2; ++i__)
    {
        /* L30: */
        dcopy___(&l, &g[i__ + mc1 * g_dim1], *lg, &w[ig - 1 + i__], *mg);
    }
    if (*mg > 0)
    {
        goto L40;
    }
    /*  SOLVE LS WITHOUT INEQUALITY CONSTRAINTS */
    *mode = 7;
    k = MAX2(*le, *n);
    t = sqrt(epmach);
    hfti_(&w[ie], me, me, &l, &w[if__], &k, &c__1, &t, &krank, xnrm, &w[1], &w[l + 1], &jw[1]);
    dcopy___(&l, &w[if__], 1, &x[mc1], 1);
    if (krank != l)
    {
        goto L75;
    }
    *mode = 1;
    goto L50;
/*  MODIFY H AND SOLVE INEQUALITY CONSTRAINED LS PROBLEM */
L40:
    i__2 = *mg;
    numBlocks = ( (i__2* *mc)  + blockSize - 1) / blockSize;
    __lsei_dot___<<<numBlocks, blockSize>>>(i__2 * *mc, *mc, g, *lg, x, 1, h__, g_dim1);
    cudaDeviceSynchronize();

    // for (i__ = 1; i__ <= i__2; ++i__)
    // {
    //     /* L45: */
    //     h__[i__] -= ddot_sl__(mc, &g[i__ + g_dim1], *lg, &x[1], 1);
    // }
    lsi_(&w[ie], &w[if__], &w[ig], &h__[1], me, me, mg, mg, &l, &x[mc1], xnrm,
         &w[mc1], &jw[1], mode);
    if (*mc == 0)
    {
        goto L75;
    }
    t = dnrm2___(mc, &x[1], 1);
    *xnrm = sqrt(*xnrm * *xnrm + t * t);
    if (*mode != 1)
    {
        goto L75;
    }
/*  SOLUTION OF ORIGINAL PROBLEM AND LAGRANGE MULTIPLIERS */
L50:
    i__2 = *me;
    cudaMallocManaged(&m_dot, i__2 * sizeof(double));
    numBlocks = ( (i__2)  + blockSize - 1) / blockSize;
    __zeros<<<numBlocks, blockSize>>>(i__2, m_dot, -1);
    numBlocks = ( (i__2* *n)  + blockSize - 1) / blockSize;
    __lsei_dot2___<<<numBlocks, blockSize>>>(i__2 * *n, *n, e, *le, x, 1, m_dot, e_dim1);
    numBlocks = (i__2  + blockSize - 1) / blockSize;
    __lsei_combine___<<<numBlocks, blockSize>>>(i__2, f, m_dot);
    cudaDeviceSynchronize();
    // for (i__ = 1; i__ <= i__2; ++i__)
    // {
    //     f[i__] = m_dot[i__ - 1] - f[i__];
    // }
    // for (i__ = 1; i__ <= i__2; ++i__)
    // {
    //     /* L55: */
    //     f[i__] = ddot_sl__(n, &e[i__ + e_dim1], *le, &x[1], 1) - f[i__];
    // }
    i__2 = *mc;
    numBlocks = (i__2  + blockSize - 1) / blockSize;
    __zeros<<<numBlocks, blockSize>>>(i__2, d__, 0);
    numBlocks = (i__2* (*me+*mg)  + blockSize - 1) / blockSize;
    __lsei_dot3___<<<numBlocks, blockSize>>>(i__2 * (*me + *mg), *me + *mg, *me, e, f, g, w, d__, e_dim1, g_dim1, mc1);
    cudaDeviceSynchronize();
    // for (i__ = 1; i__ <= i__2; ++i__)
    // {
    //     /* L60: */
    //     d__[i__] = ddot_sl__(me, &e[i__ * e_dim1 + 1], 1, &f[1], 1) -
    //                ddot_sl__(mg, &g[i__ * g_dim1 + 1], 1, &w[mc1], 1);
    // }
    for (i__ = *mc; i__ >= 1; --i__)
    {
        /* L65: */
        i__2 = i__ + 1;
        h12_(&c__2, &i__, &i__2, n, &c__[i__ + c_dim1], lc, &w[iw + i__], &x[1], &c__1, &c__1, &c__1);
    }
    for (i__ = *mc; i__ >= 1; --i__)
    {
        /* Computing MIN */
        i__2 = i__ + 1;
        j = MIN2(i__2, *lc);
        i__2 = *mc - i__;
        w[i__] = (d__[i__] - ddot_sl__(&i__2, &c__[j + i__ * c_dim1], 1, &w[j], 1)) / c__[i__ + i__ * c_dim1];
        /* L70: */
    }
/*  END OF SUBROUTINE LSEI */
L75:
    cudaFree(m_dot);
    return;
} /* lsei_ */

void lsq_(int *m, int *meq, int *n, int *nl, 
	int *la, double *l, double *g, double *a, double *
	b, const double *xl, const double *xu, double *x, double *y, 
	double *w, int *jw, int *mode)
{
    /* Initialized data */
    int numBlocks;
    int blockSize = 256;
    const double one = 1.;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    /* Local variables */
    int i__, i1, i2, i3, i4, m1, n1, n2, n3, ic, id, ie, if__, ig, ih, il,
        im, ip, iu, iw;
    double diag;
    int mineq;
    double xnorm;

    /*   MINIMIZE with respect to X */
    /*             ||E*X - F|| */
    /*                                      1/2  T */
    /*   WITH UPPER TRIANGULAR MATRIX E = +D   *L , */
    /*                                      -1/2  -1 */
    /*                     AND VECTOR F = -D    *L  *G, */
    /*  WHERE THE UNIT LOWER TRIDIANGULAR MATRIX L IS STORED COLUMNWISE */
    /*  DENSE IN THE N*(N+1)/2 ARRAY L WITH VECTOR D STORED IN ITS */
    /* 'DIAGONAL' THUS SUBSTITUTING THE ONE-ELEMENTS OF L */
    /*   SUBJECT TO */
    /*             A(J)*X - B(J) = 0 ,         J=1,...,MEQ, */
    /*             A(J)*X - B(J) >=0,          J=MEQ+1,...,M, */
    /*             XL(I) <= X(I) <= XU(I),     I=1,...,N, */
    /*     ON ENTRY, THE USER HAS TO PROVIDE THE ARRAYS L, G, A, B, XL, XU. */
    /*     WITH DIMENSIONS: L(N*(N+1)/2), G(N), A(LA,N), B(M), XL(N), XU(N) */
    /*     THE WORKING ARRAY W MUST HAVE AT LEAST THE FOLLOWING DIMENSION: */
    /*     DIM(W) =        (3*N+M)*(N+1)                        for LSQ */
    /*                    +(N-MEQ+1)*(MINEQ+2) + 2*MINEQ        for LSI */
    /*                    +(N+MINEQ)*(N-MEQ) + 2*MEQ + N        for LSEI */
    /*                      with MINEQ = M - MEQ + 2*N */
    /*     ON RETURN, NO ARRAY WILL BE CHANGED BY THE SUBROUTINE. */
    /*     X     STORES THE N-DIMENSIONAL SOLUTION VECTOR */
    /*     Y     STORES THE VECTOR OF LAGRANGE MULTIPLIERS OF DIMENSION */
    /*           M+N+N (CONSTRAINTS+LOWER+UPPER BOUNDS) */
    /*     MODE  IS A SUCCESS-FAILURE FLAG WITH THE FOLLOWING MEANINGS: */
    /*          MODE=1: SUCCESSFUL COMPUTATION */
    /*               2: ERROR RETURN BECAUSE OF WRONG DIMENSIONS (N<1) */
    /*               3: ITERATION COUNT EXCEEDED BY NNLS */
    /*               4: INEQUALITY CONSTRAINTS INCOMPATIBLE */
    /*               5: MATRIX E IS NOT OF FULL RANK */
    /*               6: MATRIX C IS NOT OF FULL RANK */
    /*               7: RANK DEFECT IN HFTI */
    /*     coded            Dieter Kraft, april 1987 */
    /*     revised                        march 1989 */
    /* Parameter adjustments */
    --y;
    --x;
    --xu;
    --xl;
    --g;
    --l;
    --b;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    --jw;

    /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq;
    m1 = mineq + *n + *n;
    /*  determine whether to solve problem */
    /*  with inconsistent linerarization (n2=1) */
    /*  or not (n2=0) */
    n2 = n1 * *n / 2 + 1;
    if (n2 == *nl)
    {
        n2 = 0;
    }
    else
    {
        n2 = 1;
    }
    n3 = *n - n2;
    /*  RECOVER MATRIX E AND VECTOR F FROM L AND G */
    i2 = 1;
    i3 = 1;
    i4 = 1;
    ie = 1;
    if__ = *n * *n + 1;
    i__1 = n3;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        i1 = n1 - i__;
        diag = sqrt(l[i2]);
        w[i3] = 0.0;
        dcopy___(&i1, &w[i3], 0, &w[i3], 1);
        i__2 = i1 - n2;
        dcopy___(&i__2, &l[i2], 1, &w[i3], *n);
        i__2 = i1 - n2;
        dscal_sl__(&i__2, &diag, &w[i3], *n);
        w[i3] = diag;
        i__2 = i__ - 1;
        w[if__ - 1 + i__] = (g[i__] - ddot_sl__(&i__2, &w[i4], 1, &w[if__], 1)) / diag;
        i2 = i2 + i1 - n2;
        i3 += n1;
        i4 += *n;
        /* L10: */
    }
    if (n2 == 1)
    {
        w[i3] = l[*nl];
        w[i4] = 0.0;
        dcopy___(&n3, &w[i4], 0, &w[i4], 1);
        w[if__ - 1 + *n] = 0.0;
    }
    d__1 = -one;
    dscal_sl__(n, &d__1, &w[if__], 1);
    ic = if__ + *n;
    id = ic + *meq * *n;
    if (*meq > 0)
    {
        /*  RECOVER MATRIX C FROM UPPER PART OF A */
        i__1 = *meq;
        for (i__ = 1; i__ <= i__1; ++i__)
        {
            dcopy___(n, &a[i__ + a_dim1], *la, &w[ic - 1 + i__], *meq);
            /* L20: */
        }
        /*  RECOVER VECTOR D FROM UPPER PART OF B */
        dcopy___(meq, &b[1], 1, &w[id], 1);
        d__1 = -one;
        dscal_sl__(meq, &d__1, &w[id], 1);
    }
    ig = id + *meq;
    if (mineq > 0)
    {
        /*  RECOVER MATRIX G FROM LOWER PART OF A */
        i__1 = mineq;
        numBlocks = (i__1 * *n  + blockSize - 1) / blockSize;
        __2d_copy__<<<numBlocks, blockSize>>>(i__1 * *n, *n, a, *la, w, m1, ig, *meq, a_dim1);
        cudaDeviceSynchronize();
        // for (i__ = 1; i__ <= i__1; ++i__)
        // {
        //     dcopy___(n, &a[*meq + i__ + a_dim1], *la, &w[ig - 1 + i__], m1);
        //     /* L30: */
        // }
    }
    /*  AUGMENT MATRIX G BY +I AND -I */
    ip = ig + mineq;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        w[ip - 1 + i__] = 0.0;
        dcopy___(n, &w[ip - 1 + i__], 0, &w[ip - 1 + i__], m1);
        /* L40: */
    }
    i__1 = m1 + 1;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
        if (!nlopt_isinf(xl[i__]))
            w[(ip - i__1) + i__ * i__1] = +1.0;
    /* Old code: w[ip] = one; dcopy___(n, &w[ip], 0, &w[ip], i__1); */
    im = ip + *n;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        w[im - 1 + i__] = 0.0;
        dcopy___(n, &w[im - 1 + i__], 0, &w[im - 1 + i__], m1);
        /* L50: */
    }
    i__1 = m1 + 1;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
        if (!nlopt_isinf(xu[i__]))
            w[(im - i__1) + i__ * i__1] = -1.0;
    /* Old code: w[im] = -one;  dcopy___(n, &w[im], 0, &w[im], i__1); */
    ih = ig + m1 * *n;
    if (mineq > 0)
    {
        /*  RECOVER H FROM LOWER PART OF B */
        dcopy___(&mineq, &b[*meq + 1], 1, &w[ih], 1);
        d__1 = -one;
        dscal_sl__(&mineq, &d__1, &w[ih], 1);
    }
    /*  AUGMENT VECTOR H BY XL AND XU */
    il = ih + mineq;
    iu = il + *n;
    /* SGJ, 2010: skip constraints for infinite bounds */
    for (i__ = 1; i__ <= *n; ++i__)
    {
        w[(il - 1) + i__] = nlopt_isinf(xl[i__]) ? 0 : xl[i__];
        w[(iu - 1) + i__] = nlopt_isinf(xu[i__]) ? 0 : -xu[i__];
    }
    /* Old code: dcopy___(n, &xl[1], 1, &w[il], 1);
                 dcopy___(n, &xu[1], 1, &w[iu], 1);
		 d__1 = -one; dscal_sl__(n, &d__1, &w[iu], 1); */
    iw = iu + *n;
    i__1 = MAX2(1, *meq);
    lsei_(&w[ic], &w[id], &w[ie], &w[if__], &w[ig], &w[ih], &i__1, meq, n, n,
          &m1, &m1, n, &x[1], &xnorm, &w[iw], &jw[1], mode);
    if (*mode == 1)
    {
        /*   restore Lagrange multipliers */
        dcopy___(m, &w[iw], 1, &y[1], 1);
        dcopy___(&n3, &w[iw + *m], 1, &y[*m + 1], 1);
        dcopy___(&n3, &w[iw + *m + *n], 1, &y[*m + n3 + 1], 1);

        /* SGJ, 2010: make sure bound constraints are satisfied, since
	   roundoff error sometimes causes slight violations and
	   NLopt guarantees that bounds are strictly obeyed */
        i__1 = *n;

        numBlocks = (i__1  + blockSize - 1) / blockSize;
        __boundary_fix__<<<numBlocks, blockSize>>>(i__1, x, xl, xu);
        cudaDeviceSynchronize();
        // for (i__ = 1; i__ <= i__1; ++i__)
        // {
        //     if (x[i__] < xl[i__])
        //         x[i__] = xl[i__];
        //     else if (x[i__] > xu[i__])
        //         x[i__] = xu[i__];
        // }
    }
    /*   END OF SUBROUTINE LSQ */
} /* lsq_ */

void ldl_(int *n, double *a, double *z__, 
	double *sigma, double *w)
{
    /* Initialized data */

    const double one = 1.;
    const double four = 4.;
    const double epmach = 2.22e-16;

    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    int i__, j;
    double t, u, v;
    int ij;
    double tp, beta, gamma_, alpha, delta;

    /*   LDL     LDL' - RANK-ONE - UPDATE */
    /*   PURPOSE: */
    /*           UPDATES THE LDL' FACTORS OF MATRIX A BY RANK-ONE MATRIX */
    /*           SIGMA*Z*Z' */
    /*   INPUT ARGUMENTS: (* MEANS PARAMETERS ARE CHANGED DURING EXECUTION) */
    /*     N     : ORDER OF THE COEFFICIENT MATRIX A */
    /*   * A     : POSITIVE DEFINITE MATRIX OF DIMENSION N; */
    /*             ONLY THE LOWER TRIANGLE IS USED AND IS STORED COLUMN BY */
    /*             COLUMN AS ONE DIMENSIONAL ARRAY OF DIMENSION N*(N+1)/2. */
    /*   * Z     : VECTOR OF DIMENSION N OF UPDATING ELEMENTS */
    /*     SIGMA : SCALAR FACTOR BY WHICH THE MODIFYING DYADE Z*Z' IS */
    /*             MULTIPLIED */
    /*   OUTPUT ARGUMENTS: */
    /*     A     : UPDATED LDL' FACTORS */
    /*   WORKING ARRAY: */
    /*     W     : VECTOR OP DIMENSION N (USED ONLY IF SIGMA .LT. ZERO) */
    /*   METHOD: */
    /*     THAT OF FLETCHER AND POWELL AS DESCRIBED IN : */
    /*     FLETCHER,R.,(1974) ON THE MODIFICATION OF LDL' FACTORIZATION. */
    /*     POWELL,M.J.D.      MATH.COMPUTATION 28, 1067-1078. */
    /*   IMPLEMENTED BY: */
    /*     KRAFT,D., DFVLR - INSTITUT FUER DYNAMIK DER FLUGSYSTEME */
    /*               D-8031  OBERPFAFFENHOFEN */
    /*   STATUS: 15. JANUARY 1980 */
    /*   SUBROUTINES REQUIRED: NONE */
    /* Parameter adjustments */
    --w;
    --z__;
    --a;

    /* Function Body */
    if (*sigma == 0.0)
    {
        goto L280;
    }
    ij = 1;
    t = one / *sigma;
    if (*sigma > 0.0)
    {
        goto L220;
    }
    /* PREPARE NEGATIVE UPDATE */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        /* L150: */
        w[i__] = z__[i__];
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        v = w[i__];
        t += v * v / a[ij];
        i__2 = *n;
        for (j = i__ + 1; j <= i__2; ++j)
        {
            ++ij;
            /* L160: */
            w[j] -= v * a[ij];
        }
        /* L170: */
        ++ij;
    }
    if (t >= 0.0)
    {
        t = epmach / *sigma;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        j = *n + 1 - i__;
        ij -= i__;
        u = w[j];
        w[j] = t;
        /* L210: */
        t -= u * u / a[ij];
    }
L220:
    /* HERE UPDATING BEGINS */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__)
    {
        v = z__[i__];
        delta = v / a[ij];
        if (*sigma < 0.0)
        {
            tp = w[i__];
        }
        else /* if (*sigma > 0.0), since *sigma != 0 from above */
        {
            tp = t + delta * v;
        }
        alpha = tp / t;
        a[ij] = alpha * a[ij];
        if (i__ == *n)
        {
            goto L280;
        }
        beta = delta / tp;
        if (alpha > four)
        {
            goto L240;
        }
        i__2 = *n;
        for (j = i__ + 1; j <= i__2; ++j)
        {
            ++ij;
            z__[j] -= v * a[ij];
            /* L230: */
            a[ij] += beta * z__[j];
        }
        goto L260;
    L240:
        gamma_ = t / tp;
        i__2 = *n;
        for (j = i__ + 1; j <= i__2; ++j)
        {
            ++ij;
            u = a[ij];
            a[ij] = gamma_ * u + beta * z__[j];
            /* L250: */
            z__[j] -= v * u;
        }
    L260:
        ++ij;
        /* L270: */
        t = tp;
    }
L280:
    return;
    /* END OF LDL */
} /* ldl_ */

#define SS(var) state->var = var
#define SAVE_STATE \
     SS(t); SS(f0); SS(h1); SS(h2); SS(h3); SS(h4);	\
     SS(n1); SS(n2); SS(n3); \
     SS(t0); SS(gs); \
     SS(tol); \
     SS(line); \
     SS(alpha); \
     SS(iexact); \
     SS(incons); SS(ireset); SS(itermx)

#define RS(var) var = state->var
#define RESTORE_STATE \
     RS(t); RS(f0); RS(h1); RS(h2); RS(h3); RS(h4);	\
     RS(n1); RS(n2); RS(n3); \
     RS(t0); RS(gs); \
     RS(tol); \
     RS(line); \
     RS(alpha); \
     RS(iexact); \
     RS(incons); RS(ireset); RS(itermx)

void slsqpb_(int *m, int *meq, int *la, int *
		    n, double *x, const double *xl, const double *xu, double *f, 
		    double *c__, double *g, double *a, double *acc, 
		    int *iter, int *mode, double *r__, double *l, 
		    double *x0, double *mu, double *s, double *u, 
		    double *v, double *w, int *iw, 
		    slsqpb_state *state)
{
    /* Initialized data */

    const double one = 1.;
    const double alfmin = .1;
    const double hun = 100.;
    const double ten = 10.;
    const double two = 2.;

    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;
    double d__1, d__2;

    /* Local variables */
    int i__, j, k;

    /* saved state from one call to the next;
       SGJ 2010: save/restore via state parameter, to make re-entrant. */
    double t, f0, h1, h2, h3, h4;
    int n1, n2, n3;
    double t0, gs;
    double tol;
    int line;
    double alpha;
    int iexact;
    int incons, ireset, itermx;
    RESTORE_STATE;

/*   NONLINEAR PROGRAMMING BY SOLVING SEQUENTIALLY QUADRATIC PROGRAMS */
/*        -  L1 - LINE SEARCH,  POSITIVE DEFINITE  BFGS UPDATE  - */
/*                      BODY SUBROUTINE FOR SLSQP */
/*     dim(W) =         N1*(N1+1) + MEQ*(N1+1) + MINEQ*(N1+1)  for LSQ */
/*                     +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ */
/*                     +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1       for LSEI */
/*                      with MINEQ = M - MEQ + 2*N1  &  N1 = N+1 */
    /* Parameter adjustments */
    --mu;
    --c__;
    --v;
    --u;
    --s;
    --x0;
    --l;
    --r__;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --g;
    --xu;
    --xl;
    --x;
    --w;
    --iw;

    /* Function Body */
    if (*mode == -1) {
	goto L260;
    } else if (*mode == 0) {
	goto L100;
    } else {
	goto L220;
    }
L100:
    itermx = *iter;
    if (*acc >= 0.0) {
	iexact = 0;
    } else {
	iexact = 1;
    }
    *acc = fabs(*acc);
    tol = ten * *acc;
    *iter = 0;
    ireset = 0;
    n1 = *n + 1;
    n2 = n1 * *n / 2;
    n3 = n2 + 1;
    s[1] = 0.0;
    mu[1] = 0.0;
    dcopy___(n, &s[1], 0, &s[1], 1);
    dcopy___(m, &mu[1], 0, &mu[1], 1);
/*   RESET BFGS MATRIX */
L110:
    ++ireset;
    if (ireset > 5) {
	goto L255;
    }
    l[1] = 0.0;
    dcopy___(&n2, &l[1], 0, &l[1], 1);
    j = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	l[j] = one;
	j = j + n1 - i__;
/* L120: */
    }
/*   MAIN ITERATION : SEARCH DIRECTION, STEPLENGTH, LDL'-UPDATE */
L130:
    ++(*iter);
    *mode = 9;
    if (*iter > itermx && itermx > 0) { /* SGJ 2010: ignore if itermx <= 0 */
	goto L330;
    }
/*   SEARCH DIRECTION AS SOLUTION OF QP - SUBPROBLEM */
    dcopy___(n, &xl[1], 1, &u[1], 1);
    dcopy___(n, &xu[1], 1, &v[1], 1);
    d__1 = -one;
    daxpy_sl__(n, &d__1, &x[1], 1, &u[1], 1);
    d__1 = -one;
    daxpy_sl__(n, &d__1, &x[1], 1, &v[1], 1);
    h4 = one;
    lsq_(m, meq, n, &n3, la, &l[1], &g[1], &a[a_offset], &c__[1], &u[1], &v[1]
	    , &s[1], &r__[1], &w[1], &iw[1], mode);

/*   AUGMENTED PROBLEM FOR INCONSISTENT LINEARIZATION */
    if (*mode == 6) {
	if (*n == *meq) {
	    *mode = 4;
	}
    }
    if (*mode == 4) {
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    if (j <= *meq) {
		a[j + n1 * a_dim1] = -c__[j];
	    } else {
/* Computing MAX */
		d__1 = -c__[j];
		a[j + n1 * a_dim1] = MAX2(d__1,0.0);
	    }
/* L140: */
	}
	s[1] = 0.0;
	dcopy___(n, &s[1], 0, &s[1], 1);
	h3 = 0.0;
	g[n1] = 0.0;
	l[n3] = hun;
	s[n1] = one;
	u[n1] = 0.0;
	v[n1] = one;
	incons = 0;
L150:
	lsq_(m, meq, &n1, &n3, la, &l[1], &g[1], &a[a_offset], &c__[1], &u[1],
		 &v[1], &s[1], &r__[1], &w[1], &iw[1], mode);
	h4 = one - s[n1];
	if (*mode == 4) {
	    l[n3] = ten * l[n3];
	    ++incons;
	    if (incons > 5) {
		goto L330;
	    }
	    goto L150;
	} else if (*mode != 1) {
	    goto L330;
	}
    } else if (*mode != 1) {
	goto L330;
    }
/*   UPDATE MULTIPLIERS FOR L1-TEST */
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1);
/* L160: */
    }
    f0 = *f;
    dcopy___(n, &x[1], 1, &x0[1], 1);
    gs = ddot_sl__(n, &g[1], 1, &s[1], 1);
    h1 = fabs(gs);
    h2 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h3 = c__[j];
	} else {
	    h3 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h2 += MAX2(d__1,h3);
	h3 = (d__1 = r__[j], fabs(d__1));
/* Computing MAX */
	d__1 = h3, d__2 = (mu[j] + h3) / two;
	mu[j] = MAX2(d__1,d__2);
	h1 += h3 * (d__1 = c__[j], fabs(d__1));
/* L170: */
    }
/*   CHECK CONVERGENCE */
    *mode = 0;
    if (h1 < *acc && h2 < *acc) {
	goto L330;
    }
    h1 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h3 = c__[j];
	} else {
	    h3 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h1 += mu[j] * MAX2(d__1,h3);
/* L180: */
    }
    t0 = *f + h1;
    h3 = gs - h1 * h4;
    *mode = 8;
    if (h3 >= 0.0) {
	goto L110;
    }
/*   LINE SEARCH WITH AN L1-TESTFUNCTION */
    line = 0;
    alpha = one;
    if (iexact == 1) {
	goto L210;
    }
/*   INEXACT LINESEARCH */
L190:
    ++line;
    h3 = alpha * h3;
    dscal_sl__(n, &alpha, &s[1], 1);
    dcopy___(n, &x0[1], 1, &x[1], 1);
    daxpy_sl__(n, &one, &s[1], 1, &x[1], 1);
    
    /* SGJ 2010: ensure roundoff doesn't push us past bound constraints */
    i__1 = *n; for (i__ = 1; i__ <= i__1; ++i__) {
	 if (x[i__] < xl[i__]) x[i__] = xl[i__];
	 else if (x[i__] > xu[i__]) x[i__] = xu[i__];
    }

    /* SGJ 2010: optimizing for the common case where the inexact line
       search succeeds in one step, use special mode = -2 here to
       eliminate a a subsequent unnecessary mode = -1 call, at the 
       expense of extra gradient evaluations when more than one inexact
       line-search step is required */
    *mode = line == 1 ? -2 : 1;
    goto L330;
L200:
    if (nlopt_isfinite(h1)) {
	    if (h1 <= h3 / ten || line > 10) {
		    goto L240;
	    }
	    /* Computing MAX */
	    d__1 = h3 / (two * (h3 - h1));
	    alpha = MAX2(d__1,alfmin);
    } else {
	    alpha = MAX2(alpha*.5,alfmin);
    }
    goto L190;
/*   EXACT LINESEARCH */
L210:
#if 0
    /* SGJ: see comments by linmin_ above: if we want to do an exact linesearch
       (which usually we probably don't), we should call NLopt recursively */
    if (line != 3) {
	alpha = linmin_(&line, &alfmin, &one, &t, &tol);
	dcopy___(n, &x0[1], 1, &x[1], 1);
	daxpy_sl__(n, &alpha, &s[1], 1, &x[1], 1);
	*mode = 1;
	goto L330;
    }
#else
    *mode = 9 /* will yield nlopt_failure */; return;
#endif
    dscal_sl__(n, &alpha, &s[1], 1);
    goto L240;
/*   CALL FUNCTIONS AT CURRENT X */
L220:
    t = *f;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h1 = c__[j];
	} else {
	    h1 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	t += mu[j] * MAX2(d__1,h1);
/* L230: */
    }
    h1 = t - t0;
    switch (iexact + 1) {
	case 1:  goto L200;
	case 2:  goto L210;
    }
/*   CHECK CONVERGENCE */
L240:
    h3 = 0.0;
    i__1 = *m;
    for (j = 1; j <= i__1; ++j) {
	if (j <= *meq) {
	    h1 = c__[j];
	} else {
	    h1 = 0.0;
	}
/* Computing MAX */
	d__1 = -c__[j];
	h3 += MAX2(d__1,h1);
/* L250: */
    }
    if (((d__1 = *f - f0, fabs(d__1)) < *acc || dnrm2___(n, &s[1], 1) < *
	    acc) && h3 < *acc) {
	*mode = 0;
    } else {
	*mode = -1;
    }
    goto L330;
/*   CHECK relaxed CONVERGENCE in case of positive directional derivative */
L255:
    if (((d__1 = *f - f0, fabs(d__1)) < tol || dnrm2___(n, &s[1], 1) < tol)
	     && h3 < tol) {
	*mode = 0;
    } else {
	*mode = 8;
    }
    goto L330;
/*   CALL JACOBIAN AT CURRENT X */
/*   UPDATE CHOLESKY-FACTORS OF HESSIAN MATRIX BY MODIFIED BFGS FORMULA */
L260:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	u[i__] = g[i__] - ddot_sl__(m, &a[i__ * a_dim1 + 1], 1, &r__[1], 1) - v[i__];
/* L270: */
    }
/*   L'*S */
    k = 0;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	h1 = 0.0;
	++k;
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    ++k;
	    h1 += l[k] * s[j];
/* L280: */
	}
	v[i__] = s[i__] + h1;
/* L290: */
    }
/*   D*L'*S */
    k = 1;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	v[i__] = l[k] * v[i__];
	k = k + n1 - i__;
/* L300: */
    }
/*   L*D*L'*S */
    for (i__ = *n; i__ >= 1; --i__) {
	h1 = 0.0;
	k = i__;
	i__1 = i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    h1 += l[k] * v[j];
	    k = k + *n - j;
/* L310: */
	}
	v[i__] += h1;
/* L320: */
    }
    h1 = ddot_sl__(n, &s[1], 1, &u[1], 1);
    h2 = ddot_sl__(n, &s[1], 1, &v[1], 1);
    h3 = h2 * .2;
    if (h1 < h3) {
	h4 = (h2 - h3) / (h2 - h1);
	h1 = h3;
	dscal_sl__(n, &h4, &u[1], 1);
	d__1 = one - h4;
	daxpy_sl__(n, &d__1, &v[1], 1, &u[1], 1);
    }
    d__1 = one / h1;
    ldl_(n, &l[1], &u[1], &d__1, &v[1]);
    d__1 = -one / h2;
    ldl_(n, &l[1], &v[1], &d__1, &u[1]);
/*   END OF MAIN ITERATION */
    goto L130;
/*   END OF SLSQPB */
L330:
    SAVE_STATE;
} /* slsqpb_ */

/* *********************************************************************** */
/*                              optimizer                               * */
/* *********************************************************************** */
int slsqp(int *m, int *meq, int *la, int *n,
		  double *x, const double *xl, const double *xu, double *f, 
		  const double *c__, const double *g, const double *a, double *acc, 
		  int *iter, int *mode, double *w, int *l_w__, int *
		  jw, int *l_jw__, slsqpb_state *state)
{
    //printf("x: %f, %f, %f\n", x[0], x[1], x[2]);
    /* System generated locals */
    int a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    int n1, il, im, ir, is, iu, iv, iw, ix, mineq;

/*   SLSQP       S EQUENTIAL  L EAST  SQ UARES  P ROGRAMMING */
/*            TO SOLVE GENERAL NONLINEAR OPTIMIZATION PROBLEMS */
/* *********************************************************************** */
/* *                                                                     * */
/* *                                                                     * */
/* *            A NONLINEAR PROGRAMMING METHOD WITH                      * */
/* *            QUADRATIC  PROGRAMMING  SUBPROBLEMS                      * */
/* *                                                                     * */
/* *                                                                     * */
/* *  THIS SUBROUTINE SOLVES THE GENERAL NONLINEAR PROGRAMMING PROBLEM   * */
/* *                                                                     * */
/* *            MINIMIZE    F(X)                                         * */
/* *                                                                     * */
/* *            SUBJECT TO  C (X) .EQ. 0  ,  J = 1,...,MEQ               * */
/* *                         J                                           * */
/* *                                                                     * */
/* *                        C (X) .GE. 0  ,  J = MEQ+1,...,M             * */
/* *                         J                                           * */
/* *                                                                     * */
/* *                        XL .LE. X .LE. XU , I = 1,...,N.             * */
/* *                          I      I       I                           * */
/* *                                                                     * */
/* *  THE ALGORITHM IMPLEMENTS THE METHOD OF HAN AND POWELL              * */
/* *  WITH BFGS-UPDATE OF THE B-MATRIX AND L1-TEST FUNCTION              * */
/* *  WITHIN THE STEPLENGTH ALGORITHM.                                   * */
/* *                                                                     * */
/* *    PARAMETER DESCRIPTION:                                           * */
/* *    ( * MEANS THIS PARAMETER WILL BE CHANGED DURING CALCULATION )    * */
/* *                                                                     * */
/* *    M              IS THE TOTAL NUMBER OF CONSTRAINTS, M .GE. 0      * */
/* *    MEQ            IS THE NUMBER OF EQUALITY CONSTRAINTS, MEQ .GE. 0 * */
/* *    LA             SEE A, LA .GE. MAX(M,1)                           * */
/* *    N              IS THE NUMBER OF VARIBLES, N .GE. 1               * */
/* *  * X()            X() STORES THE CURRENT ITERATE OF THE N VECTOR X  * */
/* *                   ON ENTRY X() MUST BE INITIALIZED. ON EXIT X()     * */
/* *                   STORES THE SOLUTION VECTOR X IF MODE = 0.         * */
/* *    XL()           XL() STORES AN N VECTOR OF LOWER BOUNDS XL TO X.  * */
/* *    XU()           XU() STORES AN N VECTOR OF UPPER BOUNDS XU TO X.  * */
/* *    F              IS THE VALUE OF THE OBJECTIVE FUNCTION.           * */
/* *    C()            C() STORES THE M VECTOR C OF CONSTRAINTS,         * */
/* *                   EQUALITY CONSTRAINTS (IF ANY) FIRST.              * */
/* *                   DIMENSION OF C MUST BE GREATER OR EQUAL LA,       * */
/* *                   which must be GREATER OR EQUAL MAX(1,M).          * */
/* *    G()            G() STORES THE N VECTOR G OF PARTIALS OF THE      * */
/* *                   OBJECTIVE FUNCTION; DIMENSION OF G MUST BE        * */
/* *                   GREATER OR EQUAL N+1.                             * */
/* *    A(),LA,M,N     THE LA BY N + 1 ARRAY A() STORES                  * */
/* *                   THE M BY N MATRIX A OF CONSTRAINT NORMALS.        * */
/* *                   A() HAS FIRST DIMENSIONING PARAMETER LA,          * */
/* *                   WHICH MUST BE GREATER OR EQUAL MAX(1,M).          * */
/* *    F,C,G,A        MUST ALL BE SET BY THE USER BEFORE EACH CALL.     * */
/* *  * ACC            ABS(ACC) CONTROLS THE FINAL ACCURACY.             * */
/* *                   IF ACC .LT. ZERO AN EXACT LINESEARCH IS PERFORMED,* */
/* *                   OTHERWISE AN ARMIJO-TYPE LINESEARCH IS USED.      * */
/* *  * ITER           PRESCRIBES THE MAXIMUM NUMBER OF ITERATIONS.      * */
/* *                   ON EXIT ITER INDICATES THE NUMBER OF ITERATIONS.  * */
/* *  * MODE           MODE CONTROLS CALCULATION:                        * */
/* *                   REVERSE COMMUNICATION IS USED IN THE SENSE THAT   * */
/* *                   THE PROGRAM IS INITIALIZED BY MODE = 0; THEN IT IS* */
/* *                   TO BE CALLED REPEATEDLY BY THE USER UNTIL A RETURN* */
/* *                   WITH MODE .NE. IABS(1) TAKES PLACE.               * */
/* *                   IF MODE = -1 GRADIENTS HAVE TO BE CALCULATED,     * */
/* *                   WHILE WITH MODE = 1 FUNCTIONS HAVE TO BE CALCULATED */
/* *                   MODE MUST NOT BE CHANGED BETWEEN SUBSEQUENT CALLS * */
/* *                   OF SQP.                                           * */
/* *                   EVALUATION MODES:                                 * */
/* *        MODE = -2,-1: GRADIENT EVALUATION, (G&A)                     * */
/* *                0: ON ENTRY: INITIALIZATION, (F,G,C&A)               * */
/* *                   ON EXIT : REQUIRED ACCURACY FOR SOLUTION OBTAINED * */
/* *                1: FUNCTION EVALUATION, (F&C)                        * */
/* *                                                                     * */
/* *                   FAILURE MODES:                                    * */
/* *                2: NUMBER OF EQUALITY CONTRAINTS LARGER THAN N       * */
/* *                3: MORE THAN 3*N ITERATIONS IN LSQ SUBPROBLEM        * */
/* *                4: INEQUALITY CONSTRAINTS INCOMPATIBLE               * */
/* *                5: SINGULAR MATRIX E IN LSQ SUBPROBLEM               * */
/* *                6: SINGULAR MATRIX C IN LSQ SUBPROBLEM               * */
/* *                7: RANK-DEFICIENT EQUALITY CONSTRAINT SUBPROBLEM HFTI* */
/* *                8: POSITIVE DIRECTIONAL DERIVATIVE FOR LINESEARCH    * */
/* *                9: MORE THAN ITER ITERATIONS IN SQP                  * */
/* *             >=10: WORKING SPACE W OR JW TOO SMALL,                  * */
/* *                   W SHOULD BE ENLARGED TO L_W=MODE/1000             * */
/* *                   JW SHOULD BE ENLARGED TO L_JW=MODE-1000*L_W       * */
/* *  * W(), L_W       W() IS A ONE DIMENSIONAL WORKING SPACE,           * */
/* *                   THE LENGTH L_W OF WHICH SHOULD BE AT LEAST        * */
/* *                   (3*N1+M)*(N1+1)                        for LSQ    * */
/* *                  +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ         for LSI    * */
/* *                  +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1       for LSEI   * */
/* *                  + N1*N/2 + 2*M + 3*N + 3*N1 + 1         for SLSQPB * */
/* *                   with MINEQ = M - MEQ + 2*N1  &  N1 = N+1          * */
/* *        NOTICE:    FOR PROPER DIMENSIONING OF W IT IS RECOMMENDED TO * */
/* *                   COPY THE FOLLOWING STATEMENTS INTO THE HEAD OF    * */
/* *                   THE CALLING PROGRAM (AND REMOVE THE COMMENT C)    * */
/* ####################################################################### */
/*     INT LEN_W, LEN_JW, M, N, N1, MEQ, MINEQ */
/*     PARAMETER (M=... , MEQ=... , N=...  ) */
/*     PARAMETER (N1= N+1, MINEQ= M-MEQ+N1+N1) */
/*     PARAMETER (LEN_W= */
/*    $           (3*N1+M)*(N1+1) */
/*    $          +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ */
/*    $          +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1 */
/*    $          +(N+1)*N/2 + 2*M + 3*N + 3*N1 + 1, */
/*    $           LEN_JW=MINEQ) */
/*     DOUBLE PRECISION W(LEN_W) */
/*     INT          JW(LEN_JW) */
/* ####################################################################### */
/* *                   THE FIRST M+N+N*N1/2 ELEMENTS OF W MUST NOT BE    * */
/* *                   CHANGED BETWEEN SUBSEQUENT CALLS OF SLSQP.        * */
/* *                   ON RETURN W(1) ... W(M) CONTAIN THE MULTIPLIERS   * */
/* *                   ASSOCIATED WITH THE GENERAL CONSTRAINTS, WHILE    * */
/* *                   W(M+1) ... W(M+N(N+1)/2) STORE THE CHOLESKY FACTOR* */
/* *                   L*D*L(T) OF THE APPROXIMATE HESSIAN OF THE        * */
/* *                   LAGRANGIAN COLUMNWISE DENSE AS LOWER TRIANGULAR   * */
/* *                   UNIT MATRIX L WITH D IN ITS 'DIAGONAL' and        * */
/* *                   W(M+N(N+1)/2+N+2 ... W(M+N(N+1)/2+N+2+M+2N)       * */
/* *                   CONTAIN THE MULTIPLIERS ASSOCIATED WITH ALL       * */
/* *                   ALL CONSTRAINTS OF THE QUADRATIC PROGRAM FINDING  * */
/* *                   THE SEARCH DIRECTION TO THE SOLUTION X*           * */
/* *  * JW(), L_JW     JW() IS A ONE DIMENSIONAL INT WORKING SPACE   * */
/* *                   THE LENGTH L_JW OF WHICH SHOULD BE AT LEAST       * */
/* *                   MINEQ                                             * */
/* *                   with MINEQ = M - MEQ + 2*N1  &  N1 = N+1          * */
/* *                                                                     * */
/* *  THE USER HAS TO PROVIDE THE FOLLOWING SUBROUTINES:                 * */
/* *     LDL(N,A,Z,SIG,W) :   UPDATE OF THE LDL'-FACTORIZATION.          * */
/* *     LINMIN(A,B,F,TOL) :  LINESEARCH ALGORITHM IF EXACT = 1          * */
/* *     LSQ(M,MEQ,LA,N,NC,C,D,A,B,XL,XU,X,LAMBDA,W,....) :              * */
/* *                                                                     * */
/* *        SOLUTION OF THE QUADRATIC PROGRAM                            * */
/* *                QPSOL IS RECOMMENDED:                                * */
/* *     PE GILL, W MURRAY, MA SAUNDERS, MH WRIGHT:                      * */
/* *     USER'S GUIDE FOR SOL/QPSOL:                                     * */
/* *     A FORTRAN PACKAGE FOR QUADRATIC PROGRAMMING,                    * */
/* *     TECHNICAL REPORT SOL 83-7, JULY 1983                            * */
/* *     DEPARTMENT OF OPERATIONS RESEARCH, STANFORD UNIVERSITY          * */
/* *     STANFORD, CA 94305                                              * */
/* *     QPSOL IS THE MOST ROBUST AND EFFICIENT QP-SOLVER                * */
/* *     AS IT ALLOWS WARM STARTS WITH PROPER WORKING SETS               * */
/* *                                                                     * */
/* *     IF IT IS NOT AVAILABLE USE LSEI, A CONSTRAINT LINEAR LEAST      * */
/* *     SQUARES SOLVER IMPLEMENTED USING THE SOFTWARE HFTI, LDP, NNLS   * */
/* *     FROM C.L. LAWSON, R.J.HANSON: SOLVING LEAST SQUARES PROBLEMS,   * */
/* *     PRENTICE HALL, ENGLEWOOD CLIFFS, 1974.                          * */
/* *     LSEI COMES WITH THIS PACKAGE, together with all necessary SR's. * */
/* *                                                                     * */
/* *     TOGETHER WITH A COUPLE OF SUBROUTINES FROM BLAS LEVEL 1         * */
/* *                                                                     * */
/* *     SQP IS HEAD SUBROUTINE FOR BODY SUBROUTINE SQPBDY               * */
/* *     IN WHICH THE ALGORITHM HAS BEEN IMPLEMENTED.                    * */
/* *                                                                     * */
/* *  IMPLEMENTED BY: DIETER KRAFT, DFVLR OBERPFAFFENHOFEN               * */
/* *  as described in Dieter Kraft: A Software Package for               * */
/* *                                Sequential Quadratic Programming     * */
/* *                                DFVLR-FB 88-28, 1988                 * */
/* *  which should be referenced if the user publishes results of SLSQP  * */
/* *                                                                     * */
/* *  DATE:           APRIL - OCTOBER, 1981.                             * */
/* *  STATUS:         DECEMBER, 31-ST, 1984.                             * */
/* *  STATUS:         MARCH   , 21-ST, 1987, REVISED TO FORTAN 77        * */
/* *  STATUS:         MARCH   , 20-th, 1989, REVISED TO MS-FORTRAN       * */
/* *  STATUS:         APRIL   , 14-th, 1989, HESSE   in-line coded       * */
/* *  STATUS:         FEBRUARY, 28-th, 1991, FORTRAN/2 Version 1.04      * */
/* *                                         accepts Statement Functions * */
/* *  STATUS:         MARCH   ,  1-st, 1991, tested with SALFORD         * */
/* *                                         FTN77/386 COMPILER VERS 2.40* */
/* *                                         in protected mode           * */
/* *                                                                     * */
/* *********************************************************************** */
/* *                                                                     * */
/* *  Copyright 1991: Dieter Kraft, FHM                                  * */
/* *                                                                     * */
/* *********************************************************************** */
/*     dim(W) =         N1*(N1+1) + MEQ*(N1+1) + MINEQ*(N1+1)  for LSQ */
/*                    +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ          for LSI */
/*                    +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1        for LSEI */
/*                    + N1*N/2 + 2*M + 3*N +3*N1 + 1           for SLSQPB */
/*                      with MINEQ = M - MEQ + 2*N1  &  N1 = N+1 */
/*   CHECK LENGTH OF WORKING ARRAYS */
    /* Parameter adjustments */
    // use managed memory
    double *m_x, *m_xl, *m_xu, *m_g, *m_w,  *m_c, *m_a;
    int *m_jw;
    cudaMallocManaged(&m_x, *n * sizeof(double));
    cudaMallocManaged(&m_xl, *n * sizeof(double));
    cudaMallocManaged(&m_xu, *n * sizeof(double));

    cudaMallocManaged(&m_c, *m * sizeof(double));
    cudaMallocManaged(&m_a, *la *(*n + 1) * sizeof(double));
    cudaMallocManaged(&m_g, (*n + 1) * sizeof(double));

    cudaMallocManaged(&m_w, (*l_w__) * sizeof(double));
    cudaMallocManaged(&m_jw, (*l_jw__) * sizeof(int));

    int blockSize = 256;
    int numBlocks = (*n + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*n, m_x, x, 1, 1, 0);
    __copy_2<<<numBlocks, blockSize>>>(*n, m_xl, xl, 1, 1, 0);
    __copy_2<<<numBlocks, blockSize>>>(*n, m_xu, xu, 1, 1, 0);
    numBlocks = (*m + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*m, m_c, c__, 1, 1, 0);
    numBlocks = (*n + 1 + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*n + 1, m_g, g, 1, 1, 0);
    numBlocks = (*la * (*n + 1) + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*la * (*n + 1), m_a, a, 1, 1, 0);
    numBlocks = (*l_w__ + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*l_w__, m_w, w, 1, 1, 0);
    numBlocks = (*l_jw__ + blockSize - 1) / blockSize;
    __copy_2_int<<<numBlocks, blockSize>>>(*l_jw__, m_jw, jw, 1, 1);

    cudaDeviceSynchronize();

    --m_c;
    a_dim1 = *la;
    a_offset = 1 + a_dim1;
    m_a -= a_offset;
    --m_g;
    --m_xu;
    --m_xl;
    --m_x;
    --m_w;
    --m_jw;

    /* Function Body */
    n1 = *n + 1;
    mineq = *m - *meq + n1 + n1;
    il = (n1 * 3 + *m) * (n1 + 1) + (n1 - *meq + 1) * (mineq + 2) + (mineq << 
	    1) + (n1 + mineq) * (n1 - *meq) + (*meq << 1) + n1 * *n / 2 + (*m 
	    << 1) + *n * 3 + (n1 << 2) + 1;
/* Computing MAX */
    i__1 = mineq, i__2 = n1 - *meq;
    im = MAX2(i__1,i__2);
    if (*l_w__ < il || *l_jw__ < im) {
	*mode = MAX2(10,il) * 1000;
	*mode += MAX2(10,im);
	return 0;
    }
/*   PREPARE DATA FOR CALLING SQPBDY  -  INITIAL ADDRESSES IN W */
    im = 1;
    il = im + MAX2(1,*m);
    il = im + *la;
    ix = il + n1 * *n / 2 + 1;
    ir = ix + *n;
    is = ir + *n + *n + MAX2(1,*m);
    is = ir + *n + *n + *la;
    iu = is + n1;
    iv = iu + n1;
    iw = iv + n1;

    slsqpb_(m, meq, la, n, &m_x[1], &m_xl[1], &m_xu[1], f, &m_c[1], &m_g[1], &m_a[
	    a_offset], acc, iter, mode, &m_w[ir], &m_w[il], &m_w[ix], &m_w[im], &m_w[is]
	    , &m_w[iu], &m_w[iv], &m_w[iw], &m_jw[1], state);


    numBlocks = (*n + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*n, x, m_x, 1, 1, 1);

    numBlocks = (*l_w__ + blockSize - 1) / blockSize;
    __copy_2<<<numBlocks, blockSize>>>(*l_w__, w, m_w, 1, 1, 1);
    cudaDeviceSynchronize();


    state->x0 = &w[ix+1];

    cudaFree(m_x);
    cudaFree(m_xl);
    cudaFree(m_xu);
    cudaFree(m_c);
    cudaFree(m_a);
    cudaFree(m_g);
    cudaFree(m_jw);
    cudaFree(m_w);
    return 0;
} /* slsqp_ */
