#include "driver.hpp"

 
void test_dsscfg_cuda(std::function<void(double*, double&, unsigned long&, int len)> func, double *x, double *xl, double *xu, int *nbd, int elements, double ftol, double gtol, double eps, int maxiter, int *iterations, int *stats, int m){
  // initialize LBFGSB option
  LBFGSB_CUDA_OPTION<double> lbfgsb_options;
  lbfgsbcuda::lbfgsbdefaultoption<double>(lbfgsb_options);
  lbfgsb_options.mode = LCM_CUDA;
  lbfgsb_options.eps_f = static_cast<double>(ftol);
  lbfgsb_options.eps_g = static_cast<double>(gtol);
  lbfgsb_options.eps_x = static_cast<double>(eps);
  lbfgsb_options.max_iteration = maxiter;
  lbfgsb_options.hessian_approximate_dimension = m;

  // initialize LBFGSB state
  LBFGSB_CUDA_STATE<double> state;
  memset(&state, 0, sizeof(state));
  cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
  if (CUBLAS_STATUS_SUCCESS != stat) {
    std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
    exit(0);
  }
  // setup callback function that evaluate function value and its gradient
  state.m_funcgrad_callback = [&func, &elements](
                                  double* x, double& f, double* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<double>& summary) {
    unsigned long addr;
    double * newadd;
    func(x, f, addr, elements);
    newadd =  reinterpret_cast<double*>(addr);
    cudaMemcpy(g, newadd, elements * sizeof(double), cudaMemcpyDeviceToDevice);
    return 0;
  };

  LBFGSB_CUDA_SUMMARY<double> summary;
  memset(&summary, 0, sizeof(summary));

  lbfgsbcuda::lbfgsbminimize<double>(elements, state, lbfgsb_options, x, nbd,
                                   xl, xu, summary);
  *iterations = summary.num_iteration;
  *stats =  summary.info;
  if (summary.info==5){
    (*iterations)--;
  }
  // std::cout<<"info "<<summary.info<<std::endl;
  // std::cout<<"res_g "<<summary.residual_g<<" "<<gtol<<std::endl;
  // std::cout<<"res_f "<<summary.residual_f<<" "<<ftol<<std::endl;
  // std::cout<<"res_x "<<summary.residual_x<<" "<<eps<<std::endl;
  // std::cout<<"max_iter "<<summary.num_iteration<<" "<<maxiter<<std::endl;
  cublasDestroy(state.m_cublas_handle);
}

void test_dsscfg_cuda_float(std::function<void(float*, float&, unsigned long&, int len)> func, float *x, float *xl, float *xu, int *nbd, int elements, float ftol, float gtol, float eps, int maxiter, int *iterations, int *stats, int m){
  // initialize LBFGSB option
  LBFGSB_CUDA_OPTION<float> lbfgsb_options;
  lbfgsbcuda::lbfgsbdefaultoption<float>(lbfgsb_options);
  lbfgsb_options.mode = LCM_CUDA;
  lbfgsb_options.eps_f = static_cast<float>(ftol);
  lbfgsb_options.eps_g = static_cast<float>(gtol);
  lbfgsb_options.eps_x = static_cast<float>(eps);
  lbfgsb_options.max_iteration = maxiter;
  lbfgsb_options.hessian_approximate_dimension = m;

  // initialize LBFGSB state
  LBFGSB_CUDA_STATE<float> state;
  memset(&state, 0, sizeof(state));
  cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
  if (CUBLAS_STATUS_SUCCESS != stat) {
    std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
    exit(0);
  }
  // setup callback function that evaluate function value and its gradient
  state.m_funcgrad_callback = [&func, &elements](
                                  float* x, float& f, float* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<float>& summary) {
    unsigned long addr;
    float * newadd;
    func(x, f, addr, elements);
    newadd =  reinterpret_cast<float*>(addr);
    cudaMemcpy(g, newadd, elements * sizeof(float), cudaMemcpyDeviceToDevice);
    return 0;
  };

  LBFGSB_CUDA_SUMMARY<float> summary;
  memset(&summary, 0, sizeof(summary));

  lbfgsbcuda::lbfgsbminimize<float>(elements, state, lbfgsb_options, x, nbd,
                                   xl, xu, summary);
  *iterations = summary.num_iteration;
  *stats =  summary.info;
  if (summary.info==5){
    (*iterations)--;
  }
  // std::cout<<"info "<<summary.info<<std::endl;
  // std::cout<<"res_g "<<summary.residual_g<<" "<<gtol<<std::endl;
  // std::cout<<"res_f "<<summary.residual_f<<" "<<ftol<<std::endl;
  // std::cout<<"res_x "<<summary.residual_x<<" "<<eps<<std::endl;
  // std::cout<<"max_iter "<<summary.num_iteration<<" "<<maxiter<<std::endl;
  cublasDestroy(state.m_cublas_handle);
}