#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

#include <cublas_v2.h>
#include <cuda.h>

/**
 * CUDA error checking utilities are provided here
 */
#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cublasStatus_t status, const char *msg) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cout << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != CUDA_SUCCESS) {
    std::cout << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

/**
 * The CUDA device selector is reused from the previous exercise
 */
class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    if (device.get_platform().get_backend() == sycl::backend::cuda)
      return 1;
    else 
      return -1;
  }
};

int main(int argc, char *argv[]) {

  /**
   * In this exercise we will call the cuBLAS sgemm library function from a SYCL
   * command group. The host side data vectors are initialised for you, these will
   * represent the matrices we will use in the sgemm operation.
   */

  using namespace sycl;

  constexpr size_t WIDTH = 1024;
  constexpr size_t HEIGHT = 1024;
  constexpr float ALPHA = 1.0f;
  constexpr float BETA = 0.0f;

  /**
   * A is an identity matrix
   * B is a matrix of ones
   * C will store the result of the sgemm operation
   */
  std::vector<float> h_A(WIDTH * HEIGHT);
  std::vector<float> h_B(WIDTH * HEIGHT);
  std::vector<float> h_C(WIDTH * HEIGHT);

  std::fill(std::begin(h_A), std::end(h_A), 0.0f);
  for (size_t i = 0; i < WIDTH; i++) {
    h_A[i * WIDTH + i] = 1.0f;
  }

  std::fill(std::begin(h_B), std::end(h_B), 1.0f);

  /**
   * Exercise:
   * 
   * Implement and invoke an interop_task that calls cublasSgemm.
   * 
   * Verify the results of cublasSgemm (multiplication of identity matrix with
   * a matrix of ones)
   */


  // Set the return code corresponding to the sgemm operation verification
  return EXIT_FAILURE;
}
