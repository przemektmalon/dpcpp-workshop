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
    using namespace sycl::info;

    const std::string DriverVersion = device.get_info<device::driver_version>();

    if (device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    };
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

  /**
   * Some parameters for cublasSgemm
   */
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

  sycl::queue myQueue{CUDADeviceSelector()};

  cublasHandle_t cublasHandle;
  CHECK_ERROR(cublasCreate(&cublasHandle));

  {
    sycl::buffer<float, 2> b_A{h_A.data(), range<2>{WIDTH, HEIGHT}};
    sycl::buffer<float, 2> b_B{h_B.data(), range<2>{WIDTH, HEIGHT}};
    sycl::buffer<float, 2> b_C{h_C.data(), range<2>{WIDTH, HEIGHT}};

    myQueue.submit([&](sycl::handler &h) {
      auto d_A = b_A.get_access<sycl::access::mode::read>(h);
      auto d_B = b_B.get_access<sycl::access::mode::read>(h);
      auto d_C = b_C.get_access<sycl::access::mode::write>(h);

      h.interop_task([=](sycl::interop_handler ih) {
        auto cudaStreamHandle = sycl::get_native<sycl::backend::cuda>(myQueue);
        cublasSetStream(cublasHandle, cudaStreamHandle);

        auto cuA =
            reinterpret_cast<float *>(ih.get_mem<sycl::backend::cuda>(d_A));
        auto cuB =
            reinterpret_cast<float *>(ih.get_mem<sycl::backend::cuda>(d_B));
        auto cuC =
            reinterpret_cast<float *>(ih.get_mem<sycl::backend::cuda>(d_C));

        CHECK_ERROR(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, WIDTH, 
                                HEIGHT, WIDTH, &ALPHA, cuA, WIDTH, cuB, WIDTH, 
                                &BETA, cuC, WIDTH));
      });
    });
  }

  int i = 0;
  const bool allEqual =
      std::all_of(std::begin(h_C), std::end(h_C), [&i](float num) {
        ++i;
        if (num != 1) {
          std::cout << i << " Not one : " << num << std::endl;
        }
        return num == 1;
      });

  if (!allEqual) {
    std::cout << " Incorrect result " << std::endl;
  } else {
    std::cout << " Correct! " << std::endl;
  }

  CHECK_ERROR(cublasDestroy(cublasHandle));

  // Set the return code corresponding to the sgemm operation verification
  return allEqual ? EXIT_SUCCESS : EXIT_FAILURE;
}
