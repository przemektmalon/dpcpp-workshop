// Original source reproduced unmodified here from:
// https://github.com/olcf/vector_addition_tutorials/blob/master/CUDA/vecAdd.cu

#include <algorithm>
#include <iostream>
#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/backend/cuda.hpp>

/**
 * The CUDA device selector is reused from the previous exercise
 */
class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &Device) const override {
    using namespace sycl::info;

    const std::string DriverVersion = Device.get_info<device::driver_version>();

    if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos)) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    };
    return -1;
  }
};

/**
 * A simple CUDA vector addition kernel is provided, which we will later call
 * from within a SYCL command group. Each thread takes care of one element of c
 */
__global__ void vecAdd(double *a, double *b, double *c, int n) {
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n) {
    c[id] = a[id] + b[id];
  }
}

int main(int argc, char *argv[]) {

  /**
   * Exercise:
   * 
   * Implement an interop_task that calls the "vecAdd" CUDA kernel defined above
   * 
   * The queue and buffers are already constructed and initialized.
   */

  sycl::device myCUDADevice{CUDADeviceSelector().select_device()};
  sycl::context myContext{myCUDADevice};
  sycl::queue myQueue{myContext, myCUDADevice};

  // Size of vectors
  int N = 10000;

  {
    sycl::buffer<double> bA{sycl::range<1>(N)};
    sycl::buffer<double> bB{sycl::range<1>(N)};
    sycl::buffer<double> bC{sycl::range<1>(N)};

    {
      auto hA = bA.get_access<sycl::access::mode::write>();
      auto hB = bB.get_access<sycl::access::mode::write>();

      // Initialize vectors on host
      for (int i = 0; i < N; i++) {
        hA[i] = sin(i) * sin(i);
        hB[i] = cos(i) * cos(i);
      }
    }

    myQueue.submit([&](sycl::handler &h) {
      /**
       * Exercise
       */
    });

    /**
     * Verification code provided
     */
    {
      auto hC = bC.get_access<sycl::access::mode::read>();
      // Sum up vector c and print result divided by N, this should equal 1
      // within error
      double sum = 0;
      for (int i = 0; i < N; i++) {
        sum += hC[i];
      }
      std::cout << "Final result " << sum / N << " : Expected result 1"
                << std::endl;
    }
  }

  return 0;
}
