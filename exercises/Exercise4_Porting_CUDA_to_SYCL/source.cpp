#include <CL/sycl.hpp>

class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {
    using namespace sycl::info;

    const std::string driverVersion = device.get_info<device::driver_version>();

    if (device.is_gpu() && (driverVersion.find("CUDA") != std::string::npos)) {
      std::cout << "CUDA device found\n";
      return 1;
    }

    return -1;
  }
};

int main(int argc, char *argv[]) {

  /* Size of matrix */
  constexpr int N = 473;

  /* Create and fill matrices on host */
  std::vector<float> h_a(N * N);
  std::vector<float> h_b(N * N);
  std::vector<float> h_c(N * N);
  std::fill(h_a.begin(), h_a.end(), 3.0);
  std::fill(h_b.begin(), h_b.end(), 2.0);

  /**
   * Exercise:
   * 
   * Port the CUDA code in "cuda_matmul.cu" to SYCL
   * 
   * Note that "cuda_matmul.cu" invoked a 2D kernel,
   * in your SYCL code you may invoke `parallel_for` 
   * with a sycl::nd_range<2>.
   * 
   * sycl::nd_range consists of a global range, and a local range.
   * The global range must be divisible by the local range.
   * 
   */




  /**
   * Verification code
   */
  std::vector<float> expected(N * N, 0.f);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < N; ++k) {
        expected[i * N + j] += h_a[i * N + k] * h_b[k * N + j];
      }
    }
  }

  for (int i = 0; i < N * N; ++i) {
    if (h_c[i] != expected[i]) {
      std::cout << "Result incorrect!\n";
      std::cout << "Expected " << expected[i] << " Actual " << h_c[i] << "\n";
      return 1;
    }
  }

  std::cout << "Result correct!\n";
  return 0;
}
