#include <CL/sycl.hpp>

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
   * Make sure to implement error handling, otherwise your solution
   * might fail silently.
   *
   * Once you have a working solution with error handling, you can try to break
   * it to see what kinds of errors are thrown
   *
   */

  try {
    /* Create SYCL buffers corresponding to the host-side data vectors */
    const sycl::range<1> bufRange{N * N};
    sycl::buffer<float, 1> d_a{h_a.data(), bufRange};
    sycl::buffer<float, 1> d_b{h_b.data(), bufRange};
    sycl::buffer<float, 1> d_c{h_c.data(), bufRange};

    /* Create queue using a CUDA device selector */
    sycl::queue myQueue{CUDADeviceSelector()};

    /* Define the command group */
    auto cg = [&](sycl::handler &h) {
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;

      /* Accessors */
      auto a = d_a.get_access<read_t>(h);
      auto b = d_b.get_access<read_t>(h);
      auto c = d_c.get_access<write_t>(h);

      /* Define block size (work group local size) */
      int blockSize = 32;

      /* Define the 2D local and global ranges */
      auto localRange = sycl::range<2>(blockSize, blockSize);
      auto globalRange =
          localRange *
          sycl::range<2>(ceil(double(N) / double(localRange.get(0))),
                         ceil(double(N) / double(localRange.get(1))));

      /* Define the kernel's ND range */
      auto kernelNDRange = sycl::nd_range<2>(globalRange, localRange);

      /* Invoke the kernel using our kernel's ND range */
      h.parallel_for<class matmul>(kernelNDRange, [=](sycl::nd_item<2> ndItem) {
        /* Get column and row */
        int col = ndItem.get_global_id(0);
        int row = ndItem.get_global_id(1);

        /* Perform the multiplication */
        float sum = 0.f;
        if (row < N && col < N) {
          for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
          }
          c[row * N + col] = sum;
        }
      });
    };

    myQueue.submit(cg);

    myQueue.wait_and_throw();

  } catch (sycl::exception e) {
    std::cout << "SYCL Exception Caught: " << e.what() << std::endl;
  } catch (std::exception e) {
    std::cout << "Non-SYCL Exception Caught: " << e.what() << std::endl;
  }

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
