#include <CL/sycl.hpp>

/**
 * A skeleton of a device selector is provided here.
 *
 * Implement this selector so that it only selects CUDA enabled devices
 */
class CUDADeviceSelector : public sycl::device_selector {
public:
  int operator()(const sycl::device &device) const override {

    /**
     * Query the device's platform properties to determine which backend it uses.
     */

  }
};

int main(int argc, char *argv[]) {

  /**
   * This queue uses the default device selector.
   * Modify this so that the queue uses your CUDA device selector
   */
  sycl::queue myQueue{cl::sycl::default_selector()};

  /**
   * Before proceeding, query the name of the device that the queue is using.
   *
   * Retrieve the queue's device using the queue's `get_device` method
   */
  std::string deviceName;

  std::cout << "Queue is running on " << deviceName << "\n";

  /**
   * Buffers provided will be used in the vector addition kernel
   */
  constexpr int N = 1000;
  std::vector<float> h_a(N);
  std::vector<float> h_b(N);
  std::vector<float> h_c(N);
  std::fill(h_a.begin(), h_a.end(), 40.0);
  std::fill(h_b.begin(), h_b.end(), 2.0);

  const sycl::range<1> bufRange{N};
  sycl::buffer<float, 1> d_a{h_a};
  sycl::buffer<float, 1> d_b{h_b};
  sycl::buffer<float, 1> d_c{h_c};

  /**
   * An empty command group is provided below.
   * 
   * You need to implement appropriate accessors for our buffers,
   * and a simple vector addition kernel.
   *
   * You can implement the kernel as a lambda function, and use the
   * `parallel_for` interface of the handler to invoke the kernel.
   *
   * You should add each item in buffers a and b, and store the result
   * in buffer c.
   */
  auto cg = [&](sycl::handler &h) {
    /**
     * Define your accessors here
     */


    /**
     * Implement the vector addition kernel here and invoke using `parallel_for`
     */

  };

  /**
   * Once you have implemented the command group, submit it to your queue.
   */


  /**
   * Finally, validate the results are as expected.
   * Use a host accessor to access the data within the SYCL buffer holding the
   * result of our vector addition.
   */

  {
    /**
     * Recall that host accessors should be placed inside a nested scope, so
     * that they are destructed properly upon exiting the scope and release
     * their ownership of the buffer data.
     */


    /**
     * Create an appropriate host accessor here
     */


    /**
     * Now verify that the the contents of the results buffer "d_c" are as expected.
     */

  }

  return 0;
}
