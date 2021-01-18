# Exercise 1: Enabling the CUDA backend in DPC++

In this exercise you will:
- Implement a CUDA device selector, enabling the CUDA backend of DPC++
- Verify that you are using the CUDA device by querying the queue's device properties
- Implement and execute a simple kernel to verify that everything is working correctly

---

## Build instructions

```bash
$ mkdir build && cd build
$ cmake .. -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++
$ make -j8
```

To enable building of the solution, pass `-DBUILD_SOLUTION=ON` to the cmake invocation
