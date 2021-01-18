# Exercise 2: Calling cuBLAS functions inside a SYCL command group

In this exercise you will implement an `interop_task` that calls `cublasSgemm`.

## Build instructions

```bash
$ mkdir build && cd build
$ cmake .. -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++
$ make -j8
```

To enable building of the solution, pass `-DBUILD_SOLUTION=ON` to the cmake invocation.
