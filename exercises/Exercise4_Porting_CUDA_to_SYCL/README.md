# Exercise 4: Porting CUDA code to SYCL

In this exercise you are provided with CUDA code, `cuda_matmul.cu`, which you must port to SYCL.

The CUDA application implements a simple matrix multiplication kernel.

## Build instructions

```bash
$ mkdir build && cd build
$ cmake .. -DSYCL_ROOT=${SYCL_ROOT_DIR} -DCMAKE_CXX_COMPILER=${SYCL_ROOT_DIR}/bin/clang++
$ make -j8
```
