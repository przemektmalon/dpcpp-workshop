
# SYCL for Nvidia Exercises

These exercises introduce the CUDA backend of the DPC++ SYCL compiler and runtime.


# Prerequisites

TODO: docker image + instructions

---

## Exercise 1 - Using the CUDA backend
To get started, in the first exercise you will learn how to select a CUDA device for a SYCL queue and verify that your application running on a CUDA device. You will also implement a simple vector addition kernel and verify the results are correct running on the CUDA device.

## Exercise 2 - Calling cuBLAS sgemm from a SYCL command group
The second exercise introduces SYCL interoperability with CUDA libraries. You will need to implement a command group that calls `cublasSgemm`. In order to achieve this, you will use the `interop_task` interface, and will have to retrieve native CUDA memory handles from the SYCL runtime API to pass to `cublasSgemm`.

## Exercise 3 - Calling a native CUDA kernel from a SYCL command group
The third exercise introduces SYCL interoperability with native CUDA kernels. You will need to implement a command group that calls a provided CUDA kernel. To do this you will use the `interop_task` interface, and similarly to exercise 2, you will have to retrieve native CUDA memory handles to pass to the CUDA kernel.

## Exercise 4 - Porting CUDA to SYCL
In the fourth and final exercise, you will need to port CUDA code to SYCL. The CUDA application implements a simple matrix multiplication.

---