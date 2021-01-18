
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(using-cublas-in-sycl LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/lib/clang/12.0.0/include/" "${SYCL_ROOT}/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS 
      "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
      "-fsycl-unnamed-lambda")

add_executable(exercise2 source.cpp)
target_compile_features(exercise2 PUBLIC cxx_std_17)
target_compile_options(exercise2 PUBLIC ${SYCL_FLAGS})
target_link_libraries(exercise2 PUBLIC ${SYCL_FLAGS})
target_compile_definitions(exercise2 PUBLIC CUDA_NO_HALF)
target_include_directories(exercise2 PUBLIC ${SYCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(exercise2 PUBLIC CUDA::toolkit CUDA::cuda_driver CUDA::cublas)

if (BUILD_SOLUTION)
  add_executable(solution2 solution.cpp)
  target_compile_features(solution2 PUBLIC cxx_std_17)
  target_compile_options(solution2 PUBLIC ${SYCL_FLAGS})
  target_link_libraries(solution2 PUBLIC ${SYCL_FLAGS})
  target_compile_definitions(solution2 PUBLIC CUDA_NO_HALF)
  target_include_directories(solution2 PUBLIC ${SYCL_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(solution2 PUBLIC CUDA::toolkit CUDA::cuda_driver CUDA::cublas)
endif()
