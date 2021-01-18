cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(using-cuda-backend LANGUAGES CXX)

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS 
      "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice")

add_executable(exercise1 source.cpp)
target_compile_features(exercise1 PUBLIC cxx_std_17)
target_compile_options(exercise1 PUBLIC ${SYCL_FLAGS})
target_link_libraries(exercise1 PUBLIC ${SYCL_FLAGS})
target_include_directories(exercise1 PUBLIC ${SYCL_INCLUDE_DIR})

if (BUILD_SOLUTION)
  add_executable(solution1 solution.cpp)
  target_compile_features(solution1 PUBLIC cxx_std_17)
  target_compile_options(solution1 PUBLIC ${SYCL_FLAGS})
  target_link_libraries(solution1 PUBLIC ${SYCL_FLAGS})
  target_include_directories(solution1 PUBLIC ${SYCL_INCLUDE_DIR})
endif()
