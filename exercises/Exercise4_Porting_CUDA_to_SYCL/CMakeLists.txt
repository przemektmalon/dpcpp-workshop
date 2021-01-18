cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(porting-cuda-to-sycl LANGUAGES CXX)

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "SYCL_ROOT not specified! Please specify -DSYCL_ROOT=<DPCPP_ROOT_PATH>")
endif()

set(SYCL_INCLUDE_DIR "${SYCL_ROOT}/include/")
set(SYCL_LIB "${SYCL_ROOT}/lib/libsycl.so")
set(SYCL_FLAGS 
      "-fsycl" 
      "-fsycl-targets=nvptx64-nvidia-cuda-sycldevice"
      "-fsycl-unnamed-lambda")

add_executable(exercise4 source.cpp)
target_compile_features(exercise4 PUBLIC cxx_std_17)
target_compile_options(exercise4 PUBLIC ${SYCL_FLAGS})
target_link_libraries(exercise4 PUBLIC ${SYCL_FLAGS})
target_include_directories(exercise4 PUBLIC ${SYCL_INCLUDE_DIR})

if (BUILD_SOLUTION)
  add_executable(solution4 example_solution.cpp)
  target_compile_features(solution4 PUBLIC cxx_std_17)
  target_compile_options(solution4 PUBLIC ${SYCL_FLAGS})
  target_link_libraries(solution4 PUBLIC ${SYCL_FLAGS})
  target_include_directories(solution4 PUBLIC ${SYCL_INCLUDE_DIR})
endif()

