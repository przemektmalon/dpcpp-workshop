
CUDACXX=${SYCL_ROOT_DIR}/bin/clang++

SYCL_INCLUDE=${SYCL_ROOT_DIR}/include/sycl/

CUDAFLAGS=--cuda-gpu-arch=sm_50

CXXFLAGS=-std=c++17 ${CUDAFLAGS} -I${SYCL_INCLUDE} -g

LIBS=-L${SYCL_ROOT_DIR}/lib -lOpenCL -lsycl -L${CUDA_ROOT_DIR}/lib64 -lcudart

default: exercise

exercise: source.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@

solution: solution.cu
	${CUDACXX} ${CXXFLAGS} $< ${LIBS} -o $@


clean:
	rm -f exercise solution