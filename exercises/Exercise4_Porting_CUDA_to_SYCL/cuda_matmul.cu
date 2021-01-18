#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>
#include <vector>

#define CHECK_ERROR(FUNC) checkCudaErrorMsg(FUNC, " " #FUNC)

void inline checkCudaErrorMsg(cudaError status, const char *msg) {
  if (status != cudaSuccess) {
    std::cout << msg << " - " << status << std::endl;
    exit(EXIT_FAILURE);
  }
}

/* A simple CUDA kernel that performs matrix multiplication */
__global__ void matmul(const float *a, const float *b, float *c, int n) {

  int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;

  float sum = 0.f;

  if (row < n && col < n) {
    for (int i = 0; i < n; ++i) {
      sum += a[row * n + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
  }
}

int main(int argc, char *argv[]) {

  /* Size of matrix */
  constexpr int N = 473;

  /* Create and fill matrices on host */
  std::vector<float> h_a(N * N);
  std::vector<float> h_b(N * N);
  std::vector<float> h_c(N * N);
  std::fill(h_a.begin(), h_a.end(), 3.f);
  std::fill(h_b.begin(), h_b.end(), 2.f);

  /* Create matrices on device and copy host data */
  float *d_a = nullptr;
  float *d_b = nullptr;
  float *d_c = nullptr;
  size_t matSizeInBytes = N * N * sizeof(float);
  CHECK_ERROR(cudaMalloc((void **)&d_a, matSizeInBytes));
  CHECK_ERROR(cudaMalloc((void **)&d_b, matSizeInBytes));
  CHECK_ERROR(cudaMalloc((void **)&d_c, matSizeInBytes));

  CHECK_ERROR(
      cudaMemcpy(d_a, h_a.data(), matSizeInBytes, cudaMemcpyHostToDevice));
  CHECK_ERROR(
      cudaMemcpy(d_b, h_b.data(), matSizeInBytes, cudaMemcpyHostToDevice));

  /* Invoke the matmul kernel */
  dim3 blockSize(32, 32);
  dim3 gridSize;
  gridSize.x = ceil(static_cast<double>(N) / static_cast<double>(blockSize.x));
  gridSize.y = ceil(static_cast<double>(N) / static_cast<double>(blockSize.y));

  matmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

  /* Synchronize and copy data back to host */
  CHECK_ERROR(cudaDeviceSynchronize());

  CHECK_ERROR(
      cudaMemcpy(h_c.data(), d_c, matSizeInBytes, cudaMemcpyDeviceToHost));

  /* Free device memory */
  CHECK_ERROR(cudaFree(d_a));
  CHECK_ERROR(cudaFree(d_b));
  CHECK_ERROR(cudaFree(d_c));

  /* Verify results */
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
