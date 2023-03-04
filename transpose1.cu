//https://stackoverflow.com/a/15892972
#include "utils.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;
constexpr int TILE_DIM = 16;
constexpr int BLOCK_ROWS = 16;


template <typename T>
__global__ void transpose(T *a, T *c, int m, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int smallest = m < n ? m : n;

  while (j < smallest) {
    i = threadIdx.x + blockIdx.x * blockDim.x;
    while (i < j) {
      c[i * m + j] = a[j * n + i];
      c[j * m + i] = a[i * n + j];
      i += blockDim.x * gridDim.x;
    }
    if (i == j)
      c[j * m + i] = a[i * n + j];

    j += blockDim.y * gridDim.y;
  }

  if (m > n) {
    i = threadIdx.x + blockIdx.x * blockDim.x + n;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    while (i < m) {
      j = threadIdx.y + blockIdx.y * blockDim.y;
      while (j < n) {
        c[j * m + i] = a[i * n + j];
        j += blockDim.y * gridDim.y;
      }
      i += blockDim.x * gridDim.x;
    }
  } else {
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y + m;

    while (i < m) {
      j = threadIdx.y + blockIdx.y * blockDim.y + m;
      while (j < n) {
        c[j * m + i] = a[i * n + j];
        j += blockDim.y * gridDim.y;
      }
      i += blockDim.x * gridDim.x;
    }
  }
}

int main() {
  unsigned int blockx = 16;
  unsigned int blocky = 8;
  int nrows = 4;
  int ncols = 3;
  auto d_data_in = gpu_alloc<float>(ncols * nrows);
  auto d_data_out = gpu_alloc<float>(ncols * nrows);
  vector<float> data_in(nrows * ncols);
  vector<float> data_out(nrows * ncols);

  for (int i = 0; i < nrows * ncols; ++i) {
    data_in[i] = i;
  }

  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      cout << data_in[i * ncols + j] << " ";
    }
    cout << endl;
  }
  cout << "==" << std::endl;
  cudaMemcpy(d_data_in.get(), data_in.data(), nrows * ncols * sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 grid(16, 16);
  dim3 block(32, 32);

  transpose<<<grid, block>>>(d_data_in.get(), d_data_out.get(), nrows, ncols);

  cudaMemcpy(data_out.data(), d_data_out.get(), nrows * ncols * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < ncols; ++i) {
    for (int j = 0; j < nrows; ++j) {
      cout << data_out[i * nrows + j] << " ";
    }
    cout << endl;
  }
}
