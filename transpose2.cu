#include "utils.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

#define TILE_DIM 8
#define BLOCK_ROWS 1
template <typename T>
__global__ void transpose_kernel(T *A, T *A_t, int a_width, int a_height) {
  __shared__ T block[TILE_DIM][TILE_DIM + 1];

  int blockIdx_x, blockIdx_y;

  if (a_width == a_height) {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  } else {
    int bid = blockIdx.x + gridDim.x * blockIdx.y;
    blockIdx_y = bid % gridDim.y;
    blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
  }

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + a_width * yIndex;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + a_height * yIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    block[threadIdx.y + i][threadIdx.x] = A[index_in + i * a_width];
  }

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    A_t[index_out + i * a_height] = block[threadIdx.x][threadIdx.y + i];
  }
}

template <typename T>
void transpose(const T *d_a, T *d_b, const size_t nrows, const size_t ncols) {

  auto d_data_in = gpu_alloc<T>(ncols * nrows);
  auto d_data_out = gpu_alloc<T>(ncols * nrows);
  CHECK_CUDA_ERROR(cudaMemcpy(d_data_in.get(), d_a, nrows * ncols * sizeof(T),
                              cudaMemcpyHostToDevice));
  dim3 grid(128, 128);
  dim3 block(32, 32, 1);
  transpose_kernel<<<grid, block>>>(d_data_in.get(), d_data_out.get(), nrows,
                                    ncols);
  CHECK_CUDA_ERROR(cudaMemcpy(d_b, d_data_out.get(), nrows * ncols * sizeof(T),
                              cudaMemcpyDeviceToHost));
  // CHECK_CUDA_ERROR(cudaDeviceReset());
}

int main() {

  using T = float;
  int nrows = 88;
  int ncols = 937500;
  auto d_data_in = gpu_alloc<T>(ncols * nrows);
  auto d_data_out = gpu_alloc<T>(ncols * nrows);
  vector<T> data_in(nrows * ncols);
  vector<T> data_out(nrows * ncols);

  std::cerr << "filling" << std::endl;
  for (int i = 0; i < nrows * ncols; ++i) {
    data_in[i] = i;
  }

  /*
  for (int i = 0; i < nrows; ++i) {
    for (int j = 0; j < ncols; ++j) {
      cout << data_in[i * ncols + j] << " ";
    }
    cout << endl;
  }*/

  // while (1)
  for (int i = 0; i < 1; ++i) {
    std::cout << i << std::endl;
    transpose(data_in.data(), data_out.data(), nrows, ncols);
  }

  std::cerr << "validating" << std::endl;
  for (int j = 0; j < nrows; ++j) {
    for (int i = 0; i < ncols; ++i) {
      assert(data_out[i * nrows + j] == data_in[j * ncols + i]);
    }
  }
}