#ifndef TRANSPOSE_HPP
#define TRANSPOSE_HPP

#include "utils.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;

template <typename T>
__global__ void transpose_kernel(const T *d_a, T *d_b, const size_t rows, const size_t cols) {
    constexpr size_t BLOCK_SIZE = 32;
    __shared__ T mat[BLOCK_SIZE][BLOCK_SIZE + 1];
    size_t bh = ceil((float) rows / BLOCK_SIZE);
    size_t bw = ceil((float) cols / BLOCK_SIZE);

    for (size_t blocky = blockIdx.y; blocky < bh; blocky += gridDim.y) {
        for (size_t blockx = blockIdx.x; blockx < bw; blockx += gridDim.x) {
            size_t bx = blockx * BLOCK_SIZE;
            size_t by = blocky * BLOCK_SIZE;

            size_t i = by + threadIdx.y;
            size_t j = bx + threadIdx.x;

            if (i < rows && j < cols) {
                mat[threadIdx.x][threadIdx.y] = d_a[i * cols + j];
            }

            __syncthreads();

            size_t ti = bx + threadIdx.y;
            size_t tj = by + threadIdx.x;

            if (tj < rows && ti < cols) {
                d_b[ti * rows + tj] = mat[threadIdx.y][threadIdx.x];
            }

            __syncthreads();
        }
    }
}


template <typename T>
void transpose(const T *d_a, T *d_b, const size_t nrows, const size_t ncols) {
    auto d_data_in = gpu_alloc<T>(ncols * nrows);
    auto d_data_out = gpu_alloc<T>(ncols * nrows);
    CHECK_CUDA_ERROR(cudaMemcpy(d_data_in.get(), d_a, nrows * ncols * sizeof(T), cudaMemcpyHostToDevice));
    dim3 grid(128, 128);
    dim3 block(32, 32, 1);
    // for (int i = 0; i < 100; ++i) {
    // std::cout << i << std::endl;
    transpose_kernel<<<grid, block>>>(d_data_in.get(), d_data_out.get(), nrows, ncols);
    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    //}

    CHECK_CUDA_ERROR(cudaMemcpy(d_b, d_data_out.get(), nrows * ncols * sizeof(T), cudaMemcpyDeviceToHost));
    // CHECK_CUDA_ERROR(cudaDeviceReset());
}

#endif
