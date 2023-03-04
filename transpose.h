#include <cooperative_groups.h>
#include <cuda_runtime.h>
constexpr int TILE_DIM = 16;
constexpr int BLOCK_ROWS = 16;

template <typename T>
__global__ void transpose_kernel(T* matTran, T* matIn, size_t n, size_t m){
    __shared__ T tile[TILE_DIM][TILE_DIM];
    size_t i_n = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t i_m = blockIdx.y * TILE_DIM + threadIdx.y; // <- threadIdx.y only between 0 and 7

    // Load matrix into tile
    // Every Thread loads in this case 4 elements into tile.
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        if(i_n < n  && (i_m+i) < m){
            tile[threadIdx.y+i][threadIdx.x] = matIn[(i_m+i)*n + i_n];
        }
    }
    __syncthreads();

    i_n = blockIdx.y * TILE_DIM + threadIdx.x; 
    i_m = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS){
        if(i_n < m  && (i_m+i) < n){
            matTran[(i_m+i)*m + i_n] = tile[threadIdx.x][threadIdx.y + i]; // <- multiply by m, non-squared!
        }
    }
}