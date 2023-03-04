#ifndef MATRIX_TRANS
#define MATRIX_TRANS

#include "common.h"
//#include <cstdio>
//#include <cstdlib>
//#include <cstring>
//https://github.com/Luca-Dalmasso/matrixTransposeCUDA

/**
 * @defgroup set of macros for this application (default all to 0)
 * @{
 */

/*enable verbose stdout (disable this when profiling)*/
#define VERBOSE 0
/*size X of shared memory tile*/
#define BDIMX 16
/*size Y of shared memory tile*/
#define BDIMY 16
/* shared memory padding size (1= used for 4byte banks, 2=used when shared
 * memory has 8byte banks)*/
#define IPAD 2
/*enable host computations for error checking*/
#define CHECK 1

/** @} */

// transpose kernels

/**
 * @defgroup matrix transpose kernels
 * @{
 */

/**
 * @brief NAIVE row based version of matrix transpose algorithm
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeNaiveRow(T *in, T *out, unsigned int nx,
                                  unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= nx || iy >= ny)
    return;
  out[ix * ny + iy] = in[iy * nx + ix];
}

/**
 * @brief NAIVE columns based version of matrix transpose algorithm
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeNaiveCol(T *in, T *out, unsigned int nx,
                                  unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= nx || iy >= ny)
    return;
  out[iy * nx + ix] = in[ix * ny + iy];
}

/**
 * @brief read in rows and write in columns + unroll 4 blocks
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeUnroll4Row(T *in, T *out, unsigned int nx,
                                    unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

  unsigned int ti = iy * nx + ix; // access in rows
  unsigned int to = ix * ny + iy; // access in columns

  if (ix + 3 * blockDim.x < nx && iy < ny) {
    out[to] = in[ti];
    out[to + ny * blockDim.x] = in[ti + blockDim.x];
    out[to + ny * 2 * blockDim.x] = in[ti + 2 * blockDim.x];
    out[to + ny * 3 * blockDim.x] = in[ti + 3 * blockDim.x];
  }
}

/**
 * @brief read in columns and write in rows + unroll 4 blocks
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeUnroll4Col(T *in, T *out, unsigned int nx,
                                    unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x * 4 + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

  unsigned int ti = iy * nx + ix; // access in rows
  unsigned int to = ix * ny + iy; // access in columns

  if (ix + 3 * blockDim.x < nx && iy < ny) {
    out[ti] = in[to];
    out[ti + blockDim.x] = in[to + blockDim.x * ny];
    out[ti + 2 * blockDim.x] = in[to + 2 * blockDim.x * ny];
    out[ti + 3 * blockDim.x] = in[to + 3 * blockDim.x * ny];
  }
}

/**
 * @brief read in rows and write in colunms + diagonal coordinate transform,
 * diagonal coordinate system allow to reduce the partition camping phenomenom.
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 * @see REAMDE.md: there's a section in which diagonal system and related
 * partition camping issue are explained
 */
 template <typename T>
__global__ void transposeDiagonalRow(T *in, T *out, unsigned int nx,
                                     unsigned int ny) {
  unsigned int blk_y = blockIdx.x;
  unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

  unsigned int ix = blockDim.x * blk_x + threadIdx.x;
  unsigned int iy = blockDim.y * blk_y + threadIdx.y;

  if (ix < nx && iy < ny) {
    out[ix * ny + iy] = in[iy * nx + ix];
  }
}

/**
 * @brief read in colunms and write in rows + diagonal coordinate transform,
 * diagonal coordinate system allow to reduce the partition camping phenomenom.
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 * @see REAMDE.md: there's a section in which diagonal system and related
 * partition camping issue are explained
 */
 template <typename T>
__global__ void transposeDiagonalCol(T *in, T *out, unsigned int nx,
                                     unsigned int ny) {
  unsigned int blk_y = blockIdx.x;
  unsigned int blk_x = (blockIdx.x + blockIdx.y) % gridDim.x;

  unsigned int ix = blockDim.x * blk_x + threadIdx.x;
  unsigned int iy = blockDim.y * blk_y + threadIdx.y;

  if (ix < nx && iy < ny) {
    out[iy * nx + ix] = in[ix * ny + iy];
  }
}

/**
 * @brief read in rows + write in rows by using shared memory (both access are
 * aligned and coalesced)
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeSmem(T *in, T *out, unsigned int nx,
                              unsigned int ny) {
  // static shared memory
  __shared__ T tile[BDIMY][BDIMX];

  // coordinate in original matrix
  unsigned int ix, iy, ti, to;
  ix = blockDim.x * blockIdx.x + threadIdx.x;
  iy = blockDim.y * blockIdx.y + threadIdx.y;

  // linear global memory index for original matrix
  ti = iy * nx + ix;

  // thread index in transposed block
  unsigned int bidx, irow, icol;
  bidx = threadIdx.y * blockDim.x + threadIdx.x;
  irow = bidx / blockDim.y;
  icol = bidx % blockDim.y;

  // coordinate in transposed matrix
  ix = blockDim.y * blockIdx.y + icol;
  iy = blockDim.x * blockIdx.x + irow;

  // linear global memory index for transposed matrix
  to = iy * ny + ix;

  // transpose with boundary test
  if (ix < nx && iy < ny) {
    // load data from global memory to shared memory
    tile[threadIdx.y][threadIdx.x] = in[ti];

    // thread synchronization
    __syncthreads();

    // store data to global memory from shared memory
    out[to] = tile[icol][irow];
  }
}

/**
 * @brief shared memory is used with padding to avoid bank conflicts
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeSmemPad(T *in, T *out, unsigned int nx,
                                 unsigned int ny) {
  // static shared memory with padding
  __shared__ T tile[BDIMY][BDIMX + IPAD];

  // coordinate in original matrix
  unsigned int ix, iy, ti, to;
  ix = blockDim.x * blockIdx.x + threadIdx.x;
  iy = blockDim.y * blockIdx.y + threadIdx.y;

  // linear global memory index for original matrix
  ti = iy * nx + ix;

  // thread index in transposed block
  unsigned int bidx, irow, icol;
  bidx = threadIdx.y * blockDim.x + threadIdx.x;
  irow = bidx / blockDim.y;
  icol = bidx % blockDim.y;

  // coordinate in transposed matrix
  ix = blockDim.y * blockIdx.y + icol;
  iy = blockDim.x * blockIdx.x + irow;

  // linear global memory index for transposed matrix
  to = iy * ny + ix;

  // transpose with boundary test
  if (ix < nx && iy < ny) {
    // load data from global memory to shared memory
    tile[threadIdx.y][threadIdx.x] = in[ti];

    // thread synchronization
    __syncthreads();

    // store data to global memory from shared memory
    out[to] = tile[icol][irow];
  }
}

/**
 * @brief dynamic shared memory used with padding and block unrolling to
 * increase throughput
 * @param in: source matrix
 * @param out: destination matrix
 * @param nx: #columns
 * @param ny: #rows
 */
 template <typename T>
__global__ void transposeSmemUnrollPadDyn(T *in, T *out,
                                          unsigned int nx, unsigned int ny) {
  extern __shared__ T tile[];

  unsigned int ix = blockDim.x * blockIdx.x * 2 + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

  unsigned int ti = iy * nx + ix;

  unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
  unsigned int irow = bidx / blockDim.y;
  unsigned int icol = bidx % blockDim.y;

  // coordinate in transposed matrix
  unsigned int ix2 = blockDim.y * blockIdx.y + icol;
  unsigned int iy2 = blockDim.x * 2 * blockIdx.x + irow;
  unsigned int to = iy2 * ny + ix2;

  // transpose with boundary test
  if (ix + blockDim.x < nx && iy < ny) {
    // load data from global memory to shared memory
    unsigned int row_idx = threadIdx.y * (blockDim.x * 2 + IPAD) + threadIdx.x;
    tile[row_idx] = in[ti];
    tile[row_idx + BDIMX] = in[ti + BDIMX];

    // thread synchronization
    __syncthreads();

    unsigned int col_idx = icol * (blockDim.x * 2 + IPAD) + irow;
    out[to] = tile[col_idx];
    out[to + ny * BDIMX] = tile[col_idx + BDIMX];
  }
}


template <typename T>
__global__ void copyRow(T *src, T *dest, unsigned int nx,
                        unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= nx || iy >= ny)
    return;
  dest[iy * nx + ix] = src[iy * nx + ix];
}

template <typename T>
__global__ void copyCol(T *src, T *dest, unsigned int nx,
                        unsigned int ny) {
  unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
  if (ix >= nx || iy >= ny)
    return;
  dest[ix * ny + iy] = src[ix * ny + iy];
}


#endif
