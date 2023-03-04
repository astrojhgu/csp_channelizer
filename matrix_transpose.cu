/**
 * @file matrixTranspose.cu
 * @brief main application, performs matrix trnspose with different
 * optimizations
 */

#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;

#include "matrix_transpose.h"

void deviceInfor(void) {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    printf("There are no available device(s) that support CUDA\n");
    exit(1);
  }
  printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  int dev = 0, driverVersion = 0, runtimeVersion = 0;
  CHECK_CUDA(cudaSetDevice(dev));
  cudaDeviceProp deviceProp;
  CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Device %d: \"%s\"\n", dev, deviceProp.name);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
         driverVersion / 1000, (driverVersion % 100) / 10,
         runtimeVersion / 1000, (runtimeVersion % 100) / 10);
  printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
         deviceProp.major, deviceProp.minor);
  printf("  Total amount of global memory:                 %.2f MBytes (%llu "
         "bytes)\n",
         (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
         (unsigned long long)deviceProp.totalGlobalMem);
  printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
         "GHz)\n",
         deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf("  Memory Clock rate:                             %.0f Mhz\n",
         deviceProp.memoryClockRate * 1e-3f);
  printf("  Memory Bus Width:                              %d-bit\n",
         deviceProp.memoryBusWidth);
  if (deviceProp.l2CacheSize) {
    printf("  L2 Cache Size:                                 %d bytes\n",
           deviceProp.l2CacheSize);
  }

  printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
         "2D=(%d,%d), 3D=(%d,%d,%d)\n",
         deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
         deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
         deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
         "2D=(%d,%d) x %d\n",
         deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
         deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
         deviceProp.maxTexture2DLayered[2]);
  printf("  Total amount of constant memory:               %lu bytes\n",
         deviceProp.totalConstMem);
  printf("  Total amount of shared memory per block:       %lu bytes\n",
         deviceProp.sharedMemPerBlock);
  printf("  Total number of registers available per block: %d\n",
         deviceProp.regsPerBlock);
  printf("  Warp size:                                     %d\n",
         deviceProp.warpSize);
  printf("  Maximum number of threads per multiprocessor:  %d\n",
         deviceProp.maxThreadsPerMultiProcessor);
  printf("  Maximum number of threads per block:           %d\n",
         deviceProp.maxThreadsPerBlock);
  printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
         deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
         deviceProp.maxThreadsDim[2]);
  printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
         deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
         deviceProp.maxGridSize[2]);
  printf("  Maximum memory pitch:                          %lu bytes\n",
         deviceProp.memPitch);
}

uint_8 randomUint8(void) {
  time_t t;
  srand((unsigned)time(&t));
  return rand() % 0xff;
}

template <typename T>
uint_8 checkRes(T *host, T *device, unsigned int nx, unsigned int ny) {
  unsigned int i;
  for (i = 0; i < (nx * ny); i++)
    if (host[i] != device[i])
      return 1;
  std::cout << "matched" << std::endl;
  return 0;
}

static void computeHost(float *hSource, float *hDest, unsigned int nx,
                        unsigned ny) {
  unsigned int i, j;
  for (i = 0; i < ny; i++)
    for (j = 0; j < nx; j++)
      hDest[j * nx + i] = hSource[i * nx + j];
}

int main(int argc, char **argv) {
#if (VERBOSE)
  deviceInfor();
#endif

  int iKernel;
  unsigned int nx = 1 << 10;
  unsigned int ny = 1 << 11;
  unsigned int blockx = 16;
  unsigned int blocky = 16;
  unsigned int i;

  if (argc < 2) {
    fprintf(stderr,
            "usage: <%s> <iKernel> [optional <blockx>] [optional <blocky>] "
            "[optional <nx>] [optional <ny>]\n",
            argv[0]);
    fprintf(stderr, "iKernel=0: copyRow\n");
    fprintf(stderr, "iKernel=1: copyCol\n");
    fprintf(stderr, "ikernel=2: transposeNaiveRow\n");
    fprintf(stderr, "ikernel=3: transposeNaiveCol\n");
    fprintf(stderr, "ikernel=4: transposeUnroll4Row\n");
    fprintf(stderr, "ikernel=5: transposeUnroll4Col\n");
    fprintf(stderr, "ikernel=6: transposeDiagonalRow\n");
    fprintf(stderr, "ikernel=7: transposeSmem\n");
    fprintf(stderr, "ikernel=8: transposeSmemPad\n");
    fprintf(stderr, "ikernel=9: transposeSmemUnrollPadDyn\n");
    exit(1);
  }

  iKernel = atoi(argv[1]);
  if (argc > 2)
    blockx = atoi(argv[2]);
  if (argc > 3)
    blocky = atoi(argv[3]);
  if (argc > 4)
    nx = atoi(argv[4]);
  if (argc > 5)
    ny = atoi(argv[5]);

  dim3 block(blockx, blocky);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
  // double iStart, iElaps;
  double effBW;

  // data
  float *hSource, *hDest;
  float *dSource, *dDest;
  float *gpuRes;
  // alloc on host
  hSource = (float *)malloc(nx * ny * sizeof(float));
  CHECK_PTR(hSource);
  hDest = (float *)malloc(nx * ny * sizeof(float));
  CHECK_PTR(hDest);
  gpuRes = (float *)malloc(nx * ny * sizeof(float));
  CHECK_PTR(gpuRes);
  // alloc on device
  CHECK_CUDA(cudaMalloc((void **)&dSource, nx * ny * sizeof(float)));
  CHECK_CUDA(cudaMalloc((void **)&dDest, nx * ny * sizeof(float)));

  // init on host
  for (i = 0; i < nx * ny; i++)
    hSource[i] = randomUint8() / (float)1.0f;
  // copy on GPU
  CHECK_CUDA(cudaMemcpy(dSource, hSource, nx * ny * sizeof(float),
                        cudaMemcpyHostToDevice));

#if (VERBOSE)
  fprintf(stdout,
          "nx=%d, ny=%d, %lu Bytes, grid(%d,%d), block(%d,%d), #threads=%llu\n",
          nx, ny, (nx * ny * sizeof(float)), grid.x, grid.y, block.x, block.y,
          (long long unsigned int)(block.x * block.y * grid.x * grid.y));
#endif

  void (*kernel)(float *, float *, unsigned int, unsigned int);
  char *kernelName;

  switch (iKernel) {
  /*setup copyRow*/
  case 0:
#if (VERBOSE)
    fprintf(stdout, "copyRow kernel selected\n");
#endif
    kernelName = strdup("copyRow");
    kernel = &copyRow;
    break;
  /*setup copyCol*/
  case 1:
#if (VERBOSE)
    fprintf(stdout, "copyCol kernel selected\n");
#endif
    kernelName = strdup("copyCol");
    kernel = &copyCol;
    break;
  /*setup transposeNaiveRow*/
  case 2:
#if (VERBOSE)
    fprintf(stdout, "transposeNaiveRow kernel selected\n");
#endif
    kernelName = strdup("transposeNaiveRow");
    kernel = &transposeNaiveRow;
    break;
  /*setup transposeNaiveCol*/
  case 3:
#if (VERBOSE)
    fprintf(stdout, "transposeNaiveCol kernel selected\n");
#endif
    kernelName = strdup("transposeNaiveCol");
    kernel = &transposeNaiveCol;
    break;
  /*setup transposeUnroll4Row*/
  case 4:
#if (VERBOSE)
    fprintf(stdout, "transposeUnroll4Row kernel selected\n");
#endif
    kernelName = strdup("transposeUnroll4Row");
    kernel = &transposeUnroll4Row;
    grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
    break;
  /*setup transposeUnroll4Col*/
  case 5:
#if (VERBOSE)
    fprintf(stdout, "transposeUnroll4Col kernel selected\n");
#endif
    kernelName = strdup("transposeUnroll4Col");
    kernel = &transposeUnroll4Col;
    grid.x = (nx + block.x * 4 - 1) / (block.x * 4);
    break;
  /*setup transposeDiagonalRow*/
  case 6:
#if (VERBOSE)
    fprintf(stdout, "transposeDiagonalRow kernel selected\n");
#endif
    kernelName = strdup("transposeDiagonalRow");
    kernel = &transposeDiagonalRow;
    break;
  /*setup transposeDiagonalCol*/
  case 7:
#if (VERBOSE)
    fprintf(stdout, "transposeDiagonalCol kernel selected\n");
#endif
    kernelName = strdup("transposeDiagonalCol");
    kernel = &transposeDiagonalCol;
    break;
  /*setup transposeSmem*/
  case 8:
#if (VERBOSE)
    fprintf(stdout, "transposeSmem kernel selected\n");
#endif
    kernelName = strdup("transposeSmem");
    kernel = &transposeSmem;
    break;
  /*setup transposeSmemPad*/
  case 9:
#if (VERBOSE)
    fprintf(stdout, "transposeSmemPad kernel selected\n");
#endif
    kernelName = strdup("transposeSmemPad");
    kernel = &transposeSmemPad;
    break;
  /*setup transposeSmemUnrollPadDyn*/
  case 10:
#if (VERBOSE)
    fprintf(stdout, "transposeSmemUnrollPadDyn kernel selected\n");
#endif
    kernelName = strdup("transposeSmemUnrollPadDyn");
    kernel = &transposeSmemUnrollPadDyn;
    grid.x = (nx + block.x * 2 - 1) / (block.x * 2);
    break;
  default:
#if (VERBOSE)
    fprintf(stderr,
            "error in kernel selection, only values between 0-9 are allowed\n");
#endif
    exit(1);
    break;
  }

  if (iKernel == 0 || iKernel == 1) {
    // run templatized kernels
    // iStart = cpuSecond();
    kernel<<<grid, block>>>(dSource, dDest, nx, ny);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
  } else if (iKernel == 9) {
    // run kernel with dynamic shared memory
    // iStart = cpuSecond();
    kernel<<<grid, block, (BDIMX * 2 + IPAD) * BDIMY * sizeof(float)>>>(
        dSource, dDest, nx, ny);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
  } else {
    // run normal kernel
    // iStart = cpuSecond();
    kernel<<<grid, block>>>(dSource, dDest, nx, ny);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    // iElaps = cpuSecond() - iStart;
  }

  // get data back from gpu
  CHECK_CUDA(cudaMemcpy(gpuRes, dDest, nx * ny * sizeof(float),
                        cudaMemcpyDeviceToHost));

#if (CHECK)
  // compute result on host
  computeHost(hSource, hDest, nx, ny);
  // check kernel results
  if (iKernel > 1) {
    if (checkRes(hDest, gpuRes, nx, ny) == 1) {
      fprintf(stderr, "GPU and CPU result missmatch!\n");
      exit(1);
    }
  } else {
    if (checkRes(hSource, gpuRes, nx, ny) == 1) {
      fprintf(stderr, "GPU and CPU result missmatch!\n");
      exit(1);
    }
  }
#endif

  // calculate effective_bandwidth (MB/s)
  // effBW=(2 * nx * ny * sizeof(float)) / ((1e+6f)*iElaps);
  /*printf on stdout used for profiling
   * <kernelName>,<elapsedTime>,<bandwidth>,<grid(x,y)>,<block(x,y)>*/
  // fprintf(stdout,"%s,%f,%f,grid(%d.%d),block(%d.%d)\n",kernelName, effBW,
  // iElaps, grid.x, grid.y, block.x, block.y);

  // free host and device memory
  CHECK_CUDA(cudaFree(dSource));
  CHECK_CUDA(cudaFree(dDest));
  free(hSource);
  free(hDest);
  free(gpuRes);

  // reset device
  CHECK_CUDA(cudaDeviceReset());

  return 0;
}
