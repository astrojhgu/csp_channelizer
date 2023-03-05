#include <cuComplex.h>
#include "utils.hpp"
#include <complex>
#include <cstdint>
#include <cassert>
#include "kernels.h"

#include "types.hpp"
__global__ void shift_freq_cast_kernel(const RawComplex *transposed_data,
                                       cuComplex *shifted_data,
                                       const cuComplex *factor,
                                       size_t nsteps,
                                       size_t nch_coarse,
                                       size_t nch_fine_per_carse_full) {
    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x;
    size_t threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    size_t nthreads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;
    size_t ntasks = nsteps * nch_coarse;
    size_t ntasks_per_thread = ceil(float(ntasks) / float(nthreads));
    // if (threadId == 0) {
    //     printf("ntask per thread: %ld\n", ntasks_per_thread);
    // }

    for (size_t i = 0; i < ntasks_per_thread; ++i) {
        size_t task_id = threadId * ntasks_per_thread + i;
        if (task_id < nsteps * nch_coarse) {
            size_t ch = task_id / nsteps;
            size_t step = task_id - ch * nsteps;
            auto f = factor[step % (nch_fine_per_carse_full * 2)];
            auto x = transposed_data[task_id];
            auto xc = cuComplex(make_float2(x.real, x.imag));

            shifted_data[task_id] = make_float2(xc.x * f.x - xc.y * f.y, xc.x * f.y + xc.y * f.x);
        }
    }
}

__global__ void pfb_conv_kernel(const cuComplex *input,
                                cuComplex *output,
                                const FloatType *coeff,
                                size_t nsteps,
                                size_t nch_coarse,
                                size_t nch_fine_per_coarse,
                                size_t coeff_len) {
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(blockDim.x == nch_fine_per_coarse);
    // assert(gridDim.x == nch_coarse);
    // assert(gridDim.y  nsteps / nch_fine_per_coarse);
    assert(gridDim.z == 1);
    if (blockIdx.x >= nch_coarse) {
        return;
    }

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    /*if (threadId == 0) {
    printf("pfb\n");
    }*/

    assert(coeff_len % nch_fine_per_coarse == 0);

    size_t tap_per_branch = coeff_len / nch_fine_per_coarse;
    size_t ntasks = nsteps / nch_fine_per_coarse;
    size_t ntasks_per_thread = ceil((float) ntasks / (float) gridDim.y);
    // printf("y=%d\n", (blockIdx.y+1)*ntasks_per_thread*nch_fine_per_coarse);
    for (int t = 0; t < ntasks_per_thread; ++t) {
        const cuComplex *base_in =
            input + blockIdx.x * nsteps + (blockIdx.y * ntasks_per_thread + t) * nch_fine_per_coarse;
        cuComplex *base_out = output + blockIdx.x * nsteps + (blockIdx.y * ntasks_per_thread + t) * nch_fine_per_coarse;
        base_out[threadIdx.x] =
            make_float2(base_in[threadIdx.x].x * coeff[threadIdx.x], base_in[threadIdx.x].y * coeff[threadIdx.x]);

        int imax = min(tap_per_branch, (nsteps / nch_fine_per_coarse - (blockIdx.y * ntasks_per_thread + t)));

        for (int i = 1; i < imax; ++i) {
            FloatType c = coeff[threadIdx.x + i * nch_fine_per_coarse];
            base_out[threadIdx.x].x += base_in[threadIdx.x + i * nch_fine_per_coarse].x * c;
            base_out[threadIdx.x].y += base_in[threadIdx.x + i * nch_fine_per_coarse].y * c;
        }
    }
}
