#ifndef CHANNELIZER_HPP
#define CHANNELIZER_HPP
#include <cuComplex.h>
#include <cufft.h>
#include "utils.hpp"
#include <complex>
#include "transpose.hpp"
#include <cstdint>
#include <cassert>
using RawDataType = int16_t;
using FloatType = float;
struct RawComplex {
    RawDataType real;
    RawDataType imag;
};

static constexpr float PI = 3.14159265358979323846f;

__global__ void shift_freq_cast(const RawComplex *transposed_data,
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

struct Channelizer {
    size_t nsteps;
    size_t nch_coarse;
    size_t nch_fine_per_coarse_full;
    std::unique_ptr<RawComplex[], Deleter<RawComplex>> raw_data;
    std::unique_ptr<RawComplex[], Deleter<RawComplex>> transposed_data;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> working_mem1;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> working_mem2;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> freq_shift_factor;  // with length nch_fine_per_coarse*2
    cufftHandle fft_handle{0};

    Channelizer() = delete;
    Channelizer(const Channelizer &) = delete;
    Channelizer &operator=(const Channelizer &) = delete;

    virtual ~Channelizer() {
        cufftDestroy(this->fft_handle);
    }

    Channelizer(size_t nsteps, size_t nch_coarse1, size_t nch_fine_per_coarse_full1)
        : nsteps(nsteps),
          nch_coarse(nch_coarse1),
          nch_fine_per_coarse_full(nch_fine_per_coarse_full1),
          raw_data(gpu_alloc<RawComplex>(nsteps * nch_coarse)),
          transposed_data(gpu_alloc<RawComplex>(nsteps * nch_coarse)),
          working_mem1(gpu_alloc<cuComplex>(nsteps * nch_coarse)),
          working_mem2(gpu_alloc<cuComplex>(nsteps * nch_coarse)),
          freq_shift_factor(gpu_alloc<cuComplex>(nch_fine_per_coarse_full1 * 2)) {
        assert(nsteps % (2 * nch_fine_per_coarse_full1) == 0);
        std::vector<std::complex<FloatType>> factor(nch_fine_per_coarse_full1 * 2);
        for (int i = 0; i < factor.size(); ++i) {
            factor[i] = std::exp(std::complex<FloatType>{0.0f, -i * PI / nch_fine_per_coarse_full1});
            // factor[i] = std::complex<FloatType>(1.0, 0.0);
        }
        cuda_mem_cpy(
            this->freq_shift_factor.get(), factor.data(), nch_fine_per_coarse_full1 * 2, cudaMemcpyHostToDevice);

        CUFFT_CALL(cufftPlan1d(
            &this->fft_handle, nch_fine_per_coarse_full1, CUFFT_C2C, nch_coarse * nsteps / nch_fine_per_coarse_full1));
    }

    void put_raw(const std::complex<RawDataType> *data) {
        cuda_mem_cpy(this->raw_data.get(), data, nsteps * nch_coarse, cudaMemcpyHostToDevice);
    }

    void transpose() {
        dim3 grid(32, 32);
        dim3 block(32, 32, 1);
        transpose_kernel<<<grid, block>>>(raw_data.get(), transposed_data.get(), nsteps, nch_coarse);
    }

    void get_transposed(std::complex<RawDataType> *dst) const {
        cuda_mem_cpy(dst, this->transposed_data.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
    }

    void shift() {
        dim3 grid(256, 256);
        dim3 block(32, 32, 1);
        shift_freq_cast<<<grid, block>>>(this->transposed_data.get(),
                                         this->working_mem1.get(),
                                         this->freq_shift_factor.get(),
                                         this->nsteps,
                                         this->nch_coarse,
                                         this->nch_fine_per_coarse_full);
    }

    void get_working_mem(std::complex<FloatType> *dst, int n) const {
        if (n == 1) {
            cuda_mem_cpy(dst, this->working_mem1.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
        } else {
            cuda_mem_cpy(dst, this->working_mem2.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
        }
    }

    void fft() {
        cufftExecC2C(this->fft_handle, working_mem1.get(), working_mem2.get(), CUFFT_FORWARD);
    }

    void rearrange() {
        dim3 grid(64, 64);
        dim3 block(32, 32, 1);
        transpose_kernel<<<grid, block>>>(working_mem2.get(),
                                          working_mem1.get(),
                                          nsteps * nch_coarse / nch_fine_per_coarse_full,
                                          nch_fine_per_coarse_full);
        // size_t i = 0;
        for (size_t i1 = nch_fine_per_coarse_full * 3 / 4, i = 0; i1 < nch_fine_per_coarse_full * 5 / 4; ++i1, ++i) {
            size_t ich = i1 % nch_fine_per_coarse_full;
            for (size_t ich_coarse = 0; ich_coarse != this->nch_coarse; ++ich_coarse) {
                cuda_mem_cpy(working_mem2.get() +
                                 nsteps / nch_fine_per_coarse_full * (ich_coarse * nch_fine_per_coarse_full / 2 + i),
                             working_mem1.get() + nch_coarse * nsteps / nch_fine_per_coarse_full * ich +
                                 ich_coarse * nsteps / nch_fine_per_coarse_full,
                             nsteps / nch_fine_per_coarse_full,
                             cudaMemcpyDeviceToDevice);
            }
        }
    }

    std::vector<std::complex<FloatType>> peek_channelized() {
        std::vector<std::complex<FloatType>> result(nsteps * nch_coarse / 2);
        cuda_mem_cpy(result.data(), working_mem2.get(), nsteps * nch_coarse / 2, cudaMemcpyDeviceToHost);
        return result;
    }
};

#endif
