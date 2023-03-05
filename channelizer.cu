#include <cuComplex.h>
#include <cufft.h>
#include "utils.hpp"
#include <complex>
#include "transpose.hpp"
#include <cstdint>
#include <cassert>
#include "channelizer.hpp"
#include "types.hpp"
#include "kernels.h"

Channelizer::~Channelizer() {
    cufftDestroy(this->fft_handle);
}

Channelizer::Channelizer(size_t nsteps,
                         size_t nch_coarse1,
                         size_t nch_fine_per_coarse_full1,
                         const vector<FloatType> &coeffs1)
    : nsteps(nsteps),
      nch_coarse(nch_coarse1),
      nch_fine_per_coarse_full(nch_fine_per_coarse_full1),
      raw_data(gpu_alloc<RawComplex>(nsteps * nch_coarse)),
      transposed_data(gpu_alloc<RawComplex>(nsteps * nch_coarse)),
      working_mem1(gpu_alloc<cuComplex>(nsteps * nch_coarse)),
      working_mem2(gpu_alloc<cuComplex>(nsteps * nch_coarse)),
      freq_shift_factor(gpu_alloc<cuComplex>(nch_fine_per_coarse_full1 * 2)),
      coeffs(gpu_alloc<FloatType>(coeffs1.size())),
      coeff_len(coeffs1.size()) {
    assert(nsteps % (2 * nch_fine_per_coarse_full1) == 0);
    std::vector<std::complex<FloatType>> factor(nch_fine_per_coarse_full1 * 2);
    for (int i = 0; i < factor.size(); ++i) {
        factor[i] = std::exp(std::complex<FloatType>{0.0f, -i * PI / nch_fine_per_coarse_full1});
        // factor[i] = std::complex<FloatType>(1.0, 0.0);
    }

    assert(coeffs1.size() % nch_fine_per_coarse_full1 == 0);
    cuda_mem_cpy(this->freq_shift_factor.get(), factor.data(), nch_fine_per_coarse_full1 * 2, cudaMemcpyHostToDevice);

    cuda_mem_cpy(this->coeffs.get(), coeffs1.data(), coeffs1.size(), cudaMemcpyHostToDevice);

    CUFFT_CALL(cufftPlan1d(
        &this->fft_handle, nch_fine_per_coarse_full1, CUFFT_C2C, nch_coarse * nsteps / nch_fine_per_coarse_full1));
}

void Channelizer::put_raw(const std::complex<RawDataType> *data) {
    cuda_mem_cpy(this->raw_data.get(), data, nsteps * nch_coarse, cudaMemcpyHostToDevice);
}

void Channelizer::transpose() {
    dim3 grid(32, 32);
    dim3 block(32, 32, 1);
    transpose_kernel<<<grid, block>>>(raw_data.get(), transposed_data.get(), nsteps, nch_coarse);
}

void Channelizer::get_transposed(std::complex<RawDataType> *dst) const {
    cuda_mem_cpy(dst, this->transposed_data.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
}

void Channelizer::shift() {
    dim3 grid(nch_coarse, nsteps/512);
    dim3 block(512, 1, 1);
    shift_freq_cast_kernel<<<grid, block>>>(this->transposed_data.get(),
                                            this->working_mem1.get(),
                                            this->freq_shift_factor.get(),
                                            this->nsteps,
                                            this->nch_coarse,
                                            this->nch_fine_per_coarse_full);
}

void Channelizer::get_working_mem(std::complex<FloatType> *dst, int n) const {
    if (n == 1) {
        cuda_mem_cpy(dst, this->working_mem1.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
    } else {
        cuda_mem_cpy(dst, this->working_mem2.get(), nsteps * nch_coarse, cudaMemcpyDeviceToHost);
    }
}

void Channelizer::filter() {
    pfb_conv_kernel<<<dim3(this->nch_coarse, 1024), this->nch_fine_per_coarse_full>>>(this->working_mem1.get(),
                                                                                      this->working_mem2.get(),
                                                                                      this->coeffs.get(),
                                                                                      this->nsteps,
                                                                                      this->nch_coarse,
                                                                                      this->nch_fine_per_coarse_full,
                                                                                      coeff_len);
}

void Channelizer::fft() {
    cufftExecC2C(this->fft_handle, working_mem2.get(), working_mem1.get(), CUFFT_FORWARD);
}

void Channelizer::rearrange() {
    dim3 grid(64, 64);
    dim3 block(32, 32, 1);
    transpose_kernel<<<grid, block>>>(working_mem1.get(),
                                      working_mem2.get(),
                                      nsteps * nch_coarse / nch_fine_per_coarse_full,
                                      nch_fine_per_coarse_full);
    // size_t i = 0;
    for (size_t i1 = nch_fine_per_coarse_full * 3 / 4, i = 0; i1 < nch_fine_per_coarse_full * 5 / 4; ++i1, ++i) {
        size_t ich = i1 % nch_fine_per_coarse_full;
        for (size_t ich_coarse = 0; ich_coarse != this->nch_coarse; ++ich_coarse) {
            cuda_mem_cpy(working_mem1.get() +
                             nsteps / nch_fine_per_coarse_full * (ich_coarse * nch_fine_per_coarse_full / 2 + i),
                         working_mem2.get() + nch_coarse * nsteps / nch_fine_per_coarse_full * ich +
                             ich_coarse * nsteps / nch_fine_per_coarse_full,
                         nsteps / nch_fine_per_coarse_full,
                         cudaMemcpyDeviceToDevice);
        }
    }
}

void Channelizer::channelize(const std::vector<std::complex<RawDataType>> data, std::vector<std::complex<FloatType>>& output) {
    assert(data.size()==nsteps*nch_coarse);
    assert(output.size()==nsteps*nch_coarse/2);
    put_raw(data.data());
    transpose();
    shift();
    filter();
    fft();
    rearrange();
    cuda_mem_cpy(output.data(), working_mem1.get(), nsteps * nch_coarse / 2, cudaMemcpyDeviceToHost);
}

std::vector<std::complex<FloatType>> Channelizer::peek_channelized() {
    std::vector<std::complex<FloatType>> result(nsteps * nch_coarse / 2);
    cuda_mem_cpy(result.data(), working_mem1.get(), nsteps * nch_coarse / 2, cudaMemcpyDeviceToHost);
    return result;
}
