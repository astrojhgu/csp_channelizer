#ifndef KERNELS_H
#define KERNELS_H
#include <cuComplex.h>
#include "utils.hpp"
#include <complex>
#include <cstdint>
#include <cassert>

#include "types.hpp"
__global__ void shift_freq_cast_kernel(const RawComplex *transposed_data,
                                       cuComplex *shifted_data,
                                       const cuComplex *factor,
                                       size_t nsteps,
                                       size_t nch_coarse,
                                       size_t nch_fine_per_coarse_full,
                                       size_t coeff_len);

__global__ void pfb_conv_kernel(const cuComplex *input,
                                cuComplex *output,
                                const FloatType *coeff,
                                size_t nsteps,
                                size_t nch_coarse,
                                size_t nch_fine_per_coarse,
                                size_t coeff_len);

#endif
