#ifndef CHANNELIZER_HPP
#define CHANNELIZER_HPP
#include <cuComplex.h>
#include <cufft.h>
#include "utils.hpp"
#include <complex>
#include "transpose.hpp"
#include <cstdint>
#include <cassert>



#include "types.hpp"



struct Channelizer {
    size_t nsteps;
    size_t nch_coarse;
    size_t nch_fine_per_coarse_full;
    std::unique_ptr<RawComplex[], Deleter<RawComplex>> raw_data;
    std::unique_ptr<RawComplex[], Deleter<RawComplex>> transposed_data;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> working_mem1;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> working_mem2;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> freq_shift_factor;  // with length nch_fine_per_coarse*2
    std::unique_ptr<FloatType[], Deleter<FloatType>> coeffs;
    std::unique_ptr<cuComplex[], Deleter<cuComplex>> buffer;
    size_t coeff_len{};
    cufftHandle fft_handle{0};

    Channelizer() = delete;
    Channelizer(const Channelizer &) = delete;
    Channelizer &operator=(const Channelizer &) = delete;

    virtual ~Channelizer();

    Channelizer(size_t nsteps, size_t nch_coarse1, size_t nch_fine_per_coarse_full1, const vector<FloatType> &coeffs1);
    void put_raw(const std::complex<RawDataType> *data);

    void transpose();

    void get_transposed(std::complex<RawDataType> *dst) const;

    void shift();

    std::vector<std::complex<FloatType>> get_working_mem(int n) const;

    void filter();

    void fft();

    void rearrange();

    void channelize(const std::vector<std::complex<RawDataType>>& data, std::vector<std::complex<FloatType>>& output);
    std::vector<std::complex<FloatType>> peek_channelized();


    //following functions are for debug purpose
    std::vector<std::complex<FloatType>> get_shifted()const;
    std::vector<std::complex<FloatType>> get_filtered()const;
    std::vector<std::complex<FloatType>> get_buffer()const;
};

#endif
