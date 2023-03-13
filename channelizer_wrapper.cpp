#include "channelizer.hpp"
#include "fir_coeffs.h"
#include <cstring>

using namespace std;

extern "C" Channelizer *create_channelizer(
    size_t nsteps, size_t nch_coarse1, size_t nch_fine_per_coarse_full1, const float *coeff, size_t coeff_len) {
    std::vector<float> coeff1(coeff_len);
    memcpy((char *) coeff1.data(), (char *) coeff, coeff_len * sizeof(float));
    return new Channelizer(nsteps, nch_coarse1, nch_fine_per_coarse_full1, coeff1);
}

extern "C" void destroy_channelizer(Channelizer *ptr) {
    delete ptr;
}

extern "C" void channelize(Channelizer *ptr,
                const int16_t *input /*1-d array with (nch_coarse*2, nsteps) elements*/,
                float *output /*1-d array with (nch_coarse, nsteps) elements*/) {
    ptr->channelize((const std::complex<int16_t> *) input, (std::complex<float> *) output);
}

extern "C" void calc_coeff(std::size_t nch, std::size_t tap_per_ch, FloatType k, float* coeff){
    auto coeff1=pfb_coeff(nch, tap_per_ch, k);
    std::cout<<coeff1.size()<<" "<<nch<<" "<<tap_per_ch<<std::endl;
    memcpy((char*) coeff, coeff1.data(), coeff1.size()*sizeof(float));
}
