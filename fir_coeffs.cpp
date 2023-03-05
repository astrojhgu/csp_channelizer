#include "fir_coeffs.h"
#include "fftw3.h"
#include <vector>
#include <complex>
#include "types.hpp"
#include "types.hpp"
#include <iostream>
#include <cmath>

using namespace std;

static constexpr FloatType PI = 3.14159265358979323846f;
vector<FloatType> to_time_domain(const vector<FloatType> &input) {
    vector<std::complex<FloatType>> cinput;
    for (auto x : input) {
        cinput.push_back(std::complex<FloatType>{x, 0.0});
    }
    vector<std::complex<FloatType>> coutput(input.size());
    fftwf_plan p = fftwf_plan_dft_1d(
        input.size(), (fftwf_complex *) cinput.data(), (fftwf_complex *) coutput.data(), FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);
    vector<FloatType> result(input.size());
    FloatType s = 0;

    for(int i=0;i<input.size();++i){
        auto j=(i+input.size()/2)%input.size();
        result[j]=coutput[i].real();
        s+=coutput[i].real();
    }
    for (auto &x : result) {
        x /= s;
    }

    return result;
}

void symmetrize(vector<FloatType>& workpiece){
    auto n1=workpiece.size();
    for(int n=0;n<=(n1/2-2);++n){
        workpiece[n1-n-1]=workpiece[n+1];
    }
}

std::vector<FloatType> lp_coeff(size_t tap, FloatType k){
    std::vector<FloatType> result(tap);
    for(int i=0;i<tap;++i){
        FloatType x = i<tap/2? i: tap-i;
        FloatType y=k*(tap/2.0);
        FloatType r=0;
        if (x>=y){
            r=0;
        }else if (x<std::floor(y)){
            r=1;
        }else{
            r=y-std::floor(y);
        }
        //std::cout<<i<<" "<<r<<std::endl;
        result[i]=r;

    }
    return result;
}

FloatType blackman_window(size_t i, size_t n){
    FloatType a0 = 0.3635819;
    FloatType a1 = 0.4891775;
    FloatType a2 = 0.1365995;
    FloatType a3 = 0.0106411;
    FloatType x = i / (FloatType)n *PI;
    return a0 - a1 * cos(2.0 * x) + a2 * cos(4 * x) - a3 * cos(6 * x);
}

void apply_blackman_window(std::vector<FloatType>& workpiece){
    auto n=workpiece.size();
    for(int i=0;i<n;++i){
        workpiece[i]*=blackman_window(i, n);
    }
}

std::vector<FloatType> pfb_coeff(size_t nch, size_t tap_per_ch, FloatType k){
    auto a=lp_coeff(nch * tap_per_ch, k/nch);
    
    symmetrize(a);
    auto b=to_time_domain(a);
    apply_blackman_window(b);
    return b;
}
