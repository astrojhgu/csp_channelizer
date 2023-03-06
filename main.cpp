#include "utils.hpp"
#include "cuComplex.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>
#include "fir_coeffs.h"
using namespace std;

//#include "transpose.hpp"
#include "channelizer.hpp"

int main() {
    auto nsteps = 1 << 20;
    auto nch = 88;
    auto nch_fine_per_coarse=32;

    std::vector<std::complex<int16_t>> raw_data(nsteps * nch);
    auto dphi_dpt = 2.1 * 3.1415926 / 16.0;
    auto phi = 0.0;
    for (int i = 0; i < nsteps; ++i) {
        int j = 0;
        auto x = std::polar<float>(270, phi);
        raw_data[i * nch + j] = std::complex<int16_t>(x.real(), x.imag());
        phi += dphi_dpt;
    }

    std::vector<float> coeffs=pfb_coeff(nch_fine_per_coarse, 8, 0.8);
    for(int i=0;i<coeffs.size();++i){
        coeffs[i]*=100;
    }

    std::cout << "initialization finished" << std::endl;
    Channelizer channelizer(nsteps, nch, nch_fine_per_coarse, coeffs);
    Channelizer channelizer1(nsteps, nch, nch_fine_per_coarse, coeffs);
    // exit(0);
    std::cout << (gpu_mem_used / 1024.0 / 1024 / 1024) << " GB" << std::endl;
    
    ofstream ofs;
    for (int i = 0; i < 100; ++i) {
        raw_data[i]=raw_data[0];
        channelizer.put_raw(raw_data.data());
        channelizer1.put_raw(raw_data.data());
        channelizer.transpose();
        channelizer1.transpose();
        channelizer.shift();
        channelizer1.shift();
        channelizer.filter();
        channelizer1.filter();
        channelizer.fft();
        channelizer1.fft();
        channelizer.rearrange();
        channelizer1.rearrange();
        //CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cout << i << std::endl;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::vector<std::complex<int16_t>> raw_data1(nsteps * nch);
    std::vector<std::complex<int16_t>> transposed(nch*nsteps);
    
    channelizer.get_transposed(raw_data1.data());
    std::cout << "validating" << std::endl;
    for (int i = 0; i < nsteps; ++i) {
        for (int j = 0; j < nch; ++j) {
            assert(raw_data[i * nch + j] == raw_data1[j * nsteps + i]);
        }
    }

    ofs.open("raw.bin", std::ios::binary);
    ofs.write((char*)raw_data.data(), raw_data.size()*sizeof(complex<int16_t>));
    ofs.close();

    cout<<(transposed.size()*sizeof(complex<int16_t>))<<" "<<(raw_data1.size()*sizeof(complex<int16_t>))<<endl;
    //std::vector<std::complex<float>> fft_result(nsteps * nch);

    auto fft_result=channelizer.get_working_mem(1);

    auto channelized = channelizer.peek_channelized();
    ofs.open("result.bin", std::ios::binary);
    ofs.write((const char *) channelized.data(), channelized.size() * sizeof(std::complex<float>));
    ofs.close();
}