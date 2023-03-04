#include "utils.hpp"
#include "cuComplex.h"
#include <cassert>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <thread>

using namespace std;

#include "transpose.hpp"
#include "channelizer.hpp"

int main() {
    auto nsteps = 1<<20;
    auto nch = 88;

    std::vector<std::complex<int16_t>> raw_data(nsteps * nch);
    auto dphi_dpt=2.0*3.1415926/32.0;
    auto phi=0.0;
    for(int i=0;i<nsteps;++i){
        int j=0;
        auto x=std::polar<int16_t>(200.0, phi);
        raw_data[i*nch+j]=x;
        phi+=dphi_dpt;
    }


    std::cout << "initialization finished" << std::endl;
    Channelizer channelizer(nsteps, nch, 64);
    Channelizer channelizer1(nsteps, nch, 64);
    //exit(0);
    std::cout << (gpu_mem_used / 1024.0 / 1024 / 1024) << " GB" << std::endl;
    

    for(int i=0;i<100;++i){
        channelizer.put_raw(raw_data.data());
        channelizer.transpose();
        channelizer.shift();
        //channelizer.shift();
        channelizer.fft();
        channelizer.rearrange();
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cout<<i<<std::endl;
    }

    std::vector<std::complex<int16_t>> raw_data1(nsteps * nch);
    channelizer.get_transposed(raw_data1.data());
    std::cout<<"validating"<<std::endl;
    for (int i = 0; i < nsteps; ++i) {
        for (int j = 0; j < nch; ++j) {
            assert(raw_data[i * nch + j] == raw_data1[j * nsteps + i]);
        }
    }
    std::vector<std::complex<float>> fft_result(nsteps * nch);

    channelizer.get_working_mem(fft_result.data(), 1);

    
    ofstream ofs("a.bin", std::ios::binary);
    ofs.write((const char*)fft_result.data(), fft_result.size()*sizeof(std::complex<float>));

    auto channelized=channelizer.peek_channelized();
    ofstream ofs1("result.bin", std::ios::binary);
    ofs1.write((const char*)channelized.data(), channelized.size()*sizeof(std::complex<float>));

}