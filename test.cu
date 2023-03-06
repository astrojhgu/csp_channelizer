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

#include "transpose.hpp"
#include "channelizer.hpp"

int main() {
    auto nsteps = 1 << 20;
    auto nch = 88;
    auto nch_fine_per_coarse = 32;
    auto tap_per_ch = 8;

    std::vector<float> coeffs = pfb_coeff(nch_fine_per_coarse, tap_per_ch, 0.8);
    for (int i = 0; i < coeffs.size(); ++i) {
        coeffs[i] *= 100;
    }

    std::vector<std::complex<int16_t>> raw_data(nsteps * nch);
    auto dphi_dpt = 2 * 3.1415926 / 16.0;
    auto phi = 0.0;
    for (int i = 0; i < nsteps; ++i) {
        auto x = std::polar<float>(270, phi);
        std::complex<int16_t> x1(x.real(), x.imag());
        for (int j = 0; j < nch; ++j) {
            raw_data[i * nch + j] = x1;
        }
        phi += dphi_dpt;
    }

    std::cout << "initialization finished" << std::endl;
    Channelizer channelizer(nsteps, nch, nch_fine_per_coarse, coeffs);

    // exit(0);
    std::cout << (gpu_mem_used / 1024.0 / 1024 / 1024) << " GB" << std::endl;

    ofstream ofs;
    for (int i = 0; i < 2; ++i) {
        raw_data[i] = raw_data[0];
        channelizer.put_raw(raw_data.data());
        channelizer.transpose();

        channelizer.shift();
        auto shifted=channelizer.get_shifted();
        dump_data(shifted, "shifted.bin");
        

        channelizer.filter();
        auto filtered=channelizer.get_filtered();
        dump_data(filtered, "filtered.bin");

        channelizer.fft();

        channelizer.rearrange();

        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cout << i << std::endl;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
