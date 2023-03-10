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

    dump_data(coeffs, "coeffs.bin");

    std::vector<std::complex<int16_t>> raw_data(nsteps * nch);
    auto dphi_dpt = 1.5 * 3.1415926 / 16.0;
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
        for (int j = 0; j < nsteps; ++j) {
            auto x = std::polar<float>(270, phi);
            std::complex<int16_t> x1(x.real(), x.imag());
            for (int k = 0; k < nch; ++k) {
                raw_data[j * nch + k] = x1;
            }
            phi += dphi_dpt;
        }
        // raw_data[i] = raw_data[0];
        channelizer.put_raw(raw_data.data());
        channelizer.transpose();

        channelizer.shift();
        auto shifted = channelizer.get_shifted();
        dump_data(shifted, "shifted.bin");

        auto buffer = channelizer.get_buffer();
        dump_data(buffer, "buffer.bin");

        auto data = channelizer.get_working_mem(1);
        dump_data(data, "input.bin");
        channelizer.filter();
        data = channelizer.get_working_mem(2);
        dump_data(data, "output.bin");

        channelizer.fft();

        channelizer.rearrange();

        // CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cout << i << std::endl;
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
