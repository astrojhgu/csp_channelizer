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

#include "channelizer.hpp"

int main() {
    auto nsteps = 1 << 20;
    auto nch = 88;
    auto nch_fine_per_coarse = 32;
    auto tap_per_ch=16;
    std::vector<float> coeffs = pfb_coeff(nch_fine_per_coarse, tap_per_ch, 1.1);
    dump_data(coeffs, "coeff.bin");
    
    std::cout << "initialization finished" << std::endl;
    Channelizer channelizer(nsteps, nch, nch_fine_per_coarse, coeffs);
    // exit(0);
    std::cout << (gpu_mem_used / 1024.0 / 1024 / 1024) << " GB" << std::endl;

    std::vector<std::complex<int16_t>> raw_data(nsteps * nch);
    std::vector<std::complex<float>> channelized(nch * nsteps / 2);
    ofstream ofs("spec.bin", std::ios::binary);

    //for (float f = -0.5; f < 0.5; f += 0.001) 
    double f=0.1;
    {
        std::cout << f << std::endl;
        auto omega = 2.0 * 3.1415926 * f;
        auto phi = 0.0;
#pragma omp parallel for
        for (int i = 0; i < nsteps; ++i) {
            int j = 0;
            auto x = std::polar<double>(270, phi);
            raw_data[i * nch + j] = std::complex<int16_t>(x.real(), x.imag());
            phi += omega;
        }
        dump_data(raw_data, "raw.bin");

        channelizer.channelize(raw_data, channelized);
        dump_data(channelized, "out.bin");
        std::vector<float> spec(nch * nch_fine_per_coarse / 2);
        auto nsteps_fine = nsteps / nch_fine_per_coarse;
        for (int i = 0; i < nch * nch_fine_per_coarse / 2; ++i) {
            for (int j = 0; j < nsteps_fine; ++j) {
                spec[i] += std::norm(channelized[i * nsteps_fine + j]);
            }
        }
        ofs.write((char *) spec.data(), spec.size() * sizeof(float));
    }
}
