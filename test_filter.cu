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
#include "kernels.h"

int main() {
    auto nsteps = 1 << 20;
    auto nch = 1;
    auto nch_fine_per_coarse = 32;
    auto tap_per_ch = 8;

    std::vector<float> coeffs = pfb_coeff(nch_fine_per_coarse, tap_per_ch, 0.8);
    for (int i = 0; i < coeffs.size(); ++i) {
        coeffs[i] *= 100;
    }

    dump_data(coeffs, "coeffs.bin");

    std::vector<std::complex<float>> raw_data(nsteps * nch);
    auto dphi_dpt = 1.5 * 3.1415926 / 16.0;
    auto phi = 0.0;
    for (int i = 0; i < nsteps; ++i) {
        auto x = std::polar<float>(270, phi);

        for (int j = 0; j < nch; ++j) {
            raw_data[i * nch + j] = x;
        }
        phi += dphi_dpt;
    }

    auto d_input = gpu_alloc<cuComplex>(nsteps + coeffs.size());

    auto d_output = gpu_alloc<cuComplex>(nsteps + coeffs.size());
    auto d_coeff = gpu_alloc<float>(coeffs.size());

    cuda_mem_cpy(d_input.get(), raw_data.data(), raw_data.size(), cudaMemcpyHostToDevice);
    cuda_mem_cpy(d_coeff.get(), coeffs.data(), coeffs.size(), cudaMemcpyHostToDevice);

    pfb_conv_kernel<<<dim3(nch, 1024), nch_fine_per_coarse>>>(
        d_input.get(), d_output.get(), d_coeff.get(), nsteps, nch, nch_fine_per_coarse, coeffs.size());

    std::vector<std::complex<float>> filtered(nsteps*nch) ;
    cuda_mem_cpy(filtered.data(), d_output.get(), filtered.size(), cudaMemcpyDeviceToHost);

    for(int i=0;i<64;++i){
        std::cout<<i<<" "<<filtered[i].real()<<" "<<raw_data[i].real()<<std::endl;
    }

    dump_data(raw_data, "raw.bin");
    dump_data(filtered, "filtered.bin");
}
