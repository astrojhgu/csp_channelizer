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
#include "correlate.h"
using namespace std;

#include "transpose.hpp"
#include "channelizer.hpp"

void __global__ func(cuComplex* x, cuComplex* y, cuComplex* z){
    *x=make_float2(5e6, 5e6);
    *y=make_float2(5e6, -5e6);
    *z=cuCmulf(*x, cuConjf(*y));
    //printf("%e %e\n", x->x, x->y );
    //printf("%e %e\n", y->x, y->y );
    printf("%e\n", x->x*y->x+x->y*y->y);
    printf("%e %e\n", z->x, z->y);
}


int main() {
    int nch_coarse = 88;
    int nch_fine_per_coarse = 16;
    int nch_total = nch_coarse * nch_fine_per_coarse;
    int nsteps = (1 << 20) / nch_fine_per_coarse / 2;

    std::vector<std::complex<float>> input1_h(nsteps * nch_total, {1, 0});
    std::vector<std::complex<float>> input2_h(nsteps * nch_total, {1, 0});
    std::vector<std::complex<float>> output_h(nsteps * nch_total, {0.0, 0.0});
    std::vector<std::complex<float>> answer_h(nsteps * nch_total, {0.0, 0.0});
    for (int i = 0; i < nsteps * nch_total; ++i) {
        input1_h[i] = {float(i), float(i)};
        input2_h[i] = {float(i), -float(i)};
        answer_h[i] = input1_h[i] * std::conj(input2_h[i]);
    }

    std::vector<std::complex<float>> answer(nch_total, {0.0, 0.0});
    for (int ich = 0; ich < nch_total; ++ich) {
        answer[ich] = {0.0, 0.0};
        for (int istep = 0; istep < nsteps; ++istep) {
            answer[ich] += answer_h[ich * nsteps + istep];
        }
        // std::cout << answer[ich] << " ";
    }

    auto input1 = gpu_alloc<cuComplex>(input1_h.size());
    auto input2 = gpu_alloc<cuComplex>(input2_h.size());

    cuda_mem_cpy(input1.get(), input1_h.data(), input1_h.size(), cudaMemcpyHostToDevice);
    cuda_mem_cpy(input2.get(), input2_h.data(), input2_h.size(), cudaMemcpyHostToDevice);

    Correlator corr(nch_total, nsteps);
    std::vector<complex<float>> x(nch_total*nsteps);
    corr.correlate(input1.get(), input2.get(), x.data());
    // for (int i = 0; i < 1000; ++i) {
    //auto x = correlate(input1.get(), input2.get(), buffer.get(), nch_total, nsteps);
    for (int i = 0; i < nch_total; ++i) {
        //std::cout <<  << std::endl;
        std::cout<<answer[i]<<" "<<x[i]<<" "<<std::abs(x[i] - answer[i]) / std::abs(x[i] + answer[i]) * 2.0f<<std::endl;
    }
    //}
}
