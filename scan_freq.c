#include "cspch.h"
#include "stdlib.h"
#include "stdio.h"
#include <string.h>
#include "math.h"
int main() {
    FILE* outfile;
    size_t nch = 88;
    size_t nsteps = 1 << 20;
    size_t tap_per_ch = 16;
    size_t nch_fine_per_coarse = 32;
    size_t coeff_len = nch_fine_per_coarse * tap_per_ch;
    float *coeff = (float *) malloc(tap_per_ch * nch_fine_per_coarse * sizeof(float));
    calc_coeff(nch_fine_per_coarse, tap_per_ch, 1.1, coeff);
    outfile=fopen("coeff_c.bin", "wb");
    fwrite((char*)coeff, sizeof(float), tap_per_ch*nch_fine_per_coarse, outfile);
    fclose(outfile);
    


    struct Channelizer *ptr = create_channelizer(nsteps, nch, nch_fine_per_coarse, coeff, coeff_len);
    free((char *) coeff);

    int16_t *input_data = malloc(2 * nch * nsteps * sizeof(int16_t));
    float *output_data = malloc(nch * nsteps * sizeof(float));
    float *spec = malloc(nch * nch_fine_per_coarse / 2 * sizeof(float));

    size_t nsteps_fine = nsteps / nch_fine_per_coarse;
    outfile = fopen("spec_c.bin", "wb");
    for (float f = -0.5; f < 0.5; f += 0.001) 
    {
        printf("%f\n", f);
        double omega = 2.0 * 3.1415926 * f;
        double phi = 0.0;
        for (int i = 0; i < nsteps; ++i) {
            int j = 0;
            input_data[i * 2*nch + 2*j] = 270*cos(phi);
            input_data[i * 2*nch + 2*j + 1] = 270*sin(phi);
            phi += omega;
        }

        
        channelize(ptr, input_data, output_data);
        
        
        memset(spec, 0, nch * nch_fine_per_coarse / 2 * sizeof(float));
        for (int i = 0; i < nch * nch_fine_per_coarse / 2; ++i) {
            for (int j = 0; j < nsteps_fine; ++j) {
                float real=output_data[(i * nsteps_fine + j)*2];
                float imag=output_data[(i * nsteps_fine + j)*2+1];
                spec[i] += (real*real+imag*imag);
            }
        }

        fwrite((void *) spec, sizeof(float), nch*nch_fine_per_coarse/2, outfile);
    }

    fclose(outfile);

    free((char *) input_data);
    free((char *) output_data);
    destroy_channelizer(ptr);
}
