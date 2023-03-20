#ifndef CSPCH_H
#define CSPCH_H
#include <stddef.h>
#include <stdint.h>

struct Channelizer;

#ifdef __cplusplus
extern "C" {
#endif

struct Channelizer *create_channelizer(
    size_t nsteps, size_t nch_coarse1, size_t nch_fine_per_coarse_full1, const float *coeff, size_t coeff_len);

void destroy_channelizer(struct Channelizer *ptr);

void channelize(struct Channelizer *ptr,
                const int16_t *input /*1-d array with (nch_coarse*2, nsteps) elements*/,
                float *output /*1-d array with (nch_coarse, nsteps) elements*/);

void channelize_no_out(struct Channelizer *ptr, const int16_t *input /*1-d array with (nch_coarse*2, nsteps) elements*/);

void calc_coeff(size_t nch, size_t tap_per_ch, float k, float *coeff);

struct Correlator *create_correlator(size_t nch, size_t nsteps);

void correlate(struct Correlator *ptr, const struct Channelizer *ch1,const struct Channelizer *ch2, float *result);

#ifdef __cplusplus
}
#endif

#endif
