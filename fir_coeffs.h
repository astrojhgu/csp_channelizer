#ifndef FIR_COEFF_H
#define FIR_COEFF_H
#include <vector>
#include "types.hpp"


std::vector<FloatType> to_time_domain(const std::vector<FloatType> &input);
void symmetrize(std::vector<FloatType> &workpiece);
std::vector<FloatType> lp_coeff(std::size_t tap, FloatType k);
void apply_blackman_window(std::vector<FloatType> &workpiece);
std::vector<FloatType> pfb_coeff(std::size_t nch, std::size_t tap_per_ch, FloatType k);

#endif