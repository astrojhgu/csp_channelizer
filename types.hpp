#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>


using RawDataType = int16_t;
using FloatType = float;


struct RawComplex {
    RawDataType real;
    RawDataType imag;
};

#endif
