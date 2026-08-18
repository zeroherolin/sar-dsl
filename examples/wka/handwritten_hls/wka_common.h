//===- wka_common.h - Hand-written omega-K HLS types -----------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_COMMON_H
#define WKA_COMMON_H

#include "config.h"
#include <cmath>
#include <cstddef>
#include <cstdint>

using real_t = float;
using acc_t = float;
using io_t = float;
using bus_t = ap_uint<WKA_AXI_BUS_BITS>;

static_assert(sizeof(io_t) * 8 == WKA_IO_SCALAR_BITS,
              "External DDR packing assumes fixed-width IO scalar data.");
static_assert(WKA_COMPLEX_SAMPLE_BITS == 64, "Memory packing requires 64-bit complex samples.");
static_assert((WKA_AXI_BUS_BITS % WKA_COMPLEX_SAMPLE_BITS) == 0,
              "AXI bus width must be an integer multiple of the packed sample width.");

struct data_t {
    float r;
    float i;
    data_t() = default;
    data_t(float real, float imag) : r(real), i(imag) {}
    float real() const { return r; }
    float imag() const { return i; }
};

static inline real_t wka_cos_real(real_t x) {
#pragma HLS INLINE
    return std::cos(x);
}

static inline real_t wka_sin_real(real_t x) {
#pragma HLS INLINE
    return std::sin(x);
}

static inline real_t wka_sqrt_real(real_t x) {
#pragma HLS INLINE
    return std::sqrt(x);
}

static inline int wka_floor_to_int(real_t x) {
#pragma HLS INLINE
    return static_cast<int>(std::floor(x));
}

inline uint64_t pack_data(const data_t &value) {
    union {
        io_t f[2];
        uint64_t u;
    } u;
    u.f[0] = value.r;
    u.f[1] = value.i;
    return u.u;
}

inline data_t unpack_data(uint64_t packed) {
    union {
        uint64_t u;
        io_t f[2];
    } u;
    u.u = packed;
    return data_t(u.f[0], u.f[1]);
}

inline uint32_t pack_real(float value) {
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = value;
    return bits.u;
}

inline float unpack_real(uint32_t packed) {
    union {
        uint32_t u;
        float f;
    } bits;
    bits.u = packed;
    return bits.f;
}

static const int N = WKA_N;
static const int LOG2_N = WKA_LOG2_N;
static const int TILE_SIZE = WKA_TILE_SIZE;
static const int MEM_DEPTH = WKA_MEM_DEPTH;
static const int BUS_BITS = WKA_AXI_BUS_BITS;
static const int BUS_LANES = BUS_BITS / WKA_COMPLEX_SAMPLE_BITS;
static const int REAL_BUS_LANES = BUS_BITS / WKA_IO_SCALAR_BITS;
static const int MEM_WORDS = MEM_DEPTH / BUS_LANES;

static_assert((N & (N - 1)) == 0, "WKA_N must be a power of two.");
static_assert((LOG2_N % 2) == 0, "Radix-4 FFT requires an even LOG2_N.");
static_assert((N % BUS_LANES) == 0, "WKA_N must align to one AXI word.");
static_assert((N % TILE_SIZE) == 0, "WKA_N must be divisible by TILE_SIZE.");
static_assert((TILE_SIZE % BUS_LANES) == 0, "TILE_SIZE must align to one AXI word.");
static_assert((N % WKA_FFT_PAR_ROWS) == 0, "WKA_N must be divisible by WKA_FFT_PAR_ROWS.");
static_assert((N % WKA_STOLT_OUT_LANES) == 0, "WKA_N must be divisible by WKA_STOLT_OUT_LANES.");
static_assert((BUS_LANES % WKA_STOLT_OUT_LANES) == 0,
              "Stolt lanes must divide the number of AXI lanes.");

static const real_t PI = WKA_PI;
static const real_t C0 = WKA_C0;
static const real_t FC = WKA_FC;
static const real_t FS = WKA_FS;
static const real_t PRF = WKA_PRF;
static const real_t VR = WKA_VR;
static const real_t R0 = WKA_R0;
static const real_t KR = WKA_KR;
static const real_t STOLT_TIME_SHIFT = WKA_STOLT_TIME_SHIFT;
static const int VALID_COLS = WKA_VALID_COLS;

#endif
