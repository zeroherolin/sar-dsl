//===- config.h - Hand-written omega-K HLS configuration -------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_CONFIG_H
#define WKA_CONFIG_H

#include <ap_int.h>

#define WKA_SIZE_SMOKE 0
#define WKA_SIZE_MEDIUM 1
#define WKA_SIZE_PRODUCTION 2

#ifndef WKA_SIZE_PROFILE
#define WKA_SIZE_PROFILE WKA_SIZE_PRODUCTION
#endif

// AXI and external sample layout
#define WKA_AXI_BUS_BITS 512
#define WKA_AXI_MAX_WIDEN_BITWIDTH 512
#define WKA_AXI_MAX_READ_BURST_LENGTH 256
#define WKA_AXI_MAX_WRITE_BURST_LENGTH 256
#define WKA_AXI_NUM_READ_OUTSTANDING 8
#define WKA_AXI_NUM_WRITE_OUTSTANDING 8
#define WKA_COMPLEX_SAMPLE_BITS 64
#define WKA_IO_SCALAR_BITS 32

// Problem size and validation profiles
#if WKA_SIZE_PROFILE == WKA_SIZE_SMOKE
#define WKA_N 64
#define WKA_LOG2_N 6
#define WKA_TILE_SIZE 16
#elif WKA_SIZE_PROFILE == WKA_SIZE_MEDIUM
#define WKA_N 256
#define WKA_LOG2_N 8
#define WKA_TILE_SIZE 32
#elif WKA_SIZE_PROFILE == WKA_SIZE_PRODUCTION
#define WKA_N 16384
#define WKA_LOG2_N 14
#define WKA_TILE_SIZE 256
#else
#error Unsupported WKA_SIZE_PROFILE
#endif
#define WKA_MEM_DEPTH (WKA_N * WKA_N)

// Radar parameters
#define WKA_PI 3.14159265358979323846f
#define WKA_C0 299792458.0f
#define WKA_FC 1269999750.06f
#define WKA_FS 32000000.0f
#define WKA_PRF 2155.172f
#define WKA_VR 7072.0f
#define WKA_R0 843013.994f
#define WKA_KR (-1.037e12f)
#if WKA_SIZE_PROFILE == WKA_SIZE_PRODUCTION
#define WKA_STOLT_TIME_SHIFT_SAMPLES 4800
#else
#define WKA_STOLT_TIME_SHIFT_SAMPLES (WKA_N / 2)
#endif
#define WKA_STOLT_TIME_SHIFT (WKA_STOLT_TIME_SHIFT_SAMPLES / WKA_FS)
#define WKA_VALID_COLS 9600
#define WKA_DISPLAY_INC_ANGLE_DEG 38.0f

// Row transform
#define WKA_ROW_LOAD_STORE_II 1
#if WKA_SIZE_PROFILE == WKA_SIZE_SMOKE
#define WKA_FFT_PAR_ROWS 4
#define WKA_FFT_BLOCK_PART_FACTOR 4
#define WKA_FFT_BUF_STORAGE_IMPL bram
#elif WKA_SIZE_PROFILE == WKA_SIZE_MEDIUM
#define WKA_FFT_PAR_ROWS 8
#define WKA_FFT_BLOCK_PART_FACTOR 8
#define WKA_FFT_BUF_STORAGE_IMPL bram
#else
#define WKA_FFT_PAR_ROWS 16
#define WKA_FFT_BLOCK_PART_FACTOR 8
#define WKA_FFT_BUF_STORAGE_IMPL uram
#endif
#define WKA_FFT_BUTTERFLY_II 4
#define WKA_FFT_BITREV_II 1

// Corner turn
#define WKA_CORNER_LOAD_STORE_II 1

// Bulk compression and Stolt interpolation
#define WKA_BULK_SAFE_SQRT_EPS 1e-10f
#define WKA_STOLT_WEIGHT_LUT_SIZE 1024
#define WKA_STOLT_OUT_LANES 2
#define WKA_STOLT_READ_WRITE_II 1
#define WKA_STOLT_INTERP_II 1
#define WKA_STOLT_READ_FMUL_LIMIT 32
#define WKA_STOLT_INTERP_FMUL_LIMIT 48
#if WKA_SIZE_PROFILE == WKA_SIZE_SMOKE
#define WKA_STOLT_ROW_PART_FACTOR 4
#define WKA_STOLT_ROW_STORAGE_IMPL bram
#define WKA_STOLT_PACKED_STORAGE_IMPL bram
#else
#define WKA_STOLT_ROW_PART_FACTOR 16
#define WKA_STOLT_ROW_STORAGE_IMPL uram
#define WKA_STOLT_PACKED_STORAGE_IMPL uram
#endif
#define WKA_STOLT_WEIGHT_TAP_START (-3)
#define WKA_STOLT_WEIGHT_TAP_END 4
#define WKA_STOLT_WEIGHT_TAP_COUNT (WKA_STOLT_WEIGHT_TAP_END - WKA_STOLT_WEIGHT_TAP_START + 1)

// Testbench defaults
#define WKA_TB_NONZERO_MAG_EPS 1e-6f
#define WKA_TB_INPUT_PATH "data/alos_raw_16384x16384.bin"
#define WKA_TB_OUTPUT_PATH "sar_wka_output.bmp"

#endif
