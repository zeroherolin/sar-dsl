//===- wka_top.cpp - Hand-written omega-K HLS top -------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "wka_top.h"
#include "corner_turn.h"
#include "fft_core.h"

#ifndef __SYNTHESIS__
#include <iostream>
#define WKA_SIM_LOG(message)                                                   \
  std::cout << "\n[FPGA-SIM] " << message << std::endl
#else
#define WKA_SIM_LOG(message)
#endif

void wka_sar_top(plane_t raw_re[N][PLANE_WORDS_PER_ROW],
                 plane_t raw_im[N][PLANE_WORDS_PER_ROW], real_t win_r[N],
                 real_t win_a[N], plane_t out0[N][PLANE_WORDS_PER_ROW],
                 plane_t scratch0[PLANE_WORDS], plane_t scratch1[PLANE_WORDS],
                 plane_t scratch2[PLANE_WORDS], plane_t scratch3[PLANE_WORDS]) {
#pragma HLS INTERFACE m_axi port = raw_re offset = slave depth =               \
    PLANE_WORDS bundle = axi_0 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = raw_im offset = slave depth =               \
    PLANE_WORDS bundle = axi_1 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = win_r offset = slave depth = N bundle =     \
    axi_2 max_widen_bitwidth =                                                 \
        WKA_AXI_SCALAR_MAX_WIDEN_BITWIDTH max_read_burst_length =              \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = win_a offset = slave depth = N bundle =     \
    axi_2 max_widen_bitwidth =                                                 \
        WKA_AXI_SCALAR_MAX_WIDEN_BITWIDTH max_read_burst_length =              \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out0 offset = slave depth =                 \
    PLANE_WORDS bundle = axi_3 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = scratch0 offset = slave depth =             \
    PLANE_WORDS bundle = axi_4 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = scratch1 offset = slave depth =             \
    PLANE_WORDS bundle = axi_5 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = scratch2 offset = slave depth =             \
    PLANE_WORDS bundle = axi_6 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = scratch3 offset = slave depth =             \
    PLANE_WORDS bundle = axi_7 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = raw_re bundle = ctrl
#pragma HLS INTERFACE s_axilite port = raw_im bundle = ctrl
#pragma HLS INTERFACE s_axilite port = win_r bundle = ctrl
#pragma HLS INTERFACE s_axilite port = win_a bundle = ctrl
#pragma HLS INTERFACE s_axilite port = out0 bundle = ctrl
#pragma HLS INTERFACE s_axilite port = scratch0 bundle = ctrl
#pragma HLS INTERFACE s_axilite port = scratch1 bundle = ctrl
#pragma HLS INTERFACE s_axilite port = scratch2 bundle = ctrl
#pragma HLS INTERFACE s_axilite port = scratch3 bundle = ctrl
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl
#pragma HLS ALLOCATION function instances = run_row_forward limit = 1
#pragma HLS ALLOCATION function instances = corner_turn_complex limit = 1

  WKA_SIM_LOG("1/8: Range FFT");
  run_row_forward(raw_re[0], raw_im[0], scratch0, scratch1);

  WKA_SIM_LOG("2/8: Corner turn");
  corner_turn_complex(scratch0, scratch1, scratch2, scratch3);

  WKA_SIM_LOG("3/8: Azimuth FFT");
  run_row_forward(scratch2, scratch3, scratch0, scratch1);

  WKA_SIM_LOG("4/8: Corner turn");
  corner_turn_complex(scratch0, scratch1, scratch2, scratch3);

  WKA_SIM_LOG("5/8: Bulk, Stolt, range window and IFFT");
  run_row_bulk_stolt_window_inverse(scratch2, scratch3, win_r, scratch0,
                                    scratch1);

  WKA_SIM_LOG("6/8: Corner turn");
  corner_turn_complex(scratch0, scratch1, scratch2, scratch3);

  WKA_SIM_LOG("7/8: Azimuth window and IFFT");
  run_row_window_inverse(scratch2, scratch3, win_a, scratch0, scratch1);

  WKA_SIM_LOG("8/8: Magnitude corner turn");
  corner_turn_magnitude(scratch0, scratch1, out0[0]);
}

#undef WKA_SIM_LOG
