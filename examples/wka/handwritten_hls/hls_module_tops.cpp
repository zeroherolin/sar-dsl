//===- hls_module_tops.cpp - Hand-written HLS module entry points --------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "corner_turn.h"
#include "fft_core.h"

void corner_turn_top(bus_t *__restrict in, bus_t *__restrict out) {
#pragma HLS INTERFACE m_axi port = in offset = slave depth =                   \
    MEM_WORDS bundle = gmem max_widen_bitwidth =                               \
        WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                     \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out offset = slave depth =                  \
    MEM_WORDS bundle = gmem max_widen_bitwidth =                               \
        WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                     \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  corner_turn(in, out, false);
}

void row_transform_top(bus_t *__restrict in, bus_t *__restrict out, int mode) {
#pragma HLS INTERFACE m_axi port = in offset = slave depth =                   \
    MEM_WORDS bundle = gmem max_widen_bitwidth =                               \
        WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                     \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out offset = slave depth =                  \
    MEM_WORDS bundle = gmem max_widen_bitwidth =                               \
        WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                     \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = in bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = mode bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
  row_transform_mode_t transform_mode = WKA_ROW_FORWARD;
  switch (mode) {
  case WKA_ROW_INVERSE:
    transform_mode = WKA_ROW_INVERSE;
    break;
  case WKA_ROW_WINDOW_INVERSE:
    transform_mode = WKA_ROW_WINDOW_INVERSE;
    break;
  case WKA_ROW_BULK_STOLT_WINDOW_INVERSE:
    transform_mode = WKA_ROW_BULK_STOLT_WINDOW_INVERSE;
    break;
  default:
    break;
  }
  run_row_transform(in, out, transform_mode);
}
