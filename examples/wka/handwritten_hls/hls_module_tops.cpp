//===- hls_module_tops.cpp - Hand-written HLS module entry points ---------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "corner_turn.h"
#include "fft_core.h"

void corner_turn_top(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im) {
#pragma HLS INTERFACE m_axi port = in_re offset = slave depth =                \
    PLANE_WORDS bundle = axi_0 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = in_im offset = slave depth =                \
    PLANE_WORDS bundle = axi_1 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out_re offset = slave depth =               \
    PLANE_WORDS bundle = axi_2 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out_im offset = slave depth =               \
    PLANE_WORDS bundle = axi_3 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = in_re bundle = ctrl
#pragma HLS INTERFACE s_axilite port = in_im bundle = ctrl
#pragma HLS INTERFACE s_axilite port = out_re bundle = ctrl
#pragma HLS INTERFACE s_axilite port = out_im bundle = ctrl
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl
  corner_turn_complex(in_re, in_im, out_re, out_im);
}

void row_forward_top(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im) {
#pragma HLS INTERFACE m_axi port = in_re offset = slave depth =                \
    PLANE_WORDS bundle = axi_0 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = in_im offset = slave depth =                \
    PLANE_WORDS bundle = axi_1 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out_re offset = slave depth =               \
    PLANE_WORDS bundle = axi_2 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = out_im offset = slave depth =               \
    PLANE_WORDS bundle = axi_3 max_widen_bitwidth =                            \
        WKA_AXI_PLANE_MAX_WIDEN_BITWIDTH max_read_burst_length =               \
            WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =             \
                WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =          \
                    WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding =       \
                        WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = in_re bundle = ctrl
#pragma HLS INTERFACE s_axilite port = in_im bundle = ctrl
#pragma HLS INTERFACE s_axilite port = out_re bundle = ctrl
#pragma HLS INTERFACE s_axilite port = out_im bundle = ctrl
#pragma HLS INTERFACE s_axilite port = return bundle = ctrl
  run_row_forward(in_re, in_im, out_re, out_im);
}
