//===- wka_top.cpp - Hand-written omega-K HLS top ------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "wka_top.h"
#include "corner_turn.h"
#include "fft_core.h"

#ifndef __SYNTHESIS__
#include <iostream>
#define WKA_SIM_LOG(message) std::cout << "\n[FPGA-SIM] " << message << std::endl
#else
#define WKA_SIM_LOG(message)
#endif

void wka_sar_top(bus_t *__restrict ddr_in, bus_t *__restrict ddr_out, bus_t *__restrict ddr_tmp) {
#pragma HLS INTERFACE m_axi port = ddr_in offset = slave depth = MEM_WORDS bundle =                \
    gmem max_widen_bitwidth = WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                   \
        WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =                                     \
            WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =                                  \
                WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding = WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = ddr_out offset = slave depth = MEM_WORDS bundle =               \
    gmem max_widen_bitwidth = WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                   \
        WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =                                     \
            WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =                                  \
                WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding = WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE m_axi port = ddr_tmp offset = slave depth = MEM_WORDS bundle =               \
    gmem max_widen_bitwidth = WKA_AXI_MAX_WIDEN_BITWIDTH max_read_burst_length =                   \
        WKA_AXI_MAX_READ_BURST_LENGTH max_write_burst_length =                                     \
            WKA_AXI_MAX_WRITE_BURST_LENGTH num_read_outstanding =                                  \
                WKA_AXI_NUM_READ_OUTSTANDING num_write_outstanding = WKA_AXI_NUM_WRITE_OUTSTANDING
#pragma HLS INTERFACE s_axilite port = ddr_in bundle = control
#pragma HLS INTERFACE s_axilite port = ddr_out bundle = control
#pragma HLS INTERFACE s_axilite port = ddr_tmp bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
#pragma HLS ALLOCATION function instances = run_row_transform limit = 1
#pragma HLS ALLOCATION function instances = corner_turn limit = 1

    WKA_SIM_LOG("1/8: Range FFT");
    run_row_transform(ddr_in, ddr_tmp, WKA_ROW_FORWARD);

    WKA_SIM_LOG("2/8: Corner turn");
    corner_turn(ddr_tmp, ddr_out, false);

    WKA_SIM_LOG("3/8: Azimuth FFT");
    run_row_transform(ddr_out, ddr_tmp, WKA_ROW_FORWARD);

    WKA_SIM_LOG("4/8: Corner turn");
    corner_turn(ddr_tmp, ddr_out, false);

    WKA_SIM_LOG("5/8: Bulk, Stolt, range window and IFFT");
    run_row_transform(ddr_out, ddr_tmp, WKA_ROW_BULK_STOLT_WINDOW_INVERSE);

    WKA_SIM_LOG("6/8: Corner turn");
    corner_turn(ddr_tmp, ddr_out, false);

    WKA_SIM_LOG("7/8: Azimuth window and IFFT");
    run_row_transform(ddr_out, ddr_tmp, WKA_ROW_WINDOW_INVERSE);

    WKA_SIM_LOG("8/8: Final corner turn");
    corner_turn(ddr_tmp, ddr_out, true);
}

#undef WKA_SIM_LOG
