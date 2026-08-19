//===- wka_top.h - Hand-written omega-K HLS top API ------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_TOP_H
#define WKA_TOP_H

#include "wka_common.h"

// Buffers hold N x N complex FP32 samples.
// Each buffer is 64-byte aligned and occupies a distinct address range.
void wka_sar_top(bus_t *__restrict ddr_in, bus_t *__restrict ddr_out,
                 bus_t *__restrict ddr_tmp);

#endif
