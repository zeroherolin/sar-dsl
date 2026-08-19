//===- fft_core.h - Hand-written radix-4 HLS engine API --------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef FFT_CORE_H
#define FFT_CORE_H

#include "wka_common.h"

enum row_transform_mode_t {
  WKA_ROW_FORWARD = 0,
  WKA_ROW_INVERSE = 1,
  WKA_ROW_WINDOW_INVERSE = 2,
  WKA_ROW_BULK_STOLT_WINDOW_INVERSE = 3
};

void run_row_transform(bus_t *__restrict in, bus_t *__restrict out,
                       row_transform_mode_t mode);

#endif
