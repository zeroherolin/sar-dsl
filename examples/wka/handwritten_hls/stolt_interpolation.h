//===- stolt_interpolation.h - Hand-written Stolt HLS API -------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_STOLT_INTERPOLATION_H
#define WKA_STOLT_INTERPOLATION_H

#include "wka_common.h"

void load_bulk_stolt_block(const plane_t *__restrict in_re,
                           const plane_t *__restrict in_im,
                           const real_t range_window[N],
                           data_t fft_input[WKA_FFT_PAR_ROWS][N], int base_row);

#endif
