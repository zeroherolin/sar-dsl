//===- fft_core.h - Hand-written radix-4 HLS engine API ---------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef FFT_CORE_H
#define FFT_CORE_H

#include "wka_common.h"

void run_row_forward(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im);

void run_row_inverse(const plane_t *__restrict in_re,
                     const plane_t *__restrict in_im,
                     plane_t *__restrict out_re, plane_t *__restrict out_im);

void run_row_window_inverse(const plane_t *__restrict in_re,
                            const plane_t *__restrict in_im,
                            const real_t *__restrict window,
                            plane_t *__restrict out_re,
                            plane_t *__restrict out_im);

void run_row_bulk_stolt_window_inverse(const plane_t *__restrict in_re,
                                       const plane_t *__restrict in_im,
                                       const real_t *__restrict range_window,
                                       plane_t *__restrict out_re,
                                       plane_t *__restrict out_im);

#endif
