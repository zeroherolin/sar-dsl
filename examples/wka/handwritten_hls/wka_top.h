//===- wka_top.h - Hand-written omega-K HLS top API -------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_TOP_H
#define WKA_TOP_H

#include "wka_common.h"

void wka_sar_top(plane_t raw_re[N][PLANE_WORDS_PER_ROW],
                 plane_t raw_im[N][PLANE_WORDS_PER_ROW], real_t win_r[N],
                 real_t win_a[N], plane_t out0[N][PLANE_WORDS_PER_ROW],
                 plane_t scratch0[PLANE_WORDS], plane_t scratch1[PLANE_WORDS],
                 plane_t scratch2[PLANE_WORDS], plane_t scratch3[PLANE_WORDS]);

#endif
