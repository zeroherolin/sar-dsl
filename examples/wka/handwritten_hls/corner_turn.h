//===- corner_turn.h - Hand-written HLS corner-turn API ---------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_CORNER_TURN_H
#define WKA_CORNER_TURN_H

#include "wka_common.h"

void corner_turn_complex(const plane_t *__restrict in_re,
                         const plane_t *__restrict in_im,
                         plane_t *__restrict out_re,
                         plane_t *__restrict out_im);

void corner_turn_magnitude(const plane_t *__restrict in_re,
                           const plane_t *__restrict in_im,
                           plane_t *__restrict out);

#endif
