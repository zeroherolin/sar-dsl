//===- corner_turn.h - Hand-written HLS corner-turn API --------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef WKA_CORNER_TURN_H
#define WKA_CORNER_TURN_H

#include "wka_common.h"

void corner_turn(bus_t *__restrict in, bus_t *__restrict out, bool magnitude_output);

#endif
