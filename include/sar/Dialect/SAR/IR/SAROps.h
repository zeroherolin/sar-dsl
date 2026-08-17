//===- SAROps.h - SAR operation declarations --------------------*- C++-*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_SAR_IR_SAROPS_H
#define SAR_DIALECT_SAR_IR_SAROPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "sar/Dialect/SAR/IR/SARDialect.h"

#define GET_OP_CLASSES
#include "sar/Dialect/SAR/IR/SAROps.h.inc"

#endif // SAR_DIALECT_SAR_IR_SAROPS_H
