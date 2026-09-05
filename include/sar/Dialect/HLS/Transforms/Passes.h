//===- Passes.h - HLS transform passes --------------------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_PASSES_H
#define SAR_DIALECT_HLS_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "sar/Dialect/HLS/IR/HLS.h"
#include <memory>

namespace mlir {
class Pass;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

namespace mlir {
namespace sar {

void registerHLSPipeline();
void registerHLSPasses();

void addSimplifyAffineLoopPasses(OpPassManager &pm);

std::unique_ptr<Pass> createFuncPreprocessPass(std::string hlsTopFunc = "main",
                                               bool rewriteOps = true);

/// Dataflow construction and legalization, in pipeline order.
std::unique_ptr<Pass> createCreateDataflowFromAffinePass();
std::unique_ptr<Pass> createLowerDataflowPass();
std::unique_ptr<Pass> createEliminateMultiProducerPass();
std::unique_ptr<Pass> createEliminateMultiConsumerPass();
std::unique_ptr<Pass>
createScheduleDataflowNodePass(bool ignoreViolations = false);
std::unique_ptr<Pass> createBalanceDataflowNodePass();
std::unique_ptr<Pass> createLegalizeDataflowPass();
std::unique_ptr<Pass> createPlaceDataflowBufferPass(
    unsigned threshold = 1024, unsigned bramBytes = 9907200,
    unsigned uramBytes = 37748736, unsigned lutramBytes = 2883584,
    unsigned lutramMaxBytes = 64, unsigned bramBlockBytes = 4608,
    unsigned uramBlockBytes = 36864, bool rebalanceOnly = false,
    bool allowDram = true);
std::unique_ptr<Pass> createCreateTokenStreamPass();
std::unique_ptr<Pass> createConvertDataflowToFuncPass();

/// Loop transformations.
std::unique_ptr<Pass> createAffineLoopOrderOptPass();
std::unique_ptr<Pass> createAffineLoopTilePass(unsigned loopTileSize = 1,
                                               unsigned tileBufferBytes = 512 *
                                                                          1024);
std::unique_ptr<Pass>
createFuseSiblingLoopsPass(bool preserveDataflowRoles = false);
std::unique_ptr<Pass> createReduceInitiationIntervalPass();

/// Memory and copy simplification.
std::unique_ptr<Pass> createAffineStoreForwardPass();
std::unique_ptr<Pass> createRaiseAffineToCopyPass();
std::unique_ptr<Pass> createSimplifyCopyPass();
std::unique_ptr<Pass>
createLowerCopyToAffinePass(bool internalCopyOnly = false);

/// Interface and directive generation.
std::unique_ptr<Pass>
createCreateAxiInterfacePass(std::string hlsTopFunc = "main",
                             bool streamInterface = false,
                             unsigned maxScratchArenas = 4);
std::unique_ptr<Pass> createWidenExternalMemoryPass(
    unsigned busBits = 512, unsigned maxLanes = 8, unsigned minElements = 4096,
    bool packPublicOutputs = true, unsigned computeMaxLanes = 0);
std::unique_ptr<Pass> createArrayPartitionPass(unsigned lutramMaxBits = 512,
                                               unsigned lutramBytes = 0,
                                               unsigned bramBytes = 9907200,
                                               unsigned uramBytes = 37748736,
                                               unsigned bramBlockBytes = 4608,
                                               unsigned uramBlockBytes = 36864,
                                               unsigned maxFactor = 32);
std::unique_ptr<Pass> createLoopPipeliningPass(unsigned maxUnrolledOps = 4096,
                                               unsigned maxUnrollFactor = 32);

/// Module-level sharing.
std::unique_ptr<Pass> createShareEquivalentFunctionsPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_TRANSFORMS_PASSES_H
