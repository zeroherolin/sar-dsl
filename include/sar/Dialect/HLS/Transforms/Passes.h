//===----------------------------------------------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_TRANSFORMS_PASSES_H
#define SAR_DIALECT_HLS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

/// Fusion mode to attempt. The default mode `Greedy` does both
/// producer-consumer and sibling fusion.
enum AffineFusionMode { Greedy, ProducerConsumer, Sibling };
enum CreateSubviewMode { Point, Reduction };

void registerHLSPipeline();
void registerTransformsPasses();

void addSimplifyAffineLoopPasses(OpPassManager &pm);
void addCreateSubviewPasses(
    OpPassManager &pm,
    CreateSubviewMode mode = CreateSubviewMode::Point);

std::unique_ptr<Pass>
createFuncPreprocessPass(std::string hlsTopFunc = "forward");

/// Dataflow-related passes.
std::unique_ptr<Pass> createBalanceDataflowNodePass();
std::unique_ptr<Pass>
createConvertDataflowToFuncPass(bool splitExternalAccess = true);
std::unique_ptr<Pass> createCreateDataflowFromAffinePass();
std::unique_ptr<Pass> createCreateTokenStreamPass();
std::unique_ptr<Pass> createEliminateMultiConsumerPass();
std::unique_ptr<Pass> createEliminateMultiProducerPass();
std::unique_ptr<Pass> createLegalizeDataflowPass();
std::unique_ptr<Pass> createLowerDataflowPass(bool splitExternalAccess = true);
std::unique_ptr<Pass> createParallelizeDataflowNodePass(
    unsigned loopUnrollFactor = 1, bool unrollPointLoopOnly = false,
    bool complexityAware = true, bool correlationAware = true);
std::unique_ptr<Pass>
createPlaceDataflowBufferPass(unsigned threshold = 1024,
                              bool placeExternalBuffer = true);
std::unique_ptr<Pass>
createScheduleDataflowNodePass(bool ignoreViolations = false);
std::unique_ptr<Pass> createStreamDataflowTaskPass();

/// Tensor-related passes.

/// Loop-related passes.
std::unique_ptr<Pass> createAffineLoopFusionPass(
    double computeToleranceThreshold = 0.3, unsigned fastMemorySpace = 0,
    uint64_t localBufSizeThreshold = 0, bool maximalFusion = false,
    enum AffineFusionMode fusionMode = AffineFusionMode::Greedy);
std::unique_ptr<Pass> createAffineLoopOrderOptPass();
std::unique_ptr<Pass> createAffineLoopPerfectionPass();
std::unique_ptr<Pass>
createAffineLoopTilePass(unsigned loopTileSize = 1,
                        unsigned tileBufferBytes = 512 * 1024);
std::unique_ptr<Pass> createRemoveVariableBoundPass();

/// Memory-related passes.
std::unique_ptr<Pass> createAffineStoreForwardPass();
std::unique_ptr<Pass> createCollapseMemrefUnitDimsPass();
std::unique_ptr<Pass> createCreateMemrefSubviewPass(
    CreateSubviewMode createSubviewMode = CreateSubviewMode::Point);
std::unique_ptr<Pass>
createCreateLocalBufferPass(bool externalBufferOnly = true,
                            bool registerOnly = false);
std::unique_ptr<Pass>
createLowerCopyToAffinePass(bool internalCopyOnly = false);
std::unique_ptr<Pass> createRaiseAffineToCopyPass();
std::unique_ptr<Pass> createReduceInitialIntervalPass();
std::unique_ptr<Pass> createSimplifyAffineIfPass();
std::unique_ptr<Pass> createSimplifyCopyPass();

/// Directive-related passes.
std::unique_ptr<Pass>
createArrayPartitionPass(unsigned lutramMaxBits = 1024,
                         unsigned maxFactor = 32);
std::unique_ptr<Pass>
createCreateAxiInterfacePass(std::string hlsTopFunc = "forward");
std::unique_ptr<Pass> createCreateHLSPrimitivePass();
std::unique_ptr<Pass> createLoopPipeliningPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_TRANSFORMS_PASSES_H
