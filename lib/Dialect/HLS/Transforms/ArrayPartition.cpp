//===- ArrayPartition.cpp - array partition -------------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_ARRAYPARTITION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

#define DEBUG_TYPE "array-partition"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace hls;

/// Whether `array` is ever cut into a view, following calls into the callee.
///
/// The check has to cross call boundaries because a partitioned type
/// propagates from the caller into the callee's argument, so a view built
/// inside the callee stops verifying even though nothing at the call site
/// looks like a view.
static bool isViewedAnywhere(Value array, DenseSet<Value> &visited) {
  if (!visited.insert(array).second)
    return false;
  for (auto &use : array.getUses()) {
    auto *owner = use.getOwner();
    if (isa<memref::SubViewOp>(owner))
      return true;
    if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal())
        continue;
      if (isViewedAnywhere(callee.getArgument(use.getOperandNumber()), visited))
        return true;
    }
  }
  return false;
}

static void updateSubFuncs(func::FuncOp func, Builder builder) {
  func.walk([&](func::CallOp op) {
    auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
    auto subFunc = dyn_cast_or_null<func::FuncOp>(callee);
    if (!subFunc || subFunc.isExternal())
      return;

    // Set sub-function type.
    auto subResultTypes = op.getResultTypes();
    auto subInputTypes = op.getOperandTypes();
    auto newType = builder.getFunctionType(subInputTypes, subResultTypes);

    if (subFunc.getFunctionType() != newType) {
      subFunc.setType(newType);

      // Set arguments type.
      unsigned index = 0;
      for (auto inputType : op.getOperandTypes())
        subFunc.getArgument(index++).setType(inputType);

      // Set results type.
      auto returnOp = cast<func::ReturnOp>(subFunc.front().getTerminator());
      index = 0;
      for (auto resultType : op.getResultTypes())
        returnOp.getOperand(index++).setType(resultType);

      // Recursively apply array partition strategy.
      updateSubFuncs(subFunc, builder);
    }
  });
}

/// Apply the specified array partition factors and kinds.
bool sar::applyArrayPartition(Value array, ArrayRef<unsigned> factors,
                              ArrayRef<hls::PartitionKind> kinds,
                              bool updateFuncSignature, unsigned lutramMaxBits,
                              uint64_t lutramBitsBudget,
                              uint64_t *lutramBitsUsed) {
  auto arrayType = dyn_cast<MemRefType>(array.getType());
  if (!arrayType || isExtBuffer(array) || !arrayType.hasStaticShape() ||
      (int64_t)factors.size() != arrayType.getRank() ||
      (int64_t)kinds.size() != arrayType.getRank())
    return false;

  // A partition layout is not a strided layout, and `memref.subview`
  // requires both its base and its result to be strided. So neither end of a
  // view can carry one: the base because its views would stop verifying, the
  // result because the verifier derives its layout from the base. Views are
  // left unpartitioned, and the buffer they are cut from keeps whatever
  // partitioning the accesses through it already earned.
  DenseSet<Value> visited;
  if (array.getDefiningOp<memref::SubViewOp>() ||
      isViewedAnywhere(array, visited))
    return false;

  LLVM_DEBUG(llvm::dbgs() << "\nApply array partition to " << array << " at "
                          << array.getLoc(););
  LLVM_DEBUG(llvm::dbgs() << "\nfactors: ";);
  LLVM_DEBUG(for (auto factor : factors) llvm::dbgs() << factor << ", ";);
  LLVM_DEBUG(llvm::dbgs() << "\nkinds: ";);
  LLVM_DEBUG(for (auto kind : kinds) llvm::dbgs() << kind << ", ";);
  LLVM_DEBUG(llvm::dbgs() << "\n";);

  // Calculate the actual depth of the partitioned array.
  uint64_t actualDepth = 1;
  for (auto [factor, dimSize] : llvm::zip(factors, arrayType.getShape())) {
    if (factor == 0)
      continue;
    if (dimSize % factor != 0)
      return false;
    actualDepth *= dimSize / factor;
  }

  // Construct and set new array type.
  auto layoutAttr = PartitionLayoutAttr::getWithActualFactors(
      array.getContext(), kinds, SmallVector<int64_t>(factors),
      arrayType.getShape());
  auto memorySpaceAttr = arrayType.getMemorySpace();
  auto kindAttr =
      memorySpaceAttr ? cast<MemoryKindAttr>(memorySpaceAttr) : nullptr;
  // Distributed RAM is built out of LUTs, so what decides whether a bank
  // belongs there is how many bits it holds, not how many elements. Judging
  // by element count puts a 512-deep bank of doubles -- 32 Kbit, more than a
  // block RAM primitive holds -- into LUTs, and a design with a hundred of
  // them asks for more distributed RAM than the device has.
  auto elementBits = arrayType.getElementTypeBitWidth();
  uint64_t bankBits = (uint64_t)actualDepth * elementBits;
  // Inclusive, matching the placement pass (`bytes <= lutramMax`): the
  // threshold is one bus beat, and a bank holding exactly one beat is the
  // canonical distributed-RAM case. With the strict compare the FFT row
  // buffers -- 256 doubles split cyclic by 32, 512-bit banks against the
  // 512-bit threshold -- each kept a whole BRAM primitive per bank, and a
  // chain's worth of them asked the device for more block RAM than it has.
  if (bankBits <= lutramMaxBits) {
    // Distributed RAM comes out of the SLICEM LUTs the datapath is built
    // from, so it is the one tier that has to be rationed against the
    // budget here as well: past it, the bank keeps whatever placement
    // already chose rather than quietly overdrawing the fabric.
    uint64_t banks = 1;
    for (unsigned f : factors)
      banks *= std::max(1u, f);
    uint64_t cost = bankBits * banks;
    // A zero budget forbids distributed RAM, matching the configuration and
    // placement contracts. "Unlimited" is not a useful hardware budget and
    // must not be encoded by the same value as "disabled".
    bool affordable = lutramBitsBudget != 0 && lutramBitsUsed &&
                      *lutramBitsUsed + cost <= lutramBitsBudget;
    if (affordable) {
      if (lutramBitsUsed)
        *lutramBitsUsed += cost;
      kindAttr = MemoryKindAttr::get(array.getContext(), MemoryKind::LUTRAM_2P);
    }
  }
  array.setType(MemRefType::get(
      arrayType.getShape(), arrayType.getElementType(), layoutAttr, kindAttr));

  if (updateFuncSignature)
    if (auto func = array.getParentRegion()->getParentOfType<func::FuncOp>()) {
      auto builder = Builder(array.getContext());

      // Align function type with entry block argument types only if the array
      // is defined as an argument of the function.
      if (!array.getDefiningOp()) {
        auto resultTypes = func.front().getTerminator()->getOperandTypes();
        auto inputTypes = func.front().getArgumentTypes();
        func.setType(builder.getFunctionType(inputTypes, resultTypes));
      }

      // Update the types of all sub-functions.
      updateSubFuncs(func, builder);
    }
  return true;
}

static AffineMap getIdentityAffineMap(const SmallVectorImpl<Value> &operands,
                                      unsigned rank, MLIRContext *context) {
  SmallVector<AffineExpr, 4> exprs;
  exprs.reserve(rank);
  unsigned dimCount = 0;
  unsigned symbolCount = 0;

  for (auto operand : operands) {
    if (isValidDim(operand))
      exprs.push_back(getAffineDimExpr(dimCount++, context));
    else if (isValidSymbol(operand))
      exprs.push_back(getAffineSymbolExpr(symbolCount++, context));
    else
      return AffineMap();
  }
  return AffineMap::get(dimCount, symbolCount, exprs, context);
}

static AffineValueMap getAffineValueMap(Operation *op) {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  SmallVector<Value, 4> operands;
  if (auto loadOp = dyn_cast<mlir::affine::AffineReadOpInterface>(op)) {
    operands = loadOp.getMapOperands();
    map = loadOp.getAffineMap();

  } else if (auto storeOp =
                 dyn_cast<mlir::affine::AffineWriteOpInterface>(op)) {
    operands = storeOp.getMapOperands();
    map = storeOp.getAffineMap();

  } else if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    operands = readOp.getIndices();
    map = getIdentityAffineMap(operands, readOp.getShapedType().getRank(),
                               readOp.getContext());
  } else {
    auto writeOp = cast<vector::TransferWriteOp>(op);
    operands = writeOp.getIndices();
    map = getIdentityAffineMap(operands, writeOp.getShapedType().getRank(),
                               writeOp.getContext());
  }

  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  return AffineValueMap(map, operands);
}

static SmallVector<AffineMap, 4>
getDimAccessMaps(Operation *op, AffineValueMap valueMap, int64_t dim) {
  // Only keep the mapping result of the target dimension.
  auto baseMap = AffineMap::get(valueMap.getNumDims(), valueMap.getNumSymbols(),
                                valueMap.getResult(dim));

  AffineMap permuteMap;
  ArrayRef<int64_t> vectorShape;
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    permuteMap = readOp.getPermutationMap();
    vectorShape = readOp.getVectorType().getShape();
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    permuteMap = writeOp.getPermutationMap();
    vectorShape = writeOp.getVectorType().getShape();
  }

  SmallVector<AffineMap, 4> maps({baseMap});
  if (!permuteMap)
    return maps;

  // Traverse each dimension of the transferred vector.
  for (unsigned i = 0, e = permuteMap.getNumResults(); i < e; ++i) {
    auto dimExpr = dyn_cast<AffineDimExpr>(permuteMap.getResult(i));

    // If the permutation result of the current dimension is equal to the target
    // dimension, we push back the access map of each element of the vector into
    // the "maps" to be returned.
    if (dimExpr && dimExpr.getPosition() == dim) {
      for (int64_t offset = 0, size = vectorShape[i]; offset < size; ++offset) {
        auto map = AffineMap::get(baseMap.getNumDims(), baseMap.getNumSymbols(),
                                  baseMap.getResult(0) + offset);
        maps.push_back(map);
      }
      break;
    }
  }
  return maps;
}

SmallVector<int64_t> createPermutationMap(ArrayRef<Value> vec1,
                                          ArrayRef<Value> vec2) {
  if (llvm::SmallDenseSet<Value>(vec1.begin(), vec1.end()) !=
      llvm::SmallDenseSet<Value>(vec2.begin(), vec2.end()))
    return {};

  SmallVector<int64_t> permutation_map(vec1.size());
  llvm::SmallDenseMap<Value, int> index_map;

  for (size_t i = 0; i < vec1.size(); ++i) {
    index_map[vec1[i]] = i;
  }
  for (size_t i = 0; i < vec2.size(); ++i) {
    permutation_map[i] = index_map[vec2[i]];
  }
  return permutation_map;
}

/// Find the suitable array partition factors and kinds for all arrays in the
/// targeted function.
bool sar::applyAutoArrayPartition(func::FuncOp func, unsigned lutramMaxBits,
                                  unsigned maxFactor, uint64_t lutramBitsBudget,
                                  uint64_t *lutramBitsUsed) {
  // Check whether the input function is pipelined.
  bool funcPipeline = false;
  if (auto attr = getFuncDirective(func))
    funcPipeline = attr.getPipeline();

  // Collect target basic blocks to be considered.
  SmallVector<Block *, 4> targetBlocks;
  if (funcPipeline)
    targetBlocks.push_back(&func.front());
  else {
    AffineLoopBands targetBands;
    getLoopBands(func.front(), targetBands);
    for (auto &band : targetBands)
      targetBlocks.push_back(band.back().getBody());
  }

  // The partition choice per memref. Several blocks may access the same
  // memref with different best fashions and factors; the largest factor
  // wins, since a bank can be read below capacity but too few banks stall
  // the widest loop. A MapVector: the application loop below spends the
  // LUTRAM budget first come, first served, so iteration order has to be
  // the program order the accesses were collected in, not pointer order.
  using Partition = std::pair<PartitionKind, int64_t>;
  llvm::MapVector<Value, SmallVector<Partition, 4>> partitionsMap;

  // Traverse all blocks that requires to be considered.
  for (auto block : targetBlocks) {
    MemAccessesMap accessesMap;
    getMemAccessesMap(*block, accessesMap, /*includeVectorTransfer=*/true);

    for (auto [memref, loadStores] : accessesMap) {
      auto memrefType = cast<MemRefType>(memref.getType());
      auto &partitions = partitionsMap[memref];

      // If the current partitionsMap is empty, initialize it with no partition.
      if (partitions.empty())
        partitions = SmallVector<Partition, 4>(
            memrefType.getRank(), Partition(PartitionKind::NONE, 1));

      LLVM_DEBUG(llvm::dbgs()
                     << "\n----------\nArray partition for " << memref;);

      // Find the best partition solution for each dimensions of the
      // memref.
      for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
        // Collect all array access indices of the current dimension.
        SmallVector<AffineValueMap, 4> indices;

        LLVM_DEBUG(llvm::dbgs() << "\n\nDimension " << dim << "";);

        for (auto accessOp : loadStores) {
          auto valueMap = getAffineValueMap(accessOp);
          if (valueMap.getAffineMap().isEmpty())
            continue;

          auto dimMaps = getDimAccessMaps(accessOp, valueMap, dim);
          for (auto dimMap : dimMaps) {
            AffineValueMap dimValueMap(dimMap, valueMap.getOperands());
            (void)dimValueMap.canonicalize();

            // Only add unique index.
            if (find_if(indices, [&](auto index) {
                  return index.getAffineMap() == dimValueMap.getAffineMap() &&
                         index.getOperands() == dimValueMap.getOperands();
                }) == indices.end()) {
              indices.push_back(dimValueMap);
              LLVM_DEBUG(llvm::dbgs()
                             << "\nIndex: " << dimValueMap.getResult(0););
            }
          }
        }
        auto accessNum = indices.size();

        // Find the max array access distance in the current block.
        unsigned maxDistance = 0;
        unsigned maxCommonDivisor = 0;
        bool requireMux = false;

        for (unsigned i = 0; i < accessNum; ++i) {
          for (unsigned j = i + 1; j < accessNum; ++j) {
            auto lhsIndex = indices[i];
            auto rhsIndex = indices[j];
            auto lhsExpr = lhsIndex.getResult(0);
            auto rhsExpr = rhsIndex.getResult(0);

            if (lhsIndex.getOperands() != rhsIndex.getOperands()) {
              // Here, we try to find a permutation map to make the two index
              // identical.
              auto possiblePermutation = createPermutationMap(
                  lhsIndex.getOperands(), rhsIndex.getOperands());

              if (possiblePermutation.empty()) {
                // If no permutation map is found, we need to use a mux to
                // select value from the partitioned array. Meanwhile, we cannot
                // calculate the distance in this case, so continue.
                requireMux = true;
                continue;
              } else {
                // If a permutation map is found, we need to apply it to the
                // rhsExpr.
                SmallVector<AffineExpr, 4> dimReplacements;
                SmallVector<AffineExpr, 4> symReplacements;
                for (auto i : possiblePermutation) {
                  if (i < rhsIndex.getNumDims())
                    dimReplacements.push_back(
                        getAffineDimExpr(i, func.getContext()));
                  else
                    symReplacements.push_back(getAffineSymbolExpr(
                        i - rhsIndex.getNumDims(), func.getContext()));
                }
                rhsExpr = rhsExpr.replaceDimsAndSymbols(dimReplacements,
                                                        symReplacements);
              }
            }

            LLVM_DEBUG(llvm::dbgs() << "\nDistance: "
                                    << "(" << lhsExpr << ")"
                                    << " - "
                                    << "(" << rhsExpr << ")";);
            auto newExpr =
                simplifyAffineExpr(rhsExpr - lhsExpr, lhsIndex.getNumDims(),
                                   lhsIndex.getNumSymbols());

            if (auto constDistance = dyn_cast<AffineConstantExpr>(newExpr)) {
              LLVM_DEBUG(llvm::dbgs() << " = " << constDistance.getValue(););

              unsigned distance = std::abs(constDistance.getValue());
              maxDistance = std::max(maxDistance, distance);
              maxCommonDivisor = std::gcd(distance, maxCommonDivisor);
            } else
              requireMux = true;
          }
        }
        ++maxDistance;

        // This means all accesses have the same index, and this dimension
        // should not be partitioned.
        if (maxDistance == 1)
          continue;

        // Determine array partition factor and kind. The factor covers the
        // accesses a pipelined iteration has in flight; which physical
        // memory the banks land in is decided afterwards (placement chose a
        // tier, and the bit-count check below may move small banks to
        // distributed RAM).
        int64_t factor = 1;
        PartitionKind kind = PartitionKind::NONE;
        if (accessNum >= maxDistance) {
          // This means some elements are accessed more than once or exactly
          // once, and successive elements are accessed. In most cases, apply
          // "cyclic" partition should be the best solution.
          factor = maxDistance;
          kind = PartitionKind::CYCLIC;
        } else if (maxCommonDivisor > 1) {
          // This means the memory access is perfectly strided.
          factor = maxDistance;
          while (factor % maxCommonDivisor != 0)
            factor++;
          kind = PartitionKind::CYCLIC;
        } else {
          // This means elements are accessed in a descrete manner however not
          // strided. Typically, "block" partition will be the most benefitial
          // partition strategy.
          factor = accessNum;
          kind = PartitionKind::BLOCK;
        }

        LLVM_DEBUG(llvm::dbgs() << "\nStrategy: "
                                << " factor=" << factor << " kind=" << kind;);

        // The strategies above derive the factor from the distance between
        // accesses, which on a loop that is pipelined rather than unrolled
        // can reach the trip count -- a 4096-point transform would ask for
        // 4096 banks, turning one block RAM into 4096 registers behind a
        // crossbar. Banks only have to cover the accesses in flight, so cap
        // the factor, staying on a divisor of the dimension because a factor
        // that does not divide it disables partitioning altogether.
        if (factor > (int64_t)maxFactor) {
          auto dimSize = memrefType.getDimSize(dim);
          int64_t capped = 1;
          for (int64_t d = maxFactor; d > 1; --d)
            if (dimSize % d == 0) {
              capped = d;
              break;
            }
          factor = capped;
          if (factor == 1)
            kind = PartitionKind::NONE;
        }

        // Across the loops accessing this array, keep the largest factor
        // seen: a factor that satisfies the widest access pattern also
        // serves the narrower ones (banks can be read below capacity, but
        // too few banks stall the widest loop).
        if (factor > partitions[dim].second) {
          LLVM_DEBUG(llvm::dbgs() << " (update)";);

          // When the accessed bank cannot be determined statically and the
          // factor exceeds 3, the tool inserts a multiplexer and wraps the
          // access in a function call, which worsens latency and II; cap
          // the factor at the largest divisor no greater than 3.
          if (requireMux) {
            for (auto i = 3; i > 0; --i)
              if (factor % i == 0) {
                partitions[dim] = Partition(kind, i);
                break;
              }
          } else
            partitions[dim] = Partition(kind, factor);
        }
      }

      LLVM_DEBUG(llvm::dbgs() << "\n\nAccesses: ";);
      for (auto op : loadStores)
        LLVM_DEBUG(llvm::dbgs() << "\n" << *op;);
    }
  }

  // Apply partition to all sub-functions and traverse all function to update
  // the "partitionsMap".
  func.walk([&](func::CallOp op) {
    auto subFunc = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()));
    if (!subFunc)
      return;

    // Apply array partition to the sub-function.
    applyAutoArrayPartition(subFunc, lutramMaxBits, maxFactor, lutramBitsBudget,
                            lutramBitsUsed);

    for (auto [type, operand] :
         llvm::zip(subFunc.getArgumentTypes(), op.getOperands())) {
      if (auto memrefType = dyn_cast<MemRefType>(type)) {
        auto &partitions = partitionsMap[operand];

        // If the current partitionsMap is empty, initialize it with no
        // partition.
        if (partitions.empty())
          partitions = SmallVector<Partition, 4>(
              memrefType.getRank(), Partition(PartitionKind::NONE, 1));

        // Traverse all dimension of the memref.
        if (auto attr = dyn_cast<PartitionLayoutAttr>(memrefType.getLayout()))
          for (int64_t dim = 0; dim < memrefType.getRank(); ++dim) {
            auto kind = attr.getKinds()[dim];
            auto factor = attr.getFactors()[dim];

            // If the factor from the sub-function is larger than the current
            // factor, replace it.
            if (factor > partitions[dim].second)
              partitions[dim] = Partition(kind, factor);
          }
      } else
        operand.setType(type);
    }
  });

  // Constuct and set new type to each partitioned MemRefType.
  auto builder = Builder(func);
  for (auto [memref, partitions] : partitionsMap) {
    SmallVector<hls::PartitionKind, 4> kinds;
    SmallVector<unsigned, 4> factors;
    for (auto [kind, factor] : partitions) {
      kinds.push_back(kind);
      factors.push_back(factor);
    }

    if (llvm::any_of(kinds, [](PartitionKind kind) {
          return kind != PartitionKind::NONE;
        }))
      applyArrayPartition(memref, factors, kinds, false, lutramMaxBits,
                          lutramBitsBudget, lutramBitsUsed);

    if (auto axiPort = memref.getDefiningOp<AxiPortOp>()) {
      auto axiType = AxiType::get(memref.getContext(), memref.getType());
      LLVM_DEBUG(llvm::dbgs() << "\nUpdate AxiPort type: " << *axiPort
                              << ", Type: " << axiType << "\n";);
      axiPort.getAxi().setType(axiType);
      LLVM_DEBUG(llvm::dbgs() << "Updated op: " << *axiPort << "\n";);
    }
  }

  // Align function type with entry block argument types.
  auto resultTypes = func.front().getTerminator()->getOperandTypes();
  auto inputTypes = func.front().getArgumentTypes();
  func.setType(builder.getFunctionType(inputTypes, resultTypes));

  // Update the types of all sub-functions.
  updateSubFuncs(func, builder);
  return true;
}

namespace {
static uint64_t roundUpTo(uint64_t value, uint64_t grain) {
  return ((value + grain - 1) / grain) * grain;
}

static LogicalResult verifyFinalMemoryBudget(ModuleOp module,
                                             uint64_t bramBudget,
                                             uint64_t uramBudget,
                                             uint64_t lutramBudget) {
  constexpr uint64_t bramBlockBytes = 4608;
  constexpr uint64_t uramBlockBytes = 36864;
  struct Usage {
    uint64_t bram = 0;
    uint64_t uram = 0;
    uint64_t lutram = 0;
  };
  DenseMap<Operation *, Usage> functionUsage;
  DenseSet<Value> charged;

  auto charge = [&](Value value, uint64_t copies, Usage &usage) {
    if (!charged.insert(value).second)
      return;
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!type || !type.hasStaticShape() ||
        !type.getElementType().isIntOrFloat())
      return;
    auto kind = getMemoryKind(type);
    if (kind == MemoryKind::UNKNOWN || kind == MemoryKind::DRAM)
      return;
    uint64_t bytes = (uint64_t)type.getNumElements() *
                     ((type.getElementTypeBitWidth() + 7) / 8);
    uint64_t banks = std::max<int64_t>(1, getPartitionFactors(type));
    uint64_t bankBytes = (bytes + banks - 1) / banks;
    switch (kind) {
    case MemoryKind::LUTRAM_1P:
    case MemoryKind::LUTRAM_2P:
    case MemoryKind::LUTRAM_S2P:
      usage.lutram += copies * bytes;
      break;
    case MemoryKind::BRAM_1P:
    case MemoryKind::BRAM_2P:
    case MemoryKind::BRAM_S2P:
    case MemoryKind::BRAM_T2P:
      usage.bram += copies * banks * roundUpTo(bankBytes, bramBlockBytes);
      break;
    case MemoryKind::URAM_1P:
    case MemoryKind::URAM_2P:
    case MemoryKind::URAM_S2P:
    case MemoryKind::URAM_T2P:
      usage.uram += copies * banks * roundUpTo(bankBytes, uramBlockBytes);
      break;
    case MemoryKind::UNKNOWN:
    case MemoryKind::DRAM:
      break;
    }
  };

  module.walk([&](hls::BufferLikeInterface buffer) {
    auto func = buffer->getParentOfType<func::FuncOp>();
    if (func)
      charge(buffer.getMemref(), isa<ConstBufferOp>(*buffer) ? 1 : 2,
             functionUsage[func.getOperation()]);
  });

  func::FuncOp designTop;
  for (auto func : module.getOps<func::FuncOp>())
    if (hasTopFuncAttr(func))
      designTop = func;
  bool hasRuntimeWrapper =
      llvm::any_of(module.getOps<func::FuncOp>(),
                   [](func::FuncOp func) { return hasRuntimeAttr(func); });
  if (!hasRuntimeWrapper)
    for (auto func : module.getOps<func::FuncOp>())
      if (hasTopFuncAttr(func))
        for (BlockArgument arg : func.getArguments())
          charge(arg, 1, functionUsage[func.getOperation()]);

  // Outlined helpers execute under the design top. Their local arrays have
  // disjoint call lifetimes, so peak storage is the top's persistent buffers
  // plus the largest helper frame, not the sum of every helper in the module.
  Usage peak = designTop ? functionUsage[designTop.getOperation()] : Usage{};
  Usage helperPeak;
  for (auto &[func, usage] : functionUsage)
    if (!designTop || func != designTop.getOperation()) {
      helperPeak.bram = std::max(helperPeak.bram, usage.bram);
      helperPeak.uram = std::max(helperPeak.uram, usage.uram);
      helperPeak.lutram = std::max(helperPeak.lutram, usage.lutram);
    }
  uint64_t bram = peak.bram + helperPeak.bram;
  uint64_t uram = peak.uram + helperPeak.uram;
  uint64_t lutram = peak.lutram + helperPeak.lutram;

  auto over = [](uint64_t used, uint64_t budget) {
    return used > budget || (budget == 0 && used != 0);
  };
  if (!over(bram, bramBudget) && !over(uram, uramBudget) &&
      !over(lutram, lutramBudget))
    return success();
  module.emitError() << "SAR_HLS_RETRYABLE_PARTITION_OVERFLOW: final banked "
                        "memories exceed the resource budgets: BRAM "
                     << bram << "/" << bramBudget << " bytes, URAM " << uram
                     << "/" << uramBudget << " bytes, LUTRAM " << lutram << "/"
                     << lutramBudget << " bytes";
  return failure();
}

struct ArrayPartition : public sar::impl::ArrayPartitionBase<ArrayPartition> {
  using sar::impl::ArrayPartitionBase<ArrayPartition>::ArrayPartitionBase;

  ArrayPartition() = default;
  ArrayPartition(unsigned argLutramMaxBits, unsigned argLutramBytes,
                 unsigned argBramBytes, unsigned argUramBytes,
                 unsigned argMaxFactor) {
    lutramMaxBits = argLutramMaxBits;
    lutramBytes = argLutramBytes;
    bramBytes = argBramBytes;
    uramBytes = argUramBytes;
    maxFactor = argMaxFactor;
  }

  void runOnOperation() override {
    auto module = getOperation();

    // Get the top function. After AXI interface creation the module holds a
    // runtime wrapper that calls the design top; partitioning must start from
    // the outermost caller so the partitioned types propagate through every
    // sub-function signature, so the wrapper wins when present.
    func::FuncOp topFunc;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (hasRuntimeAttr(func)) {
        topFunc = func;
        break;
      } else if (hasTopFuncAttr(func))
        topFunc = func;
    }

    if (!topFunc) {
      emitError(module.getLoc(), "fail to find the top function");
      return signalPassFailure();
    }
    // Buffer placement already spent part of the distributed-RAM cap on
    // whole buffers; banking draws from what is left of the same cap. The
    // figure sits on the design function, which under an AXI wrapper is
    // not the outermost one, so every function is consulted.
    uint64_t lutramBitsUsed = 0;
    for (auto func : module.getOps<func::FuncOp>())
      if (auto spent = func->getAttrOfType<IntegerAttr>("lutram_spent")) {
        lutramBitsUsed += (uint64_t)spent.getInt() * 8;
        func->removeAttr("lutram_spent");
      }
    applyAutoArrayPartition(topFunc, lutramMaxBits, maxFactor,
                            (uint64_t)lutramBytes * 8, &lutramBitsUsed);
    if (failed(
            verifyFinalMemoryBudget(module, bramBytes, uramBytes, lutramBytes)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> sar::createArrayPartitionPass(unsigned lutramMaxBits,
                                                    unsigned lutramBytes,
                                                    unsigned bramBytes,
                                                    unsigned uramBytes,
                                                    unsigned maxFactor) {
  return std::make_unique<ArrayPartition>(lutramMaxBits, lutramBytes, bramBytes,
                                          uramBytes, maxFactor);
}
