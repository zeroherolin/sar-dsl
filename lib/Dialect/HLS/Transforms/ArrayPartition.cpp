//===- ArrayPartition.cpp - bank on-chip arrays within the tiers ----------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"
#include "sar/Support/HLSHints.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"

#include <functional>
#include <limits>

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_ARRAYPARTITION
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

#define DEBUG_TYPE "hls-array-partition"

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

static void updateSubFuncs(func::FuncOp func, Builder builder);

/// Re-derives `func`'s type from its entry block after a retyped array, then
/// pushes the new operand types into every callee. A retyped block argument
/// leaves the signature stale on its own function, which no callee walk can
/// repair.
static void retypeEnclosingFuncs(Value array, Builder builder) {
  auto func = array.getParentRegion()->getParentOfType<func::FuncOp>();
  if (!func)
    return;
  if (isa<BlockArgument>(array))
    func.setType(builder.getFunctionType(
        func.front().getArgumentTypes(),
        func.front().getTerminator()->getOperandTypes()));
  updateSubFuncs(func, builder);
}

static void updateSubFuncs(func::FuncOp func, Builder builder) {
  func.walk([&](func::CallOp op) {
    auto callee = SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr());
    auto subFunc = dyn_cast_or_null<func::FuncOp>(callee);
    if (!subFunc || subFunc.isExternal())
      return;

    auto subResultTypes = op.getResultTypes();
    auto subInputTypes = op.getOperandTypes();
    auto newType = builder.getFunctionType(subInputTypes, subResultTypes);

    if (subFunc.getFunctionType() != newType) {
      subFunc.setType(newType);

      unsigned index = 0;
      for (auto inputType : op.getOperandTypes())
        subFunc.getArgument(index++).setType(inputType);

      auto returnOp = cast<func::ReturnOp>(subFunc.front().getTerminator());
      index = 0;
      for (auto resultType : op.getResultTypes())
        returnOp.getOperand(index++).setType(resultType);

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
    // already chose, keeping the emitted design within the fabric budget.
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
    // dimension, the access map of each vector element goes into
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

static SmallVector<int64_t> createPermutationMap(ArrayRef<Value> vec1,
                                                 ArrayRef<Value> vec2) {
  if (llvm::SmallDenseSet<Value>(vec1.begin(), vec1.end()) !=
      llvm::SmallDenseSet<Value>(vec2.begin(), vec2.end()))
    return {};

  SmallVector<int64_t> permutationMap(vec1.size());
  llvm::SmallDenseMap<Value, int> indexMap;

  for (size_t i = 0; i < vec1.size(); ++i) {
    indexMap[vec1[i]] = i;
  }
  for (size_t i = 0; i < vec2.size(); ++i) {
    permutationMap[i] = indexMap[vec2[i]];
  }
  return permutationMap;
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

  // Traverse all blocks that need to be considered.
  for (auto block : targetBlocks) {
    MemAccessesMap accessesMap;
    getMemAccessesMap(*block, accessesMap, /*includeVectorTransfer=*/true);

    for (auto [memref, loadStores] : accessesMap) {
      auto memrefType = cast<MemRefType>(memref.getType());
      // A partition layout in the type means the banking was pinned by a
      // hint (or an outer call); the search must not override it.
      if (isa<PartitionLayoutAttr>(memrefType.getLayout()))
        continue;
      auto &partitions = partitionsMap[memref];

      // If the current partitionsMap is empty, initialize it with no partition.
      if (partitions.empty())
        partitions = SmallVector<Partition, 4>(
            memrefType.getRank(), Partition(PartitionKind::NONE, 1));

      LLVM_DEBUG(llvm::dbgs()
                     << "\n----------\nArray partition for " << memref;);

      // Find the best partition solution for each dimension of the
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
              // Look for a permutation map making the two indices
              // identical.
              auto possiblePermutation = createPermutationMap(
                  lhsIndex.getOperands(), rhsIndex.getOperands());

              if (possiblePermutation.empty()) {
                // Without one, the array needs a mux to select the bank
                // and the distance is not computable; skip the pair.
                requireMux = true;
                continue;
              } else {
                // Otherwise apply it to the right-hand expression.
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
        // Convert the largest constant distance into the number of elements
        // it spans.
        ++maxDistance;

        // An invariant index does not benefit from partitioning.
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
          // Cyclic partitioning spreads repeated or consecutive accesses.
          factor = maxDistance;
          kind = PartitionKind::CYCLIC;
        } else if (maxCommonDivisor > 1) {
          // A uniform stride maps naturally to cyclic banks.
          factor = maxDistance;
          while (factor % maxCommonDivisor != 0)
            factor++;
          kind = PartitionKind::CYCLIC;
        } else {
          // Irregular discrete accesses use block partitioning.
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

      LLVM_DEBUG({
        llvm::dbgs() << "\n\nAccesses: ";
        for (auto op : loadStores)
          llvm::dbgs() << "\n" << *op;
      });
    }
  }

  // Partition callees before propagating their argument layouts to call sites.
  func.walk([&](func::CallOp op) {
    auto subFunc = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()));
    if (!subFunc)
      return;

    applyAutoArrayPartition(subFunc, lutramMaxBits, maxFactor, lutramBitsBudget,
                            lutramBitsUsed);

    for (auto [type, operand] :
         llvm::zip(subFunc.getArgumentTypes(), op.getOperands())) {
      if (auto memrefType = dyn_cast<MemRefType>(type)) {
        // Pinned by a hint before the search ran; leave it alone.
        if (auto operandType = dyn_cast<MemRefType>(operand.getType()))
          if (isa<PartitionLayoutAttr>(operandType.getLayout()))
            continue;
        auto &partitions = partitionsMap[operand];

        // If the current partitionsMap is empty, initialize it with no
        // partition.
        if (partitions.empty())
          partitions = SmallVector<Partition, 4>(
              memrefType.getRank(), Partition(PartitionKind::NONE, 1));

        // Traverse all dimensions of the memref.
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

  // Construct and set new type to each partitioned MemRefType.
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

/// Number of synthesized reader processes that may need a private ROM copy.
///
/// A table handed down a call chain is instantiated once per process that
/// actually reads it, so the count has to follow it into the callees rather
/// than stop at the top-level call sites: a transform's twiddle table reaches
/// one engine but is read by each of its stages, and counting the engine alone
/// understates the block RAM by the stage count. Copies multiply along call
/// paths for the same reason. A shared helper is one allocated instance
/// however often it is called.
static uint64_t getConstReaderCopies(Value value) {
  DenseSet<Operation *> sharedInstances;

  std::function<uint64_t(Value)> count = [&](Value current) -> uint64_t {
    uint64_t copies = 0;
    bool readHere = false;
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      auto call = dyn_cast<func::CallOp>(owner);
      if (!call) {
        readHere = true;
        continue;
      }
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal() ||
          use.getOperandNumber() >= callee.getNumArguments()) {
        readHere = true;
        continue;
      }
      if (callee->hasAttr("hls.shared_instance") &&
          !sharedInstances.insert(callee).second)
        continue;
      copies += count(callee.getArgument(use.getOperandNumber()));
    }
    return copies + readHere;
  };

  return std::max<uint64_t>(1, count(value));
}

/// Re-binds banked UltraRAM arrays whose banks cannot fill an UltraRAM.
///
/// Placement picks a tier from the unbanked array, but banking is decided
/// later and a memory primitive is claimed whole: a 64 KiB array bound to
/// UltraRAM and then split into 64 banks holds a kilobyte in each of 64
/// UltraRAMs. Block RAM has an eighth of the granularity, so such a bank
/// fits it far better -- but the block tier is also the smaller budget, so
/// the move is only worth making while it has room. Arrays are taken
/// worst-waste first, which spends the block budget where it buys the most.
static void rebalanceBankedStorage(ModuleOp module, uint64_t bramBudget,
                                   uint64_t uramBudget, uint64_t bramBlockBytes,
                                   uint64_t uramBlockBytes) {
  struct Candidate {
    Value array;
    uint64_t uramBytes;
    uint64_t bramBytes;
  };
  SmallVector<Candidate> candidates;
  uint64_t bramUsed = 0;

  auto measure = [](MemRefType type, uint64_t block) {
    uint64_t bytes = (uint64_t)type.getNumElements() *
                     ((type.getElementTypeBitWidth() + 7) / 8);
    uint64_t banks = std::max<int64_t>(1, getPartitionFactors(type));
    return banks * roundUpTo((bytes + banks - 1) / banks, block);
  };

  // Block-RAM-bound arrays, most expensive first: when the block tier is the
  // one under pressure they are the candidates to move the other way.
  SmallVector<Candidate> blockBound;
  uint64_t uramUsed = 0;

  DenseSet<Value> seen;
  auto visit = [&](Value array, uint64_t copies) {
    auto type = dyn_cast<MemRefType>(array.getType());
    if (!type || !type.hasStaticShape() ||
        !type.getElementType().isIntOrFloat() || !seen.insert(array).second)
      return;
    switch (getMemoryKind(type)) {
    case MemoryKind::BRAM_1P:
    case MemoryKind::BRAM_2P:
    case MemoryKind::BRAM_S2P:
    case MemoryKind::BRAM_T2P: {
      uint64_t bram = copies * measure(type, bramBlockBytes);
      bramUsed += bram;
      blockBound.push_back(
          {array, uramBlockBytes ? copies * measure(type, uramBlockBytes) : 0,
           bram});
      return;
    }
    case MemoryKind::URAM_1P:
    case MemoryKind::URAM_2P:
    case MemoryKind::URAM_S2P:
    case MemoryKind::URAM_T2P:
      if (uramBlockBytes)
        uramUsed += copies * measure(type, uramBlockBytes);
      break;
    default:
      return;
    }
    // An UltraRAM binding is worth moving whenever block RAM would hold the
    // same array in fewer bytes -- which is the case exactly when a bank
    // cannot fill an UltraRAM, since a primitive is claimed whole and the
    // block primitive is an eighth the size.
    uint64_t uram = uramBlockBytes ? copies * measure(type, uramBlockBytes) : 0;
    uint64_t bram = copies * measure(type, bramBlockBytes);
    if (uramBlockBytes && bram < uram)
      candidates.push_back({array, uram, bram});
  };

  // Charged the way the final budget check charges: a dataflow buffer is
  // ping-ponged, and a constant table is instantiated once per reader.
  module.walk([&](hls::BufferLikeInterface buffer) {
    visit(buffer.getMemref(), isa<ConstBufferOp>(*buffer)
                                  ? getConstReaderCopies(buffer.getMemref())
                                  : 2);
  });
  for (auto func : module.getOps<func::FuncOp>())
    if (hasTopFuncAttr(func))
      for (BlockArgument arg : func.getArguments())
        visit(arg, 1);

  llvm::stable_sort(candidates, [](const Candidate &lhs, const Candidate &rhs) {
    return lhs.uramBytes - lhs.bramBytes > rhs.uramBytes - rhs.bramBytes;
  });
  for (Candidate &candidate : candidates) {
    // Block RAM is the smaller budget, so a move that would overrun it trades
    // one overflow for another and is not made. Once the UltraRAM tier is
    // inside its budget there is nothing left to relieve.
    if (uramUsed <= uramBudget)
      break;
    if (bramUsed + candidate.bramBytes > bramBudget)
      continue;
    auto type = cast<MemRefType>(candidate.array.getType());
    candidate.array.setType(MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(),
        MemoryKindAttr::get(type.getContext(), MemoryKind::BRAM_T2P)));
    // The tier is part of the memref type, so the owning function and every
    // callee handed this array have to be retyped with it, or the signature
    // and the call stop verifying.
    retypeEnclosingFuncs(candidate.array, Builder(type.getContext()));
    bramUsed += candidate.bramBytes;
    uramUsed -= candidate.uramBytes;
  }

  // The other direction, for a design whose pressure is on the block tier
  // instead: a transform's twiddle ROM is instantiated once per reading
  // stage, and a chain of engines can spend more of the block budget on
  // those copies than on its working buffers while UltraRAM sits idle.
  // Moved largest-first, since each move frees the most block RAM per
  // UltraRAM spent.
  if (bramUsed <= bramBudget)
    return;
  llvm::stable_sort(blockBound, [](const Candidate &lhs, const Candidate &rhs) {
    return lhs.bramBytes > rhs.bramBytes;
  });
  for (Candidate &candidate : blockBound) {
    if (bramUsed <= bramBudget)
      break;
    if (!uramBlockBytes || uramUsed + candidate.uramBytes > uramBudget)
      continue;
    auto type = cast<MemRefType>(candidate.array.getType());
    candidate.array.setType(MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(),
        MemoryKindAttr::get(type.getContext(), MemoryKind::URAM_T2P)));
    retypeEnclosingFuncs(candidate.array, Builder(type.getContext()));
    uramUsed += candidate.uramBytes;
    bramUsed -= candidate.bramBytes;
  }
}

static LogicalResult
verifyFinalMemoryBudget(ModuleOp module, uint64_t bramBudget,
                        uint64_t uramBudget, uint64_t lutramBudget,
                        uint64_t bramBlockBytes, uint64_t uramBlockBytes) {
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
      if (uramBlockBytes)
        usage.uram += copies * banks * roundUpTo(bankBytes, uramBlockBytes);
      else
        usage.uram = std::numeric_limits<uint64_t>::max();
      break;
    case MemoryKind::UNKNOWN:
    case MemoryKind::DRAM:
      break;
    }
  };

  // A helper carrying `#pragma HLS inline` is folded into its caller, so its
  // buffers live in the caller's frame. Walk up to the first function that
  // stays a module and charge them there.
  auto owningFrame = [&](Operation *operation) -> func::FuncOp {
    auto func = operation->getParentOfType<func::FuncOp>();
    while (func && func->getAttr("inline")) {
      func::FuncOp caller;
      module.walk([&](func::CallOp call) {
        if (call.getCallee() == func.getName() && !caller)
          caller = call->getParentOfType<func::FuncOp>();
      });
      if (!caller || caller == func)
        break;
      func = caller;
    }
    return func;
  };

  module.walk([&](hls::BufferLikeInterface buffer) {
    auto func = owningFrame(buffer.getOperation());
    if (!func)
      return;
    // Vitis double-buffers a dataflow channel so producer and consumer can
    // run concurrently; a buffer in a sequentially scheduled region is one
    // instance. Charging every buffer twice overstates a design that carries
    // no `#pragma HLS dataflow` by the whole working set.
    auto directive = getFuncDirective(func);
    bool pingPong = directive && directive.getDataflow();
    uint64_t copies = isa<ConstBufferOp>(*buffer)
                          ? getConstReaderCopies(buffer.getMemref())
                          : (pingPong ? 2 : 1);
    charge(buffer.getMemref(), copies, functionUsage[func.getOperation()]);
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

  // A helper that stays a module becomes its own RTL block holding its own
  // memories, whether or not the calls overlap in time -- storage is bound at
  // elaboration, not scheduled -- so the design costs one frame per such
  // helper. Charging only the largest of them, as if sequential calls could
  // share primitives, understates a chain of four transform engines by a
  // factor of four. A helper carrying `#pragma HLS inline` is folded into its
  // caller instead, and its buffers were already charged there.
  Usage total = designTop ? functionUsage[designTop.getOperation()] : Usage{};
  for (auto &[func, usage] : functionUsage) {
    if (designTop && func == designTop.getOperation())
      continue;
    if (func->getAttr("inline"))
      continue;
    total.bram += usage.bram;
    total.uram += usage.uram;
    total.lutram += usage.lutram;
  }
  uint64_t bram = total.bram;
  uint64_t uram = total.uram;
  uint64_t lutram = total.lutram;

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
                 unsigned argBramBlockBytes, unsigned argUramBlockBytes,
                 unsigned argMaxFactor) {
    lutramMaxBits = argLutramMaxBits;
    lutramBytes = argLutramBytes;
    bramBytes = argBramBytes;
    uramBytes = argUramBytes;
    bramBlockBytes = argBramBlockBytes;
    uramBlockBytes = argUramBlockBytes;
    maxFactor = argMaxFactor;
  }

  void runOnOperation() override {
    auto module = getOperation();
    if (bramBlockBytes == 0 || (uramBytes != 0 && uramBlockBytes == 0)) {
      module.emitError("invalid target memory primitive geometry");
      return signalPassFailure();
    }

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
    // Banking hints first: a lowering that shaped an access pattern knows
    // the banking it needs, which local distance analysis cannot always
    // recover (compact unrolled lanes, data-dependent gathers). A buffer
    // partitioned here is final -- the automatic search below skips any
    // type that already carries a partition layout.
    module.walk([&](hls::BufferLikeInterface buffer) {
      auto kindsAttr =
          buffer->getAttrOfType<ArrayAttr>(sar::kPartitionKindsAttr);
      auto factorsAttr =
          buffer->getAttrOfType<ArrayAttr>(sar::kPartitionFactorsAttr);
      if (!kindsAttr || !factorsAttr)
        return;
      SmallVector<PartitionKind> kinds;
      SmallVector<unsigned> factors;
      for (auto [kind, factor] :
           llvm::zip(kindsAttr.getAsRange<StringAttr>(),
                     factorsAttr.getAsRange<IntegerAttr>())) {
        kinds.push_back(llvm::StringSwitch<PartitionKind>(kind.getValue())
                            .Case("complete", PartitionKind::COMPLETE)
                            .Case("cyclic", PartitionKind::CYCLIC)
                            .Case("block", PartitionKind::BLOCK)
                            .Default(PartitionKind::NONE));
        factors.push_back(factor.getInt());
      }
      applyArrayPartition(buffer.getMemref(), factors, kinds,
                          /*updateFuncSignature=*/true, lutramMaxBits,
                          (uint64_t)lutramBytes * 8, &lutramBitsUsed);
      buffer->removeAttr(sar::kPartitionKindsAttr);
      buffer->removeAttr(sar::kPartitionFactorsAttr);
    });
    applyAutoArrayPartition(topFunc, lutramMaxBits, maxFactor,
                            (uint64_t)lutramBytes * 8, &lutramBitsUsed);
    // Banking is settled here, so this is the first point at which a bank's
    // real cost in whole primitives is known.
    rebalanceBankedStorage(module, bramBytes, uramBytes, bramBlockBytes,
                           uramBlockBytes);
    if (failed(verifyFinalMemoryBudget(module, bramBytes, uramBytes,
                                       lutramBytes, bramBlockBytes,
                                       uramBlockBytes)))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass>
sar::createArrayPartitionPass(unsigned lutramMaxBits, unsigned lutramBytes,
                              unsigned bramBytes, unsigned uramBytes,
                              unsigned bramBlockBytes, unsigned uramBlockBytes,
                              unsigned maxFactor) {
  return std::make_unique<ArrayPartition>(lutramMaxBits, lutramBytes, bramBytes,
                                          uramBytes, bramBlockBytes,
                                          uramBlockBytes, maxFactor);
}
