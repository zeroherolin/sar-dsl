//===- Analysis.h - dataflow analyses ---------------------------*- C++ -*-===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#ifndef SAR_DIALECT_HLS_IR_ANALYSIS_H
#define SAR_DIALECT_HLS_IR_ANALYSIS_H

#include "sar/Dialect/HLS/IR/HLS.h"
#include "sar/Dialect/HLS/IR/Utils.h"

namespace mlir {
namespace sar {

/// Node and Schedule complexity analysis.
class ComplexityAnalysis {
public:
  ComplexityAnalysis(func::FuncOp func);

  std::optional<unsigned long>
  getScheduleComplexity(hls::ScheduleOp schedule) const;
  std::optional<unsigned long> getNodeComplexity(hls::NodeOp node) const;

private:
  std::optional<unsigned long> calculateBlockComplexity(Block *block) const;
  llvm::SmallDenseMap<hls::NodeOp, unsigned long> nodeComplexityMap;
};

/// Record a pair of correlated nodes. The analysis reads one loop band per
/// node -- the node structure the dataflow passes build; a node with several
/// bands contributes only its first (see CorrelationAnalysis).
class Correlation {
public:
  Correlation(hls::NodeOp sourceNode, hls::NodeOp targetNode,
              hls::BufferLikeInterface sharedBuffer, Value sourceBuffer,
              Value targetBuffer, SmallVector<float> sourceScaleFactors,
              SmallVector<float> targetScaleFactors,
              SmallVector<int64_t> sourceToTargetMap,
              SmallVector<int64_t> targetToSourceMap)
      : sourceNode(sourceNode), targetNode(targetNode),
        sharedBuffer(sharedBuffer), sourceBuffer(sourceBuffer),
        targetBuffer(targetBuffer), sourceScaleFactors(sourceScaleFactors),
        targetScaleFactors(targetScaleFactors),
        sourceToTargetMap(sourceToTargetMap),
        targetToSourceMap(targetToSourceMap) {
    // Make sure the source-to-target and target-to-source map is valid.
    if (!sourceToTargetMap.empty()) {
      assert(getNodeLoopBand(targetNode).size() == sourceToTargetMap.size() &&
             "invalid source-to-target map size");

      for (auto i : llvm::enumerate(sourceToTargetMap))
        if (i.value() >= 0 && i.value() < (int64_t)targetToSourceMap.size())
          assert((int64_t)i.index() == targetToSourceMap[i.value()] &&
                 "mismatched source-to-target map");
        else
          assert(i.value() == -1 && "invalid source-to-target map");
    }

    if (!targetToSourceMap.empty()) {
      assert(getNodeLoopBand(sourceNode).size() == targetToSourceMap.size() &&
             "invalid target-to-source map size");

      for (auto i : llvm::enumerate(targetToSourceMap))
        if (i.value() >= 0 && i.value() < (int64_t)sourceToTargetMap.size())
          assert((int64_t)i.index() == sourceToTargetMap[i.value()] &&
                 "mismatched target-to-source map");
        else
          assert(i.value() == -1 && "invalid target-to-source map");
    }

    assert(getNodeLoopBand(sourceNode).size() == sourceScaleFactors.size() &&
           "invalid source strides size");
    assert(getNodeLoopBand(targetNode).size() == targetScaleFactors.size() &&
           "invalid target strides size");

    assert(sharedBuffer.getMemrefType() == sourceBuffer.getType() &&
           sharedBuffer.getMemrefType() == targetBuffer.getType() &&
           "source or target argument type not align with buffer type");
  }

  /// Get the shared buffer.
  hls::BufferLikeInterface getBuffer() const { return sharedBuffer; }

  /// Check whether a node is source node.
  bool isSourceNode(hls::NodeOp currentNode) const {
    assert((currentNode == sourceNode || currentNode == targetNode) &&
           "invalid input node");
    return currentNode == sourceNode;
  }

  // Get the correlated node of the current node.
  hls::NodeOp getCorrelatedNode(hls::NodeOp currentNode) const {
    return isSourceNode(currentNode) ? targetNode : sourceNode;
  }

  SmallVector<int64_t> getCorrelateMap(hls::NodeOp currentNode) const {
    if (isSourceNode(currentNode))
      return sourceToTargetMap;
    else
      return targetToSourceMap;
  }

  SmallVector<float> getScaleFactors(hls::NodeOp currentNode) const {
    if (isSourceNode(currentNode))
      return sourceScaleFactors;
    else
      return targetScaleFactors;
  }

  // Permute factors of the current node to the correlated node. If any of the
  // scaled factors is less than 1 and rounded to 1, return true.
  std::pair<FactorList, bool>
  permuteAndScaleFactors(hls::NodeOp currentNode, const FactorList &factors) {
    assert(factors.size() == getNodeLoopBand(currentNode).size() &&
           "invalid permutation factors");
    auto correlateMap = getCorrelateMap(currentNode);
    auto permutedFactors = permuteFactorsWithMap(factors, correlateMap);

    auto scaleFactors = getScaleFactors(getCorrelatedNode(currentNode));
    FactorList scaledFactors;
    bool roundedFlag = false;
    for (auto [factor, scaleFactor] :
         llvm::zip(permutedFactors, scaleFactors)) {
      auto scaledFactor = factor * scaleFactor;
      if (scaledFactor < 1.0) {
        scaledFactor = 1.0;
        roundedFlag = true;
      }
      // A non-integer ratio between the two bands cannot be expressed as an
      // unroll factor; round it and report, rather than assert on input.
      if (scaledFactor != (unsigned)scaledFactor) {
        scaledFactor = std::max(1.0f, std::floor(scaledFactor));
        roundedFlag = true;
      }
      scaledFactors.push_back(scaledFactor);
    }
    return {scaledFactors, roundedFlag};
  }

private:
  /// Permute "factors" with "map" and return the permuted factors. Note that
  /// "-1" in the permutation map indicates the output factor on the
  /// corresponding position is one.
  FactorList permuteFactorsWithMap(const FactorList &factors,
                                   SmallVectorImpl<int64_t> &map) const {
    FactorList permutedFactors;
    for (auto i : map) {
      if (i >= 0 && i < (int64_t)factors.size())
        permutedFactors.push_back(factors[i]);
      else if (i == -1)
        permutedFactors.push_back(1);
      else
        llvm_unreachable("invalid factors or map");
    }
    return permutedFactors;
  }

  hls::NodeOp sourceNode;
  hls::NodeOp targetNode;
  hls::BufferLikeInterface sharedBuffer;
  Value sourceBuffer;
  Value targetBuffer;
  SmallVector<float> sourceScaleFactors;
  SmallVector<float> targetScaleFactors;
  SmallVector<int64_t> sourceToTargetMap;
  SmallVector<int64_t> targetToSourceMap;
};

/// Correlations analysis between dataflow nodes.
class CorrelationAnalysis {
  using CorrelationList = SmallVector<Correlation, 4>;

public:
  CorrelationAnalysis(func::FuncOp func);

  CorrelationList getCorrelations(hls::NodeOp node) const {
    return nodeCorrelationMap.lookup(node);
  }

  auto begin() { return nodeCorrelationMap.begin(); }
  auto end() { return nodeCorrelationMap.end(); }

private:
  /// A MapVector: consumers build worklists by iterating this map, and the
  /// order has to be the walk order, not pointer order.
  llvm::MapVector<hls::NodeOp, CorrelationList> nodeCorrelationMap;
};

} // namespace sar
} // namespace mlir

#endif // SAR_DIALECT_HLS_IR_ANALYSIS_H
