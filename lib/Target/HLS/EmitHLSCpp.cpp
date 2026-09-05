//===- EmitHLSCpp.cpp - scheduled HLS IR to Vitis HLS C++ -----------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "sar/Target/HLS/EmitHLSCpp.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "sar/Dialect/HLS/IR/Utils.h"
#include "sar/Dialect/HLS/IR/Visitor.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::affine;
using namespace sar;
using namespace sar::hls;

static llvm::cl::opt<bool> emitVitisDirectives("emit-vitis-directives",
                                               llvm::cl::init(false));
static llvm::cl::opt<bool> enforceFalseDependency("enforce-false-dependency",
                                                  llvm::cl::init(false));
static llvm::cl::opt<unsigned>
    axiBusBits("axi-bus-bits",
               llvm::cl::desc("Data width, in bits, of the AXI master ports"),
               llvm::cl::init(512));
static llvm::cl::opt<unsigned>
    axiMaxBurstLength("axi-max-burst-length",
                      llvm::cl::desc("Maximum beats in one AXI burst"),
                      llvm::cl::init(64));
static llvm::cl::opt<unsigned> axiMaxOutstanding(
    "axi-max-outstanding",
    llvm::cl::desc("Maximum AXI bursts in flight per direction"),
    llvm::cl::init(16));

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// AXI master shaping for a buffer: how wide a beat is, how many beats a
/// burst carries, and how many bursts may be in flight.
struct AxiShape {
  unsigned widenBits;
  unsigned burstBeats;
  unsigned outstanding;
};

/// Derives the shaping from the buffer itself.
///
/// The innermost dimension is what the loops walk contiguously, so it bounds
/// both how far a beat can widen and how long a burst can usefully run. AXI4
/// bursts may not cross a 4 KiB boundary, so the physical beat width also
/// bounds their useful length.
static AxiShape getAxiShape(MemRefType type) {
  Type elementType = type.getElementType();
  unsigned elementBits = 0;
  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else if (auto vectorType = dyn_cast<VectorType>(elementType)) {
    Type scalarType = vectorType.getElementType();
    if (scalarType.isIntOrFloat())
      elementBits =
          vectorType.getNumElements() * scalarType.getIntOrFloatBitWidth();
  }
  if (elementBits == 0 || !type.hasStaticShape())
    return {0, 0, 0};

  // A single-element buffer moves no bulk data, so shaping it is noise; the
  // Vitis defaults serve. Real buffers always have a contiguous run to shape.
  if (type.getNumElements() == 1)
    return {0, 0, 0};

  // Widen while the beat still divides the contiguous run: a partial beat
  // at the end of every row would cost more than the widening gains.
  int64_t contiguous = type.getShape().back();
  unsigned widenBits = elementBits;
  while (!isa<VectorType>(elementType) && widenBits < axiBusBits &&
         (contiguous * elementBits) % (2 * widenBits) == 0)
    widenBits *= 2;

  // Vitis caps a burst at 256 beats. The 4 KiB boundary can be tighter for
  // wide ports, and a row shorter than either cap ends the burst first.
  auto beatsPerRow = (unsigned)(contiguous * elementBits / widenBits);
  unsigned boundaryBeats = std::max<unsigned>(1, 32768 / widenBits);
  unsigned burstBeats =
      std::min({256u, std::max(1u, axiMaxBurstLength.getValue()), boundaryBeats,
                std::max(1u, beatsPerRow)});
  unsigned outstanding =
      std::min(32u, std::max(1u, axiMaxOutstanding.getValue()));
  return {widenBits, burstBeats, outstanding};
}

static Type peelAxiType(Type type) {
  if (auto axiType = dyn_cast<AxiType>(type))
    return axiType.getElementType();
  return type;
}

static std::string getDataTypeName(Type type) {
  auto valType = peelAxiType(type);

  // Handle aggregated types, including memref, vector, and stream.
  if (auto arrayType = dyn_cast<MemRefType>(valType))
    return getDataTypeName(arrayType.getElementType());
  else if (auto streamType = dyn_cast<StreamType>(valType)) {
    std::string streamName = "hls::stream<";
    streamName += getDataTypeName(streamType.getElementType());
    streamName += ">";
    return streamName;
  } else if (auto vectorType = dyn_cast<VectorType>(valType)) {
    std::string vectorName = "hls::vector<";
    vectorName += getDataTypeName(vectorType.getElementType());
    vectorName += ", " + std::to_string(vectorType.getNumElements()) + ">";
    return vectorName;
  }

  // Handle scalar types, including float and integer.
  if (isa<Float32Type>(valType))
    return "float";
  else if (isa<Float64Type>(valType))
    return "double";
  else if (isa<IndexType>(valType))
    return "int64_t";
  else if (auto intType = dyn_cast<IntegerType>(valType)) {
    if (intType.getWidth() == 1)
      return "bool";
    std::string intName = "ap_";
    intName += intType.isUnsigned() ? "u" : "";
    intName += "int<" + std::to_string(intType.getWidth()) + ">";
    return intName;
  }
  return "unknown_type";
}

static std::string getUnsignedDataTypeName(Type type) {
  type = peelAxiType(type);
  if (isa<IndexType>(type))
    return "uint64_t";
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer)
    return "unknown_type";
  if (integer.getWidth() == 1)
    return "bool";
  return "ap_uint<" + std::to_string(integer.getWidth()) + ">";
}

static std::string getStorageTypeAndImpl(MemoryKind kind, std::string typeStr,
                                         std::string implStr) {
  switch (kind) {
  case MemoryKind::LUTRAM_1P:
    return typeStr + "=ram_1p " + implStr + "=lutram";
  case MemoryKind::LUTRAM_2P:
    return typeStr + "=ram_2p " + implStr + "=lutram";
  case MemoryKind::LUTRAM_S2P:
    return typeStr + "=ram_s2p " + implStr + "=lutram";
  case MemoryKind::BRAM_1P:
    return typeStr + "=ram_1p " + implStr + "=bram";
  case MemoryKind::BRAM_2P:
    return typeStr + "=ram_2p " + implStr + "=bram";
  case MemoryKind::BRAM_S2P:
    return typeStr + "=ram_s2p " + implStr + "=bram";
  case MemoryKind::BRAM_T2P:
    return typeStr + "=ram_t2p " + implStr + "=bram";
  case MemoryKind::URAM_1P:
    return typeStr + "=ram_1p " + implStr + "=uram";
  case MemoryKind::URAM_2P:
    return typeStr + "=ram_2p " + implStr + "=uram";
  case MemoryKind::URAM_S2P:
    return typeStr + "=ram_s2p " + implStr + "=uram";
  case MemoryKind::URAM_T2P:
    return typeStr + "=ram_t2p " + implStr + "=uram";
  default:
    return typeStr + "=ram_t2p " + implStr + "=bram";
  }
}

static std::string getVivadoStorageTypeAndImpl(MemoryKind kind) {
  switch (kind) {
  case MemoryKind::LUTRAM_1P:
    return "ram_1p_lutram";
  case MemoryKind::LUTRAM_2P:
    return "ram_2p_lutram";
  case MemoryKind::LUTRAM_S2P:
    return "ram_s2p_lutram";
  case MemoryKind::BRAM_1P:
    return "ram_1p_bram";
  case MemoryKind::BRAM_2P:
    return "ram_2p_bram";
  case MemoryKind::BRAM_S2P:
    return "ram_s2p_bram";
  case MemoryKind::BRAM_T2P:
    return "ram_t2p_bram";
  case MemoryKind::URAM_1P:
    return "ram_1p_uram";
  case MemoryKind::URAM_2P:
    return "ram_2p_uram";
  case MemoryKind::URAM_S2P:
    return "ram_s2p_uram";
  case MemoryKind::URAM_T2P:
    return "ram_t2p_uram";
  default:
    return "ram_t2p_bram";
  }
}

//===----------------------------------------------------------------------===//
// Port role analysis
//===----------------------------------------------------------------------===//

namespace {
/// Which way data crosses a port. A buffer that a callee only fills is an
/// output at the caller's boundary too, so the walk follows calls.
enum class PortRole { Unused, In, Out, InOut };
} // namespace

static const char *getPortRolePrefix(PortRole role) {
  switch (role) {
  case PortRole::Unused:
    return "unused";
  case PortRole::In:
    return "in";
  case PortRole::Out:
    return "out";
  case PortRole::InOut:
    return "inout";
  }
  return "arg";
}

/// Accumulates whether `value` is read and/or written. Recursion through
/// calls is bounded by `visited`, which keys the callee block arguments
/// already on the stack.
static void
collectAccess(Value value, bool &isRead, bool &isWritten,
              llvm::DenseSet<std::pair<Operation *, unsigned>> &visited) {
  for (auto &use : value.getUses()) {
    auto *owner = use.getOwner();
    unsigned idx = use.getOperandNumber();

    if (auto load = dyn_cast<AffineLoadOp>(owner)) {
      if (load.getMemRef() == value)
        isRead = true;
    } else if (auto store = dyn_cast<AffineStoreOp>(owner)) {
      isWritten |= store.getMemRef() == value;
      isRead |= store.getValueToStore() == value;
    } else if (auto load = dyn_cast<memref::LoadOp>(owner)) {
      if (load.getMemRef() == value)
        isRead = true;
    } else if (auto store = dyn_cast<memref::StoreOp>(owner)) {
      isWritten |= store.getMemRef() == value;
      isRead |= store.getValueToStore() == value;
    } else if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
      isRead |= copy.getSource() == value;
      isWritten |= copy.getTarget() == value;
    } else if (auto read = dyn_cast<vector::TransferReadOp>(owner)) {
      isRead |= read.getBase() == value;
    } else if (auto write = dyn_cast<vector::TransferWriteOp>(owner)) {
      isWritten |= write.getBase() == value;
      isRead |= write.getVector() == value;
    } else if (isa<StreamReadOp>(owner)) {
      isRead = true;
    } else if (auto write = dyn_cast<StreamWriteOp>(owner)) {
      isWritten |= write.getChannel() == value;
      isRead |= write.getValue() == value;
    } else if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal() || idx >= callee.getNumArguments()) {
        // Cannot see the callee: assume the worst rather than mislabel.
        isRead = isWritten = true;
        continue;
      }
      if (!visited.insert({callee.getOperation(), idx}).second)
        continue;
      collectAccess(callee.getArgument(idx), isRead, isWritten, visited);
    } else if (isa<func::ReturnOp>(owner)) {
      isRead = true;
    } else {
      // An operation the emitter does not model precisely; both directions
      // are possible, and over-reporting only costs a vaguer name.
      isRead = isWritten = true;
    }

    if (isRead && isWritten)
      return;
  }
}

static PortRole getPortRole(Value value) {
  bool isRead = false, isWritten = false;
  llvm::DenseSet<std::pair<Operation *, unsigned>> visited;
  collectAccess(value, isRead, isWritten, visited);
  if (isRead && isWritten)
    return PortRole::InOut;
  if (isWritten)
    return PortRole::Out;
  if (isRead)
    return PortRole::In;
  return PortRole::Unused;
}

/// AXI4-Stream has no address channel. A memref may use it only when the whole
/// call chain contains exactly one monotonic row-major sweep, with every
/// element consumed or produced once.
static bool isProvablySequentialStreamPort(Value value, PortRole role,
                                           unsigned &accesses) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto call = dyn_cast<func::CallOp>(owner)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal() || isNestedInLoop(call) ||
          use.getOperandNumber() >= callee.getNumArguments() ||
          !isProvablySequentialStreamPort(
              callee.getArgument(use.getOperandNumber()), role, accesses))
        return false;
      continue;
    }

    // The interface pass has already converted a proven affine sweep to
    // StreamRead/StreamWrite operations. There are no indices left to
    // inspect, but the recursive walk still verifies one FIFO direction.
    if (isa<StreamType>(value.getType())) {
      bool expected = (role == PortRole::In && isa<StreamReadOp>(owner)) ||
                      (role == PortRole::Out && isa<StreamWriteOp>(owner));
      if (!expected)
        return false;
      ++accesses;
      continue;
    }

    bool expected = (role == PortRole::In && isa<AffineLoadOp>(owner)) ||
                    (role == PortRole::Out && isa<AffineStoreOp>(owner));
    if (!expected || !isCompleteRowMajorSweep(owner, value))
      return false;
    ++accesses;
  }
  return true;
}

static bool isProvablySequentialStreamPort(Value value, PortRole role) {
  if (role != PortRole::In && role != PortRole::Out)
    return false;
  unsigned accesses = 0;
  return isProvablySequentialStreamPort(value, role, accesses) && accesses == 1;
}

/// Returns memrefs whose accesses in `loop` are proven independent across
/// iterations of that loop. This is useful for a carved arena: two logical
/// planes may share one physical pointer while occupying disjoint affine
/// ranges. Vitis otherwise treats the pointer as loop-carried memory and
/// raises the II even though no iteration can observe another's write.
static SmallVector<Value>
getInterIterationIndependentMemrefs(affine::AffineForOp loop) {
  SmallVector<Value> result;
  llvm::MapVector<Value, SmallVector<Operation *>> accesses;
  loop.getBody()->walk([&](Operation *operation) {
    if (auto read = dyn_cast<affine::AffineReadOpInterface>(operation))
      accesses[read.getMemRef()].push_back(operation);
    else if (auto write = dyn_cast<affine::AffineWriteOpInterface>(operation))
      accesses[write.getMemRef()].push_back(operation);
  });

  for (auto &entry : accesses) {
    bool independent = true;
    bool hasRead = false;
    bool hasWrite = false;
    auto &ops = entry.second;
    for (unsigned i = 0; i < ops.size() && independent; ++i) {
      auto *lhs = ops[i];
      bool lhsWrite = isa<affine::AffineWriteOpInterface>(lhs);
      hasWrite |= lhsWrite;
      hasRead |= !lhsWrite;
      for (unsigned j = 0; j < ops.size(); ++j) {
        auto *rhs = ops[j];
        if (lhs == rhs)
          continue;
        bool rhsWrite = isa<affine::AffineWriteOpInterface>(rhs);
        if (!lhsWrite && !rhsWrite)
          continue;

        unsigned common = affine::getNumCommonSurroundingLoops(*lhs, *rhs);
        if (common == 0) {
          independent = false;
          break;
        }
        affine::MemRefAccess lhsAccess(lhs), rhsAccess(rhs);
        auto dependence = affine::checkMemrefAccessDependence(
            lhsAccess, rhsAccess, common + 1);
        if (!affine::noDependence(dependence)) {
          independent = false;
          break;
        }
      }
    }
    if (independent && hasRead && hasWrite)
      result.push_back(entry.first);
  }

  // Outlined dataflow nodes may receive the same arena value for two
  // distinct arguments. Their block arguments therefore look unrelated to
  // local analysis even though the generated caller aliases them. Re-run the
  // same affine proof across such argument pairs, treating the two accesses
  // as views of one memref. This is what lets a carved read range and write
  // range share storage without forcing a false loop-carried dependence.
  auto func = loop->getParentOfType<func::FuncOp>();
  auto module = func ? func->getParentOfType<ModuleOp>() : ModuleOp();
  if (!func || !module)
    return result;

  SmallVector<std::pair<BlockArgument, BlockArgument>> aliases;
  module.walk([&](func::CallOp call) {
    auto callee = dyn_cast_or_null<func::FuncOp>(
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
    if (callee != func)
      return;
    for (unsigned i = 0; i < call.getNumOperands(); ++i)
      for (unsigned j = i + 1; j < call.getNumOperands(); ++j)
        if (call.getOperand(i) == call.getOperand(j) &&
            isa<MemRefType>(callee.getArgument(i).getType()) &&
            isa<MemRefType>(callee.getArgument(j).getType()))
          aliases.emplace_back(callee.getArgument(i), callee.getArgument(j));
  });

  for (auto [lhsArg, rhsArg] : aliases) {
    auto lhsIt = accesses.find(lhsArg), rhsIt = accesses.find(rhsArg);
    if (lhsIt == accesses.end() || rhsIt == accesses.end())
      continue;
    bool independent = true;
    bool hasRead = false, hasWrite = false;
    for (Operation *lhs : lhsIt->second)
      for (Operation *rhs : rhsIt->second) {
        bool lhsWrite = isa<affine::AffineWriteOpInterface>(lhs);
        bool rhsWrite = isa<affine::AffineWriteOpInterface>(rhs);
        hasRead |= !lhsWrite || !rhsWrite;
        hasWrite |= lhsWrite || rhsWrite;
        if (!lhsWrite && !rhsWrite)
          continue;
        unsigned common = affine::getNumCommonSurroundingLoops(*lhs, *rhs);
        if (common == 0) {
          independent = false;
          break;
        }
        affine::MemRefAccess lhsAccess(lhs), rhsAccess(rhs);
        rhsAccess.memref = lhsAccess.memref;
        auto forward = affine::checkMemrefAccessDependence(lhsAccess, rhsAccess,
                                                           common + 1);
        auto reverse = affine::checkMemrefAccessDependence(rhsAccess, lhsAccess,
                                                           common + 1);
        if (!affine::noDependence(forward) || !affine::noDependence(reverse)) {
          independent = false;
          break;
        }
      }
    if (independent && hasRead && hasWrite) {
      result.push_back(lhsArg);
      result.push_back(rhsArg);
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Some Base Classes
//===----------------------------------------------------------------------===//

namespace {
/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class HLSEmitterState {
public:
  explicit HLSEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;

  // Temporaries are numbered per function, so a reader sees `v3` in a
  // twenty-line node instead of `v1546` counted across the whole design.
  unsigned localNameIdx = 0;

  // Constant tables lifted to file scope, and the names they were given.
  DenseMap<Value, std::string> globalTables;
  DenseMap<Value, std::string> globalTableDeclNames;
  DenseMap<Value, std::pair<double, double>> linearTables;

  // Sub-functions renamed for readability: original symbol -> emitted name.
  llvm::StringMap<std::string> funcNames;

private:
  HLSEmitterState(const HLSEmitterState &) = delete;
  void operator=(const HLSEmitterState &) = delete;
};
} // namespace

namespace {
/// This is the base class for all of the HLSCpp Emitter components.
class HLSEmitterBase {
public:
  explicit HLSEmitterBase(HLSEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // Mutable emitter state.
  HLSEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  SmallString<8> addName(Value val, bool isPtr = false);

  /// Binds `val` to an explicit name, uniqued against names already handed
  /// out. Used for ports and buffers, whose names carry meaning.
  SmallString<8> addNamedValue(Value val, const Twine &base,
                               bool isPtr = false);

  SmallString<8> addAlias(Value val, Value alias);

  SmallString<8> getName(Value val);

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    } else
      return true;
  }

private:
  HLSEmitterBase(const HLSEmitterBase &) = delete;
  void operator=(const HLSEmitterBase &) = delete;
};
} // namespace

SmallString<8> HLSEmitterBase::addName(Value val, bool isPtr) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";

  valName += StringRef("v" + std::to_string(state.localNameIdx++));
  state.nameTable[val] = valName;

  return valName;
}

SmallString<8> HLSEmitterBase::addNamedValue(Value val, const Twine &base,
                                             bool isPtr) {
  assert(!isDeclared(val) && "has been declared before.");

  SmallString<8> valName;
  if (isPtr)
    valName += "*";
  base.toVector(valName);
  state.nameTable[val] = valName;

  return valName;
}

SmallString<8> HLSEmitterBase::addAlias(Value val, Value alias) {
  assert(!isDeclared(alias) && "has been declared before.");
  assert(isDeclared(val) && "hasn't been declared before.");

  auto valName = getName(val);
  state.nameTable[alias] = valName;

  return valName;
}

/// Formats a float with enough digits to round-trip exactly
/// (std::to_string truncates to 6 decimals, destroying twiddle factors
/// and small phase coefficients).
template <typename FloatT>
static std::string formatFloat(FloatT value, const char *format) {
  if (std::isnan(value))
    return "NAN";
  if (!std::isfinite(value))
    return value > 0 ? "INFINITY" : "-INFINITY";
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), format, value);
  return buffer;
}

static SmallString<8> getConstantString(Type type, Attribute attr) {
  SmallString<8> string;
  if (type.isInteger(1)) {
    auto value = cast<BoolAttr>(attr).getValue();
    string.append(value ? "true" : "false");

  } else if (type.isIndex()) {
    string.append("(int64_t)");
    auto value = cast<IntegerAttr>(attr).getInt();
    string.append(std::to_string(value));

  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    if (floatType.getWidth() == 32) {
      string.append("(float)");
      auto value = cast<FloatAttr>(attr).getValue().convertToFloat();
      string.append(formatFloat(value, "%.9g"));
    } else if (floatType.getWidth() == 64) {
      string.append("(double)");
      auto value = cast<FloatAttr>(attr).getValue().convertToDouble();
      string.append(formatFloat(value, "%.17g"));
    }
  } else if (auto intType = dyn_cast<IntegerType>(type)) {
    std::string signedness = "";
    if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
      signedness = "u";

    string.append("(");
    string.append("ap_" + signedness + "int<" +
                  std::to_string(intType.getWidth()) + ">)");

    if (intType.isSigned()) {
      auto value = cast<IntegerAttr>(attr).getValue().getSExtValue();
      string.append(std::to_string(value));
    } else if (intType.isUnsigned()) {
      auto value = cast<IntegerAttr>(attr).getValue().getZExtValue();
      string.append(std::to_string(value));
    } else {
      auto value = cast<IntegerAttr>(attr).getInt();
      string.append(std::to_string(value));
    }
  }
  return string;
}

SmallString<8> HLSEmitterBase::getName(Value val) {
  // For constant scalar operations, the constant number will be returned rather
  // than the value name.
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>())
    if (!isa<ShapedType>(constOp.getType())) {
      auto string = getConstantString(constOp.getType(), constOp.getValue());
      if (string.empty())
        constOp.emitOpError("constant has invalid value");
      return string;
    }
  return state.nameTable.lookup(val);
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public HLSEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(HLSEmitterState &state) : HLSEmitterBase(state) {}

  /// HLS dialect operation emitters.
  void emitConstBuffer(ConstBufferOp op);
  void emitStreamChannel(StreamOp op);
  void emitStreamRead(StreamReadOp op);
  void emitStreamWrite(StreamWriteOp op);
  void emitAxiPort(AxiPortOp op);
  template <typename AssignOpType> void emitAssign(AssignOpType op);
  void emitExtUI(arith::ExtUIOp op);
  template <typename CastOpType> void emitFPToInt(CastOpType op);
  void emitAffineSelect(hls::AffineSelectOp op);

  /// Control flow operation emitters.
  void emitCall(func::CallOp op);

  /// Loop-carried values (`iter_args`): one variable per carry, declared
  /// before the loop, aliased by the region argument and reassigned by the
  /// yield.
  LogicalResult emitLoopCarries(ValueRange results, ValueRange inits,
                                ValueRange iterArgs, Operation *op);
  void emitLoopCarryAssign(ValueRange results, ValueRange yields,
                           Operation *op);

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op);
  void emitScfIf(scf::IfOp op);
  void emitScfYield(scf::YieldOp op);

  /// Affine statement emitters.
  void emitAffineFor(AffineForOp op);
  void emitAffineIf(AffineIfOp op);
  void emitAffineParallel(AffineParallelOp op);
  void emitAffineApply(AffineApplyOp op);
  template <typename OpType>
  void emitAffineMaxMin(OpType op, const char *syntax);
  void emitAffineLoad(AffineLoadOp op);
  void emitAffineStore(AffineStoreOp op);
  void emitAffineYield(AffineYieldOp op);

  /// Vector-related statement emitters.
  void emitInsert(vector::InsertOp op);
  void emitExtract(vector::ExtractOp op);
  void emitFromElements(vector::FromElementsOp op);
  void emitTransferRead(vector::TransferReadOp op);
  void emitTransferWrite(vector::TransferWriteOp op);
  void emitBroadcast(vector::BroadcastOp);

  /// Memref-related statement emitters.
  template <typename OpType> void emitAlloc(OpType op);
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);
  void emitMemCpy(memref::CopyOp op);

  /// Standard expression emitters.
  void emitUnary(Operation *op, const char *syntax);
  void emitBinary(Operation *op, const char *syntax);
  void emitUnsignedBinary(Operation *op, const char *syntax);
  void emitUnsignedAssign(Operation *op);
  void emitFPToUnsigned(arith::FPToUIOp op);
  template <typename OpType> void emitMaxMin(OpType op, const char *syntax);
  template <typename OpType>
  void emitUnsignedMaxMin(OpType op, const char *syntax);

  /// Special expression emitters.
  void emitSelect(arith::SelectOp op);
  template <typename OpType> void emitConstant(OpType op);

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// Helper to get the string indices of TransferRead/Write operations.
  template <typename TransferOpType>
  SmallVector<SmallString<8>, 4> getTransferIndices(TransferOpType op);

  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 bool isRef = false);
  void emitValueDecl(Value val, bool isPtr = false);
  void emitArrayDecl(Value array);
  unsigned emitNestedLoopHeader(Value val);
  void emitNestedLoopFooter(unsigned rank);

  /// MLIR component and HLS C++ pragma emitters.
  void emitBlock(Block &block);
  void emitLoopDirectives(Operation *op);
  void emitArrayDirectives(Value memref, bool isInterface = false,
                           bool emitStorage = true);
  void emitAxiShape(MemRefType type);
  void emitFunctionDirectives(func::FuncOp func, ArrayRef<Value> portList);
  void emitFunction(func::FuncOp func);
  void emitInterfaceSchema(func::FuncOp func);

  /// Naming and layout helpers.
  StringRef getEmittedFuncName(StringRef symbol);
  void assignFuncNames(ArrayRef<func::FuncOp> funcs, func::FuncOp topFunc);
  SmallVector<Value, 8> assignPortNames(func::FuncOp func);
  void emitPortDecl(Value port, bool isConst);
  void emitFunctionSignature(func::FuncOp func, ArrayRef<Value> portList,
                             bool asPrototype);
  void nameLoopIV(Value iv);

  /// Constant table hoisting.
  void collectGlobalTables(ModuleOp module, ArrayRef<func::FuncOp> funcs);
  void emitGlobalTables();
  bool isConstParam(func::FuncOp func, unsigned idx) const;

  /// Loop induction variables are numbered per function.
  unsigned loopIVIdx = 0;

  /// Intermediate buffers are numbered per function.
  unsigned bufferIdx = 0;

  /// Ops whose results became file-scope tables; their bodies emit nothing.
  llvm::SetVector<Operation *> hoistedTables;

  /// Lifted tables that must stay writable because some callee parameter
  /// they reach could not be made const.
  DenseSet<Value> mutableTables;

  /// Callee parameters that only ever receive a file-scope const table.
  DenseMap<Operation *, llvm::BitVector> constParams;
};
} // namespace

//===----------------------------------------------------------------------===//
// AffineEmitter Class
//===----------------------------------------------------------------------===//

namespace {
class AffineExprEmitter : public HLSEmitterBase,
                          public AffineExprVisitor<AffineExprEmitter> {
public:
  using operand_range = Operation::operand_range;
  explicit AffineExprEmitter(HLSEmitterState &state, unsigned numDim,
                             operand_range operands)
      : HLSEmitterBase(state), numDim(numDim), operands(operands) {}

  void visitAddExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "+"); }
  void visitMulExpr(AffineBinaryOpExpr expr) { emitAffineBinary(expr, "*"); }
  void visitModExpr(AffineBinaryOpExpr expr) {
    emitAffineCall(expr, "sar_hls_mod");
  }
  void visitFloorDivExpr(AffineBinaryOpExpr expr) {
    emitAffineCall(expr, "sar_hls_floor_div");
  }
  void visitCeilDivExpr(AffineBinaryOpExpr expr) {
    emitAffineCall(expr, "sar_hls_ceil_div");
  }

  void visitConstantExpr(AffineConstantExpr expr) { os << expr.getValue(); }

  void visitDimExpr(AffineDimExpr expr) {
    os << getName(operands[expr.getPosition()]);
  }
  void visitSymbolExpr(AffineSymbolExpr expr) {
    os << getName(operands[numDim + expr.getPosition()]);
  }

  /// Affine expression emitters.
  void emitAffineCall(AffineBinaryOpExpr expr, const char *function) {
    os << function << "(";
    visit(expr.getLHS());
    os << ", ";
    visit(expr.getRHS());
    os << ")";
  }

  /// Emits `expr` in infix form. A negated right-hand side folds into unary
  /// minus or subtraction (`x * -1` as `-x`, `x + -3` as `x - 3`, and
  /// `x + y * -1` as `x - y`) so the generated C++ reads naturally.
  void emitAffineBinary(AffineBinaryOpExpr expr, const char *syntax) {
    os << "(";
    if (auto constRHS = dyn_cast<AffineConstantExpr>(expr.getRHS())) {
      if ((unsigned)*syntax == (unsigned)*"*" && constRHS.getValue() == -1) {
        os << "-";
        visit(expr.getLHS());
        os << ")";
        return;
      }
      if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() < 0) {
        visit(expr.getLHS());
        os << " - ";
        os << -constRHS.getValue();
        os << ")";
        return;
      }
    }
    if (auto binaryRHS = dyn_cast<AffineBinaryOpExpr>(expr.getRHS())) {
      if (auto constRHS = dyn_cast<AffineConstantExpr>(binaryRHS.getRHS())) {
        if ((unsigned)*syntax == (unsigned)*"+" && constRHS.getValue() == -1 &&
            binaryRHS.getKind() == AffineExprKind::Mul) {
          visit(expr.getLHS());
          os << " - ";
          visit(binaryRHS.getLHS());
          os << ")";
          return;
        }
      }
    }
    visit(expr.getLHS());
    os << " " << syntax << " ";
    visit(expr.getRHS());
    os << ")";
  }

  void emitAffineExpr(AffineExpr expr) { visit(expr); }

private:
  unsigned numDim;
  operand_range operands;
};
} // namespace

//===----------------------------------------------------------------------===//
// StmtVisitor, ExprVisitor, and PragmaVisitor Classes
//===----------------------------------------------------------------------===//

namespace {
class StmtVisitor : public HLSVisitorBase<StmtVisitor, bool> {
public:
  StmtVisitor(ModuleEmitter &emitter) : emitter(emitter) {}
  using HLSVisitorBase::visitOp;

  /// HLS dialect operations.
  bool visitOp(BufferOp op) {
    if (op.getDepth() == 1)
      return emitter.emitAlloc(op), true;
    return op.emitOpError("only support depth of 1"), false;
  }
  bool visitOp(ConstBufferOp op) { return emitter.emitConstBuffer(op), true; }
  bool visitOp(StreamOp op) { return emitter.emitStreamChannel(op), true; }
  bool visitOp(StreamReadOp op) { return emitter.emitStreamRead(op), true; }
  bool visitOp(StreamWriteOp op) { return emitter.emitStreamWrite(op), true; }
  bool visitOp(AxiBundleOp op) { return true; }
  bool visitOp(AxiPortOp op) { return emitter.emitAxiPort(op), true; }
  bool visitOp(AxiPackOp op) { return false; }
  bool visitOp(hls::AffineSelectOp op) {
    return emitter.emitAffineSelect(op), true;
  }

  /// Function operations.
  bool visitOp(func::CallOp op) { return emitter.emitCall(op), true; }
  bool visitOp(func::ReturnOp op) { return true; }

  /// SCF statements.
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::IfOp op) { return emitter.emitScfIf(op), true; };
  bool visitOp(scf::ParallelOp op) { return false; };
  bool visitOp(scf::ReduceOp op) { return false; };
  bool visitOp(scf::ReduceReturnOp op) { return false; };
  bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true; };

  /// Affine statements.
  bool visitOp(AffineForOp op) { return emitter.emitAffineFor(op), true; }
  bool visitOp(AffineIfOp op) { return emitter.emitAffineIf(op), true; }
  bool visitOp(AffineParallelOp op) {
    return emitter.emitAffineParallel(op), true;
  }
  bool visitOp(AffineApplyOp op) { return emitter.emitAffineApply(op), true; }
  bool visitOp(AffineMaxOp op) {
    return emitter.emitAffineMaxMin(op, "max"), true;
  }
  bool visitOp(AffineMinOp op) {
    return emitter.emitAffineMaxMin(op, "min"), true;
  }
  bool visitOp(AffineLoadOp op) { return emitter.emitAffineLoad(op), true; }
  bool visitOp(AffineStoreOp op) { return emitter.emitAffineStore(op), true; }
  bool visitOp(AffineVectorLoadOp op) { return false; }
  bool visitOp(AffineVectorStoreOp op) { return false; }
  bool visitOp(AffineYieldOp op) { return emitter.emitAffineYield(op), true; }

  /// Vector statements.
  bool visitOp(vector::InsertOp op) { return emitter.emitInsert(op), true; };
  bool visitOp(vector::ExtractOp op) { return emitter.emitExtract(op), true; };
  bool visitOp(vector::FromElementsOp op) {
    return emitter.emitFromElements(op), true;
  };
  bool visitOp(vector::TransferReadOp op) {
    return emitter.emitTransferRead(op), true;
  };
  bool visitOp(vector::TransferWriteOp op) {
    return emitter.emitTransferWrite(op), true;
  };
  bool visitOp(vector::BroadcastOp op) {
    return emitter.emitBroadcast(op), true;
  };

  /// Memref statements.
  bool visitOp(memref::AllocOp op) { return emitter.emitAlloc(op), true; }
  bool visitOp(memref::AllocaOp op) { return emitter.emitAlloc(op), true; }
  bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true; }
  bool visitOp(memref::StoreOp op) { return emitter.emitStore(op), true; }
  bool visitOp(memref::DeallocOp op) { return true; }
  bool visitOp(memref::CopyOp op) { return emitter.emitMemCpy(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(ModuleEmitter &emitter) : emitter(emitter) {}
  using HLSVisitorBase::visitOp;

  /// Unary expressions.
  bool visitOp(math::AbsIOp op) {
    return emitter.emitUnary(op, "std::abs"), true;
  }
  bool visitOp(math::AbsFOp op) {
    return emitter.emitUnary(op, "std::abs"), true;
  }
  bool visitOp(math::CeilOp op) {
    return emitter.emitUnary(op, "std::ceil"), true;
  }
  bool visitOp(math::CosOp op) {
    return emitter.emitUnary(op, "std::cos"), true;
  }
  bool visitOp(math::IsFiniteOp op) {
    return emitter.emitUnary(op, "std::isfinite"), true;
  }
  bool visitOp(math::SinOp op) {
    return emitter.emitUnary(op, "std::sin"), true;
  }
  bool visitOp(math::TanhOp op) {
    return emitter.emitUnary(op, "std::tanh"), true;
  }
  bool visitOp(math::SqrtOp op) {
    return emitter.emitUnary(op, "std::sqrt"), true;
  }
  bool visitOp(math::RsqrtOp op) {
    return emitter.emitUnary(op, "1.0 / std::sqrt"), true;
  }
  bool visitOp(math::ExpOp op) {
    return emitter.emitUnary(op, "std::exp"), true;
  }
  bool visitOp(math::Exp2Op op) {
    return emitter.emitUnary(op, "std::exp2"), true;
  }
  bool visitOp(math::LogOp op) {
    return emitter.emitUnary(op, "std::log"), true;
  }
  bool visitOp(math::Log2Op op) {
    return emitter.emitUnary(op, "std::log2"), true;
  }
  bool visitOp(math::Log10Op op) {
    return emitter.emitUnary(op, "std::log10"), true;
  }
  bool visitOp(arith::NegFOp op) { return emitter.emitUnary(op, "-"), true; }

  /// Float binary expressions.
  bool visitOp(arith::CmpFOp op);
  bool visitOp(arith::AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulFOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivFOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemFOp op) {
    return emitter.emitMaxMin(op, "std::fmod"), true;
  }
  bool visitOp(arith::MaximumFOp op) {
    return emitter.emitMaxMin(op, "sar_hls_maximum"), true;
  }
  bool visitOp(arith::MinimumFOp op) {
    return emitter.emitMaxMin(op, "sar_hls_minimum"), true;
  }
  bool visitOp(math::PowFOp op) {
    return emitter.emitMaxMin(op, "std::pow"), true;
  }
  bool visitOp(math::Atan2Op op) {
    return emitter.emitMaxMin(op, "std::atan2"), true;
  }

  /// Integer binary expressions.
  bool visitOp(arith::CmpIOp op);
  bool visitOp(arith::AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulIOp op) { return emitter.emitBinary(op, "*"), true; }
  bool visitOp(arith::DivSIOp op) { return emitter.emitBinary(op, "/"), true; }
  bool visitOp(arith::RemSIOp op) { return emitter.emitBinary(op, "%"), true; }
  bool visitOp(arith::DivUIOp op) {
    return emitter.emitUnsignedBinary(op, "/"), true;
  }
  bool visitOp(arith::RemUIOp op) {
    return emitter.emitUnsignedBinary(op, "%"), true;
  }
  bool visitOp(arith::XOrIOp op) { return emitter.emitBinary(op, "^"), true; }
  bool visitOp(arith::AndIOp op) { return emitter.emitBinary(op, "&"), true; }
  bool visitOp(arith::OrIOp op) { return emitter.emitBinary(op, "|"), true; }
  bool visitOp(arith::ShLIOp op) { return emitter.emitBinary(op, "<<"), true; }
  bool visitOp(arith::ShRSIOp op) { return emitter.emitBinary(op, ">>"), true; }
  bool visitOp(arith::ShRUIOp op) {
    return emitter.emitUnsignedBinary(op, ">>"), true;
  }
  bool visitOp(arith::MaxSIOp op) {
    return emitter.emitMaxMin(op, "std::max"), true;
  }
  bool visitOp(arith::MinSIOp op) {
    return emitter.emitMaxMin(op, "std::min"), true;
  }
  bool visitOp(arith::MaxUIOp op) {
    return emitter.emitUnsignedMaxMin(op, "std::max"), true;
  }
  bool visitOp(arith::MinUIOp op) {
    return emitter.emitUnsignedMaxMin(op, "std::min"), true;
  }

  /// Special expressions.
  bool visitOp(arith::SelectOp op) { return emitter.emitSelect(op), true; }
  bool visitOp(arith::ConstantOp op) { return emitter.emitConstant(op), true; }
  bool visitOp(arith::IndexCastOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::UIToFPOp op) {
    return emitter.emitUnsignedAssign(op), true;
  }
  bool visitOp(arith::SIToFPOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::FPToUIOp op) {
    return emitter.emitFPToUnsigned(op), true;
  }
  bool visitOp(arith::FPToSIOp op) { return emitter.emitFPToInt(op), true; }

  /// Width conversions emit as plain assignments: the emitted types
  /// (`ap_int<W>`, float/double) carry truncation and extension in their
  /// conversion operators. The one exception is zero-extension, where the
  /// signed source type would sign-extend, so it casts through unsigned.
  bool visitOp(arith::TruncIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::TruncFOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::ExtUIOp op) { return emitter.emitExtUI(op), true; }
  bool visitOp(arith::ExtSIOp op) { return emitter.emitAssign(op), true; }
  bool visitOp(arith::ExtFOp op) { return emitter.emitAssign(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

bool ExprVisitor::visitOp(arith::CmpFOp op) {
  switch (op.getPredicate()) {
  case arith::CmpFPredicate::OEQ:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpFPredicate::UNE:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpFPredicate::OLT:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpFPredicate::OLE:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpFPredicate::OGT:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpFPredicate::OGE:
    return emitter.emitBinary(op, ">="), true;
  default:
    emitter.emitError(op, "has an unsupported floating-point predicate; "
                          "unordered comparisons other than une require "
                          "explicit NaN handling");
    return false;
  }
}

bool ExprVisitor::visitOp(arith::CmpIOp op) {
  switch (op.getPredicate()) {
  case arith::CmpIPredicate::eq:
    return emitter.emitBinary(op, "=="), true;
  case arith::CmpIPredicate::ne:
    return emitter.emitBinary(op, "!="), true;
  case arith::CmpIPredicate::slt:
    return emitter.emitBinary(op, "<"), true;
  case arith::CmpIPredicate::ult:
    return emitter.emitUnsignedBinary(op, "<"), true;
  case arith::CmpIPredicate::sle:
    return emitter.emitBinary(op, "<="), true;
  case arith::CmpIPredicate::ule:
    return emitter.emitUnsignedBinary(op, "<="), true;
  case arith::CmpIPredicate::sgt:
    return emitter.emitBinary(op, ">"), true;
  case arith::CmpIPredicate::ugt:
    return emitter.emitUnsignedBinary(op, ">"), true;
  case arith::CmpIPredicate::sge:
    return emitter.emitBinary(op, ">="), true;
  case arith::CmpIPredicate::uge:
    return emitter.emitUnsignedBinary(op, ">="), true;
  }
  llvm_unreachable("covered switch");
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Definition
//===----------------------------------------------------------------------===//

/// HLS dialect operation emitters.
void ModuleEmitter::emitConstBuffer(ConstBufferOp op) {
  // Tables lifted to file scope are already defined and named; a local
  // copy would only duplicate the ROM.
  if (hoistedTables.count(op.getOperation()))
    return;

  emitConstant(op);
  emitArrayDirectives(op.getResult());
}

void ModuleEmitter::emitStreamChannel(StreamOp op) {
  indent();
  emitValue(op.getChannel());
  os << ";";
  os << "\n";
  indent() << "#pragma HLS stream variable=";
  emitValue(op.getChannel());
  os << " depth=" << op.getDepth() << "\n";
}

void ModuleEmitter::emitStreamRead(StreamReadOp op) {
  indent();
  if (op.getResult()) {
    emitValue(op.getResult());
    os << " = ";
  }
  emitValue(op.getChannel());
  os << ".read(";
  os << ");";
  os << "\n";
}

void ModuleEmitter::emitStreamWrite(StreamWriteOp op) {
  indent();
  emitValue(op.getChannel());
  os << ".write(";
  emitValue(op.getValue());
  os << ");";
  os << "\n";
}

/// Emits the AXI master shaping options for `type`, if it has any.
void ModuleEmitter::emitAxiShape(MemRefType type) {
  auto shape = getAxiShape(type);
  if (!shape.widenBits)
    return;
  os << " max_widen_bitwidth=" << shape.widenBits
     << " max_read_burst_length=" << shape.burstBeats
     << " max_write_burst_length=" << shape.burstBeats
     << " num_read_outstanding=" << shape.outstanding
     << " num_write_outstanding=" << shape.outstanding;
}

void ModuleEmitter::emitAxiPort(AxiPortOp op) {
  addAlias(op.getAxi(), op.getElement());

  // A design compiled for AXI4-Stream carries this on its top function. The
  // port types are unchanged -- a memref must be mapped to an MM bundle for
  // the IR to verify -- so the protocol is decided here, at the pragma.
  auto func = op->getParentOfType<func::FuncOp>();
  bool streamInterface = func && func->hasAttr("stream_interface");

  // A port the design both reads and writes is a scratch buffer that spilled
  // to DRAM, not a data stream: an AXI4-Stream is unidirectional and consumed
  // once, so `axis` on such a port cannot synthesize. Neither can a port the
  // design never touches -- a stream nobody drains would stall the host.
  // Only pure inputs and pure outputs stream; everything else stays
  // memory-mapped.
  auto role = getPortRole(op.getElement());
  bool requestedStream = op.getBundleType().getKind() == AxiKind::STREAM ||
                         (streamInterface && !op->hasAttr("hls.scratch"));
  bool streaming =
      requestedStream && isProvablySequentialStreamPort(op.getElement(), role);
  if (requestedStream && !streaming)
    emitError(op, "cannot use AXI4-Stream without one complete monotonic "
                  "row-major access sweep");

  indent() << "#pragma HLS interface";

  if (streaming) {
    // `axis` takes a bundle but no burst/latency shaping: those describe
    // addressed access to DRAM, which a stream port does not perform.
    os << " axis";
  } else if (op.getBundleType().getKind() == AxiKind::MM) {
    // A memory-mapped bundle is the contract with the host, so the port is
    // an AXI master whatever the compiler decided about keeping a copy on
    // chip. Downgrading it to `bram` because the buffer happened to fit
    // would emit `bundle=` on a mode that does not take one -- a pragma
    // Vitis rejects -- and would make the signature depend on placement.
    auto type = cast<MemRefType>(op.getElement().getType());
    os << " m_axi offset=slave depth=" << type.getNumElements();
    emitAxiShape(type);
  } else
    llvm_unreachable("AXI element type must be a memref or stream");

  os << " port=";
  emitValue(op.getElement());
  // Vitis HLS 2022.2 rejects `bundle=` on an AXI4-Stream pragma. Bundles
  // only name addressed AXI masters; stream references are independent.
  if (!streaming)
    os << " bundle=" << op.getBundleName();
  os << "\n";

  // ``offset=slave`` puts the base address in an AXI-Lite register.  Name
  // that register explicitly on the same control bundle as ``return``;
  // otherwise Vitis 2022.2 silently creates its default ``control`` bundle
  // beside the generated ``ctrl`` bundle, exposing two control interfaces.
  if (!streaming) {
    indent() << "#pragma HLS interface s_axilite port=";
    emitValue(op.getElement());
    os << " bundle=ctrl\n";
  }

  // An m_axi port is one physical interface. Array partitioning is an
  // on-chip banking directive; attaching it here makes Vitis split one
  // declared master into several RTL masters, violating the generated ABI.
  // Packing and burst shaping above provide external width. Any banking a
  // consumer needs belongs on a staged local buffer, never on this port.
  if (auto init = op->getAttrOfType<TypedAttr>("init_value")) {
    auto type = cast<MemRefType>(op.getElement().getType());
    Type scalarType = type.getElementType();
    unsigned lanes = 1;
    if (auto vector = dyn_cast<VectorType>(scalarType)) {
      scalarType = vector.getElementType();
      lanes = vector.getNumElements();
    }
    auto value = getConstantString(scalarType, init);
    if (value.empty()) {
      emitError(op, "has an initial value the C++ target cannot emit");
    } else {
      for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
        indent() << "for (int64_t init_i" << dim << " = 0; init_i" << dim
                 << " < " << extent << "; ++init_i" << dim << ") {\n";
        addIndent();
      }
      if (lanes > 1) {
        indent() << "for (int64_t init_lane = 0; init_lane < " << lanes
                 << "; ++init_lane) {\n";
        addIndent();
      }
      indent();
      emitValue(op.getElement());
      for (unsigned dim = 0; dim < type.getRank(); ++dim)
        os << "[init_i" << dim << "]";
      if (lanes > 1)
        os << "[init_lane]";
      os << " = " << value << ";\n";
      if (lanes > 1) {
        reduceIndent();
        indent() << "}\n";
      }
      for (unsigned dim = 0; dim < type.getRank(); ++dim) {
        reduceIndent();
        indent() << "}\n";
      }
    }
  }
  os << "\n";
}

/// Zero-extension. Integers are declared as signed `ap_int`, whose widening
/// conversion sign-extends; reinterpreting the source as unsigned at its own
/// width makes the assignment fill with zeros instead.
void ModuleEmitter::emitExtUI(arith::ExtUIOp op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = (ap_uint<" << op.getIn().getType().getIntOrFloatBitWidth() << ">)";
  emitValue(op.getIn(), rank);
  os << ";";
  os << "\n";
  emitNestedLoopFooter(rank);
}

template <typename CastOpType> void ModuleEmitter::emitFPToInt(CastOpType op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  auto type = cast<IntegerType>(op.getResult().getType());
  const char *prefix = type.isUnsigned() ? "uint" : "int";
  unsigned width = type.getWidth();
  if (width != 8 && width != 16 && width != 32 && width != 64)
    width = 64;
  indent();
  emitValue(op.getResult(), rank);
  os << " = (" << prefix << width << "_t)";
  emitValue(op.getIn(), rank);
  os << ";\n";
  emitNestedLoopFooter(rank);
}

template <typename AssignOpType>
void ModuleEmitter::emitAssign(AssignOpType op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getOperand(), rank);
  os << ";";
  os << "\n";
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitAffineSelect(hls::AffineSelectOp op) {
  indent();
  emitValue(op.getResult());
  os << " = (";
  auto constrSet = op.getIntegerSet();
  AffineExprEmitter constrEmitter(state, constrSet.getNumDims(),
                                  op.getOperands());

  // Emit all constraints.
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    constrEmitter.emitAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      os << " == 0";
    else
      os << " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      os << " && ";
  }
  os << ") ? ";
  emitValue(op.getTrueValue());
  os << " : ";
  emitValue(op.getFalseValue());
  os << ";";
  os << "\n";
}

/// Control flow operation emitters.
void ModuleEmitter::emitCall(func::CallOp op) {
  // Handle values returned by the callee.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  // Emit the function call.
  indent() << getEmittedFuncName(op.getCallee()) << "(";

  // Handle input arguments.
  unsigned argIdx = 0;
  for (auto arg : op.getOperands()) {
    emitValue(arg);

    if (argIdx++ != op.getNumOperands() - 1)
      os << ", ";
  }

  // Handle output arguments.
  for (auto result : op.getResults()) {
    // The address should be passed in for scalar result arguments.
    if (isa<ShapedType>(result.getType()))
      os << ", ";
    else
      os << ", &";

    emitValue(result);
  }

  os << ");";
  os << "\n";
}

/// Loop-carried values compile to ordinary variables: declared and
/// initialized ahead of the loop, read through the region's iter arg (an
/// alias of the same variable) inside it, and reassigned by the yield. The
/// loop's results are those variables after the last iteration.
///
/// A carried *array* cannot be reassigned in C, so it lives in its init
/// buffer instead: the iter arg and the result both alias it, and the
/// yield copies back into it unless the body already wrote it in place.
LogicalResult ModuleEmitter::emitLoopCarries(ValueRange results,
                                             ValueRange inits,
                                             ValueRange iterArgs,
                                             Operation *op) {
  for (auto [result, init, iterArg] : llvm::zip(results, inits, iterArgs)) {
    if (isa<MemRefType>(result.getType())) {
      addAlias(init, iterArg);
      addAlias(init, result);
      continue;
    }
    if (isa<ShapedType>(result.getType()) &&
        !isa<VectorType>(result.getType())) {
      emitError(op,
                "only scalar, vector and memref loop carries are supported");
      return failure();
    }
    indent();
    emitValueDecl(result);
    os << " = ";
    emitValue(init);
    os << ";\n";
    addAlias(result, iterArg);
  }
  return success();
}

/// Assigns what a loop yields into its carried variables. With more than
/// one scalar carry the values stage through temporaries first: a permuted
/// yield (`yield %b, %a`) must read this iteration's values, not the ones
/// the first assignment just overwrote. Array carries cannot stage, so a
/// permuted array yield is rejected rather than miscompiled.
void ModuleEmitter::emitLoopCarryAssign(ValueRange results, ValueRange yields,
                                        Operation *op) {
  for (auto [result, yielded] : llvm::zip(results, yields)) {
    if (!isa<MemRefType>(result.getType()))
      continue;
    if (getName(yielded) == getName(result))
      continue; // the body wrote the carry buffer in place
    for (Value other : results)
      if (other != result && isa<MemRefType>(other.getType()) &&
          getName(other) == getName(yielded)) {
        emitError(op, "swapped array carries are not supported; copy "
                      "through a scratch buffer inside the loop instead");
        return;
      }
    unsigned rank = emitNestedLoopHeader(result);
    indent();
    emitValue(result, rank);
    os << " = ";
    emitValue(yielded, rank);
    os << ";";
    os << "\n";
    emitNestedLoopFooter(rank);
  }

  SmallVector<unsigned> scalars;
  for (auto [idx, result] : llvm::enumerate(results))
    if (!isa<MemRefType>(result.getType()))
      scalars.push_back(idx);
  llvm::DenseMap<unsigned, std::string> temps;
  if (scalars.size() > 1) {
    for (unsigned idx : scalars) {
      auto temp = "carry" + std::to_string(state.localNameIdx++);
      indent() << getDataTypeName(yields[idx].getType()) << " " << temp
               << " = ";
      emitValue(yields[idx]);
      os << ";";
      os << "\n";
      temps[idx] = temp;
    }
  }
  for (unsigned idx : scalars) {
    indent();
    emitValue(results[idx]);
    os << " = ";
    if (auto it = temps.find(idx); it != temps.end())
      os << it->second;
    else
      emitValue(yields[idx]);
    os << ";";
    os << "\n";
  }
}

/// SCF statement emitters.
void ModuleEmitter::emitScfFor(scf::ForOp op) {
  if (failed(emitLoopCarries(op.getResults(), op.getInitArgs(),
                             op.getRegionIterArgs(), op)))
    return;

  indent() << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  nameLoopIV(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  os << "\n";

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent() << "}\n";
}

void ModuleEmitter::emitScfIf(scf::IfOp op) {
  // Declare all values returned by scf::YieldOp. They will be further handled
  // by the scf::YieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent() << "if (";
  emitValue(op.getCondition());
  os << ") {";
  os << "\n";

  addIndent();
  emitBlock(op.getThenRegion().front());
  reduceIndent();

  if (!op.getElseRegion().empty()) {
    indent() << "} else {\n";
    addIndent();
    emitBlock(op.getElseRegion().front());
    reduceIndent();
  }

  indent() << "}\n";
}

void ModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  if (auto forOp = dyn_cast<scf::ForOp>(op->getParentOp())) {
    emitLoopCarryAssign(forOp.getResults(), op.getOperands(), op);
    return;
  }

  if (auto parentOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      os << "\n";
      emitNestedLoopFooter(rank);
    }
  }
}

/// Affine statement emitters.
void ModuleEmitter::emitAffineFor(AffineForOp op) {
  if (failed(emitLoopCarries(op.getResults(), op.getInits(),
                             op.getRegionIterArgs(), op)))
    return;

  indent() << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  nameLoopIV(iterVar);
  os << " = ";
  auto lowerMap = op.getLowerBoundMap();
  AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                 op.getLowerBoundOperands());
  if (lowerMap.getNumResults() == 1)
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
  else {
    for (unsigned i = 0, e = lowerMap.getNumResults() - 1; i < e; ++i)
      os << "std::max(";
    lowerEmitter.emitAffineExpr(lowerMap.getResult(0));
    for (auto &expr : llvm::drop_begin(lowerMap.getResults(), 1)) {
      os << ", ";
      lowerEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  auto upperMap = op.getUpperBoundMap();
  AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                 op.getUpperBoundOperands());
  if (upperMap.getNumResults() == 1)
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
  else {
    for (unsigned i = 0, e = upperMap.getNumResults() - 1; i < e; ++i)
      os << "std::min(";
    upperEmitter.emitAffineExpr(upperMap.getResult(0));
    for (auto &expr : llvm::drop_begin(upperMap.getResults(), 1)) {
      os << ", ";
      upperEmitter.emitAffineExpr(expr);
      os << ")";
    }
  }
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += " << op.getStep() << ") {";
  os << "\n";

  addIndent();

  emitLoopDirectives(op);
  emitBlock(*op.getBody());
  reduceIndent();

  indent() << "}\n";
}

void ModuleEmitter::emitAffineIf(AffineIfOp op) {
  // Declare all values returned by AffineYieldOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  indent() << "if (";
  auto constrSet = op.getIntegerSet();
  AffineExprEmitter constrEmitter(state, constrSet.getNumDims(),
                                  op.getOperands());

  // Emit all constraints.
  unsigned constrIdx = 0;
  for (auto &expr : constrSet.getConstraints()) {
    constrEmitter.emitAffineExpr(expr);
    if (constrSet.isEq(constrIdx))
      os << " == 0";
    else
      os << " >= 0";

    if (constrIdx++ != constrSet.getNumConstraints() - 1)
      os << " && ";
  }
  os << ") {";
  os << "\n";

  addIndent();
  emitBlock(*op.getThenBlock());
  reduceIndent();

  if (op.hasElse()) {
    indent() << "} else {\n";
    addIndent();
    emitBlock(*op.getElseBlock());
    reduceIndent();
  }

  indent() << "}\n";
}

void ModuleEmitter::emitAffineParallel(AffineParallelOp op) {
  // Declare all values returned by AffineParallelOp. They will be further
  // handled by the AffineYieldOp emitter.
  for (auto result : op.getResults()) {
    if (!isDeclared(result)) {
      indent();
      if (isa<MemRefType>(result.getType()))
        emitArrayDecl(result);
      else
        emitValue(result);
      os << ";\n";
    }
  }

  auto steps = getIntArrayAttrValue(op, op.getStepsAttrName());
  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    indent() << "for (";
    auto iterVar = op.getBody()->getArgument(i);

    // Emit lower bound.
    nameLoopIV(iterVar);
    os << " = ";
    auto lowerMap = op.getLowerBoundsValueMap().getAffineMap();
    AffineExprEmitter lowerEmitter(state, lowerMap.getNumDims(),
                                   op.getLowerBoundsOperands());
    lowerEmitter.emitAffineExpr(lowerMap.getResult(i));
    os << "; ";

    // Emit upper bound.
    emitValue(iterVar);
    os << " < ";
    auto upperMap = op.getUpperBoundsValueMap().getAffineMap();
    AffineExprEmitter upperEmitter(state, upperMap.getNumDims(),
                                   op.getUpperBoundsOperands());
    upperEmitter.emitAffineExpr(upperMap.getResult(i));
    os << "; ";

    // Emit increase step.
    emitValue(iterVar);
    os << " += " << steps[i] << ") {";
    os << "\n";

    addIndent();
  }

  emitBlock(*op.getBody());

  for (unsigned i = 0, e = op.getNumDims(); i < e; ++i) {
    reduceIndent();

    indent() << "}\n";
  }
}

void ModuleEmitter::emitAffineApply(AffineApplyOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter(state, affineMap.getNumDims(), op.getOperands())
      .emitAffineExpr(affineMap.getResult(0));
  os << ";";
  os << "\n";
}

template <typename OpType>
void ModuleEmitter::emitAffineMaxMin(OpType op, const char *syntax) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getOperands());
  for (unsigned i = 0, e = affineMap.getNumResults() - 1; i < e; ++i)
    os << syntax << "(";
  affineEmitter.emitAffineExpr(affineMap.getResult(0));
  for (auto &expr : llvm::drop_begin(affineMap.getResults(), 1)) {
    os << ", ";
    affineEmitter.emitAffineExpr(expr);
    os << ")";
  }
  os << ";";
  os << "\n";
}

void ModuleEmitter::emitAffineLoad(AffineLoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << ";";
  os << "\n";
}

void ModuleEmitter::emitAffineStore(AffineStoreOp op) {
  indent();
  emitValue(op.getMemRef());
  auto affineMap = op.getAffineMap();
  AffineExprEmitter affineEmitter(state, affineMap.getNumDims(),
                                  op.getMapOperands());
  for (auto index : affineMap.getResults()) {
    os << "[";
    affineEmitter.emitAffineExpr(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  os << "\n";
}

// The parent op declares the values a yield produces ahead of its regions
// (`emitAffineIf`, and the carried variables of `emitAffineFor`), so the
// yield only assigns into names that already exist at the outer scope.
void ModuleEmitter::emitAffineYield(AffineYieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  if (auto forOp = dyn_cast<AffineForOp>(op->getParentOp())) {
    emitLoopCarryAssign(forOp.getResults(), op.getOperands(), op);
    return;
  }

  if (auto parentOp = dyn_cast<AffineIfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      os << "\n";
      emitNestedLoopFooter(rank);
    }
  } else if (auto parentOp = dyn_cast<AffineParallelOp>(op->getParentOp())) {
    indent() << "if (";
    unsigned ivIdx = 0;
    for (auto iv : parentOp.getBody()->getArguments()) {
      emitValue(iv);
      os << " == 0";
      if (ivIdx++ != parentOp.getBody()->getNumArguments() - 1)
        os << " && ";
    }
    os << ") {\n";

    // When all induction values are 0, generated values will be directly
    // assigned to the current results, correspondingly.
    addIndent();
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      os << " = ";
      emitValue(op.getOperand(resultIdx++), rank);
      os << ";";
      os << "\n";
      emitNestedLoopFooter(rank);
    }
    reduceIndent();

    indent() << "} else {\n";

    // Otherwise, generated values will be accumulated/reduced to the
    // current results with corresponding arith::AtomicRMWKind operations.
    addIndent();
    auto RMWAttrs =
        getIntArrayAttrValue(parentOp, parentOp.getReductionsAttrName());
    resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      unsigned rank = emitNestedLoopHeader(result);
      indent();
      emitValue(result, rank);
      switch ((arith::AtomicRMWKind)RMWAttrs[resultIdx]) {
      case (arith::AtomicRMWKind::addf):
      case (arith::AtomicRMWKind::addi):
        os << " += ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::assign):
        os << " = ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::maximumf):
      case (arith::AtomicRMWKind::maxs):
      case (arith::AtomicRMWKind::maxu):
        os << " = std::max(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (arith::AtomicRMWKind::minimumf):
      case (arith::AtomicRMWKind::mins):
      case (arith::AtomicRMWKind::minu):
        os << " = std::min(";
        emitValue(result, rank);
        os << ", ";
        emitValue(op.getOperand(resultIdx++), rank);
        os << ")";
        break;
      case (arith::AtomicRMWKind::mulf):
      case (arith::AtomicRMWKind::muli):
        os << " *= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::ori):
        os << " |= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      case (arith::AtomicRMWKind::andi):
        os << " &= ";
        emitValue(op.getOperand(resultIdx++), rank);
        break;
      default:
        // Emitting the assignment without a right-hand side would produce
        // C++ that silently computes something else, so refuse instead.
        op.emitOpError("unsupported reduction kind in affine.yield");
        os << " = /*unsupported*/";
        break;
      }
      os << ";";
      os << "\n";
      emitNestedLoopFooter(rank);
    }
    reduceIndent();

    indent() << "}\n";
  }
}

/// Helper to get the string indices of TransferRead/Write operations.
template <typename TransferOpType>
SmallVector<SmallString<8>, 4>
ModuleEmitter::getTransferIndices(TransferOpType op) {
  // Get the head indices of the transfer read/write.
  SmallVector<SmallString<8>, 4> indices;
  for (auto index : op.getIndices()) {
    assert(isDeclared(index) && "index has not been declared");
    indices.push_back(getName(index));
  }
  // Construct the physical indices.
  for (unsigned i = 0, e = op.getPermutationMap().getNumResults(); i < e; ++i) {
    auto expr = op.getPermutationMap().getResult(i);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr))
      indices[dimExpr.getPosition()] += " + iv" + std::to_string(i);
  }
  return indices;
}

/// Helper to get the TransferRead/Write condition.
template <typename TransferOpType>
static SmallString<16>
getTransferCondition(TransferOpType op,
                     const SmallVector<SmallString<8>, 4> &indices) {
  // Figure out whether the transfer read/write could be out of bound.
  SmallVector<unsigned, 4> outOfBoundDims;
  for (unsigned i = 0, e = op.getVectorType().getRank(); i < e; ++i)
    if (!op.isDimInBounds(i))
      outOfBoundDims.push_back(i);

  // Construct the condition of transfer if required.
  SmallString<16> condition;
  for (auto i : outOfBoundDims) {
    auto expr = op.getPermutationMap().getResult(i);
    if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
      auto pos = dimExpr.getPosition();
      condition += indices[pos];
      condition += " < " + std::to_string(op.getShapedType().getDimSize(pos));
      if (i != outOfBoundDims.back())
        condition += " && ";
    }
  }
  return condition;
}

/// Vector-related statement emitters.
void ModuleEmitter::emitInsert(vector::InsertOp op) {
  addAlias(op.getDest(), op.getResult());
  indent();
  emitValue(op.getDest());
  os << "[";
  {
    auto position = op.getMixedPosition()[0];
    if (auto attr = dyn_cast<Attribute>(position))
      os << cast<IntegerAttr>(attr).getInt();
    else
      emitValue(cast<Value>(position));
  }
  os << "] = ";
  emitValue(op.getValueToStore());
  os << ";";
  os << "\n";
}

void ModuleEmitter::emitExtract(vector::ExtractOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getSource());
  os << "[";
  auto position = op.getMixedPosition()[0];
  if (auto attr = dyn_cast<Attribute>(position))
    os << cast<IntegerAttr>(attr).getInt();
  else
    emitValue(cast<Value>(position));
  os << "];";
  os << "\n";
}

void ModuleEmitter::emitFromElements(vector::FromElementsOp op) {
  indent();
  emitValue(op.getResult());
  os << ";\n";
  for (auto [index, element] : llvm::enumerate(op.getElements())) {
    indent();
    emitValue(op.getResult());
    os << "[" << index << "] = ";
    emitValue(element);
    os << ";\n";
  }
}

void ModuleEmitter::emitTransferRead(vector::TransferReadOp op) {
  auto rank = emitNestedLoopHeader(op.getVector());
  auto indices = getTransferIndices(op);
  auto condition = getTransferCondition(op, indices);

  if (!condition.empty()) {
    indent() << "if (" << condition << ")\n";
    addIndent();
  }

  indent();
  emitValue(op.getVector(), rank);
  os << " = ";
  emitValue(op.getBase());
  for (auto index : indices)
    os << "[" << index << "]";
  os << ";";
  os << "\n";

  if (!condition.empty()) {
    reduceIndent();
    indent() << "else\n";
    addIndent();

    indent();
    emitValue(op.getVector(), rank);
    os << " = ";
    emitValue(op.getPadding());
    os << ";\n";
    reduceIndent();
  }
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitTransferWrite(vector::TransferWriteOp op) {
  auto rank = emitNestedLoopHeader(op.getVector());
  auto indices = getTransferIndices(op);
  auto condition = getTransferCondition(op, indices);

  if (!condition.empty()) {
    indent() << "if (" << condition << ")\n";
    addIndent();
  }

  indent();
  emitValue(op.getBase());
  for (auto index : indices)
    os << "[" << index << "]";
  os << " = ";
  emitValue(op.getVector(), rank);
  os << ";";
  os << "\n";

  if (!condition.empty())
    reduceIndent();
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitBroadcast(vector::BroadcastOp op) {
  auto rank = emitNestedLoopHeader(op.getVector());
  indent();
  emitValue(op.getVector(), rank);
  os << " = ";
  emitValue(op.getSource());

  // Figure out whether each dimension is broadcast or multicast.
  if (auto type = dyn_cast<ShapedType>(op.getSource().getType()))
    for (unsigned dim = 0, e = type.getRank(); dim < e; ++dim) {
      if (type.getDimSize(dim) == 1)
        os << "[0]";
      else
        os << "[iv" << dim + op.getType().getRank() - type.getRank() << "]";
    }

  os << ";";
  os << "\n";
  emitNestedLoopFooter(rank);
}

/// Memref-related statement emitters.
template <typename OpType> void ModuleEmitter::emitAlloc(OpType op) {
  // A declared result indicates that the memref is an output of the function,
  // and has been declared in the function signature.
  if (isDeclared(op.getResult()))
    return;

  // On-chip memory must have a static shape.
  if (!op.getType().hasStaticShape())
    emitError(op, "is unranked or has dynamic shape.");

  // Intermediate storage reads as storage, not as another temporary.
  if (!isDeclared(op.getResult()))
    addNamedValue(op.getResult(), "buf" + std::to_string(bufferIdx++));

  indent();
  emitArrayDecl(op.getResult());
  os << ";";
  os << "\n";
  emitArrayDirectives(op.getResult());

  auto buffer = dyn_cast<BufferOp>(op.getOperation());
  if (!buffer || !buffer.getInitValue())
    return;
  auto type = buffer.getType();
  auto init = getConstantString(type.getElementType(), *buffer.getInitValue());
  if (init.empty()) {
    emitError(buffer, "has an initial value the C++ target cannot emit");
    return;
  }
  for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
    indent() << "for (int64_t init_i" << dim << " = 0; init_i" << dim << " < "
             << extent << "; ++init_i" << dim << ") {\n";
    addIndent();
  }
  indent();
  emitValue(buffer.getResult());
  for (unsigned dim = 0; dim < type.getRank(); ++dim)
    os << "[init_i" << dim << "]";
  os << " = " << init << ";\n";
  for (unsigned dim = 0; dim < type.getRank(); ++dim) {
    reduceIndent();
    indent() << "}\n";
  }
}

void ModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  emitValue(op.getResult());
  os << " = ";
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  os << "\n";
}

void ModuleEmitter::emitStore(memref::StoreOp op) {
  indent();
  emitValue(op.getMemRef());
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << " = ";
  emitValue(op.getValueToStore());
  os << ";";
  os << "\n";
}

void ModuleEmitter::emitMemCpy(memref::CopyOp op) {
  indent() << "memcpy(";
  emitValue(op.getTarget());
  os << ", ";
  emitValue(op.getSource());
  os << ", ";

  auto type = cast<MemRefType>(op.getTarget().getType());
  os << type.getNumElements() << " * sizeof("
     << getDataTypeName(op.getTarget().getType()) << "));";
  os << "\n";
  os << "\n";
}

/// Standard expression emitters.
void ModuleEmitter::emitUnary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = " << syntax << "(";
  emitValue(op->getOperand(0), rank);
  os << ");";
  os << "\n";
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = ";
  emitValue(op->getOperand(0), rank);
  os << " " << syntax << " ";
  emitValue(op->getOperand(1), rank);
  os << ";";
  os << "\n";
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitUnsignedBinary(Operation *op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  std::string type = getUnsignedDataTypeName(op->getOperand(0).getType());
  os << " = (" << type << ")(";
  emitValue(op->getOperand(0), rank);
  os << ") " << syntax << " (" << type << ")(";
  emitValue(op->getOperand(1), rank);
  os << ");\n";
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitUnsignedAssign(Operation *op) {
  auto rank = emitNestedLoopHeader(op->getResult(0));
  indent();
  emitValue(op->getResult(0), rank);
  os << " = (" << getUnsignedDataTypeName(op->getOperand(0).getType()) << ")(";
  emitValue(op->getOperand(0), rank);
  os << ");\n";
  emitNestedLoopFooter(rank);
}

void ModuleEmitter::emitFPToUnsigned(arith::FPToUIOp op) {
  auto rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = (" << getUnsignedDataTypeName(op.getResult().getType()) << ")(";
  emitValue(op.getIn(), rank);
  os << ");\n";
  emitNestedLoopFooter(rank);
}

template <typename OpType>
void ModuleEmitter::emitUnsignedMaxMin(OpType op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  std::string type = getUnsignedDataTypeName(op.getLhs().getType());
  os << " = " << syntax << "((" << type << ")(";
  emitValue(op.getLhs(), rank);
  os << "), (" << type << ")(";
  emitValue(op.getRhs(), rank);
  os << "));\n";
  emitNestedLoopFooter(rank);
}

template <typename OpType>
void ModuleEmitter::emitMaxMin(OpType op, const char *syntax) {
  auto rank = emitNestedLoopHeader(op.getResult());
  indent();
  emitValue(op.getResult(), rank);
  os << " = " << syntax << "(";
  emitValue(op.getLhs(), rank);
  os << ", ";
  emitValue(op.getRhs(), rank);
  os << ");";
  os << "\n";
  emitNestedLoopFooter(rank);
}

/// Special expression emitters.
void ModuleEmitter::emitSelect(arith::SelectOp op) {
  unsigned rank = emitNestedLoopHeader(op.getResult());
  unsigned conditionRank = rank;
  if (!isa<ShapedType>(op.getCondition().getType()))
    conditionRank = 0;

  indent();
  emitValue(op.getResult(), rank);
  os << " = ";
  emitValue(op.getCondition(), conditionRank);
  os << " ? ";
  emitValue(op.getTrueValue(), rank);
  os << " : ";
  emitValue(op.getFalseValue(), rank);
  os << ";";
  os << "\n";
  emitNestedLoopFooter(rank);
}

template <typename OpType> void ModuleEmitter::emitConstant(OpType op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
    indent();
    Type resultType = op.getResult().getType();
    Type elementType;
    if (auto memref = dyn_cast<MemRefType>(resultType)) {
      emitArrayDecl(op.getResult());
      elementType = memref.getElementType();
    } else if (auto vector = dyn_cast<VectorType>(resultType)) {
      emitValue(op.getResult());
      elementType = vector.getElementType();
    } else {
      emitError(op, "has a dense constant with unsupported result type");
      return;
    }
    os << " = {";

    unsigned elementIdx = 0;
    for (auto element : denseAttr.template getValues<Attribute>()) {
      auto string = getConstantString(elementType, element);
      if (string.empty())
        op.emitOpError("constant has invalid value");
      os << string;
      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    os << "\n";
  } else
    emitError(op, "has unsupported constant type.");
}

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              bool isRef) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  // Emit the type of the value.
  os << getDataTypeName(val.getType()) << " ";
  if (isRef)
    os << "&";

  // Add the new value to nameTable and emit its name.
  os << addName(val, isPtr);
  for (unsigned i = 0; i < rank; ++i)
    os << "[iv" << i << "]";
}

/// Prints `<type> <name>` for a value being introduced. Unlike emitValue,
/// the type is always printed: a value may have been named ahead of its
/// declaration so that a signature and its body agree on the spelling.
void ModuleEmitter::emitValueDecl(Value val, bool isPtr) {
  os << getDataTypeName(val.getType()) << " ";
  if (isDeclared(val))
    os << getName(val);
  else
    os << addName(val, isPtr);
}

void ModuleEmitter::emitArrayDecl(Value array) {
  auto arrayType = cast<MemRefType>(peelAxiType(array.getType()));

  if (arrayType.hasStaticShape()) {
    emitValueDecl(array);
    for (auto &shape : arrayType.getShape())
      os << "[" << shape << "]";
  } else
    emitValueDecl(array, /*isPtr=*/true);
}

unsigned ModuleEmitter::emitNestedLoopHeader(Value val) {
  unsigned rank = 0;

  if (auto type = dyn_cast<MemRefType>(val.getType())) {
    if (!type.hasStaticShape()) {
      emitError(val.getDefiningOp(), "is unranked or has dynamic shape.");
      return 0;
    }

    // Declare a new array.
    if (!isDeclared(val)) {
      indent();
      emitArrayDecl(val);
      os << ";\n";
      // A vector value's lanes are one hardware word, all live at once, so
      // a complete partition is exact rather than a guess.
      if (isa<VectorType>(type)) {
        indent() << "#pragma HLS array_partition variable=";
        emitValue(val);
        os << " complete dim=0\n";
      }
    }

    // Create nested loop.
    unsigned dimIdx = 0;
    for (auto &shape : type.getShape()) {
      indent() << "for (int iv" << dimIdx << " = 0; ";
      os << "iv" << dimIdx << " < " << shape << "; ";
      os << "++iv" << dimIdx++ << ") {\n";

      addIndent();
      // The lanes of a vector value are one hardware word, so the sweep
      // over them unrolls fully by construction.
      if (isa<VectorType>(type))
        indent() << "#pragma HLS unroll\n";
    }
    rank = type.getRank();
  }

  return rank;
}

void ModuleEmitter::emitNestedLoopFooter(unsigned rank) {
  for (unsigned i = 0; i < rank; ++i) {
    reduceIndent();

    indent() << "}\n";
  }
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}

void ModuleEmitter::emitLoopDirectives(Operation *loop) {
  if (auto factor = loop->getAttrOfType<IntegerAttr>("hls.unroll_factor"))
    indent() << "#pragma HLS unroll factor=" << factor.getInt() << "\n";

  auto loopDirect = getLoopDirective(loop);
  if (!loopDirect)
    return;

  if (auto affineLoop = dyn_cast<affine::AffineForOp>(loop)) {
    for (Value memref : getInterIterationIndependentMemrefs(affineLoop)) {
      indent() << "#pragma HLS dependence variable=";
      emitValue(memref);
      os << " inter false\n";
    }
  }

  if (!hasParallelAttr(loop) && enforceFalseDependency.getValue())
    indent() << "#pragma HLS dependence false\n";

  if (loopDirect.getPipeline())
    indent() << "#pragma HLS pipeline II=" << loopDirect.getTargetII() << "\n";
}

void ModuleEmitter::emitArrayDirectives(Value memref, bool isInterface,
                                        bool emitStorage) {
  bool emitPragmaFlag = false;
  auto type = cast<MemRefType>(memref.getType());

  // Emit array_partition pragma(s).
  if (auto attr = dyn_cast<PartitionLayoutAttr>(type.getLayout())) {
    unsigned dim = 0;
    for (auto [kind, factor] :
         llvm::zip(attr.getKinds(), attr.getActualFactors(type.getShape()))) {
      if (factor != 1) {
        emitPragmaFlag = true;

        // Partition attributes never land on external (DRAM) buffers:
        // `hls-array-partition` skips them (`isExtBuffer`), so this pragma
        // only ever names an on-chip array.
        indent() << "#pragma HLS array_partition";
        os << " variable=";
        emitValue(memref);

        // Emit partition type. A complete partition splits every element
        // into its own register, so it takes no factor -- Vitis 2022.2
        // warns and ignores one that is given (HLS 207-5529).
        os << " " << stringifyPartitionKind(kind);
        if (kind != hls::PartitionKind::COMPLETE)
          os << " factor=" << factor;

        // Vitis HLS automatically collapses the first dimension when its
        // size is one, so the directive dimension shifts to match.
        auto directiveDim = dim + 1;
        if (emitVitisDirectives.getValue())
          if (type.getShape().front() == 1)
            directiveDim = dim;
        os << " dim=" << directiveDim << "\n";
      }
      ++dim;
    }
  }

  // Emit resource pragma when the array is not DRAM kind and is not fully
  // partitioned.
  if (!isInterface && emitStorage) {
    auto kind = getMemoryKind(type);
    if (kind != MemoryKind::DRAM && !isFullyPartitioned(type)) {
      emitPragmaFlag = true;

      if (emitVitisDirectives.getValue()) {
        indent() << "#pragma HLS bind_storage";
        os << " variable=";
        emitValue(memref);
        os << " " << getStorageTypeAndImpl(kind, "type", "impl");
      } else {
        indent() << "#pragma HLS resource";
        os << " variable=";
        emitValue(memref);
        os << " core=" << getVivadoStorageTypeAndImpl(kind);
      }
      os << "\n";
    }
  }

  // A trailing blank line separates the pragma block from the code.
  if (emitPragmaFlag)
    os << "\n";
}

void ModuleEmitter::emitFunctionDirectives(func::FuncOp func,
                                           ArrayRef<Value> portList) {
  // Only the top function should emit interface pragmas.
  if (hasTopFuncAttr(func)) {
    if (auto module = func->getParentOfType<ModuleOp>())
      for (auto helper : module.getOps<func::FuncOp>())
        if (helper->hasAttr("hls.shared_instance"))
          indent() << "#pragma HLS allocation function instances="
                   << getEmittedFuncName(helper.getName()) << " limit=1\n";

    // Inside a dataflow region every array is checked as a channel: one
    // writer, one reader. A read-only port is not a channel -- its value
    // is a run-long constant -- and several nodes reading it is the
    // normal fan-out of an input raster. `stable` states exactly that
    // contract, and without it Vitis rejects the design (HLS 200-779).
    auto funcDirective = getFuncDirective(func);
    bool isDataflow = funcDirective && funcDirective.getDataflow();

    indent() << "#pragma HLS interface s_axilite port=return bundle=ctrl\n";
    for (auto &port : portList) {
      // Axi ports are handled separately.
      if (isa<AxiType>(port.getType()))
        continue;

      // Handle normal memref or stream types.
      if (isa<MemRefType, StreamType>(port.getType())) {
        indent() << "#pragma HLS interface";

        if (auto memrefPortType = dyn_cast<MemRefType>(port.getType())) {
          if (getMemoryKind(memrefPortType) == MemoryKind::DRAM) {
            os << " m_axi offset=slave depth="
               << memrefPortType.getNumElements();
            emitAxiShape(memrefPortType);
          } else
            os << " bram";
        } else
          os << " axis";

        os << " port=";
        emitValue(port);
        os << "\n";

        if (auto memrefPortType = dyn_cast<MemRefType>(port.getType());
            memrefPortType &&
            getMemoryKind(memrefPortType) == MemoryKind::DRAM) {
          indent() << "#pragma HLS interface s_axilite port=";
          emitValue(port);
          os << " bundle=ctrl\n";
        }

        if (isDataflow && isa<MemRefType>(port.getType()) &&
            getPortRole(port) == PortRole::In) {
          indent() << "#pragma HLS stable variable=";
          emitValue(port);
          os << "\n";
        }

      } else {
        // Scalar arguments are always AXI-Lite ports.
        auto name = getName(port);
        if (name.front() == "*"[0])
          name.erase(name.begin());
        indent() << "#pragma HLS interface s_axilite port=" << name
                 << " bundle=ctrl\n";
      }

      if (isa<MemRefType>(port.getType()))
        emitArrayDirectives(port, true);
    }

    // Vitis rejects HLS pragmas at file scope, but a directive inside the top
    // function may name a file-scope constant. Keep hoisted ROMs readable
    // without dropping the placement and banking the IR selected.
    for (Operation *operation : hoistedTables)
      emitArrayDirectives(cast<ConstBufferOp>(operation).getResult(),
                          /*isInterface=*/false,
                          /*emitStorage=*/false);
    // Hoisted ROMs are immutable globals shared by several outlined stages.
    // Mark them stable in the top dataflow region so Vitis does not treat a
    // global read as an implicit process dependency (HLS 214-113/200-471).
    if (isDataflow)
      for (Operation *operation : hoistedTables) {
        indent() << "#pragma HLS stable variable=";
        emitValue(cast<ConstBufferOp>(operation).getResult());
        os << "\n";
      }
  }

  if (func->getAttr("inline"))
    indent() << "#pragma HLS inline\n";

  if (auto funcDirect = getFuncDirective(func)) {
    if (funcDirect.getPipeline()) {
      indent() << "#pragma HLS pipeline II=" << funcDirect.getTargetInterval()
               << "\n";
      os << "\n";
    } else if (funcDirect.getDataflow()) {
      indent() << "#pragma HLS dataflow\n";
      os << "\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// Naming, ordering, and constant hoisting
//===----------------------------------------------------------------------===//

/// Describes a sub-function by the arithmetic it actually contains.
///
/// This is all the lowered IR still knows: by the time a design reaches the
/// emitter the pipeline has erased which algorithm stage a node came from,
/// so a name like `stolt` cannot be recovered here without guessing. Only
/// classifications that follow directly from the operations present are
/// used; everything else falls back to the bare stage index.
///
/// `twiddleArguments` marks the parameters that receive an FFT twiddle
/// table, which is the one stage identity the lowering still records by
/// name. The walk follows calls, because a stage that drives its work
/// through helpers holds no arithmetic of its own: an FFT stage is a loop
/// over butterfly calls, and classifying it on its own body alone would
/// name it after the one thing it does not do.
static StringRef
classifyStage(func::FuncOp func,
              const llvm::SmallDenseSet<Operation *> &twiddleReaders) {
  if (func->hasAttr("hls.shared_instance"))
    return "engine";
  if (twiddleReaders.contains(func))
    return "fft";

  bool hasArith = false, hasExt = false, hasTrunc = false;
  bool hasSin = false, hasCos = false;

  llvm::SmallDenseSet<Operation *> visited;
  std::function<void(func::FuncOp)> scan = [&](func::FuncOp callee) {
    if (!visited.insert(callee).second)
      return;
    callee.walk([&](Operation *op) {
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (auto target = dyn_cast_or_null<func::FuncOp>(
                SymbolTable::lookupNearestSymbolFrom(call,
                                                     call.getCalleeAttr())))
          scan(target);
        return;
      }
      if (isa<math::SinOp>(op))
        hasSin = true;
      if (isa<math::CosOp>(op))
        hasCos = true;

      if (isa<arith::ExtFOp>(op)) {
        hasExt = true;
        return;
      }
      if (isa<arith::TruncFOp>(op)) {
        hasTrunc = true;
        return;
      }
      if (isa<arith::ConstantOp, arith::IndexCastOp>(op))
        return;
      if (isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
        hasArith = true;
    });
  };
  scan(func);

  // A node that computes a sine and a cosine is applying a phase factor.
  if (hasSin && hasCos)
    return "phase";
  if (hasArith)
    return "";
  if (hasExt && !hasTrunc)
    return "widen";
  if (hasTrunc && !hasExt)
    return "narrow";
  if (!hasExt && !hasTrunc)
    return "copy";
  return "";
}

/// Functions reached, directly or through further calls, by a value the FFT
/// lowering emitted as a twiddle table.
static llvm::SmallDenseSet<Operation *>
findTwiddleReaders(func::FuncOp topFunc) {
  llvm::SmallDenseSet<Operation *> readers;
  if (!topFunc)
    return readers;

  std::function<void(func::FuncOp, const llvm::SmallDenseSet<unsigned> &)>
      propagate = [&](func::FuncOp caller,
                      const llvm::SmallDenseSet<unsigned> &tableArgs) {
        caller.walk([&](func::CallOp call) {
          auto callee = dyn_cast_or_null<func::FuncOp>(
              SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
          if (!callee)
            return;
          llvm::SmallDenseSet<unsigned> calleeArgs;
          for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
            bool isTable = false;
            if (auto buffer = operand.getDefiningOp<ConstBufferOp>()) {
              if (auto source = buffer.getSourceName())
                isTable = source->contains("fft_twiddle");
            } else if (auto argument = dyn_cast<BlockArgument>(operand)) {
              isTable = argument.getOwner()->getParentOp() == caller &&
                        tableArgs.contains(argument.getArgNumber());
            }
            if (isTable)
              calleeArgs.insert(index);
          }
          if (calleeArgs.empty())
            return;
          readers.insert(callee);
          propagate(callee, calleeArgs);
        });
      };
  propagate(topFunc, {});
  return readers;
}

StringRef ModuleEmitter::getEmittedFuncName(StringRef symbol) {
  auto it = state.funcNames.find(symbol);
  return it == state.funcNames.end() ? symbol : StringRef(it->second);
}

/// Sub-functions arrive numbered in the order the dataflow pass happened to
/// create them, which runs roughly backwards against execution. Renumbering
/// them in call order lets the reader follow the pipeline top to bottom.
void ModuleEmitter::assignFuncNames(ArrayRef<func::FuncOp> funcs,
                                    func::FuncOp topFunc) {
  if (!topFunc)
    return;

  // Order sub-functions by first call from the top function, then by any
  // remaining calls, so nested helpers follow their caller.
  SmallVector<func::FuncOp, 16> ordered;
  llvm::SmallDenseSet<Operation *> seen;
  std::function<void(func::FuncOp)> visit = [&](func::FuncOp caller) {
    caller.walk([&](func::CallOp call) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee == topFunc || !seen.insert(callee).second)
        return;
      ordered.push_back(callee);
      visit(callee);
    });
  };
  visit(topFunc);

  // Anything never reached from the top keeps a slot at the end.
  for (auto func : funcs)
    if (func != topFunc && seen.insert(func).second)
      ordered.push_back(func);

  auto twiddleReaders = findTwiddleReaders(topFunc);
  unsigned stageIdx = 0;
  for (auto func : ordered) {
    SmallString<32> name;
    llvm::raw_svector_ostream nameOs(name);
    nameOs << topFunc.getName() << "_s";
    if (stageIdx < 10)
      nameOs << "0";
    nameOs << stageIdx++;
    if (auto role = classifyStage(func, twiddleReaders); !role.empty())
      nameOs << "_" << role;
    state.funcNames[func.getName()] = std::string(name);
  }
}

/// Whether a user-supplied port name can be emitted verbatim: a C identifier
/// that is neither a C++ keyword nor shaped like the names the emitter hands
/// out itself (`v3`, `i0`, `buf1`, `in0`, `out0`, ...), which are not uniqued
/// against it.
static bool isUsablePortName(StringRef name) {
  if (name.empty() || (!llvm::isAlpha(name.front()) && name.front() != '_'))
    return false;
  if (!llvm::all_of(name, [](char c) { return llvm::isAlnum(c) || c == '_'; }))
    return false;

  static const llvm::StringSet<> keywords = {
      "alignas",   "alignof",  "asm",     "auto",     "bool",     "break",
      "case",      "catch",    "char",    "class",    "const",    "constexpr",
      "continue",  "default",  "delete",  "do",       "double",   "else",
      "enum",      "explicit", "extern",  "false",    "float",    "for",
      "friend",    "goto",     "if",      "inline",   "int",      "long",
      "namespace", "new",      "nullptr", "operator", "private",  "protected",
      "public",    "register", "return",  "short",    "signed",   "sizeof",
      "static",    "struct",   "switch",  "template", "this",     "throw",
      "true",      "try",      "typedef", "typeid",   "typename", "union",
      "unsigned",  "using",    "virtual", "void",     "volatile", "while"};
  // Vitis carries top-level argument names into Verilog and VHDL. Keep common
  // HDL keywords out of emitted port names so every generated view shares a
  // valid, stable ABI.
  static const llvm::StringSet<> hdlKeywords = {
      "architecture",  "array",     "assert",    "assign",    "begin",
      "block",         "body",      "buffer",    "case",      "component",
      "configuration", "constant",  "context",   "default",   "design",
      "entity",        "event",     "file",      "function",  "generate",
      "generic",       "genvar",    "in",        "initial",   "inout",
      "input",         "integer",   "is",        "library",   "literal",
      "localparam",    "map",       "module",    "out",       "output",
      "package",       "parameter", "port",      "primitive", "procedure",
      "process",       "property",  "protected", "range",     "real",
      "record",        "reg",       "register",  "report",    "sequence",
      "shared",        "signal",    "specify",   "specparam", "subtype",
      "table",         "task",      "time",      "transport", "type",
      "units",         "use",       "variable",  "wait",      "wire"};
  if (keywords.contains(name) || hdlKeywords.contains(name.lower()))
    return false;

  // Reserved generated-name shapes: a known stem followed by digits only.
  StringRef digits = name;
  for (StringRef stem :
       {"v", "i", "buf", "in", "out", "arg", "out_scalar", "iv", "val"})
    if (name.starts_with(stem) && name.size() > stem.size()) {
      digits = name.drop_front(stem.size());
      if (llvm::all_of(digits, [](char c) { return llvm::isDigit(c); }))
        return false;
    }
  return true;
}

/// Names the ports of `func` and returns them in signature order. Naming
/// happens before any emission so a prototype and its definition agree.
///
/// Arguments covered by the `sar.arg_names` attribute -- the kernel's Python
/// parameter names, split per plane by `sar-decomplexify` -- keep those
/// names; everything else (out-params, the DRAM scratch, sub-function ports)
/// falls back to the role-based scheme.
SmallVector<Value, 8> ModuleEmitter::assignPortNames(func::FuncOp func) {
  SmallVector<Value, 8> portList;
  unsigned roleIdx[4] = {0, 0, 0, 0};
  unsigned scalarIdx = 0;

  auto argNames = func->getAttrOfType<ArrayAttr>("sar.arg_names");

  // The names are emitted verbatim, so a duplicate -- a complex `raw`
  // splitting into `raw_re` beside a parameter already called `raw_re` --
  // would redeclare a port. Fall back to the generated scheme entirely
  // rather than emit C++ that does not compile.
  if (argNames) {
    llvm::StringSet<> seen;
    for (auto attr : argNames)
      if (auto str = dyn_cast<StringAttr>(attr))
        if (!seen.insert(str.getValue()).second) {
          argNames = nullptr;
          break;
        }
  }

  // Each port must be a distinct value. `verifyHLSCppTarget` rejects an
  // aliased signature before any C++ is written, so reaching this with a
  // duplicate means the module was emitted without that check.
  DenseSet<Value> seenPorts;

  auto nameOne = [&](Value port, bool isResult, StringRef given) {
    portList.push_back(port);

    assert(seenPorts.insert(port).second &&
           "aliased port reached emission; verifyHLSCppTarget should have "
           "rejected it");
    (void)seenPorts;

    // Prototype and definition both ask for the port list; the names are
    // handed out once and reused so the two always agree.
    if (isDeclared(port))
      return;

    auto type = peelAxiType(port.getType());
    bool isArray = isa<MemRefType>(type);
    bool isStream = isa<StreamType>(type);

    // A returned scalar is passed by pointer, which is how Vitis HLS
    // spells an output.
    bool isPtr = isResult && !isa<ShapedType>(port.getType());
    if (isArray)
      if (auto memref = dyn_cast<MemRefType>(type);
          memref && !memref.hasStaticShape())
        isPtr = true;

    if (!given.empty() && isUsablePortName(given)) {
      addNamedValue(port, given, isPtr);
      return;
    }

    if (!isArray && !isStream) {
      addNamedValue(port,
                    isResult ? "out_scalar" + std::to_string(scalarIdx++)
                             : "arg" + std::to_string(scalarIdx++),
                    isPtr);
      return;
    }

    // Internal DRAM arenas are implementation storage, not logical results.
    // Give them an explicit scratch name so the top signature and manifest
    // read like a hand-written design instead of exposing generic `inoutN`
    // placeholders beside the algorithm's public ports.
    AxiPortOp axiPort;
    if (hasTopFuncAttr(func)) {
      func.walk([&](AxiPortOp candidate) {
        if (candidate.getAxi() == port)
          axiPort = candidate;
      });
      if (axiPort && axiPort->hasAttr("hls.scratch")) {
        addNamedValue(port, "scratch" + std::to_string(roleIdx[3]++), isPtr);
        return;
      }
    }

    // The design function receives an AxiType wrapper, while the actual
    // reads/writes are uses of the element produced by hls.axi.port.  Role
    // analysis on the wrapper sees that plumbing as unknown and labels a
    // write-only result ``inout``. Follow the port element so public outputs
    // receive the same clear ``outN`` name their manifest already reports.
    auto role = isResult ? PortRole::Out
                         : getPortRole(axiPort ? axiPort.getElement() : port);
    if (role == PortRole::Unused)
      role = PortRole::In;

    // Depth-one boolean channels are the dataflow handshake tokens, not
    // payload; saying so beats calling them streams.
    StringRef stem = "";
    if (isStream) {
      auto streamType = cast<StreamType>(type);
      stem = streamType.getElementType().isInteger(1) ? "sync_" : "strm_";
    }

    SmallString<24> name;
    llvm::raw_svector_ostream nameOs(name);
    nameOs << stem << getPortRolePrefix(role) << roleIdx[(unsigned)role]++;
    addNamedValue(port, name, isPtr);
  };

  for (auto [idx, arg] : llvm::enumerate(func.getArguments())) {
    StringRef given;
    if (argNames && idx < argNames.size())
      if (auto str = dyn_cast<StringAttr>(argNames[idx]))
        given = str.getValue();
    nameOne(arg, /*isResult=*/false, given);
  }

  if (!func.getBlocks().empty())
    if (auto ret = dyn_cast<func::ReturnOp>(func.front().getTerminator()))
      for (auto result : ret.getOperands())
        nameOne(result, /*isResult=*/true, StringRef());

  return portList;
}

bool ModuleEmitter::isConstParam(func::FuncOp func, unsigned idx) const {
  auto it = constParams.find(func.getOperation());
  return it != constParams.end() && idx < it->second.size() && it->second[idx];
}

void ModuleEmitter::emitPortDecl(Value port, bool isConst) {
  auto type = peelAxiType(port.getType());

  if (isConst)
    os << "const ";
  os << getDataTypeName(port.getType()) << " ";
  if (isa<StreamType>(type))
    os << "&";
  os << getName(port);

  if (auto arrayType = dyn_cast<MemRefType>(type))
    if (arrayType.hasStaticShape())
      for (auto &shape : arrayType.getShape())
        os << "[" << shape << "]";
}

void ModuleEmitter::emitFunctionSignature(func::FuncOp func,
                                          ArrayRef<Value> portList,
                                          bool asPrototype) {
  os << "void " << getEmittedFuncName(func.getName()) << "(";
  if (portList.empty()) {
    os << ")";
    return;
  }

  os << "\n";
  addIndent();
  unsigned numArgs = func.getNumArguments();
  for (auto [idx, port] : llvm::enumerate(portList)) {
    indent();
    emitPortDecl(port, idx < numArgs && isConstParam(func, idx));
    if (idx + 1 != portList.size())
      os << ",";
    os << "\n";
  }
  reduceIndent();
  os << ")";
  if (asPrototype)
    os << ";\n";
}

/// Declares a loop counter named `i<n>`, numbered within the function.
/// Counters read far better as `i0`/`i1` than as entries in the temporary
/// sequence, and they are the identifiers a reader scans for.
void ModuleEmitter::nameLoopIV(Value iv) {
  if (isDeclared(iv)) {
    os << getName(iv);
    return;
  }
  os << getDataTypeName(iv.getType()) << " "
     << addNamedValue(iv, "i" + std::to_string(loopIVIdx++));
}

/// Names a lifted table after what can actually be proven about its values,
/// so the name never claims more than the data supports.
static std::optional<std::pair<double, double>>
getExactLinearTable(DenseElementsAttr attr) {
  auto type = cast<ShapedType>(attr.getType());
  auto elementType = dyn_cast<FloatType>(type.getElementType());
  if (!elementType || type.getNumElements() <= 2)
    return std::nullopt;

  SmallVector<double> values;
  values.reserve(type.getNumElements());
  for (APFloat element : attr.getValues<APFloat>()) {
    bool lossy = false;
    element.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                    &lossy);
    values.push_back(element.convertToDouble());
  }
  if (elementType.isF32()) {
    float base = static_cast<float>(values[0]);
    float step = static_cast<float>(values[1]) - base;
    if (step == 0.0f)
      return std::nullopt;
    for (size_t i = 2; i < values.size(); ++i)
      if (base + step * static_cast<float>(i) != static_cast<float>(values[i]))
        return std::nullopt;
    return std::pair<double, double>(base, step);
  }
  double base = values[0], step = values[1] - base;
  if (step == 0.0)
    return std::nullopt;
  for (size_t i = 2; i < values.size(); ++i)
    if (base + step * static_cast<double>(i) != values[i])
      return std::nullopt;
  return std::pair(base, step);
}

static std::optional<std::string> describeNamedTable(ConstBufferOp op) {
  auto source = op.getSourceName();
  if (!source)
    return std::nullopt;

  SmallVector<StringRef> parts;
  source->split(parts, '_', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  if (parts.size() == 7 && parts[0] == "sar" && parts[1] == "fft" &&
      parts[2] == "twiddle" && (parts[3] == "cos" || parts[3] == "sin") &&
      parts[5].starts_with("s")) {
    std::string kind = parts[3] == "cos" ? "Cos" : "Sin";
    return (Twine("kTwiddle") + kind + "_" + parts[4] + parts[5].upper()).str();
  }
  return std::nullopt;
}

static std::string describeTable(ConstBufferOp op, DenseElementsAttr attr,
                                 unsigned idx) {
  if (auto named = describeNamedTable(op))
    return *named;
  auto type = cast<ShapedType>(attr.getType());
  int64_t n = type.getNumElements();

  SmallVector<double, 0> values;
  if (isa<FloatType>(type.getElementType()) && n > 2) {
    values.reserve(n);
    for (auto element : attr.getValues<APFloat>()) {
      APFloat copy = element;
      bool lossy = false;
      copy.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &lossy);
      values.push_back(copy.convertToDouble());
    }
  }

  if (!values.empty()) {
    // A constant-step ramp is a sampling grid: a frequency or spatial axis.
    double step = values[1] - values[0];
    bool isRamp = step != 0.0;
    for (int64_t i = 2; i < n && isRamp; ++i)
      isRamp =
          std::abs((values[i] - values[i - 1]) - step) <= 1e-9 * std::abs(step);
    if (isRamp)
      return ("kAxis" + Twine(idx) + "_" + Twine(n)).str();

    // Unit-bounded tables are the FFT twiddle layout: L-1 entries for an
    // L-point transform, padded up to a multiple of three so the three
    // radix-4 reads of one butterfly can be banked apart. Depending on
    // log2(L) that leaves L-1 or L+1 stored entries; the first entry
    // separates the sine table from the cosine one.
    bool unitBounded = true;
    for (double v : values)
      unitBounded &= std::abs(v) <= 1.0;
    if (unitBounded && n >= 3) {
      for (int64_t points : {n + 1, n - 1})
        if (points >= 4 && (points & (points - 1)) == 0 &&
            (points - 1 + 2) / 3 * 3 == n) {
          if (values[0] == 0.0)
            return ("kTwiddleSin_" + Twine(points)).str();
          if (values[0] == 1.0)
            return ("kTwiddleCos_" + Twine(points)).str();
        }
    }
  }

  return ("kTable" + Twine(idx) + "_" + Twine(n)).str();
}

/// Finds the constant buffers worth lifting to file scope and works out
/// which callee parameters can therefore be `const`.
void ModuleEmitter::collectGlobalTables(ModuleOp module,
                                        ArrayRef<func::FuncOp> funcs) {
  llvm::StringSet<> usedNames;
  unsigned tableIdx = 0;

  module.walk([&](ConstBufferOp op) {
    auto attr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!attr || !isa<MemRefType>(op.getResult().getType()))
      return;

    // Uniquing keeps two tables that describe the same thing apart.
    std::string base = describeTable(op, attr, tableIdx++);
    std::string name = base;
    for (unsigned suffix = 2; !usedNames.insert(name).second; ++suffix)
      name = base + "_" + std::to_string(suffix);

    state.globalTables.try_emplace(op.getResult(), name);
    state.globalTableDeclNames.try_emplace(op.getResult(), name);
    if (auto linear = getExactLinearTable(attr))
      state.linearTables.try_emplace(op.getResult(), *linear);
    hoistedTables.insert(op.getOperation());
  });

  if (state.globalTables.empty())
    return;

  // A parameter may be const when every call passes a table or an already
  // const parameter, so the property has to reach a fixed point.
  for (auto func : funcs)
    constParams[func.getOperation()] = llvm::BitVector(func.getNumArguments());

  auto isConstSource = [&](Value value) {
    if (state.globalTables.count(value))
      return true;
    if (auto arg = dyn_cast<BlockArgument>(value))
      if (auto owner =
              dyn_cast_or_null<func::FuncOp>(arg.getOwner()->getParentOp()))
        return isConstParam(owner, arg.getArgNumber());
    return false;
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto func : funcs) {
      auto &bits = constParams[func.getOperation()];
      for (unsigned idx = 0, e = func.getNumArguments(); idx != e; ++idx) {
        if (bits[idx] ||
            !isa<MemRefType>(peelAxiType(func.getArgument(idx).getType())))
          continue;
        bool anyCall = false, allConst = true;
        auto uses = func.getSymbolUses(func->getParentOp());
        if (!uses)
          continue;
        for (auto use : *uses) {
          auto call = dyn_cast<func::CallOp>(use.getUser());
          if (!call || idx >= call.getNumOperands())
            continue;
          anyCall = true;
          allConst &= isConstSource(call.getOperand(idx));
        }
        if (anyCall && allConst) {
          bits.set(idx);
          changed = true;
        }
      }
    }
  }

  // A table handed to a parameter that stayed mutable cannot be const
  // itself; it still gets lifted, just without the qualifier.
  for (auto &entry : state.globalTables) {
    for (auto &use : entry.first.getUses()) {
      auto call = dyn_cast<func::CallOp>(use.getOwner());
      if (!call)
        continue;
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || !isConstParam(callee, use.getOperandNumber())) {
        mutableTables.insert(entry.first);
        break;
      }
    }
  }
  for (auto &[value, linear] : state.linearTables)
    if (!mutableTables.count(value))
      state.globalTables[value] =
          state.globalTableDeclNames.lookup(value) + ".values";
}

void ModuleEmitter::emitGlobalTables() {
  if (state.globalTables.empty())
    return;

  // The block is marked so the artifact writer can lift it into
  // <top>_tables.h: the tables are data, often most of the file's lines, and
  // separating them leaves the implementation file reading as logic. Direct
  // sar-translate output stays valid as a single file.
  os << "//===----------------------------------------------------------"
        "------------===//\n"
        "// Constant tables. Lifted to file scope so the design reads as\n"
        "// logic and the tool can map them to ROM.\n"
        "//===----------------------------------------------------------"
        "------------===//\n"
        "// SAR_DSL_TABLES_BEGIN\n";

  bool hasLinearTable =
      llvm::any_of(state.linearTables, [&](const auto &entry) {
        return !mutableTables.count(entry.first);
      });
  if (hasLinearTable)
    os << "template <typename T, int N>\n"
          "struct LinearTable {\n"
          "  T values[N];\n"
          "  constexpr LinearTable(T base, T step) : values{} {\n"
          "    for (int i = 0; i < N; ++i)\n"
          "      values[i] = base + step * static_cast<T>(i);\n"
          "  }\n"
          "};\n\n";

  // Emit in creation order for a stable, diffable file.
  for (auto *op : hoistedTables) {
    auto constOp = cast<ConstBufferOp>(op);
    Value result = constOp.getResult();
    auto it = state.globalTables.find(result);
    if (it == state.globalTables.end())
      continue;

    auto memrefType = cast<MemRefType>(result.getType());
    auto elementType = memrefType.getElementType();
    auto attr = cast<DenseElementsAttr>(constOp.getValue());
    const std::string &declName = state.globalTableDeclNames.lookup(result);

    if (auto linear = state.linearTables.find(result);
        linear != state.linearTables.end() && !mutableTables.count(result)) {
      os << "static constexpr LinearTable<" << getDataTypeName(elementType)
         << ", " << memrefType.getNumElements() << "> " << declName << "(";
      if (cast<FloatType>(elementType).isF32())
        os << formatFloat(static_cast<float>(linear->second.first), "%.9g")
           << "f, "
           << formatFloat(static_cast<float>(linear->second.second), "%.9g")
           << "f";
      else
        os << formatFloat(linear->second.first, "%.17g") << ", "
           << formatFloat(linear->second.second, "%.17g");
      os << ");\n\n";
      state.nameTable[result] = SmallString<8>(StringRef(it->second));
      continue;
    }

    os << "static ";
    if (!mutableTables.count(result))
      os << "const ";
    os << getDataTypeName(elementType) << " " << declName;
    for (auto &shape : memrefType.getShape())
      os << "[" << shape << "]";
    os << " = {";

    // The declared element type already fixes the value's type, so the cast
    // every scalar constant carries is noise across a 256-entry table.
    bool stripCast = isa<FloatType, IndexType>(elementType);
    unsigned perLine = elementType.getIntOrFloatBitWidth() > 32 ? 4 : 8;

    unsigned elementIdx = 0;
    for (auto element : attr.getValues<Attribute>()) {
      auto string = getConstantString(elementType, element);
      if (string.empty())
        constOp.emitOpError("constant has invalid value");
      StringRef value(string);
      if (stripCast && value.starts_with("("))
        value = value.drop_front(value.find(')') + 1);
      if (elementIdx % perLine == 0)
        os << "\n    ";
      os << value;
      if (++elementIdx != attr.getNumElements())
        os << ",";
      if (elementIdx % perLine != 0)
        os << " ";
    }
    os << "\n};\n\n";

    // Bind the name so every reader of the buffer resolves to the table.
    state.nameTable[result] = SmallString<8>(StringRef(it->second));
  }
  os << "// SAR_DSL_TABLES_END\n\n";
}

void ModuleEmitter::emitInterfaceSchema(func::FuncOp func) {
  auto ports = assignPortNames(func);
  for (auto [index, port] : llvm::enumerate(ports)) {
    AxiPortOp axiPort;
    func.walk([&](AxiPortOp candidate) {
      if (candidate.getAxi() == port)
        axiPort = candidate;
    });

    Type type = peelAxiType(port.getType());
    Type elementType = type;
    SmallVector<int64_t> physicalShape;
    unsigned depth = 0;
    if (auto memref = dyn_cast<MemRefType>(type)) {
      elementType = memref.getElementType();
      physicalShape.assign(memref.getShape().begin(), memref.getShape().end());
    } else if (auto stream = dyn_cast<StreamType>(type)) {
      elementType = stream.getElementType();
      depth = stream.getDepth();
      if (axiPort)
        if (auto shape = axiPort->getAttrOfType<ArrayAttr>("stream_shape"))
          for (Attribute extent : shape)
            physicalShape.push_back(cast<IntegerAttr>(extent).getInt());
    }

    unsigned lanes = 1;
    Type scalarType = elementType;
    if (auto vector = dyn_cast<VectorType>(elementType)) {
      lanes = vector.getNumElements();
      scalarType = vector.getElementType();
    }
    unsigned scalarBits = 0;
    if (scalarType.isIndex())
      scalarBits = 64;
    else if (scalarType.isIntOrFloat())
      scalarBits = scalarType.getIntOrFloatBitWidth();
    uint64_t logicalElements = lanes;
    for (int64_t extent : physicalShape)
      logicalElements *= extent;

    PortRole role = index >= func.getNumArguments()
                        ? PortRole::Out
                        : getPortRole(axiPort ? axiPort.getElement() : port);
    if (role == PortRole::Unused)
      role = PortRole::In;

    StringRef protocol = "s_axilite";
    StringRef bundle = "ctrl";
    if (axiPort) {
      bool requestedStream =
          axiPort.getBundleType().getKind() == AxiKind::STREAM ||
          (func->hasAttr("stream_interface") &&
           !axiPort->hasAttr("hls.scratch"));
      bool streaming = requestedStream && isProvablySequentialStreamPort(
                                              axiPort.getElement(), role);
      protocol = streaming ? "axis" : "m_axi";
      bundle = streaming ? StringRef() : axiPort.getBundleName();
    } else if (auto memref = dyn_cast<MemRefType>(type)) {
      protocol = getMemoryKind(memref) == MemoryKind::DRAM ? "m_axi" : "bram";
      bundle = "";
    } else if (isa<StreamType>(type)) {
      protocol = "axis";
      bundle = "";
    }

    std::string name = std::string(getName(port));
    if (!name.empty() && name.front() == '*')
      name.erase(name.begin());

    os << "// SAR_DSL_INTERFACE: {\"name\":\"" << name << "\",\"protocol\":\""
       << protocol << "\",\"direction\":\"" << getPortRolePrefix(role)
       << "\",\"kind\":\""
       << (axiPort && axiPort->hasAttr("hls.scratch") ? "scratch" : "public")
       << "\",\"bundle\":\"" << bundle << "\",\"c_type\":\""
       << getDataTypeName(elementType) << "\",\"scalar_type\":\""
       << getDataTypeName(scalarType) << "\",\"vector_lanes\":" << lanes
       << ",\"data_bits\":" << scalarBits * lanes << ",\"depth\":" << depth
       << ",\"physical_shape\":[";
    for (auto [dim, extent] : llvm::enumerate(physicalShape)) {
      if (dim)
        os << ",";
      os << extent;
    }
    os << "],\"logical_shape\":[";
    for (auto [dim, extent] : llvm::enumerate(physicalShape)) {
      if (dim)
        os << ",";
      if (dim + 1 == physicalShape.size())
        extent *= lanes;
      os << extent;
    }
    os << "],\"logical_elements\":" << logicalElements << "}\n";
  }
}

void ModuleEmitter::emitFunction(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "must have exactly one basic block.");

  // Temporaries and loop counters restart in every function.
  state.localNameIdx = 0;
  loopIVIdx = 0;
  bufferIdx = 0;

  if (hasTopFuncAttr(func))
    os << "/// Top function of the design.\n";

  auto portList = assignPortNames(func);
  emitFunctionSignature(func, portList, /*asPrototype=*/false);
  os << " {\n";

  // Emit function body.
  addIndent();

  emitFunctionDirectives(func, portList);
  SmallVector<std::pair<Value, SmallString<8>>> tableNames;
  auto directive = getFuncDirective(func);
  if (directive && directive.getDataflow()) {
    for (Operation *operation : hoistedTables) {
      Value table = cast<ConstBufferOp>(operation).getResult();
      if (!state.globalTables.count(table))
        continue;
      bool usedHere = llvm::any_of(table.getUses(), [&](OpOperand &use) {
        return use.getOwner()->getParentOfType<func::FuncOp>() == func;
      });
      if (!usedHere)
        continue;
      SmallString<8> globalName = getName(table);
      SmallString<8> localName("v" + std::to_string(state.localNameIdx++));
      indent() << "const auto &" << localName << " = " << globalName << ";\n";
      tableNames.push_back({table, globalName});
      state.nameTable[table] = localName;
    }
    if (!tableNames.empty())
      os << "\n";
  }
  emitBlock(func.front());
  for (auto &[table, globalName] : tableNames)
    state.nameTable[table] = globalName;
  reduceIndent();
  os << "}\n";
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  // Gather the functions once: the order they are emitted in is decided
  // below, and several decisions need the whole set up front.
  SmallVector<func::FuncOp, 16> funcs;
  func::FuncOp topFunc;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (hasRuntimeAttr(func))
      continue;
    funcs.push_back(func);
    if (hasTopFuncAttr(func))
      topFunc = func;
  }
  assignFuncNames(funcs, topFunc);

  // Only include what the design actually uses. Emitting the full Vitis
  // header set unconditionally drags in headers the design never touches
  // and breaks the portable C++ fallback over stub headers.
  bool usesStream = false, usesVector = false, usesApInt = false;
  bool usesMemCpy = false;
  auto noteType = [&](Type type) {
    if (!type)
      return;
    auto peeled = peelAxiType(type);
    usesStream |= isa<StreamType>(peeled);
    usesVector |= isa<VectorType>(peeled);
    if (auto memref = dyn_cast<MemRefType>(peeled))
      peeled = memref.getElementType();
    if (auto streamType = dyn_cast<StreamType>(peeled))
      peeled = streamType.getElementType();
    usesVector |= isa<VectorType>(peeled);
    if (auto intType = dyn_cast<IntegerType>(peeled))
      usesApInt |= intType.getWidth() != 1;
  };

  for (auto func : funcs) {
    for (auto type : func.getArgumentTypes())
      noteType(type);
    for (auto type : func.getResultTypes())
      noteType(type);
    func.walk([&](Operation *op) {
      usesMemCpy |= isa<memref::CopyOp>(op);
      for (auto type : op->getResultTypes())
        noteType(type);
      for (auto type : op->getOperandTypes())
        noteType(type);
    });
  }

  os << "//===- SAR-DSL generated Vitis HLS design -----------------------"
        "*- C++ -*-===//\n"
        "//\n"
        "// Generated by sar-translate --hls-emit-hlscpp.\n"
        "// Regenerate this file instead of editing it.\n"
        "//\n";
  if (topFunc)
    os << "// Top function : " << topFunc.getName() << "\n";
  os << "// Directives   : " << (emitVitisDirectives ? "vitis" : "vivado")
     << "\n";
  // What a board integrator has to wire: how many top-level ports there are
  // and what protocol they speak. Only a design that has such ports reports
  // them, so an `ap_memory` package says nothing about a bus it does not use.
  if (topFunc) {
    unsigned ports = 0;
    topFunc.walk([&](AxiPortOp port) {
      if (isa<MemRefType>(port.getElement().getType()))
        ++ports;
    });
    if (ports) {
      bool isStream = topFunc->hasAttr("stream_interface");
      os << "// Interface    : " << ports
         << (isStream ? " AXI4-Stream port" : " AXI master")
         << (ports == 1 ? "" : "s") << "\n";
      if (!isStream)
        os << "// AXI bus      : " << axiBusBits << "-bit\n";
    }
    if (topFunc->hasAttr("hls.scratch_arena_overflow")) {
      os << "// Scratch ABI  : conflict graph compacted to configured master "
            "limit\n";
      if (auto penalty =
              topFunc->getAttrOfType<IntegerAttr>("hls.scratch_arena_penalty"))
        os << "// Scratch cost : predicted serialization penalty "
           << penalty.getInt() << "\n";
    }
  }
  os << "// Sub-functions: " << (funcs.empty() ? 0 : funcs.size() - 1) << "\n";
  // The machine-readable port records follow the human-readable summary so
  // they do not interrupt it; `design.py` parses them back out.
  if (topFunc)
    emitInterfaceSchema(topFunc);
  os << "//\n"
        "//===------------------------------------------------------------"
        "----------===//\n\n";

  os << "#include <algorithm>\n";
  os << "#include <cmath>\n";
  os << "#include <cstdint>\n";
  if (usesMemCpy)
    os << "#include <cstring>\n";
  if (usesApInt) {
    os << "#include <ap_int.h>\n";
  }
  if (usesStream)
    os << "#include <hls_stream.h>\n";
  if (usesVector)
    os << "#include <hls_vector.h>\n";
  os << "\n"
        "template <typename T>\n"
        "static T sar_hls_maximum(T lhs, T rhs);\n\n"
        "template <typename T>\n"
        "static T sar_hls_minimum(T lhs, T rhs);\n\n"
        "static int64_t sar_hls_floor_div(int64_t lhs, int64_t rhs);\n"
        "static int64_t sar_hls_ceil_div(int64_t lhs, int64_t rhs);\n"
        "static int64_t sar_hls_mod(int64_t lhs, int64_t rhs);\n\n";

  collectGlobalTables(module, funcs);
  emitGlobalTables();

  // Order the sub-functions the way the pipeline runs, so the file reads
  // in the same direction as the data.
  SmallVector<func::FuncOp, 16> subFuncs;
  for (auto func : funcs)
    if (func != topFunc)
      subFuncs.push_back(func);
  llvm::stable_sort(subFuncs, [&](func::FuncOp lhs, func::FuncOp rhs) {
    return getEmittedFuncName(lhs.getName()) <
           getEmittedFuncName(rhs.getName());
  });

  // Keep a self-contained declaration block in the translation output. The
  // Python artifact writer moves this marked block into <top>.h for packaged
  // projects, while direct sar-translate output remains valid as one file.
  if (topFunc) {
    os << "//===----------------------------------------------------------"
          "------------===//\n"
          "// Sub-function prototypes, in dataflow order.\n"
          "//===----------------------------------------------------------"
          "------------===//\n"
          "// SAR_DSL_DECLARATIONS_BEGIN\n";
    for (auto func : subFuncs) {
      state.localNameIdx = 0;
      auto portList = assignPortNames(func);
      emitFunctionSignature(func, portList, /*asPrototype=*/true);
      os << "\n";
    }
    os << "// SAR_DSL_DECLARATIONS_END\n\n";
  }

  if (topFunc)
    emitFunction(topFunc);

  os << "// Internal arithmetic helpers.\n\n"
        "template <typename T>\n"
        "static T sar_hls_maximum(T lhs, T rhs) {\n"
        "  if (std::isnan(lhs))\n"
        "    return lhs;\n"
        "  if (std::isnan(rhs))\n"
        "    return rhs;\n"
        "  if (lhs == rhs && lhs == T(0))\n"
        "    return std::signbit(lhs) ? rhs : lhs;\n"
        "  return lhs > rhs ? lhs : rhs;\n"
        "}\n\n"
        "template <typename T>\n"
        "static T sar_hls_minimum(T lhs, T rhs) {\n"
        "  if (std::isnan(lhs))\n"
        "    return lhs;\n"
        "  if (std::isnan(rhs))\n"
        "    return rhs;\n"
        "  if (lhs == rhs && lhs == T(0))\n"
        "    return std::signbit(lhs) ? lhs : rhs;\n"
        "  return lhs < rhs ? lhs : rhs;\n"
        "}\n\n"
        "static int64_t sar_hls_floor_div(int64_t lhs, int64_t rhs) {\n"
        "  int64_t quotient = lhs / rhs;\n"
        "  int64_t remainder = lhs % rhs;\n"
        "  return quotient - (remainder < 0);\n"
        "}\n\n"
        "static int64_t sar_hls_ceil_div(int64_t lhs, int64_t rhs) {\n"
        "  int64_t quotient = lhs / rhs;\n"
        "  int64_t remainder = lhs % rhs;\n"
        "  return quotient + (remainder > 0);\n"
        "}\n\n"
        "static int64_t sar_hls_mod(int64_t lhs, int64_t rhs) {\n"
        "  int64_t remainder = lhs % rhs;\n"
        "  return remainder < 0 ? remainder + rhs : remainder;\n"
        "}\n\n";

  for (auto func : subFuncs)
    emitFunction(func);

  // Anything that is not a function and not a constant is unexpected.
  for (auto &op : *module.getBody())
    if (!isa<func::FuncOp>(op) && !op.hasTrait<OpTrait::ConstantLike>() &&
        op.getName().getStringRef() != "ml_program.global")
      emitError(&op, "is unsupported operation");
}

//===----------------------------------------------------------------------===//
// Entry of the HLS C++ translation
//===----------------------------------------------------------------------===//

static bool isSupportedHLSCppType(Type type) {
  type = peelAxiType(type);
  if (auto bundle = dyn_cast<BundleType>(type))
    return isSupportedHLSCppType(bundle.getDataType());
  if (auto memref = dyn_cast<MemRefType>(type))
    return memref.hasStaticShape() &&
           isSupportedHLSCppType(memref.getElementType());
  if (auto stream = dyn_cast<StreamType>(type))
    return isSupportedHLSCppType(stream.getElementType());
  if (auto vector = dyn_cast<VectorType>(type))
    return vector.hasStaticShape() &&
           isSupportedHLSCppType(vector.getElementType());
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth() == 32 || floating.getWidth() == 64;
  return isa<IndexType, IntegerType>(type);
}

static LogicalResult verifyHLSCppTarget(ModuleOp module) {
  bool failed = false;
  unsigned tops = 0;
  module.walk([&](Operation *op) {
    if (auto func = dyn_cast<func::FuncOp>(op)) {
      if (!hasRuntimeAttr(func) && hasTopFuncAttr(func))
        ++tops;
      if (func.isExternal()) {
        func.emitError("HLS C++ target does not support external functions");
        failed = true;
      } else if (func.getBlocks().size() != 1) {
        func.emitError("HLS C++ target requires exactly one basic block");
        failed = true;
      } else {
        // Each port must be a distinct value: a function returning one of
        // its arguments, or the same value twice, would declare two ports
        // with one name. Caught here rather than while naming them, because
        // this runs before any C++ is written -- reporting it mid-emission
        // would leave an uncompilable definition on the output stream for
        // any caller that reads the stream without checking the status.
        DenseSet<Value> seen;
        for (Value arg : func.getArguments())
          seen.insert(arg);
        if (auto ret = dyn_cast<func::ReturnOp>(func.front().getTerminator()))
          for (Value result : ret.getOperands())
            if (!seen.insert(result).second) {
              func.emitError(
                  "HLS C++ target does not support port aliasing: a value "
                  "reaches the signature twice (a result that is also an "
                  "argument, or one value returned twice)");
              failed = true;
              break;
            }
      }
    }
    for (Region &region : op->getRegions())
      if (!region.empty() && !llvm::hasSingleElement(region.getBlocks())) {
        op->emitError("HLS C++ target does not support multi-block regions");
        failed = true;
      }
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()))
      if (!isSupportedHLSCppType(type)) {
        op->emitError("HLS C++ target does not support type ") << type;
        failed = true;
      }
    if (isa<memref::ViewOp, memref::SubViewOp, memref::ReinterpretCastOp,
            memref::ExpandShapeOp, memref::CollapseShapeOp>(op)) {
      op->emitError("HLS C++ target requires views to be folded before "
                    "emission");
      failed = true;
    }
    if (isa<AffineParallelOp>(op)) {
      op->emitError("HLS C++ target requires affine.parallel to be lowered "
                    "before emission");
      failed = true;
    }
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (!callee || callee.isExternal()) {
        call.emitError("HLS C++ target requires every call to resolve to a "
                       "defined function");
        failed = true;
      }
    }
    if (auto cmp = dyn_cast<arith::CmpFOp>(op))
      if (!llvm::is_contained(
              {arith::CmpFPredicate::OEQ, arith::CmpFPredicate::UNE,
               arith::CmpFPredicate::OLT, arith::CmpFPredicate::OLE,
               arith::CmpFPredicate::OGT, arith::CmpFPredicate::OGE},
              cmp.getPredicate())) {
        cmp.emitError("HLS C++ target requires explicit NaN handling for "
                      "this floating-point predicate");
        failed = true;
      }
  });
  if (tops > 1) {
    module.emitError("HLS C++ target allows at most one top function; found ")
        << tops;
    failed = true;
  }
  DenseMap<Operation *, unsigned> colors;
  std::function<bool(func::FuncOp)> visit = [&](func::FuncOp func) {
    unsigned &color = colors[func.getOperation()];
    if (color == 1)
      return true;
    if (color == 2 || func.isExternal())
      return false;
    color = 1;
    bool recursive = false;
    func.walk([&](func::CallOp call) {
      auto callee = dyn_cast_or_null<func::FuncOp>(
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
      if (callee && visit(callee)) {
        call.emitError("HLS C++ target does not support recursive calls");
        recursive = true;
      }
    });
    color = 2;
    return recursive;
  };
  for (auto func : module.getOps<func::FuncOp>())
    failed |= visit(func);
  return failure(failed);
}

LogicalResult sar::emitHLSCpp(ModuleOp module, llvm::raw_ostream &os) {
  if (failed(verifyHLSCppTarget(module)))
    return failure();
  HLSEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void sar::registerEmitHLSCppTranslation() {
  static TranslateFromMLIRRegistration toHLSCpp(
      "hls-emit-hlscpp", "Translate MLIR into synthesizable C++", emitHLSCpp,
      [&](DialectRegistry &registry) {
        registry.insert<hls::HLSDialect>();
        registerAllDialects(registry);
      });
}
