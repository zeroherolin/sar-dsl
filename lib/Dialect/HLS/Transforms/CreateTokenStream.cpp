//===- CreateTokenStream.cpp - create token stream ------------------------===//
//
// Part of the SAR-DSL Project. Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "sar/Dialect/HLS/Transforms/Passes.h"
#include "sar/Dialect/HLS/Transforms/Utils.h"

namespace mlir {
namespace sar {
#define GEN_PASS_DEF_CREATETOKENSTREAM
#include "sar/Dialect/HLS/Transforms/Passes.h.inc"
} // namespace sar
} // namespace mlir

using namespace mlir;
using namespace sar;
using namespace hls;

namespace {
struct CreateTokenStream
    : public sar::impl::CreateTokenStreamBase<CreateTokenStream> {
  void runOnOperation() override {
    auto func = getOperation();
    auto context = func.getContext();
    OpBuilder b(context);
    auto loc = b.getUnknownLoc();

    func.walk([&](ScheduleOp schedule) {
      if (!schedule.getIsLegal())
        return WalkResult::advance();

      SmallVector<Value> buffers;
      for (auto arg : schedule.getBody().getArguments())
        if (isExtBuffer(arg))
          buffers.push_back(arg);
      for (auto bufferOp : schedule.getOps<BufferOp>())
        if (isExtBuffer(bufferOp))
          buffers.push_back(bufferOp);

      for (auto buffer : buffers) {
        auto producers = getProducers(buffer);
        if (!llvm::hasSingleElement(producers))
          continue;

        auto producer = producers.front();
        auto outputIdx = llvm::find(producer.getOutputs(), buffer) -
                         producer.getOutputs().begin();
        SmallVector<Value, 8> outputs(producer.getOutputs());
        SmallVector<std::pair<StreamOp, NodeOp>, 4> tokenConsumers;

        auto consumers = getDependentConsumers(buffer, producer);
        if (consumers.empty())
          continue;

        for (auto consumer : consumers) {
          if (consumer == producer)
            continue;
          // Unscheduled nodes carry no level; a reversed pair would wrap an
          // unsigned diff into a huge FIFO depth, so compute signed and keep
          // at least one slot.
          if (!producer.getLevel() || !consumer.getLevel())
            continue;
          int64_t levelDiff = std::max<int64_t>(
              1, (int64_t)*producer.getLevel() - (int64_t)*consumer.getLevel());
          b.setInsertionPointAfterValue(buffer);
          auto token = StreamOp::create(
              b, loc, StreamType::get(b.getContext(), b.getI1Type(), levelDiff),
              levelDiff);
          tokenConsumers.push_back({token, consumer});

          // Add the stream channel as a new output argument of the producer.
          outputs.insert(std::next(outputs.begin(), outputIdx),
                         token.getChannel());
          auto tokenArg = producer.getBody().insertArgument(
              outputIdx++ + producer.getNumInputs(), token.getType(),
              token.getLoc());

          // Construct stream write on the producer side.
          b.setInsertionPointToEnd(&producer.getBody().front());
          auto value = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));
          StreamWriteOp::create(b, loc, tokenArg, value);
        }
        if (tokenConsumers.empty())
          continue;

        // Construct a new producer node.
        b.setInsertionPoint(producer);
        auto newProducer =
            NodeOp::create(b, producer.getLoc(), producer.getInputs(), outputs,
                           producer.getParams(), producer.getInputTapsAttr(),
                           producer.getLevelAttr());
        newProducer.getBody().getBlocks().splice(
            newProducer.getBody().end(), producer.getBody().getBlocks());
        producer.erase();

        for (auto [token, consumer] : tokenConsumers) {

          // Add the stream channel as a new input argument of the consumer.
          auto inputIdx = llvm::find(consumer.getInputs(), buffer) -
                          consumer.getInputs().begin();
          SmallVector<Value, 8> inputs(consumer.getInputs());
          SmallVector<unsigned> inputTaps(consumer.getInputTapsAsInt());

          inputs.insert(std::next(inputs.begin(), inputIdx),
                        token.getChannel());
          inputTaps.insert(std::next(inputTaps.begin(), inputIdx),
                           token.getType().getDepth() - 1);
          auto tokenArg = consumer.getBody().insertArgument(
              inputIdx, token.getType(), token.getLoc());

          // Construct stream write on the producer side.
          b.setInsertionPointToStart(&consumer.getBody().front());
          StreamReadOp::create(b, loc, Type(), tokenArg);

          // Construct a new consumer node.
          b.setInsertionPoint(consumer);
          auto newConsumer = NodeOp::create(
              b, consumer.getLoc(), inputs, consumer.getOutputs(),
              consumer.getParams(), inputTaps, consumer.getLevelAttr());
          newConsumer.getBody().getBlocks().splice(
              newConsumer.getBody().end(), consumer.getBody().getBlocks());
          consumer.erase();
        }
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<Pass> sar::createCreateTokenStreamPass() {
  return std::make_unique<CreateTokenStream>();
}
