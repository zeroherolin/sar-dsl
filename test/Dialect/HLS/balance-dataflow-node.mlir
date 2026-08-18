// RUN: sar-opt %s --hls-balance-dataflow-node | FileCheck %s

// A dataflow region runs its nodes as a pipeline: when a buffer's producer
// and consumer sit more than one level apart, the value must be carried
// across the intermediate beats. For an on-chip buffer the pass inserts
// copy nodes, one per skipped level. An external (DRAM) buffer keeps depth
// 1 and gets no copies: DRAM is persistent, so a consumer several levels
// down simply reads it later, and where the levels do have to be held
// apart CreateTokenStream carries that with a token of the matching depth.
// Giving the buffer a depth here would duplicate that, at one full-size
// plane per level.

// CHECK-LABEL: func.func @skip_level
func.func @skip_level(%arg0: memref<16xf32, #hls.mem<dram>>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<16xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<16xf32, #hls.mem<dram>>):
    %onchip = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<bram_t2p>>
    %dram = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<dram>>
    %sink1 = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<bram_t2p>>
    %sink2 = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<bram_t2p>>
    %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<16xf32, #hls.mem<bram_t2p>>

    // The on-chip buffer read two levels below its producer is re-carried
    // through a copy node (one new buffer); every buffer keeps depth 1.
    // CHECK-COUNT-6: hls.dataflow.buffer {depth = 1 : i32}
    // CHECK-NOT: depth = 2

    // Level-2 producer fills the on-chip and the DRAM buffer.
    hls.dataflow.node(%in) -> (%onchip, %dram) {inputTaps = [0 : i32], level = 2 : i32}
        : (memref<16xf32, #hls.mem<dram>>) -> (memref<16xf32, #hls.mem<bram_t2p>>, memref<16xf32, #hls.mem<dram>>) {
    ^bb0(%i: memref<16xf32, #hls.mem<dram>>, %o1: memref<16xf32, #hls.mem<bram_t2p>>, %o2: memref<16xf32, #hls.mem<dram>>):
      affine.for %j = 0 to 16 {
        %v = affine.load %i[%j] : memref<16xf32, #hls.mem<dram>>
        affine.store %v, %o1[%j] : memref<16xf32, #hls.mem<bram_t2p>>
        affine.store %v, %o2[%j] : memref<16xf32, #hls.mem<dram>>
      }
    }

    // A level-1 hop so the consumers below sit two levels from the producer.
    hls.dataflow.node(%onchip) -> (%mid) {inputTaps = [0 : i32], level = 1 : i32}
        : (memref<16xf32, #hls.mem<bram_t2p>>) -> memref<16xf32, #hls.mem<bram_t2p>> {
    ^bb0(%i: memref<16xf32, #hls.mem<bram_t2p>>, %o: memref<16xf32, #hls.mem<bram_t2p>>):
      affine.for %j = 0 to 16 {
        %v = affine.load %i[%j] : memref<16xf32, #hls.mem<bram_t2p>>
        affine.store %v, %o[%j] : memref<16xf32, #hls.mem<bram_t2p>>
      }
    }

    // Exactly one copy node appears (for the on-chip skip), none for DRAM.
    // CHECK: memref.copy
    // CHECK-NOT: memref.copy

    // Level-0 consumers: one reads the on-chip buffer produced two levels
    // up, the other reads the DRAM buffer produced two levels up.
    hls.dataflow.node(%onchip, %mid) -> (%sink1) {inputTaps = [0 : i32, 0 : i32], level = 0 : i32}
        : (memref<16xf32, #hls.mem<bram_t2p>>, memref<16xf32, #hls.mem<bram_t2p>>) -> memref<16xf32, #hls.mem<bram_t2p>> {
    ^bb0(%a: memref<16xf32, #hls.mem<bram_t2p>>, %b: memref<16xf32, #hls.mem<bram_t2p>>, %o: memref<16xf32, #hls.mem<bram_t2p>>):
      affine.for %j = 0 to 16 {
        %v = affine.load %a[%j] : memref<16xf32, #hls.mem<bram_t2p>>
        %w = affine.load %b[%j] : memref<16xf32, #hls.mem<bram_t2p>>
        %s = arith.addf %v, %w : f32
        affine.store %s, %o[%j] : memref<16xf32, #hls.mem<bram_t2p>>
      }
    }
    hls.dataflow.node(%dram, %mid) -> (%sink2) {inputTaps = [0 : i32, 0 : i32], level = 0 : i32}
        : (memref<16xf32, #hls.mem<dram>>, memref<16xf32, #hls.mem<bram_t2p>>) -> memref<16xf32, #hls.mem<bram_t2p>> {
    ^bb0(%a: memref<16xf32, #hls.mem<dram>>, %b: memref<16xf32, #hls.mem<bram_t2p>>, %o: memref<16xf32, #hls.mem<bram_t2p>>):
      affine.for %j = 0 to 16 {
        %v = affine.load %a[%j] : memref<16xf32, #hls.mem<dram>>
        %w = affine.load %b[%j] : memref<16xf32, #hls.mem<bram_t2p>>
        %s = arith.addf %v, %w : f32
        affine.store %s, %o[%j] : memref<16xf32, #hls.mem<bram_t2p>>
      }
    }
  }
  return
}
