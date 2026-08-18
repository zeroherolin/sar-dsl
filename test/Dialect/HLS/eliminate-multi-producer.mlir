// RUN: sar-opt %s --hls-eliminate-multi-producer | FileCheck %s

// Two nodes writing one internal buffer would be two processes driving one
// channel. The pass privatizes: the later producer writes a fresh buffer and
// takes the original as an extra input, and consumers below it move to the
// new buffer -- every internal buffer ends with a single producer.

// CHECK-LABEL: func.func @privatize
func.func @privatize(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    // CHECK: %[[A:.*]] = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // CHECK: %[[B:.*]] = hls.dataflow.buffer
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %b = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // CHECK: hls.dataflow.node(%{{.*}}) -> (%[[A]])
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 4 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    // The dominated producer is rebuilt around a private buffer, with the
    // original buffer threaded in as an input.
    // CHECK: %[[NEW:.*]] = hls.dataflow.buffer
    // CHECK: hls.dataflow.node(%{{.*}}, %[[A]]) -> (%[[NEW]])
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 4 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    // The downstream consumer now reads the private buffer.
    // CHECK: hls.dataflow.node(%[[NEW]]) -> (%[[B]])
    hls.dataflow.node(%a) -> (%b) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
  }
  return
}

// An external buffer cannot be privatized -- it is the interface -- so two
// adjacent producers of it are fused into one node instead.

// CHECK-LABEL: func.func @merge_external
// CHECK: hls.dataflow.node() -> (%{{.*}})
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.for %{{.*}} = 4 to 8
// CHECK-NOT: hls.dataflow.node
func.func @merge_external(%arg0: memref<8xf32, #hls.mem<dram>>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<8xf32, #hls.mem<dram>>):
    hls.dataflow.node() -> (%in) {inputTaps = []} : () -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%o: memref<8xf32, #hls.mem<dram>>):
      %c = arith.constant 1.0 : f32
      affine.for %j = 0 to 4 {
        affine.store %c, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
    hls.dataflow.node() -> (%in) {inputTaps = []} : () -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%o: memref<8xf32, #hls.mem<dram>>):
      %c = arith.constant 2.0 : f32
      affine.for %j = 4 to 8 {
        affine.store %c, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
  }
  return
}
