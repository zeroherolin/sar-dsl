// RUN: sar-opt -hls-legalize-dataflow %s | FileCheck %s

// Two nodes that share an input sit at the same level and are candidates for
// fusion, but a consumer of the first one's output sits between them. Fusion
// collapses the group into the position of its last member, which would put
// the merged body after the consumer that reads the first one's result -- so
// the value would be read before it is written. The pass has to leave this
// schedule alone.

// CHECK-LABEL: func.func @unfusable_multi_consumer
// CHECK-COUNT-3: hls.dataflow.node
// CHECK-NOT: hls.dataflow.node
func.func @unfusable_multi_consumer(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %b = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %c = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>

    // Producer of %a.
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32], level = 2 : i32}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }

    // Consumer of %a, sitting between the two producers.
    hls.dataflow.node(%a) -> (%c) {inputTaps = [0 : i32], level = 1 : i32}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }

    // Producer of %b, sharing %in with the first node and at the same level.
    hls.dataflow.node(%in) -> (%b) {inputTaps = [0 : i32], level = 2 : i32}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
  }
  return
}

// -----

// An external buffer written by two nodes is a plane being reused across
// lifetimes that do not overlap. Nothing orders the write-after-read edge
// reuse introduces, so the schedule has to stay sequential: without the
// `legal` marker the emitter attaches no dataflow pragma.

// CHECK-LABEL: func.func @reused_external_buffer
// CHECK-NOT: hls.dataflow.schedule legal
func.func @reused_external_buffer(%arg0: memref<8xf32, #hls.mem<dram>>)
    attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<8xf32, #hls.mem<dram>>):
    %shared = hls.dataflow.buffer {depth = 1 : i32}
        : memref<8xf32, #hls.mem<dram>>

    hls.dataflow.node(%in) -> (%shared) {inputTaps = [0 : i32], level = 1 : i32}
        : (memref<8xf32, #hls.mem<dram>>) -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%i: memref<8xf32, #hls.mem<dram>>, %o: memref<8xf32, #hls.mem<dram>>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32, #hls.mem<dram>>
        affine.store %v, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }

    hls.dataflow.node(%in) -> (%shared) {inputTaps = [0 : i32], level = 0 : i32}
        : (memref<8xf32, #hls.mem<dram>>) -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%i: memref<8xf32, #hls.mem<dram>>, %o: memref<8xf32, #hls.mem<dram>>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32, #hls.mem<dram>>
        affine.store %v, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
  }
  return
}
