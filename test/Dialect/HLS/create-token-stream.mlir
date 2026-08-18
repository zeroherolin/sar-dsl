// RUN: sar-opt %s --hls-create-token-stream | FileCheck %s

// A DRAM buffer is persistent memory, not a channel: the schedule carries no
// structural edge between its producer and consumer. The ordering must be
// carried by a token whose FIFO depth equals the level distance, so the
// producer can run that many beats ahead without the consumer starting early.

// CHECK-LABEL: func.func @token
func.func @token(%arg0: memref<8xf32, #hls.mem<dram>>,
                 %arg1: memref<8xf32, #hls.mem<dram>>) attributes {top_func} {
  hls.dataflow.schedule legal (%arg0, %arg1)
      : memref<8xf32, #hls.mem<dram>>, memref<8xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<8xf32, #hls.mem<dram>>, %out: memref<8xf32, #hls.mem<dram>>):
    %dram = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32, #hls.mem<dram>>
    // Producer at level 2, consumer at level 0: the token depth must be the
    // level difference, and the consumer taps it one beat before the end.
    // CHECK: %[[TOK:.*]] = hls.dataflow.stream {depth = 2 : i32} : <i1, 2>
    // CHECK: hls.dataflow.node(%{{.*}}) -> (%[[TOK]], %{{.*}})
    // CHECK: hls.dataflow.stream_write %{{.*}}, %true
    hls.dataflow.node(%in) -> (%dram) {inputTaps = [0 : i32], level = 2 : i32}
        : (memref<8xf32, #hls.mem<dram>>) -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%i: memref<8xf32, #hls.mem<dram>>, %o: memref<8xf32, #hls.mem<dram>>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32, #hls.mem<dram>>
        affine.store %v, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
    // CHECK: hls.dataflow.node(%[[TOK]], %{{.*}}) -> (%{{.*}}) {inputTaps = [1 : i32, 0 : i32]
    // CHECK-NEXT: ^bb0
    // CHECK-NEXT: hls.dataflow.stream_read
    hls.dataflow.node(%dram) -> (%out) {inputTaps = [0 : i32], level = 0 : i32}
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

// Until the schedule is marked legal the levels are not final: tokens created
// now would encode a stale ordering. An unmarked schedule must be left alone.

// CHECK-LABEL: func.func @not_legal
// CHECK-NOT: hls.dataflow.stream
func.func @not_legal(%arg0: memref<8xf32, #hls.mem<dram>>,
                     %arg1: memref<8xf32, #hls.mem<dram>>) attributes {top_func} {
  hls.dataflow.schedule(%arg0, %arg1)
      : memref<8xf32, #hls.mem<dram>>, memref<8xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<8xf32, #hls.mem<dram>>, %out: memref<8xf32, #hls.mem<dram>>):
    %dram = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32, #hls.mem<dram>>
    hls.dataflow.node(%in) -> (%dram) {inputTaps = [0 : i32], level = 1 : i32}
        : (memref<8xf32, #hls.mem<dram>>) -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%i: memref<8xf32, #hls.mem<dram>>, %o: memref<8xf32, #hls.mem<dram>>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32, #hls.mem<dram>>
        affine.store %v, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
    hls.dataflow.node(%dram) -> (%out) {inputTaps = [0 : i32], level = 0 : i32}
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
