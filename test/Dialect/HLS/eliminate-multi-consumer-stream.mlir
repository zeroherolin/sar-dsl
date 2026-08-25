// RUN: sar-opt %s --hls-eliminate-multi-consumer | FileCheck %s

// A scalar stream with two readers needs a stream fork. Creating BufferOp and
// memref.copy for this value would be invalid because it has no memref.

// CHECK-LABEL: func.func @stream_fanout
// CHECK-COUNT-3: hls.dataflow.stream
// CHECK: hls.dataflow.node(%{{.*}}) -> (%{{.*}}, %{{.*}})
// CHECK: %[[VALUE:.*]] = hls.dataflow.stream_read
// CHECK-COUNT-2: hls.dataflow.stream_write {{.*}}, %[[VALUE]]
// CHECK-NOT: memref.copy
func.func @stream_fanout(%in: memref<1xf32>, %out0: memref<1xf32>,
                         %out1: memref<1xf32>) attributes {top_func} {
  hls.dataflow.schedule(%in, %out0, %out1)
      : memref<1xf32>, memref<1xf32>, memref<1xf32> {
  ^bb0(%input: memref<1xf32>, %output0: memref<1xf32>,
       %output1: memref<1xf32>):
    %channel = hls.dataflow.stream {depth = 2 : i32} : <f32, 2>
    hls.dataflow.node(%input) -> (%channel) {inputTaps = [0 : i32]}
        : (memref<1xf32>) -> !hls.stream<f32, 2> {
    ^bb0(%arg0: memref<1xf32>, %arg1: !hls.stream<f32, 2>):
      %value = affine.load %arg0[0] : memref<1xf32>
      hls.dataflow.stream_write %arg1, %value : !hls.stream<f32, 2>, f32
    }
    hls.dataflow.node(%channel) -> (%output0) {inputTaps = [0 : i32]}
        : (!hls.stream<f32, 2>) -> memref<1xf32> {
    ^bb0(%arg0: !hls.stream<f32, 2>, %arg1: memref<1xf32>):
      %value = hls.dataflow.stream_read %arg0
          : (!hls.stream<f32, 2>) -> f32
      affine.store %value, %arg1[0] : memref<1xf32>
    }
    hls.dataflow.node(%channel) -> (%output1) {inputTaps = [0 : i32]}
        : (!hls.stream<f32, 2>) -> memref<1xf32> {
    ^bb0(%arg0: !hls.stream<f32, 2>, %arg1: memref<1xf32>):
      %value = hls.dataflow.stream_read %arg0
          : (!hls.stream<f32, 2>) -> f32
      affine.store %value, %arg1[0] : memref<1xf32>
    }
  }
  return
}

// A multi-token stream gets a counted fork, not a one-token body.
// CHECK-LABEL: func.func @stream_loop_fanout
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: %[[LOOP_VALUE:.*]] = hls.dataflow.stream_read
// CHECK-COUNT-2: hls.dataflow.stream_write {{.*}}, %[[LOOP_VALUE]]
func.func @stream_loop_fanout(%in: memref<4xf32>, %out0: memref<4xf32>,
                              %out1: memref<4xf32>) attributes {top_func} {
  hls.dataflow.schedule(%in, %out0, %out1)
      : memref<4xf32>, memref<4xf32>, memref<4xf32> {
  ^bb0(%input: memref<4xf32>, %output0: memref<4xf32>,
       %output1: memref<4xf32>):
    %channel = hls.dataflow.stream {depth = 2 : i32} : <f32, 2>
    hls.dataflow.node(%input) -> (%channel) {inputTaps = [0 : i32]}
        : (memref<4xf32>) -> !hls.stream<f32, 2> {
    ^bb0(%arg0: memref<4xf32>, %arg1: !hls.stream<f32, 2>):
      affine.for %i = 0 to 4 {
        %value = affine.load %arg0[%i] : memref<4xf32>
        hls.dataflow.stream_write %arg1, %value : !hls.stream<f32, 2>, f32
      }
    }
    hls.dataflow.node(%channel) -> (%output0) {inputTaps = [0 : i32]}
        : (!hls.stream<f32, 2>) -> memref<4xf32> {
    ^bb0(%arg0: !hls.stream<f32, 2>, %arg1: memref<4xf32>):
      affine.for %i = 0 to 4 {
        %value = hls.dataflow.stream_read %arg0
            : (!hls.stream<f32, 2>) -> f32
        affine.store %value, %arg1[%i] : memref<4xf32>
      }
    }
    hls.dataflow.node(%channel) -> (%output1) {inputTaps = [0 : i32]}
        : (!hls.stream<f32, 2>) -> memref<4xf32> {
    ^bb0(%arg0: !hls.stream<f32, 2>, %arg1: memref<4xf32>):
      affine.for %i = 0 to 4 {
        %value = hls.dataflow.stream_read %arg0
            : (!hls.stream<f32, 2>) -> f32
        affine.store %value, %arg1[%i] : memref<4xf32>
      }
    }
  }
  return
}
