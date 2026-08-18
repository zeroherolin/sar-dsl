// RUN: sar-opt %s --hls-stream-dataflow-task | FileCheck %s

// A scalar crossing a task boundary has no memory to live in once tasks
// become concurrent stages: it must travel through a stream channel. The
// channel is hoisted out of the producing task so both stages see it, the
// producer writes it before its yield, and every consuming task reads it
// on entry.

// CHECK-LABEL: func.func @scalar_crossing
func.func @scalar_crossing(%in: memref<8xf32>, %out: memref<8xf32>)
    attributes {top_func} {
  hls.dataflow.dispatch {
    // CHECK: %[[CH:.*]] = hls.dataflow.stream {depth = 1 : i32} : <f32, 1>
    // CHECK: hls.dataflow.task : !hls.stream<f32, 1>
    %sum = hls.dataflow.task : f32 {
      %init = arith.constant 0.0 : f32
      %r = affine.for %i = 0 to 8 iter_args(%acc = %init) -> f32 {
        %v = affine.load %in[%i] : memref<8xf32>
        %n = arith.addf %acc, %v : f32
        affine.yield %n : f32
      }
      // CHECK: hls.dataflow.stream_write %[[CH]], %{{.*}}
      // CHECK-NEXT: hls.dataflow.yield %[[CH]]
      hls.dataflow.yield %r : f32
    }
    // CHECK: hls.dataflow.task {
    // CHECK-NEXT: %[[V:.*]] = hls.dataflow.stream_read %[[CH]]
    // CHECK: arith.addf %{{.*}}, %[[V]]
    hls.dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf32>
        %w = arith.addf %v, %sum : f32
        affine.store %w, %out[%i] : memref<8xf32>
      }
    }
  }
  return
}
