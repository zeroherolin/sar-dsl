// RUN: sar-opt %s --hls-lower-dataflow | FileCheck %s

// Lowering from task level to node level is where implicit dataflow becomes
// explicit: a dispatch's live-ins become schedule block arguments (the region
// turns isolated), and each task's live-ins are classified by use -- written
// buffers become node outputs, read-only ones inputs -- which is the producer/
// consumer information every later dataflow pass keys on. Local allocs must
// come out as dataflow buffers so they participate in that graph.

// CHECK-LABEL: func.func @lower
func.func @lower(%in: memref<8xf32>, %out: memref<8xf32>) attributes {top_func} {
  // CHECK: hls.dataflow.schedule(%{{.*}}, %{{.*}}) : memref<8xf32>, memref<8xf32> {
  // CHECK-NEXT: ^bb0(%[[A:.*]]: memref<8xf32>, %[[B:.*]]: memref<8xf32>):
  hls.dataflow.dispatch {
    %tmp = memref.alloc() : memref<8xf32>
    // CHECK: %[[BUF:.*]] = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // %in is only read: it becomes the node's input; %tmp is written: output.
    // CHECK: hls.dataflow.node(%[[A]]) -> (%[[BUF]])
    hls.dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf32>
        %w = arith.mulf %v, %v : f32
        affine.store %w, %tmp[%i] : memref<8xf32>
      }
    }
    // CHECK: hls.dataflow.node(%[[BUF]]) -> (%[[B]])
    hls.dataflow.task {
      affine.for %i = 0 to 8 {
        %v = affine.load %tmp[%i] : memref<8xf32>
        %w = arith.addf %v, %v : f32
        affine.store %w, %out[%i] : memref<8xf32>
      }
    }
  }
  // No task, dispatch, or raw alloc may survive: they are illegal targets.
  // CHECK-NOT: hls.dataflow.task
  // CHECK-NOT: hls.dataflow.dispatch
  // CHECK-NOT: memref.alloc
  return
}
