// RUN: sar-opt %s --hls-affine-loop-order-opt | FileCheck %s

// The pass exists to stretch loop-carried dependencies: a reduction sitting
// innermost forces each iteration to wait on the previous accumulation. With
// the reduction loop moved outermost, consecutive iterations of the new inner
// loop touch different accumulators, so the recurrence is 8 iterations apart
// instead of back-to-back.

// CHECK-LABEL: func.func @reduction_innermost
// CHECK: affine.for %[[K:.*]] = 0 to 16
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to 8
// CHECK: affine.load %{{.*}}[%[[I]], %[[K]]]
func.func @reduction_innermost(%a: memref<8x16xf32>, %b: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %k = 0 to 16 {
      %v = affine.load %a[%i, %k] : memref<8x16xf32>
      %s = affine.load %b[%i] : memref<8xf32>
      %n = arith.addf %s, %v : f32
      affine.store %n, %b[%i] : memref<8xf32>
    }
  }
  return
}

// With no loop-carried dependence there is nothing to stretch: a fully
// parallel nest keeps its order.

// CHECK-LABEL: func.func @all_parallel
// CHECK: affine.for %{{.*}} = 0 to 8
// CHECK-NEXT: affine.for %{{.*}} = 0 to 16
func.func @all_parallel(%a: memref<8x16xf32>, %b: memref<8x16xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 16 {
      %v = affine.load %a[%i, %j] : memref<8x16xf32>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %b[%i, %j] : memref<8x16xf32>
    }
  }
  return
}
