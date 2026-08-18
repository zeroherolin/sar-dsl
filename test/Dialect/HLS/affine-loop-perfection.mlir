// RUN: sar-opt %s --hls-affine-loop-perfection | FileCheck %s

// Later loop passes (order opt, tiling, pipelining) only operate on perfect
// bands, so perfection has to sink loop-level prefix and suffix operations
// into the innermost body -- guarded so they still execute exactly once: a
// write before the inner loop runs only on its first iteration, one after it
// only on the last.

// CHECK: #set = affine_set<(d0) : (d0 == 0)>
// CHECK: #set1 = affine_set<(d0) : (-d0 + 7 == 0)>

// CHECK-LABEL: func.func @imperfect_sum
func.func @imperfect_sum(%a: memref<8x8xf32>, %b: memref<8xf32>,
                         %out: memref<8xf32>) {
  %zero = arith.constant 0.0 : f32
  // The two loops must end up perfectly nested: nothing between them.
  // CHECK: affine.for %{{.*}} = 0 to 8
  // CHECK-NEXT: affine.for %[[J:.*]] = 0 to 8
  affine.for %i = 0 to 8 {
    // Prefix store: runs only when the inner loop is at its first iteration.
    // CHECK-NEXT: affine.if #set(%[[J]])
    // CHECK-NEXT: affine.store
    affine.store %zero, %b[%i] : memref<8xf32>
    affine.for %j = 0 to 8 {
      %v = affine.load %a[%i, %j] : memref<8x8xf32>
      %s = affine.load %b[%i] : memref<8xf32>
      %n = arith.addf %s, %v : f32
      affine.store %n, %b[%i] : memref<8xf32>
    }
    // Suffix store: guarded to the last iteration of the inner loop.
    // CHECK: arith.mulf
    // CHECK-NEXT: affine.if #set1(%[[J]])
    // CHECK-NEXT: affine.store
    %r = affine.load %b[%i] : memref<8xf32>
    %d = arith.mulf %r, %r : f32
    affine.store %d, %out[%i] : memref<8xf32>
  }
  return
}
