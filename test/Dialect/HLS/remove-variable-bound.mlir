// RUN: sar-opt %s --hls-remove-variable-bound | FileCheck %s

// HLS scheduling needs constant trip counts. When an inner bound is an affine
// expression of an outer induction variable, the loop can run to the worst
// case with the original bound re-imposed as a guard: same iterations
// executed, but the loop itself is now schedulable.

// CHECK: #set = affine_set<(d0, d1) : (d0 - d1 >= 0)>

// CHECK-LABEL: func.func @triangular
// CHECK: affine.for %[[I:.*]] = 0 to 8
// CHECK-NEXT: affine.for %[[J:.*]] = 0 to 8
// CHECK-NEXT: affine.if #set(%[[I]], %[[J]])
// CHECK-NEXT: affine.store
#map = affine_map<(d0) -> (d0 + 1)>
func.func @triangular(%m: memref<8x8xf32>) {
  %c = arith.constant 1.0 : f32
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to #map(%i) {
      affine.store %c, %m[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// A bound driven by a function argument has no worst case the pass can
// compute: the loop must keep its variable bound.

// CHECK-LABEL: func.func @symbolic_bound
// CHECK: affine.for %{{.*}} = 0 to %
// CHECK-NOT: affine.if
func.func @symbolic_bound(%m: memref<8xf32>, %n: index) {
  %c = arith.constant 1.0 : f32
  affine.for %j = 0 to %n {
    affine.store %c, %m[%j] : memref<8xf32>
  }
  return
}

// Modulo expressions can reach extrema away from the iteration-box corners,
// so the pass must leave this bound unchanged.

#mod = affine_map<(d0) -> ((d0 mod 4) + 1)>
// CHECK-LABEL: func.func @nonlinear_bound
// CHECK: affine.for %{{.*}} = 1 to 7
// CHECK: affine.for %{{.*}} = 0 to #{{.*}}(%{{.*}})
// CHECK-NOT: affine.if
func.func @nonlinear_bound(%m: memref<8x8xf32>) {
  %c = arith.constant 1.0 : f32
  affine.for %i = 1 to 7 {
    affine.for %j = 0 to #mod(%i) {
      affine.store %c, %m[%i, %j] : memref<8x8xf32>
    }
  }
  return
}
