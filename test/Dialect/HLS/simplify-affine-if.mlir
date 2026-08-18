// RUN: sar-opt %s --hls-simplify-affine-if | FileCheck %s

// Perfection and unrolling guard sunk operations with affine ifs; once the
// surrounding bounds make a guard decidable it is pure overhead -- a branch
// the schedule must still allocate. An infeasible condition must collapse to
// the else region, a constant-true one to the then region.

// The condition d0 - 8 >= 0 has no solution under the loop domain 0..8:
// only the else branch survives, unwrapped into the loop body.

// CHECK-LABEL: func.func @never_taken
// CHECK: %[[C:.*]] = arith.constant 2.0
// CHECK: affine.for
// CHECK-NEXT: affine.store %[[C]]
// CHECK-NOT: affine.if
#never = affine_set<(d0) : (d0 - 8 >= 0)>
func.func @never_taken(%m: memref<8xf32>) {
  %c1 = arith.constant 1.0 : f32
  %c2 = arith.constant 2.0 : f32
  affine.for %i = 0 to 8 {
    affine.if #never(%i) {
      affine.store %c1, %m[%i] : memref<8xf32>
    } else {
      affine.store %c2, %m[%i] : memref<8xf32>
    }
  }
  return
}

// A constant condition needs no domain reasoning at all: the then block is
// spliced into the parent and the if disappears.

// CHECK-LABEL: func.func @always_taken
// CHECK: affine.for
// CHECK-NEXT: affine.store
// CHECK-NOT: affine.if
#true = affine_set<() : (0 == 0)>
func.func @always_taken(%m: memref<8xf32>) {
  %c = arith.constant 1.0 : f32
  affine.for %i = 0 to 8 {
    affine.if #true() {
      affine.store %c, %m[%i] : memref<8xf32>
    }
  }
  return
}

// Identical guards separated by an effectful operation must retain their
// original ordering.

#nonnegative = affine_set<(d0) : (d0 >= 0)>
func.func private @observe(memref<8xf32>)

// CHECK-LABEL: func.func @effect_between_guards
// CHECK: affine.if
// CHECK: call @observe
// CHECK: affine.if
func.func @effect_between_guards(%m: memref<8xf32>, %i: index) {
  %c = arith.constant 1.0 : f32
  affine.if #nonnegative(%i) {
    affine.store %c, %m[%i] : memref<8xf32>
  }
  func.call @observe(%m) : (memref<8xf32>) -> ()
  affine.if #nonnegative(%i) {
    affine.store %c, %m[%i] : memref<8xf32>
  }
  return
}
