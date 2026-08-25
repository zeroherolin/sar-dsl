// RUN: sar-opt %s --hls-affine-store-forward | FileCheck %s

// Forwarding is what lets a scratch buffer disappear entirely: once the load
// takes the stored value directly, the memref is left with only stores and
// the pass sweeps buffer, stores, and alloc away -- one fewer memory to bank.

// CHECK-LABEL: func.func @forward
func.func @forward(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK-NOT: memref.alloc
  %tmp = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    // CHECK: %[[V:.*]] = affine.load %arg0[%{{.*}}]
    // CHECK-NEXT: arith.addf %[[V]], %[[V]]
    // CHECK-NEXT: affine.store
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %tmp[%i] : memref<8xf32>
    %w = affine.load %tmp[%i] : memref<8xf32>
    %x = arith.addf %w, %w : f32
    affine.store %x, %out[%i] : memref<8xf32>
  }
  return
}

// The case plain scalar replacement cannot handle: a store guarded by an if
// still forwards
// -- the branch collapses into a select between the stored value and what the
// memory held, so the load after the if is resolved without a branch.

// CHECK-LABEL: func.func @conditional_forward
#half = affine_set<(d0) : (d0 - 4 >= 0)>
func.func @conditional_forward(%in: memref<8xf32>, %out: memref<8xf32>) {
  %tmp = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    // CHECK: %[[T:.*]] = affine.load %arg0
    // CHECK: %[[F:.*]] = affine.load %{{.*}}
    // CHECK: %[[S:.*]] = hls.affine.select #{{.*}}(%{{.*}}) %[[T]], %[[F]]
    // CHECK-NOT: affine.if
    %v = affine.load %in[%i] : memref<8xf32>
    affine.if #half(%i) {
      affine.store %v, %tmp[%i] : memref<8xf32>
    }
    %w = affine.load %tmp[%i] : memref<8xf32>
    affine.store %w, %out[%i] : memref<8xf32>
  }
  return
}

// A store fully overwritten before any read never happens in hardware either:
// only the surviving store may remain.

// CHECK-LABEL: func.func @dead_store
func.func @dead_store(%in: memref<8xf32>, %out: memref<8xf32>) {
  %c = arith.constant 0.0 : f32
  affine.for %i = 0 to 8 {
    // CHECK: affine.load %arg0
    // CHECK-NEXT: affine.store
    // CHECK-NOT: affine.store
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %c, %out[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}
