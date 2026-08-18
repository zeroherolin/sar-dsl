// RUN: sar-opt %s --sar-demote-loop-carries --canonicalize | FileCheck %s

// The carry becomes side effects: the body iterates in the init buffer, a
// copy makes each iteration's result current, and canonicalization folds
// the dead iter_args away -- what the HLS dataflow model requires.

// CHECK-LABEL: func.func @carried
// CHECK: %[[INIT:.*]] = memref.alloc
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: memref.copy %{{.*}}, %[[INIT]]
// CHECK: return %[[INIT]]
func.func @carried(%f: memref<4x4xf64>) -> memref<4x4xf64> {
  %init = memref.alloc() : memref<4x4xf64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1
      iter_args(%acc = %init) -> (memref<4x4xf64>) {
    %next = memref.alloc() : memref<4x4xf64>
    affine.for %a = 0 to 4 {
      affine.for %b = 0 to 4 {
        %x = affine.load %acc[%a, %b] : memref<4x4xf64>
        %y = affine.load %f[%a, %b] : memref<4x4xf64>
        %m = arith.mulf %x, %y : f64
        affine.store %m, %next[%a, %b] : memref<4x4xf64>
      }
    }
    scf.yield %next : memref<4x4xf64>
  }
  return %r : memref<4x4xf64>
}

// An init that is a function argument is a top-level port of the design,
// and a dataflow node may not read and write the same port. The loop
// iterates in a local buffer primed from the port instead: the port keeps
// a single reader (the priming copy) and stays read-only.

// CHECK-LABEL: func.func @port_init
// CHECK: %[[LOCAL:.*]] = memref.alloc
// CHECK-NEXT: memref.copy %{{.*}}, %[[LOCAL]]
// CHECK: scf.for
// CHECK-NOT: iter_args
// CHECK: memref.copy %{{.*}}, %[[LOCAL]]
// CHECK: return %[[LOCAL]]
func.func @port_init(%z: memref<4x4xf64>, %f: memref<4x4xf64>)
    -> memref<4x4xf64> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %r = scf.for %i = %c0 to %c8 step %c1
      iter_args(%acc = %z) -> (memref<4x4xf64>) {
    %next = memref.alloc() : memref<4x4xf64>
    affine.for %a = 0 to 4 {
      affine.for %b = 0 to 4 {
        %x = affine.load %acc[%a, %b] : memref<4x4xf64>
        %y = affine.load %f[%a, %b] : memref<4x4xf64>
        %m = arith.mulf %x, %y : f64
        affine.store %m, %next[%a, %b] : memref<4x4xf64>
      }
    }
    scf.yield %next : memref<4x4xf64>
  }
  return %r : memref<4x4xf64>
}
