// RUN: sar-opt %s --hls-parallelize-dataflow-node="max-unroll-factor=2 complexity-aware=false correlation-aware=false" \
// RUN:   | FileCheck %s
// RUN: sar-opt %s --hls-parallelize-dataflow-node="max-unroll-factor=2 complexity-aware=true correlation-aware=false" \
// RUN:   | FileCheck %s --check-prefix=COMPLEX

// In naive mode every node's loop band is unroll-jammed by exactly the given
// factor: the loop steps by 2 with two access pairs per iteration. The
// replacement constants the unroller creates must also stay local to the node
// region -- hoisting them to the function entry would break node isolation.

// CHECK-LABEL: func.func @unroll
func.func @unroll(%arg0: memref<8xf32>) attributes {top_func} {
  // CHECK: hls.dataflow.schedule
  // CHECK-NEXT: ^bb0
  // CHECK-NOT: arith.constant
  // CHECK: hls.dataflow.node
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      // CHECK: affine.for %{{.*}} = 0 to 8 step 2
      // CHECK-COUNT-2: affine.load
      // CHECK-NOT: affine.load
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
  }
  return
}

// A schedule with no arithmetic complexity must not divide by zero.
// COMPLEX-LABEL: func.func @straight_line
// COMPLEX: memref.copy
func.func @straight_line(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      memref.copy %i, %o : memref<8xf32> to memref<8xf32>
    }
  }
  return
}
