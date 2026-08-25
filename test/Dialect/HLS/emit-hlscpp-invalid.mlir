// RUN: ! sar-translate --hls-emit-hlscpp %s 2>&1 | FileCheck %s

func.func @unordered_compare(%lhs: f32, %rhs: f32) attributes {top_func} {
  // CHECK: error: HLS C++ target requires explicit NaN handling
  %0 = arith.cmpf ult, %lhs, %rhs : f32
  return
}

// CHECK: error: HLS C++ target does not support external functions
func.func private @external()

func.func @calls_external() {
  // CHECK: error: HLS C++ target requires every call to resolve
  call @external() : () -> ()
  return
}

func.func @recursive() {
  // CHECK: error: HLS C++ target does not support recursive calls
  call @recursive() : () -> ()
  return
}

func.func @parallel() {
  // CHECK: error: HLS C++ target requires affine.parallel to be lowered
  affine.parallel (%i) = (0) to (4) {
    affine.yield
  }
  return
}
