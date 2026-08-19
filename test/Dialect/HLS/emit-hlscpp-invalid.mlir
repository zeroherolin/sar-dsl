// RUN: ! sar-translate --hls-emit-hlscpp %s 2>&1 | FileCheck %s

func.func @unordered_compare(%lhs: f32, %rhs: f32) attributes {top_func} {
  // CHECK: error: HLS C++ target requires explicit NaN handling
  %0 = arith.cmpf ult, %lhs, %rhs : f32
  return
}
