// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

func.func @float_semantics(%lhs: memref<1xf32>, %rhs: memref<1xf32>,
                           %out: memref<1xf32>) attributes {top_func} {
  %c0 = arith.constant 0 : index
  %a = memref.load %lhs[%c0] : memref<1xf32>
  %b = memref.load %rhs[%c0] : memref<1xf32>
  // CHECK: std::fmod(
  %remainder = arith.remf %a, %b : f32
  %nan = arith.constant 0x7FC00000 : f32
  // CHECK: std::sin(
  // CHECK: std::cos(
  %sin = math.sin %a : f32
  %cos = math.cos %a : f32
  %trig = arith.addf %sin, %cos : f32
  // CHECK: std::sqrt(
  %root = math.sqrt %trig : f32
  %partial = arith.addf %remainder, %root : f32
  // CHECK: (float)NAN
  %result = arith.addf %partial, %nan : f32
  memref.store %result, %out[%c0] : memref<1xf32>
  return
}
