// RUN: sar-translate --hls-emit-hlscpp %s | FileCheck %s

module {
  func.func @disjoint(%arena: memref<32xf32>) attributes {top_func} {
    affine.for %i = 0 to 16 {
      %read = affine.load %arena[%i] : memref<32xf32>
      affine.store %read, %arena[%i + 16] : memref<32xf32>
    } {loop_directive = #hls.loop<pipeline = true, target_ii = 1>}
    return
  }
}

// CHECK: #pragma HLS dependence variable=inout0 inter false
// CHECK-NEXT: #pragma HLS pipeline II=1
