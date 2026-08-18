// RUN: sar-translate --hls-emit-hlscpp -emit-vitis-directives %s | FileCheck %s

// A non-dataflow top keeps the plain interface pragmas: outside a
// dataflow region `stable` asserts nothing the checker wants, so the
// emitter does not claim it.

// CHECK:      void plain(
// CHECK:        #pragma HLS interface bram port=in0
// CHECK-NOT:    #pragma HLS stable
func.func @plain(%f: memref<4x4xf64>, %out: memref<4x4xf64>)
    attributes {top_func} {
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 4 {
      %x = affine.load %f[%i, %j] : memref<4x4xf64>
      affine.store %x, %out[%i, %j] : memref<4x4xf64>
    }
  }
  return
}
