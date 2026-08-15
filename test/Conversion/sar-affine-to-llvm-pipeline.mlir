// RUN: sar-opt %s --sar-affine-to-llvm-pipeline | FileCheck %s

// The affine path continued to LLVM: no sar/linalg/affine ops may remain
// and the kernel entry point gets a C interface wrapper.
// CHECK-NOT: sar.
// CHECK-NOT: linalg.
// CHECK-NOT: affine.for
// CHECK-LABEL: llvm.func @_mlir_ciface_affine_fft
func.func @affine_fft(%data: tensor<8x16xcomplex<f32>>)
    -> tensor<8x16xcomplex<f32>> {
  %0 = sar.fft %data {dim = 1 : i64} : tensor<8x16xcomplex<f32>>
  return %0 : tensor<8x16xcomplex<f32>>
}
