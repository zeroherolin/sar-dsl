// RUN: sar-opt %s --sar-to-llvm-pipeline | FileCheck %s
// Full CPU lowering smoke test: everything must reach the LLVM dialect,
// kernels become destination-passing-style C-interface functions.

func.func @kernel(%x: tensor<8x16xcomplex<f32>>, %s: tensor<8x16xf32>)
    -> tensor<8x16xcomplex<f32>> {
  %0 = sar.fft %x {dim = 1 : i64} : tensor<8x16xcomplex<f32>>
  %1 = sar.fftshift %0 {dim = 1 : i64} : tensor<8x16xcomplex<f32>>
  %sc = sar.cos %s : tensor<8x16xf32>
  %ss = sar.sin %s : tensor<8x16xf32>
  %2 = sar.complex %sc, %ss : tensor<8x16xf32> -> tensor<8x16xcomplex<f32>>
  %3 = sar.mul %1, %2 : tensor<8x16xcomplex<f32>>
  return %3 : tensor<8x16xcomplex<f32>>
}

// CHECK-NOT: sar.
// CHECK-NOT: linalg.
// CHECK-NOT: memref.
// CHECK-DAG: llvm.func @_mlir_ciface_sar_rt_fft_2d_c64
// CHECK-DAG: llvm.func @_mlir_ciface_kernel
// CHECK-DAG: llvm.call @_mlir_ciface_sar_rt_fft_2d_c64
