// RUN: sar-opt %s --convert-sar-signal-to-runtime | FileCheck %s

// Runtime declarations are inserted at the module top (newest first), so
// they are order-independent CHECK-DAGs.
// CHECK-DAG: func.func private @sar_rt_fft_2d_c64(memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>, memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>, i64, i1) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @sar_rt_fft_1d_c128(memref<?xcomplex<f64>, strided<[?], offset: ?>>, memref<?xcomplex<f64>, strided<[?], offset: ?>>, i64, i1) attributes {llvm.emit_c_interface}
// CHECK-DAG: func.func private @sar_rt_interp1d_2d_c64(memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>, memref<?x?xf64, strided<[?, ?], offset: ?>>, memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>, i64, i64, i64, f64) attributes {llvm.emit_c_interface}

// CHECK-LABEL: func.func @fft2d
func.func @fft2d(%x: tensor<16x32xcomplex<f32>>) -> tensor<16x32xcomplex<f32>> {
  // CHECK: %[[IN:.*]] = bufferization.to_buffer
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-DAG: %[[DIM:.*]] = arith.constant 1 : i64
  // CHECK-DAG: %[[INV:.*]] = arith.constant false
  // CHECK: call @sar_rt_fft_2d_c64(%{{.*}}, %{{.*}}, %[[DIM]], %[[INV]])
  // CHECK: %[[RES:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
  %0 = sar.fft %x {dim = 1 : i64} : tensor<16x32xcomplex<f32>>
  // CHECK: return %[[RES]]
  return %0 : tensor<16x32xcomplex<f32>>
}

// CHECK-LABEL: func.func @ifft1d
func.func @ifft1d(%x: tensor<64xcomplex<f64>>) -> tensor<64xcomplex<f64>> {
  // CHECK: arith.constant true
  // CHECK: call @sar_rt_fft_1d_c128
  %0 = sar.ifft %x {dim = 0 : i64} : tensor<64xcomplex<f64>>
  return %0 : tensor<64xcomplex<f64>>
}

// CHECK-LABEL: func.func @interp1d
func.func @interp1d(%data: tensor<8x16xcomplex<f32>>,
                     %pos: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f32>> {
  // CHECK: bufferization.to_buffer
  // CHECK: bufferization.to_buffer
  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-DAG: %[[KERNEL:.*]] = arith.constant 3 : i64
  // CHECK-DAG: %[[TAPS:.*]] = arith.constant 8 : i64
  // CHECK-DAG: %[[WINDOW:.*]] = arith.constant 1 : i64
  // CHECK-DAG: %[[BETA:.*]] = arith.constant 2.500000e+00 : f64
  // CHECK: call @sar_rt_interp1d_2d_c64(%{{.*}}, %{{.*}}, %{{.*}}, %[[KERNEL]], %[[TAPS]], %[[WINDOW]], %[[BETA]])
  // CHECK: %[[RES:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable
  %0 = sar.interp1d %data, %pos {kernel = "sinc", taps = 8 : i64}
      : (tensor<8x16xcomplex<f32>>, tensor<8x16xf64>)
      -> tensor<8x16xcomplex<f32>>
  // CHECK: return %[[RES]]
  return %0 : tensor<8x16xcomplex<f32>>
}
