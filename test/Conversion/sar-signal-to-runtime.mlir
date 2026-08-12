// RUN: sar-opt %s --convert-sar-signal-to-runtime | FileCheck %s

// CHECK: func.func private @sar_rt_fft_2d_c64(
// CHECK-SAME: memref<?x?xcomplex<f32>, strided<[?, ?], offset: ?>>
// CHECK-SAME: attributes {llvm.emit_c_interface}

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

// CHECK-LABEL: func.func @stolt
func.func @stolt(%d: tensor<8x16xcomplex<f64>>, %fa: tensor<8xf64>,
                 %fr: tensor<16xf64>) -> tensor<8x16xcomplex<f64>> {
  // CHECK: call @sar_rt_stolt_2d_c128
  // CHECK-SAME: (memref<?x?xcomplex<f64>, strided<[?, ?], offset: ?>>, memref<?xf64, strided<[?], offset: ?>>, memref<?xf64, strided<[?], offset: ?>>, memref<?x?xcomplex<f64>, strided<[?, ?], offset: ?>>, f64, f64, f64, f64)
  %0 = sar.stolt_interp %d, %fa, %fr {c = 3.0e8, fc = 1.0e9, vr = 7000.0,
        t_shift = 1.0e-4}
      : (tensor<8x16xcomplex<f64>>, tensor<8xf64>, tensor<16xf64>)
      -> (tensor<8x16xcomplex<f64>>)
  return %0 : tensor<8x16xcomplex<f64>>
}
