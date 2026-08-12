// RUN: sar-opt %s | sar-opt | FileCheck %s
// Round-trip test for every SAR operation (custom assembly).

// CHECK-LABEL: func.func @elementwise
func.func @elementwise(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sar.add %{{.*}}, %{{.*}} : tensor<4x8xf32>
  %0 = sar.add %a, %b : tensor<4x8xf32>
  // CHECK: sar.sub
  %1 = sar.sub %0, %b : tensor<4x8xf32>
  // CHECK: sar.mul
  %2 = sar.mul %1, %b : tensor<4x8xf32>
  // CHECK: sar.div
  %3 = sar.div %2, %b : tensor<4x8xf32>
  // CHECK: sar.add_scalar %{{.*}}, 1.500000e+00 : tensor<4x8xf32>
  %4 = sar.add_scalar %3, 1.5 : tensor<4x8xf32>
  // CHECK: sar.mul_scalar
  %5 = sar.mul_scalar %4, 2.0 : tensor<4x8xf32>
  // CHECK: sar.max_scalar
  %6 = sar.max_scalar %5, 1.0e-10 : tensor<4x8xf32>
  // CHECK: sar.neg
  %7 = sar.neg %6 : tensor<4x8xf32>
  // CHECK: sar.sqrt
  %8 = sar.sqrt %7 : tensor<4x8xf32>
  return %8 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @complex_ops
func.func @complex_ops(%p: tensor<16x16xf64>, %d: tensor<16x16xcomplex<f64>>)
    -> tensor<16x16xf32> {
  // CHECK: sar.expj %{{.*}} : tensor<16x16xf64> -> tensor<16x16xcomplex<f64>>
  %0 = sar.expj %p : tensor<16x16xf64> -> tensor<16x16xcomplex<f64>>
  // CHECK: sar.mul
  %1 = sar.mul %d, %0 : tensor<16x16xcomplex<f64>>
  // CHECK: sar.abs %{{.*}} : tensor<16x16xcomplex<f64>> -> tensor<16x16xf64>
  %2 = sar.abs %1 : tensor<16x16xcomplex<f64>> -> tensor<16x16xf64>
  // CHECK: sar.cast %{{.*}} : tensor<16x16xf64> -> tensor<16x16xf32>
  %3 = sar.cast %2 : tensor<16x16xf64> -> tensor<16x16xf32>
  return %3 : tensor<16x16xf32>
}

// CHECK-LABEL: func.func @structure
func.func @structure(%m: tensor<4x8xf32>, %v: tensor<8xf32>) -> tensor<8x4xf32> {
  // CHECK: sar.broadcast %{{.*}} {dim = 1 : i64} : tensor<8xf32> -> tensor<4x8xf32>
  %0 = sar.broadcast %v {dim = 1 : i64} : tensor<8xf32> -> tensor<4x8xf32>
  // CHECK: sar.mul
  %1 = sar.mul %m, %0 : tensor<4x8xf32>
  // CHECK: sar.fftshift %{{.*}} {dim = 1 : i64} : tensor<4x8xf32>
  %2 = sar.fftshift %1 {dim = 1 : i64} : tensor<4x8xf32>
  // CHECK: sar.fftshift %{{.*}} {dim = 0 : i64, inverse} : tensor<4x8xf32>
  %3 = sar.fftshift %2 {dim = 0 : i64, inverse} : tensor<4x8xf32>
  // CHECK: sar.transpose %{{.*}} : tensor<4x8xf32> -> tensor<8x4xf32>
  %4 = sar.transpose %3 : tensor<4x8xf32> -> tensor<8x4xf32>
  return %4 : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @signal
func.func @signal(%x: tensor<16x32xcomplex<f32>>, %fa: tensor<16xf64>,
                  %fr: tensor<32xf64>) -> tensor<16x32xcomplex<f32>> {
  // CHECK: sar.fft %{{.*}} {dim = 1 : i64} : tensor<16x32xcomplex<f32>>
  %0 = sar.fft %x {dim = 1 : i64} : tensor<16x32xcomplex<f32>>
  // CHECK: sar.ifft %{{.*}} {dim = 0 : i64} : tensor<16x32xcomplex<f32>>
  %1 = sar.ifft %0 {dim = 0 : i64} : tensor<16x32xcomplex<f32>>
  // CHECK: sar.stolt_interp %{{.*}}, %{{.*}}, %{{.*}} {c = {{.*}}, fc = {{.*}}, t_shift = {{.*}}, vr = {{.*}}}
  %2 = sar.stolt_interp %1, %fa, %fr {c = 2.99792458e8, fc = 1.27e9,
        vr = 7100.0, t_shift = 1.5e-4}
      : (tensor<16x32xcomplex<f32>>, tensor<16xf64>, tensor<32xf64>)
      -> (tensor<16x32xcomplex<f32>>)
  return %2 : tensor<16x32xcomplex<f32>>
}

// CHECK-LABEL: func.func @constants
func.func @constants() -> tensor<2x2xf32> {
  // CHECK: sar.constant dense<
  %0 = sar.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
