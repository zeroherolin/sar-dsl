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
  // CHECK: sar.sqrt
  %8 = sar.sqrt %5 : tensor<4x8xf32>
  return %8 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @complex_ops
func.func @complex_ops(%p: tensor<16x16xf64>, %d: tensor<16x16xcomplex<f64>>)
    -> tensor<16x16xf32> {
  // CHECK: sar.complex %{{.*}}, %{{.*}} : tensor<16x16xf64> -> tensor<16x16xcomplex<f64>>
  %pc = sar.cos %p : tensor<16x16xf64>
  %ps = sar.sin %p : tensor<16x16xf64>
  %0 = sar.complex %pc, %ps : tensor<16x16xf64> -> tensor<16x16xcomplex<f64>>
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
  return %1 : tensor<16x32xcomplex<f32>>
}

// CHECK-LABEL: func.func @constants
func.func @constants() -> tensor<2x2xf32> {
  // CHECK: sar.constant dense<
  %0 = sar.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func.func @complex_access
func.func @complex_access(%z: tensor<4x8xcomplex<f32>>)
    -> tensor<4x8xcomplex<f32>> {
  // CHECK: sar.conj %{{.*}} : tensor<4x8xcomplex<f32>>
  %0 = sar.conj %z : tensor<4x8xcomplex<f32>>
  // CHECK: sar.real %{{.*}} : tensor<4x8xcomplex<f32>> -> tensor<4x8xf32>
  %1 = sar.real %0 : tensor<4x8xcomplex<f32>> -> tensor<4x8xf32>
  // CHECK: sar.imag %{{.*}} : tensor<4x8xcomplex<f32>> -> tensor<4x8xf32>
  %2 = sar.imag %0 : tensor<4x8xcomplex<f32>> -> tensor<4x8xf32>
  // CHECK: sar.atan2 %{{.*}}, %{{.*}} : tensor<4x8xf32>
  %4 = sar.atan2 %2, %1 : tensor<4x8xf32>
  // CHECK: sar.complex %{{.*}}, %{{.*}} : tensor<4x8xf32> -> tensor<4x8xcomplex<f32>>
  %5 = sar.complex %1, %4 : tensor<4x8xf32> -> tensor<4x8xcomplex<f32>>
  return %5 : tensor<4x8xcomplex<f32>>
}

// CHECK-LABEL: func.func @transcendentals
func.func @transcendentals(%x: tensor<8xf64>) -> tensor<8xf64> {
  // CHECK: sar.exp %{{.*}} : tensor<8xf64>
  %0 = sar.exp %x : tensor<8xf64>
  // CHECK: sar.log %{{.*}} : tensor<8xf64>
  %1 = sar.log %0 : tensor<8xf64>
  return %1 : tensor<8xf64>
}

// CHECK-LABEL: func.func @reductions
func.func @reductions(%x: tensor<4x8xf64>, %z: tensor<4x8xcomplex<f64>>)
    -> (tensor<4xf64>, tensor<8xf64>, tensor<4xi64>, tensor<4xcomplex<f64>>) {
  // CHECK: sar.reduce %{{.*}} {dim = 1 : i64, kind = "sum"} : tensor<4x8xf64> -> tensor<4xf64>
  %0 = sar.reduce %x {kind = "sum", dim = 1 : i64} : tensor<4x8xf64> -> tensor<4xf64>
  // CHECK: sar.reduce %{{.*}} {dim = 0 : i64, kind = "max"} : tensor<4x8xf64> -> tensor<8xf64>
  %1 = sar.reduce %x {kind = "max", dim = 0 : i64} : tensor<4x8xf64> -> tensor<8xf64>
  // CHECK: sar.argmax %{{.*}} {dim = 1 : i64} : tensor<4x8xf64> -> tensor<4xi64>
  %2 = sar.argmax %x {dim = 1 : i64} : tensor<4x8xf64> -> tensor<4xi64>
  // CHECK: sar.reduce %{{.*}} {dim = 1 : i64, kind = "sum"} : tensor<4x8xcomplex<f64>> -> tensor<4xcomplex<f64>>
  %3 = sar.reduce %z {kind = "sum", dim = 1 : i64} : tensor<4x8xcomplex<f64>> -> tensor<4xcomplex<f64>>
  return %0, %1, %2, %3 : tensor<4xf64>, tensor<8xf64>, tensor<4xi64>, tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @cumsum_f32
func.func @cumsum_f32(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: sar.cumsum %{{.*}} {dim = 1 : i64} : tensor<4x8xf32>
  %0 = sar.cumsum %x {dim = 1 : i64} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @cumsum_complex
func.func @cumsum_complex(%x: tensor<4x8xcomplex<f64>>) -> tensor<4x8xcomplex<f64>> {
  // CHECK: sar.cumsum %{{.*}} {dim = 0 : i64} : tensor<4x8xcomplex<f64>>
  %0 = sar.cumsum %x {dim = 0 : i64} : tensor<4x8xcomplex<f64>>
  return %0 : tensor<4x8xcomplex<f64>>
}

// CHECK-LABEL: func.func @rank_filter_median
func.func @rank_filter_median(%x: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK: sar.rank_filter %{{.*}} {dim = 1 : i64, rank = 3 : i64, window = 7 : i64} : tensor<8x16xf32>
  %0 = sar.rank_filter %x {window = 7 : i64, rank = 3 : i64, dim = 1 : i64} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func.func @layout
func.func @layout(%x: tensor<4x8xf64>) -> tensor<7x8xf64> {
  // CHECK: sar.slice %{{.*}} {offsets = array<i64: 1, 0>, sizes = array<i64: 2, 8>, strides = array<i64: 1, 1>}
  %0 = sar.slice %x {offsets = array<i64: 1, 0>, sizes = array<i64: 2, 8>, strides = array<i64: 1, 1>} : tensor<4x8xf64> -> tensor<2x8xf64>
  // CHECK: sar.concat %{{.*}}, %{{.*}} {dim = 0 : i64}
  %1 = sar.concat %0, %x {dim = 0 : i64} : (tensor<2x8xf64>, tensor<4x8xf64>) -> (tensor<6x8xf64>)
  // CHECK: sar.pad %{{.*}} {high = array<i64: 1, 0>, low = array<i64: 0, 0>, value = 0.000000e+00 : f64}
  %2 = sar.pad %1 {low = array<i64: 0, 0>, high = array<i64: 1, 0>, value = 0.0} : tensor<6x8xf64> -> tensor<7x8xf64>
  return %2 : tensor<7x8xf64>
}

// CHECK-LABEL: func.func @int_casts
func.func @int_casts(%x: tensor<4x8xf64>) -> tensor<4x8xf64> {
  // CHECK: sar.cast %{{.*}} : tensor<4x8xf64> -> tensor<4x8xi64>
  %0 = sar.cast %x : tensor<4x8xf64> -> tensor<4x8xi64>
  // CHECK: sar.cast %{{.*}} : tensor<4x8xi64> -> tensor<4x8xf64>
  %1 = sar.cast %0 : tensor<4x8xi64> -> tensor<4x8xf64>
  return %1 : tensor<4x8xf64>
}

// CHECK-LABEL: func.func @interp_kernels
func.func @interp_kernels(%z: tensor<8x4xcomplex<f64>>, %p: tensor<8x4xf64>)
    -> (tensor<8x4xcomplex<f64>>, tensor<8x4xcomplex<f64>>) {
  // CHECK: sar.interp1d %{{.*}}, %{{.*}} {kernel = "cubic"}
  %0 = sar.interp1d %z, %p {kernel = "cubic"}
      : (tensor<8x4xcomplex<f64>>, tensor<8x4xf64>)
      -> (tensor<8x4xcomplex<f64>>)
  // CHECK: sar.interp1d %{{.*}}, %{{.*}} {beta = 4.000000e+00 : f64, taps = 16 : i64, window = "kaiser"}
  %1 = sar.interp1d %z, %p {taps = 16 : i64, window = "kaiser",
                            beta = 4.0 : f64}
      : (tensor<8x4xcomplex<f64>>, tensor<8x4xf64>)
      -> (tensor<8x4xcomplex<f64>>)
  return %0, %1 : tensor<8x4xcomplex<f64>>, tensor<8x4xcomplex<f64>>
}

// CHECK-LABEL: func.func @gather
func.func @gather(%d: tensor<8x8xcomplex<f64>>, %r: tensor<4x4xf64>,
                  %c: tensor<4x4xf64>) -> tensor<4x4xcomplex<f64>> {
  // CHECK: sar.gather2d %{{.*}}, %{{.*}}, %{{.*}} {boundary = "edge", kernel = "nearest"}
  %0 = sar.gather2d %d, %r, %c {kernel = "nearest", boundary = "edge"}
      : (tensor<8x8xcomplex<f64>>, tensor<4x4xf64>, tensor<4x4xf64>)
      -> tensor<4x4xcomplex<f64>>
  // CHECK: sar.gather2d
  %1 = sar.gather2d %0, %r, %c
      : (tensor<4x4xcomplex<f64>>, tensor<4x4xf64>, tensor<4x4xf64>)
      -> tensor<4x4xcomplex<f64>>
  return %1 : tensor<4x4xcomplex<f64>>
}

// CHECK-LABEL: func.func @iterate
func.func @iterate(%z: tensor<8x8xcomplex<f64>>, %f: tensor<8x8xcomplex<f64>>)
    -> tensor<8x8xcomplex<f64>> {
  // CHECK: sar.iterate(%{{.*}}) {trips = 4 : i64}
  // CHECK: sar.yield
  %0 = sar.iterate(%z) {trips = 4 : i64}
      : (tensor<8x8xcomplex<f64>>) -> tensor<8x8xcomplex<f64>> {
  ^bb0(%acc: tensor<8x8xcomplex<f64>>):
    %1 = sar.mul %acc, %f : tensor<8x8xcomplex<f64>>
    sar.yield %1 : tensor<8x8xcomplex<f64>>
  }
  return %0 : tensor<8x8xcomplex<f64>>
}
