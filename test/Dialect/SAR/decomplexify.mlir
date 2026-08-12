// RUN: sar-opt %s --sar-decomplexify | FileCheck %s

// CHECK-LABEL: func.func @complex_mul
// CHECK-SAME: (%[[ARE:.*]]: tensor<4xf64>, %[[AIM:.*]]: tensor<4xf64>, %[[BRE:.*]]: tensor<4xf64>, %[[BIM:.*]]: tensor<4xf64>) -> (tensor<4xf64>, tensor<4xf64>)
func.func @complex_mul(%a: tensor<4xcomplex<f64>>, %b: tensor<4xcomplex<f64>>)
    -> tensor<4xcomplex<f64>> {
  // (a+bi)(c+di) = (ac - bd) + (ad + bc)i
  // CHECK-DAG: sar.mul
  // CHECK-DAG: sar.sub
  // CHECK-DAG: sar.add
  // CHECK-NOT: complex
  %0 = sar.mul %a, %b : tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @expj_becomes_cos_sin
func.func @expj_becomes_cos_sin(%p: tensor<8xf32>) -> tensor<8xcomplex<f32>> {
  // CHECK-DAG: sar.cos %arg0
  // CHECK-DAG: sar.sin %arg0
  %0 = sar.expj %p : tensor<8xf32> -> tensor<8xcomplex<f32>>
  return %0 : tensor<8xcomplex<f32>>
}

// CHECK-LABEL: func.func @abs_becomes_magnitude
func.func @abs_becomes_magnitude(%x: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  // CHECK: sar.mul
  // CHECK: sar.mul
  // CHECK: sar.add
  // CHECK: sar.sqrt
  %0 = sar.abs %x : tensor<4xcomplex<f32>> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @fft_becomes_split
func.func @fft_becomes_split(%x: tensor<4x8xcomplex<f32>>)
    -> tensor<4x8xcomplex<f32>> {
  // CHECK: sar.fft_split %arg0, %arg1 {dim = 1 : i64} : tensor<4x8xf32>
  %0 = sar.fft %x {dim = 1 : i64} : tensor<4x8xcomplex<f32>>
  // CHECK: sar.fft_split %{{.*}} {dim = 0 : i64, inverse}
  %1 = sar.ifft %0 {dim = 0 : i64} : tensor<4x8xcomplex<f32>>
  return %1 : tensor<4x8xcomplex<f32>>
}

// CHECK-LABEL: func.func @interp_becomes_split
func.func @interp_becomes_split(%d: tensor<8x16xcomplex<f32>>,
                                %p: tensor<8x16xf64>)
    -> tensor<8x16xcomplex<f32>> {
  // CHECK: sar.interp1d_split %arg0, %arg1, %arg2
  %0 = sar.interp1d %d, %p
      : (tensor<8x16xcomplex<f32>>, tensor<8x16xf64>)
      -> (tensor<8x16xcomplex<f32>>)
  return %0 : tensor<8x16xcomplex<f32>>
}

// CHECK-LABEL: func.func @stolt_becomes_split
func.func @stolt_becomes_split(%d: tensor<8x16xcomplex<f32>>,
                               %fa: tensor<8xf64>, %fr: tensor<16xf64>)
    -> tensor<8x16xcomplex<f32>> {
  // CHECK: sar.stolt_interp_split %arg0, %arg1, %arg2, %arg3
  // CHECK-SAME: t_shift
  %0 = sar.stolt_interp %d, %fa, %fr {c = 3.0e8, fc = 1.0e9, vr = 7000.0,
        t_shift = 1.0e-4}
      : (tensor<8x16xcomplex<f32>>, tensor<8xf64>, tensor<16xf64>)
      -> (tensor<8x16xcomplex<f32>>)
  return %0 : tensor<8x16xcomplex<f32>>
}

// CHECK-LABEL: func.func @complex_constant_splits
func.func @complex_constant_splits() -> tensor<2xcomplex<f64>> {
  // CHECK-DAG: sar.constant dense<[1.000000e+00, 3.000000e+00]>
  // CHECK-DAG: sar.constant dense<[2.000000e+00, -4.000000e+00]>
  %0 = sar.constant dense<[(1.0, 2.0), (3.0, -4.0)]> : tensor<2xcomplex<f64>>
  return %0 : tensor<2xcomplex<f64>>
}

// CHECK-LABEL: func.func @float_only_untouched
func.func @float_only_untouched(%x: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: sar.mul_scalar %arg0
  %0 = sar.mul_scalar %x, 2.0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}
