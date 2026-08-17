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

// The frontend's `sar.arg_names` splits with the arguments it names: a
// complex parameter keeps its name on both planes.
// CHECK-LABEL: func.func @named_args
// CHECK-SAME: attributes {sar.arg_names = ["raw_re", "raw_im", "win"]}
func.func @named_args(%a: tensor<4xcomplex<f64>>, %w: tensor<4xf64>)
    -> tensor<4xcomplex<f64>>
    attributes {sar.arg_names = ["raw", "win"]} {
  %0 = sar.mul %a, %a : tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @complex_from_planes
func.func @complex_from_planes(%p: tensor<8xf32>) -> tensor<8xcomplex<f32>> {
  // CHECK-DAG: sar.cos %arg0
  // CHECK-DAG: sar.sin %arg0
  // CHECK: return
  %c = sar.cos %p : tensor<8xf32>
  %s = sar.sin %p : tensor<8xf32>
  %0 = sar.complex %c, %s : tensor<8xf32> -> tensor<8xcomplex<f32>>
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

// CHECK-LABEL: func.func @conj_negates_imag
// CHECK-SAME: (%[[RE:.*]]: tensor<4xf32>, %[[IM:.*]]: tensor<4xf32>)
func.func @conj_negates_imag(%z: tensor<4xcomplex<f32>>)
    -> tensor<4xcomplex<f32>> {
  // CHECK: %[[NEG:.*]] = sar.mul_scalar %[[IM]], -1
  // CHECK: return %[[RE]], %[[NEG]]
  %0 = sar.conj %z : tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}


// CHECK-LABEL: func.func @plane_roundtrip
// CHECK-SAME: (%[[RE:.*]]: tensor<4xf64>, %[[IM:.*]]: tensor<4xf64>)
func.func @plane_roundtrip(%z: tensor<4xcomplex<f64>>)
    -> tensor<4xcomplex<f64>> {
  // CHECK: return %[[IM]], %[[RE]]
  %re = sar.imag %z : tensor<4xcomplex<f64>> -> tensor<4xf64>
  %im = sar.real %z : tensor<4xcomplex<f64>> -> tensor<4xf64>
  %0 = sar.complex %re, %im : tensor<4xf64> -> tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @complex_sum_splits
// CHECK-SAME: (%[[RE:.*]]: tensor<4x8xf64>, %[[IM:.*]]: tensor<4x8xf64>)
func.func @complex_sum_splits(%z: tensor<4x8xcomplex<f64>>)
    -> tensor<4xcomplex<f64>> {
  // CHECK-DAG: sar.reduce %[[RE]] {dim = 1 : i64, kind = "sum"}
  // CHECK-DAG: sar.reduce %[[IM]] {dim = 1 : i64, kind = "sum"}
  %0 = sar.reduce %z {kind = "sum", dim = 1 : i64}
      : tensor<4x8xcomplex<f64>> -> tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// CHECK-LABEL: func.func @pad_planes
// CHECK-SAME: (%[[RE:.*]]: tensor<4xf32>, %[[IM:.*]]: tensor<4xf32>)
func.func @pad_planes(%z: tensor<4xcomplex<f32>>) -> tensor<6xcomplex<f32>> {
  // CHECK-DAG: sar.pad %[[RE]] {high = array<i64: 1>, low = array<i64: 1>, value = 2.500000e-01
  // CHECK-DAG: sar.pad %[[IM]] {high = array<i64: 1>, low = array<i64: 1>, value = 0.000000e+00
  %0 = sar.pad %z {low = array<i64: 1>, high = array<i64: 1>, value = 0.25} : tensor<4xcomplex<f32>> -> tensor<6xcomplex<f32>>
  return %0 : tensor<6xcomplex<f32>>
}

// CHECK-LABEL: func.func @where_selects_planes
// CHECK-SAME: (%[[M:.*]]: tensor<4xf32>, %[[ARE:.*]]: tensor<4xf32>, %[[AIM:.*]]: tensor<4xf32>, %[[BRE:.*]]: tensor<4xf32>, %[[BIM:.*]]: tensor<4xf32>)
func.func @where_selects_planes(%m: tensor<4xf32>, %a: tensor<4xcomplex<f32>>,
                                %b: tensor<4xcomplex<f32>>)
    -> tensor<4xcomplex<f32>> {
  // CHECK-DAG: sar.where %[[M]], %[[ARE]], %[[BRE]]
  // CHECK-DAG: sar.where %[[M]], %[[AIM]], %[[BIM]]
  %0 = sar.where %m, %a, %b : (tensor<4xf32>, tensor<4xcomplex<f32>>, tensor<4xcomplex<f32>>) -> (tensor<4xcomplex<f32>>)
  return %0 : tensor<4xcomplex<f32>>
}

// CHECK-LABEL: func.func @cumsum_planes
// CHECK-SAME: (%[[RE:.*]]: tensor<4x8xf32>, %[[IM:.*]]: tensor<4x8xf32>)
func.func @cumsum_planes(%z: tensor<4x8xcomplex<f32>>) -> tensor<4x8xcomplex<f32>> {
  // A prefix sum is linear, so each plane scans on its own.
  // CHECK-DAG: sar.cumsum %[[RE]] {dim = 1 : i64}
  // CHECK-DAG: sar.cumsum %[[IM]] {dim = 1 : i64}
  %0 = sar.cumsum %z {dim = 1 : i64} : tensor<4x8xcomplex<f32>>
  return %0 : tensor<4x8xcomplex<f32>>
}

// A complex gather splits into one gather per plane, sharing the (real)
// position tensors.
// CHECK-LABEL: func.func @gather_planes
func.func @gather_planes(%d: tensor<8x8xcomplex<f64>>, %r: tensor<4x4xf64>,
                         %c: tensor<4x4xf64>) -> tensor<4x4xcomplex<f64>> {
  // CHECK: sar.gather2d_split %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  // CHECK-NOT: complex
  %0 = sar.gather2d %d, %r, %c
      : (tensor<8x8xcomplex<f64>>, tensor<4x4xf64>, tensor<4x4xf64>)
      -> tensor<4x4xcomplex<f64>>
  return %0 : tensor<4x4xcomplex<f64>>
}

// A complex loop carry expands into an adjacent (re, im) pair, and the
// body is rewritten inside the loop.
// CHECK-LABEL: func.func @iterate_carries
func.func @iterate_carries(%z: tensor<4x4xcomplex<f64>>)
    -> tensor<4x4xcomplex<f64>> {
  // CHECK: sar.iterate(%{{.*}}, %{{.*}}) {trips = 3 : i64} : (tensor<4x4xf64>, tensor<4x4xf64>)
  // CHECK: ^bb0(%{{.*}}: tensor<4x4xf64>, %{{.*}}: tensor<4x4xf64>):
  // CHECK: sar.yield %{{.*}}, %{{.*}} : tensor<4x4xf64>, tensor<4x4xf64>
  // CHECK-NOT: complex
  %0 = sar.iterate(%z) {trips = 3 : i64}
      : (tensor<4x4xcomplex<f64>>) -> tensor<4x4xcomplex<f64>> {
  ^bb0(%acc: tensor<4x4xcomplex<f64>>):
    %1 = sar.mul %acc, %acc : tensor<4x4xcomplex<f64>>
    sar.yield %1 : tensor<4x4xcomplex<f64>>
  }
  return %0 : tensor<4x4xcomplex<f64>>
}
