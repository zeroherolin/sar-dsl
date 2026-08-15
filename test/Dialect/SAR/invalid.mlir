// RUN: sar-opt %s -split-input-file -verify-diagnostics
// Verifier coverage for ill-formed SAR operations.

func.func @fft_too_small(%x: tensor<4x1xcomplex<f32>>) -> tensor<4x1xcomplex<f32>> {
  // expected-error @+1 {{size along dim must be at least 2}}
  %0 = sar.fft %x {dim = 1 : i64} : tensor<4x1xcomplex<f32>>
  return %0 : tensor<4x1xcomplex<f32>>
}

// -----

func.func @fft_bad_dim(%x: tensor<4x8xcomplex<f32>>) -> tensor<4x8xcomplex<f32>> {
  // expected-error @+1 {{dim is out of range for the input rank}}
  %0 = sar.fft %x {dim = 2 : i64} : tensor<4x8xcomplex<f32>>
  return %0 : tensor<4x8xcomplex<f32>>
}

// -----

func.func @transpose_bad_shape(%x: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // expected-error @+1 {{result shape must be the transpose of the input shape}}
  %0 = sar.transpose %x : tensor<4x8xf32> -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// -----

func.func @broadcast_bad_length(%v: tensor<8xf32>) -> tensor<4x16xf32> {
  // expected-error @+1 {{result size along dim must equal the input length}}
  %0 = sar.broadcast %v {dim = 1 : i64} : tensor<8xf32> -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// -----

func.func @abs_bad_precision(%x: tensor<4xcomplex<f64>>) -> tensor<4xf32> {
  // expected-error @+1 {{result element type must be the float precision}}
  %0 = sar.abs %x : tensor<4xcomplex<f64>> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @cast_complex_to_float(%x: tensor<4xcomplex<f32>>) -> tensor<4xf32> {
  // expected-error @+1 {{cannot cast a complex tensor to a real tensor}}
  %0 = sar.cast %x : tensor<4xcomplex<f32>> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @real_precision_mismatch(%z: tensor<4xcomplex<f32>>) -> tensor<4xf64> {
  // expected-error @+1 {{result element type must be the float precision of the input element type}}
  %0 = sar.real %z : tensor<4xcomplex<f32>> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

func.func @complex_precision_mismatch(%re: tensor<4xf32>, %im: tensor<4xf32>)
    -> tensor<4xcomplex<f64>> {
  // expected-error @+1 {{result complex precision must match the plane float type}}
  %0 = sar.complex %re, %im : tensor<4xf32> -> tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}

// -----

func.func @reduce_bad_kind(%x: tensor<4x8xf64>) -> tensor<4xf64> {
  // expected-error @+1 {{kind must be one of sum, max, min}}
  %0 = sar.reduce %x {kind = "mean", dim = 1 : i64} : tensor<4x8xf64> -> tensor<4xf64>
  return %0 : tensor<4xf64>
}

// -----

func.func @reduce_complex_max(%z: tensor<4x8xcomplex<f32>>) -> tensor<4xcomplex<f32>> {
  // expected-error @+1 {{max/min reductions require float elements}}
  %0 = sar.reduce %z {kind = "max", dim = 1 : i64} : tensor<4x8xcomplex<f32>> -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----

func.func @reduce_bad_result(%x: tensor<4x8xf64>) -> tensor<8xf64> {
  // expected-error @+1 {{result must be rank-1 with the size of the kept axis}}
  %0 = sar.reduce %x {kind = "sum", dim = 1 : i64} : tensor<4x8xf64> -> tensor<8xf64>
  return %0 : tensor<8xf64>
}

// -----

func.func @slice_out_of_bounds(%x: tensor<4x8xf64>) -> tensor<2x8xf64> {
  // expected-error @+1 {{slice bounds exceed the input along dim 0}}
  %0 = sar.slice %x {offsets = array<i64: 3, 0>, sizes = array<i64: 2, 8>, strides = array<i64: 1, 1>} : tensor<4x8xf64> -> tensor<2x8xf64>
  return %0 : tensor<2x8xf64>
}

// -----

func.func @concat_shape_mismatch(%a: tensor<4x8xf64>, %b: tensor<4x6xf64>) -> tensor<8x8xf64> {
  // expected-error @+1 {{operand shapes must match outside dim}}
  %0 = sar.concat %a, %b {dim = 0 : i64} : (tensor<4x8xf64>, tensor<4x6xf64>) -> (tensor<8x8xf64>)
  return %0 : tensor<8x8xf64>
}

// -----

func.func @pad_bad_result(%x: tensor<4xf64>) -> tensor<5xf64> {
  // expected-error @+1 {{result shape must be the padded input shape}}
  %0 = sar.pad %x {low = array<i64: 2>, high = array<i64: 0>, value = 0.0} : tensor<4xf64> -> tensor<5xf64>
  return %0 : tensor<5xf64>
}



// -----

func.func @cmp_bad_predicate(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{predicate must be one of eq, ne, lt, le, gt, ge}}
  %0 = sar.cmp %a, %b {predicate = "approx"} : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @where_mask_precision(%m: tensor<4xf32>, %a: tensor<4xcomplex<f64>>, %b: tensor<4xcomplex<f64>>) -> tensor<4xcomplex<f64>> {
  // expected-error @+1 {{mask precision must match the branch element precision}}
  %0 = sar.where %m, %a, %b : (tensor<4xf32>, tensor<4xcomplex<f64>>, tensor<4xcomplex<f64>>) -> (tensor<4xcomplex<f64>>)
  return %0 : tensor<4xcomplex<f64>>
}

// -----

func.func @interp_bad_kernel(%z: tensor<4x8xcomplex<f64>>,
                             %p: tensor<4x8xf64>)
    -> tensor<4x8xcomplex<f64>> {
  // expected-error @+1 {{kernel must be one of: nearest, linear, cubic, sinc}}
  %0 = sar.interp1d %z, %p {kernel = "spline"}
      : (tensor<4x8xcomplex<f64>>, tensor<4x8xf64>)
      -> (tensor<4x8xcomplex<f64>>)
  return %0 : tensor<4x8xcomplex<f64>>
}

// -----

func.func @interp_odd_taps(%z: tensor<4x8xcomplex<f64>>, %p: tensor<4x8xf64>)
    -> tensor<4x8xcomplex<f64>> {
  // expected-error @+1 {{taps must be even and in [4, 32]}}
  %0 = sar.interp1d %z, %p {taps = 7 : i64}
      : (tensor<4x8xcomplex<f64>>, tensor<4x8xf64>)
      -> (tensor<4x8xcomplex<f64>>)
  return %0 : tensor<4x8xcomplex<f64>>
}

// -----

func.func @cast_int_to_complex(%x: tensor<4xi64>)
    -> tensor<4xcomplex<f64>> {
  // expected-error @+1 {{cannot cast an integer tensor directly to complex}}
  %0 = sar.cast %x : tensor<4xi64> -> tensor<4xcomplex<f64>>
  return %0 : tensor<4xcomplex<f64>>
}
