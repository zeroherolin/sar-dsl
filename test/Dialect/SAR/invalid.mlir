// RUN: sar-opt %s -split-input-file -verify-diagnostics
// Verifier coverage for ill-formed SAR operations.

func.func @fft_non_pow2(%x: tensor<4x6xcomplex<f32>>) -> tensor<4x6xcomplex<f32>> {
  // expected-error @+1 {{size along dim must be a power of two}}
  %0 = sar.fft %x {dim = 1 : i64} : tensor<4x6xcomplex<f32>>
  return %0 : tensor<4x6xcomplex<f32>>
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
  // expected-error @+1 {{cannot cast a complex tensor to a float tensor}}
  %0 = sar.cast %x : tensor<4xcomplex<f32>> -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @stolt_bad_axis(%d: tensor<8x16xcomplex<f32>>, %fa: tensor<4xf64>,
                          %fr: tensor<16xf64>) -> tensor<8x16xcomplex<f32>> {
  // expected-error @+1 {{fa length must equal the azimuth (first) dimension}}
  %0 = sar.stolt_interp %d, %fa, %fr {c = 3.0e8, fc = 1.0e9, vr = 7000.0,
        t_shift = 0.0}
      : (tensor<8x16xcomplex<f32>>, tensor<4xf64>, tensor<16xf64>)
      -> (tensor<8x16xcomplex<f32>>)
  return %0 : tensor<8x16xcomplex<f32>>
}
