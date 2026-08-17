// RUN: sar-opt %s --convert-sar-fft-to-affine -split-input-file -verify-diagnostics

// Non-power-of-two sizes are legal (Bluestein handles them); what the op
// rejects is a shape the transform cannot be defined on at all. These are
// verifier errors, so they fire before the pass sees the op.

func.func @dim_out_of_range(%re: tensor<4x8xf64>, %im: tensor<4x8xf64>)
    -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  // expected-error @+1 {{'sar.fft_split' op dim is out of range for the input rank}}
  %r, %i = sar.fft_split %re, %im {dim = 2 : i64} : tensor<4x8xf64>
  return %r, %i : tensor<4x8xf64>, tensor<4x8xf64>
}

// -----

func.func @size_below_two(%re: tensor<4x1xf64>, %im: tensor<4x1xf64>)
    -> (tensor<4x1xf64>, tensor<4x1xf64>) {
  // expected-error @+1 {{'sar.fft_split' op size along dim must be at least 2}}
  %r, %i = sar.fft_split %re, %im {dim = 1 : i64} : tensor<4x1xf64>
  return %r, %i : tensor<4x1xf64>, tensor<4x1xf64>
}

// -----

func.func @rank_three(%re: tensor<2x4x8xf64>, %im: tensor<2x4x8xf64>)
    -> (tensor<2x4x8xf64>, tensor<2x4x8xf64>) {
  // expected-error @+1 {{'sar.fft_split' op expects a rank-1 or rank-2 input}}
  %r, %i = sar.fft_split %re, %im {dim = 2 : i64} : tensor<2x4x8xf64>
  return %r, %i : tensor<2x4x8xf64>, tensor<2x4x8xf64>
}
