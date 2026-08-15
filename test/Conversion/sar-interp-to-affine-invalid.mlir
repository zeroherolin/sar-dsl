// RUN: sar-opt %s --convert-sar-interp-to-affine -split-input-file -verify-diagnostics

// dim = 0 is normalized into transposes around the dim = 1 form by
// canonicalization. Running this pass standalone must report that
// precondition rather than silently leaving the op unconverted.
func.func @dim0_requires_canonicalization(%re: tensor<4x8xf64>,
                                          %im: tensor<4x8xf64>,
                                          %p: tensor<4x8xf64>)
    -> (tensor<4x8xf64>, tensor<4x8xf64>) {
  // expected-error @+2 {{must be canonicalized to dim = 1}}
  // expected-error @+1 {{failed to legalize operation 'sar.interp1d_split'}}
  %0, %1 = sar.interp1d_split %re, %im, %p {dim = 0 : i64}
      : (tensor<4x8xf64>, tensor<4x8xf64>, tensor<4x8xf64>)
      -> (tensor<4x8xf64>, tensor<4x8xf64>)
  return %0, %1 : tensor<4x8xf64>, tensor<4x8xf64>
}
