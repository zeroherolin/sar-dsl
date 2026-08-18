// RUN: sar-opt %s --convert-sar-signal-to-runtime --split-input-file --verify-diagnostics

// The runtime lowering only implements the dim = 1 form; dim = 0 is
// rewritten into transposes by canonicalization, which every pipeline runs
// first. Standalone, the precondition must be reported explicitly instead of
// leaving the op silently unconverted.
func.func @dim0_requires_canonicalization(%z: tensor<4x8xcomplex<f64>>,
                                          %p: tensor<4x8xf64>)
    -> tensor<4x8xcomplex<f64>> {
  // expected-error @+2 {{must be canonicalized to dim = 1}}
  // expected-error @+1 {{failed to legalize operation 'sar.interp1d'}}
  %0 = sar.interp1d %z, %p {dim = 0 : i64}
      : (tensor<4x8xcomplex<f64>>, tensor<4x8xf64>)
      -> tensor<4x8xcomplex<f64>>
  return %0 : tensor<4x8xcomplex<f64>>
}
