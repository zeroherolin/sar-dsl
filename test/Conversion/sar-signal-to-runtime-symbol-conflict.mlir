// RUN: sar-opt %s --convert-sar-signal-to-runtime --verify-diagnostics

// Runtime symbols cannot be user definitions or incompatible declarations.
// expected-error @+1 {{symbol reserved for the SAR runtime ABI must be a matching private declaration with llvm.emit_c_interface}}
func.func @sar_rt_fft_1d_c128(%x: f64) -> f64 {
  return %x : f64
}

func.func @kernel(%x: tensor<8xcomplex<f64>>) -> tensor<8xcomplex<f64>> {
  // expected-error @+1 {{failed to legalize operation 'sar.fft' that was explicitly marked illegal}}
  %0 = sar.fft %x {dim = 0 : i64} : tensor<8xcomplex<f64>>
  return %0 : tensor<8xcomplex<f64>>
}
