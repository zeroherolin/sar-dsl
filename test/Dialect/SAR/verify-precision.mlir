// RUN: sar-opt %s --split-input-file --sar-verify-precision="precision=f32" --verify-diagnostics

func.func @narrow(%x: tensor<4xf32>) -> tensor<4xf32> {
  %0 = sar.mul_scalar %x, 2.0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @internal_widening(%x: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{result type 'tensor<4xf64>' violates precision=f32}}
  %wide = sar.cast %x : tensor<4xf32> -> tensor<4xf64>
  %scaled = sar.mul_scalar %wide, 1.25 : tensor<4xf64>
  %narrow = sar.cast %scaled : tensor<4xf64> -> tensor<4xf32>
  return %narrow : tensor<4xf32>
}
