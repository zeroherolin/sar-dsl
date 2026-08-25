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

// -----

// A sampling coordinate is an index, not a sample: the dialect fixes it at
// f64 so a fractional bin still resolves over a full-size raster. Charging
// that against an f32 data path would make the policy unsatisfiable for
// every resampling kernel, so positions and the arithmetic feeding only
// them are exempt.
func.func @position_is_not_the_data_path(%z: tensor<8x8xcomplex<f32>>,
                                         %p: tensor<8x8xf64>)
    -> tensor<8x8xcomplex<f32>> {
  %shifted = sar.add_scalar %p, 0.5 : tensor<8x8xf64>
  %r = sar.interp1d %z, %shifted
      : (tensor<8x8xcomplex<f32>>, tensor<8x8xf64>) -> tensor<8x8xcomplex<f32>>
  return %r : tensor<8x8xcomplex<f32>>
}

// -----

// The exemption follows use, not type: a plane that is both a position and
// a sample is data, and is checked as data.
func.func @shared_plane_is_data(%p: tensor<8x8xf64>)
    -> (tensor<8x8xcomplex<f64>>, tensor<8x8xf64>) {
  // expected-error @+1 {{result type 'tensor<8x8xcomplex<f64>>' violates precision=f32}}
  %z = sar.complex %p, %p : tensor<8x8xf64> -> tensor<8x8xcomplex<f64>>
  %r = sar.interp1d %z, %p
      : (tensor<8x8xcomplex<f64>>, tensor<8x8xf64>) -> tensor<8x8xcomplex<f64>>
  %m = sar.abs %r : tensor<8x8xcomplex<f64>> -> tensor<8x8xf64>
  return %r, %m : tensor<8x8xcomplex<f64>>, tensor<8x8xf64>
}

// -----

// The same rule one step back: it is the *arithmetic feeding* a position
// that the exemption covers, and only while it feeds nothing else. Here the
// scaled plane reaches an interpolation and a cast into the data path, so
// it is data and its f64 width is checked. Following operands instead of
// results would exempt it -- the position operand it also feeds would be
// enough to carry the whole chain out of the policy.
func.func @position_arithmetic_shared_with_data(%z: tensor<4x8xcomplex<f32>>,
                                                %p: tensor<4x8xf64>)
    -> (tensor<4x8xcomplex<f32>>, tensor<4x8xf32>) {
  // expected-error @+1 {{operand type 'tensor<4x8xf64>' violates precision=f32}}
  %s = sar.mul %p, %p : tensor<4x8xf64>
  %r = sar.interp1d %z, %s
      : (tensor<4x8xcomplex<f32>>, tensor<4x8xf64>) -> tensor<4x8xcomplex<f32>>
  %c = sar.cast %s : tensor<4x8xf64> -> tensor<4x8xf32>
  return %r, %c : tensor<4x8xcomplex<f32>>, tensor<4x8xf32>
}
