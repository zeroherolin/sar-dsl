// RUN: sar-opt %s --sar-decomplexify --verify-diagnostics

// The rewrite walks one straight-line body; branch successors would need
// their block arguments remapped, so anything but a single block is
// rejected rather than miscompiled. (External declarations are skipped
// before this check.)

// expected-error @below {{carries complex tensors but does not have a single-block body}}
func.func @multi_block(%a: tensor<4xcomplex<f32>>, %c: i1)
    -> tensor<4xcomplex<f32>> {
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  %0 = sar.conj %a : tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
^bb2:
  return %a : tensor<4xcomplex<f32>>
}
