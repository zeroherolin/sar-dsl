// RUN: sar-opt %s --convert-sar-interp-to-affine | FileCheck %s

// CHECK-LABEL: func.func @interp
func.func @interp(%re: tensor<8x16xf32>, %im: tensor<8x16xf32>,
                  %p: tensor<8x16xf64>)
    -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK: bufferization.to_buffer
  // CHECK: memref.alloc
  // CHECK: affine.for
  // CHECK: affine.for
  // Position inputs are scalarized at the output indices.
  // CHECK: tensor.extract
  // 8 statically unrolled taps, each a masked (select) gather.
  // CHECK-COUNT-8: arith.select
  // CHECK: memref.load
  // CHECK-NOT: sar.interp1d_split
// CHECK-NOT: scf.if
  %r, %i = sar.interp1d_split %re, %im, %p
      : (tensor<8x16xf32>, tensor<8x16xf32>, tensor<8x16xf64>)
      -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %r, %i : tensor<8x16xf32>, tensor<8x16xf32>
}

// The taps of one sample sit at compile-time integer offsets from a single
// fractional position, so the sinc numerator and the raised-cosine window
// angle are evaluated once and combined per tap rather than per tap
// evaluated: three transcendentals for the whole gather, not two per tap.

// RUN: sar-opt %s --convert-sar-interp-to-affine \
// RUN:   | grep -c -E 'math\.(sin|cos)' | FileCheck %s --check-prefix=COUNT
// COUNT: 3
