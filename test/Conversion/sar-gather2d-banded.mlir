// RUN: sar-opt %s --convert-sar-interp-to-affine="banded-profit-threshold=1" | FileCheck %s
// RUN: sar-opt %s --convert-sar-interp-to-affine="enable-banded-gather=0" | FileCheck %s --check-prefix=FULL

// When the row coordinate is provably the output row plus a bounded
// displacement, the 2-D gather runs against a sliding band of whole
// source rows: displacement [1.5, 1.5] plus the bilinear tap support
// widens to a 4-row band (next power of two), and each source row is
// staged exactly once per output row.

// CHECK-LABEL: func.func @banded
// CHECK: memref.alloc() : memref<4x16xf64>
// CHECK: memref.alloc() : memref<4x16xf64>
// CHECK-NOT: sar.gather2d_split
// CHECK: arith.andi
// CHECK: affine.for

// Disabled, the op survives for the full-plane linalg path.
// FULL: sar.gather2d_split
func.func @banded(%re: tensor<16x16xf64>, %im: tensor<16x16xf64>)
    -> (tensor<16x16xf64>, tensor<16x16xf64>) {
  %iota = sar.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                              8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                              15.0]> : tensor<16xf64>
  %rows0 = sar.broadcast %iota {dim = 0 : i64} : tensor<16xf64> -> tensor<16x16xf64>
  %rows = sar.add_scalar %rows0, 1.5 : tensor<16x16xf64>
  %cols = sar.broadcast %iota {dim = 1 : i64} : tensor<16xf64> -> tensor<16x16xf64>
  %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
      : (tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>)
      -> (tensor<16x16xf64>, tensor<16x16xf64>)
  return %or, %oi : tensor<16x16xf64>, tensor<16x16xf64>
}

// -----

// A row coordinate the analysis cannot bound (a kernel argument) keeps
// the op for the full-plane path -- the fallback is automatic.

// CHECK-LABEL: func.func @unbounded
// CHECK: sar.gather2d_split
func.func @unbounded(%re: tensor<16x16xf64>, %im: tensor<16x16xf64>,
                     %rows: tensor<16x16xf64>, %cols: tensor<16x16xf64>)
    -> (tensor<16x16xf64>, tensor<16x16xf64>) {
  %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
      : (tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>)
      -> (tensor<16x16xf64>, tensor<16x16xf64>)
  return %or, %oi : tensor<16x16xf64>, tensor<16x16xf64>
}

// -----

// f32 arithmetic can round an apparent identity ramp away. The analysis
// declines rather than deriving an unsound one-row band.

// CHECK-LABEL: func.func @rounded_f32
// CHECK: sar.gather2d_split
func.func @rounded_f32(%re: tensor<16x16xf64>, %im: tensor<16x16xf64>)
    -> (tensor<16x16xf64>, tensor<16x16xf64>) {
  %iota32 = sar.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0]> : tensor<16xf32>
  %rows0 = sar.broadcast %iota32 {dim = 0 : i64}
      : tensor<16xf32> -> tensor<16x16xf32>
  %large = sar.add_scalar %rows0, 1.0e8 : tensor<16x16xf32>
  %rounded = sar.add_scalar %large, -1.0e8 : tensor<16x16xf32>
  %rows = sar.cast %rounded : tensor<16x16xf32> -> tensor<16x16xf64>
  %iota64 = sar.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                                15.0]> : tensor<16xf64>
  %cols = sar.broadcast %iota64 {dim = 1 : i64}
      : tensor<16xf64> -> tensor<16x16xf64>
  %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
      : (tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>,
         tensor<16x16xf64>) -> (tensor<16x16xf64>, tensor<16x16xf64>)
  return %or, %oi : tensor<16x16xf64>, tensor<16x16xf64>
}

// -----

// A nonlinear row field built entirely from constants is folded exactly.
// The symbolic affine form cannot follow sqrt(i*i), but the folded result is
// the identity row ramp and therefore needs only the bilinear two-row band.

// CHECK-LABEL: func.func @folded_rows
// CHECK: memref.alloc() : memref<2x16xf64>
// CHECK: memref.alloc() : memref<2x16xf64>
// CHECK-NOT: sar.gather2d_split
func.func @folded_rows(%re: tensor<16x16xf64>, %im: tensor<16x16xf64>)
    -> (tensor<16x16xf64>, tensor<16x16xf64>) {
  %iota = sar.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                              8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                              15.0]> : tensor<16xf64>
  %rows0 = sar.broadcast %iota {dim = 0 : i64}
      : tensor<16xf64> -> tensor<16x16xf64>
  %rows2 = sar.mul %rows0, %rows0 : tensor<16x16xf64>
  %rows = sar.sqrt %rows2 : tensor<16x16xf64>
  %cols = sar.broadcast %iota {dim = 1 : i64}
      : tensor<16xf64> -> tensor<16x16xf64>
  %or, %oi = sar.gather2d_split %re, %im, %rows, %cols
      : (tensor<16x16xf64>, tensor<16x16xf64>, tensor<16x16xf64>,
         tensor<16x16xf64>) -> (tensor<16x16xf64>, tensor<16x16xf64>)
  return %or, %oi : tensor<16x16xf64>, tensor<16x16xf64>
}
