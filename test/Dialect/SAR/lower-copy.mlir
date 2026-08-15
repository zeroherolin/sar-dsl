// RUN: sar-opt %s --sar-lower-copy | FileCheck %s

// `memref.copy` lowers to a call into the MLIR runtime support library,
// which neither the standalone kernel nor an HLS target links against.
// Expanding it into a loop nest is what keeps those cases working.

// CHECK-LABEL: func.func @copy_1d
// CHECK-NOT: memref.copy
// CHECK: affine.for %[[I:.*]] = 0 to 64
// CHECK: %[[V:.*]] = affine.load %arg0[%[[I]]]
// CHECK: affine.store %[[V]], %arg1[%[[I]]]
func.func @copy_1d(%a: memref<64xf64>, %b: memref<64xf64>) {
  memref.copy %a, %b : memref<64xf64> to memref<64xf64>
  return
}

// -----

// Rank is not fixed: a copy sweeps one loop per dimension, innermost last
// so both sides stay contiguous.

// CHECK-LABEL: func.func @copy_2d
// CHECK-NOT: memref.copy
// CHECK: affine.for %[[I:.*]] = 0 to 8
// CHECK: affine.for %[[J:.*]] = 0 to 16
// CHECK: %[[V:.*]] = affine.load %arg0[%[[I]], %[[J]]]
// CHECK: affine.store %[[V]], %arg1[%[[I]], %[[J]]]
func.func @copy_2d(%a: memref<8x16xf32>, %b: memref<8x16xf32>) {
  memref.copy %a, %b : memref<8x16xf32> to memref<8x16xf32>
  return
}
