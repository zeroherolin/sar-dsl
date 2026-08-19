// RUN: sar-opt %s --hls-fuse-sibling-loops --cse | FileCheck %s

// CHECK-LABEL: func.func @share_phase
// CHECK: affine.for
// CHECK: math.sin
// CHECK: affine.store
// CHECK: affine.store
// CHECK: affine.for
// CHECK: math.sin
// CHECK: affine.store
func.func @share_phase(%input: memref<16xf64>, %re: memref<16xf64>,
                       %im: memref<16xf64>, %extra: memref<16xf64>) {
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    affine.store %s, %re[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    %neg = arith.negf %s : f64
    affine.store %neg, %im[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    affine.store %s, %extra[%i] : memref<16xf64>
  }
  return
}

// CHECK-LABEL: func.func @dependence_blocks_fusion
// CHECK-COUNT-2: affine.for
func.func @dependence_blocks_fusion(%buffer: memref<16xf64>,
                                    %out: memref<16xf64>) {
  affine.for %i = 0 to 16 {
    %zero = arith.constant 0.0 : f64
    affine.store %zero, %buffer[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %value = affine.load %buffer[%i] : memref<16xf64>
    affine.store %value, %out[%i] : memref<16xf64>
  }
  return
}
