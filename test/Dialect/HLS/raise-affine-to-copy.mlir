// RUN: sar-opt %s --hls-raise-affine-to-copy | FileCheck %s

// A whole-buffer copy expressed as a loop nest hides its burst nature from
// everything downstream; raised to memref.copy it becomes a unit the copy
// passes can place, split, or elide. Only a nest that provably touches every
// element exactly once may be raised.

// CHECK-LABEL: func.func @whole_copy
// CHECK-NOT: affine.for
// CHECK: memref.copy %arg0, %arg1
func.func @whole_copy(%src: memref<8x8xf32>, %dst: memref<8x8xf32>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %src[%i, %j] : memref<8x8xf32>
      affine.store %v, %dst[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// A diagonal sweep covers one slice of the buffer; replacing it with a
// whole-buffer copy would overwrite elements the loop never touched. The
// nest must stay a nest.

// CHECK-LABEL: func.func @diagonal
// CHECK: affine.for
// CHECK: affine.load %{{.*}}[%[[I:arg[0-9]+]], %[[I]]]
// CHECK-NOT: memref.copy
func.func @diagonal(%src: memref<8x8xf32>, %dst: memref<8x8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %src[%i, %i] : memref<8x8xf32>
    affine.store %v, %dst[%i, %i] : memref<8x8xf32>
  }
  return
}
