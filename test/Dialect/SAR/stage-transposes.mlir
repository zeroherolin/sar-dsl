// RUN: sar-opt %s --sar-stage-transposes="block-bytes=128" | FileCheck %s

// A transposing nest reads one buffer along its rows and writes the other
// along its columns, so one of the two strides by a whole row whichever loop
// is innermost. Staging a square block makes both sides contiguous: the
// block is filled row-wise and drained column-wise, and it is the only thing
// left striding.

// CHECK-LABEL: func.func @corner_turn
// CHECK: memref.alloc() : memref<4x4xf64>
// The read half sweeps the source's last index, the write half the
// destination's; neither jumps a row.
// CHECK: affine.load %arg0[%{{.*}} * 4 + %{{.*}}, %{{.*}} * 4 + %[[R:.*]]]
// CHECK: affine.store %{{.*}}, %alloc[%[[R]], %{{.*}}]
// CHECK: affine.load %alloc[%{{.*}}, %{{.*}}]
// CHECK: affine.store %{{.*}}, %arg1[%{{.*}} * 4 + %{{.*}}, %{{.*}} * 4 + %{{.*}}]
func.func @corner_turn(%in: memref<8x8xf64>, %out: memref<8x8xf64>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%j, %i] : memref<8x8xf64>
      affine.store %v, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// -----

// A corner turn is usually fused with the window or phase multiply that
// follows it, so the staging has to replay whatever the body computes --
// re-expressing every other access over the tile and point loops.

// CHECK-LABEL: func.func @fused_with_window
// CHECK: memref.alloc() : memref<4x4xf64>
// CHECK: affine.load %alloc[
// CHECK: affine.load %arg1[
// CHECK: arith.mulf
// CHECK: affine.store
func.func @fused_with_window(%in: memref<8x8xf64>, %w: memref<8xf64>,
                             %out: memref<8x8xf64>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%j, %i] : memref<8x8xf64>
      %t = affine.load %w[%j] : memref<8xf64>
      %m = arith.mulf %v, %t : f64
      affine.store %m, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// -----

// Several planes may be read against the grain at once. Staging only some
// of them would leave the rest striding, so each gets a block and they
// share the budget -- two f64 blocks in 128 bytes gives a 2 x 2 edge.

// CHECK-LABEL: func.func @two_transposed_reads
// CHECK-COUNT-2: memref.alloc() : memref<2x2xf64>
// CHECK-NOT: memref.alloc()
func.func @two_transposed_reads(%a: memref<8x8xf64>, %b: memref<8x8xf64>,
                                %out: memref<8x8xf64>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %x = affine.load %a[%j, %i] : memref<8x8xf64>
      %y = affine.load %b[%j, %i] : memref<8x8xf64>
      %s = arith.addf %x, %y : f64
      affine.store %s, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// -----

// A nest that already streams both sides the same way has nothing to gain:
// staging would add a copy without making anything contiguous.

// CHECK-LABEL: func.func @already_contiguous
// CHECK-NOT: memref.alloc
func.func @already_contiguous(%in: memref<8x8xf64>, %out: memref<8x8xf64>) {
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i, %j] : memref<8x8xf64>
      affine.store %v, %out[%i, %j] : memref<8x8xf64>
    }
  }
  return
}

// -----

// Rank and nest depth are not fixed: only the two levels the staging swaps
// are split, and every other level keeps its extent and its place. A batch
// of transposes stages one block and reuses it per batch element.

// CHECK-LABEL: func.func @batched
// CHECK: memref.alloc() : memref<4x4xf64>
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.for %{{.*}} = 0 to 2
// CHECK: affine.for %{{.*}} = 0 to 2
// CHECK-COUNT-2: affine.for %{{.*}} = 0 to 4
func.func @batched(%in: memref<4x8x8xf64>, %out: memref<4x8x8xf64>) {
  affine.for %b = 0 to 4 {
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %v = affine.load %in[%b, %j, %i] : memref<4x8x8xf64>
        affine.store %v, %out[%b, %i, %j] : memref<4x8x8xf64>
      }
    }
  }
  return
}
