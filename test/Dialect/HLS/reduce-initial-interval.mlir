// RUN: sar-opt %s --hls-reduce-initial-interval | FileCheck %s

// The II of an accumulation loop is bound by the recurrence: load the
// accumulator, walk the whole add chain, store it back. Since the adds are
// commutative, the independent operands can be summed first and the
// accumulator folded in by a single add at the end -- the load moves down the
// chain, the store depends on one add instead of three, and the summation
// order of each individual chain is preserved (results stay bit-exact).

// CHECK-LABEL: func.func @acc_chain
func.func @acc_chain(%a: memref<8xf32>, %b: memref<8xf32>, %c: memref<8xf32>,
                     %acc: memref<1xf32>) {
  affine.for %i = 0 to 8 {
    // CHECK: %[[X:.*]] = affine.load %arg0
    // CHECK: %[[Y:.*]] = affine.load %arg1
    // CHECK: %[[Z:.*]] = affine.load %arg2
    // CHECK: %[[S1:.*]] = arith.addf %[[X]], %[[Y]]
    // CHECK-NEXT: %[[S2:.*]] = arith.addf %[[S1]], %[[Z]]
    // CHECK-NEXT: %[[ACC:.*]] = affine.load %arg3[0]
    // CHECK-NEXT: %[[S3:.*]] = arith.addf %[[ACC]], %[[S2]]
    // CHECK-NEXT: affine.store %[[S3]], %arg3[0]
    %s = affine.load %acc[0] : memref<1xf32>
    %x = affine.load %a[%i] : memref<8xf32>
    %y = affine.load %b[%i] : memref<8xf32>
    %z = affine.load %c[%i] : memref<8xf32>
    %s1 = arith.addf %s, %x : f32
    %s2 = arith.addf %s1, %y : f32
    %s3 = arith.addf %s2, %z : f32
    affine.store %s3, %acc[0] : memref<1xf32>
  }
  return
}
