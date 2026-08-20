// RUN: sar-opt %s --hls-reduce-initiation-interval | FileCheck %s

// Floating-point accumulation is left in source order. Commutativity does not
// make reassociation bit-exact because it changes rounding and NaN behavior.

// CHECK-LABEL: func.func @float_acc_chain
func.func @float_acc_chain(%a: memref<8xf32>, %b: memref<8xf32>,
                           %c: memref<8xf32>, %acc: memref<1xf32>) {
  affine.for %i = 0 to 8 {
    // CHECK: %[[ACC:.*]] = affine.load %arg3[0]
    // CHECK-NEXT: %[[X:.*]] = affine.load %arg0
    // CHECK-NEXT: %[[Y:.*]] = affine.load %arg1
    // CHECK-NEXT: %[[Z:.*]] = affine.load %arg2
    // CHECK-NEXT: %[[S1:.*]] = arith.addf %[[ACC]], %[[X]]
    // CHECK-NEXT: %[[S2:.*]] = arith.addf %[[S1]], %[[Y]]
    // CHECK-NEXT: %[[S3:.*]] = arith.addf %[[S2]], %[[Z]]
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

// Exact integer arithmetic may move the recurrence load to the end of the
// chain, reducing its loop-carried path without changing values.

// CHECK-LABEL: func.func @int_acc_chain
func.func @int_acc_chain(%a: memref<8xi32>, %b: memref<8xi32>,
                         %c: memref<8xi32>, %acc: memref<1xi32>) {
  affine.for %i = 0 to 8 {
    // CHECK: %[[X:.*]] = affine.load %arg0
    // CHECK-NEXT: %[[Y:.*]] = affine.load %arg1
    // CHECK-NEXT: %[[Z:.*]] = affine.load %arg2
    // CHECK-NEXT: %[[S1:.*]] = arith.addi %[[X]], %[[Y]]
    // CHECK-NEXT: %[[S2:.*]] = arith.addi %[[S1]], %[[Z]]
    // CHECK-NEXT: %[[ACC:.*]] = affine.load %arg3[0]
    // CHECK-NEXT: %[[S3:.*]] = arith.addi %[[ACC]], %[[S2]]
    // CHECK-NEXT: affine.store %[[S3]], %arg3[0]
    %s = affine.load %acc[0] : memref<1xi32>
    %x = affine.load %a[%i] : memref<8xi32>
    %y = affine.load %b[%i] : memref<8xi32>
    %z = affine.load %c[%i] : memref<8xi32>
    %s1 = arith.addi %s, %x : i32
    %s2 = arith.addi %s1, %y : i32
    %s3 = arith.addi %s2, %z : i32
    affine.store %s3, %acc[0] : memref<1xi32>
  }
  return
}
