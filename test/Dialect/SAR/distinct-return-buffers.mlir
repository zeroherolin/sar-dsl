// RUN: sar-opt %s --sar-distinct-return-buffers | FileCheck %s

// CHECK-LABEL: func.func @duplicate_alloc
func.func @duplicate_alloc(%src: memref<8xf32>)
    -> (memref<8xf32>, memref<8xf32>, memref<8xf32>) {
  // CHECK: %[[ORIGINAL:.*]] = memref.alloc
  %result = memref.alloc() : memref<8xf32>
  // CHECK: memref.copy %arg0, %[[ORIGINAL]]
  memref.copy %src, %result : memref<8xf32> to memref<8xf32>
  // CHECK: %[[COPY0:.*]] = memref.alloc
  // CHECK: memref.copy %[[ORIGINAL]], %[[COPY0]]
  // CHECK: %[[COPY1:.*]] = memref.alloc
  // CHECK: memref.copy %[[ORIGINAL]], %[[COPY1]]
  // CHECK: return %[[ORIGINAL]], %[[COPY0]], %[[COPY1]]
  return %result, %result, %result
      : memref<8xf32>, memref<8xf32>, memref<8xf32>
}

// CHECK-LABEL: func.func @duplicate_argument
func.func @duplicate_argument(%src: memref<?xf64>)
    -> (memref<?xf64>, memref<?xf64>) {
  // CHECK: %[[N:.*]] = memref.dim %arg0, %c0
  // CHECK: %[[COPY:.*]] = memref.alloc(%[[N]])
  // CHECK: memref.copy %arg0, %[[COPY]]
  // CHECK: return %arg0, %[[COPY]]
  return %src, %src : memref<?xf64>, memref<?xf64>
}
