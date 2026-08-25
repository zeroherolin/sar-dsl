// RUN: ! sar-opt %s "--hls-create-axi-interface=stream-interface=true" 2>&1 | FileCheck %s

// A square transpose can look like an identity map when only operand extents
// are inspected. The induction variables are reversed, so the physical access
// order is column-major and cannot be represented by an addressless stream.

// CHECK: error: cannot use AXI4-Stream without one complete monotonic row-major access sweep
module {
  func.func @transposed(%input: memref<8x8xf32>,
                        %output: memref<8x8xf32>) attributes {top_func} {
    affine.for %j = 0 to 8 {
      affine.for %i = 0 to 8 {
        %value = affine.load %input[%i, %j] : memref<8x8xf32>
        affine.store %value, %output[%j, %i] : memref<8x8xf32>
      }
    }
    return
  }
}
