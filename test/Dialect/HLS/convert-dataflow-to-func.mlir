// RUN: sar-opt %s --hls-convert-dataflow-to-func | FileCheck %s

// Globals materialized into const buffers disappear when unused, but globals
// referenced by unrelated functions must remain valid.

// CHECK: memref.global "private" constant @used
// CHECK-NOT: @unused
memref.global "private" constant @used : memref<4xf32> = dense<1.0>
memref.global "private" constant @unused : memref<4xf32> = dense<2.0>

// CHECK-LABEL: func.func @helper
// CHECK: memref.get_global @used
func.func @helper() -> memref<4xf32> {
  %0 = memref.get_global @used : memref<4xf32>
  return %0 : memref<4xf32>
}

// A direct child of a legal top-level schedule is a dataflow process. It must
// remain a function boundary rather than carrying an inline pragma that Vitis
// reports as conflicting with dataflow canonical form.
// CHECK-LABEL: func.func @legal_process_node0
// CHECK-NOT: inline
// CHECK: return
// CHECK-LABEL: func.func @legal_process
// CHECK-SAME: dataflow = true
func.func @legal_process(%input: memref<8xf32>, %output: memref<8xf32>) {
  hls.dataflow.schedule legal(%input, %output)
      : memref<8xf32>, memref<8xf32> {
  ^bb0(%in: memref<8xf32>, %out: memref<8xf32>):
    hls.dataflow.node(%in) -> (%out) {inputTaps = [0 : i32], level = 0 : i32}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%src: memref<8xf32>, %dst: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %value = affine.load %src[%i] : memref<8xf32>
        affine.store %value, %dst[%i] : memref<8xf32>
      }
    }
  }
  return
}
