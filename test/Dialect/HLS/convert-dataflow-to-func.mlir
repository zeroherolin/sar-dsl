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
