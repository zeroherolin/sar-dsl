// RUN: sar-opt --sar-emit-c-interface %s | FileCheck %s

// Only public functions (kernel entry points) get llvm.emit_c_interface.

// CHECK: func.func @kernel
// CHECK-SAME: llvm.emit_c_interface
func.func @kernel(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// CHECK: func.func private @helper
// CHECK-NOT: llvm.emit_c_interface
func.func private @helper(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// CHECK: func.func private @sar_rt_decl
// CHECK-NOT: llvm.emit_c_interface
func.func private @sar_rt_decl(tensor<8xf32>)
