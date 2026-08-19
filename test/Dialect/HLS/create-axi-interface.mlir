// RUN: sar-opt %s --split-input-file --hls-create-axi-interface --verify-diagnostics | FileCheck %s

// The top function's ports are the design's contract with its host: the
// kernel's own arrays, then two scratch pointers. Internal DRAM buffers are
// distributed between those allocations and carved at compile-time offsets
// instead of becoming more ports. Every port takes its own AXI bundle:
// ports sharing one serialize their bus requests. A buffer that cannot be
// redirected to the scratch allocation is rejected instead of silently
// changing the host ABI.

// CHECK-LABEL: func.func @carved(
// CHECK-SAME: %arg0: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg1: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg2: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg3: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK: hls.axi.bundle
// CHECK-NEXT: hls.axi.port
// CHECK: hls.axi.bundle
// CHECK-NEXT: hls.axi.port
// CHECK: hls.axi.bundle
// CHECK-NEXT: hls.axi.port
// CHECK: hls.axi.bundle
// CHECK-NEXT: hls.axi.port
// CHECK-NOT: hls.axi.bundle
// CHECK-NOT: hls.axi.port
module {
  func.func @carved_node0(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %arg1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
  func.func @carved_node1(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %c = arith.constant 1.0 : f32
      %w = arith.addf %v, %c : f32
      affine.store %w, %arg1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
  func.func @carved_node2(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %c = arith.constant 2.0 : f32
      %w = arith.mulf %v, %c : f32
      affine.store %w, %arg1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
  // Two 64-element DRAM intermediates occupy separate physical masters.
  func.func @carved(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) attributes {top_func} {
    %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<64xf32, #hls.mem<dram>>
    %mid2 = hls.dataflow.buffer {depth = 1 : i32} : memref<64xf32, #hls.mem<dram>>
    call @carved_node0(%arg0, %mid) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    call @carved_node1(%mid, %mid2) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    call @carved_node2(%mid2, %arg1) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    return
  }
}

// The runtime wrapper binds the carved buffers for the host: same array
// arguments, no AXI types.
// CHECK: func.func @main(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>)
// CHECK-SAME: runtime

// -----

// The default top name is also the wrapper name. Rename the implementation
// before creating the wrapper rather than rejecting a default invocation.
module {
  // CHECK-LABEL: func.func @main_impl(
  // CHECK: hls.axi.port
  func.func @main(%arg0: memref<8xf32, #hls.mem<dram>>) attributes {top_func} {
    return
  }
  // CHECK-LABEL: func.func @main(
  // CHECK-SAME: attributes {runtime}
  // CHECK: call @main_impl
}

// -----

// A DRAM buffer handed to a callee that several call sites share cannot be
// carved because one offset cannot serve two buffers. Reject the design:
// optimizer-created buffers may not leak into the top-level interface.
module {
  func.func @shared_callee_node0(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %arg1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
  func.func @shared_callee(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) attributes {top_func} {
    // expected-error @+1 {{internal DRAM buffer cannot be carved}}
    %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<64xf32, #hls.mem<dram>>
    call @shared_callee_node0(%arg0, %mid) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    call @shared_callee_node0(%mid, %arg1) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    return
  }
}
