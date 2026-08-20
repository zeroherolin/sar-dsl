// RUN: sar-opt %s --split-input-file --hls-create-axi-interface --verify-diagnostics | FileCheck %s

// The top function's ports are the design's contract with its host, so they
// are the algorithm's own I/O and nothing else: the kernel's arrays, then the
// scratch pointers the spilled buffers need. Internal DRAM buffers are carved
// out of an arena at compile-time offsets instead of becoming ports of their
// own, and a type with no spill contributes no port. Buffers a single node
// reads and writes get separate arenas -- one pointer both loaded from and
// stored to forces HLS to serialize that node's bus traffic. Every remaining
// port takes its own AXI bundle: ports sharing one serialize their bus
// requests. A buffer that cannot be redirected to a scratch allocation is
// rejected instead of silently changing the host ABI.

// CHECK-LABEL: func.func @carved(
// CHECK-SAME: %arg0: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg1: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg2: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg3: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-NOT: %arg4
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
  // The middle node reads one intermediate and writes the other, so the two
  // cannot share a pointer: they get an arena each. Neither becomes a port in
  // its own right -- the ports are the two arenas, carved at aligned offsets.
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

// -----

// One arena per element type that spills, and none for a type that does not.
// The f64 kernel array is an argument, not an internal buffer, so it stays a
// port of its own and contributes no f64 arena: the design gets a single f32
// scratch port, not one per type in the signature.

// CHECK-LABEL: func.func @typed_arenas(
// CHECK-SAME: %arg0: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-SAME: %arg1: !hls.axi<memref<64xf64, #hls.mem<dram>>>
// CHECK-SAME: %arg2: !hls.axi<memref<64xf32, #hls.mem<dram>>>
// CHECK-NOT: %arg3
module {
  func.func @typed_arenas_node0(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf32, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %w = arith.mulf %v, %v : f32
      affine.store %w, %arg1[%i] : memref<64xf32, #hls.mem<dram>>
    }
    return
  }
  func.func @typed_arenas_node1(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf64, #hls.mem<dram>>) {
    affine.for %i = 0 to 64 {
      %v = affine.load %arg0[%i] : memref<64xf32, #hls.mem<dram>>
      %w = arith.extf %v : f32 to f64
      affine.store %w, %arg1[%i] : memref<64xf64, #hls.mem<dram>>
    }
    return
  }
  func.func @typed_arenas(%arg0: memref<64xf32, #hls.mem<dram>>, %arg1: memref<64xf64, #hls.mem<dram>>) attributes {top_func} {
    %mid = hls.dataflow.buffer {depth = 1 : i32} : memref<64xf32, #hls.mem<dram>>
    call @typed_arenas_node0(%arg0, %mid) : (memref<64xf32, #hls.mem<dram>>, memref<64xf32, #hls.mem<dram>>) -> ()
    call @typed_arenas_node1(%mid, %arg1) : (memref<64xf32, #hls.mem<dram>>, memref<64xf64, #hls.mem<dram>>) -> ()
    return
  }
}
