// RUN: sar-opt %s --sar-reuse-buffers="min-elements=0" | FileCheck %s
// RUN: sar-opt %s --sar-reuse-buffers="min-elements=100" \
// RUN:   | FileCheck %s --check-prefix=ABOVE
// RUN: sar-opt %s --sar-reuse-buffers="min-elements=0 allow-retype=true" \
// RUN:   | FileCheck %s --check-prefix=RETYPE

// Three buffers, each live only between adjacent passes: the first is dead
// by the time the third is written, so two allocations suffice.

// CHECK-LABEL: func.func @chain
// CHECK-COUNT-2: memref.alloc() : memref<64xf64>
// CHECK-NOT: memref.alloc()

// `min-elements` selects which buffers take part. Above the threshold a
// buffer acts as a dataflow channel, where distinct producers are what let
// a backend pipeline the stages, so none are shared.

// ABOVE-LABEL: func.func @chain
// ABOVE-COUNT-3: memref.alloc() : memref<64xf64>
// ABOVE: return
func.func @chain(%in: memref<64xf64>, %out: memref<64xf64>) {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<64xf64>
  %c = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf64>
    affine.store %v, %b[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %b[%i] : memref<64xf64>
    affine.store %v, %c[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %c[%i] : memref<64xf64>
    affine.store %v, %out[%i] : memref<64xf64>
  }
  return
}

// -----

// Buffers of different types never share, whatever their lifetimes, unless
// `allow-retype` says a byte-addressed backing allocation is acceptable.
// Here the two overlap anyway -- the second loop reads %a while writing %b --
// so neither setting may merge them.

// CHECK-LABEL: func.func @distinct_types
// CHECK: memref.alloc() : memref<64xf64>
// CHECK: memref.alloc() : memref<64xf32>

// RETYPE-LABEL: func.func @distinct_types
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE: memref.alloc() : memref<64xf32>
func.func @distinct_types(%in: memref<64xf64>, %out: memref<64xf32>) {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf64>
    %t = arith.truncf %v : f64 to f32
    affine.store %t, %b[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %b[%i] : memref<64xf32>
    affine.store %v, %out[%i] : memref<64xf32>
  }
  return
}

// -----

// A mixed-precision chain: an f32 plane retires before a c64 plane of half
// the element count is first written, and both occupy 256 bytes. Without
// `allow-retype` they stay separate; with it one byte-addressed allocation
// backs both, each reinterpreting it at offset zero.

// CHECK-LABEL: func.func @mixed_precision
// CHECK: memref.alloc() : memref<64xf32>
// CHECK: memref.alloc() : memref<32xcomplex<f32>>
// CHECK-NOT: memref.view

// RETYPE-LABEL: func.func @mixed_precision
// RETYPE: %[[BACK:.*]] = memref.alloc() {alignment = 8 : i64} : memref<256xi8>
// RETYPE-DAG: memref.view %[[BACK]]{{.*}} : memref<256xi8> to memref<64xf32>
// RETYPE-DAG: memref.view %[[BACK]]{{.*}} : memref<256xi8> to memref<32xcomplex<f32>>
// RETYPE-NOT: memref.alloc
func.func @mixed_precision(%in: memref<64xf32>, %tmp: memref<64xf32>,
                           %out: memref<32xcomplex<f32>>) {
  %a = memref.alloc() : memref<64xf32>
  %b = memref.alloc() : memref<32xcomplex<f32>>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf32>
    affine.store %v, %a[%i] : memref<64xf32>
  }
  // %a's last use; %b is not live yet.
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf32>
    affine.store %v, %tmp[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %tmp[%i] : memref<64xf32>
    %c = complex.create %v, %v : complex<f32>
    affine.store %c, %b[%i] : memref<32xcomplex<f32>>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %b[%i] : memref<32xcomplex<f32>>
    affine.store %v, %out[%i] : memref<32xcomplex<f32>>
  }
  return
}

// -----

// The backing allocation takes the strictest alignment of the group. MLIR's
// data layout gives complex<f64> the alignment of its f64 element, but the
// value is 16 bytes wide, so the shared storage is aligned to 16 rather than
// to the 8 an f64 plane alone would have needed.

// RETYPE-LABEL: func.func @alignment_follows_widest
// RETYPE: memref.alloc() {alignment = 16 : i64} : memref<512xi8>
// RETYPE-DAG: memref.view {{.*}} to memref<64xf64>
// RETYPE-DAG: memref.view {{.*}} to memref<32xcomplex<f64>>
func.func @alignment_follows_widest(%in: memref<64xf64>, %tmp: memref<64xf64>,
                                    %out: memref<32xcomplex<f64>>) {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<32xcomplex<f64>>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf64>
    affine.store %v, %tmp[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %tmp[%i] : memref<64xf64>
    %c = complex.create %v, %v : complex<f64>
    affine.store %c, %b[%i] : memref<32xcomplex<f64>>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %b[%i] : memref<32xcomplex<f64>>
    affine.store %v, %out[%i] : memref<32xcomplex<f64>>
  }
  return
}

// -----

// A retired buffer backs a newer one only when the newer one fits. Here the
// f32 plane retires first but holds 128 bytes, and the c64 plane that follows
// needs 256, so the group never forms -- the footprint of a shared allocation
// is fixed by the buffer that opened it and never grows.

// RETYPE-LABEL: func.func @too_small_to_back
// RETYPE: memref.alloc() : memref<32xf32>
// RETYPE: memref.alloc() : memref<32xcomplex<f32>>
// RETYPE-NOT: memref.view
func.func @too_small_to_back(%in: memref<32xf32>, %tmp: memref<32xf32>,
                             %out: memref<32xcomplex<f32>>) {
  %a = memref.alloc() : memref<32xf32>
  %b = memref.alloc() : memref<32xcomplex<f32>>
  affine.for %i = 0 to 32 {
    %v = affine.load %in[%i] : memref<32xf32>
    affine.store %v, %a[%i] : memref<32xf32>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %a[%i] : memref<32xf32>
    affine.store %v, %tmp[%i] : memref<32xf32>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %tmp[%i] : memref<32xf32>
    %c = complex.create %v, %v : complex<f32>
    affine.store %c, %b[%i] : memref<32xcomplex<f32>>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %b[%i] : memref<32xcomplex<f32>>
    affine.store %v, %out[%i] : memref<32xcomplex<f32>>
  }
  return
}

// -----

// A buffer handing out an alias is never a candidate: the subview outlives
// the span the scan measures, so sharing %a would write through it.

// RETYPE-LABEL: func.func @live_subview
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE-NOT: memref.view {{.*}} to memref<64xf64>
func.func @live_subview(%in: memref<64xf64>, %out: memref<64xf64>) {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  %s = memref.subview %a[0][32][1] : memref<64xf64> to memref<32xf64, strided<[1]>>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %b[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %s[%i] : memref<32xf64, strided<[1]>>
    affine.store %v, %out[%i] : memref<64xf64>
  }
  return
}

// -----

// A tensor handed out by `bufferization.to_tensor` aliases the buffer just as
// a memref would, and the tensor here outlives the buffer's own last use.
// Sharing %a would overwrite what the returned tensor refers to.

// RETYPE-LABEL: func.func @live_tensor_alias
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE-NOT: memref.view
func.func @live_tensor_alias(%in: memref<64xf64>) -> tensor<64xf64> {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  %t = bufferization.to_tensor %a : memref<64xf64> to tensor<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %b[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %b[%i] : memref<64xf64>
    affine.store %v, %b[%i] : memref<64xf64>
  }
  return %t : tensor<64xf64>
}

// -----

// The shared allocation has to dominate every use it takes over. Here the
// c64 plane is allocated late -- after the f32 plane has already been read --
// so the backing allocation belongs where the earliest member's did, not
// where the group's first-used member happens to sit.

// RETYPE-LABEL: func.func @late_allocation
// RETYPE: %[[BACK:.*]] = memref.alloc() {alignment = 8 : i64} : memref<256xi8>
// RETYPE-NEXT: %[[ZERO:.*]] = arith.constant 0 : index
// RETYPE-NEXT: memref.view %[[BACK]][%[[ZERO]]][] : memref<256xi8> to memref<64xf32>
// RETYPE-NEXT: memref.view %[[BACK]][%[[ZERO]]][] : memref<256xi8> to memref<32xcomplex<f32>>
func.func @late_allocation(%in: memref<64xf32>, %tmp: memref<64xf32>,
                           %out: memref<32xcomplex<f32>>) {
  %a = memref.alloc() : memref<64xf32>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf32>
    affine.store %v, %a[%i] : memref<64xf32>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf32>
    affine.store %v, %tmp[%i] : memref<64xf32>
  }
  // Allocated only once the f32 plane has retired.
  %b = memref.alloc() : memref<32xcomplex<f32>>
  affine.for %i = 0 to 32 {
    %v = affine.load %tmp[%i] : memref<64xf32>
    %c = complex.create %v, %v : complex<f32>
    affine.store %c, %b[%i] : memref<32xcomplex<f32>>
  }
  affine.for %i = 0 to 32 {
    %v = affine.load %b[%i] : memref<32xcomplex<f32>>
    affine.store %v, %out[%i] : memref<32xcomplex<f32>>
  }
  return
}

// -----

// A linear scan orders operations only along a single straight-line block. A
// function with branches has no such order, so nothing is shared -- and the
// scan must not walk off the top of the IR looking for a position.

// RETYPE-LABEL: func.func @branching
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE: memref.alloc() : memref<64xf64>
// RETYPE-NOT: memref.view
func.func @branching(%in: memref<64xf64>, %out: memref<64xf64>) {
  %a = memref.alloc() : memref<64xf64>
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %in[%i] : memref<64xf64>
    affine.store %v, %a[%i] : memref<64xf64>
  }
  cf.br ^bb1
^bb1:
  affine.for %i = 0 to 64 {
    %v = affine.load %a[%i] : memref<64xf64>
    affine.store %v, %b[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %b[%i] : memref<64xf64>
    affine.store %v, %out[%i] : memref<64xf64>
  }
  return
}

// -----

// An alias escaping through an *unranked* cast pins the buffer just like a
// ranked one: the cast result may be read after the alloc's last visible use,
// so %b may not take over %a.

// CHECK-LABEL: func.func @unranked_cast_alias
// CHECK-COUNT-2: memref.alloc() : memref<64xf64>
func.func @unranked_cast_alias(%out: memref<64xf64>) {
  %a = memref.alloc() : memref<64xf64>
  %u = memref.cast %a : memref<64xf64> to memref<*xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 1.0 : f64
    affine.store %c, %a[%i] : memref<64xf64>
  }
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 2.0 : f64
    affine.store %c, %b[%i] : memref<64xf64>
  }
  %r = memref.cast %u : memref<*xf64> to memref<64xf64>
  affine.for %i = 0 to 64 {
    %v = affine.load %r[%i] : memref<64xf64>
    %w = affine.load %b[%i] : memref<64xf64>
    %s = arith.addf %v, %w : f64
    affine.store %s, %out[%i] : memref<64xf64>
  }
  return
}

// -----

// A buffer stored *as a value* into another memref escapes through that
// table: whoever loads it back holds an alias the scan cannot see.

// CHECK-LABEL: func.func @stored_into_table
// CHECK-COUNT-2: memref.alloc() : memref<64xf64>
func.func @stored_into_table(%table: memref<1xmemref<64xf64>>,
                             %out: memref<64xf64>) {
  %c0 = arith.constant 0 : index
  %a = memref.alloc() : memref<64xf64>
  memref.store %a, %table[%c0] : memref<1xmemref<64xf64>>
  affine.for %i = 0 to 64 {
    %c = arith.constant 1.0 : f64
    affine.store %c, %a[%i] : memref<64xf64>
  }
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 2.0 : f64
    affine.store %c, %b[%i] : memref<64xf64>
  }
  %r = memref.load %table[%c0] : memref<1xmemref<64xf64>>
  affine.for %i = 0 to 64 {
    %v = affine.load %r[%i] : memref<64xf64>
    %w = affine.load %b[%i] : memref<64xf64>
    %s = arith.addf %v, %w : f64
    affine.store %s, %out[%i] : memref<64xf64>
  }
  return
}

// -----

// Raw pointer extraction escapes the address itself; nothing downstream of
// an index can be tracked, so the buffer is left alone.

// CHECK-LABEL: func.func @pointer_escape
// CHECK-COUNT-2: memref.alloc() : memref<64xf64>
func.func @pointer_escape(%out: memref<index>) {
  %a = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 1.0 : f64
    affine.store %c, %a[%i] : memref<64xf64>
  }
  %p = memref.extract_aligned_pointer_as_index %a : memref<64xf64> -> index
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 2.0 : f64
    affine.store %c, %b[%i] : memref<64xf64>
  }
  memref.store %p, %out[] : memref<index>
  return
}

// -----

// A terminator forwarding the buffer (`scf.yield`) hands it to its parent
// op, whose result aliases it past the span the scan measures.

// CHECK-LABEL: func.func @yielded_alias
// CHECK-COUNT-2: memref.alloc() : memref<64xf64>
func.func @yielded_alias(%cond: i1, %out: memref<64xf64>) {
  %a = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 1.0 : f64
    affine.store %c, %a[%i] : memref<64xf64>
  }
  %m = scf.if %cond -> memref<64xf64> {
    scf.yield %a : memref<64xf64>
  } else {
    scf.yield %a : memref<64xf64>
  }
  %b = memref.alloc() : memref<64xf64>
  affine.for %i = 0 to 64 {
    %c = arith.constant 2.0 : f64
    affine.store %c, %b[%i] : memref<64xf64>
  }
  affine.for %i = 0 to 64 {
    %v = affine.load %m[%i] : memref<64xf64>
    %w = affine.load %b[%i] : memref<64xf64>
    %s = arith.addf %v, %w : f64
    affine.store %s, %out[%i] : memref<64xf64>
  }
  return
}
