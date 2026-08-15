// RUN: sar-opt %s --sar-reuse-buffers="min-elements=0" | FileCheck %s
// RUN: sar-opt %s --sar-reuse-buffers="min-elements=100" \
// RUN:   | FileCheck %s --check-prefix=ABOVE

// Bufferization gives every intermediate its own allocation, so a chain of
// whole-raster passes holds far more buffers than are live at once. A
// buffer whose last use has passed can carry a later one of the same type.

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

// Buffers of different types never share, whatever their lifetimes.

// CHECK-LABEL: func.func @distinct_types
// CHECK: memref.alloc() : memref<64xf64>
// CHECK: memref.alloc() : memref<64xf32>
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
