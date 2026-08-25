// RUN: sar-opt %s --hls-fuse-sibling-loops="preserve-dataflow-roles=true" --cse | FileCheck %s

// CHECK-LABEL: func.func @share_phase
// CHECK-COUNT-1: affine.for
// CHECK-COUNT-1: math.sin
// CHECK-COUNT-3: affine.store
func.func @share_phase(%input: memref<16xf64>, %re: memref<16xf64>,
                       %im: memref<16xf64>, %extra: memref<16xf64>) {
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    affine.store %s, %re[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    %neg = arith.negf %s : f64
    affine.store %neg, %im[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %x = affine.load %input[%i] : memref<16xf64>
    %s = math.sin %x : f64
    affine.store %s, %extra[%i] : memref<16xf64>
  }
  return
}

// A pointwise producer-consumer pair is safe to interleave. The store
// forwarding pass later removes the intermediate traffic entirely.
// CHECK-LABEL: func.func @pointwise_producer_consumer
// CHECK-COUNT-1: affine.for
func.func @pointwise_producer_consumer(%buffer: memref<16xf64>,
                                       %out: memref<16xf64>) {
  affine.for %i = 0 to 16 {
    %zero = arith.constant 0.0 : f64
    affine.store %zero, %buffer[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %value = affine.load %buffer[%i] : memref<16xf64>
    affine.store %value, %out[%i] : memref<16xf64>
  }
  return
}

// A center load plus a dynamic neighbour load is not pointwise. Interleaving
// the loops would let the consumer observe a neighbour the producer has not
// written yet.
// CHECK-LABEL: func.func @dynamic_neighbour_dependency
// CHECK-COUNT-2: affine.for
func.func @dynamic_neighbour_dependency(%input: memref<16xf64>,
                                        %buffer: memref<16xf64>,
                                        %out: memref<16xf64>) {
  affine.for %i = 0 to 16 {
    %value = affine.load %input[%i] : memref<16xf64>
    affine.store %value, %buffer[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %c1 = arith.constant 1 : index
    %c15 = arith.constant 15 : index
    %next = arith.addi %i, %c1 : index
    %clamped = arith.minui %next, %c15 : index
    %center = affine.load %buffer[%i] : memref<16xf64>
    %neighbour = memref.load %buffer[%clamped] : memref<16xf64>
    %sum = arith.addf %center, %neighbour : f64
    affine.store %sum, %out[%i] : memref<16xf64>
  }
  return
}

// A process that feeds an internal successor must not be fused with an
// independent public-result sweep. Vitis otherwise reports HLS 200-1450 and
// may serialize the successor behind the caller output.
// CHECK-LABEL: func.func @result_and_successor_roles
// CHECK-COUNT-3: affine.for
func.func @result_and_successor_roles(%lhs: memref<16xf64>,
                                     %rhs: memref<16xf64>,
                                     %out: memref<16xf64>,
                                     %extra: memref<8xf64>)
    attributes {sar.arg_names = ["lhs", "rhs"]} {
  %channel = memref.alloc() : memref<16xf64>
  affine.for %i = 0 to 16 {
    %value = affine.load %lhs[%i] : memref<16xf64>
    affine.store %value, %channel[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 16 {
    %value = affine.load %rhs[%i] : memref<16xf64>
    affine.store %value, %out[%i] : memref<16xf64>
  }
  affine.for %i = 0 to 8 {
    %value = affine.load %channel[%i] : memref<16xf64>
    affine.store %value, %extra[%i] : memref<8xf64>
  }
  return
}
