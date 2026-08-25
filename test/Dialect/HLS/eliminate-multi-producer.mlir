// RUN: sar-opt %s --hls-eliminate-multi-producer | FileCheck %s

// Two nodes writing one internal buffer would be two processes driving one
// channel. The pass privatizes: the later producer writes a fresh buffer and
// takes the original as an extra input, and consumers below it move to the
// new buffer -- every internal buffer ends with a single producer.

// CHECK-LABEL: func.func @privatize
func.func @privatize(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    // CHECK: %[[A:.*]] = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // CHECK: %[[B:.*]] = hls.dataflow.buffer
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %b = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // CHECK: hls.dataflow.node(%{{.*}}) -> (%[[A]])
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 4 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    // The dominated write-only producer is rebuilt around a private buffer;
    // it does not carry the old buffer as an unused input.
    // CHECK: %[[NEW:.*]] = hls.dataflow.buffer
    // CHECK: hls.dataflow.node(%{{.*}}) -> (%[[NEW]])
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 4 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    // The downstream consumer now reads the private buffer.
    // CHECK: hls.dataflow.node(%[[NEW]]) -> (%[[B]])
    hls.dataflow.node(%a) -> (%b) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
  }
  return
}

// An external buffer cannot be privatized -- it is the interface -- so two
// adjacent producers of it are fused into one node instead.

// CHECK-LABEL: func.func @merge_external
// CHECK: hls.dataflow.node() -> (%{{.*}})
// CHECK: affine.for %{{.*}} = 0 to 4
// CHECK: affine.for %{{.*}} = 4 to 8
// CHECK-NOT: hls.dataflow.node
func.func @merge_external(%arg0: memref<8xf32, #hls.mem<dram>>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32, #hls.mem<dram>> {
  ^bb0(%in: memref<8xf32, #hls.mem<dram>>):
    hls.dataflow.node() -> (%in) {inputTaps = []} : () -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%o: memref<8xf32, #hls.mem<dram>>):
      %c = arith.constant 1.0 : f32
      affine.for %j = 0 to 4 {
        affine.store %c, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
    hls.dataflow.node() -> (%in) {inputTaps = []} : () -> memref<8xf32, #hls.mem<dram>> {
    ^bb0(%o: memref<8xf32, #hls.mem<dram>>):
      %c = arith.constant 2.0 : f32
      affine.for %j = 4 to 8 {
        affine.store %c, %o[%j] : memref<8xf32, #hls.mem<dram>>
      }
    }
  }
  return
}

// A later producer that fully writes its private buffer before reading it
// must not copy the old buffer over the new values.
// CHECK-LABEL: func.func @full_write_before_read
func.func @full_write_before_read(%input: memref<8xf32>)
    attributes {top_func} {
  hls.dataflow.schedule(%input) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    // CHECK: %[[STATE:.*]] = hls.dataflow.buffer
    %state = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    // CHECK: %[[SINK:.*]] = hls.dataflow.buffer
    %sink = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%state) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%src: memref<8xf32>, %dst: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %src[%i] : memref<8xf32>
        affine.store %v, %dst[%i] : memref<8xf32>
      }
    }
    // CHECK: %[[PRIVATE:.*]] = hls.dataflow.buffer
    // CHECK: hls.dataflow.node(%{{.*}}) -> (%[[PRIVATE]], %[[SINK]])
    // CHECK-NOT: memref.copy
    hls.dataflow.node(%in) -> (%state, %sink) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> (memref<8xf32>, memref<8xf32>) {
    ^bb0(%src: memref<8xf32>, %dst: memref<8xf32>, %out: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %src[%i] : memref<8xf32>
        affine.store %v, %dst[%i] : memref<8xf32>
      }
      affine.for %i = 0 to 8 {
        %v = affine.load %dst[%i] : memref<8xf32>
        affine.store %v, %out[%i] : memref<8xf32>
      }
    }
  }
  return
}

// A partial producer that reads the prior contents before writing still needs
// an entry copy into its private buffer.
// CHECK-LABEL: func.func @read_before_partial_write
func.func @read_before_partial_write(%input: memref<8xf32>)
    attributes {top_func} {
  hls.dataflow.schedule(%input) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %state = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %sink = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%state) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%src: memref<8xf32>, %dst: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %src[%i] : memref<8xf32>
        affine.store %v, %dst[%i] : memref<8xf32>
      }
    }
    // CHECK: hls.dataflow.node(%{{.*}}, %{{.*}}) -> (%{{.*}}, %{{.*}})
    // CHECK: ^bb0(%{{.*}}, %[[OLD:[a-zA-Z0-9_]+]]: memref<8xf32>, %[[NEW:[a-zA-Z0-9_]+]]: memref<8xf32>, %{{.*}}):
    // CHECK-NEXT: memref.copy %[[OLD]], %[[NEW]]
    hls.dataflow.node(%in) -> (%state, %sink) {inputTaps = [0 : i32]}
        : (memref<8xf32>) -> (memref<8xf32>, memref<8xf32>) {
    ^bb0(%src: memref<8xf32>, %dst: memref<8xf32>, %out: memref<8xf32>):
      affine.for %i = 0 to 8 {
        %v = affine.load %dst[%i] : memref<8xf32>
        affine.store %v, %out[%i] : memref<8xf32>
      }
      affine.for %i = 4 to 8 {
        %v = affine.load %src[%i] : memref<8xf32>
        affine.store %v, %dst[%i] : memref<8xf32>
      }
    }
  }
  return
}
