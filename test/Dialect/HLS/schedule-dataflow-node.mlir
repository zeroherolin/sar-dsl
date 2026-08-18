// RUN: sar-opt %s --hls-schedule-dataflow-node | FileCheck %s
// RUN: sar-opt %s --hls-schedule-dataflow-node="ignore-violations=true" \
// RUN:   | FileCheck %s --check-prefix=IGNORE

// ALAP scheduling: a node's level must be one above its highest consumer, so
// sinks land at level 0 and levels count hops to the end of the pipeline.

// CHECK-LABEL: func.func @chain
// CHECK: hls.dataflow.node(%{{.*}}) -> (%{{.*}}) {inputTaps = [0 : i32], level = 1 : i32}
// CHECK: hls.dataflow.node(%{{.*}}) -> (%{{.*}}) {inputTaps = [0 : i32], level = 0 : i32}
func.func @chain(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %b = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
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

// An internal buffer read by two nodes is a violation the fork pass has not
// resolved yet: assigning its producer a level now would bake the broken graph
// into the schedule. The producer must stay unscheduled (no level attribute,
// pinned by the bare "} :" after inputTaps) until told to ignore violations.

// CHECK-LABEL: func.func @multi_consumer
// CHECK: hls.dataflow.node(%{{.*}}) -> (%{{.*}}) {inputTaps = [0 : i32]} :
// CHECK-COUNT-2: level = 0 : i32

// IGNORE-LABEL: func.func @multi_consumer
// IGNORE: hls.dataflow.node(%{{.*}}) -> (%{{.*}}) {inputTaps = [0 : i32], level = 1 : i32}
// IGNORE-COUNT-2: level = 0 : i32
func.func @multi_consumer(%arg0: memref<8xf32>) attributes {top_func} {
  hls.dataflow.schedule(%arg0) : memref<8xf32> {
  ^bb0(%in: memref<8xf32>):
    %a = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %b = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    %c = hls.dataflow.buffer {depth = 1 : i32} : memref<8xf32>
    hls.dataflow.node(%in) -> (%a) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    hls.dataflow.node(%a) -> (%b) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
    hls.dataflow.node(%a) -> (%c) {inputTaps = [0 : i32]} : (memref<8xf32>) -> memref<8xf32> {
    ^bb0(%i: memref<8xf32>, %o: memref<8xf32>):
      affine.for %j = 0 to 8 {
        %v = affine.load %i[%j] : memref<8xf32>
        affine.store %v, %o[%j] : memref<8xf32>
      }
    }
  }
  return
}
