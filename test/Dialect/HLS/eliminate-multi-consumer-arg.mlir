// RUN: sar-opt %s --hls-eliminate-multi-consumer | FileCheck %s

// A top-level input has no producer node, so the fork pattern for node
// outputs never fires on it. Vitis binds a dual-port memory to at most
// two reader processes; a wider fan-out needs an explicit fork copying
// the input into per-consumer buffers, leaving the port one reader.

// CHECK-LABEL: func.func @wide_fanout
// CHECK: hls.dataflow.schedule
// CHECK: ^bb0(%[[F:arg[0-9]+]]: memref<8xf64>,
// CHECK-COUNT-3: hls.dataflow.buffer {depth = 1 : i32} : memref<8xf64>
// CHECK: hls.dataflow.node(%[[F]]) -> (%[[B0:[0-9]+]], %[[B1:[0-9]+]], %[[B2:[0-9]+]])
// CHECK-COUNT-3: memref.copy
// CHECK-DAG: hls.dataflow.node(%[[B0]])
// CHECK-DAG: hls.dataflow.node(%[[B1]])
// CHECK-DAG: hls.dataflow.node(%[[B2]])
func.func @wide_fanout(%f: memref<8xf64>, %a: memref<8xf64>,
                       %b: memref<8xf64>, %c: memref<8xf64>) {
  hls.dataflow.schedule(%f, %a, %b, %c)
      : memref<8xf64>, memref<8xf64>, memref<8xf64>, memref<8xf64> {
  ^bb0(%sf: memref<8xf64>, %sa: memref<8xf64>, %sb: memref<8xf64>,
       %sc: memref<8xf64>):
    hls.dataflow.node(%sf) -> (%sa) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
    hls.dataflow.node(%sf) -> (%sb) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
    hls.dataflow.node(%sf) -> (%sc) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
  }
  return
}

// Even two readers need a fork: Vitis HLS dataflow legality is per process,
// not per BRAM port count. Constant tables are the separate exception below.

// CHECK-LABEL: func.func @two_readers
// CHECK-COUNT-2: hls.dataflow.buffer {depth = 1 : i32} : memref<8xf64>
// CHECK: hls.dataflow.node
// CHECK-COUNT-2: memref.copy
func.func @two_readers(%f: memref<8xf64>, %a: memref<8xf64>,
                       %b: memref<8xf64>) {
  hls.dataflow.schedule(%f, %a, %b)
      : memref<8xf64>, memref<8xf64>, memref<8xf64> {
  ^bb0(%sf: memref<8xf64>, %sa: memref<8xf64>, %sb: memref<8xf64>):
    hls.dataflow.node(%sf) -> (%sa) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
    hls.dataflow.node(%sf) -> (%sb) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
  }
  return
}

// A constant table read by three nodes stays shared: the emitter hoists
// it to file scope where the tool replicates the ROM per reader, so a
// fork here would only turn one ROM into per-consumer ping-pong RAM.

// CHECK-LABEL: func.func @const_table_fanout
// CHECK-NOT: hls.dataflow.buffer
// CHECK-NOT: memref.copy
func.func @const_table_fanout(%a: memref<8xf64>, %b: memref<8xf64>,
                              %c: memref<8xf64>) {
  %t = hls.dataflow.const_buffer {value = dense<1.0> : tensor<8xf64>}
      : memref<8xf64>
  hls.dataflow.schedule(%t, %a, %b, %c)
      : memref<8xf64>, memref<8xf64>, memref<8xf64>, memref<8xf64> {
  ^bb0(%st: memref<8xf64>, %sa: memref<8xf64>, %sb: memref<8xf64>,
       %sc: memref<8xf64>):
    hls.dataflow.node(%st) -> (%sa) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
    hls.dataflow.node(%st) -> (%sb) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
    hls.dataflow.node(%st) -> (%sc) {inputTaps = [0 : i32]}
        : (memref<8xf64>) -> memref<8xf64> {
    ^bb0(%in: memref<8xf64>, %out: memref<8xf64>):
      affine.for %i = 0 to 8 {
        %v = affine.load %in[%i] : memref<8xf64>
        affine.store %v, %out[%i] : memref<8xf64>
      }
    }
  }
  return
}
