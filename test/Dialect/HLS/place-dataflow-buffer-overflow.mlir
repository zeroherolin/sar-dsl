// RUN: sar-opt %s --hls-place-dataflow-buffer="threshold=4096 bram-bytes=4608 uram-bytes=0 lutram-bytes=0 lutram-max-bytes=64" --verify-diagnostics

// The budgets are hard caps: a per-iteration scratch cannot stream (no
// AXI port reaches through an scf.for region), and with block RAM
// smaller than the scratch and the URAM tier forbidden the pass refuses
// the design instead of overcommitting the device -- an over-budget
// design is permanently invalid for the user.

// expected-error @below {{constant tables and per-iteration buffers need}}
func.func @scratch_over_budget(%v: f32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %scratch = hls.dataflow.buffer {depth = 1 : i32} : memref<8192xf32>
    affine.for %j = 0 to 8192 {
      affine.store %v, %scratch[%j] : memref<8192xf32>
    }
  }
  return
}
