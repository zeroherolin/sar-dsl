// RUN: sar-opt %s --hls-lower-dataflow --verify-diagnostics

// A high-level task result cannot become a node result. The pass must reject
// it directly instead of entering dialect-conversion rollback with live uses.
func.func @task_result() -> f32 {
  %result = hls.dataflow.dispatch : f32 {
    // expected-error@+1 {{'hls.dataflow.task' op should not yield any results}}
    %task = hls.dataflow.task : f32 {
      %value = arith.constant 1.0 : f32
      hls.dataflow.yield %value : f32
    }
    hls.dataflow.yield %task : f32
  }
  return %result : f32
}
