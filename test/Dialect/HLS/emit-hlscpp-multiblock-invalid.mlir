// RUN: ! sar-translate --hls-emit-hlscpp %s 2>&1 | FileCheck %s

func.func @multi_block(%flag: i1) attributes {top_func} {
  // CHECK: error: HLS C++ target requires exactly one basic block
  cf.cond_br %flag, ^left, ^right
^left:
  return
^right:
  return
}
