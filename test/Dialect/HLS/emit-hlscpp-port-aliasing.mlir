// RUN: ! sar-translate --hls-emit-hlscpp %s 2>%t.err >%t.cpp
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err
// RUN: FileCheck %s --check-prefix=EMPTY --allow-empty --input-file=%t.cpp

// A value that reaches the signature twice -- a result that is also an
// argument -- would declare two ports with one name. The check runs before
// any C++ is written, so the failure is reported and nothing lands on the
// output stream: a caller reading stdout without checking the status must
// not receive a definition that cannot compile.
//
// The two streams are checked separately, and the diagnostic is matched by
// text. The emitter keeps an assertion for the same condition, and an
// assertion failure is also a nonzero exit that writes to stderr -- so
// checking the status alone, or folding the streams together, would pass on
// a build where the pre-emission check had been lost.

// ERR: error: HLS C++ target does not support port aliasing: a value
// ERR-SAME: reaches the signature twice
// ERR-NOT: Assertion
// EMPTY-NOT: void alias
func.func @alias(%a: memref<8xf32>) -> memref<8xf32> attributes {top_func} {
  return %a : memref<8xf32>
}
