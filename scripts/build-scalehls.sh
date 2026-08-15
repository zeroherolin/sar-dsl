#!/usr/bin/env bash
# Builds the ScaleHLS-HIDA toolchain (scalehls-opt / scalehls-translate)
# against the project's LLVM tree (externals/llvm-project).
#
# The submodule tracks the `dev` branch of our fork
# (github.com/zeroherolin/ScaleHLS-HIDA), which carries the LLVM 22 port
# and bug fixes as regular commits -- no patches, no second LLVM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALEHLS_DIR="$ROOT/externals/ScaleHLS-HIDA"
LLVM_BUILD="$ROOT/externals/llvm-project/build"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -x "$LLVM_BUILD/bin/clang" ]]; then
  echo "clang not found at $LLVM_BUILD/bin; run scripts/build-llvm.sh first"
  exit 1
fi

if [[ ! -f "$SCALEHLS_DIR/CMakeLists.txt" ]]; then
  echo "ScaleHLS-HIDA submodule missing; run:" \
       "git submodule update --init externals/ScaleHLS-HIDA"
  exit 1
fi

cmake -G Ninja \
  -S "$SCALEHLS_DIR" \
  -B "$SCALEHLS_DIR/build" \
  -DMLIR_DIR="$LLVM_BUILD/lib/cmake/mlir" \
  -DLLVM_DIR="$LLVM_BUILD/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER="$LLVM_BUILD/bin/clang" \
  -DCMAKE_CXX_COMPILER="$LLVM_BUILD/bin/clang++" \
  -DCMAKE_BUILD_TYPE=Release

ninja -C "$SCALEHLS_DIR/build" -j "$JOBS" scalehls-opt scalehls-translate

echo "ScaleHLS toolchain ready: $SCALEHLS_DIR/build/bin"
