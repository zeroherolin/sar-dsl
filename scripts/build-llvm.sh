#!/usr/bin/env bash
# Builds the in-tree LLVM/MLIR/Clang toolchain used by SAR-DSL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_DIR="$ROOT/externals/llvm-project"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -f "$LLVM_DIR/llvm/CMakeLists.txt" ]]; then
  echo "llvm-project submodule missing; run: git submodule update --init externals/llvm-project"
  exit 1
fi

cmake -G Ninja -S "$LLVM_DIR/llvm" -B "$LLVM_DIR/build" \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF

ninja -C "$LLVM_DIR/build" -j "$JOBS"
echo "LLVM/MLIR/Clang toolchain ready: $LLVM_DIR/build/bin"
