#!/usr/bin/env bash
# Builds the ScaleHLS-HIDA toolchain (scalehls-opt / scalehls-translate).
#
# ScaleHLS pins an older LLVM through its polygeist submodule, so it gets its
# own LLVM build. The freshly built clang from externals/llvm-project is used
# as the host compiler (old LLVM sources predate current GCC strictness).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCALEHLS_DIR="$ROOT/externals/ScaleHLS-HIDA"
CLANG="$ROOT/externals/llvm-project/build/bin/clang"
CLANGXX="$ROOT/externals/llvm-project/build/bin/clang++"
JOBS="${JOBS:-$(nproc)}"

if [[ ! -x "$CLANG" ]]; then
  echo "clang not found at $CLANG; run scripts/build-llvm.sh first"
  exit 1
fi

if [[ ! -f "$SCALEHLS_DIR/polygeist/llvm-project/llvm/CMakeLists.txt" ]]; then
  echo "Fetching polygeist submodule (contains ScaleHLS' pinned LLVM) ..."
  git -C "$SCALEHLS_DIR" submodule update --init --recursive --depth 1 polygeist \
    || git -C "$SCALEHLS_DIR" submodule update --init --recursive polygeist
fi

# Upstream fixes applied idempotently: missing MLIRAnalysis/Presburger link
# deps on MLIRHLS (breaks single-pass GNU ld) and arith.extf missing from
# the HLS emitter's dispatch table.
PATCH="$ROOT/scripts/patches/scalehls-hida-fixes.patch"
if ! git -C "$SCALEHLS_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
  git -C "$SCALEHLS_DIR" apply "$PATCH"
  echo "Applied scalehls-hida-fixes.patch"
fi

# 1. ScaleHLS' pinned LLVM.
cmake -G Ninja \
  -S "$SCALEHLS_DIR/polygeist/llvm-project/llvm" \
  -B "$SCALEHLS_DIR/polygeist/llvm-project/build" \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER="$CLANG" \
  -DCMAKE_CXX_COMPILER="$CLANGXX" \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DLLVM_ENABLE_LIBXML2=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_TESTS=OFF
ninja -C "$SCALEHLS_DIR/polygeist/llvm-project/build" -j "$JOBS"

# 2. ScaleHLS itself (only the tools SAR-DSL needs).
cmake -G Ninja -S "$SCALEHLS_DIR" -B "$SCALEHLS_DIR/build" \
  -DMLIR_DIR="$SCALEHLS_DIR/polygeist/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$SCALEHLS_DIR/polygeist/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CLANG" \
  -DCMAKE_CXX_COMPILER="$CLANGXX"
ninja -C "$SCALEHLS_DIR/build" -j "$JOBS" scalehls-opt scalehls-translate
echo "ScaleHLS toolchain ready: $SCALEHLS_DIR/build/bin"
