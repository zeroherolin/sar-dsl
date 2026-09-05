#!/usr/bin/env bash
# Builds the in-tree LLVM/MLIR/Clang toolchain used by SAR-DSL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLVM_DIR="$ROOT/externals/llvm-project"
JOBS="${JOBS:-16}"
LLVM_PROJECTS="${LLVM_PROJECTS:-mlir;clang;clang-tools-extra}"
LLVM_INCLUDE_TESTS="${LLVM_INCLUDE_TESTS:-ON}"
LLVM_BUILD_TARGETS="${LLVM_BUILD_TARGETS:-all}"

if [[ ! -f "$LLVM_DIR/llvm/CMakeLists.txt" ]]; then
  echo "llvm-project submodule missing; run: git submodule update --init externals/llvm-project"
  exit 1
fi

# clang-tools-extra provides clangd for local editor integration; it is not
# needed to build or test SAR-DSL, so override LLVM_PROJECTS to omit it for a
# leaner build. The openmp runtime remains required by the CPU backend.
cmake_args=(
  -G Ninja
  -S "$LLVM_DIR/llvm"
  -B "$LLVM_DIR/build"
  "-DLLVM_ENABLE_PROJECTS=$LLVM_PROJECTS"
  -DLLVM_ENABLE_RUNTIMES=openmp
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF
  -DLLVM_TARGETS_TO_BUILD=host
  -DCMAKE_BUILD_TYPE=Release
  -DLLVM_ENABLE_ASSERTIONS=ON
  "-DLLVM_INCLUDE_TESTS=$LLVM_INCLUDE_TESTS"
  "-DCLANG_INCLUDE_TESTS=$LLVM_INCLUDE_TESTS"
  "-DMLIR_INCLUDE_TESTS=$LLVM_INCLUDE_TESTS"
  -DLLVM_INSTALL_UTILS=ON
  -DLLVM_ENABLE_ZSTD=OFF
  -DLLVM_ENABLE_LIBXML2=OFF
  -DLLVM_ENABLE_TERMINFO=OFF
  -DLLVM_ENABLE_BINDINGS=OFF
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_DOCS=OFF
)

if command -v ccache >/dev/null 2>&1; then
  cmake_args+=(
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  )
fi

cmake "${cmake_args[@]}"

read -r -a build_targets <<<"$LLVM_BUILD_TARGETS"
echo "Building LLVM targets '${build_targets[*]}' with $JOBS jobs"
ninja -C "$LLVM_DIR/build" -j "$JOBS" "${build_targets[@]}"

# The runtimes build leaves libomp.so in its own subtree; stage it next to
# the toolchain libraries where the CPU backend expects it.
cp -f "$LLVM_DIR/build/runtimes/runtimes-bins/openmp/runtime/src/libomp.so" \
  "$LLVM_DIR/build/lib/"

echo "LLVM/MLIR/Clang toolchain ready: $LLVM_DIR/build/bin"
