#!/usr/bin/env bash
# Download and unpack the pinned LLVM/MLIR release used by CI.
set -euo pipefail

VERSION="${LLVM_RELEASE_VERSION:-22.1.8}"
ROOT="${LLVM_RELEASE_ROOT:-${RUNNER_TEMP:-/tmp}/sar-dsl-llvm-${VERSION}}"
CACHE_DIR="${LLVM_RELEASE_CACHE:-${HOME}/.cache/sar-dsl}"
CACHE_DIR="${CACHE_DIR/#\~/${HOME}}"
ARCHIVE="${CACHE_DIR}/LLVM-${VERSION}-Linux-X64.tar.xz"
URL="${LLVM_RELEASE_URL:-https://github.com/llvm/llvm-project/releases/download/llvmorg-${VERSION}/LLVM-${VERSION}-Linux-X64.tar.xz}"
EXPECTED_SHA256="${LLVM_RELEASE_SHA256:-df0e1ecf16caf3489a272a5eea4eec9b0d82878f6477fa309504f918a0006384}"

if [[ -z "$ROOT" || "$ROOT" == "/" || "$ROOT" == "." ]]; then
  echo "refusing unsafe LLVM_RELEASE_ROOT=$ROOT" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
if [[ ! -f "$ARCHIVE" ]] ||
   [[ "$(sha256sum "$ARCHIVE" | awk '{print $1}')" != "$EXPECTED_SHA256" ]]; then
  rm -f -- "$ARCHIVE"
  curl --fail --location --retry 3 --retry-delay 2 --silent --show-error \
       --output "$ARCHIVE" "$URL"
fi

echo "$EXPECTED_SHA256  $ARCHIVE" | sha256sum --check --status

marker="$ROOT/.sar-dsl-llvm-release"
if [[ ! -f "$marker" ]] ||
   [[ "$(<"$marker")" != "$VERSION:$EXPECTED_SHA256" ]]; then
  rm -rf -- "$ROOT"
  mkdir -p "$ROOT"
  tar --extract --xz --file "$ARCHIVE" --strip-components=1 --directory "$ROOT"
  printf '%s\n' "$VERSION:$EXPECTED_SHA256" > "$marker"
fi

for required in \
  "$ROOT/bin/clang" \
  "$ROOT/bin/mlir-opt" \
  "$ROOT/bin/mlir-tblgen" \
  "$ROOT/bin/mlir-translate" \
  "$ROOT/lib/cmake/llvm/LLVMConfig.cmake" \
  "$ROOT/lib/cmake/mlir/MLIRConfig.cmake"; do
  if [[ ! -e "$required" ]]; then
    echo "LLVM release is missing required path: $required" >&2
    exit 1
  fi
done

omp=$(find "$ROOT/lib" -type f -name 'libomp.so' -print -quit)
if [[ -z "$omp" ]]; then
  echo "LLVM release is missing libomp.so" >&2
  exit 1
fi

echo "LLVM release ready: $ROOT"
echo "LLVM_RELEASE_ROOT=$ROOT"
echo "LLVM_RELEASE_OMP=$omp"
