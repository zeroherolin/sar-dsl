#!/usr/bin/env bash
# Tests for scripts/fetch-llvm-release.sh. No download is reached: unsafe
# roots must be rejected while the target path is still untouched, and the
# install cases run from pre-seeded fake archives in the cache.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
script="$repo_root/scripts/fetch-llvm-release.sh"
scratch="$(mktemp -d)"
trap 'rm -rf -- "$scratch"' EXIT

expect_refusal() {
  local root="$1"
  if LLVM_RELEASE_ROOT="$root" LLVM_RELEASE_CACHE="$scratch/cache" \
      bash "$script" >"$scratch/stdout" 2>"$scratch/stderr"; then
    echo "expected LLVM_RELEASE_ROOT=$root to be refused" >&2
    exit 1
  fi
}

# Seeds the cache with a minimal fake release archive for "$1", tagging
# every file with "$2" so installed trees are distinguishable. Prints the
# archive's sha256.
make_release_archive() {
  local version="$1" tag="$2"
  local tree="$scratch/tree-$version"
  mkdir -p "$tree/pkg/bin" "$tree/pkg/lib/cmake/llvm" \
    "$tree/pkg/lib/cmake/mlir"
  local tool
  for tool in clang mlir-opt mlir-tblgen mlir-translate; do
    printf '%s\n' "$tag" >"$tree/pkg/bin/$tool"
  done
  printf '%s\n' "$tag" >"$tree/pkg/lib/cmake/llvm/LLVMConfig.cmake"
  printf '%s\n' "$tag" >"$tree/pkg/lib/cmake/mlir/MLIRConfig.cmake"
  printf '%s\n' "$tag" >"$tree/pkg/lib/libomp.so"
  mkdir -p "$scratch/cache"
  tar --create --xz --file "$scratch/cache/LLVM-$version-Linux-X64.tar.xz" \
    -C "$tree" pkg
  sha256sum "$scratch/cache/LLVM-$version-Linux-X64.tar.xz" | awk '{print $1}'
}

install_release() {
  local root="$1" version="$2" sha="$3"
  LLVM_RELEASE_ROOT="$root" LLVM_RELEASE_CACHE="$scratch/cache" \
    LLVM_RELEASE_VERSION="$version" LLVM_RELEASE_SHA256="$sha" \
    LLVM_RELEASE_URL="file://$scratch/offline" \
    bash "$script" >"$scratch/stdout" 2>"$scratch/stderr"
}

expect_refusal "."
expect_refusal "/"
expect_refusal "///"

existing="$scratch/existing"
mkdir -p "$existing"
printf '%s\n' "keep" >"$existing/user-data"
expect_refusal "$existing"
test "$(<"$existing/user-data")" = "keep"

# A trailing slash on LLVM_RELEASE_ROOT once relocated the derived
# install/backup paths inside the root, corrupting it. Both the fresh
# install and the upgrade over an existing tree must succeed and leave
# the marker in place (regression).
sha_one="$(make_release_archive fake1 one)"
sha_two="$(make_release_archive fake2 two)"
root="$scratch/release-root"

install_release "$root/" fake1 "$sha_one"
test "$(<"$root/.sar-dsl-llvm-release")" = "fake1:$sha_one"
test "$(<"$root/bin/clang")" = "one"

install_release "$root/" fake2 "$sha_two"
test "$(<"$root/.sar-dsl-llvm-release")" = "fake2:$sha_two"
test "$(<"$root/bin/clang")" = "two"
grep -q "LLVM_RELEASE_ROOT=$root\$" "$scratch/stdout"

echo "ok"
