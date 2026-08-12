#!/usr/bin/env bash
# Links the third_party backends into the sar Python package (development
# mode; `setup.py` copies them for installed packages) and adds the package
# to the current interpreter via a .pth-style hint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKENDS_DIR="$ROOT/python/sar/backends"

for vendor_dir in "$ROOT"/third_party/*/backend; do
  vendor="$(basename "$(dirname "$vendor_dir")")"
  link="$BACKENDS_DIR/$vendor"
  if [[ ! -e "$link" ]]; then
    ln -s "$vendor_dir" "$link"
    echo "linked backend: $vendor"
  fi
done

echo
echo "Add SAR-DSL to your environment with:"
echo "  export PYTHONPATH=$ROOT/python:\$PYTHONPATH"
