#!/usr/bin/bash

set -e

# ---------- Set log ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
die()  { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
header() { echo -e "\n${YELLOW}========== $* ==========${NC}"; }

# ---------- Set dir ----------
ROOT_DIR=$(pwd)
BUILD_DIR="$ROOT_DIR/build"
LLVM_DIR="$ROOT_DIR/externals/llvm-project"
SCALEHLS_DIR="$ROOT_DIR/externals/ScaleHLS-HIDA"

# ---------- Set env ----------
log "Setting env ..."
export PATH="$BUILD_DIR/bin:$PATH"
export PATH="$BUILD_DIR/test:$PATH"
export PATH="$LLVM_DIR/build/bin:$PATH"
export PATH="$SCALEHLS_DIR/build/bin:$PATH"
export PYTHONPATH="$LLVM_DIR/build/tools/mlir/python_packages/mlir_core:$PYTHONPATH"
export PYTHONPATH="$BUILD_DIR/python/python_packages:$PYTHONPATH"

# ---------- Run tests ----------
run_tests() {
    log "Starting tests ..."

    mkdir -p "$ROOT_DIR/test/MLIR"
    mkdir -p "$ROOT_DIR/test/emitHLS"

    # 1. Test generate mlir
    header "Test generate mlir"

    log "Generating test_gen_elem.mlir ..."
    if command -v test-gen-elem >/dev/null 2>&1; then
        test-gen-elem -o "$ROOT_DIR/test/MLIR/test_gen_elem.mlir"
    else
        die "test-gen-elem not found in PATH"
    fi
    ok "MLIR/test_gen_elem.mlir generated."

    log "Generating test_gen_fft.mlir ..."
    if command -v test-gen-fft >/dev/null 2>&1; then
        test-gen-fft  -o "$ROOT_DIR/test/MLIR/test_gen_fft.mlir"
    else
        die "test-gen-fft not found in PATH"
    fi
    ok "MLIR/test_gen_fft.mlir generated."

    log "Testing test_shape_mismatch ..."
    if command -v test-shape-mismatch >/dev/null 2>&1; then
        test-shape-mismatch
    else
        die "test-shape-mismatch not found in PATH"
    fi
    ok "test_shape_mismatch passed."

    # 2. Lowering to Linalg
    header "Lowering to Linalg"

    log "Lowering SAR to Linalg ..."
    if command -v sar-opt >/dev/null 2>&1; then
        sar-opt "$ROOT_DIR/test/MLIR/test_gen_elem.mlir" \
            --sar-to-linalg-pipeline \
            > "$ROOT_DIR/test/MLIR/test_gen_elem_output.mlir"
    else
        die "sar-opt not found in PATH"
    fi
    ok "SAR to Linalg done."

    # 3. Test LLVM output
    header "Test LLVM output"

    log "Lowering to LLVM ..."
    if command -v mlir-opt >/dev/null 2>&1 && command -v mlir-translate >/dev/null 2>&1; then
        mlir-opt "$ROOT_DIR/test/MLIR/test_gen_elem_output.mlir" \
            --one-shot-bufferize="bufferize-function-boundaries" \
            --convert-linalg-to-loops \
            --convert-scf-to-cf \
            --finalize-memref-to-llvm \
            --convert-arith-to-llvm \
            --convert-func-to-llvm \
            --convert-cf-to-llvm \
            --reconcile-unrealized-casts \
            | mlir-translate --mlir-to-llvmir > "$ROOT_DIR/test/output.ll"
    else
        die "mlir-opt/mlir-translate not found in PATH"
    fi
    ok "Lowering to LLVM done."

    log "Compiling to executable ..."
    clang -c "$ROOT_DIR/test/output.ll" -o "$ROOT_DIR/test/output.o" -Wno-override-module
    clang -c "$ROOT_DIR/test/ir_test.c" -o "$ROOT_DIR/test/ir_test.o"
    clang "$ROOT_DIR/test/ir_test.o" "$ROOT_DIR/test/output.o" -o "$ROOT_DIR/test/ir_test"
    ok "Compiling to executable done."

    log "Running executable ..."
    "$ROOT_DIR/test/ir_test"
    ok "Executable run done."

    # 4. Test ScaleHLS-HIDA
    header "Test ScaleHLS-HIDA"

    if command -v scalehls-opt >/dev/null 2>&1 && command -v scalehls-translate >/dev/null 2>&1; then
        log "Running ScaleHLS-HIDA affine_matmul ..."
        scalehls-opt "$ROOT_DIR/test/test-scalehls-hida/affine_matmul.mlir" \
            -hida-pytorch-pipeline="top-func=affine_matmul" \
            | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives \
            > "$ROOT_DIR/test/emitHLS/hls_affine_matmul.cpp"
        ok "ScaleHLS-HIDA affine_matmul done."
    else
        die "scalehls-opt/scalehls-translate not found in PATH"
    fi

    # 5. Test emit SAR to HLS
    header "Test emit SAR to HLS"

    if command -v scalehls-opt >/dev/null 2>&1 && command -v scalehls-translate >/dev/null 2>&1; then
        log "Emitting SAR to HLS ..."
        scalehls-opt "$ROOT_DIR/test/MLIR/test_gen_elem_output.mlir" \
            -hida-pytorch-pipeline="top-func=forward loop-tile-size=8 loop-unroll-factor=4" \
            | scalehls-translate -scalehls-emit-hlscpp -emit-vitis-directives \
            > "$ROOT_DIR/test/emitHLS/hls_output.cpp"
        ok "Emit SAR to HLS done."
    else
        die "scalehls-opt/scalehls-translate not found in PATH"
    fi

    # 6. Test python frontend
    header "Test python frontend"

    log "Running Python frontend tests ..."
    python3 "$ROOT_DIR/test/test_debug.py"
    python3 "$ROOT_DIR/test/test_elem.py"
    python3 "$ROOT_DIR/test/test_fft.py"
    python3 "$ROOT_DIR/test/test_compr.py"
    python3 "$ROOT_DIR/test/sar_domain/test_filter.py"
    ok "Python frontend tests done."

    header "All tests done."
}

run_tests
