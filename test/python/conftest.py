"""Shared pytest fixtures and backend availability markers."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "examples"))

import sar  # noqa: E402


def _backend_available(name: str) -> bool:
    backends = sar.list_backends()
    return name in backends and backends[name].is_available()


requires_cpu = pytest.mark.skipif(
    not _backend_available("cpu"),
    reason="CPU backend toolchain not available (build the project first)")

requires_hls = pytest.mark.skipif(not _backend_available("hls"),
                                  reason="HLS toolchain not available")


# Shared utilities for split-complex affine/HLS testing
def compile_split_kernel(mlir_text: str,
                         name: str,
                         tmp_path,
                         pipeline: str = "--sar-affine-to-llvm-pipeline"):
    """Compiles a module through the split-complex affine path into a
    shared library and returns the `_mlir_ciface_<name>` symbol."""
    import ctypes
    import subprocess

    from sar.compiler.toolchain import find_tool

    llvm_mlir = subprocess.run([find_tool("sar-opt"), pipeline, "-"],
                               input=mlir_text,
                               capture_output=True,
                               text=True,
                               check=True,
                               timeout=300).stdout
    llvm_ir = subprocess.run(
        [find_tool("mlir-translate"), "--mlir-to-llvmir", "-"],
        input=llvm_mlir,
        capture_output=True,
        text=True,
        check=True,
        timeout=300).stdout
    ll = tmp_path / "kernel.ll"
    ll.write_text(llvm_ir)
    so = tmp_path / "kernel.so"
    subprocess.run([
        find_tool("clang"), "-O2", "-shared", "-fPIC",
        str(ll), "-o",
        str(so), "-lm", "-Wno-override-module"
    ],
                   check=True,
                   timeout=300)
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, f"_mlir_ciface_{name}")
    fn.restype = None
    return lib, fn


def run_split(fn, inputs, out_shapes, dtype):
    """Invokes a split-complex C interface function."""
    import numpy as np

    from sar.runtime import make_descriptor
    import ctypes

    outs = [np.empty(s, dtype=dtype) for s in out_shapes]
    descriptors = [make_descriptor(a) for a in list(inputs) + outs]
    fn(*[ctypes.byref(d) for d in descriptors])
    return outs
