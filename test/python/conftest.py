"""Shared pytest fixtures and backend availability markers."""

import os
import shutil
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


requires_cpu = pytest.mark.requires_cpu
requires_hls = pytest.mark.requires_hls


def _vitis_available() -> bool:
    """Whether a Vitis HLS installation can be driven from this shell.

    RTL co-simulation is opt-in because it takes minutes per design. C-sim
    through Vitis HLS is selected automatically by `run_hls_csim` whenever
    the executable exists; this gate applies only to RTL co-simulation.
    """
    if os.environ.get("SAR_DSL_TEST_VITIS", "") not in ("1", "true", "yes"):
        return False
    return shutil.which("vitis_hls") is not None


requires_vitis = pytest.mark.requires_vitis


def pytest_collection_modifyitems(items):
    """Skips capability-marked tests only when their dependency is absent."""
    unavailable = {
        "requires_cpu":
        (not _backend_available("cpu"),
         "CPU backend toolchain not available (build the project first)"),
        "requires_hls":
        (not _backend_available("hls"), "HLS toolchain not available"),
        "requires_vitis":
        (not _vitis_available(),
         "Vitis HLS run not requested (set SAR_DSL_TEST_VITIS=1)"),
    }
    for item in items:
        for marker, (missing, reason) in unavailable.items():
            if missing and item.get_closest_marker(marker) is not None:
                item.add_marker(pytest.mark.skip(reason=reason))
                break


def run_hls_csim(package,
                 top: str,
                 timeout: float = 1800.0,
                 expect_success: bool = True):
    """Runs C-sim through Vitis HLS, or portable C++ without Vitis."""
    import subprocess

    package = Path(package)
    vitis = shutil.which("vitis_hls")
    if vitis is not None:
        command = [vitis, f"{top}_hls_csim.tcl"]
        mode = "hls_csim"
    else:
        command = ["sh", f"{top}_portable_cpp_sim.sh"]
        mode = "portable_cpp_sim"
    result = subprocess.run(command,
                            cwd=package,
                            capture_output=True,
                            text=True,
                            timeout=timeout)
    output = result.stdout + result.stderr
    if expect_success:
        assert result.returncode == 0, (
            f"{mode} simulation failed for {top}:\n{output[-4000:]}")
        if mode == "hls_csim":
            assert "CSim done with 0 errors" in output, output[-4000:]
        else:
            assert "PASS" in output, output[-4000:]
    else:
        assert result.returncode != 0, output[-4000:]
    return mode, result


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
