"""CPU backend: compiles SAR kernels to a native shared library.

"CPU" means the *host* CPU: kernels are JIT-compiled for the machine the
driver runs on (clang targets the host triple, optionally tuned with
native codegen flags). Nothing in the flow is architecture-specific;
x86-64 Linux is the tested platform, and any Linux host with an LLVM host
target (e.g. AArch64) is expected to work.

Stages:
    sar    -- verified `sar` dialect module (input)
    llvm   -- `sar-opt --sar-to-llvm-pipeline` (LLVM dialect)
    ll     -- `mlir-translate --mlir-to-llvmir` (LLVM IR)
    shared -- `clang -shared` linked against libsar_runtime

The resulting library exports `_mlir_ciface_<kernel>` and is executed
through `sar.runtime.CompiledKernel` (ctypes + memref descriptors).
"""

from __future__ import annotations

import functools
import os
import subprocess

from sar.backends.base import BaseBackend, KernelMetadata
from sar.compiler.toolchain import find_runtime_library, find_tool, run_tool
from sar.errors import ToolchainError
from sar.runtime import CompiledKernel


@functools.lru_cache(maxsize=None)
def _native_arch_flags() -> tuple:
    """Best flags for tuning to the host CPU.

    `-march=native` is well supported on x86; on other architectures clang
    historically prefers `-mcpu=native`. Probe once and fall back to
    generic codegen if neither is accepted.
    """
    clang = find_tool("clang")
    for flag in ("-march=native", "-mcpu=native"):
        probe = subprocess.run([clang, flag, "-x", "c", "-", "-fsyntax-only"],
                               input="int main(void){return 0;}",
                               capture_output=True,
                               text=True)
        if probe.returncode == 0:
            return (flag, )
    return ()


def _openmp_link_flags() -> list:
    """Locates libomp next to the clang toolchain (LLVM build tree layout)
    and returns the flags linking the OpenMP runtime the generated code's
    `__kmpc_*` calls resolve against."""
    override = os.environ.get("SAR_DSL_OMP_LIB")
    candidates = [override] if override else []
    clang = find_tool("clang")
    lib_dir = os.path.join(os.path.dirname(os.path.dirname(clang)), "lib")
    candidates.append(os.path.join(lib_dir, "libomp.so"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            directory = os.path.dirname(candidate)
            return [f"-L{directory}", "-lomp", f"-Wl,-rpath,{directory}"]
    # Fall back to the driver's default -fopenmp resolution.
    return ["-fopenmp"]


class Backend(BaseBackend):
    name = "cpu"

    @classmethod
    def is_available(cls) -> bool:
        try:
            find_tool("sar-opt")
            find_tool("mlir-translate")
            find_tool("clang")
            find_runtime_library()
            return True
        except ToolchainError:
            return False

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def add_stages(self, stages, metadata: KernelMetadata) -> None:
        stages["llvm"] = self._stage_llvm_dialect
        stages["ll"] = self._stage_llvm_ir
        stages["shared"] = self._stage_shared_library

    @staticmethod
    def _stage_llvm_dialect(module_text: str, metadata: KernelMetadata,
                            cache) -> str:
        if cache.has("kernel.llvm.mlir"):
            return cache.read_text("kernel.llvm.mlir")
        cache.write_text("kernel.sar.mlir", module_text)
        out = run_tool("sar-to-llvm",
                       [find_tool("sar-opt"), "--sar-to-llvm-pipeline", "-"],
                       input_text=module_text)
        cache.write_text("kernel.llvm.mlir", out)
        return out

    @staticmethod
    def _stage_llvm_ir(llvm_dialect: str, metadata: KernelMetadata,
                       cache) -> str:
        if cache.has("kernel.ll"):
            return cache.read_text("kernel.ll")
        out = run_tool("mlir-translate",
                       [find_tool("mlir-translate"), "--mlir-to-llvmir", "-"],
                       input_text=llvm_dialect)
        cache.write_text("kernel.ll", out)
        return out

    @staticmethod
    def _stage_shared_library(llvm_ir: str, metadata: KernelMetadata,
                              cache) -> str:
        so_path = cache.path("kernel.so")
        if cache.has("kernel.so"):
            return str(so_path)
        ll_path = cache.write_text("kernel.ll", llvm_ir)
        runtime = find_runtime_library()
        opt_level = str(metadata.options.get("opt_level", 3))
        scratch = cache.scratch_path("kernel.so")
        command = [
            find_tool("clang"), f"-O{opt_level}", "-shared", "-fPIC",
            str(ll_path), runtime, "-o",
            str(scratch), "-lm", f"-Wl,-rpath,{os.path.dirname(runtime)}",
            "-Wno-override-module"
        ]
        if metadata.options.get("native_codegen", True):
            command += _native_arch_flags()
        command += _openmp_link_flags()
        run_tool("clang-link", command)
        cache.publish(scratch, "kernel.so")
        return str(so_path)

    # ------------------------------------------------------------------ #
    # Launcher
    # ------------------------------------------------------------------ #

    def make_launcher(self, artifact: str, metadata: KernelMetadata):
        return CompiledKernel(artifact, metadata.name, metadata.arg_types,
                              metadata.result_types)
