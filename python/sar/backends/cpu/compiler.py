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

from ..base import BaseBackend, KernelMetadata, cached_stage
from ...compiler.toolchain import find_runtime_library, find_tool, run_tool
from ...errors import ToolchainError
from ...runtime import CompiledKernel


@functools.lru_cache(maxsize=None)
def _native_arch_flags() -> tuple:
    """Best flags for tuning to the host CPU.

    Clang commonly accepts `-march=native` on x86 and `-mcpu=native` on
    other targets. Probe once and fall back to generic codegen if neither is
    accepted.
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
            # -Xlinker rather than -Wl: the -Wl spelling splits its
            # argument on commas, so a build path containing one would
            # reach the linker in pieces.
            return [
                f"-L{directory}", "-lomp", "-Xlinker", f"-rpath={directory}"
            ]
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

    #: The whole option schema of this backend. Anything else is a typo
    #: or an option meant for another backend, and silently ignoring it
    #: would let the user believe it took effect.
    OPTIONS = ("opt_level", "native_codegen")

    def add_stages(self, stages, metadata: KernelMetadata) -> None:
        unknown = sorted(set(metadata.options) - set(self.OPTIONS))
        if unknown:
            names = ", ".join(map(repr, unknown))
            raise ToolchainError(
                f"unknown cpu backend option(s) {names}; valid options "
                f"are: {', '.join(self.OPTIONS)}")
        opt_level = metadata.options.get("opt_level", 3)
        native_codegen = metadata.options.get("native_codegen", True)
        if isinstance(opt_level, bool) or not isinstance(
                opt_level, int) or (not 0 <= opt_level <= 3):
            raise ToolchainError("cpu option 'opt_level' must be an integer "
                                 "in [0, 3]")
        if not isinstance(native_codegen, bool):
            raise ToolchainError(
                "cpu option 'native_codegen' must be true or false")
        metadata.options.clear()
        metadata.options.update({
            "opt_level": opt_level,
            "native_codegen": native_codegen
        })
        stages["llvm"] = self._stage_llvm_dialect
        stages["ll"] = self._stage_llvm_ir
        stages["shared"] = self._stage_shared_library

    @staticmethod
    def _stage_llvm_dialect(module_text: str, metadata: KernelMetadata,
                            cache) -> str:

        def build() -> str:
            cache.write_text("kernel.sar.mlir", module_text)
            return run_tool(
                "sar-to-llvm",
                [find_tool("sar-opt"), "--sar-to-llvm-pipeline", "-"],
                input_text=module_text)

        return cached_stage(cache, "kernel.llvm.mlir", build)

    @staticmethod
    def _stage_llvm_ir(llvm_dialect: str, metadata: KernelMetadata,
                       cache) -> str:

        def build() -> str:
            return run_tool(
                "mlir-translate",
                [find_tool("mlir-translate"), "--mlir-to-llvmir", "-"],
                input_text=llvm_dialect)

        return cached_stage(cache, "kernel.ll", build)

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
            str(scratch), "-lm", "-Xlinker",
            f"-rpath={os.path.dirname(runtime)}", "-Wno-override-module"
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
