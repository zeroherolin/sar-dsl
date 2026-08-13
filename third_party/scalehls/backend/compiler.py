"""ScaleHLS-HIDA backend: emits Vitis HLS C++ for FPGA synthesis.

Two lowering flows, selected automatically (or via the `flow` option):

* ``linalg`` -- float-only element-wise kernels go through
  `sar-to-linalg-pipeline` and HIDA's PyTorch entry point
  (`-hida-pytorch-pipeline`), which applies dataflow decomposition.
* ``affine`` -- kernels using complex arithmetic, FFTs or interpolation
  are decomplexified (complex tensors become re/im float planes), FFTs
  become Stockham loop nests and interpolation becomes windowed-sinc
  gather loops (`sar-to-affine-pipeline`); the result enters HIDA's C++
  entry point (`-hida-cpp-pipeline`).

The compilation result is an `HLSDesign` handle pointing at the emitted
C++ source, not an executable kernel. In the affine flow the generated
top function takes each complex tensor as two adjacent float arrays
(re, im), inputs first, then one output array per result plane.

Every SAR operation has an HLS lowering, so complete imaging chains
(omega-K, range-Doppler, chirp scaling) emit as single designs.

Options:
    flow               -- "auto" (default), "linalg" or "affine"
    top_func           -- HIDA top function (defaults to the kernel name)
    loop_tile_size     -- HIDA loop tiling factor, linalg flow (default 8)
    loop_unroll_factor -- HIDA unroll factor, linalg flow (default 4)
"""

from __future__ import annotations

import re
from pathlib import Path

from sar.backends.base import BaseBackend, KernelMetadata
from sar.compiler.toolchain import find_tool, run_tool
from sar.errors import ToolchainError

_AFFINE_TRIGGER_OPS = ("sar.fft", "sar.ifft", "sar.fftshift",
                       "sar.stolt_interp", "sar.interp1d")


class HLSDesign:
    """Handle to an emitted HLS C++ design."""

    def __init__(self, cpp_path: str, name: str, flow: str):
        self.cpp_path = str(cpp_path)
        self.name = name
        self.flow = flow

    def source(self) -> str:
        return Path(self.cpp_path).read_text()

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "HLS designs are emitted as C++ for Vitis HLS synthesis and "
            f"cannot be executed directly; see {self.cpp_path}")

    def __repr__(self) -> str:
        return f"HLSDesign({self.name} [{self.flow}] @ {self.cpp_path})"


def _uses_op(module_text: str, op: str) -> bool:
    return re.search(rf'"{re.escape(op)}"', module_text) is not None


def _select_flow(module_text: str, options: dict) -> str:
    flow = options.get("flow", "auto")
    if flow != "auto":
        return flow
    if any(_uses_op(module_text, op) for op in _AFFINE_TRIGGER_OPS):
        return "affine"
    if "complex<" in module_text:
        return "affine"
    return "linalg"


class Backend(BaseBackend):
    name = "scalehls"

    @classmethod
    def is_available(cls) -> bool:
        try:
            find_tool("sar-opt")
            find_tool("scalehls-opt")
            find_tool("scalehls-translate")
            return True
        except ToolchainError:
            return False

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def add_stages(self, stages, metadata: KernelMetadata) -> None:
        stages["select-flow"] = self._stage_select_flow
        stages["lower"] = self._stage_lower
        stages["hls"] = self._stage_hls_cpp

    @staticmethod
    def _stage_select_flow(module_text: str, metadata: KernelMetadata,
                           cache) -> str:
        metadata.extra["hls_flow"] = _select_flow(module_text,
                                                  metadata.options)
        return module_text

    @staticmethod
    def _stage_lower(module_text: str, metadata: KernelMetadata,
                     cache) -> str:
        flow = metadata.extra["hls_flow"]
        if cache.has(f"kernel.{flow}.mlir"):
            return cache.read_text(f"kernel.{flow}.mlir")
        cache.write_text("kernel.sar.mlir", module_text)
        pipeline = ("--sar-to-affine-pipeline" if flow == "affine"
                    else "--sar-to-linalg-pipeline")
        out = run_tool(
            "sar-lower",
            [find_tool("sar-opt"), pipeline, "-"],
            input_text=module_text)
        cache.write_text(f"kernel.{flow}.mlir", out)
        return out

    @staticmethod
    def _stage_hls_cpp(lowered: str, metadata: KernelMetadata,
                       cache) -> str:
        cpp = cache.path("kernel.hls.cpp")
        if cache.has("kernel.hls.cpp"):
            return str(cpp)

        options = metadata.options
        flow = metadata.extra["hls_flow"]
        top_func = options.get("top_func", metadata.name)

        if flow == "affine":
            hida_arg = f"-hida-cpp-pipeline=top-func={top_func}"
        else:
            tile = options.get("loop_tile_size", 8)
            unroll = options.get("loop_unroll_factor", 4)
            hida_arg = (f"-hida-pytorch-pipeline=top-func={top_func} "
                        f"loop-tile-size={tile} loop-unroll-factor={unroll}")

        hida = run_tool(
            "scalehls-hida",
            [find_tool("scalehls-opt"), hida_arg, "-"],
            input_text=lowered)
        cache.write_text("kernel.hida.mlir", hida)

        cpp_text = run_tool(
            "scalehls-emit",
            [find_tool("scalehls-translate"), "-scalehls-emit-hlscpp",
             "-emit-vitis-directives", "-"],
            input_text=hida)
        cache.write_text("kernel.hls.cpp", cpp_text)
        return str(cpp)

    # ------------------------------------------------------------------ #
    # Launcher
    # ------------------------------------------------------------------ #

    def make_launcher(self, artifact: str, metadata: KernelMetadata):
        return HLSDesign(artifact, metadata.name,
                         metadata.extra.get("hls_flow", "linalg"))
