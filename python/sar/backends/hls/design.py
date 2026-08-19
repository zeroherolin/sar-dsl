"""User-facing handle and artifact writers for emitted HLS designs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ...errors import LaunchError

__all__ = ["HLSDesign"]


class HLSDesign:
    """Handle to an emitted HLS C++ design."""

    def __init__(self, cpp_path: str, name: str, metadata=None, config=None):
        self.cpp_path = str(cpp_path)
        self.name = name
        self.config = config
        self._metadata = metadata
        try:
            self._source = Path(self.cpp_path).read_text()
        except OSError as exc:
            raise LaunchError(
                f"HLS artifact for {self.name!r} is unavailable at "
                f"{self.cpp_path}; recompile the kernel") from exc

    def source(self) -> str:
        return self._source

    def _write_manifest(self, output_dir: Path) -> Path:
        config = self.config.as_dict() if self.config is not None else {}
        provenance = (self.config.provenance
                      if self.config is not None else {})
        path = output_dir / "design_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version":
                    1,
                    "top":
                    self.name,
                    "source_sha256":
                    hashlib.sha256(self.source().encode()).hexdigest(),
                    "config":
                    config,
                    "config_provenance":
                    provenance,
                },
                indent=2,
                sort_keys=True) + "\n")
        return path

    def write_testbench(self,
                        inputs,
                        expected,
                        output_dir=None,
                        rtol: float = 1e-4,
                        atol: float = 1e-5) -> Path:
        """Writes a self-contained C-simulation package."""
        from . import compiler

        tolerances = {}
        for name, value in (("rtol", rtol), ("atol", atol)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise LaunchError(
                    f"{name} must be a finite non-negative number") from None
            if not math.isfinite(value) or value < 0:
                raise LaunchError(
                    f"{name} must be a finite non-negative number")
            tolerances[name] = value

        meta = self._metadata
        if self.config is not None and self.config.interface == "axi":
            raise LaunchError(
                "testbench generation needs interface='ap_memory'; AXI ports "
                "for compiler-managed scratch buffers have no golden input "
                "data")
        if self.config is not None and self.config.interface == "stream":
            raise LaunchError(
                "testbench generation needs interface='ap_memory'; stream "
                "ports require a FIFO-driving system harness")
        for tensor_type in list(meta.arg_types) + list(meta.result_types):
            if tensor_type.dtype.is_int:
                raise LaunchError(
                    "testbench generation does not support integer "
                    "kernel arguments or results")

        in_ports = compiler._split_planes("in", meta.arg_types)
        out_ports = compiler._split_planes("out", meta.result_types)
        arrays = (list(compiler._plane_data(inputs, meta.arg_types)) +
                  list(compiler._plane_data(expected, meta.result_types)))

        out = Path(
            output_dir if output_dir is not None else Path("hls_project") /
            self.name)
        data_dir = out / f"{self.name}_tb_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        for (port, _, _), values in zip(in_ports + out_ports, arrays):
            np.savetxt(data_dir / f"{port}.dat",
                       values.reshape(-1),
                       fmt="%.17g")

        (out / f"{self.name}.cpp").write_text(self.source())
        testbench = out / f"{self.name}_tb.cpp"
        testbench.write_text(
            compiler._testbench_source(self.name, in_ports, out_ports,
                                       tolerances["rtol"], tolerances["atol"]))
        part, clock = compiler._part_and_clock(self.config)
        (out / f"{self.name}_csim.tcl").write_text(
            compiler._csim_script(self.name, part, clock))
        (out / f"{self.name}_csynth.tcl").write_text(
            compiler._csynth_script(self.name, part, clock))
        compiler._write_header_stubs(out / "stubs")
        self._write_manifest(out)
        return testbench

    def write_synthesis_script(self, output_dir=None) -> Path:
        """Writes the design and its Vitis HLS synthesis script."""
        from . import compiler

        out = Path(
            output_dir if output_dir is not None else Path("hls_project") /
            self.name)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{self.name}.cpp").write_text(self.source())
        self._write_manifest(out)
        part, clock = compiler._part_and_clock(self.config)
        script = out / f"{self.name}_csynth.tcl"
        script.write_text(compiler._csynth_script(self.name, part, clock))
        return script

    def __call__(self, *args, **kwargs):
        raise LaunchError(
            "HLS designs are emitted as C++ for Vitis HLS synthesis and "
            f"cannot be executed directly; see {self.cpp_path}")

    def __repr__(self) -> str:
        return f"HLSDesign({self.name} @ {self.cpp_path})"
