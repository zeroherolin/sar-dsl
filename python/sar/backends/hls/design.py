"""User-facing handle and artifact writers for emitted HLS designs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path

import numpy as np

from ..._version import __version__
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

    def _interface_schema(self) -> list:
        match = re.search(rf"^void {re.escape(self.name)}\(\n(.*?)\n\) \{{",
                          self.source(), re.S | re.M)
        if not match:
            return []
        ports = []
        type_pattern = (
            r"(?P<type>float|double|"
            r"hls::vector<(?P<scalar>float|double),\s*(?P<lanes>\d+)>)")
        for declaration in match.group(1).split(",\n"):
            parsed = re.fullmatch(
                rf"\s*(?:const\s+)?{type_pattern}\s+(?P<name>\w+)"
                rf"(?P<dims>(?:\[\d+\])+)\s*", declaration)
            if not parsed:
                continue
            scalar = parsed.group("scalar") or parsed.group("type")
            lanes = int(parsed.group("lanes") or 1)
            shape = [
                int(extent)
                for extent in re.findall(r"\[(\d+)\]", parsed.group("dims"))
            ]
            physical_elements = math.prod(shape)
            ports.append({
                "name":
                parsed.group("name"),
                "c_type":
                parsed.group("type"),
                "scalar_type":
                scalar,
                "vector_lanes":
                lanes,
                "physical_shape":
                shape,
                "logical_elements":
                physical_elements * lanes,
                "data_bits": (32 if scalar == "float" else 64) * lanes,
            })
        return ports

    def _write_manifest(self, output_dir: Path) -> Path:
        config = self.config.as_dict() if self.config is not None else {}
        provenance = (self.config.provenance
                      if self.config is not None else {})
        metadata = self._metadata

        def tensor_schema(tensor_type):
            return {
                "shape": list(tensor_type.shape),
                "dtype": tensor_type.dtype.name,
                "mlir": tensor_type.mlir,
            }

        path = output_dir / "design_manifest.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version":
                    2,
                    "top":
                    self.name,
                    "source_sha256":
                    hashlib.sha256(self.source().encode()).hexdigest(),
                    "generator": {
                        "name": "sar-dsl",
                        "version": __version__,
                        "backend": "hls",
                    },
                    "kernel": {
                        "name":
                        metadata.name,
                        "parameters":
                        list(metadata.param_names or []),
                        "arguments":
                        [tensor_schema(t) for t in metadata.arg_types],
                        "results":
                        [tensor_schema(t) for t in metadata.result_types],
                    },
                    "requested_options":
                    dict(metadata.extra.get("hls_requested_options", {})),
                    "interfaces":
                    self._interface_schema(),
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
        physical_ports = None
        if self.config is not None and self.config.interface == "axi":
            physical_ports = self._interface_schema()
            if (len(physical_ports) < len(in_ports) + len(out_ports)):
                raise LaunchError(
                    "cannot recover the generated AXI top signature")
            if any(port["vector_lanes"] != 1 for port in physical_ports):
                raise LaunchError(
                    "AXI testbench generation currently requires scalar "
                    "physical ports; pin external_vector_min_elements above "
                    "the kernel's arrays for RTL co-simulation")
        schema = physical_ports or self._interface_schema()
        static_bytes = sum(
            math.prod(port["physical_shape"]) * port["data_bits"] // 8
            for port in schema)
        static_bytes += max((math.prod(port[1]) * 8 for port in out_ports),
                            default=0)
        raw_limit = os.environ.get("SAR_DSL_HLS_TESTBENCH_MAX_BYTES",
                                   str(1 << 30))
        try:
            limit = int(raw_limit)
        except ValueError:
            raise LaunchError(
                "SAR_DSL_HLS_TESTBENCH_MAX_BYTES must be an integer") from None
        if limit < 1:
            raise LaunchError(
                "SAR_DSL_HLS_TESTBENCH_MAX_BYTES must be positive")
        if static_bytes > limit:
            raise LaunchError(
                f"generated testbench needs about {static_bytes / 2**30:.2f} "
                f"GiB of static arrays, above the configured "
                f"{limit / 2**30:.2f} GiB limit")
        arrays = (
            list(compiler._plane_data(inputs, meta.arg_types)) +
            list(compiler._golden_plane_data(expected, meta.result_types)))

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
                                       tolerances["rtol"], tolerances["atol"],
                                       physical_ports))
        part, clock = compiler._part_and_clock(self.config)
        (out / f"{self.name}_csim.tcl").write_text(
            compiler._csim_script(self.name, part, clock))
        (out / f"{self.name}_csynth.tcl").write_text(
            compiler._csynth_script(self.name, part, clock, self.config))
        (out / f"{self.name}_cosim.tcl").write_text(
            compiler._cosim_script(self.name, part, clock, self.config))
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
        script.write_text(
            compiler._csynth_script(self.name, part, clock, self.config))
        return script

    def __call__(self, *args, **kwargs):
        raise LaunchError(
            "HLS designs are emitted as C++ for Vitis HLS synthesis and "
            f"cannot be executed directly; see {self.cpp_path}")

    def __repr__(self) -> str:
        return f"HLSDesign({self.name} @ {self.cpp_path})"
