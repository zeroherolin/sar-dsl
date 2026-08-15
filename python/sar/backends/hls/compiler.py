"""HLS backend: emits Vitis HLS C++ for FPGA synthesis.

A kernel is decomplexified (complex tensors become re/im float planes),
FFTs become Stockham loop nests, interpolation becomes windowed-sinc
gather loops (`sar-to-affine-pipeline`), and the resulting affine IR
enters the HLS pipeline (`-hls-pipeline`), which builds the dataflow
hierarchy, places buffers on or off chip, and shapes the interfaces.
`sar-translate` then writes the C++.

The compilation result is an `HLSDesign` handle pointing at the emitted
C++ source, not an executable kernel. The generated top function takes
each complex tensor as two adjacent float arrays (re, im), inputs
first, then one output array per result plane;
`HLSDesign.write_testbench` emits a matching C-simulation testbench
with golden data.

Every SAR operation has an HLS lowering, so complete imaging chains
(omega-K, range-Doppler, chirp scaling) emit as single designs.

Options:
    top_func       -- top function name (defaults to the kernel name)
    axi_interface  -- wrap the top function in AXI ports for SoC
                      integration (default False: the plain-array
                      signature, csim-able and synthesizable)
    on_chip_budget -- on-chip memory the design may use, in bytes
                      (default 4 MiB). Which buffers stay resident and how
                      large a block a transposing loop nest stages both
                      follow from it; pass 0 to keep everything on chip.
    axi_bus_bits   -- data width of the AXI masters (default 512). Beat
                      width, burst length and how many bursts are in
                      flight are derived from it.
    loop_tile_size -- loop tiling factor (default 8)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from sar.backends.base import BaseBackend, KernelMetadata
from sar.compiler.toolchain import find_tool, run_tool
from sar.errors import LaunchError, ToolchainError

_C_TYPES = {"f32": "float", "f64": "double"}

#: On-chip memory a design may occupy before buffers are pushed to DRAM.
#: Sized well below a mid-range device (a VU13P carries ~51 MiB) so the
#: default is portable; raise it to trade DRAM traffic for BRAM/URAM.
_DEFAULT_ON_CHIP_BUDGET = 4 * 1024 * 1024

#: Threshold that leaves every buffer resident (no buffer has this many
#: elements).
_KEEP_ON_CHIP = 2**32 - 1

#: Byte width of the element types the affine flow emits.
_ELEMENT_BYTES = {"f32": 4, "f64": 8}


def _largest_plane_elements(metadata: KernelMetadata) -> int:
    """Element count of the largest plane the kernel's signature names."""
    planes = list(metadata.arg_types) + list(metadata.result_types)
    return max((int(np.prod(t.shape)) for t in planes), default=0)


def _resident_bytes(metadata: KernelMetadata, lowered: str,
                    min_elements: int) -> int:
    """On-chip memory the lowered kernel would need for its full-size planes.

    The kernel's own planes come from the signature; the intermediates are
    counted in the IR, after buffers with disjoint lifetimes have been
    shared. Counting rather than estimating is what makes the placement
    decision hold for any algorithm: the same four-stage chain needs six
    live planes under range-Doppler and ten under chirp-scaling, and nothing
    in the signature says which.
    """
    planes = list(metadata.arg_types) + list(metadata.result_types)
    total = sum(
        int(np.prod(t.shape)) * t.dtype.to_numpy().itemsize for t in planes)
    for match in re.finditer(r"memref\.alloc\(\)[^:]*: memref<([^>]+)>",
                             lowered):
        parts = match.group(1).split("x")
        width = _ELEMENT_BYTES.get(parts[-1])
        if width is None:
            continue
        elements = int(np.prod([int(d) for d in parts[:-1]]))
        if elements >= min_elements:
            total += elements * width
    return total


def _external_buffer_threshold(metadata: KernelMetadata, lowered: str) -> int:
    """Element count above which a buffer is moved off chip.

    Full-scene planes dominate the working set; the constant tables
    (twiddles, interpolation weights) and the transform scratch, which the
    affine lowering keeps one line wide, are orders of magnitude smaller. So
    the decision is about the planes: keep them resident while the whole set
    fits the budget, and stream them once it does not, which is what lets
    scene size grow past the device.
    """
    budget = metadata.options.get("on_chip_budget", _DEFAULT_ON_CHIP_BUDGET)
    elements = _largest_plane_elements(metadata)
    if budget <= 0 or not elements:
        return _KEEP_ON_CHIP
    if _resident_bytes(metadata, lowered, elements) <= budget:
        return _KEEP_ON_CHIP
    return elements


def _split_planes(kind: str, types) -> list:
    """(name, shape, dtype-name) per top-function port: complex tensors
    contribute adjacent (re, im) float planes."""
    ports = []
    for i, t in enumerate(types):
        if t.dtype.is_complex:
            plane = "f32" if t.dtype.name == "c64" else "f64"
            ports.append((f"{kind}{i}_re", t.shape, plane))
            ports.append((f"{kind}{i}_im", t.shape, plane))
        else:
            ports.append((f"{kind}{i}", t.shape, t.dtype.name))
    return ports


class HLSDesign:
    """Handle to an emitted HLS C++ design."""

    def __init__(self, cpp_path: str, name: str, metadata=None):
        self.cpp_path = str(cpp_path)
        self.name = name
        self._metadata = metadata

    def source(self) -> str:
        return Path(self.cpp_path).read_text()

    def write_testbench(self,
                        inputs,
                        expected,
                        output_dir=None,
                        rtol: float = 1e-4,
                        atol: float = 1e-5) -> Path:
        """Writes a self-contained C-simulation package into `output_dir`:
        the design, a testbench comparing against golden data, the data
        files, and a Vitis HLS csim script.

        `inputs` / `expected` are numpy arrays matching the kernel's
        original argument/result types (complex arrays for complex
        tensors); golden outputs typically come from the NumPy reference
        or the cpu backend. `output_dir` defaults to
        `hls_project/<top>`. Returns the testbench path.
        """
        meta = self._metadata
        if meta.options.get("axi_interface", False):
            raise LaunchError(
                "testbench generation needs the plain-array top signature. "
                "With axi_interface=True every DRAM buffer -- including the "
                "FFT scratch planes -- is promoted to its own AXI port, so "
                "the top function takes far more arguments than the kernel "
                "has inputs and results, and the testbench has no data to "
                "drive them with. Emit the csim package without "
                "axi_interface, and use axi_interface for the design you "
                "hand to Vitis.")
        for t in list(meta.arg_types) + list(meta.result_types):
            if t.dtype.is_int:
                raise LaunchError(
                    "testbench generation does not support integer "
                    "kernel arguments/results yet")

        in_ports = _split_planes("in", meta.arg_types)
        out_ports = _split_planes("out", meta.result_types)
        arrays = (list(_plane_data(inputs, meta.arg_types)) +
                  list(_plane_data(expected, meta.result_types)))

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
        tb = out / f"{self.name}_tb.cpp"
        tb.write_text(
            _testbench_source(self.name, in_ports, out_ports, rtol, atol))
        (out / f"{self.name}_csim.tcl").write_text(_csim_script(self.name))
        _write_header_stubs(out / "stubs")
        return tb

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "HLS designs are emitted as C++ for Vitis HLS synthesis and "
            f"cannot be executed directly; see {self.cpp_path}")

    def __repr__(self) -> str:
        return f"HLSDesign({self.name} @ {self.cpp_path})"


def _plane_data(arrays, types):
    """Float planes (in port order) for numpy arrays of the given types."""
    if len(arrays) != len(types):
        raise LaunchError(f"expected {len(types)} array(s), got {len(arrays)}")
    for arr, t in zip(arrays, types):
        arr = np.asarray(arr)
        if tuple(arr.shape) != t.shape:
            raise LaunchError(
                f"array shape {arr.shape} does not match {t.shape}")
        if t.dtype.is_complex:
            plane = np.float32 if t.dtype.name == "c64" else np.float64
            yield np.ascontiguousarray(arr.real, dtype=plane)
            yield np.ascontiguousarray(arr.imag, dtype=plane)
        else:
            yield np.ascontiguousarray(arr, dtype=t.dtype.to_numpy())


def _c_decl(port) -> str:
    name, shape, dtype = port
    dims = "".join(f"[{d}]" for d in shape)
    return f"{_C_TYPES[dtype]} {name}{dims}"


def _testbench_source(top, in_ports, out_ports, rtol, atol) -> str:
    """C-simulation testbench: loads the .dat files, runs the design,
    compares each output plane against the golden data."""
    proto = ",\n    ".join(_c_decl(p) for p in in_ports + out_ports)
    decls = "\n".join(f"  static {_c_decl(p)};" for p in in_ports + out_ports)
    loads = "\n".join(
        f'  if (!load(dir, "{p[0]}.dat", &{p[0]}{"[0]" * len(p[1])}, '
        f"{int(np.prod(p[1]))})) return 2;" for p in in_ports)
    call_args = ", ".join(p[0] for p in in_ports + out_ports)
    checks = "\n".join(f'  fail |= check(dir, "{p[0]}.dat", "{p[0]}", '
                       f'&{p[0]}{"[0]" * len(p[1])}, {int(np.prod(p[1]))});'
                       for p in out_ports)
    return f"""\
// Auto-generated C-simulation testbench for `{top}`.
//
// Vitis HLS:  vitis_hls -f {top}_csim.tcl
// Plain C++ (stub headers stand in for the Vitis ones):
//   c++ -O2 -I stubs {top}.cpp {top}_tb.cpp -o csim -pthread && ./csim
//
// The design runs on a dedicated 1 GiB stack: in csim the on-chip
// buffers are stack arrays, far beyond the default thread stack.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <pthread.h>
#include <string>

void {top}(
    {proto});

namespace {{

// Vitis HLS may copy testbench data flat into the csim working
// directory; fall back from `dir`/`name` to the bare name.
template <typename T>
bool load(const std::string &dir, const std::string &name, T *dst,
          long count) {{
  std::string path = dir + "/" + name;
  std::ifstream f(path);
  if (!f) {{
    path = name;
    f.open(path);
  }}
  if (!f) {{
    std::fprintf(stderr, "cannot open %s/%s or ./%s\\n", dir.c_str(),
                 name.c_str(), name.c_str());
    return false;
  }}
  for (long i = 0; i < count; ++i)
    if (!(f >> dst[i])) {{
      std::fprintf(stderr, "short read in %s\\n", path.c_str());
      return false;
    }}
  return true;
}}

template <typename T>
int check(const std::string &dir, const std::string &name, const char *port,
          const T *got, long count) {{
  static double gold[{max(int(np.prod(p[1])) for p in out_ports)}];
  if (!load(dir, name, gold, count))
    return 2;
  double max_err = 0.0;
  long bad = 0;
  for (long i = 0; i < count; ++i) {{
    double err = std::abs(double(got[i]) - gold[i]);
    max_err = std::fmax(max_err, err);
    if (err > {atol} + {rtol} * std::abs(gold[i]))
      ++bad;
  }}
  std::printf("%-12s max |err| = %.3e  (%ld/%ld out of tolerance)\\n",
              port, max_err, bad, count);
  return bad != 0;
}}

int run(const std::string &dir) {{
{decls}

{loads}

  {top}({call_args});

  int fail = 0;
{checks}

  std::printf("%s\\n", fail ? "FAIL" : "PASS");
  return fail;
}}

}} // namespace

int main(int argc, char **argv) {{
  const std::string dir = argc > 1 ? argv[1] : "{top}_tb_data";
  auto trampoline = [](void *arg) -> void * {{
    auto rc = new int(run(*static_cast<const std::string *>(arg)));
    return rc;
  }};
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 1ull << 30);
  pthread_t worker;
  if (pthread_create(&worker, &attr, trampoline, (void *)&dir) != 0) {{
    std::fprintf(stderr, "cannot start the csim worker thread\\n");
    return 3;
  }}
  void *result = nullptr;
  pthread_join(worker, &result);
  int fail = *static_cast<int *>(result);
  delete static_cast<int *>(result);
  return fail;
}}
"""


_AP_INT_STUB = """\
#pragma once
// Minimal stand-in for the Vitis `ap_int`/`ap_uint` types: wide enough
// for the index arithmetic the designs emit, so any C++ compiler can
// run the C simulation without the Vitis headers.
template <int W> struct ap_int {
  long long v;
  ap_int() : v(0) {}
  ap_int(long long x) : v(x) {}
  operator long long() const { return v; }
};
template <int W> struct ap_uint {
  unsigned long long v;
  ap_uint() : v(0) {}
  ap_uint(unsigned long long x) : v(x) {}
  operator unsigned long long() const { return v; }
};
"""


def _write_header_stubs(stub_dir: Path) -> None:
    """Vitis header stand-ins so the package csims with any compiler."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "ap_int.h").write_text(_AP_INT_STUB)
    for header in ("ap_axi_sdata.h", "ap_fixed.h", "hls_math.h",
                   "hls_stream.h", "hls_vector.h"):
        (stub_dir / header).write_text("#pragma once\n#include <cmath>\n")


def _csim_script(top) -> str:
    """Vitis HLS csim script (written against Vitis HLS 2022.2)."""
    return f"""\
# Vitis HLS C simulation for `{top}`: vitis_hls -f {top}_csim.tcl
open_project -reset {top}_csim_proj
set_top {top}
add_files {top}.cpp
add_files -tb {top}_tb.cpp
add_files -tb {top}_tb_data
open_solution -reset sol1 -flow_target vivado
set_part xczu9eg-ffvb1156-2-e ;# ZCU102; adjust to your device
create_clock -period 10
csim_design -ldflags {{-lpthread}}
exit
"""


class Backend(BaseBackend):
    name = "hls"

    @classmethod
    def is_available(cls) -> bool:
        try:
            find_tool("sar-opt")
            find_tool("sar-translate")
            return True
        except ToolchainError:
            return False

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def add_stages(self, stages, metadata: KernelMetadata) -> None:
        stages["lower"] = self._stage_lower
        stages["hls"] = self._stage_hls_cpp

    @staticmethod
    def _stage_lower(module_text: str, metadata: KernelMetadata, cache) -> str:
        if cache.has("kernel.affine.mlir"):
            return cache.read_text("kernel.affine.mlir")
        cache.write_text("kernel.sar.mlir", module_text)

        options = metadata.options
        planes = _largest_plane_elements(metadata)
        budget = max(
            1, int(options.get("on_chip_budget", _DEFAULT_ON_CHIP_BUDGET)))

        # Three decisions, all keyed on the size of a full-scene plane:
        #
        # - share planes whose lifetimes do not overlap, so a longer chain
        #   costs no more buffers than a short one;
        # - recompute a full-size producer into each consumer rather than
        #   store it, trading an arithmetic unit for a raster of traffic;
        # - stage a transposing nest through an on-chip block, so both of
        #   its sides stream contiguously.
        #
        # Smaller tensors keep the usual rules: below plane size a buffer is
        # a dataflow channel, where one producer each is what lets the
        # backend pipeline the stages.
        pipeline = (f"--sar-to-affine-pipeline=reuse-buffer-min-elements={planes} "
                    f"recompute-min-elements={planes} "
                    f"transpose-block-bytes={budget // 8}")
        out = run_tool("sar-lower", [find_tool("sar-opt"), pipeline, "-"],
                       input_text=module_text)
        cache.write_text("kernel.affine.mlir", out)
        return out

    @staticmethod
    def _stage_hls_cpp(lowered: str, metadata: KernelMetadata, cache) -> str:
        cpp = cache.path("kernel.hls.cpp")
        if cache.has("kernel.hls.cpp"):
            return str(cpp)

        options = metadata.options
        top_func = options.get("top_func", metadata.name)

        # `axi_interface` decides how off-chip buffers reach the top
        # function: as AXI master ports (SoC integration) or as plain arrays
        # (what the generated csim testbench drives). Which buffers go off
        # chip is the compiler's call -- see `_external_buffer_threshold`.
        axi = ("axi-interface=true" if options.get("axi_interface", False) else
               "axi-interface=false")
        threshold = options.get("external_buffer_threshold")
        if threshold is None:
            threshold = _external_buffer_threshold(metadata, lowered)
        budget = max(
            1, int(options.get("on_chip_budget", _DEFAULT_ON_CHIP_BUDGET)))
        tile = options.get("loop_tile_size", 8)
        hls_arg = (f"-hls-pipeline=top-func={top_func} "
                   f"loop-tile-size={tile} on-chip-bytes={budget} {axi} "
                   f"external-buffer-threshold={int(threshold)}")

        scheduled = run_tool("sar-hls",
                             [find_tool("sar-opt"), hls_arg, "-"],
                             input_text=lowered)
        cache.write_text("kernel.hls.mlir", scheduled)

        # The bus width is a board property, so the shaping of the AXI
        # masters -- beat width, burst length, how many are in flight --
        # is derived from it rather than fixed.
        bus_bits = int(options.get("axi_bus_bits", 512))
        cpp_text = run_tool("sar-emit-hls", [
            find_tool("sar-translate"), "-hls-emit-hlscpp",
            "-emit-vitis-directives", f"-axi-bus-bits={bus_bits}", "-"
        ],
                            input_text=scheduled)
        cache.write_text("kernel.hls.cpp", cpp_text)
        return str(cpp)

    # ------------------------------------------------------------------ #
    # Launcher
    # ------------------------------------------------------------------ #

    def make_launcher(self, artifact: str, metadata: KernelMetadata):
        return HLSDesign(artifact, metadata.name, metadata=metadata)
